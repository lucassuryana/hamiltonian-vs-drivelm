"""
scripts/run_pipeline.py

Standalone script version of smoke_test_v5.ipynb's inference pipeline,
built for SLURM batch execution rather than interactive notebook use.

Combines (from the notebook):
  - CELL 0   : hostname/path detection
  - CELL 1   : model loading
  - CELL 2   : data/prompt loading + helper functions
  - CELL 2b  : build_context_header, run_condition_v5 (Hamiltonian),
               run_planning_drivelm (DriveLM-style, reuses frozen Steps 1-3)
  - CELL 4   : main loop logic (notebook-only display/plot calls removed)

Designed to process a SLICE of the full sample list, so it can be run
as a SLURM job array: each array task handles samples[start:end] and
checkpoints to its own file, safe to interrupt/resume independently.

Usage:
    python scripts/run_pipeline.py \
        --preprocessed results/preprocessed_n100.json \
        --start 0 --end 25 \
        --arm both \
        --out-dir results/n100_run

After all chunks finish, use scripts/merge_chunks.py to combine them.
"""
import argparse
import json
import math
import os
import re
import socket
import time

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from scripts.labels import continuous_to_drivelm_label


# ── CELL 0: hostname/path detection ─────────────────────────────────────────
def detect_paths():
    hostname = socket.gethostname()
    if "daic" in hostname or "tudelft" in hostname or "hpc" in hostname:
        os.environ["HF_HOME"] = "/tudelft.net/staff-umbrella/lsuryana/huggingface"
        model_path = "Qwen/Qwen2-VL-7B-Instruct"
        data_path = "/tudelft.net/staff-umbrella/lsuryana/rq2-posthoc-driving/data/nuscenes"
    else:
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
        model_path = "Qwen/Qwen2-VL-7B-Instruct"
        data_path = os.path.expanduser("~/rq2-posthoc-driving/data/nuscenes")
    print(f"Running on: {hostname}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"MODEL_PATH: {model_path}")
    print(f"DATA_PATH: {data_path}")
    return model_path, data_path


# ── CELL 1: model loading ────────────────────────────────────────────────────
def load_model(model_path):
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model loaded. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")
    return model, processor


IMAGE_SIZE = (640, 360)


def resize_image(path):
    tmp = path.replace('.jpg', '_r.jpg')
    if not os.path.exists(tmp):
        img = Image.open(path).convert('RGB').resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(tmp, quality=85)
    return tmp


def get_image_sequence(s, condition):
    fd = s['frame_dir']
    n_past = s.get('n_past', 0)
    past_frames = []
    if n_past >= 3:
        past_frames = [f"{fd}/tm3.jpg", f"{fd}/tm2.jpg", f"{fd}/tm1.jpg"]
    elif n_past == 2:
        past_frames = [f"{fd}/tm2.jpg", f"{fd}/tm1.jpg"]
    elif n_past == 1:
        past_frames = [f"{fd}/tm1.jpg"]
    current = [f"{fd}/t0.jpg"]
    future = [f"{fd}/t1.jpg", f"{fd}/t2.jpg", f"{fd}/t3.jpg"] \
        if condition == 'posthoc' else []
    return past_frames + current + future


def build_image_description(s, condition):
    n_past = s.get('n_past', 0)
    lines = []
    if n_past == 3:
        lines += [
            "- Image 1: t-3 (1.5 seconds before decision point)",
            "- Image 2: t-2 (1.0 second before decision point)",
            "- Image 3: t-1 (0.5 seconds before decision point)",
            "- Image 4: t0  (current frame -- DECISION POINT)",
        ]
        if condition == 'posthoc':
            lines += [
                "- Image 5: t+1 (0.5 seconds after decision point)",
                "- Image 6: t+2 (1.0 second after decision point)",
                "- Image 7: t+3 (1.5 seconds after decision point)",
            ]
    elif n_past == 2:
        lines += [
            "- Image 1: t-2 (1.0 second before decision point)",
            "- Image 2: t-1 (0.5 seconds before decision point)",
            "- Image 3: t0  (current frame -- DECISION POINT)",
        ]
        if condition == 'posthoc':
            lines += [
                "- Image 4: t+1 (0.5 seconds after decision point)",
                "- Image 5: t+2 (1.0 second after decision point)",
                "- Image 6: t+3 (1.5 seconds after decision point)",
            ]
    return "\n".join(lines)


def get_gt_continuous(s):
    return s.get('gt_continuous') or {}


def get_prev_continuous(s):
    return s.get('prev_continuous') or {}


def get_past_behavior_label(past_traj, n_past):
    if n_past == 0 or not past_traj:
        return continuous_to_drivelm_label(0.0, 0.0)
    earliest = past_traj[0]
    dx = 0.0 - earliest['x']
    dy = 0.0 - earliest['y']
    dt = n_past * 0.5
    dist = math.sqrt(dx ** 2 + dy ** 2)
    avg_speed_mps = dist / dt
    angle_deg = math.degrees(math.atan2(dy, dx)) if dist > 0.01 else 0.0
    return continuous_to_drivelm_label(avg_speed_mps, angle_deg)


PARSE_PATTERNS = {
    'perception_summary': r'PERCEPTION_SUMMARY:\s*(.*?)$',
    'comprehension_summary': r'COMPREHENSION_SUMMARY:\s*(.*?)$',
    'prediction_summary': r'PREDICTION_SUMMARY:\s*(.*?)$',
    'planning': r'PLANNING:\s*(.*?)$',
    'direction': r'DIRECTION:\s*(.+?)(?:\n|$)',
    'speed': r'SPEED:\s*(.+?)(?:\n|$)',
    'reasoning': r'REASONING:\s*(.*?)$',
}


def parse_step(raw, step):
    match = re.search(PARSE_PATTERNS[step], raw, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ''



def extract_fallback_plan(raw):
    """Used only when PLANNING: is missing (truncated generation)."""
    m = re.search(r"Chosen pair:\s*(.+?)(?:\s*--|\s*\(copied|\n|$)", raw)
    if m and m.group(1).strip():
        return m.group(1).strip()
    dm = re.search(r"Direction:\s*(.+?)(?:\n|$)", raw)
    sm = re.search(r"Speed:\s*(.+?)(?:\n|$)", raw)
    if dm and sm and dm.group(1).strip() and sm.group(1).strip():
        return f"{dm.group(1).strip()} + {sm.group(1).strip()}"
    return None


class Pipeline:
    """Wraps model/processor/prompts so functions don't need globals."""

    def __init__(self, model, processor, p1, p2, p3, p4, p5, p4_drivelm):
        self.model = model
        self.processor = processor
        self.p1, self.p2, self.p3, self.p4, self.p5 = p1, p2, p3, p4, p5
        self.p4_drivelm = p4_drivelm

    def run_inference(self, messages, max_tokens=128):
        torch.cuda.empty_cache()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        has_images = any(
            isinstance(item, dict) and item.get('type') == 'image'
            for msg in messages for item in msg.get('content', []))
        if has_images:
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt").to("cuda")
        else:
            inputs = self.processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
        trimmed = out[:, inputs['input_ids'].shape[1]:]
        result = self.processor.decode(trimmed[0], skip_special_tokens=True)
        del inputs, out, trimmed
        torch.cuda.empty_cache()
        return result

    def build_context_header(self, s, condition):
        images = get_image_sequence(s, condition)
        n_total = len(images)
        n_past = s.get('n_past', 0)
        traj = s['trajectory']
        past_traj = s.get('past_trajectory', [])
        gt_cont = get_gt_continuous(s)
        prev_cont = get_prev_continuous(s)
        img_desc = build_image_description(s, condition)
        ego_goal = s.get('ego_goal', 'continue driving safely along the current road')

        past_traj_text = ""
        if n_past >= 3:
            past_traj_text = (
                f"  t-1.5s: x={past_traj[0]['x']}, y={past_traj[0]['y']}\n"
                f"  t-1.0s: x={past_traj[1]['x']}, y={past_traj[1]['y']}\n"
                f"  t-0.5s: x={past_traj[2]['x']}, y={past_traj[2]['y']}\n")
        elif n_past == 2:
            past_traj_text = (
                f"  t-1.0s: x={past_traj[0]['x']}, y={past_traj[0]['y']}\n"
                f"  t-0.5s: x={past_traj[1]['x']}, y={past_traj[1]['y']}\n")

        if n_past >= 1:
            dir_lbl, spd_lbl = get_past_behavior_label(past_traj, n_past)
            past_traj_text += (
                f"Ego vehicle past behavior (aggregated over past window):\n"
                f"  speed:     {spd_lbl}\n"
                f"  direction: {dir_lbl}\n")

        if prev_cont:
            past_motion_text = (
                f"  avg speed:    {prev_cont.get('avg_speed_mps', 0)} m/s "
                f"({prev_cont.get('avg_speed_kph', 0)} km/h)\n"
                f"  acceleration: {prev_cont.get('acceleration_mps2', 0)} m/s2\n"
                f"  heading:      {prev_cont.get('angle_overall_deg', 0)} degrees\n"
                f"  yaw rate:     {prev_cont.get('yaw_rate_deg_s', 0)} deg/s\n")
        else:
            past_motion_text = "  not available\n"

        header = (
            f"You have access to {n_total} images showing the driving scene:\n"
            f"{img_desc}\n\n"
            f"Ego vehicle goal: {ego_goal}\n\n"
            f"Ego vehicle past trajectory (x=forward meters, y=left meters, relative to t0):\n"
            f"{past_traj_text}"
            f"Ego vehicle past motion state:\n"
            f"{past_motion_text}")

        if condition == 'posthoc':
            header += (
                f"\nEgo vehicle future trajectory (ground-truth):\n"
                f"  t+0.5s: x={traj[0]['x']}, y={traj[0]['y']}\n"
                f"  t+1.0s: x={traj[1]['x']}, y={traj[1]['y']}\n"
                f"  t+1.5s: x={traj[2]['x']}, y={traj[2]['y']}\n"
                f"Ego vehicle future motion state (ground-truth):\n"
                f"  avg speed:    {gt_cont.get('avg_speed_mps', 0)} m/s "
                f"({gt_cont.get('avg_speed_kph', 0)} km/h)\n"
                f"  heading:      {gt_cont.get('angle_overall_deg', 0)} degrees\n"
                f"  acceleration: {gt_cont.get('acceleration_mps2', 0)} m/s2\n"
                f"  yaw rate:     {gt_cont.get('yaw_rate_deg_s', 0)} deg/s\n")
        return header

    def run_condition_v5(self, s, condition):
        """Hamiltonian-structured pipeline (Steps 1-5)."""
        images = get_image_sequence(s, condition)
        resized = [resize_image(img) for img in images]
        n_total = len(images)
        context_header = self.build_context_header(s, condition)

        def img_content():
            return [{"type": "image", "image": p} for p in resized]

        prompt1 = context_header + "\n" + self.p1
        msg1 = [{"role": "user", "content": img_content() + [{"type": "text", "text": prompt1}]}]
        raw1 = self.run_inference(msg1, max_tokens=1024)
        perc_summary = parse_step(raw1, 'perception_summary') or raw1.strip()

        prompt2 = (context_header + f"\nPerception analysis:\n{raw1}\n\n"
                   f"Perception summary: {perc_summary}\n\n") + self.p2
        msg2 = [{"role": "user", "content": [{"type": "text", "text": prompt2}]}]
        raw2 = self.run_inference(msg2, max_tokens=1024)
        comp_summary = parse_step(raw2, 'comprehension_summary') or raw2.strip()

        prompt3 = (context_header + f"\nComprehension output:\n{raw2}\n\n"
                   f"Comprehension summary: {comp_summary}\n\n") + self.p3
        msg3 = [{"role": "user", "content": [{"type": "text", "text": prompt3}]}]
        raw3 = self.run_inference(msg3, max_tokens=1024)
        pred_summary = parse_step(raw3, 'prediction_summary') or raw3.strip()

        prompt4 = (context_header + f"\nProjection output:\n{raw3}\n\n"
                   f"Projection summary: {pred_summary}\n\n") + self.p4
        msg4 = [{"role": "user", "content": [{"type": "text", "text": prompt4}]}]
        raw4 = self.run_inference(msg4, max_tokens=4096)
        ans_plan = parse_step(raw4, 'planning') or extract_fallback_plan(raw4) or raw4.strip()

        prompt5 = (context_header + f"\nPerception summary:    {perc_summary}\n"
                   f"Comprehension summary: {comp_summary}\n"
                   f"Prediction summary:    {pred_summary}\n"
                   f"Planning:              {ans_plan}\n\n") + self.p5
        msg5 = [{"role": "user", "content": [{"type": "text", "text": prompt5}]}]
        raw5 = self.run_inference(msg5, max_tokens=512)

        direction = parse_step(raw5, 'direction')
        speed = parse_step(raw5, 'speed')
        reasoning = parse_step(raw5, 'reasoning')

        gt = get_gt_continuous(s)
        gt_direction = gt.get('direction_label', '')
        gt_speed = gt.get('speed_label', '')
        direction_correct = (direction == gt_direction)
        speed_correct = (speed == gt_speed)

        return {
            'perception_raw': raw1, 'perception_summary': perc_summary,
            'comprehension_raw': raw2, 'comprehension_summary': comp_summary,
            'prediction_raw': raw3, 'prediction_summary': pred_summary,
            'planning_raw': raw4, 'planning': ans_plan,
            'direction': direction, 'speed': speed, 'reasoning': reasoning,
            'raw_step5': raw5, 'gt_direction': gt_direction, 'gt_speed': gt_speed,
            'direction_correct': direction_correct, 'speed_correct': speed_correct,
            'full_correct': direction_correct and speed_correct, 'n_images': n_total,
        }

    def run_planning_drivelm(self, saved_result, s, condition):
        """DriveLM-style Step 4b + 5b, reusing frozen Steps 1-3 from run_condition_v5."""
        context_header = self.build_context_header(s, condition)
        raw3 = saved_result['prediction_raw']
        pred_summary = saved_result['prediction_summary']
        perc_summary = saved_result['perception_summary']
        comp_summary = saved_result['comprehension_summary']

        prompt4b = (context_header + f"\nProjection output:\n{raw3}\n\n"
                    f"Projection summary: {pred_summary}\n\n") + self.p4_drivelm
        msg4b = [{"role": "user", "content": [{"type": "text", "text": prompt4b}]}]
        raw4b = self.run_inference(msg4b, max_tokens=2800)
        ans_plan_b = parse_step(raw4b, 'planning') or extract_fallback_plan(raw4b) or raw4b.strip()

        prompt5b = (context_header + f"\nPerception summary:    {perc_summary}\n"
                    f"Comprehension summary: {comp_summary}\n"
                    f"Prediction summary:    {pred_summary}\n"
                    f"Planning:              {ans_plan_b}\n\n") + self.p5
        msg5b = [{"role": "user", "content": [{"type": "text", "text": prompt5b}]}]
        raw5b = self.run_inference(msg5b, max_tokens=512)

        direction_b = parse_step(raw5b, 'direction')
        speed_b = parse_step(raw5b, 'speed')
        gt_direction = saved_result['gt_direction']
        gt_speed = saved_result['gt_speed']

        return {
            'planning_raw_drivelm': raw4b, 'planning_drivelm': ans_plan_b,
            'direction_drivelm': direction_b, 'speed_drivelm': speed_b,
            'raw_step5_drivelm': raw5b,
            'direction_correct_drivelm': direction_b == gt_direction,
            'speed_correct_drivelm': speed_b == gt_speed,
            'full_correct_drivelm': (direction_b == gt_direction) and (speed_b == gt_speed),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed', required=True,
                         help='Path to preprocessed.json (or a larger-scale variant)')
    parser.add_argument('--start', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--arm', choices=['hamiltonian', 'drivelm', 'both'], default='both')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_path, _ = detect_paths()
    model, processor = load_model(model_path)

    with open(args.preprocessed) as f:
        data = json.load(f)
    chunk = data[args.start:args.end]
    print(f"Processing samples {args.start}:{args.end} ({len(chunk)} samples)")

    p1 = open('prompts/step1_perception.txt').read()
    p2 = open('prompts/step2_comprehension.txt').read()
    p3 = open('prompts/step3_projection.txt').read()
    p4 = open('prompts/step4_planning.txt').read()
    p5 = open('prompts/step5_behavior.txt').read()
    p4_drivelm = open('prompts/step4_planning_drivelm.txt').read()

    pipe = Pipeline(model, processor, p1, p2, p3, p4, p5, p4_drivelm)

    checkpoint_file = os.path.join(args.out_dir, f'chunk_{args.start}_{args.end}.json')
    results = []
    done_tokens = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            results = json.load(f)
        done_tokens = {r['sample_token'] for r in results}
        print(f"Resuming: {len(done_tokens)}/{len(chunk)} done in this chunk")

    total_start = time.time()
    for i, s in enumerate(chunk):
        if s['sample_token'] in done_tokens:
            continue

        entry = {
            'sample_token': s['sample_token'],
            'scene_token': s.get('scene_token'),
            'gt_continuous': get_gt_continuous(s),
            'prev_continuous': get_prev_continuous(s),
            'trajectory': s['trajectory'],
            'past_trajectory': s.get('past_trajectory', []),
            'n_past': s.get('n_past', 0),
        }

        for cond in ['non_posthoc', 'posthoc']:
            t0 = time.time()
            out = {}
            if args.arm in ('hamiltonian', 'both'):
                out.update(pipe.run_condition_v5(s, cond))
            if args.arm in ('drivelm', 'both'):
                # drivelm reuses the frozen Steps 1-3 just computed above
                out.update(pipe.run_planning_drivelm(out, s, cond))
            entry[cond] = out
            print(f"  [{args.start+i+1}] {s['sample_token'][:8]} {cond}: {time.time()-t0:.1f}s")

        results.append(entry)
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[{args.start+i+1}/{args.end}] {s['sample_token'][:8]} done "
              f"({len(results)}/{len(chunk)} in this chunk)")

    print(f"\nChunk done in {(time.time()-total_start)/60:.1f} min. "
          f"Saved to {checkpoint_file}")


if __name__ == '__main__':
    main()
