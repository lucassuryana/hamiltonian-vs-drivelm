"""
Ablation study, local-Qwen backend: same 2x2 ablation as run_ablation_pilot.py
(DriveLM-inspired vs Hamiltonian-inspired planning, crossed with
explicit-categorical-table vs generic physical reachability), but run on
Qwen3-VL-8B-Instruct locally (DAIC SLURM GPU job) instead of the GPT-4.1 API.

Four planning configurations, Step 3 ONLY -- Steps 1-2 (perception,
projection) are loaded verbatim from the already-completed Qwen3-VL-8B run
(results/scenario_hamiltonian_new_variants_v3.json) and never regenerated,
so every configuration sees IDENTICAL perception/projection input for a
given scenario -- mirrors the GPT pilot's frozen-Step1/2 design exactly, just
pointed at the Qwen frozen results instead of the GPT ones.

  A - DriveLM Original            prompts/step3_planning_drivelm.txt (unmodified)
  B - DriveLM + Explicit Reach.    prompts/step3_planning_drivelm_reachability.txt
  C - Hamiltonian + Generic Reach. prompts/step3_planning_generic_reachability.txt
  D - Hamiltonian Original         prompts/step3_planning.txt (unmodified)

Scope: all 22 scenario-variants of the full dataset x 4 configs x 10
repetitions each = 880 Step-3 local-inference calls (worst case 1760 if
every single one needs ensure_final_tags()'s one-shot retry). Greedy
decoding (temperature=0.0, do_sample=False) throughout -- repeats are run to
empirically check for residual run-to-run non-determinism in greedy
decoding on GPU (floating-point reduction order varies run-to-run; see the
loop-guard docstring below), not by sampling with temperature > 0.

Writes to results/ablation_pilot/ only; does not touch any existing
result/checkpoint file. Checkpoints after every single call, keyed by
(scenario, config, repetition), so it is safe to interrupt and resume
(e.g. if a SLURM job hits its time limit) -- just resubmit.

Usage (on DAIC, inside the `rq2` conda env, on a GPU node):
    python run_ablation_pilot_qwen.py --dry-run   # build+print prompts, no model load, $0 GPU time
    python run_ablation_pilot_qwen.py              # actually runs local inference
"""
import argparse
import gc
import json
import math
import os
import re
import socket
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from scripts.labels import continuous_to_drivelm_label

MAX_TOKENS_STEP3 = 9216  # raised 3x from 3072 (2026-10-21): 6/22 scenario-variant x
# config cells (all under C_hamiltonian_generic_reachability except scene2_variantE/D)
# hit the old 3072 cap mid-reasoning, before reaching Q12/PLANNING SUMMARY --
# confirmed via exact tokenizer counts, not just character-length estimates. See
# results/ablation_pilot/all_scenes_ablation_results_qwen.BACKUP_pre_token_fix_20261021.json
# for the pre-fix (truncated) versions of those 60 records.
IMAGE_SIZE = (640, 360)

SCENARIO_TOKENS = [
    "scene1_variantA", "scene1_variantB", "scene1_variantC", "scene1_variantD",
    "scene2_variantA", "scene2_variantB", "scene2_variantD", "scene2_variantE",
    "scene3_variantA", "scene3_variantB", "scene3_variantC", "scene3_variantD",
    "scene4_variantA", "scene4_variantB", "scene4_variantC", "scene4_variantD",
    "scene5_variantA", "scene5_variantB", "scene5_variantC",
    "scene6_variantA", "scene6_variantB", "scene6_variantC",
]
N_REPETITIONS = 10

CONFIGS = [
    ("A_drivelm_original", "prompts/step3_planning_drivelm.txt"),
    ("B_drivelm_explicit_reachability", "prompts/step3_planning_drivelm_reachability.txt"),
    ("C_hamiltonian_generic_reachability", "prompts/step3_planning_generic_reachability.txt"),
    ("D_hamiltonian_original", "prompts/step3_planning.txt"),
]

OUT_DIR = "results/ablation_pilot"
FROZEN_RESULTS = "results/scenario_hamiltonian_new_variants_v3.json"
DATA_FILE = "data/scenarios/preprocessed_scenarios_new_variants.json"
RESULTS_PATH = os.path.join(OUT_DIR, "all_scenes_ablation_results_qwen.json")


# ───────────────────────── model loading (mirrors scenario_comparison_v3_new_variants.ipynb) ──

def detect_model_path():
    hostname = socket.gethostname()
    if "daic" in hostname or "tudelft" in hostname or "hpc" in hostname:
        os.environ["HF_HOME"] = "/tudelft.net/staff-umbrella/lsuryana/huggingface"
    else:
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
    model_path = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Running on: {hostname}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"MODEL_PATH: {model_path}")
    return model_path


def load_model(model_path):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model loaded. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")
    return model, processor


# ───────────────────────── shared helpers (mirrors the v3 notebook / GPT pilot) ──

def resize_image(path):
    base, ext = os.path.splitext(path)
    tmp = f"{base}_r{ext}"
    if not os.path.exists(tmp):
        from PIL import Image
        img = Image.open(path).convert('RGB').resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(tmp, quality=85)
    return tmp


def get_image_sequence(s):
    fd = s['frame_dir']
    n_past = s.get('n_past', 0)
    past_frames = []
    if n_past >= 3:
        past_frames = [f"{fd}/tm3.jpg", f"{fd}/tm2.jpg", f"{fd}/tm1.jpg"]
    elif n_past == 2:
        past_frames = [f"{fd}/tm2.jpg", f"{fd}/tm1.jpg"]
    elif n_past == 1:
        past_frames = [f"{fd}/tm1.jpg"]
    return past_frames + [f"{fd}/t0.jpg"]


def build_image_description(n_past):
    lines = []
    labels = [("tm3", "t-3 (1.5 s before the decision point)"),
              ("tm2", "t-2 (1.0 s before the decision point)"),
              ("tm1", "t-1 (0.5 s before the decision point)")]
    labels = labels[3 - n_past:] if n_past < 3 else labels
    for i, (_, desc) in enumerate(labels, start=1):
        lines.append(f"- Image {i}: {desc}")
    lines.append(f"- Image {len(labels)+1}: t0 (decision point)")
    return "\n".join(lines)


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


def build_context_header(s):
    """IDENTICAL to run_ablation_pilot.py's build_context_header (which is
    itself a copy of results/scenario_comparison_gpt_new_variants.ipynb's) --
    kept in sync by hand, not imported, since none of the three are an
    importable module."""
    n_past = s.get('n_past', 0)
    past_traj = s.get('past_trajectory', [])
    prev_cont = s.get('prev_continuous') or {}
    ego_goal = s.get('ego_goal', 'continue driving safely along the current road')
    scenario_text = s.get('scenario_text', '')
    img_desc = build_image_description(n_past)

    past_traj_text = ""
    if n_past >= 3 and len(past_traj) >= 3:
        past_traj_text = (
            f"  t-1.5s: x={past_traj[0]['x']}, y={past_traj[0]['y']}\n"
            f"  t-1.0s: x={past_traj[1]['x']}, y={past_traj[1]['y']}\n"
            f"  t-0.5s: x={past_traj[2]['x']}, y={past_traj[2]['y']}\n"
        )
    elif n_past == 2 and len(past_traj) >= 2:
        past_traj_text = (
            f"  t-1.0s: x={past_traj[0]['x']}, y={past_traj[0]['y']}\n"
            f"  t-0.5s: x={past_traj[1]['x']}, y={past_traj[1]['y']}\n"
        )

    if n_past >= 1 and past_traj:
        dir_lbl, spd_lbl = get_past_behavior_label(past_traj, n_past)
        past_traj_text += (
            f"Historical ego behaviour:\n"
            f"  Speed:     {spd_lbl}\n"
            f"  Direction: {dir_lbl}\n"
        )
    elif n_past >= 1 and not past_traj:
        past_traj_text += (
            "Historical ego behaviour (ASSUMED -- no real telemetry was "
            "recorded for this synthetic scenario; treat this as a neutral "
            "placeholder, not an observed fact):\n"
            "  Speed:     driving with normal speed\n"
            "  Direction: going straight\n"
        )

    if prev_cont:
        past_traj_text += (
            f"Historical ego motion state (numeric):\n"
            f"  avg speed:    {prev_cont.get('avg_speed_mps', 0)} m/s "
            f"({prev_cont.get('avg_speed_kph', 0)} km/h)\n"
            f"  acceleration: {prev_cont.get('acceleration_mps2', 0)} m/s2\n"
            f"  heading:      {prev_cont.get('angle_overall_deg', 0)} degrees\n"
            f"  yaw rate:     {prev_cont.get('yaw_rate_deg_s', 0)} deg/s\n"
        )

    header = (
        f"You are given four sequential images of the same driving scene.\n"
        f"Image timeline:\n{img_desc}\n\n"
        f"Ego vehicle goal:\n{ego_goal}\n\n"
        f"Historical ego trajectory (relative to t0):\n"
        f"{past_traj_text}"
    )

    if scenario_text:
        header += (
            f"\nScenario details at the decision point (t0), as specified for this "
            f"experimental variant:\n  {scenario_text}\n"
        )

    return header


def looks_like_id_loop(text, threshold=30):
    """Same runaway-repetition detector as the v3 notebook: the model
    exhausts real scene objects and starts cycling through descriptors
    under a new ID forever (P1..P100+, V1..V140, etc)."""
    ids = re.findall(r'\b(?:P|V|C|M|TL|TS|O)(\d+)\b', text or '')
    return max((int(x) for x in ids), default=0) > threshold


# ───────────────────────── parsing / canonicalization (identical to run_ablation_pilot.py) ──

SPEED_LABELS = ["not moving", "driving slowly", "driving with normal speed",
                "driving fast", "driving very fast"]
DIRECTION_LABELS = ["going straight", "slightly steering to the left",
                     "slightly steering to the right", "steering to the left",
                     "steering to the right"]
NO_SAFE_ACTION = "no safe reachable action"
_NO_SAFE_PATTERNS = re.compile(
    r"\bno safe\b|\bno viable\b|\bno reachable\b|\bnone\b|\bn/a\b",
    re.IGNORECASE,
)


def parse_step(raw, step):
    patterns = {
        'planning_summary': r'PLANNING[ _]SUMMARY:?\s*(.*?)(?=FINAL SPEED:|$)',
        'final_speed': r'FINAL SPEED:\s*(.+?)(?=\n|FINAL DIRECTION:|$)',
        'final_direction': r'FINAL DIRECTION:\s*(.+?)(?=\n|FINAL SPEED:|$)',
    }
    if step in ('final_speed', 'final_direction'):
        matches = re.findall(patterns[step], raw, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        return matches[-1].strip() if matches else ''
    match = re.search(patterns[step], raw, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ''


def _speed_keyword(cleaned):
    t = cleaned.lower()
    if 'not moving' in t or 'stationary' in t or 'stopped' in t:
        return 'not moving'
    if 'very fast' in t:
        return 'driving very fast'
    if 'fast' in t:
        return 'driving fast'
    if 'normal' in t:
        return 'driving with normal speed'
    if 'slow' in t:
        return 'driving slowly'
    return None


def _direction_keyword(cleaned):
    t = cleaned.lower()
    if 'straight' in t:
        return 'going straight'
    slightly = 'slightly' in t
    if 'left' in t:
        return 'slightly steering to the left' if slightly else 'steering to the left'
    if 'right' in t:
        return 'slightly steering to the right' if slightly else 'steering to the right'
    return None


def canonicalize_label(raw_value, canonical_list, keyword_fn):
    cleaned = raw_value.strip().strip('*').strip().rstrip('.').strip()
    if not cleaned:
        return cleaned
    if _NO_SAFE_PATTERNS.search(cleaned):
        return NO_SAFE_ACTION
    norm = re.sub(r'\s+', '', cleaned).lower()
    for canon in canonical_list:
        if re.sub(r'\s+', '', canon).lower() == norm:
            return canon
    kw = keyword_fn(cleaned)
    return kw if kw else cleaned


FINAL_TAGS_FOLLOWUP = (
    "Your previous answer did not include the required final lines. "
    "Reply with EXACTLY these two lines and nothing else:\n"
    "FINAL SPEED: <one of: not moving, driving slowly, driving with normal speed, "
    "driving fast, driving very fast, no safe reachable action>\n"
    "FINAL DIRECTION: <one of: going straight, slightly steering to the left, "
    "slightly steering to the right, steering to the left, steering to the right, "
    "no safe reachable action>"
)


def ensure_final_tags(raw, base_messages, run_inference_fn, max_retry_tokens=64):
    speed = parse_step(raw, 'final_speed')
    direction = parse_step(raw, 'final_direction')
    if speed and direction:
        return raw, canonicalize_label(speed, SPEED_LABELS, _speed_keyword), \
                    canonicalize_label(direction, DIRECTION_LABELS, _direction_keyword)

    followup_messages = base_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": raw}]},
        {"role": "user", "content": [{"type": "text", "text": FINAL_TAGS_FOLLOWUP}]},
    ]
    followup_raw = run_inference_fn(followup_messages, max_tokens=max_retry_tokens)
    combined = raw + "\n" + followup_raw
    speed = parse_step(combined, 'final_speed')
    direction = parse_step(combined, 'final_direction')
    if speed:
        speed = canonicalize_label(speed, SPEED_LABELS, _speed_keyword)
    if direction:
        direction = canonicalize_label(direction, DIRECTION_LABELS, _direction_keyword)
    return combined, speed, direction


# ───────────────────────── pilot driver ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Build and print every prompt that would be sent, "
                          "load no model, make zero inference calls, $0 GPU time.")
    args = ap.parse_args()

    with open(DATA_FILE) as f:
        data = json.load(f)
    by_token = {d['sample_token']: d for d in data}

    with open(FROZEN_RESULTS) as f:
        frozen_results = json.load(f)
    frozen_by_token = {r['sample_token']: r['result'] for r in frozen_results}

    missing = [t for t in SCENARIO_TOKENS if t not in frozen_by_token]
    if missing:
        raise SystemExit(f"Missing frozen Step1/2 results for: {missing}")

    n_calls = len(SCENARIO_TOKENS) * len(CONFIGS) * N_REPETITIONS
    print(f"Pilot plan: {len(SCENARIO_TOKENS)} scenarios x {len(CONFIGS)} configs "
          f"x {N_REPETITIONS} repetitions = {n_calls} Step-3 calls "
          f"(up to {n_calls * 2} if every single one needs the missing-tags retry).")
    print(f"Model: Qwen3-VL-8B-Instruct (local), max_tokens={MAX_TOKENS_STEP3}, "
          f"temperature=0.0, do_sample=False, repetition_penalty=1.1\n")

    if args.dry_run:
        def run_inference(messages, max_tokens=MAX_TOKENS_STEP3):
            raise RuntimeError("run_inference should not be called in --dry-run mode")
    else:
        import torch
        model_path = detect_model_path()
        model, processor = load_model(model_path)
        from qwen_vl_utils import process_vision_info

        def _generate(messages, max_tokens, **gen_kwargs):
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            has_images = any(
                isinstance(item, dict) and item.get('type') == 'image'
                for msg in messages for item in msg.get('content', []))
            if has_images:
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, return_tensors="pt").to("cuda")
            else:
                inputs = processor(text=[text], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_tokens, **gen_kwargs)
            trimmed = out[:, inputs['input_ids'].shape[1]:]
            result = processor.decode(trimmed[0], skip_special_tokens=True)
            del inputs, out, trimmed
            torch.cuda.empty_cache()
            return result

        def run_inference(messages, max_tokens=MAX_TOKENS_STEP3, max_loop_retries=2):
            """Greedy by default; on detecting the runaway-repetition failure
            mode (looks_like_id_loop), retry with light sampling to escape the
            deterministic path that led into the loop -- identical strategy to
            the v3 notebook's run_inference."""
            torch.cuda.empty_cache()
            gc.collect()
            result = _generate(messages, max_tokens, temperature=0.0, do_sample=False,
                                repetition_penalty=1.1)
            for attempt in range(max_loop_retries):
                if not looks_like_id_loop(result):
                    return result
                print(f"    [loop guard] repetition loop detected, retry {attempt+1}/{max_loop_retries} "
                      f"with sampling to escape the deterministic path...")
                result = _generate(messages, max_tokens, temperature=0.4, do_sample=True,
                                    repetition_penalty=1.2)
            if looks_like_id_loop(result):
                print(f"    [loop guard] still looping after {max_loop_retries} retries -- "
                      f"keeping this result, flag for manual review.")
            return result

    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    done_keys = set()
    if not args.dry_run and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        done_keys = {(r['scenario_id'], r['config'], r['repetition']) for r in results}
        print(f"Resuming: {len(done_keys)}/{n_calls} already done in {RESULTS_PATH}")

    call_count = 0

    for token in SCENARIO_TOKENS:
        s = by_token[token]
        frozen = frozen_by_token[token]
        raw1 = frozen['perception_raw']
        raw2 = frozen['projection_raw']

        images = get_image_sequence(s)
        resized = [resize_image(p) for p in images]
        context_header = build_context_header(s)
        img_content = [{"type": "image", "image": p} for p in resized]

        for config_label, prompt_path in CONFIGS:
            with open(prompt_path) as f:
                p3_text = f.read()

            prompt3 = (
                context_header +
                f"\nPerception results (Step 1):\n{raw1}\n\n"
                f"Behaviour assessment (Step 2):\n{raw2}\n\n"
            ) + p3_text
            msg3 = [{"role": "user", "content": img_content + [{"type": "text", "text": prompt3}]}]

            if args.dry_run:
                preview_path = os.path.join(OUT_DIR, f"DRYRUN_prompt_qwen_{token}_{config_label}.txt")
                with open(preview_path, 'w') as f:
                    f.write(prompt3)
                print(f"[{token} / {config_label}] prompt file: {prompt_path} "
                      f"({len(prompt3)} chars, {len(resized)} images)")
                print(f"    [dry-run] wrote full prompt text to {preview_path}, no inference call made")
                continue

            for rep in range(1, N_REPETITIONS + 1):
                if (token, config_label, rep) in done_keys:
                    continue

                print(f"[{token} / {config_label} / rep {rep}] prompt file: {prompt_path} "
                      f"({len(prompt3)} chars, {len(resized)} images)")

                t0 = time.time()
                raw3 = run_inference(msg3, max_tokens=MAX_TOKENS_STEP3)
                call_count += 1
                raw3, final_speed, final_direction = ensure_final_tags(raw3, msg3, run_inference)
                if (not final_speed or not final_direction):
                    call_count += 1  # ensure_final_tags made one retry call
                elapsed = time.time() - t0

                status = (
                    "no_safe_action" if final_speed == NO_SAFE_ACTION or final_direction == NO_SAFE_ACTION
                    else "parsed_ok" if final_speed and final_direction
                    else "parse_failed"
                )

                print(f"    {elapsed:.1f}s -> speed={final_speed!r}, direction={final_direction!r}, "
                      f"status={status}")

                results.append({
                    "scenario_id": token,
                    "config": config_label,
                    "repetition": rep,
                    "prompt_file": prompt_path,
                    "model": "Qwen3-VL-8B-Instruct",
                    "final_speed": final_speed,
                    "final_direction": final_direction,
                    "planning_status": status,
                    "planning_raw": raw3,
                    "elapsed_s": round(elapsed, 1),
                })

                # checkpoint after every call -- safe to interrupt and resume
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(results, f, indent=2)

    if args.dry_run:
        n_prompts = len(SCENARIO_TOKENS) * len(CONFIGS)
        print(f"\n[dry-run] Done. {n_prompts} distinct prompts previewed in {OUT_DIR}/, "
              f"zero inference calls made. (Repetitions don't change the prompt, so only "
              f"{n_prompts} files were written, not {n_calls}.)")
        return

    print(f"\nDone. {len(results)} results, {call_count} total inference calls this run "
          f"(including any missing-tags retries), saved to {RESULTS_PATH}")

    print(f"\n{'scenario':<18} {'config':<36} {'rep':<4} {'speed':<28} {'direction':<32} status")
    print("-" * 125)
    for r in results:
        print(f"{r['scenario_id']:<18} {r['config']:<36} {r['repetition']:<4} {r['final_speed']:<28} "
              f"{r['final_direction']:<32} {r['planning_status']}")


if __name__ == "__main__":
    main()
