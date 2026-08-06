"""
Build data/scenarios/preprocessed_scenarios.json from the scene folder
structure, and materialize normalized per-variant folders so the existing
notebook code (get_image_sequence, resize_image) works unchanged.

For each scenario variant, this creates:
    data/scenarios/normalized/<variant_id>/
        tm3.jpg  -> symlink to the real fixed frame
        tm2.jpg  -> symlink to the real fixed frame
        tm1.jpg  -> symlink to the real fixed frame
        t0.jpg   -> COPY of the chosen t0_N variant (converted to .jpg if
                    the source was .png, since the pipeline assumes .jpg)
        t1.jpg / t2.jpg / t3.jpg -> symlinks to real future frames, if
                    present (only used when condition == 'posthoc')

Run this from the repo root on WSL:
    python build_preprocessed_scenarios.py

Requires: the base 100-frame preprocessed JSON (for real gt_continuous /
past_trajectory per token) and, for scene 1, the scenario design xlsx.
Edit the CONFIG block below to match your actual paths before running.
"""

import json
import os
import re
import shutil
from pathlib import Path

try:
    import openpyxl
except ImportError:
    openpyxl = None

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths to match your repo layout
# ─────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(".")
IMAGES_ROOT = REPO_ROOT / "data" / "scenarios" / "images"
NORMALIZED_ROOT = REPO_ROOT / "data" / "scenarios" / "normalized"
OUTPUT_JSON = REPO_ROOT / "data" / "scenarios" / "preprocessed_scenarios.json"

# The original 100-frame dataset — source of real gt_continuous / trajectory
# per sample_token. Adjust this path to wherever it actually lives in the repo.
BASE_100_JSON = REPO_ROOT / "results" / "preprocessed_100_final.json"

# Scene 1's scenario-text design matrix (Full text sheet) — used to attach
# the pedestrian/vehicle/light description to each t0_N by ORDINAL POSITION
# (t0_1 -> row 1, t0_2 -> row 2, ...). CONFIRM this ordering matches your
# actual image generation before trusting the attached text.
SCENE1_XLSX = REPO_ROOT / "data" / "scenarios" / "30_scenarios_kmap_sorted (1).xlsx"
SCENE1_TOKEN = "dc676572"

# Scene 2's 7 real base tokens (red-light intersection, cross-traffic /
# pedestrian vary naturally across folders; t0_1..t0_5 meaning TBD — left
# unattached for now).
SCENE2_TOKENS = [
    "6dabc0fb", "37535cd3", "3ffc4360", "b8766f42",
    "e9a1c915", "27900b9c", "47188ae2",
]

# ─────────────────────────────────────────────────────────────────────────

def load_base_samples():
    if not BASE_100_JSON.exists():
        print(f"WARNING: {BASE_100_JSON} not found. gt_continuous/trajectory "
              f"fields will be left empty — fix BASE_100_JSON path and rerun.")
        return {}
    with open(BASE_100_JSON) as f:
        data = json.load(f)
    return {d["sample_token"]: d for d in data}


def load_scene1_scenario_text():
    """Returns {ordinal_index (1-based): text_description} for scene 1."""
    if openpyxl is None:
        print("WARNING: openpyxl not installed, skipping scenario-text attachment "
              "for scene 1. pip install openpyxl and rerun if you need this.")
        return {}
    if not SCENE1_XLSX.exists():
        print(f"WARNING: {SCENE1_XLSX} not found, skipping scenario-text attachment.")
        return {}
    wb = openpyxl.load_workbook(SCENE1_XLSX, data_only=True)
    if "Full text" not in wb.sheetnames:
        print("WARNING: 'Full text' sheet not found in xlsx, skipping.")
        return {}
    ws = wb["Full text"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
    text_by_idx = {}
    for row in rows:
        idx = row[0]
        if idx is None:
            continue
        parts = [str(v) for v in row[1:] if v not in (None, "")]
        text_by_idx[int(idx)] = " | ".join(parts)
    return text_by_idx


def find_base_token_dir(scene_dir: Path, token_prefix: str) -> Path | None:
    for sub in scene_dir.iterdir():
        if sub.is_dir() and sub.name.startswith(token_prefix):
            return sub
    return None


def find_t0_variants(folder: Path):
    """Returns sorted list of (index, path) for t0_1.ext, t0_2.ext, ... in folder.
    Ignores t0.jpg (the original/base) and *_r.* (resized cache files)."""
    variants = []
    for f in folder.iterdir():
        m = re.match(r"^t0_(\d+)\.(jpg|jpeg|png)$", f.name, re.IGNORECASE)
        if m:
            variants.append((int(m.group(1)), f))
    return sorted(variants, key=lambda x: x[0])


def find_fixed_frame(folder: Path, name: str) -> Path | None:
    """Find tm3/tm2/tm1/t1/t2/t3 regardless of .jpg vs .png extension."""
    for ext in ("jpg", "jpeg", "png"):
        p = folder / f"{name}.{ext}"
        if p.exists():
            return p
    return None


def materialize_variant(variant_id: str, source_folder: Path, t0_source: Path) -> str:
    """Create the normalized folder for one variant. Returns the folder path as str."""
    out_dir = NORMALIZED_ROOT / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("tm3", "tm2", "tm1", "t1", "t2", "t3"):
        src = find_fixed_frame(source_folder, name)
        if src is None:
            continue
        dst = out_dir / f"{name}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # symlink with absolute path so it resolves regardless of cwd
        dst.symlink_to(src.resolve())

    # t0.jpg: COPY (not symlink) since we may need to convert png->jpg
    t0_dst = out_dir / "t0.jpg"
    if t0_dst.exists():
        t0_dst.unlink()
    if t0_source.suffix.lower() == ".png":
        from PIL import Image
        Image.open(t0_source).convert("RGB").save(t0_dst, quality=95)
    else:
        shutil.copy(t0_source, t0_dst)

    return str(out_dir)


def build_entry(base_sample: dict, variant_id: str, frame_dir: str,
                 scenario_text: str = "") -> dict:
    entry = {
        "sample_token": variant_id,
        "frame_dir": frame_dir,
        "n_past": base_sample.get("n_past", 3),
        "trajectory": base_sample.get("trajectory", []),
        "past_trajectory": base_sample.get("past_trajectory", []),
        "gt_continuous": base_sample.get("gt_continuous", {}),
        "prev_continuous": base_sample.get("prev_continuous", {}),
        "ego_goal": base_sample.get("ego_goal", "continue driving safely along the current road"),
        "scenario_text": scenario_text,  # injected into the frozen context at prompt-build time
    }
    return entry


def main():
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    base_samples = load_base_samples()
    scene1_text = load_scene1_scenario_text()

    all_entries = []

    # ── Scene 1: dc676572, flat 30 variants ────────────────────────────
    scene1_dir = IMAGES_ROOT / "scene 1"
    if scene1_dir.exists():
        token_dir = find_base_token_dir(scene1_dir, SCENE1_TOKEN)
        if token_dir is None:
            print(f"WARNING: no folder starting with '{SCENE1_TOKEN}' found under {scene1_dir}")
        else:
            base = base_samples.get(
                next((k for k in base_samples if k.startswith(SCENE1_TOKEN)), ""), {})
            variants = find_t0_variants(token_dir)
            print(f"Scene 1 ({SCENE1_TOKEN}): found {len(variants)} t0 variants")
            for idx, t0_path in variants:
                variant_id = f"{SCENE1_TOKEN}_v{idx:02d}"
                frame_dir = materialize_variant(variant_id, token_dir, t0_path)
                text = scene1_text.get(idx, "")
                if not text:
                    print(f"  NOTE: no scenario text found for t0_{idx} (row {idx} in xlsx)")
                all_entries.append(build_entry(base, variant_id, frame_dir, text))
    else:
        print(f"WARNING: {scene1_dir} not found, skipping scene 1.")

    # ── Scene 2: 7 tokens x 5 variants each ────────────────────────────
    scene2_dir = IMAGES_ROOT / "scene 2"
    if scene2_dir.exists():
        for token in SCENE2_TOKENS:
            token_dir = find_base_token_dir(scene2_dir, token)
            if token_dir is None:
                print(f"WARNING: no folder starting with '{token}' found under {scene2_dir}")
                continue
            base = base_samples.get(
                next((k for k in base_samples if k.startswith(token)), ""), {})
            variants = find_t0_variants(token_dir)
            print(f"Scene 2 ({token}): found {len(variants)} t0 variants")
            for idx, t0_path in variants:
                variant_id = f"{token}_v{idx:02d}"
                frame_dir = materialize_variant(variant_id, token_dir, t0_path)
                # scenario_text intentionally left blank -- meaning of
                # t0_1..t0_5 for scene 2 not yet defined. Fill in once known.
                all_entries.append(build_entry(base, variant_id, frame_dir, ""))
    else:
        print(f"WARNING: {scene2_dir} not found, skipping scene 2.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nWrote {len(all_entries)} entries to {OUTPUT_JSON}")
    print(f"Normalized folders created under {NORMALIZED_ROOT}")


if __name__ == "__main__":
    main()