"""
Build a single self-contained HTML report from the Hamiltonian and DriveLM
result files, with the real frame sequence, the shared perception/
comprehension/prediction summaries (frozen upstream, same for both prompts),
and both models' Step 4 planning + reasoning shown side by side.

Run this from the repo root, wherever the results and normalized images
actually live (WSL or DAIC):
    python generate_report.py

Output: results/comparison_report.html
"""

import json
import base64
import os
from pathlib import Path

HAMILTON_FILE = "results/scenario_hamiltonian.json"
DRIVELM_FILE  = "results/scenario_drivelm.json"
OUTPUT_FILE   = "results/comparison_report.html"


def img_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_frames(frame_dir):
    frames = []
    for name, label in [("tm3", "t-3"), ("tm2", "t-2"), ("tm1", "t-1"), ("t0", "t0")]:
        path = os.path.join(frame_dir, f"{name}.jpg")
        data = img_to_base64(path)
        if data:
            frames.append((label, data))
    return frames


def highlight_diff(a, b):
    a_words, b_words = a.split(), b.split()
    same = set(a_words) & set(b_words)
    a_html = " ".join(w if w in same else f"<mark>{w}</mark>" for w in a_words)
    b_html = " ".join(w if w in same else f"<mark>{w}</mark>" for w in b_words)
    return a_html, b_html


def main():
    with open(HAMILTON_FILE) as f:
        ham_data = json.load(f)
    with open(DRIVELM_FILE) as f:
        drivelm_data = json.load(f)

    drivelm_by_token = {r["sample_token"]: r for r in drivelm_data}

    sections = []
    for r in ham_data:
        token = r["sample_token"]
        dl = drivelm_by_token.get(token)
        if dl is None:
            continue

        ham_res = r["non_posthoc"]
        dl_res  = dl["non_posthoc"]

        ham_speed = ham_res.get("speed", "")
        dl_speed  = dl_res.get("speed_drivelm", "")
        ham_dir   = ham_res.get("direction", "")
        dl_dir    = dl_res.get("direction_drivelm", "")
        agree = (ham_speed == dl_speed) and (ham_dir == dl_dir)

        ham_plan_h, dl_plan_h = highlight_diff(
            ham_res.get("planning", ""), dl_res.get("planning_drivelm", ""))

        frame_dir = f"data/scenarios/normalized/{token}"
        frames = get_frames(frame_dir)
        frame_html = "".join(
            f'<div class="frame"><img src="data:image/jpeg;base64,{data}"><span>{label}</span></div>'
            for label, data in frames
        )

        agree_badge = (
            '<span class="badge agree">AGREE</span>' if agree
            else '<span class="badge disagree">DISAGREE</span>'
        )

        # Steps 1-3 are frozen upstream, shared by both prompts -- read from
        # the Hamiltonian result (identical to DriveLM's copy since both
        # pipelines inherit the same Steps 1-3 output).
        perception    = ham_res.get("perception_summary", "")
        comprehension = ham_res.get("comprehension_summary", "")
        prediction    = ham_res.get("prediction_summary", "")

        sections.append(f"""
        <div class="scenario">
          <div class="scenario-header">
            <h2>{token}</h2>
            {agree_badge}
          </div>
          <div class="scenario-text"><strong>Scenario:</strong> {r.get('scenario_text', '(none)')}</div>
          <div class="frames">{frame_html}</div>
          <div class="sa-summary">
            <p><strong>Perception:</strong> {perception}</p>
            <p><strong>Comprehension:</strong> {comprehension}</p>
            <p><strong>Prediction:</strong> {prediction}</p>
          </div>
          <div class="compare">
            <div class="col hamiltonian">
              <h3>Hamiltonian</h3>
              <p><strong>Planning:</strong> {ham_plan_h}</p>
              <p><strong>Behavior:</strong> direction={ham_dir}, speed={ham_speed}</p>
              <p class="reasoning"><strong>Reasoning:</strong> {ham_res.get('reasoning', '')}</p>
            </div>
            <div class="col drivelm">
              <h3>DriveLM</h3>
              <p><strong>Planning:</strong> {dl_plan_h}</p>
              <p><strong>Behavior:</strong> direction={dl_dir}, speed={dl_speed}</p>
              <p class="reasoning"><strong>Reasoning:</strong> {dl_res.get('reasoning_drivelm', '')}</p>
            </div>
          </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hamiltonian vs DriveLM — Scenario Comparison Report</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ font-size: 20px; }}
  .controls {{ margin-bottom: 16px; }}
  .controls input {{ padding: 6px 10px; width: 300px; font-size: 14px; }}
  .scenario {{ background: #1c1c1c; border: 1px solid #333; border-radius: 8px;
               padding: 16px; margin-bottom: 20px; }}
  .scenario-header {{ display: flex; align-items: center; gap: 10px; }}
  .scenario-header h2 {{ font-size: 15px; margin: 0; font-family: monospace; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
  .badge.agree {{ background: #1e5c2e; color: #a6e3a1; }}
  .badge.disagree {{ background: #5c1e1e; color: #f38ba8; }}
  .scenario-text {{ margin: 8px 0; font-size: 13px; color: #bbb; }}
  .frames {{ display: flex; gap: 6px; margin: 10px 0; }}
  .frame {{ text-align: center; }}
  .frame img {{ width: 140px; border-radius: 4px; display: block; }}
  .frame span {{ font-size: 10px; color: #888; }}
  .sa-summary {{ background: #151515; padding: 10px 14px; border-radius: 6px;
                 margin: 10px 0; border-left: 3px solid #6c7086; }}
  .sa-summary p {{ font-size: 12px; line-height: 1.5; margin: 4px 0; color: #ccc; }}
  .compare {{ display: flex; gap: 20px; margin-top: 10px; }}
  .col {{ flex: 1; background: #151515; padding: 10px; border-radius: 6px; }}
  .col h3 {{ margin-top: 0; font-size: 13px; }}
  .col.hamiltonian h3 {{ color: #89b4fa; }}
  .col.drivelm h3 {{ color: #f9e2af; }}
  .col p {{ font-size: 12px; line-height: 1.5; }}
  .reasoning {{ color: #aaa; }}
  mark {{ background: #574315; color: #fff; padding: 0 2px; border-radius: 2px; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<h1>Hamiltonian vs DriveLM — {len(sections)} scenarios</h1>
<div class="controls">
  <input type="text" id="search" placeholder="Filter by sample_token or scenario text..."
         onkeyup="filterScenarios()">
  <label style="margin-left: 16px; font-size: 13px;">
    <input type="checkbox" id="onlyDisagree" onchange="filterScenarios()"> Show disagreements only
  </label>
</div>
<div id="scenarios">
{''.join(sections)}
</div>
<script>
function filterScenarios() {{
  const q = document.getElementById('search').value.toLowerCase();
  const onlyDisagree = document.getElementById('onlyDisagree').checked;
  document.querySelectorAll('.scenario').forEach(el => {{
    const text = el.innerText.toLowerCase();
    const isDisagree = el.querySelector('.badge.disagree') !== null;
    const matchesText = text.includes(q);
    const matchesFilter = !onlyDisagree || isDisagree;
    el.classList.toggle('hidden', !(matchesText && matchesFilter));
  }});
}}
</script>
</body>
</html>
"""

    Path("results").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)
    print(f"Wrote {len(sections)} scenarios to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
