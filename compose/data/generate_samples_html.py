"""
Generate an HTML page showing human inspection samples from controlled_format_v1.
Each task and split: 20 positive and 20 negative samples randomly sampled.
"""

import argparse
import base64
import json
import os
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PIL import Image

FUNCTIONS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")
SPLITS = ("train", "val", "test")
SAMPLES_PER_POLARITY = 20

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Format-Controlled Dataset v1 — Human Inspection Samples</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
  h1 { color: #333; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #2c3e50; margin-top: 40px; background: #ecf0f1; padding: 10px 15px; border-radius: 5px; }
  h3 { color: #7f8c8d; margin-top: 30px; }
  .task-section { margin-bottom: 50px; }
  .split-section { margin-bottom: 30px; }
  .polarity-section { margin: 20px 0; }
  .polarity-label { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }
  .polarity-label.positive { color: #27ae60; }
  .polarity-label.negative { color: #e74c3c; }
  .samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
                   gap: 20px; }
  .sample-card { background: white; border-radius: 8px; padding: 15px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); font-size: 13px; }
  .sample-card img { max-width: 100%; height: auto; border: 1px solid #ddd;
                     border-radius: 4px; display: block; }
  .sample-id { color: #95a5a6; font-size: 11px; margin-bottom: 5px; }
  .sample-q { font-weight: 600; margin: 8px 0; color: #2c3e50; }
  .sample-a { font-size: 1.2em; font-weight: bold; }
  .sample-a.A { color: #27ae60; }
  .sample-a.B { color: #e74c3c; }
  .sample-meta { color: #7f8c8d; font-size: 11px; margin-top: 5px;
                 word-break: break-all; }
  .summary-box { background: white; border-radius: 8px; padding: 20px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 30px; }
  .summary-box table { border-collapse: collapse; width: 100%; }
  .summary-box th { background: #3498db; color: white; padding: 8px 12px; text-align: left; }
  .summary-box td { padding: 6px 12px; border-bottom: 1px solid #eee; }
  .summary-box tr:nth-child(even) { background: #f8f9fa; }
  .stat-ok { color: #27ae60; }
  .stat-warn { color: #e67e22; }
</style>
</head>
<body>
<h1>🔍 Format-Controlled Dataset v1 — Human Inspection Samples</h1>

<div class="summary-box">
  <h3>Dataset Summary</h3>
  <table>
    <tr><th>Property</th><th>Value</th></tr>
    {summary_rows}
  </table>
</div>

{sections}
</body>
</html>"""

SAMPLE_CARD = """
<div class="sample-card">
  <div class="sample-id">ID: {sample_id} | Scene: {scene_id} | negative_type: {negative_type}</div>
  <img src="data:image/png;base64,{img_b64}" alt="{scene_id}" loading="lazy">
  <div class="sample-q">{question}</div>
  <div class="sample-a {answer}">Answer: {answer} ({answer_text}) | Polarity: {polarity}</div>
  <div class="sample-meta">Metadata: {metadata_json}</div>
</div>"""


def image_to_b64(image_path: Path) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_samples_page(data_root: str, output_path: str) -> None:
    root = Path(data_root)
    random.seed(42)  # fixed seed for sampling reproducibility

    # Count samples
    total = 0
    balance_info = []
    for task in FUNCTIONS:
        for split in SPLITS:
            with open(root / task / f"{split}.json", "r", encoding="utf-8") as f:
                samples = json.load(f)
            total += len(samples)
            a_count = sum(1 for s in samples if s["answer"] == "A")
            b_count = sum(1 for s in samples if s["answer"] == "B")
            balance_info.append(f"<tr><td>{task}/{split}</td><td>{len(samples)}</td>"
                              f"<td class='stat-ok'>{a_count}</td><td class='stat-ok'>{b_count}</td>"
                              f"<td class='stat-ok'>{a_count/len(samples)*100:.1f}% / {b_count/len(samples)*100:.1f}%</td></tr>")

    summary_rows = f"""
    <tr><td>Total Scenes</td><td>2200</td></tr>
    <tr><td>Total QA Samples</td><td>{total}</td></tr>
    <tr><td>Tasks</td><td>{', '.join(FUNCTIONS)}</td></tr>
    <tr><td>Answer Format</td><td>A = Yes, B = No</td></tr>
    <tr><td>Per-Task Train/Val/Test</td><td>1600 / 200 / 400</td></tr>
    """

    sections_html = ""

    for task in FUNCTIONS:
        sections_html += f"<div class='task-section'><h2>📋 {task}</h2>"

        for split in SPLITS:
            sections_html += f"<div class='split-section'><h3>Split: {split}</h3>"

            with open(root / task / f"{split}.json", "r", encoding="utf-8") as f:
                all_samples = json.load(f)

            pos_samples = [s for s in all_samples if s["polarity"] == "positive"]
            neg_samples = [s for s in all_samples if s["polarity"] == "negative"]

            pos_sample = random.sample(pos_samples, min(SAMPLES_PER_POLARITY, len(pos_samples)))
            neg_sample = random.sample(neg_samples, min(SAMPLES_PER_POLARITY, len(neg_samples)))

            for polarity, samples in [("positive", pos_sample), ("negative", neg_sample)]:
                label_class = "positive" if polarity == "positive" else "negative"
                sections_html += (
                    f"<div class='polarity-section'>"
                    f"<div class='polarity-label {label_class}'>"
                    f"{'✅' if polarity == 'positive' else '❌'} "
                    f"{polarity.upper()} ({len(samples)} samples shown)"
                    f"</div>"
                    f"<div class='samples-grid'>"
                )

                for s in samples:
                    img_path = root / s["image"]
                    if not img_path.exists():
                        img_b64 = "IMAGE_NOT_FOUND"
                    else:
                        img_b64 = image_to_b64(img_path)

                    neg_type = s.get("negative_type") or "none"
                    answer_text = "Yes" if s["answer"] == "A" else "No"

                    card = SAMPLE_CARD.format(
                        sample_id=s["id"],
                        scene_id=s["scene_id"],
                        negative_type=neg_type,
                        img_b64=img_b64,
                        question=s["question"],
                        answer=s["answer"],
                        answer_text=answer_text,
                        polarity=s["polarity"],
                        metadata_json=json.dumps(s["metadata"], ensure_ascii=False),
                    )
                    sections_html += card

                sections_html += "</div></div>"

            sections_html += "</div>"  # split-section

        sections_html += "</div>"  # task-section

    html = (HTML_TEMPLATE
            .replace("{summary_rows}", summary_rows)
            .replace("{sections}", sections_html))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML inspection page written to: {output_path}")
    print(f"  Tasks: {len(FUNCTIONS)}")
    print(f"  Splits per task: {len(SPLITS)}")
    print(f"  Samples per polarity: {SAMPLES_PER_POLARITY}")
    print(f"  Total cards: {len(FUNCTIONS) * len(SPLITS) * 2 * SAMPLES_PER_POLARITY}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_samples_page(args.data_root, args.output)


if __name__ == "__main__":
    main()
