"""Extract preregistered failure cases for the format-controlled experiment.

Categories (section 十七):
  1. 50 lowest-synergy samples per pair
  2. pair wrong / best single right
  3. pair right / best single wrong
  4. pair wrong / rank16 right
  5. samples failing in all three seeds
  6. worst failure per negative_type
  7. Residual B improves B_only but hurts B_plus_C

Outputs failure_cases.json and an HTML report embedding the scene images.
"""

import argparse
import base64
import json
import statistics
from pathlib import Path

SEEDS = (42, 43, 44)

PAIRS = [
    ("A_plus_B", "a_residual_b", ("expert_a", "residual_b"), "rank16_ab"),
    ("A_plus_B", "a_independent_b", ("expert_a", "independent_b"), "rank16_ab"),
    ("B_plus_C", "residual_b_c", ("residual_b", "expert_c"), "upper_bc"),
    ("B_plus_C", "independent_b_c", ("independent_b", "expert_c"), "upper_bc"),
]


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load(root: Path, seed: int, dataset: str, model: str):
    rows = _read_jsonl(root / "seed{}".format(seed) / dataset / model / "per_sample.jsonl")
    return {str(row["sample_id"]): row for row in rows}


def _case_record(seed: int, dataset: str, pair_name: str, sample_id: str,
                 rows: dict, singles: tuple, rank16: dict) -> dict:
    pair = rows["pair"][sample_id]
    single_a = rows[singles[0]][sample_id]
    single_b = rows[singles[1]][sample_id]
    rank16_row = rank16[sample_id]
    best_single = single_a if (
        single_a["answer_token_nll"] <= single_b["answer_token_nll"]
    ) else single_b
    synergy = min(single_a["answer_token_nll"], single_b["answer_token_nll"]) - pair["answer_token_nll"]
    best_single_correct = bool(single_a["correct"] or single_b["correct"])
    return {
        "seed": seed, "dataset": dataset, "pair_name": pair_name, "sample_id": sample_id,
        "scene_id": pair["scene_id"], "image": pair.get("image", ""),
        "question": pair["question"], "target": pair["target"],
        "polarity": pair["polarity"], "negative_type": pair["negative_type"],
        "required_functions": pair["required_functions"],
        "queried_shape": pair["queried_shape"], "queried_count": pair["queried_count"],
        "queried_relation": pair["queried_relation"], "true_count": pair["true_count"],
        "options": pair["options"],
        "pair": {
            "prediction": pair["prediction"], "correct": pair["correct"],
            "logit_A": pair["logit_A"], "logit_B": pair["logit_B"],
            "probability_A": pair["probability_A"], "answer_token_nll": pair["answer_token_nll"],
        },
        "best_single": {
            "name": singles[0] if single_a["answer_token_nll"] <= single_b["answer_token_nll"] else singles[1],
            "prediction": best_single["prediction"], "correct": best_single["correct"],
            "answer_token_nll": best_single["answer_token_nll"],
        },
        "single_a": {"name": singles[0], "prediction": single_a["prediction"],
                     "correct": single_a["correct"], "answer_token_nll": single_a["answer_token_nll"]},
        "single_b": {"name": singles[1], "prediction": single_b["prediction"],
                     "correct": single_b["correct"], "answer_token_nll": single_b["answer_token_nll"]},
        "rank16": {"name": rank16_row["selection_name"], "prediction": rank16_row["prediction"],
                   "correct": rank16_row["correct"], "answer_token_nll": rank16_row["answer_token_nll"]},
        "synergy": synergy,
        "best_single_correct": best_single_correct,
        "rank16_correct": bool(rank16_row["correct"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--limit-worst", type=int, default=50)
    args = parser.parse_args()
    root = Path(args.evaluation_root)
    image_folder = Path(args.image_folder)

    # load all rows once per seed/dataset/model
    all_rows = {}
    for seed in SEEDS:
        all_rows[seed] = {}
        for dataset, pair_name, singles, rank16_name in PAIRS:
            models = [pair_name, *singles, rank16_name]
            all_rows[seed].setdefault(dataset, {}).update({
                model: _load(root, seed, dataset, model) for model in models
            })

    cases = []
    for dataset, pair_name, singles, rank16_name in PAIRS:
        ids = sorted(all_rows[SEEDS[0]][dataset][pair_name])
        seed_rows = [
            {
                "pair": all_rows[seed][dataset][pair_name],
                "single_a": all_rows[seed][dataset][singles[0]],
                "single_b": all_rows[seed][dataset][singles[1]],
                "rank16": all_rows[seed][dataset][rank16_name],
            }
            for seed in SEEDS
        ]
        for seed_index, seed in enumerate(SEEDS):
            rows = seed_rows[seed_index]
            for sample_id in ids:
                case = _case_record(seed, dataset, pair_name, sample_id, {
                    "pair": rows["pair"], singles[0]: rows["single_a"],
                    singles[1]: rows["single_b"],
                }, singles, rows["rank16"])
                cases.append(case)

        # category 5: failing in all three seeds (pair wrong in all seeds)
        repeated = set()
        for sample_id in ids:
            if all(not all_rows[seed][dataset][pair_name][sample_id]["correct"] for seed in SEEDS):
                repeated.add(sample_id)

        grouped = {}
        for seed in SEEDS:
            for sample_id in ids:
                case = _case_record(seed, dataset, pair_name, sample_id, {
                    "pair": all_rows[seed][dataset][pair_name],
                    singles[0]: all_rows[seed][dataset][singles[0]],
                    singles[1]: all_rows[seed][dataset][singles[1]],
                }, singles, all_rows[seed][dataset][rank16_name])
                grouped.setdefault(sample_id, []).append(case)

        # per-seed category lists
        for seed in SEEDS:
            cases_for_seed = [c for c in cases if c["seed"] == seed and c["dataset"] == dataset and c["pair_name"] == pair_name]
            lowest = sorted(cases_for_seed, key=lambda c: c["synergy"])[:args.limit_worst]
            pair_wrong_best_right = [c for c in cases_for_seed if not c["pair"]["correct"] and c["best_single_correct"]]
            pair_right_best_wrong = [c for c in cases_for_seed if c["pair"]["correct"] and not c["best_single_correct"]]
            pair_wrong_rank16_right = [c for c in cases_for_seed if not c["pair"]["correct"] and c["rank16_correct"]]

    # assemble final structure
    final = {
        "pairs": {},
        "category_summaries": {},
    }
    for dataset, pair_name, singles, rank16_name in PAIRS:
        final["pairs"][pair_name] = {
            "dataset": dataset, "singles": list(singles), "rank16": rank16_name,
            "cases": [],
        }
    for case in cases:
        final["pairs"][case["pair_name"]]["cases"].append(case)
    for pair_name, entry in final["pairs"].items():
        per_seed = {seed: [c for c in entry["cases"] if c["seed"] == seed] for seed in SEEDS}
        sample_ids = sorted({c["sample_id"] for c in entry["cases"]})
        repeated = [
            sid for sid in sample_ids
            if all(not c["pair"]["correct"] for seed in SEEDS
                   for c in per_seed[seed] if c["sample_id"] == sid)
        ]
        worst = sorted(entry["cases"], key=lambda c: c["synergy"])[:args.limit_worst]
        entry["worst_synergy"] = [c["sample_id"] for c in worst]
        entry["repeated_failures_all_seeds"] = repeated
        # worst per negative type (mean synergy across seeds)
        by_type = {}
        for sid in sample_ids:
            type_rows = [next(c for c in per_seed[seed] if c["sample_id"] == sid) for seed in SEEDS]
            key = type_rows[0]["negative_type"] or "positive"
            mean_synergy = statistics.fmean(c["synergy"] for c in type_rows)
            by_type.setdefault(key, []).append((sid, mean_synergy))
        entry["worst_by_negative_type"] = {
            key: min(items, key=lambda item: item[1])[0]
            for key, items in by_type.items()
        }
        entry["counts"] = {
            "total_case_rows": len(entry["cases"]),
            "unique_samples": len(sample_ids),
            "repeated_failures_all_seeds": len(repeated),
        }
    final["summary"] = {
        pair_name: entry["counts"] for pair_name, entry in final["pairs"].items()
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- HTML report --------------------------------------------------------
    def img_data_uri(image: str) -> str:
        path = image_folder / image
        if not path.is_file():
            return ""
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    def seed_consistency(cases_for_sample):
        return {
            "all_wrong": all(not c["pair"]["correct"] for c in cases_for_sample),
            "all_right": all(c["pair"]["correct"] for c in cases_for_sample),
            "mixed": len({c["pair"]["correct"] for c in cases_for_sample}) == 2,
            "correct_by_seed": [bool(c["pair"]["correct"]) for c in cases_for_sample],
        }

    sections = []
    for pair_name, entry in final["pairs"].items():
        dataset = entry["dataset"]
        rows_html = []
        sample_ids = sorted({c["sample_id"] for c in entry["cases"]})
        for sid in sample_ids:
            cases_for_sample = sorted(
                [c for c in entry["cases"] if c["sample_id"] == sid], key=lambda c: c["seed"]
            )
            base = cases_for_sample[0]
            consistency = seed_consistency(cases_for_sample)
            img = img_data_uri(base["image"])
            rows_html.append(f"""
<tr>
  <td><img src="{img}" width="120" loading="lazy"/></td>
  <td><b>{base['sample_id']}</b><br/>scene {base['scene_id']}<br/>{base['polarity']} / {base['negative_type']}</td>
  <td>{base['question']}<br/><small>target: {base['target']} | required: {base['required_functions']}</small></td>
  <td>{''.join(
      f"{c['seed']}: {c['pair']['prediction']}({'✓' if c['pair']['correct'] else '✗'})<br/>" for c in cases_for_sample
  )}</td>
  <td>{''.join(f"{c['seed']}: {c['synergy']:.3f}<br/>" for c in cases_for_sample)}</td>
  <td>{''.join(f"{c['seed']}: {c['best_single']['name']}={c['best_single']['prediction']}{'✓' if c['best_single']['correct'] else '✗'}<br/>" for c in cases_for_sample)}</td>
  <td>{''.join(f"{c['seed']}: {c['rank16']['prediction']}{'✓' if c['rank16']['correct'] else '✗'}<br/>" for c in cases_for_sample)}</td>
  <td>{base['true_count']}<br/>{'all-wrong' if consistency['all_wrong'] else ('all-right' if consistency['all_right'] else 'mixed')}</td>
</tr>""")
        table = f"""
<h2>{pair_name} <span class="small">({dataset})</span></h2>
<p>unique samples: {entry['counts']['unique_samples']} | repeated failures in all 3 seeds: {entry['counts']['repeated_failures_all_seeds']}</p>
<table>
<tr><th>image</th><th>sample / scene</th><th>question</th><th>pair pred by seed</th><th>synergy by seed</th><th>best single by seed</th><th>rank16 by seed</th><th>true count / consistency</th></tr>
{''.join(rows_html)}
</table>"""
        sections.append(table)

    worst_sections = []
    for pair_name, entry in final["pairs"].items():
        worst = entry["worst_synergy"][:20]
        html_rows = []
        for sid in worst:
            cases_for_sample = sorted(
                [c for c in entry["cases"] if c["sample_id"] == sid], key=lambda c: c["seed"]
            )
            base = cases_for_sample[0]
            img = img_data_uri(base["image"])
            html_rows.append(f"""
<tr><td><img src="{img}" width="100" loading="lazy"/></td>
<td>{base['sample_id']}<br/>{base['question']}</td>
<td>{''.join(f"{c['seed']}: {c['synergy']:.3f}<br/>" for c in cases_for_sample)}</td>
<td>{''.join(f"{c['seed']}: {c['pair']['prediction']}{'✓' if c['pair']['correct'] else '✗'}<br/>" for c in cases_for_sample)}</td></tr>""")
        worst_sections.append(f"""
<h2>{pair_name}: worst-20 synergy</h2>
<table><tr><th>image</th><th>sample</th><th>synergy by seed</th><th>pair pred by seed</th></tr>
{''.join(html_rows)}
</table>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Format-Controlled Failure Cases</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 32px; }}
.small {{ color: #666; font-weight: normal; }}
table {{ border-collapse: collapse; margin-top: 8px; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px; font-size: 12px; vertical-align: top; }}
th {{ background: #f2f2f2; }}
</style></head><body>
<h1>Format-Controlled Functional Expert Composition — Failure Cases</h1>
<p>Generated {args.output_json}. Cases per pair; each row shows all three seeds (42/43/44).</p>
{''.join(worst_sections)}
<hr/>
{''.join(sections)}
</body></html>"""
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(json.dumps(final["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
