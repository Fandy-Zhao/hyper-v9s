"""Create the required Compose P1-P3 final reports and reproduction script."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    decisions = {stage: read_json(root / "gate_decisions/{}_decision.json".format(stage)) for stage in ("p1", "p2", "p3")}
    scheduler_states = [read_json(path) for path in sorted((root / "logs/scheduler_state").glob("*.json"))]
    commands = [attempt for state in scheduler_states for attempt in state.get("attempts", [])]
    gpu_hours = sum(float(item["elapsed_seconds"]) for item in commands) / 3600.0
    used_gpus = sorted({int(item["cuda_visible_devices"]) for item in commands})
    if any(device not in tuple(range(8)) for device in used_gpus):
        raise RuntimeError("final audit found forbidden GPU")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    allow_benchmark = all(decisions[stage].get("formal") is not False and decisions[stage]["decision"].startswith("PASS") for stage in decisions)
    final = {
        "method": "Compose: Keyed Functional Expert Composition",
        "git_commit": commit,
        "output_root": str(root),
        "environment": {"python": "3.10.20", "pytorch": "2.3.1+cu118", "cuda": "11.8", "transformers": "4.33.3", "peft": "0.4.0"},
        "actual_physical_gpus": used_gpus,
        "gpu_policy": "prefer physical GPUs 0-3; use authorized fallback 4-7 when 0-3 are occupied",
        "gpu_hours": gpu_hours,
        "commands": commands,
        "decisions": {stage: value["decision"] for stage, value in decisions.items()},
        "formal": {stage: value.get("formal", stage == "p1") for stage, value in decisions.items()},
        "allow_full_continual_benchmark": allow_benchmark,
        "supported_claims": ["P1 formal result under arithmetic-mean RMS calibration", "P2/P3 engineering smoke behavior"] if decisions["p1"]["decision"] != "PASS_COMPOSITION" else ["P1 formal composition result"],
        "unsupported_claims": ["task-internal multi-function discovery", "closed-loop router superiority", "full continual-learning superiority"] if not allow_benchmark else [],
    }
    (root / "reports/Compose_P1_P3_Final_Report.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    summary_rows = [
        {"stage": stage.upper(), "decision": value["decision"], "formal": final["formal"][stage], "allow_next": stage == "p1" and value["decision"] == "PASS_COMPOSITION" or stage != "p1"}
        for stage, value in decisions.items()
    ]
    with (root / "metrics/Compose_P1_P3_Summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    final_decision = {
        "allow_full_continual_benchmark": allow_benchmark,
        "stage_decisions": final["decisions"],
        "reason": "all formal gates passed" if allow_benchmark else "one or more formal gates failed or later stages were gate-limited smoke diagnostics",
    }
    (root / "gate_decisions/final_decision.json").write_text(json.dumps(final_decision, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Compose P1-P3 Final Report", "", "Method: **Compose: Keyed Functional Expert Composition**", "",
        "- Commit: `{}`".format(commit),
        "- Physical GPUs used: {}".format(used_gpus or "none (cached-feature diagnostics only)"),
        "- Total recorded GPU hours: {:.4f}".format(gpu_hours),
        "- P1: `{}`".format(final["decisions"]["p1"]),
        "- P2: `{}` ({})".format(final["decisions"]["p2"], "formal" if final["formal"]["p2"] else "single-seed smoke only"),
        "- P3: `{}` ({})".format(final["decisions"]["p3"], "formal" if final["formal"]["p3"] else "single-seed smoke only"),
        "- Allow full continual benchmark: **{}**".format("yes" if allow_benchmark else "no"), "",
        "## Commands", "",
    ]
    lines.extend("- GPU {cuda_visible_devices}, seed {seed}: `{command}`".format(**item) for item in commands)
    lines += [
        "", "## Gate interpretation", "",
        "P2/P3 smoke results do not override a failed or borderline P1 gate and do not support formal method claims.", "",
        "## Unique next step", "",
        "Do not start the full continual benchmark unless the formal P1 unseen-composition gate passes; otherwise redesign the functional experts before repeating this fixed protocol.", "",
    ]
    (root / "reports/Compose_P1_P3_Final_Report.md").write_text("\n".join(lines), encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    reproduce = scripts / "reproduce_compose_p1_p3.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nROOT=/home/zhaozhuofan/Hyper-LlaVA\nOUT={out}\nPY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python\ncd \"$ROOT\"\n\"$PY\" -m compose.experiments.build_p1_manifest --output-root \"$OUT\"\nOUTPUT_ROOT=\"$OUT\" scripts/run_compose_p1_p3_4gpu.sh\n\"$PY\" -m compose.eval.aggregate_compose_p1 --output-root \"$OUT\"\n\"$PY\" -m compose.experiments.run_gated_smokes --output-root \"$OUT\"\n\"$PY\" -m compose.experiments.finalize --output-root \"$OUT\"\n".format(out=root),
        encoding="utf-8",
    )
    reproduce.chmod(0o755)
    print(json.dumps(final_decision, sort_keys=True))


if __name__ == "__main__":
    main()
