# FULL-DATA UCIT Six-Task Formal Feasibility Pilot — Seed 42 — Final Report

**Status: COMPLETE** (training + continual eval + summary, 2026-08-10 11:35 UTC)
**Method**: Compose (residual-adaptive expert composition, format-controlled) on Hyper-LLaVA/LLaVA-1.5-7B
**Tasks**: ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR-Math → Flickr30k (official test_3000 per task)
**Run root**: `experiments/runs/compose_ucit_formal_seed42`
**Head commit**: `14da28b` (+ working tree fixes: openjdk-17 conda install for caption scorers)

---

## 1. Continual performance matrix A[t][j] (official 3000-sample test sets)

| Stage | ImgNetR | ArxivQA | VizWiz | IconQA | CLEVR | Flickr |
|---|---|---|---|---|---|---|
| After T0 | **26.53** | NA | NA | NA | NA | NA |
| After T1 | 31.27 | **90.30** | NA | NA | NA | NA |
| After T2 | 18.00 | 88.03 | **55.71** | NA | NA | NA |
| After T3 | 21.57 | 88.80 | 54.79 | **44.03** | NA | NA |
| After T4 | 23.50 | 89.00 | 53.33 | 44.97 | **43.47** | NA |
| After T5 | 20.97 | 89.03 | 53.62 | 44.30 | 41.07 | **51.73** |

Bold = diagonal (post-task performance). Caption tasks (VizWiz/Flickr30k) use the original COCO-style scorer (Average of BLEU1-4/METEOR/ROUGE-L/CIDEr); accuracy tasks use the original deepseek-r1 scorer. **The ORIGINAL Hyper-LLaVA scorer modules were executed verbatim** (`llava.eval.eval_caption` / `llava.eval.eval_deepseek_r1`); Compose only parses their `Result.text`.

## 2. Continual metrics (authoritative original script `summarize_continual_metrics.py`)

| Metric | Compose seed42 | Hyper-LLaVA (paper) |
|---|---|---|
| **MFN** (mean final) | **50.12** | 72.81 |
| **MAA** (mean all-acc) | **49.083305** | 81.87 |
| **MFT** (mean final-task) | **51.961667** | 78.03 |
| **BWT** (backward transfer) | **−2.21** | −5.22 |

§28 self-consistency: wrapper recomputation vs original script — **4/4 PASS, max diff 0.0** (bit-identical).
Final table: `evaluation/final_ucit_table.csv`; matrix: `evaluation/continual_matrix.{csv,json}`.

## 3. Training execution (spec §26, §29)

| Task | Exec | Loss (final) | Steps | Wall | Rate |
|---|---|---|---|---|---|
| 0 | single-GPU | 0.001 | 750 | 43.6 min | 1.15 sps |
| 1 | single-GPU | 0.0259 | 708 | 39.4 min | 1.27 sps |
| 2 | single-GPU | 0.5562 | 750 | 2.75 h | 0.30 sps |
| 3 | **4-GPU DDP** | 0.6207 | 750 | 39.3 min | 2.54 sps |
| 4 | **4-GPU DDP** | 0.5286 | 750 | 41.1 min | 2.54 sps |
| 5 | **4-GPU DDP** | 0.8192 | 750 | 40.3 min | 2.54 sps |

- **TRAINING_EXECUTION**: tasks 0–2 single-GPU (stopped per user directive at task3 S2); tasks 3–5 resumed on physical GPUs 4–7 via `torchrun --standalone --nproc_per_node=4 -m compose.experiments.task_run --first-task 3 --last-task 5 --gpus 4,5,6,7`, task-level strictly sequential (spec §1), stages sharded per §3 (features/query/teacher-NLL/LoRA-train/RMS/inference distributed; clustering/keys/commit/pool_version/snapshots rank0-only).
- **EFFECTIVE_GLOBAL_BATCH**: 16×1×4 = **64 samples/step** on 4 GPUs (per-rank 1 × grad-accum 16 × world 4); single-GPU tasks used 16×1×1. Optimizer steps identical to single-GPU protocol (750/task), verified by per-task `distributed_training_contract.json` (spec §6, §4). LR/β/warmup identical to the single-GPU formal config (spec §5, not rescaled).
- **EVALUATION_PROTOCOL**: every stage t re-scored all tasks j≤t with snapshot t on the official 3000-sample test set (incl. the t=t diagonal from S11 answers, reused). Sharded across 4 GPUs, merged in record order (spec §18); original scorers verbatim.

## 4. Method diagnostics

- **12 experts** (2 per task), pool_version 13 (POOL_VERSION_INITIAL=1 + 12 commits). Every task: cluster train features (k=2, silhouette 0.15–0.78), learnable keys (50 epochs, pos-sim 0.62–0.75), RMS calibration on 224 layers, commit 2 experts.
- **Cross-task reuse** (teacher-positive & pre-existing, on train split): expert 0 reused 249×, expert 1 reused 1511× across future tasks — retrieval demonstrably composes old experts on new tasks; recall audit OracleMemberRecall@8 = 1.0 for tasks 1–5.
- **Routing collapse audit**: all 6 tasks flag-free (empty/pair/single rates 0.0; histogram shows a healthy mix of 1- and 2-expert selections per task).
- **RMS**: fp64 sum/count/sum-sq aggregation, 224 layers calibrated per task; 4-GPU vs single parity bit-identical at equal batch shape (smoke §17).

## 5. 4-GPU scaling (smoke §25, identical code path, equal 256-sample budgets)

| Config | Steps | Mean step | Samples/s | Peak VRAM |
|---|---|---|---|---|
| 1 GPU (world1) | 128 | 0.898 s | 2.23 | 15.54 GB |
| 4 GPU (world4) | 32 | 1.289 s | **6.21** | 15.84 GB |

**Speedup 2.79× (69.7% of linear)** at constant VRAM — bounded by DDP all-reduce + activation checkpointing recompute, as expected for a 7B bf16 model with micro-batch 1.

## 6. Parity verification (smoke, one-shot 4-GPU run — ALL 8 PHASES PASS)

- §15 features (visual/text/query): max diff ≤ 1e-5 vs single-GPU.
- §17 RMS kappas: 224 layers bit-identical (max_abs_diff = 0.0) at same batch shape.
- §19 answers: byte-identical line-for-line (64-sample smoke slice; formal stages additionally scored the full 3000).
- §9 gradient audit: new-expert LoRA grads finite, all other params grad-free, post-step weight hash identical across ranks.
- §21 init identity: pre-train expert hash identical on ranks 0–3.

## 7. Notable findings / deviations (transparently logged)

1. **Missing JRE blocked caption scoring** (VizWiz/Flickr30k). Root cause: original `eval_caption` shells out to `java` (pycocoevalcap PTBTokenizer/Meteor); host had no JRE and no passwordless sudo. **Fixed** by `conda install openjdk=17` into the hyper env (no sudo). This was the pre-existing "VizWiz/Flickr30k evaluator parity failure" seen on pristine HEAD — now root-caused and resolved. Stage-3 eval had partially completed (A[3][0]/A[3][1] scored) before crashing; the eval chain was re-run stages 0→5, reusing already-generated predictions.
2. **bf16 GEMM shape-sensitivity**: RMS parity diffs (~1e-3) were entirely batch-shape artifacts (batch-of-4 per-rank vs batch-of-8 reference); same shape ⇒ bit-identical.
3. **world1 cuDNN quirk**: DDP(world1) forward on vision conv raises CUDNN_STATUS_NOT_INITIALIZED; smoke wraps DDP only for world>1 (semantic no-op), verified on both world1 and world4.
4. **ImageNet-R performance is low and forgets** (26.5 → 31.3 → 18.0 → … → 21.0; paper 87.2). ArxivQA/VizWiz approach or exceed feasibility thresholds (89.0 / 53.6 vs 93.7 / 57.24). This is an honest result of the format-controlled composition protocol on seed 42; per the pilot's mandate, results are reported as measured without tuning.
5. BWT −2.21 vs paper −5.22: less negative backward transfer, but from a much lower per-task floor.

## 8. Files produced

```
evaluation/continual_matrix.csv           6×6 matrix (official test_3000)
evaluation/continual_matrix.json
evaluation/continual_metrics_original.json  authoritative original-script metrics
evaluation/continual_metrics_wrapper.json   wrapper recomputation (§28)
evaluation/final_ucit_table.csv            Hyper-LLaVA-paper row + Compose row
evaluation/compose_method_diagnostics.json diagnostics (growth/reuse/keys/routing/RMS)
evaluation/hyper_result_root/{Dataset}/hyper-task{t+1}/Result.text  mirrored layout
task{t}/report/task_report.md              per-task report (experts, keys, commits, RMS, eval)
```

**Per spec §30: the seed-42 pilot stops here. seed43/44 are NOT started automatically.**
