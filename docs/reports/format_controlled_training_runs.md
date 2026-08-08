# Format-Controlled Training Runs

21 formal runs (7 model types x 3 seeds), one epoch each, global batch 64, AdamW lr 2e-4, cosine, warmup 0.03, bf16, LoRA scale 2 (alpha=2*rank), target modules q,k,v,o,gate,up,down, seeds 42/43/44. Data: `experiments/data/controlled_format_v1_training` (all answers unified to single-token A/B; 1600/200/400 train/val/test per task; scene splits disjoint).

## Run matrix

| seed | model | rank | active | trainable | data | params | status | source |
|---|---|---|---|---|---|---|---|---|
| 42 | expert_a | 8 | 0 | 0 | A_only | 19988480 | OK | primary |
| 42 | independent_b | 8 | 1 | 1 | B_only | 19988480 | OK | retry_gpu45 |
| 42 | expert_c | 8 | 2 | 2 | C_only | 19988480 | OK | primary |
| 42 | residual_b | 8 | 0,1 | 1 | A_plus_B | 39976960 | OK | primary |
| 42 | rank16_ab | 16 | 0 | 0 | A_plus_B | 39976960 | OK | retry_gpu45 |
| 42 | task_ab | 8 | 0 | 0 | A_plus_B | 19988480 | OK | primary |
| 42 | upper_bc | 16 | 0 | 0 | B_plus_C | 39976960 | OK | retry_gpu45 |
| 43 | expert_a | 8 | 0 | 0 | A_only | 19988480 | OK | primary |
| 43 | independent_b | 8 | 1 | 1 | B_only | 19988480 | OK | primary |
| 43 | expert_c | 8 | 2 | 2 | C_only | 19988480 | OK | primary |
| 43 | residual_b | 8 | 0,1 | 1 | A_plus_B | 39976960 | OK | primary |
| 43 | rank16_ab | 16 | 0 | 0 | A_plus_B | 39976960 | OK | primary |
| 43 | task_ab | 8 | 0 | 0 | A_plus_B | 19988480 | OK | primary |
| 43 | upper_bc | 16 | 0 | 0 | B_plus_C | 39976960 | OK | primary |
| 44 | expert_a | 8 | 0 | 0 | A_only | 19988480 | OK | primary |
| 44 | independent_b | 8 | 1 | 1 | B_only | 19988480 | OK | primary |
| 44 | expert_c | 8 | 2 | 2 | C_only | 19988480 | OK | primary |
| 44 | residual_b | 8 | 0,1 | 1 | A_plus_B | 39976960 | OK | primary |
| 44 | rank16_ab | 16 | 0 | 0 | A_plus_B | 39976960 | OK | primary |
| 44 | task_ab | 8 | 0 | 0 | A_plus_B | 19988480 | OK | primary |
| 44 | upper_bc | 16 | 0 | 0 | B_plus_C | 39976960 | OK | primary |

## Incidents (all preserved, none silent)

1. **seed42 independent_b primary run: CUDA OOM** on GPUs 4,5,6,7 (another user's process occupies ~8.4 GiB on GPU 6). Failure log preserved at `logs/training/seed42/independent_b/`. Retried on GPUs 4,5 with gradient accumulation 4 (global batch 64 preserved); retry logs at `logs/training_retry_gpu45/seed42/independent_b/`.
2. **seed42 upper_bc initially trained at rank 8** (runner inherited an old-F2 bug: the upper_bc case omitted the rank override). Fixed to rank 16 / alpha 32 / expected 39,976,960 params and retrained; the rank-8 checkpoint was quarantined at `seed42/upper_bc.bad_rank8` (not deleted).
3. **Evaluation logits-position bug** (LM convention: logits[t] predicts token t+1): the first evaluation pass used the answer-token position instead of answer_pos - 1. Fixed in `compose/eval/eval_controlled_ab.py`; all 90 evaluations were rerun; invalid results quarantined at `evaluation/seed42_bug_v1_invalid`.
4. **seed42 evaluation OOM on GPU 6** (3 jobs): evaluation workers moved to GPUs 0-5,7 (7 workers); OOM eval log preserved.

## Checkpoint verification

All 21 checkpoints verified: parameter counts exactly 19,988,480 (rank 8) or 39,976,960 (rank 16 / two rank-8 experts). Checkpoint selection rule: final epoch, one epoch, no test-set selection, no resume.
