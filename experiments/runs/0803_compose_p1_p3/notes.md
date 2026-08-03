# Compose P1-P3 Notes

## Initial provenance

- Branch: `exp/0803-compose-p1-p3`
- Parent commit: `e0a948538c4df96523a73d639b70ffff2a8e594f`
- Output root: `outputs/compose_p1_p3_20260803T090000Z/`
- Environment: Python 3.10.20, PyTorch 2.3.1+cu118, CUDA 11.8,
  Transformers 4.33.3, PEFT 0.4.0.
- GPU policy: prefer physical GPUs 0-3; when they are occupied, the user
  explicitly authorized physical GPUs 4-7 on 2026-08-03.

## Preserved pre-existing changes

The modified Stage 01 report and all pre-existing untracked format/data files
were present before this task. They are not part of this work and must not be
staged, overwritten, or removed.

## Initial evidence

Verified functional experts and matched controls exist for checkpoint seeds
42/43/44. Existing direct-composition results fail unseen B+C composition in
all three seeds. Stage 05 retrieval and Stage 06 routing are engineering-valid
but method-negative. These results are reusable evidence, not substitutes for
the newly specified arithmetic-mean RMS C2/C3 validation.

## Runtime preflight

- Physical GPUs 0-3 were occupied by unrelated long-running processes.
- A one-sample model-load smoke on GPU 2 failed with CUDA OOM; no external
  process was modified or stopped.
- The same one-sample end-to-end smoke passed on authorized fallback GPU 4,
  with recorded peak allocated memory of 14,842,019,328 bytes.
- Formal scheduling therefore uses automatic primary/fallback selection and
  resolves to GPUs 4-7 while GPUs 0-3 remain occupied.

## Launch incident

The initial detached SSH launch remained alive after the local SSH wrapper was
terminated. A second launch was briefly created during verification. The later
task-owned process group was stopped within its model-load phase, before any
prediction or summary file existed; the original scheduler (PID 2480764) was
kept. A PID guard was added to the checked-in launcher to prevent recurrence.

## Final outcome

- P1 formal matrix: 12/12 complete, no OOM, 3.434 recorded GPU-hours on
  physical GPUs 4/5/6/7; decision `FAIL_COMPOSITION`.
- P2 gate-limited seed-0 smoke: `FAIL_SLOT_SPECIALIZATION`, formal=false.
- P3 gate-limited seed-0 smoke: `ROUTER_QUERY_INSUFFICIENT`, formal=false.
- Final decision: `allow_full_continual_benchmark=false`.
- Canonical report: `docs/reports/0803_compose_p1_p3.md`; full machine-readable
  artifacts: `outputs/compose_p1_p3_20260803T090000Z/`.
