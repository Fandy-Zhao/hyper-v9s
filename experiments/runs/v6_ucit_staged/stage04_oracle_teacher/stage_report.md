# Stage 04 Report

- Status: **PASSED**
- Scope: answer-supervised Empty/Single/Pair Oracle teacher only; no Router and no Stage 05 work
- Formal runs: controlled 8, smoke 4, mini2 8, full seed42 subset 24
- Full subset: 32 train samples per task, seed 42, answer-token-length stratified; not full train
- Controlled max NLL regression error: `1.907e-06`
- Peak cache-miss CUDA memory: `15.11 GiB`
- Test data used: false
- OOM events: 0
- Formal report: `docs/reports/v6_ucit_stage04_oracle_teacher.md`
- Machine validation: `validation/validation.json`
- Aggregate metrics: `metrics/aggregate_metrics.json`
