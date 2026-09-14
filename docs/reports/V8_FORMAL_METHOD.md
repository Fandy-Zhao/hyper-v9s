# V8 Formal Method

V8 has one formal meaning: few-shot answer-supervised historical-capability
screening followed by full-data, query-only multi-key Global Top-2
co-evolution.

For a task, the pipeline creates collision-safe full splits and fixed query
caches; screens a bounded, seeded, train-only teacher subset; emits the
task-level reusable historical set `R_t`; initializes one current-task reuse
key per retained historical expert; and trains all full-train samples with
`query -> active route keys -> max per expert -> distinct Top-2`.

The teacher's sample answers, NLL values and selections end at reuse-key
initialization. Full-data routing has exactly two inputs: fixed query and active
route keys. Ground-truth answers contribute only normal supervised LM loss.

The V7 trainer, key-pool, router and pruner are reused numerical kernels, not a
second V8 method. The formal entry point is `python -m
compose.experiments.v8_task_run`; the launcher is `bash
scripts/Compose/Run_UCIT/v8_six_task_run_4090.sh`.

Task0 bypasses Teacher screening, uses no history, and routes among four new
candidates. Task1+ requires a bounded Teacher subset and a previous committed
pool. The committed contract contains `compose_experts.bin`,
`compose_experts.json`, `v7_keys.pt`, and RMS calibration, and is directly
consumed by the next task.

Current release gate: Task2 (VizWiz) and Task5 (Flickr30k) lack verified
per-sample capability proxies. The launcher fails before work starts for any
task set containing either one. A decomposable subset may use `TASKS=0,1,3,4`.
