# Project State

## 2026-09-03 — V7 Phase A re-audit PASS + Phase B gate progress (0903 spec §2-3/§19-21)

- HEAD `ea816a3` (branch `feat/0903-v7-throughput-equivalence`, tree clean
  after the precheck-tool commit).  GPU0/GPU1 idle at session start
  (GPU2–7 still other users' jobs, untouched).
- 双会话协调：bg 会话（cache 预计算 + smoke 驱动）转为只读；本会话接续驱动
  Phase B GPU gates → Phase C。其 smoke（`v7_gpu01_cache_smoke_task0_fixed_20260903`，
  physical GPU1，PID 3216132）S5 pruning 进行中。
- Phase A 独立复核 PASS（`compose/eval/v7_cache_precheck.py`，CPU-only，从 live
  声明文件/磁盘 cache/现役常量全量重算，不信报告文本）：18/18 splits
  （230,780 条 declared==saved==unique、miss/unk=0、1536-D fp32 finite、
  双哈希重算匹配、source/backbone/impl content binding 全绿）+ 6/6 task
  centers 用完整 train cache 重算逐位一致（max_abs_diff=0.0）。
  FULL_SAMPLE_QUERY_COVERAGE=YES / QUERY_CONTRACT_MATCH=YES /
  QUERY_CACHE_READY=YES（producer c93f51e → runtime ea816a3 git drift 记录）。
- Gate 证据在最终 HEAD 重新锚定：cache-vs-live gate rerun 于 physical GPU0
  （run root `v7_gpu01_cache_live_gate_20260903_rerun_head`）：task0/task1.train
  ×128 条 QUERY_NUMERICAL_EQUIVALENCE PASS（exact_bit_equal、max_abs_diff=0.0）
  + TOP2_ROUTING_EQUIVALENCE PASS（rate 1.0、score diff 0.0）。原 GPU1 证据
  （git 8f1a08f）依旧有效（修复 commit 不动 query 行/路由数学）。
- 回归：`tests/compose` 全量 500 passed / 2 env-limited（java 缺失，与基线一致）。
- 下一步：单卡 smoke 收尾验证（pruning trajectory/commit/encoder_calls=0）→
  2-GPU DDP cache smoke → cached-vs-online 孪生对跑 → RMS/PRUNING/EVAL
  equivalence gates → RECIPE_EXACT → formal launcher Task0→Task5。

## 2026-09-03 (evening) — V7 downstream cache adaptation (0903 spec Phase B)

- 状态：代码改动完成（commit `af716f0`、docs `1083eee`、formal launcher +
  适配报告 `0da6019`，branch `feat/0903-v7-throughput-equivalence`，HEAD
  0da6019；适配报告结论块 `DOWNSTREAM_CACHE_ADAPTATION=YES` /
  `FORMAL_TRAINING_READY=NO`）；500/502 CPU 测试通过（2 个失败为既有的
  `java` 缺失环境限制，与本次无关）。
- 已完成：S1 cache 适配器（train+val payload 从二进制 cache 直接发出，
  （Phase B smoke 实测发现并修复 d5bd9f8 潜伏缺陷：adaptive S5 scorer 把
  routes `.tolist()` 成 list 后传入需要 tensor 的 route_manifest，任何 fresh
  自适应 pruning 打分 job 必崩；修复为保留 CPU tensor，不改路由/剪枝语义，
  legacy 路径同样受益）；
  encoder_calls=0、id 序列 fail-closed、backbone/impl 内容绑定、git drift 记录）；
  S3/S4/S5 cache-origin guards；S6 与 21-cell 最终评测改为 cache test rows →
  committed-pool selection manifests（eval_task --selection-manifest，零 CLIP
  encoder 调用）；RMS 审计结论 = 不消费 queries（activation RMS），天然复用。
- 未执行（阻塞）：Phase B GPU 等价 gates + smokes（QUERY_NUMERICAL_EQUIVALENCE、
  TOP2_ROUTING_EQUIVALENCE、单卡 7B smoke、DDP smoke、cached-vs-online、
  RMS/PRUNING/EVAL gates、RECIPE_EXACT）与 Phase C formal Task0→Task5。
  GPU0 被 zangzeh+ 的 openpi serve_lerobot_policy（~8.9 GiB，运行中）占用、
  GPU1 被 caizhen+ 的 continual_train（~435 MiB）占用 → 按规则不抢占、轮询等待
  （会话 cron 每 20 分钟探测一次，双卡空闲即继续）。
- 下一步（GPU 空闲后）：gate 驱动 → 单卡/双卡 smoke → formal launcher
  （RECIPE_EXACT world2×batch1×GA32=64）→ Task0→Task5 → 21-cell cache 评测 →
  `docs/reports/V7_QUERY_CACHE_DOWNSTREAM_ADAPTATION_REPORT.md` +
  `V7_GPU01_CACHED_QUERY_FORMAL_FINAL_REPORT.md` + `artifacts/v7_gpu01_formal/`。

## 2026-09-03 — V7 fixed-query full precompute cache (GPU0+GPU1)

- Fixed-query (L2Norm(concat(LN(z_v), LN(z_t))), 1536-D fp32, detached) 全量预计算完成：
  ImageNet-R/ArxivQA/VizWiz/IconQA/CLEVR/Flickr30k × train/val/test = 230,780 条，
  18 个 split cache + 6 个 full-train task center 落盘
  `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_fixed_query_cache_gpu01_20260903/`。
- 只用 GPU0+GPU1（GPU2--7 他人占用，未触碰）；V7 主流程（训练/Top-2/RMS/Pruning/评测）
  零改动，仅新增 `compose/v7/query_cache.py`、`compose/eval/precompute_v7_queries.py`、
  `tests/compose/test_query_cache_audit.py`。
- 有界门 PASS：128 条真实样本单卡 vs 双卡逐位相同（cosine_min 0.99999976 ≥ 1-1e-6）；
  Top-2 路由对休眠 formal 2-GPU run 的 S2 候选池一致率 1.0、打分 0 差；task center
  bit-equal。全量审计：declared=saved=unique、0 dup/miss/unk、1536-D fp32、范数≈1、
  哈希+契约绑定、原子写、断点续跑幂等（真实 3 次）。
- 报告 `docs/reports/V7_FIXED_QUERY_CACHE_GPU01_REPORT.md`；manifest 双份
  （run root + `artifacts/v7_query_cache/query_cache_manifest.json`）。
- S3 训练缓存化就绪但未执行：唯一改动点 `compose/train/data.py:369` 一行替换 +
  seed 固定对拍（报告 §29/§35）。

## 2026-09-03 — V7 formal three-GPU run

- `FORMAL_EXPERIMENT_READY = YES`; GPU0--2 and all declared train,
  validation, test and annotation paths passed preflight.
- The S3-only DDP implementation uses three ranks, per-device batch 1 and
  accumulation 21 (effective global batch 63, -1.5625% from target 64), with
  no learning-rate scaling and unchanged sparse Global Top-2 execution.
- Focused regression: 39 passed. Real 7B gate: Task0, three ranks, two
  optimizer steps, finite losses/gradients, identical cross-rank Key/LoRA
  checksums, and exact current-only optimizer membership.
- Formal output root:
  `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_ucit_formal_3gpu_seed42`.
- Current status: formal Task0--Task5 launcher is being started; after Task5
  it automatically evaluates the 21-cell lower triangle and computes the
  original continual metrics.

## 2026-09-02 — V7 final formal-repair gate

- Validated implementation SHA `ac7fb8d`: 33 focused V7 tests and the complete
  Compose suite (`396 passed + 8 subtests`) pass.
- Formal defaults, answer-only NLL, RMS/preprocessing parity, iterative
  pruning, strict Top-2, leakage provenance, official validation metrics,
  accumulation-safe gradients, resume, query provenance and atomic commit are
  implemented without changing the V7 method skeleton.
- Final-HEAD GPU2/3 smoke was resource-blocked because both cards had external
  19--21 GiB allocations. Earlier real-7B Task0/Task1 evidence is retained;
  a fresh CPU 30+30 lifecycle smoke passed.
- `FORMAL_EXPERIMENT_READY = NO`: the declared six-task validation directory
  is absent, so the formal launcher intentionally fails before Task0 rather
  than reusing test data.

## 2026-09-02 — V7 real 7B GPU2/GPU3 validation

- GPU2 completed V7 stages S0--S5 on a bounded real ImageNet-R split. The
  first training step exposed an unpadded Top-2 versus four-slot execution
  boundary mismatch before any optimizer update.
- The fix pads only the V7 execution representation with `-1`/zero slots; the
  Global Top-2 route, active gates, Key loss and sparse LoRA semantics remain
  unchanged.
- GPU2 Task0 completed two finite 7B optimization steps plus 224-layer RMS,
  removal-reroute pruning and commit; Candidate 2 was retained.
- GPU3 Task1 then exposed a BF16 NumPy checksum incompatibility before its
  first optimizer update. The checksum now hashes raw tensor bytes.
- GPU3 Task1 completed two finite 7B optimization steps, preserved the Task0
  historical Key and LoRA checksums, completed 224-layer RMS and true removal-
  reroute pruning, and retained Candidates 4, 5 and 7. The committed pool now
  contains four frozen historical experts: 2, 4, 5 and 7.
- This is a bounded real-model lifecycle smoke test, not a convergence or
  benchmark-quality claim. The focused post-fix regression set is 26 passed.

## 2026-09-01 — V7 full-data global key–expert co-evolution

- Baseline: `a8f3a7860631aec8e2ea0d65ad9794838ffaffc7`.
- Active branch: `exp/v7-full-data-global-key-expert-coevolution`.
- Audit: `CURRENT_CODE_AUDIT.md`.
- Status: implementation and CPU/real-query smoke complete; V6/V6.1/V6.2 paths remain intact.
- Plan: fixed 1536-D query, four full-data candidates, global Top-2 from step
  one, selected-current Key/LoRA updates, validation removal-reroute pruning,
  frozen commit, resume-safe inference, then unit and two-task smoke tests.
- Validation: V7 acceptance 17/17; related regression 24/24; CPU Task0/Task1
  30+30 steps passed; real ImageNet-R full-declared-split query/candidate
  preparation passed. Real 7B optimization smoke awaits safe GPU capacity.

## Snapshot
- Date: 2026-07-30
- Branch: `exp/0730-compose-task1-oracle`
- Project type: Python deep learning / MLLM research (ACL 2025)
- Current focus: Staged Compose Task1 parity, multi-expert, Oracle, and capacity-control validation

## Active Work
- **Compose foundation**: Independent LLaVA model, LoRA expert composition, ExpertPool, adapter checkpoints, and UCIT Task1 entry are implemented on `feat/0729-compose-foundation`.
- **Compose validation**: Stages A--E are complete. The suite has 42 passing unit tests, grouped bf16 CUDA execution matches its reference, and full Task1 Compose/PEFT parity, expert isolation, deterministic Oracle, and rank-matched controls are recorded.
- **Compose experiment decision**: The 6,000-sample Oracle is reproducible, but mean/median synergy are negative and the L2 pair underperforms the exactly parameter-matched rank-16 adapter in NLL and accuracy. Stop condition C is active; no Set Router is implemented.
- **Project governance initialization**: Creating AGENTS.md, directory structure, archiving deprecated files, moving analysis tools to `tools/`
- **Branch `zzf`**: Active development branch for Hyper-LLaVA experiments

## Known Risks
- **Large binary files in root**: Three `flash_attn-*.whl` files (~1GB total) and `nohup.out` (23MB) are in the repository root — these should be moved to external storage or `.gitignore`'d
- **Symlinked directories**: `instructions/`, `runs/`, `ucit_instructions/` are symlinks to external paths (`/data/ckpt/`, `/data/dataset/`) — repository portability depends on these paths existing
- **Partial test coverage**: Compose has focused unit tests, while the retained LLaVA/Hyper paths still rely primarily on full training/eval runs
- **Checkpoint compatibility**: The earlier 296-layer Compose smoke checkpoint is intentionally rejected by the strict decoder-only loader and must not be used as a valid expert checkpoint.
- **Composition generality**: The negative stop decision is based on two task-trained functional proxies; it does not establish a universal result for other decompositions or larger pools.
- **`gaussian.py`**: Contains non-runnable code snippets extracted from model implementation — kept for reference only
- **`compute_routing_weights.py`**: Must remain in root — imported by `llava/train/train_MOE.py` at line 44

## Validation Status
- [x] Compose Python import isolation and syntax smoke test
- [x] Compose unit tests (42 tests)
- [x] Compose UCIT Task1 two-step DeepSpeed smoke run
- [x] Explicit LLaVA-to-Compose config conversion and core-dimension validation
- [x] Compose shell script `bash -n` check
- [x] Training dry-run (single GPU, ZeRO-2, two steps)
- [x] Strict 448-tensor checkpoint reload and output-equality validation
- [x] Explicit gate-normalization contract and active-row grouped execution
- [x] Mixed top-1/top-2 CPU and bf16 CUDA forward/backward equivalence
- [x] Full UCIT Task1 Compose and matched PEFT training/evaluation
- [x] Two-expert isolation and strict pool reload
- [x] Deterministic 6,000-sample empty/single/pair Oracle audit
- [x] Exactly parameter-matched rank-16 NLL and generation control
- [ ] Full six-task UCIT benchmark (outside this task)

## Next Steps
1. Do not implement the Compose Set Router for the current expert pool; stop condition C is recorded in ADR-0730.
2. If composition is revisited, first produce a new deterministic fixed-set audit that beats an exactly parameter-matched single adapter on NLL and accuracy.
3. Extend beyond two functional proxies only as a separately governed experiment with the same isolation and capacity controls.
4. Move or `.gitignore` the existing large `.whl` and `nohup.out` files in a separate cleanup task.
