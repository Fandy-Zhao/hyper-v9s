# Project State

## 2026-09-03 (late) — comparator semantic fixes; batch16-vs-batch32 localization (0903 spec §26/§28)

- 全 root cache-vs-online twin 对比(单卡 cache smoke root vs GPU0 live
  twin root)首轮 5 gate FAIL → 三分是 comparator 缺陷,已修复(semantic
  comparator v2,`compose/eval/v7_twin_run_compare.py` + 21 CPU 测试全
  PASS):S1 按 per-sample_id 对齐(文件顺序是 emission artifact,信息性
  记录 id_sequence_identical);json 结构 walk 对 machine-path leaf
  (output_dir/annotation_file/prediction_file)做 root 相对比较并计数
  relocated;commit gate 语义化 — 文件集 hard、bin byte-hard、json
  数值 2e-4 界、v7_keys.pt 按 per-expert max-abs-diff ≤ KEY_ATOL 2e-4
  (实测噪声包络,非自由参数),sha 改信息性。
- **剩余两类 FAIL 精确定位(§8a 五步证据链)**:S1/S3/PRUNING/COMMIT 的
  每个分歧都追踪到一个控制设置缺陷 — twin S1 用 `query_features` 默认
  batch **16** 编码,live cache 生产/有界 gate 用 batch **32**;fp16 CLIP
  conv kernels 按 batch shape 选择 → visual half 逐行 ≤1.87e-3 确定性
  fp16 差异 → S2 center 7.6e-5(共享 offset)→ pruning val rows 170/195
  (+reroute 107)边界 Top-2 翻转。S2 自身确定性由 DDP-vs-single
  candidate_keys.pt 4/4 bit-equal 证明(同一 cache binary rows)。§26
  remediation = DDP smoke 后以 `--batch-size 32` 重跑 live twin,重新对
  batch32 pair 签发 named gates。
- RMS_CACHE_EQUIVALENCE 在真实 twin pair 上 **PASS**(3 RMS 文件数值
  相等,rms_calibration/rms_statistics byte-identical;唯一结构差 =
  output_dir path leaf,已 root 相对比较);commit v7_keys.pt numeric
  sub-gate PASS(≤1.16e-4 ≤ 2e-4);compose_experts.json rerouted_expert_
  ids[107] flip 仍正确拒绝(route id 结构差不宽容忍)。twin audit v2:
  `v7_gpu01_smoke_compare_task0_20260903/compare_audit_v2.json`。
- DDP cache smoke(GPU0/1,world 2×batch1×GA32,09:34 起,PID 3711301):
  S2 done + candidate_keys.pt 已用于决定性 proof;现处 S5 pruning;完成
  后跑 `v7_twin_run_compare.py --gate-mode distributed`(DDP root vs
  单卡 cache root,same cache rows)→ DISTRIBUTED_RMS_EQUIVALENCE + batch32
  live twin rerun → Phase B gates re-issue。
- 报告 `docs/reports/V7_QUERY_CACHE_DOWNSTREAM_ADAPTATION_REPORT.md` §8a
  已写五步定位 + evidence table;HEAD `dab7d38` + comparator fixes
  (uncommitted,待 commit)。

## 2026-09-03 (morning) — cache smoke commit complete; EVAL gate running (0903 spec §29-40)

- Cache-mode smoke root `v7_gpu01_cache_smoke_task0_fixed_20260903` lifecycle
  CLOSED: `s5_pruning_commit.done` 08:40:45 + `committed/` 08:40:44
  (compose_experts.bin/json + v7_keys.pt, pool selectable (0,1,2,3) all
  origin_task 0).  Pruning job chain closed: job_0 07:43:03 (original run),
  job_1 08:40:44 (resume full re-score), job_2 08:00:17, job_3 08:11:55,
  job_4 08:23:26.  features train/val carry `encoder_calls: 0`.
- Death-time thread CLOSED (both sessions' evidence agree): old-process
  3216132 died ∈ (08:23:25, 08:23:36] (last artifact official_metric_4.json
  08:23:25; rc-marker watcher output mtime 08:23:36) = external unknown
  SIGTERM (×2 counting 07:48); resume launched 08:29:10 → zero overlap, no
  causality.  My earlier "kill -0 = ALIVE" observations were a zombie-reap
  artifact (signal 0 succeeds on zombies) — withdrawn.  Session clocks
  verified synced (both read ~08:34 at the same wall moment).
- Twin (GPU0, live mode, `v7_gpu01_smoke_task0_live_twin_20260903`, main
  3389457 since 08:16:30): S3 done (2 steps finite, train_runtime 204.96 s),
  `s4_rms.done` written, S5 pruning scoring in flight (~11 min/job × 5 →
  ETA ~09:30-45); after it: full twin-vs-cache stage compare
  (`v7_twin_run_compare.py --gate-mode cache`).
- EVALUATION_CACHE_EQUIVALENCE gate running on GPU1 (evidence root
  `v7_gpu01_eval_gate_20260903`): step 1 cached_selections full-test-row
  manifest from committed v7_keys.pt + cache test split (content-bound),
  step 2 v7_cache_live_gate `--split test --limit 128 --pool-state
  <committed v7_keys.pt>` (live CLIP rows vs cache rows routed through the
  committed pool).  Committed pool verified loadable via
  `V7ExpertKeyPool.from_state` (selectable (0,1,2,3)).

## 2026-09-03 (late morning) — Phase B smokes + comparator + resume (0903 spec §19-28)

- 我的 live twin smoke 初启误落在 physical GPU1(adaptive `--gpus 1` =
  physical id,非"1 张卡";CLI 优先于 CUDA_VISIBLE_DEVICES)—— 已 kill 重建于
  GPU0(`--gpus 0`,PID 3389457,gpu_plan available=[0]);peer 的 cache smoke
  仍在 GPU1。这不是 orchestrator bug,是我的启动参数错误。
- 新增 `compose/eval/v7_twin_run_compare.py` + 16 CPU 测试(全部 PASS):
  CPU-only 双 run 逐阶段 diff(双 root 必须同为完整 task0 lifecycle),
  输出 S1_QUERY_ROWS / S3_TRAIN_STEPS / RMS_CACHE(或 DISTRIBUTED_RMS)/
  PRUNING_TRAJECTORY / COMMIT_STATE 判定行 + 原子 audit JSON,任何 FAIL
  退出非零。服务于 cached-vs-online 收尾验证与后续 DDP-vs-single gate。
- cache smoke(单卡,fixed root)再次被外部 SIGTERM(~08:52,非本会话/peer
  所为;07:48 首次)。产物:encoder_calls=0 证据 + s0-s4 完整 + S5 job 0-4
  (job 1 被 07:48 打断),无 commit。已在 ea816a3 worktree(PID 3427469,
  GPU1)resume,score-job cache 使 job 0-4 重放,只跑剩余假设 → s5+commit。
- 当前:GPU0 = live twin(S3 training,2-step smoke);GPU1 = cache smoke
  resume(S5 scoring)。双卡各自 ~15 GiB / 30-40% util,健康。
- 下一步(按 peer handoff 顺序):twin/resume 收尾验证(full trajectory +
  commit + 双 run 全文 compare)→ DDP cache smoke(GPU0/1 空闲后,world
  2×batch1×GA32=64)→ EVALUATION_CACHE_EQUIVALENCE → formal launcher。

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
