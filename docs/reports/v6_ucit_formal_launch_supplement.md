# V6 UCIT Formal Launch Supplement — BLOCKED 修复记录（DEV 补齐）

- 日期：2026-08-06
- 前置：启动审计 `docs/reports/v6_ucit_formal_launch_audit.md` → **BLOCKED**
- 决策：用户批准「补齐缺失组件」（2026-08-06 定时任务会话）

## 修复的 BLOCKING 差异

| DEV | 差异 | 修复 | 依据 |
|---|---|---|---|
| DEV-6 | `configs/v6_ucit_formal_locked.yaml` 不存在（README.json 与 exact_commands.sh 均引用） | 新建 `configs/v6_ucit_formal_locked.yaml`，内容 = locked_config.yaml 工程默认值 + 正式运行追加（tasks/training/eval 段、config_hash） | 任务书「复制为新配置版本 + 记录修改原因 + 新 hash + 新 run root」；exact_commands.sh 注释（正式规模 full、eval 3000、4 卡 batch 6×accum 1） |
| DEV-9 | `scripts/v6_ucit/six_task_run.sh` 不存在（README.json run_entrypoint） | 新建六任务串行编排脚本（seed 参数化、每任务独立输出根、幂等 done marker） | READY.json run_entrypoint；任务书「任务之间严格串行」 |
| DEV-10 | `scripts/v6_ucit/resume_run.sh` 不存在（README.json resume_entrypoint） | 新建幂等恢复脚本（pending 事务清理 + 六任务重跑 + hash 校验） | READY.json resume_entrypoint；resume_commands.sh 模式 |
| DEV-13 | VizWiz/IconQA/CLEVR/Flickr30k test 路径声明 `test.json`，实际 `test_3000.json` | 正式配置中修正为 `test_3000.json`（数据本身完整，train 全部存在） | 实际文件系统核验 |

## 新增组件（commit `c165fca`）

1. `configs/v6_ucit_formal_locked.yaml`
   - config_hash：`d52111fc25b4b735`（sha256 前 16 位，排除 config_hash 行）
   - 与 handoff locked_config.yaml 工程默认值一致（除注释）；追加 tasks/training/eval 段
2. `compose/experiments/v6_task_run.py` — 通用任务 runner（task index ≥ 1）
   - S0..S11 幂等阶段；slot ID = (task_id+1)*10, +1（task0→10、task1→20/21 已验收，唯一性验证见测试）
   - 全部调用 E1..E12 已验收模块；未修改任何已验收模块
3. `scripts/v6_ucit/six_task_run.sh` / `resume_run.sh`
4. `tests/compose/test_v6_task_run_formal.py` — 15 项回归测试

## 测试基线

- 新增：15 passed（formal config 完整性、config_hash 自洽、路径存在、六任务顺序、feature 开关全关、runner 幂等、snapshot task_id 校验、slot 唯一、编排脚本存在）
- 全套：**354 passed + 14 subtests**（基线 339 + 15 新增，无回归）

## 说明

- 工作区其余未跟踪/修改文件（dual_lora 分析产物等）不在本批次范围，未纳入 commit。
- 未 push。
- Git HEAD：`c165fca`（补丁 commit 前为 `61ed787`；交接包记录的 `41e70bd` 为代码零差异的 docs-only 前置状态，见审计 DEV-4）。

## 运行中修复（seed 42 task0 冷启动阻塞 bug）

- commit：`6d6e2a2`（fix(compose): encode_images dtype match for multi-GPU DataParallel）
- 复现：seed 42 task0 冷启动，4 卡（4,5,6,7）batch 6 × accum 1 → `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`（mm_projector）
- 根因：dry-run 只验证过单卡 batch-1（README 警告明确）；4 卡 DataParallel replica 线程丢失 autocast 上下文，vision tower fp32 输出 × mm_projector bf16 权重
- 修复：`encode_images` 中将 features cast 到 projector 权重 dtype（autocast 下为 no-op）
- 回归：`tests/compose/test_multimodal_dtype_multigpu.py`（3 项）；全套 357 passed + 14 subtests
- 受影响 seed：42（task0 阶段失败，未提交任何专家，快照未生成——安全重跑；无已完成任务受影响）

## 运行中修复 2（seed 42 task0 冷启动阻塞 bug：DataParallel vs ComposeSelection）

- commit：`ca1318a`（fix(v6-ucit): launch candidate training via torchrun DDP）
- 复现：seed 42 task0 冷启动，4 卡 batch 6 → `ValueError: selection batch size 24 does not match input batch size 6`
- 根因：多卡 HF Trainer 未初始化分布式时回退 DataParallel；DataParallel 只切分 tensor 输入，自定义 ComposeSelection（batch 24）被逐字复制到每个 replica，与切分后的输入（batch 6）不匹配。dry-run 只跑过单卡 batch-1（README 明确 4 卡是建议、从未实测）
- 修复：train_v6_candidate 经 `torch.distributed.run`（DDP）启动——每个 rank 独立 forward，selection 与 input 天然一致；rank-0 保存逻辑已存在（--local_rank）；DDP 同时让 encode_images 的 dtype guard 成为 no-op
- 回归：`test_candidate_training_uses_torchrun_ddp`；全套 359 passed + 14 subtests

## 运行中修复 3（seed 42 task0 冷启动 OOM → config v2）

- commit：`（随 config v2 提交）`
- 复现：seed 42 task0 冷启动，4 卡 batch 6 × accum 1 → rank 3 `torch.cuda.OutOfMemoryError`（23.5GB/24GB）
- 根因：README KI-003 建议的 "batch 6 x accum 1 x 4 cards" 从未实测；每卡 batch 6 激活内存超 24GB
- 修复：config v2（`config_hash=e664569014c12f78`）：locked `global_batch_size=24` 不变，改为 per-device 3 × accum 2 × 4 卡 = 24
- 记录：修改原因写入 config 注释；run root 不变（seed 42 无已完成输出，无 superseded 需要）；旧 config v1 hash `d52111fc25b4b735` 记录于此
- 回归：config hash 自洽测试动态适配

## 运行中修复 4（OOM 根因定位 → config v4 单卡 batch-1）

- 根因链（2026-08-06 凌晨诊断）：
  1. `config.attn_implementation="flash_attention_2"` 在 transformers 4.33 下**从未生效**（LlamaDecoderLayer 硬编码 LlamaAttention）→ 单样本 forward 峰值 ~22.75GB（eager attention 激活 8.5GB）贴 24GB 上限
  2. 4 卡 DDP 下 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 报 "not supported"（每 rank 打印警告）→ DDP 每 rank 额外开销（find_unused_parameters 遍历等）导致 OOM
  3. 单卡 + expandable_segments 实测 PASSED（200 样本 14.33GB；dry-run 正是此配置）
  4. gradient checkpointing 与 frozen-backbone 冲突（KI-003 预言，实测 RuntimeError: element 0 does not require grad）
- 修复：config v4（`config_hash=904ec5950db0b333`）：单卡 batch-1 × accum 24 = global 24（locked global_batch_size=24 不变；README 明确 "or keep batch-1 single-card"）；six_task_run.sh 训练用 TRAIN_GPU=4（expandable 生效），实测 3.25 s/step → task0 全量 ~54 分钟
- 历史 config hash：v1 `d52111fc25b4b735`（batch 6×4）、v2 `e664569014c12f78`（batch 3×2×4）、v3 `6274b5498aaf825b`（batch 1×6×4）
- 未修改已验收训练代码；仅 config + 编排脚本

## 运行中修复 5（全量数据吞吐瓶颈 → config v5 dataloader workers）

- commit：`6178348`（perf(v6-ucit): dataloader_num_workers=4）
- 复现：task0 全量训练 ~13s/step（wandb epoch 0.44 / 1h38m），远慢于实测 3.3s/step
- 根因：`LazySupervisedDataset.__getitem__` 每次重载图像+预处理，`dataloader_num_workers=0` 串行执行（dry-run 200 样本图像在缓存中掩盖了瓶颈）
- 修复：train_v6_candidate 新增 `--dataloader-num-workers`（默认 0 保持 dry-run 路径不变）；config v5 设为 4；三个 runner 从 config 透传
- 回归：`test_dataloader_workers_passed_from_config`；全套 360 passed + 14 subtests
- config v5 hash：`5e9175028f59c94c`（v4 `904ec5950db0b333` 记录于上文）

## 运行中修复 6（seed 42 task0 below_tau → task1 空 registry 崩溃；退化链完整化）

- commit：`（独立 commit，见本节尾部）`
- 复现：seed 42 task0 完成（s1-s8 全绿，state COMPLETED），但 **0 个 expert 提交**
  （`commit_record.json: {"committed_expert_ids": [], "reason": "below_tau"}`，协议允许）。
  task1（ArxivQA）S1 teacher search 立即崩溃：
  `IndexError: list index out of range` @ v6_task2_dry_run.py:127
  （`"single_{}".format(old_ids[0])` 在空 registry 下无条件求值）。
  编排脚本 `set -e` 退出；后台 shell 随会话结束终止，无进程残留。
- 根因：已验收的 task2 runner 假设 task0 一定提交 expert 10；空 registry 路径
  只在 S1 的 checkpoint 回退（line 131-133 已写 cold_start 兜底）做了防护，
  其余三处硬编码 `expert_0010` / `old_ids[0]` 未防护。
- 修复（同一根因族，5 处）：
  1. `v6_task2_dry_run.py` S1：`_teacher_search_selections()` 空 registry 时
     只评估 empty baseline（不再 `old_ids[0]`）；checkpoint 走
     `_old_expert_checkpoint_or_cold_start()`（committed 专家 → prev cold start）。
  2. `v6_task2_dry_run.py` S6 assemble：`--old-expert-checkpoint` 不再硬编码
     `expert_0010`，改为镜像 S4 的 `old_checkpoint`（空 registry 时省略该参数）。
  3. `v6_task2_dry_run.py` S10 eval：fallback 从 `committed/expert_0010` 改为
     prev `candidate/cold_start`，并写入 `candidate/last_known_checkpoint.json`
     指针（把继承的 checkpoint 传向下游任务）。
  4. `v6_task2_dry_run.py` S3 / `v6_task_run.py` S4：residual 为空时跳过
     v6_query_features（原逻辑仍会启动提取器空跑，浪费一次 GPU 模型加载）。
  5. `v6_task_run.py`：`_old_expert_checkpoint_chain()` 三层回退（prev
     candidate/train → prev cold_start → last_known 指针）；全部落空时
     `_write_empty_teacher_records()` 写入退化 teacher 记录（teacher_set=()、
     empty_loss=None、summary.json 注明原因），S1-S10 正常完成、commit 0，
     仅 S11 eval 在真正无任何 adapter 状态时报清晰错误；S3 重建容忍 None 损失；
     `_record_id()` 适配 id/question_id 双 schema（VizWiz/IconQA/Flickr30k 用
     question_id，否则 S1 对这三个任务同样会 KeyError）。
- 回归：`tests/compose/test_v6_task_run_formal.py` 重写 1 项 + 新增 2 项
  （空 registry 退化链端到端、S1 selections 空 registry、指针回退）；
  全套结果见 commit 信息。
- 受影响 seed：42（task0 已完成且不受影响；task1 未提交任何专家、无快照——
  安全重跑，从 task1 幂等续跑）。

## 运行中修复 7（task0 below_tau 为 bug 产物：验证切片恒空 → config v6 + 防护）★推翻运行中修复 6 的受影响判断★

- commit：`（独立 commit，见本节尾部）`
- 复现（seed 42 task1 完成、task2 S1 进行中时审计发现）：
  - task0 `validation/summary.json` 为 `{"samples": 0, "mean_gain": 0.0,
    "support_count": 0}`，而 config `validation_samples: 256`、tau_support 8。
  - 根因链：`configs/v6_ucit_formal_locked.yaml` v5 的
    `cold_start_train_samples: 23998` **恰好等于** ImageNet-R train.json 总记录数
    （23998，manifest `train_records: 23998`）；S1 验证切片
    `records[23998:24254]` 恒为空 → `validation_subset.json` 0 样本 →
    v6_nll_eval 0 行 → `mean_gain 0.0 / support_count 0` → below_tau（0 提交）
    是**切片 bug 产物，不是数据驱动结果**。
  - 后果：task0 的 below_tau 无效 → 空 registry 退化链（task1-5 全部 0 提交）
    建立在无效输入上，**整个 seed 42 结果作废**。运行中修复 6 的
    「task0 不受影响」判断被推翻（当时只查了 task0 的 s1-s8 阶段绿，未审计
    validation 样本数）。
- 修复：
  1. config **v6**（`config_hash=30020824bf7bf084`）：task0
     `cold_start_train_samples: 23998 → 23742`（= 23998 − 256，为
     validation_samples=256 保留切片；23742+256=23998 仍为全量数据，
     train 与 validation 均来自 train.json，test 永不用于验证）。
     修改原因写入 config 注释；v5 hash `5e9175028f59c94c` 记录于此。
  2. `v6_task1_dry_run.py`：新增 `_draw_train_val_splits()`——切片不足时
     抛错（`train split is short`），从源头杜绝静默空验证；S3 对
     gains 空列表抛错（`refusing a below_tau decision on an empty validation`）。
  3. `v6_task2_dry_run.py` S5 / `v6_task_run.py` S6：同一族防护——per-slot
     gains 为空时抛错（不再静默写入 `mean_gain 0.0` 然后 below_tau）。
- 回归：新增 2 项（`test_draw_train_val_splits_raises_on_short_dataset` 单元、
  `test_task0_cold_start_split_fits_real_dataset` 真数据 config 校验——
  该测试直接复现原 bug：v5 配置下 `_draw_train_val_splits(records, 23998, 256)`
  抛错）；全套结果见 commit 信息。
- 受影响 seed：**42 全部标记 INVALID**（task0 below_tau 无效，task1-5 全为
  退化链产物）。run root 改名 `seed_42_INVALID_validation_bug` 留证，
  以 config v6 全新重跑 seed 42（任务顺序不变）。
