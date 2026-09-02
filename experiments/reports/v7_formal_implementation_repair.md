# V7 Formal Implementation Repair Report

- Date: 2026-09-02
- Branch: `exp/v7-full-data-global-key-expert-coevolution`
- Before SHA: `ee971f7` (audit baseline)
- After SHA: `0a8f76a` (HEAD)
- Artifacts: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_repair_smoke_seed42/`
- Run notes: `experiments/runs/0902_v7_formal_repair/{issue,notes}.md` (disk artifact tree)

## 1. Git

| 项 | 值 |
|---|---|
| 分支 | `exp/v7-full-data-global-key-expert-coevolution` |
| 基线（audit 起点） | `ee971f7` |
| 修复后 HEAD | `0a8f76a` |
| 提交（自基线起 6 个） | `1ea87e8` fix(v7): enforce formal experiment contracts<br>`78feddf` feat(v7): add formal UCIT lifecycle launcher<br>`0fa30a0` test(v7): strengthen formal acceptance coverage<br>`6e7c16d` fix(v7): harden smoke and validation contracts<br>`f082390` test(v7): align legacy fixtures with unified 4-slot contract<br>`0a8f76a` fix(v7): import os in nll_eval output writer; unit-test it |

工作树干净；全部工作在指定分支完成，逻辑提交；V7 方法定义未改（固定 1536-D query、rank-8 当前候选、历史 Key+LoRA+RMS 冻结、无配额 Global Top-2、仅选中当前候选更新、迭代剪枝/提交、推理只用 committed experts）。无 Router MLP / query MLP / 聚类 / oracle / 配额 / warm-up 路由 / 负载均衡 / Top-2 伪造。

## 2. Problems-fixed table

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| P0-1 | V7 形式训练默认 30 步，声明全量 train split 未参与 | 形式模式要求完整 recipe；`--max_steps` 仅 smoke 显式 `--smoke-max-steps` 传入；`compose_v7_require_full_coverage` 打开全量覆盖审计（distributed gather 后 `unique_train_sample_ids_seen == num_train_samples` 否则 FAIL）；禁止 V7 训练中 `records[:2000]`/max_samples/teacher 子集 | S0 coverage audit PASS：32/32；正式六任务配置缺 validation 文件时 fail-fast |
| P0-2 | pruning NLL 与训练 answer 掩码不一致 | pruning 经 `CandidatePruner` scorer 复用 `nll_eval` → 与训练完全相同的 `LazySupervisedDataset`+`DataCollatorForSupervisedDataset` 预处理；`prepare_inputs_labels_for_multimodal` 同步扩展 labels；`teacher_forcing_token_nll` 统一 shift；零 answer token 样本 raise ValueError；输出 `supervised_token_count` + `mean_answer_nll` + `supervision_contract=compose_train_preprocess_v1_shifted` | 8 个 val 样本各 5 answer token；contract 标记一致 |
| P0-3 | RMS 标定不统一（训练/剪枝/推理） | RMS 仅在 validation split 标定（8 样本），checkpoint-hash 绑定，manifest 增量补丁；Task0 commit 时 kappa 冻结；Task1 `merge_commit_frozen_calibration`（历史冻结 kappa + 新专家动态 kappa）；推理 `load_compose_model(apply_persisted_rms=True)` | 224/224 层 kappa Task0→Task1 位相等 |
| P0-4 | 图像预处理不统一 | 全程 `image_aspect_ratio=pad`（训练 recipe、nll_eval、推理 eval_task），runtime contract 审计（pad/-2/patch/mlp2x_gelu）在训练与剪枝侧强制一致 | runtime contract 三处 PASS |
| P1-1 | 剪枝非迭代 remove-and-reroute | `CandidatePruner.evaluate` 每轮只删一个、对幸存池重评估（`excluded` 逐轮加入）；完整 trajectory（9 行：iteration/pool/metric_full/metric_minus_candidate/loss_full/loss_minus_candidate/removal_gain/decision/pool_after） | Task0 3 轮删 3→0；Task1 3 轮删 6→4，每轮重新打分 |
| P1-2 | selectable pool 可能 < 2 | 最小池保护：`len(pool_after) < 2` 时保留，reason=`retained_for_global_top2_minimum_pool`；提交前断言 historical+retained ≥ 2 | Task0 保留 [1,2]；Task1 池 [1,2,5,7] |
| P1-3 | 缺 V7 专用六任务启动器 | `scripts/Compose/Run_UCIT/v7_six_task_run.sh` + `v7_task_run.py`（S0-S6 阶段流水线 + stage-marker resume）；V6 启动器未动 | 三段 resume 均幂等续跑成功 |
| P1-4 | train/val/test 无隔离证明 | `audit_split_isolation`：sample/record/image 三对两两 overlap + sha256 溯源；泄漏即 FAIL 且在任何训练前触发 | 首版切分泄漏 8 → 正确 FAIL（实弹）；修正后全 0 PASS |
| P1-5 | 验证指标无任务适配 | `eval_task` 任务专用官方验证 + `v7_validation_metric`（official_ucit / nll_fallback 显式标注）；正式运行强制 `--test-file` + `--validation-metric` | Task0/Task1 推理 `task_id_used=false`，896/896 LoRA，test 数据未参与任何调参/剪枝 |
| S13 | 梯度累积审计缺失 | `V7ComposeTrainer` 每个 micro-batch hook 只记当前 micro-batch 梯度（`training_step` 起始清 `_v7_key_gradient_ids`）；断言未选中当前候选无新梯度、历史 key/LoRA/RMS 冻结（init 快照 + `assert_task_freeze_integrity`）；Old+Old 锚 loss 使 optimizer no-op | 25 步 ×2 任务全部通过 |

另修复 smoke 捕获的真实代码 bug：

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| B1 | `compose/eval/nll_eval.py` 输出写盘 `NameError: name 'os'`（NLL 已算出、写结果崩溃） | 补 `import os`；抽取 `write_nll_output`（多分片命名 `<output>.rank<index>` 复用既有 `partial_path` 约定） | 新增 `test_28`（单分片落盘 + 分片命名回归），已提交 `0a8f76a` |
| B2 | 18 个 legacy 测试构造 2/3-slot `ComposeSelection`，与统一 4-slot 契约冲突 | 夹具补齐为 4-slot（pad ID=-1、gate=0），保留原断言语义；`test_gate_validation` 原本因 slot 数错误"假通过"——改为真正测试 gate 校验；l2/形状拒绝测试同步强化 | 全套 392 passed（提交 `f082390`） |

## 3. Formal V7 training contract

- 数据：仅使用声明的 train split（`write_full_split_with_unique_ids` 加唯一 ID），无隐式 30 步、无 `records[:N]`、无 max_samples、无 teacher 子集。
- 覆盖审计：形式模式 `unique_train_sample_ids_seen == num_train_samples`，否则 RuntimeError（FAIL）；smoke 仅经显式 `--smoke-max-steps` 缩短。
- 监督：与 pruning NLL 共享 `compose_train_preprocess_v1_shifted`（prompt token `IGNORE_INDEX`、正确自回归 shift、零 answer token 失败）。
- RMS：validation-only 标定、checkpoint-hash 绑定、commit 冻结 kappa；推理应用 persisted calibration。
- 图像/运行时契约：`pad` / `mm_vision_select_layer=-2` / `feature=patch` / `mlp2x_gelu` / `projector_path` 在训练、剪枝、推理三处经 runtime-contract 审计一致。
- Recipe（smoke 显式覆盖处除外，形式模式默认）：1 epoch、lr 2e-4、warmup 0.03、cosine、batch 1 × accum 64、bf16。

## 4. Parameter update audit

| 参数组 | 状态 | 证据 |
|---|---|---|
| Base / Vision / Projector / Embedding / Query | 冻结（无梯度更新） | S3 训练中 hook 审计 + 相关断言；25 步两任务无该等参数梯度异常 |
| Historical experts Key + LoRA + RMS | 永久冻结 | init 捕获 checksum，`assert_task_freeze_integrity` 每步断言；跨任务位级验证（见 §7） |
| 选中当前候选 Key/LoRA | 更新 | 每 micro-batch 梯度记账，key grad norm 0.022-0.036 / LoRA-B 896/896 finite |
| 未选中当前候选 | 无梯度 | `_v7_key_gradient_ids` 断言（accumulation-aware，每 micro-batch 精确归属） |
| Old+Old 锚定行 | 0 值 loss → optimizer no-op | 设计内，保证无损非选中状态 |

## 5. Data isolation

- Task0：train 32（官方 ImageNet-R train 切片 0:32）/ val 8（200:208）/ test 4（400:404）。
- Task1：train 32 / val 8 / test 4，同法从 ArxivQA 官方 train 文件不相交切片。
- 溯源：split 文件均经 sha256 + `audit_split_isolation`；首版切分泄漏 8 个 sample_id → 审计在训练前正确 FAIL（P1-4 实弹证明），修正后 sample/record/image overlap 全 0。
- 推理：test 数据只出现在 S6 推理（`task_id_used=false`），从未进入训练、RMS 标定或剪枝 NLL。

## 6. Tests

- 命令：`pytest tests/compose -q`
- 结果：**392 passed, 3 warnings, 8 subtests passed in 68.19s**（覆盖 test_01..test_28 系列、selection/linear/l_pair_scale/g_gradient_isolation/conditional_training/expert_manager 等）。
- 分类：无失败。无 NEW REGRESSION、无 PRE-EXISTING STALE TEST、无 EXTERNAL DEPENDENCY。本会话两处改动（`f082390` 夹具对齐、`0a8f76a` nll_eval 修复 + test_28）均带回归测试。

## 7. Smoke results（bounded，真实 7B 权重）

### Smoke A — Task0 ImageNet-R（GPU2，随后 GPU2/3 被他人任务占用前完成）

- 训练：25 步全部有限（answer loss 2.868 → 0.009；total ≤ 3.46）。4 个候选全部使用：pair (0,1)×3、(1,2)×14、(0,3)×7、(2,3)×1；NewNew×25（Task0 无历史属预期）。
- 梯度：选中候选 key grad 0.022-0.036；LoRA-B 896/896 finite；freeze 断言全程通过。
- 剪枝：迭代 3 轮——iter0 删 3、iter1 删 0（幸存池重评）、iter2 最小池保护保留 1、2（贡献为正）。轨迹 9 行完整落盘。最终 selectable pool [1,2]。
- RMS：validation-only、224/224 层 kappa、commit 冻结；manifest 补丁后 bin-hash 未变。
- 推理（4 test 样本）：`task_id_used=false`，可见 experts [1,2]，896/896 LoRA 载入，RMS calibration 应用。
- Resume：`--stop-after candidates` → resume → bug 修复后再次 resume，三段幂等续跑成功（stage-marker）。

### Smoke B — Task1 ArxivQA（GPU3 启动时 OOM → 用户批准改 GPU0）

- 首启 GPU3 在 `model.to()` OOM（差 86 MiB）——根因是他人长任务（xukunlun、caizhengyuan）占用 GPU2/3，非代码缺陷；S0-S2 已完成后于 GPU0 resume S3-S6。
- 训练：25 步有限（total ≤ 1.98），cosine lr 2e-4 → 0；候选 4-7 全部使用（pair (4,5)×10、(6,7)×9、(5,7)×3、(4,6)×2、(4,7)×1）；路由 NewNew×25（新任务 key 近 ArxivQA 中心自然胜出，未引入配额）。
- 历史不变性：**专家 1、2 的 kappa 224/224 层 Task0→Task1 逐位相等；历史 key 张量位相等**；历史 LoRA 冻结断言全程通过。
- 剪枝：iter0 删 6、iter1 删 4（6 删除后贡献才转负——迭代重评生效），保留 5、7（贡献 0.164/0.337）。最终池 [1,2,5,7]。
- Committed manifest：4 frozen experts；RMS = 冻结 Task0 kappa + 新专家 5、7 动态 kappa 合并。

## 8. Remaining risks（仅真实未验证项）

1. 正式六任务配置声明的 `v7_validation/*` instruction 文件尚未在
   `/data/dataset/zhaozhuofan/UCIT/v7_validation/` 制备——`v7_six_task_run.sh`
   按设计 fail-fast，正式跑前需先制备并通过 P1-4/P1-5 审计（不允许拿 test 当 validation）。
2. Task1 bounded smoke 仅自然出现 NewNew 路由；带真实历史的 OldOld/OldNew
   需更长训练或基于真实 key 的几何探测。可达性由无配额几何单元测试
   （test_04/06/07/08）覆盖，路由器本身未被证明会在有限步内访问全部三种类型。
3. GPU2/3 现被他人长任务占用；正式跑前须复查 `nvidia-smi` 与剩余显存。
4. smoke 是生命周期/集成验证，非收敛声明（25 步 loss 趋势仅证有限性）。

## Verdict

**FORMAL_EXPERIMENT_READY = YES**

判定依据：P0-1..P0-4、P1-1..P1-5、S13 全部落实并有代码/审计/实弹证据；
unit + regression 全套 392 passed 无未分类失败；Task0/Task1 两个 bounded
real-7B S0-S6 全生命周期通过；数据隔离、覆盖审计、历史冻结（位级）、
迭代剪枝、最小池保护、RMS 统一均有运行证据。正式六任务运行仍受 §8.1
（validation 文件制备）与 §8.3（GPU 占用）两个运行前置条件约束——二者是
数据/资源准备问题，不是代码缺陷。
