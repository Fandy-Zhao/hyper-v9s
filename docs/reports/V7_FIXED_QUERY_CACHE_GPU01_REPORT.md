# V7 Fixed-Query 全量预计算实验报告（GPU0+GPU1）

- Date: 2026-09-03
- Branch: `feat/0903-v7-throughput-equivalence` @ `c93f51e02492abf62c70f2c585bf95dffb538490`
- Repo: `/home/zhaozhuofan/Hyper-LlaVA`
- Run root: `/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_fixed_query_cache_gpu01_20260903/`
- Status doc updates: `PROJECT_STATE.md`, `CHANGELOG.md`, `docs/module_status.md`

## 1. 背景与目标

V7（Global Key–Expert co-evolution）在 S3 训练 / RMS / Pruning / 评测阶段反复用同一
定义重算 query（visual+text 双塔 CLIP-L/14@336），其中 query 计算不含任何可训练参数、
与 answer/expert 完全无关，因此可一次性"全量预计算"成缓存，供后续阶段只读复用。
本实验在独立于 V7 主流程的新文件上，用 GPU0+GPU1（仅这两张空闲 4090）把
6 任务 × train/val/test 的全部 230,780 条 query 与 6 个 full-train task center 预计算落盘，
并通过"单卡 vs 双卡有界正确性门"证明切分计算与单机单卡位等价。

## 2. 范围与铁律（执行约束）

- 只使用物理 GPU0 + GPU1（RTX 4090）；GPU2–7 有他人在用。全程未 kill 任何他人进程、
  未 reset 任何 GPU、未触碰 GPU2–7、无任何抢占行为（`_physical_gpus_free` 探测失败即拒绝执行）。
- 不修改 V7 主流程任何文件：LoRA 训练、Global Top-2、RMS、Pruning、评测全部未动
  （见 §6 变更清单：仅 3 个新增文件 + 2 个新增目录）。
- query 计算 **复用既有实现**（下述契约），不为提速改变 backbone/text encoder/tokenizer/
  预处理/LN/concat 顺序/L2Norm/dtype。

## 3. 查询定义与实现契约（精确复用）

每个样本的固定 query 严格等于 V7 现役定义：

```
q_i = L2Norm(concat(LN(z_visual), LN(z_text)))      # 1536-D fp32, detached
```

- `F.layer_norm(weight=None, bias=None)` 逐模态归一 → visual 在前、text 在后 concat
  → `F.normalize` → `.detach()`；
- 无 Router / Expert Pool / Candidate Key / answer 依赖；无可训练参数；
- 实现指纹 `query_impl_hash = "v7_fixed_layernorm_concat_l2_v1"`（写在每个 split 的
  contract 中，与既有实现一致）。

## 4. 运行期特征契约

- `CLIPModel.from_pretrained(torch_dtype=float16).eval()`；text 分支经 `CLIPModel` 前向
  后在 fp32 上做 query 数学（与既有 `compose/eval/query_features.py` 参考管线一致，
  见 §16 的 bit-exact 证据）；
- `CLIPProcessor(text=[question_text(record)], images=[PIL RGB], padding=True, truncation=True)`；
  预处理产物 `outputs.image_embeds.float()`；query 张量 fp32 `[B, 1536]`；
- backbone：`clip-vit-large-patch14-336`
  `/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336`
  （backbone 内容 sha256 `cccaae94d2376bc5ae0a94160622e328245857a3832627c8390af124158c0b1f`
  写入 contract）。

## 5. 声明数据划分（configs/v7_ucit_formal.yaml）

| task | train | val | test(指令 3000) | 数据根目录 |
| --- | ---: | ---: | ---: | --- |
| 0 ImageNet-R | 23,742 | 256 | 3,000 | `/data/dataset/zhaozhuofan/UCIT` |
| 1 ArxivQA | 39,720 | 256 | 3,000 | 同上 |
| 2 VizWiz | 39,520 | 256 | 3,000 | 同上 |
| 3 IconQA | 29,603 | 256 | 3,000 | 同上 |
| 4 CLEVR | 39,743 | 256 | 3,000 | 同上 |
| 5 Flickr30k | 38,916 | 256 | 3,000 | 同上 |
| **合计** | **211,244** | **1,536** | **18,000** | **230,780** |

source 文件：train=`UCIT/v7_train/{T}/train.json`，val=`UCIT/v7_validation/{T}/validation.json`，
test=`UCIT/instructions/{T}/test_3000.json`；image_folder=`UCIT/datasets`。
**全部按声明整段计算**：18 个 split 无 subsetting、无抽样（§17 计数审计证明
declared = saved = unique）。

## 6. 变更清单（全部新增，零修改）

| 路径 | 类型 | 内容 |
| --- | --- | --- |
| `compose/v7/query_cache.py` | 新增库 | 契约、切分、partial、merge、原子写、校验、task center 读写 |
| `compose/eval/precompute_v7_queries.py` | 新增 CLI | precompute / worker / gate 三种模式、编排、统计、manifest |
| `tests/compose/test_query_cache_audit.py` | 新增测试 | 28 个 CPU 测试（§28） |
| `docs/reports/V7_FIXED_QUERY_CACHE_GPU01_REPORT.md` | 新增报告 | 本文 |
| `artifacts/v7_query_cache/query_cache_manifest.json` | 新增机器可读 manifest 镜像 | 与 run root 内副本同内容（§27） |
| `PROJECT_STATE.md` / `CHANGELOG.md` / `docs/module_status.md` | 状态文档更新 | §36 摘要 |

V7 现役文件（训练/路由/Key-loss/Global Top-2/RMS/Pruning/评测）**零改动**。

## 7. 执行环境

- 平台：Linux 5.4.0，conda env `hyper`（torch 2.3.1+cu118，transformers 4.33.3）；
- 节点 GPU：8×RTX 4090；实验时 GPU0/GPU1 空闲（4 MiB），GPU2–7 被他人占用
  （>15 GiB 常驻），`_physical_gpus_free` 探测与"拒绝占用 GPU2+"逻辑生效；
- 每 worker 最大显存 1.738 GiB（含 fp16 backbone 双塔）。

## 8. 执行架构（真数据并行，非 torchrun）

- 单进程 orchestrator 依序处理 task×split；每个 split 起 2 个 worker 子进程，
  各绑定一张物理 GPU（`CUDA_VISIBLE_DEVICES=<gpu>` 单卡可见）；
- 每个 worker 只写自己 rank 的 tmp partial（`tmp/taskN.split.rank{R}.pt`）；
- 两 rank 全部成功退出后 orchestrator 做 10 项 merge 审计并写 official cache；
- 任一 worker 非零退出 → 该 split 抛错终止（不再并发下一任务），partial 保留供续跑。

## 9. 确定性切分规则（index % world）

- 切分：全局样本按下标 `index % world_size == rank` 划分（declared 全序来自
  JSON 文件顺序，split 定义与训练端一致）；
- merge：全局行 `j*world + rank` ← rank 的局部行 j（与切分互为逆）；
- **错误优先级**：先做排序并集覆盖检查（missing/foreign 报 "merge coverage
  mismatch: X missing, Y foreign"），再查每 rank 是否恰好覆盖自己的确定性切片——
  保证审计报错指向真实根因（有 2 个专项测试锁定该顺序，§28）。

## 10. 原子写入与暂存

- worker 只允许写 `run_root/tmp/`；official 区只在 merge 通过后写入；
- official 文件写入路径：同目录 `mkstemp` → 写 → `fsync` → `os.replace` 原子改名；
  元数据 JSON 同样式；目录内无 `.tmp` 残留（ls 复核为空，测试断言 3 条路径）；
- 官方路径一旦出现即完整：半写文件不可能以正式名存在。

## 11. 契约绑定与自动失效（resume 的合法性来源）

每个 split 的 metadata 携带 contract（`SplitContract`）：

- git：`git_sha`=`c93f51e0…`、`git_branch`=`feat/0903-v7-throughput-equivalence`；
- 数据：`source_dataset_sha256`（文件级 sha256）+ `source_content_hash`
  （逐样本 content 哈希），`image_folder`；
- 实现：`query_backbone_hash`（目录内容 sha256）、`query_impl_hash`、`query_dtype`、
  `query_dim`、schema；
- 执行：`world_size`、`shard_algorithm`、`batch_size`；
- `contract_hash` = 以上绑定值的哈希，写进 metadata 与 manifest；partial 有效性与
  official cache 是否 current 均由 contract_hash 判定——任何绑定项变化即自动失效重算
  （对 query 库的 28 个测试含 hash 变动 → 判定失效的用例）。

## 12. 校验门设计（先于全量执行）

- 阶段 A：GPU0 单卡（world=1）在真实 ImageNet-R train 样本上算 128 条；
- 阶段 B：GPU0+GPU1 双卡（world=2，index%2 切分）同 128 条；
- 门限：合并后与单卡逐样本余弦 `>= 1 - 1e-6` 才 PASS；**若失败立即停止并根因分析，
  不归咎"多卡误差"**；
- 附加烟测：用这 128 条 query 对一个**既有** key pool（休眠 formal 2-GPU run 的
  S2 `candidate_keys.pt`）跑 Global Top-2，比对路由一致性；
- 每阶段都走与全量相同的 tmp→merge→audit 管线（含 rmtree tmp 的干净现场）。

## 13. 校验门结果（verdict = PASS）

`_gate/gate_result.json`（bounded_real_data_single_vs_dual）：

| 指标 | 值 |
| --- | --- |
| bounded_samples | 128（task0 train 前 128 条真实样本） |
| 单卡 vs 双卡 | `exact_bit_equal: true`，`max_abs_diff = 0.0`，`max_relative_diff = 0.0` |
| cosine | mean `1.0`，min `0.99999976 >= 1-1e-6` |
| task center（128 条 full-train center） | `exact_bit_equal: true`，cosine_min `0.99999994` |
| 边界墙钟 | single 14.869s / multi 14.846s（speedup 1.002×，边界量小、启动占主导） |

两阶段 merge 审计均 10 项全绿（declared=merged=128、dup/missing/unknown=0、
norm∈[1−1.2e-7, 1+1.2e-7]、1536-D fp32、finite）。

## 14. 单卡 / 双卡等价性结论

128 条真实样本上双卡并行与单卡**逐位相同**（非仅近似）：cosine_min 偏差
`2.4e-7` 来自 fp16 CLIP 前向本身（单卡内同一条 query 与 L2 参考也在此量级），
切分与合并不引入任何额外误差。等价性证据链：gate 两阶段 partial bit-equal → merge
重建顺序数学可逆 → 全部 18 个全量 split 的 230,780 条与单卡管线（同 128 条上重跑对照）
同源同码（§16）。

## 15. Top-2 路由一致性烟测

pool：`…/v7_ucit_formal_2gpu_b2a16_w4_seed42_20260903/task0/state/candidate_keys.pt`
（该 run 处于休眠态，仅只读其已落盘 S2 产物作参照，未唤醒任何进程）：

- `top2_agreement_rate = 1.0`，`agreement_exact: true`，`disagreements: []`；
- `max_abs_score_diff = 0.0`（打分逐位一致），`visible_expert_ids = [0,1,2,3]`。

→ 预计算 query 喂给现役 Global Top-2 得到与现有候选池完全一致的路由结果。

## 16. 吞吐优化（不改变数值的提速，已用位等价证明）

- 根因：transformers 4.33 `CLIPProcessor` resize 在预处理主线程**串行**执行 PIL
  BICUBIC（最短边 336），约 1.4 s/batch，GPU duty cycle 仅 ~5%（曾导致一次对
  健康进度的误判——低 GPU 采样率 ≠ 卡死，prep 瓶颈时属正常）；
- 修复：把 resize 搬进 decode 线程池（与 text tokenize 并行），复用 transformers
  完全相同的数学（`get_resize_output_image_size` 最短边 336、`int` 截断、无 max_size、
  同 filter）；`processor(...)` 收到的是已 resize 好的 PIL → 之后 transform 路径不变；
- 位等价证明：同一批真实图片 `torch.equal == True`；gate 复跑 PASS；新旧管线同
  128 个 id、同 contract hash 的 partial 对比 `exact_bit_equal: true`；
- 效果：~17 → ~55–65 samples/s/worker（≈3.5×；双卡 95.5 → 111–130/s，§23）。

## 17. 全量覆盖结果（18 split × 计数审计全绿）

每 split 的 metadata `merge_audit` + manifest 计数复核：

| split | task | n(declared=saved=unique) | split | task | n |
| --- | --- | ---: | --- | --- | ---: |
| task0.train | ImageNet-R | 23,742 | task0.test | ImageNet-R | 3,000 |
| task1.train | ArxivQA | 39,720 | task1.test | ArxivQA | 3,000 |
| task2.train | VizWiz | 39,520 | task2.test | VizWiz | 3,000 |
| task3.train | IconQA | 29,603 | task3.test | IconQA | 3,000 |
| task4.train | CLEVR | 39,743 | task4.test | CLEVR | 3,000 |
| task5.train | Flickr30k | 38,916 | task5.test | Flickr30k | 3,000 |
| val 合计 | 6×256 | 1,536 | **总计** | **230,780** | 与声明逐项相等 |

train 合计 211,244 与 yaml 声明严格相等；`duplicate=missing=unknown=0` 全部 split；
每 split `num_declared_samples == num_saved_queries == num_unique_sample_ids == n`
与 manifest `sample_count` 一致。

## 18. 样本身份审计

- sample_id 规则：train 用顶层 `id`（`v7_t{task}_train_{i}` 形式），val/test 用
  `question_id`（`sample_id_of` 与训练端同一函数）；
- 每 split 存 `ordered_sample_ids`（JSON 序全列）并算两个哈希：
  `sample_id_set_hash`（无序集合）与 ordered 序列哈希；merge 时两哈希必须等于
  declared（读回 JSON 全量重算）→ 覆盖顺序与内容双重绑定；
- 全量 18 split：missing / duplicate / unknown = 0（§17 表），
  `ordered_id_hash_matches_declared` / `sample_id_set_hash_matches_declared` = true。

## 19. 张量审计

全部 18 split（读回磁盘张量复核）：

- 形状 `[n, 1536]`，`dtype=float32`，`all_finite=true`；
- 范数统计 ∈ [0.99999976, 1.00000024]（L2 归一在 fp32 舍入内，均值 1.0）；
- `query_tensor_hash`（字节级 sha256）写入 metadata/manifest，用于事后篡改检测；
- 有界门内另有 `torch.equal` 级证据（§13/§14）。

## 20. Task Center（full-train）

- 6 个 center 全部由**对应任务完整 train cache** 计算（`center_counts_equal`，
  `num_declared_train_samples == num_queries_used_for_center` 全 true，
  即 23742/39720/39520/29603/39743/38916）；
- 复用 `compose.v7.query.full_train_task_center`（与现役 full-train center 同一数学）；
- 存 `query_cache/task{N}/task_center.pt`，metadata 记 `task_center.source_query_cache_hash`
  与 center 计数；128 条有界样本上的 center 对比 `exact_bit_equal: true`（§13）。

## 21. 元数据与溯源（每 split）

`metadata.json` 顶层：kind `v7_fixed_query_split_cache`、schema_version 1、created_at、
contract（§11 全字段）、merge_audit（§17–19）、worker_stats（每 rank：samples、
prep/decode/forward 累计时间、model_load、gpu_max_memory、local/physical GPU 映射）、
runtime（physical_gpus、world_size、GPU util 采样）、queries_file、双哈希。
来源链路：git sha → JSON 文件 sha256+content hash → backbone 内容 sha256 →
query_impl_hash → contract_hash，任一层变化都会使 split 失效重算。

## 22. 断点续跑验证（真实发生）

- 中途发生过一次执行器被杀（ArxivQA 阶段，健康误判后修正，§16）；重启同一命令：
  - 已完成 split：`official cache present and contract-current; skip`（ImageNet-R
    train/val/test 等全部跳过，缓存原样保留）；
  - 未完成 split：partial 有效则从断点续算、无效则整段重算；
- 终局复跑（manifest 修复后第 3 次进入）：18 行 skip + 6 个 center 重算 + manifest/
  stats 写出，全程无重算，幂等成立；tmp 目录终态为空。

## 23. 性能统计（真实双卡全量）

worker 日志末行解析（`run_root/logs/*.log`；每 split wall = 两 rank 慢者）：

| split | n | 双卡 samples/s | r0/r1 (s) | 备注 |
| --- | ---: | ---: | --- | --- |
| ImageNet-R.train | 23,742 | 95.5 | 243.8 / 248.5 | 预优化管线（03:50，值=旧管线证据） |
| ArxivQA.train | 39,720 | 111.5 | 344.8 / 356.1 | 解码线程预 resize |
| VizWiz.train | 39,520 | 114.6 | 344.9 / 339.0 | 同上 |
| IconQA.train | 29,603 | 118.9 | 249.0 / 244.9 | 同上 |
| CLEVR.train | 39,743 | 123.8 | 320.9 / 313.2 | 同上 |
| Flickr30k.train | 38,916 | 130.3 | 298.7 / 293.5 | 同上 |
| test×6 | 18,000 | 49.9–97.4 | 30–60 | 每 rank 25–49/s；ImageNet-R.test 为预优化 |
| val×6 | 1,536 | 20–26 | 9.7–12.4 | 含 ~7.5 s 模型加载/任务，小量启动主导 |

优化后 train 单 worker ≈ 55.8–65.1 samples/s（前代管线 ~47.8）；合计有效双卡
wall ≈ **2,114.6 s ≈ 35.2 min**（train 1818.1s + test 228.7s + val 67.8s），
不含 gate（~30 s）与编排/模型加载间隙；日历时长 03:45→04:55（含误杀重跑）。

## 24. 显存与资源占用

- 每 worker 峰值显存 **1.738 GiB**（fp16 CLIP-L/14 双塔 + 批激活），两卡合计
  ~3.5 GiB，与 GPU2–7 上他人作业零冲突；
- GPU util 采样多数时刻 <30%（prep 瓶颈）属预期（§16）；无任何越界行为；
- 计算侧仅 CUDA 调用、无系统级变更；fork 出的 ~32 个进程为库级
  ProcessPoolExecutor 的空闲 `pipe_wait` 子进程（CLIPModel 加载时产生，无害）。

## 25. 磁盘占用估算

- 单条 fp32 query = 1536×4 B = **6,144 B**；
- 理论：train 1,237.8 MiB + val 9.0 MiB + test 105.5 MiB ≈ **1,352 MiB ≈ 1.32 GiB**
  （与任务书 ~1.3 GiB 估计一致）；
- 实测 `du`：**1.4 GB**（含 18×metadata.json、双哈希列表与目录块）；
- 单 split 示例：task1.train 39720 条 tensor 233 MiB + metadata ~1.1 MiB；
  运行期冗余仅 tmp partial（≤2×1 split），merge 后清理，终态 tmp 为空。

## 26. 产物目录布局

```
v7_fixed_query_cache_gpu01_20260903/
├── query_cache/
│   ├── task{N}/  (N=0..5)
│   │   ├── task_center.pt
│   │   ├── {train,val,test}/metadata.json
│   │   └── {train,val,test}/queries.pt      # [n,1536] fp32
│   ├── query_cache_manifest.json            # 机器可读清单（与仓库 artifacts/ 镜像一致）
├── metrics/precompute_stats.json            # 6×task_center 校验
├── metrics/manifest_summary.json            # manifest 汇总
├── logs/{task}.{split}.rank{0,1}.log        # 36 worker 日志
├── tmp/                                     # 终态为空（partial 已清）
└── _gate/{single,multi}/ + gate_result.json # 有界门证据
```

`queries.pt` 内含 `queries` + `sample_ids`（torch.load `weights_only` 兼容）。

## 27. 机器可读 Manifest

`query_cache_manifest.json`（run root 与仓库 `artifacts/v7_query_cache/` 双份，
sha256 一致）顶层：`kind=v7_fixed_query_cache_manifest`、schema_version 1、
git_sha/git_branch、physical_gpus=[0,1]、local_cuda_mapping、query_mode=`v7_fixed`、
query_dim 1536、dtype float32、cache_root、tasks{task_name}{train|val|test}{
sample_count、query_hash、sample_id_hash、contract_hash、source_dataset_sha256、
path、metadata_path}。6×3 全条目与 metadata 逐字段一致（读回抽样核对）。

## 28. 测试与复验

新增 `tests/compose/test_query_cache_audit.py`：**28 tests 全部通过**
（切分计划、merge 审计含覆盖-优先报错顺序、原子写无残留、contract 哈希与失效判定、
原子 partial 有效性、norm/center 对 `full_train_task_center` 的对拍、cosine 比较容差、
record→sample_id/数据契约）；既有回归 `48 passed`（V7/query 相关）不受影响。
复验命令：

```bash
PYTHONPATH=. python -m pytest tests/compose/test_query_cache_audit.py -q
# 只读审计（无 GPU）：逐 split 打印 declared/saved/unique、哈希、范数、dim/dtype
# 产物即 §17–§19 表格数据来源；重跑 orchestrator（幂等，skip 全部完成 split）：
PYTHONPATH=. python -m compose.eval.precompute_v7_queries --mode precompute \
  --config configs/v7_ucit_formal.yaml \
  --out-root /data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_fixed_query_cache_gpu01_20260903 \
  --gpus 0,1
```

## 29. S3 训练阶段复用可行性（YES）

- 现状：S3 训练每样本现算 query（DataLoader 内 CLIP 前向）。全量预计算缓存可直接替换；
- 现役消费点审计：`compose/train/train_compose.py:504-506` 打开 split JSON 建 dataset
  （缓存模式读取点），`compose/train/data.py:347-353` 构建 `V7QueryDataset`
  sample_id→query 索引，**唯一改动点 `data.py:369` 一行**：该处当前每样本现算 query，
  改为查缓存张量（id 命中即取行，miss 才回退现算）；sample_id 规则与缓存写入端同一函数，
  `full_train_task_center` 直接读 `task_center.pt`；
- 约束成立性：query 无 answer/expert 依赖、无 trainable 参数 → 训练前计算与
  训练中计算**数学上同一**（有界门 + 全量 contract 保证）；fp32 缓存张量 `.detach()`
  语义等价；缓存不进入梯度图。
- 边界提示：`compose/v7/workflow.py:64-75` 的 `queries_from_cache` 按排序后的 id
  返回行（S3 训练侧是顺序/随机访问）——若做二进制换入需按该顺序或按 id 查表
  （对拍数据已含 ordered id 列表，可无损映射）。

## 30. RMS（Root-Mean-Square 预处理）复用可行性（YES）

- 审计结论：RMS 阶段计算 **0 条固定 query**——它只消费训练产生的（Key, 统计）产物，
  与样本级 query 无关；因此无需改任何 RMS 代码；
- 唯一关联：若未来 RMS 需要 center 作为参照，可直接读 `task_center.pt`（现役
  `full_train_task_center` 同一数学，§20）。

## 31. Pruning 复用可行性（YES）

- 审计：`compose/v7/pruning.py` 的剪枝打分全部来自缓存张量（Key/LoRA 产物），
  对 6 个 full-train cache 的**实时 CLIP 计算量为 0**；本缓存不改变其输入格式；
- 若后续要剪枝样本级 Key 缓存（现在不需要），可用 `sample_ids` 列表 + 现役
  `queries_from_cache`，注意 §29 的 id 顺序提示。

## 32. 评测（S6）复用：保持现状为"非正式管线"（符合声明）

- `compose/eval/eval_task.py:315-333` 评测时**逐条现算** query（eval 分支为 live
  per-record CLIP）。任务书将 S6 定位为非正式管线（test 标签 not-formal-pipeline），
  且 test 3000×6 cache 仍已按"合规额外计算"全部预计算落盘（§17）；
- 因此评测侧不改动；若后续要测缓存在线评测，直接以本 test cache 替换同 id 现算
  路径即可（数值等价由 §13/§14 保证）。

## 33. 结论判定总表（YES/NO）

| 编号 | 结论行 | 判定 | 依据 |
| --- | --- | --- | --- |
| C1 | TWO_GPU_QUERY_COMPUTE | PASS | 6×train+6×val+6×test 全部落盘（§17） |
| C2 | FULL_SAMPLE_QUERY_COVERAGE | PASS | 230,780 = 声明 211,244+1,536+18,000，0 dup/miss/unk |
| C3 | QUERY_DIM_1536 | PASS | 全 split `[n,1536]` fp32（§19） |
| C4 | SINGLE_MULTI_GPU_EQUIVALENT | PASS | 128 条真实样本 bit-exact，cosine_min 0.99999976（§13–14） |
| C5 | TOP2_ROUTING_EQUIVALENT | PASS | 现役候选池 Top-2 一致率 1.0、打分 0 差（§15） |
| C6 | TASK_CENTER_EQUIVALENT | PASS | 6 center counts equal；128 条上 bit-equal（§13/§20） |
| C7 | CACHE_ATOMIC_AND_RESUMABLE | PASS | mkstemp→fsync→rename；真实断点续跑 3 次幂等（§10/§22） |
| C8 | QUERY_CACHE_READY | YES | 18 cache+6 center+manifest 齐备、审计全绿、契约绑定 |
| C9 | TRAIN_QUERY_CACHE_REUSE_FEASIBLE | YES | 最小改动 `data.py:369` 一行（§29） |
| C10 | RMS_QUERY_CACHE_REUSE_FEASIBLE | YES | RMS 计算 0 条固定 query，无需改动（§30） |
| C11 | PRUNING_QUERY_CACHE_REUSE_FEASIBLE | YES | Pruning 0 条实时 CLIP，输入格式不变（§31） |
| C12 | EVAL_QUERY_CACHE_REUSE_FEASIBLE | YES | S6 现算路径保留；test cache 就绪、等价有证（§32） |
| C13 | NEXT_STAGE_CODE_MODIFICATION_READY | YES | 改动点已定位并给出顺序语义边界（§29 边界提示） |

无阻塞项（C8 判定无 YES-blocker：实现、数据、GPU、契约、审计全部满足）。

结论行（独立、可机器匹配）：

```text
TWO_GPU_QUERY_COMPUTE=PASS
FULL_SAMPLE_QUERY_COVERAGE=PASS
QUERY_DIM_1536=PASS
SINGLE_MULTI_GPU_EQUIVALENT=PASS
TOP2_ROUTING_EQUIVALENT=PASS
TASK_CENTER_EQUIVALENT=PASS
CACHE_ATOMIC_AND_RESUMABLE=PASS
QUERY_CACHE_READY=YES
TRAIN_QUERY_CACHE_REUSE_FEASIBLE=YES
RMS_QUERY_CACHE_REUSE_FEASIBLE=YES
PRUNING_QUERY_CACHE_REUSE_FEASIBLE=YES
EVAL_QUERY_CACHE_REUSE_FEASIBLE=YES
NEXT_STAGE_CODE_MODIFICATION_READY=YES
```

## 34. 风险与遗留

- 低：test/val 小 split 吞吐数字受模型加载主导，不代表稳态速率（§23 注明）；
- 低：`queries_from_cache` 的 id 排序语义（§29 边界提示）——二进制换入前需按
  ordered ids 校验一次；
- 中（外部）：GPU2–7 继续被他人占用，任何"扩到 >2 卡"都需重新走 `_physical_gpus_free`
  与 gate（本实现支持任意 world，但本实验只证明 world∈{1,2}）；
- 无遗留 tmp/半写文件；无未完成 split；`_gate` 证据保留在 run root。

## 35. 变更文件、测试与下一步

变更：3 个新增源文件/测试 + 本报告 + `artifacts/v7_query_cache/query_cache_manifest.json`
+ 3 个状态文档；V7 主流程文件零改动。
测试：28 个新增全过；既有回归 48 全过；真实数据 gate PASS（§13）。
下一步（未执行，待授权）：若启动 S3 训练缓存化，按 §29 在 `compose/train/data.py:369`
做一行替换 + 一次 seed 固定对拍（缓存 vs 现算 同 batch 逐位相等），即可移除训练侧
CLIP 前向；此后本缓存由 contract_hash 自动失效机制守护。
