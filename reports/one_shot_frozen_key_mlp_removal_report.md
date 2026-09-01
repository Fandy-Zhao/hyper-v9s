# 取消 MLP Router 详细修改报告

> 实验：One-shot Learnable Expert Key 可行性验证（规范第 18 节约束之一：
> **删除/绕过 MLP Router**，仅保留 cosine 检索 + 可学习 key + 阈值路由）
> 日期：2026-08-29 ｜ 实现：`compose/router/one_shot_key.py`（commit `545efa7`）

## 1. 被取消的对象：原有 MLP Router 完整组成

正式 V6.2 推理链 `ComposeRouter.select()` 由三个可学习模块组成：

| 模块 | 位置 | 结构 | 作用 |
|------|------|------|------|
| `MultimodalQueryEncoder` | `compose/router/query_encoder.py:35` | `image_projection Linear(→128)` + `text_projection Linear(→128)` + `gate Sequential(Linear(258→128), Sigmoid)` | 768-d 多模态特征 → 128-d query（MLP 投影） |
| `ExpertSetRouter`（set_router） | `compose/router/set_router.py:25` | `CardinalityHead(query_dim, top_m)` + `SymmetricPairScorer(query_dim)` + `single_bias` | top_m 检索后：空/单/对基数分类 + pair 打分（MLP 决策） |
| `predict_sets(output, pair_threshold)` | — | MLP 输出 → 阈值决策 | 最终 empty/single/pair 集合 |

正式模式开关：`set_router_enabled`（默认 `False`，V6.2 最终正式配置启用）。

## 2. 当前实现的绕过机制（三层）

### 层 1：推理/评估路径——完全不实例化 MLP

`evaluate` / `simulate` / R1 评估全部走 `probabilities_of()`（[one_shot_key.py:144-147](compose/router/one_shot_key.py#L144-L147)）：

```python
key_matrix = torch.stack([keys[value] for value in ordered])
similarities = F.normalize(queries, dim=-1) @ key_matrix.T        # 128-d × 128-d 纯 cosine
temperature = min(2.0, max(0.01, math.exp(log_temperature)))
return torch.sigmoid((similarities - bias_vector.unsqueeze(0)) / temperature)
```

- **不调用 `ComposeRouter.select()`**，不加载 base router checkpoint，`ExpertSetRouter` / `CardinalityHead` / `SymmetricPairScorer` / `predict_sets` 从未实例化
- query 已是 128-d（Phase D 用冻结 query encoder 预提取缓存），**连 `query_encoder` 也不在推理路径上**
- 决策 = `select_from_probabilities()` 阈值判定（tau_none → empty / tau_second → single / 否则 pair），零网络参数

### 层 2：训练路径——MLP 参数物理不可训练

`r2_train`（[one_shot_key.py:231-272](compose/router/one_shot_key.py#L231-L272)）加载 base router 仅作 **key_store 元数据容器**（校验 creation_task 与规范一致），然后四重禁用：

```python
router.set_router_enabled = False                                  # ① 关闭 MLP 路由开关
router.enable_key_only_multilabel(initial_temperature=0.1)         # ② 内部再次 set_router_enabled = False；
                                                                   #    只建 key_only_bias + log_temperature（无 MLP）
for p in router.query_encoder.parameters(): p.requires_grad = False  # ③ MLP 全冻结
for p in router.set_router.parameters():     p.requires_grad = False  # ③
optimizer = torch.optim.AdamW(
    [parameter, bias_parameter, router.key_only_log_temperature], ...)  # ④ 优化器只含 key/bias/温度
```

- 损失计算走 `probabilities_of()`（纯 cosine+sigmoid），**不经 `set_router.forward()`**
- `optimizer` 参数列表 = 当前专家 key(128-d) + 当前专家 bias(标量) + 共享 log_temperature(标量)；MLP 权重不在其中，即使 requires_grad 误开也收不到梯度

### 层 3：持久化——产物不含任何 MLP 权重

| 产物 | 内容 |
|------|------|
| `r1_centroid_keys.pt` | keys(12×128) + bias(全 0) + log_temperature(0.0) + creation_task |
| `r2_sequential_keys.pt` | keys(12×128) + bias(12) + log_temperature(1) + snapshots/history |

均不含 query_encoder / set_router 任何权重；不调用 `state_dict_extra`。

## 3. 与既有 key-only 模式的区别（重要澄清）

既有 `compose/experiments/key_only_bestset.py`（joint R1）的 key-only 模式取消的是 **set_router（MLP 决策头）**，但推理仍经过 `ComposeRouter.select()` 的 key-only 分支——即仍跑 `query_encoder`（768→128 投影）。

本实验（`one_shot_key.py`）更彻底：**query_encoder 也不在推理路径上**（Phase D 已把冻结 encoder 的输出投影为 128-d 并缓存，`query_encoder_hash = d2bc62e…` 全 task 校验一致）。检索矩阵为 128-d query × 128-d key 直接点积，`query_encoder` 仅在 Phase D 一次性离线使用。

## 4. 可验证的证据链

| 检查点 | 证据 |
|--------|------|
| 优化器参数集 | `r2_train` 中 `trainable = [parameter, bias_parameter, router.key_only_log_temperature]`（[one_shot_key.py:272](compose/router/one_shot_key.py#L272)） |
| MLP requires_grad | 全部显式置 False（[one_shot_key.py:234-237](compose/router/one_shot_key.py#L234-L237)） |
| 推理不触 router 对象 | `evaluate`/`simulate`/`r1_centroid` 全程无 `ComposeRouter()` 实例化 |
| 查询无梯度 | 既有测试 `test_only_keys_bias_temperature_receive_gradient`（`test_key_only_bestset.py`）：queries.grad is None |
| 权重规模 | 可学习参数总量 = 12×128 + 12 + 1 = **1549 个标量**（原 MLP Router 含 query_encoder 3 个 Linear + CardinalityHead + SymmetricPairScorer） |
| checkpoint 校验 | `r2_train` 用 base router 元数据强校验：`creation_task` 与 `CREATION_TASK` 不一致即 raise（[one_shot_key.py:258-260](compose/router/one_shot_key.py#L258-L260)） |

## 5. 测试覆盖

- `test_one_shot_key.py::test_probability_retrieval_all_cardinalities`：纯概率检索三基数
- `test_one_shot_key.py::test_evaluate_retrieval_reports_oracle_regret`：检索+regret
- `test_key_only_bestset.py::test_compose_router_key_only_mode_has_no_mlp_dependency`：key-only 模式无 temperature 依赖、set_router_enabled False、bias/temperature 参数存在

8/8 单元测试通过；全量 413 通过（3 个环境性失败与本改动无关）。
