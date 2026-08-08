# Pre-Formal Code State（§1 冻结检查）

- **日期**：2026-08-08
- **HEAD**：`3c2a7d1ded478e0d7faaa4964097e8170bc9cb3c`
- **branch**：`main`

## 判定：**NOT FROZEN — BLOCKING（§1）**

正式实验不能基于未提交 working tree。当前 Compose 流水线核心代码**全部 untracked**。

## 状态摘要

| 类别 | 数量 | 说明 |
|---|---|---|
| staged deletions | 42 | 全部 v6_* 旧流程文件（已 staged 删除） |
| staged renames | 2 | `v6_assemble_expert→assemble_expert`、`v6_nll_eval→nll_eval` |
| unstaged modifications | 36 | 迁移修改（train_compose.py +250 等） |
| untracked | 96 | 其中 Compose 流水线核心 ≥40 个正式文件 untracked |

## Untracked 正式流水线文件（BLOCKING 证据）

### compose/ 核心
```
compose/expansion/commit.py
compose/expansion/expert_formation.py
compose/expansion/query_clustering.py
compose/expansion/residual.py
compose/experiments/snapshot.py
compose/experiments/task_run.py
compose/lora/rms.py
compose/router/functional_query.py
compose/router/key_learning.py
compose/router/router.py
compose/teacher/teacher.py
```

### configs/ + scripts/
```
configs/compose_ucit.yaml
configs/compose_ucit_smoke.yaml
scripts/Compose/Run_UCIT/six_task_run.sh
scripts/Compose/Run_UCIT/resume_run.sh
```

### tests/（A–O 关键测试 + 补充）
```
tests/compose/test_compose_a_per_sample_top_m.py  ...  test_compose_o_inference_purity.py
tests/compose/test_compose_p1_real.py
tests/compose/test_compose_root_cause.py
tests/compose/test_rms_image_question.py
```

## 其他 untracked（非 UCIT 正式路径，单独归类）

- `compose/data/`（controlled_format_v1 等 6 文件 + real_p1/）：0730 格式受控实验（另一条实验线，见 memory `format-controlled-composition-decision`）
- `compose/eval/`（d0–d6、p1_real、format_controlled 等 20+ 文件）：同上实验线与诊断脚本；其中 `query_features.py` 待确认归属
- `experiments/data/`、`outputs/`、`wandb/`、`sample_instructions/`、`docs/reports/*`：产物/报告

## 结论

§1 判定：**BLOCKING** —— 存在大量 untracked 正式代码。
本报告阶段只报告、不 commit。冻结动作留给用户决定（建议：先完成全部验证、修复后一次性提交迁移快照，再启动正式实验）。
