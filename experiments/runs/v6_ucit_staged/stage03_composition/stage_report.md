# Stage 03 Composition Report

Status: **PASSED**

## Preregistration

RMS 公式在正式测试前冻结：`c_i=sqrt(r_j/r_i)`、`c_j=sqrt(r_i/r_j)`，`pair_scale=1/sqrt(2)`，epsilon=`1e-8`，clip=`[0.25,4.0]`。校准集为 seed 42 下各 pair 训练 split 的固定前 16 个样本；config SHA-256 为 `29eacd7c25718e085a864a742b3c0d8ab343759b2be958202b11cc846923d0f2`，没有使用测试答案、测试预测或 benchmark 指标调参。

## Acceptance summary

- Compose 0/1/2 专家 `base_only`、`single`、`direct_sum`、`rms_calibrated` 全部可执行；Hyper 源码未修改。
- 完整 Compose unittest 83/83 通过；真实 4-rank bf16 pair smoke 通过，梯度隔离与 DDP 统计一致。
- Controlled direct-sum：4 pairs × 8 samples，logit/NLL 最大与平均差均为 0，prediction agreement=1。
- Single smoke：30 steps，loss=0.767935，ImageNet-R 128 samples accuracy=53.91%，checkpoint save/reload 通过。
- Mini2：`[[49.22,—],[50.00,79.69]]`；MAA=57.0325，MFN=64.845，MFT=63.675，BWT=2.34，无系统性 single 漂移。
- RMS 诊断完整保留：Independent B+C 改善；A+Independent B 与 Residual B+C 恶化；A+Residual B accuracy 持平而 NLL 恶化。不将 RMS 描述为 V6 效果修复。
- GPU 使用物理 4,5,6,7，global batch=24；未抢占其他用户、未 OOM。所有失败与 retry 均保留。

详细设计、数值、性能、失败和限制见 `docs/reports/v6_ucit_stage03_composition.md`。SHA-256 清单见 `checksums.json`。本阶段提交后停止，不进入 Stage 04。
