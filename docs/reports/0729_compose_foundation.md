# Compose Foundation 实现报告

## Branch

`feat/0729-compose-foundation`，基线提交 `8f4ae4fbcad5cf18418598331e013577156d06bf`。

## Summary

本任务新增了独立于 Hyper-LLaVA 的 Compose 基础实现。Compose 直接基于 Transformers LLaMA 与仓库现有 CLIP/projector 约定构建模型，不使用 `Hyper/peft`、HyperMOELora、Gaussian/Poincare 路由、InstanceModalityRouter、`cur_task` 或 task-ID 专家绑定。

基础版本支持：独立 LoRA 专家、ExpertPool、样本级 top-1/top-2 固定组合、版本化 adapter checkpoint、独立 UCIT Task1 训练入口，以及 DeepSpeed ZeRO-2 两步验证。

## Implementation

- `compose/model/`: `ComposeLlavaConfig`、`ComposeLlavaModel`、`ComposeLlavaForCausalLM` 与纯视觉多模态输入重组。
- `compose/adapters/`: `ComposeSelection`、上下文运行时、LoRA expert、线性层注入和统一管理。
- `compose/experts/`: expert 元数据、pool 状态和 adapter-only 保存/加载。
- `compose/train/`: 无 breakpoint 的 UCIT v1 数据处理、LLaVA Trainer 复用和独立训练入口。
- `scripts/Compose/Train_UCIT/`: 4-GPU Task1 全量脚本和 1-GPU 两步 smoke 脚本。
- `llava/__init__.py`: 顶层模型符号改为兼容惰性导入，避免工具模块导入时初始化 Hyper 栈。

## Validation

### Static and Unit Checks

- `python -m compileall -q compose tests/compose`: passed.
- `python -m unittest discover -s tests/compose -p 'test_*.py' -v`: 10/10 passed.
- `bash -n scripts/Compose/Train_UCIT/Task1.sh scripts/Compose/Train_UCIT/Task1_smoke.sh`: passed.
- Fresh-process import check: `compose.train.train_compose` did not load any `Hyper.peft` module.
- Task1 preflight: model, projector, CLIP tower, JSON, and image root all exist; Task1 has 23,998 records.

### Two-step GPU Dry-run

- Environment: server `ubuntu`, Conda env `hyper`, Torch 2.3.1+cu118, Transformers 4.33.3, one NVIDIA GPU, DeepSpeed ZeRO-2.
- Effective config: batch size 1, gradient accumulation 1, bf16, maximum length 1024, 2 steps, expert 0.
- Injected layers: 296.
- Loss: step 1 `2.3188`; step 2 `0.8084`; mean `1.5635839402675629`.
- Runtime: `2.0146s` training time after initialization; process exited successfully.
- Output: `/data/ckpt/zhaozhuofan/compose/smoke/task1_two_step_0729_v2`.
- Checkpoint: format version 1, 592 tensors, 42,545,074 bytes; 224 LoRA-B tensors were non-zero after two steps.

An earlier run with maximum length 512 exited successfully but produced zero loss because 576 image patch tokens pushed answer labels beyond the truncation boundary. It is retained at `/data/ckpt/zhaozhuofan/compose/smoke/task1_two_step_0729` as an invalid diagnostic run and is not counted as validation. The smoke script now uses 1024.

## Risks

- Full Task1 training and evaluation have not been run; only the requested bounded execution path is validated.
- Native Compose conversion of the base checkpoint is not implemented, so Transformers reports a source `llava` to target `compose_llava` model-type warning during load.
- Automatic router, expert keys, candidate retrieval, contribution teacher, oracle search, shadow update, anchor memory, dynamic expansion, and the full provisional/formal lifecycle remain out of scope.

## Next Step

Run one full Task1 epoch into a new `/data/ckpt/zhaozhuofan/compose/UCIT/Task1` output, evaluate the resulting expert, and define a separate ADR before adding any learned routing or Task2 composition policy.
