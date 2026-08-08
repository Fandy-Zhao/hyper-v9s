# Compose ↔ Hyper-LLaVA UCIT Evaluation Compatibility Contract

- **日期**：2026-08-08
- **目的**：正式六任务 UCIT seed42 实验与 Hyper-LLaVA 论文结果严格直接比较的逐项审计（spec §4）。
- **审计依据**：仓库内原始 Hyper-LLaVA 代码（`scripts/Hyper/`、`llava/eval/`），非论文描述推断。

---

## A. Backbone

| 组件 | Hyper-LLaVA | Compose | 一致 |
|---|---|---|---|
| base MLLM | `llava-v1.5-7b`（`/data/ckpt/zhaozhuofan/models/llava-v1.5-7b`） | 同路径 | ✅ 同一权重目录 |
| vision encoder | `clip-vit-large-patch14-336`（同一路径） | 同路径（CLIP 模型 + processor 均从此加载） | ✅ |
| tokenizer | llava-v1.5-7b tokenizer（`llava.model.builder.load_pretrained_model`） | 同一 tokenizer（`AutoTokenizer`，同路径） | ✅ |
| image processor | llava-v1.5-7b `image_processor`（`process_images` 同一函数） | 同一 processor / `process_images` | ✅ |
| 加载精度 | fp16（`llava/model/builder.py:42 torch_dtype=torch.float16`） | bf16（`compose/eval/load_compose.py`） | ⚠️ 见注 1 |

> **注 1（精度差异）**：两者加载同一份 llava-v1.5-7b 权重，Hyper-LLaVA 以 fp16、Compose 以 bf16 加载（Compose LoRA 适配器管线自 pre-formal 验证起即锁定 bf16）。生成协议完全一致（见 E）；greedy 解码下个别边界 token 可能因算术精度不同而不同，但这属于实现层差异而非协议差异，不影响可比较性分级（§30 其余条款全部满足）。

---

## B. UCIT Task Order

Hyper-LLaVA 训练脚本 `scripts/Hyper/Train_UCIT/Task{1..6}.sh` 与 Compose `configs/compose_ucit.yaml` 逐任务核对：

| Compose task_id | Hyper task_id | 数据集 | 一致 |
|---|---|---|---|
| 0 | 1 | ImageNet-R | ✅ |
| 1 | 2 | ArxivQA | ✅ |
| 2 | 3 | VizWiz | ✅ |
| 3 | 4 | IconQA | ✅ |
| 4 | 5 | CLEVR | ✅ |
| 5 | 6 | Flickr30k | ✅ |

顺序不可改变；两边的 continual 语义均按此顺序。

---

## C. Dataset Splits

### 测试集（正式评测唯一来源，两边逐字节相同）

| 任务 | 文件 | SHA256 | 样本数 |
|---|---|---|---|
| ImageNet-R | `instructions/ImageNet-R/test_3000.json` | `bfc603ab258d11186e776524e710764a92c05e2779ba60c3f970cccc3ec342ac` | 3000 |
| ArxivQA | `instructions/ArxivQA/test_3000.json` | `6a4da4dd77a2d339a41aa02c3b7a2db5292f9e2e3cbf9a1f23e3db9fb8d102c2` | 3000 |
| VizWiz | `instructions/VizWiz/test_3000.json` | `d7db5d9b0e298c24180f9a2b81b4c7ca76d8e935f2bbd6f37868040517ab7bfd` | 3000 |
| IconQA | `instructions/IconQA/test_3000.json` | `f4ef8b7735606fe5ff61623213945da7ecb95f203aff73a238957e66eac74896` | 3000 |
| CLEVR | `instructions/CLEVR/test_3000.json` | `463c53f3dc4d6297069b3396cce5609f6dd90173b0a979b9f181fbad4854e41d` | 3000 |
| Flickr30k | `instructions/Flickr30k/test_3000.json` | `e0a244e7456f761404daa9743321e020343e3e752f6e3c699eff051c4ed4ffec` | 3000 |

caption 任务额外使用 COCO 型标注（Hyper 原 `eval_caption` 的 `--annotation-file`）：

| 任务 | 文件 | SHA256 |
|---|---|---|
| VizWiz | `instructions/VizWiz/val_coco_type_3000.json` | `0597fe8b5d29a7815f77a72d096edb0839accc6548c99a5ab8d0a27a3e402191` |
| Flickr30k | `instructions/Flickr30k/val_coco_type_3000.json` | `2e4c16a16b788e73243ee94755cfd793f79f0b6f5022f8555a04ad73b3ce2dc7` |

结论：**Hyper-LLaVA 正式 UCIT evaluator 使用的就是这 3000 个测试样本**（`scripts/Hyper/Eval_UCIT/eval_*.sh` 的 `--question-file` 逐项核对）。无重新采样、无 quick-chain slice、无新 shuffle。

### 训练集（方法差异，见注 2）

| 任务 | 文件（两边同一） |
|---|---|
| ImageNet-R | `instructions/ImageNet-R/train.json` |
| ArxivQA | `instructions/ArxivQA/train_4w.json` |
| VizWiz | `instructions/VizWiz/train.json` |
| IconQA | `instructions/IconQA/train.json` |
| CLEVR | `instructions/CLEVR/train_4w.json` |
| Flickr30k | `instructions/Flickr30k/train_brief_4w.json` |

> **注 2（训练数据）**：Hyper-LLaVA 在完整训练 split 上训练；Compose 在**同一训练 split** 上做 teacher search（前 2000 样本）并将 residual 子集用于 LoRA 训练（方法自身的 residual selection，spec §10 "Method-specific: Residual selection" 允许）。Compose 不使用测试集、不扩大数据、不使用额外标注。

### 图像

两边均使用 `--image-folder /data/dataset/zhaozhuofan/UCIT/datasets`（同一路径），`Image.open(...).convert("RGB")` 同一预处理入口。

---

## D. Instruction / Conversation Template

- Hyper-LLaVA `llava/eval/model_answer.py`：`qs = line["text"]`；`mm_use_im_start_end` 为假时 `qs = DEFAULT_IMAGE_TOKEN + '\n' + qs`；`conv_templates['vicuna_v1']`，`append_message(roles[0], qs)` + `append_message(roles[1], None)`；`conv.get_prompt()`。
- Compose `compose/eval/eval_task.py::_prompt`：`question = question_text(record)`（UCIT 记录无 `conversations` 字段时即 `record["text"]`）；`DEFAULT_IMAGE_TOKEN` 缺失时才前置 `"<image>\n"`（UCIT 指令文件本身不含 `<image>`，因此行为与 Hyper 无条件前置**逐字符一致**）；同一 `vicuna_v1` 模板、同一 roles 结构、同一 `get_prompt()`。

结论：prompt 构造调用的是**同一套 llava.conversation 代码**，模板逐字节一致。

---

## E. Generation

| 字段 | Hyper-LLaVA（model_answer.py + eval_*.sh） | Compose（eval_task.py） | equal |
|---|---|---|---|
| do_sample | `temperature 0 → False` | `False`（固定） | ✅ |
| temperature | `0` | 不传（greedy 下无效） | ✅ |
| top_p | `None`（默认 1.0，greedy 下无效） | 不传（默认） | ✅ |
| num_beams | `1` | `1` | ✅ |
| max_new_tokens | `128` | `128`（config `eval.max_new_tokens`） | ✅ |
| use_cache | `True` | `True` | ✅ |
| batch size | 1（`batch_size == 1` 断言） | 1（逐样本） | ✅ |
| eos / stop | 默认；`skip_special_tokens=True` + `.strip()` | 同 | ✅ |
| conv_mode | `vicuna_v1` | `vicuna_v1` | ✅ |
| image dtype | fp16 | bf16（模型同为 bf16，见注 1） | ⚠️ 注 1 |
| seed | 无显式（greedy 确定性） | `torch.manual_seed(42)` | ✅ 确定性 |

正式运行前输出 `generation_config_comparison.json`（run root `metadata/`），所有用于比较的字段 `equal: true`。

---

## F. Prediction Post-processing

- 生成端：`tokenizer.batch_decode(..., skip_special_tokens=True)[0].strip()` —— 两边相同。
- 打分端：原 evaluator 不做额外 normalization，比较即为 `pred.upper() == ground_truth.upper()`（eval_deepseek_r1 `eval_single`）。Compose 不复制、不修改该语义 —— 直接执行原模块。

---

## G. Task Metric（same implementation = true，全部）

| 任务 | Hyper-LLaVA scorer | Compose 调用的 scorer | same implementation |
|---|---|---|---|
| ImageNet-R | `llava.eval.eval_deepseek_r1`（exact match, case-insensitive） | 同一模块，subprocess 原样执行 | ✅ |
| ArxivQA | 同 | 同 | ✅ |
| VizWiz | `llava.eval.eval_caption`（COCOEvalCap：Bleu_1-4, METEOR, ROUGE_L, CIDEr → Average） | 同一模块，subprocess 原样执行 | ✅ |
| IconQA | `llava.eval.eval_deepseek_r1` | 同 | ✅ |
| CLEVR | `llava.eval.eval_deepseek_r1` | 同 | ✅ |
| Flickr30k | `llava.eval.eval_caption` | 同 | ✅ |

> 说明：`eval_deepseek_r1` / `eval_caption` 中的 LLM API 打分分支（deepseek/gpt-4o）在原评估流程中已注释禁用（`eval_single` / `eval_caption` 的 `eval_single` 是实际生效路径），Compose 同样只使用 `eval_single` 路径。`Result.text` 由原模块原样写出（Accuracy / Average 行），Compose 只解析该行（`formal_ucit_eval._score_answers`）。

---

## Continual Metrics（MFN / MAA / MFT / BWT）

- 权威实现：**原仓库** `scripts/Hyper/Eval_UCIT/summarize_continual_metrics.py`，直接在 Compose 镜像的 `evaluation/hyper_result_root/{Dataset}/hyper-task{t+1}/Result.text` 布局上运行（数据集名与 1-based task id 与原脚本 `TASKS` 逐项一致，含 `CLEVR-Math`）。
- Compose wrapper（`formal_ucit_summary.wrapper_metrics`）按同一公式从矩阵重算；两结果 diff < 1e-8（回归测试 `test_ucit_evaluator_parity.py::test_continual_metrics_self_consistency`）。
- 语义（原脚本注释）：MFT=diagonal 均值（plasticity）；MFN=最终行均值；MAA=各阶段已学任务均值；BWT=旧任务最终-刚学完之差均值。

---

## 验证状态

- 6/6 任务 evaluator parity：同一 prediction 文件经原 scorer 与 Compose wrapper 得分 |A−B| < 1e-6 —— `tests/compose/test_ucit_evaluator_parity.py` 8/8 PASS（2026-08-08）。
- 最终表 per-task 列取 A[5][j]（最终行）而非 MFT 对角线 —— 回归测试覆盖（spec §29）。

## COMPARABILITY_STATUS（正式运行前判定）

`DIRECTLY_COMPARABLE_TO_HYPER_LLAVA`（前提：正式运行按本契约执行；唯一实现层差异为模型加载精度 fp16 vs bf16，已在注 1 记录，不影响协议可比性；训练数据为同一训练 split 的方法内子集，见注 2）。
