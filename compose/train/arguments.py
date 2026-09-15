from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="facebook/opt-125m")
    version: str = field(default="v1")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: int = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: str = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: str = field(default="patch")
    compose_rank: int = field(default=8)
    compose_alpha: float = field(default=16.0)
    compose_dropout: float = field(default=0.0)
    compose_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    compose_expert_ids: str = field(default="0")
    compose_trainable_expert_ids: str = field(default="")
    compose_expert_name: str = field(default="")
    compose_origin_task_id: Optional[str] = field(default=None)
    compose_expert_tags: str = field(default="")
    compose_existing_expert_origins: str = field(default="")
    compose_expert_seeds: str = field(default="")
    compose_gates: str = field(default="")
    compose_gate_normalization: str = field(default="none")
    compose_checkpoint: Optional[str] = field(default=None)
    compose_mode: str = field(default="fixed")
    compose_selection_manifest: Optional[str] = field(default=None)
    compose_cluster_expert_ids: str = field(default="")
    compose_v7_key_state: Optional[str] = field(default=None)
    compose_v7_query_cache: Optional[str] = field(default=None)
    compose_v7_config: Optional[str] = field(default=None)
    compose_v7_metrics_path: Optional[str] = field(default=None)
    compose_v7_task_index: int = field(default=0)
    compose_v7_require_full_coverage: bool = field(default=False)
    compose_v7_runtime_contract: Optional[str] = field(default=None)
    # Task-level output of the few-shot answer teacher.  It is consumed only
    # to restrict the historical routing candidates; full-data training never
    # reads teacher sample assignments, answers, or answer NLL.
    compose_v8_reusable_screening: Optional[str] = field(default=None)
    max_samples: Optional[int] = field(default=None)
    expected_adapter_parameters: Optional[int] = field(default=None)
    # --- V8 exact-accelerated execution flags (all default to the frozen
    # --- baseline behaviour; see docs/reports/V8_ACCELERATION_IMPLEMENTATION.md).
    profile_training: bool = field(default=False)
    profile_path: Optional[str] = field(default=None)
    profile_flush_every: int = field(default=25)
    profile_sync: bool = field(default=True)
    # S5: share one selection decomposition across all ComposeLinear layers per
    # micro-step instead of recomputing it (with a device sync) in each layer.
    compose_selection_plan: bool = field(default=False)
    compose_v7_query_tensor: Optional[str] = field(default=None)
    # S3 (query cache): attention kernel. Empty keeps transformers' own default.
    compose_attn_implementation: str = field(default="")
    # V8-Exact-Accelerated declaration; see configs/v8_exact_accelerated.yaml.
    compose_v8_config: Optional[str] = field(default=None)
    compose_v8_reuse_quality_enabled: bool = field(default=False)
    compose_v8_reuse_quality_temperature: float = field(default=1.0)
    compose_v8_reuse_quality_floor: float = field(default=0.10)
    # --- V9: Answer-Guided Key--Expert Co-Evolution (see compose/v9/). ---
    # The whole recipe lives in the V9 YAML; these flags only locate it and
    # name the artefacts of this task.  No V9 threshold is ever passed on the
    # command line, so a run cannot drift from the config it declares.
    compose_v9_config: Optional[str] = field(default=None)
    compose_v9_key_state: Optional[str] = field(default=None)
    compose_v9_query_cache: Optional[str] = field(default=None)
    compose_v9_retrieval_cache: Optional[str] = field(default=None)
    compose_v9_metrics_path: Optional[str] = field(default=None)
    compose_v9_task_index: int = field(default=0)
    # 0 = derive from the train dataloader the trainer actually builds.
    compose_v9_total_steps: int = field(default=0)
    compose_v9_runtime_contract: Optional[str] = field(default=None)
    #: Fail the task if any training sample never reached the answer loss.
    #: Off for deliberately compressed preflights, on for formal runs.
    compose_v9_require_full_coverage: bool = field(default=False)
    #: JSON manifest naming the held-out split the spec §30 calibration reads.
    compose_v9_calibration: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    memory_data_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = field(default=True)
    is_multimodal: bool = field(default=False)
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default="square")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=512)
    mm_projector_lr: Optional[float] = field(default=None)
    group_by_modality_length: bool = field(default=False)
    use_im_start_end: bool = field(default=False)
    #: V9 trains two parameter groups: the current candidates' LoRA (Path A of
    #: the answer gradient) and the routing keys plus bias (Path B).  A key is a
    #: direction in a 1536-D unit sphere, not a weight matrix, so it takes its
    #: own rate.  ``learning_rate`` remains the LoRA rate for every mode.
    v9_key_learning_rate: float = field(default=3e-4)
