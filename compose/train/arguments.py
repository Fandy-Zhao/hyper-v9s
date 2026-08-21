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
    max_samples: Optional[int] = field(default=None)
    expected_adapter_parameters: Optional[int] = field(default=None)


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
