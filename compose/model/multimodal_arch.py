from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


class ComposeVisionTower(nn.Module):
    """Clean adapter for the repository's CLIP-with-projection vision tower."""

    def __init__(self, vision_tower: str, args, delay_load: bool = False) -> None:
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        if delay_load:
            self.cfg_only = CLIPVisionConfig.from_pretrained(vision_tower)
        else:
            self.load_model()

    def load_model(self) -> None:
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, outputs) -> torch.Tensor:
        features = outputs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return features[:, 1:]
        if self.select_feature == "cls_patch":
            return features
        raise ValueError("unexpected vision select feature: {}".format(self.select_feature))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(images, list):
            return [self.forward(image.unsqueeze(0)) for image in images]
        outputs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        return self.feature_select(outputs).to(images.dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_tower.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size


def build_compose_vision_tower(config, delay_load: bool = False) -> ComposeVisionTower:
    path = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", None))
    if not path:
        raise ValueError("vision tower path is required")
    return ComposeVisionTower(path, config, delay_load=delay_load)


def build_compose_projector(config) -> nn.Module:
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type.startswith("mlp") and projector_type.endswith("x_gelu"):
        depth_text = projector_type[len("mlp") : -len("x_gelu")]
        if not depth_text.isdigit() or int(depth_text) < 1:
            raise ValueError("invalid projector type: {}".format(projector_type))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, int(depth_text)):
            modules.extend([nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)])
        return nn.Sequential(*modules)
    if projector_type == "identity":
        return nn.Identity()
    raise ValueError("unknown projector type: {}".format(projector_type))


class ComposeLlavaMetaModel:
    """Vision tower and projector only; no expert or continual-learning state."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_compose_vision_tower(config, delay_load=True)
            self.mm_projector = build_compose_projector(config)

    def get_vision_tower(self) -> Optional[ComposeVisionTower]:
        vision_tower = getattr(self, "vision_tower", None)
        if isinstance(vision_tower, list):
            return vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None) -> None:
        self.config.mm_vision_tower = model_args.vision_tower
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            vision_tower = build_compose_vision_tower(model_args)
            self.vision_tower = [vision_tower] if fsdp else vision_tower
        else:
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_compose_projector(self.config)
        else:
            self.mm_projector.requires_grad_(True)

        adapter_path = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        if adapter_path:
            weights = torch.load(adapter_path, map_location="cpu")
            projector_weights = {
                key.split("mm_projector.", 1)[1]: value
                for key, value in weights.items()
                if "mm_projector." in key
            }
            self.mm_projector.load_state_dict(projector_weights, strict=False)


class ComposeLlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def get_vision_tower(self) -> Optional[ComposeVisionTower]:
        return self.get_model().get_vision_tower()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.get_vision_tower()(images)
        projector = self.get_model().mm_projector
        # Match the model dtype explicitly. The vision tower may emit fp32
        # (e.g. CLIP output norm), and in multi-GPU DataParallel replica
        # threads the autocast context is lost (replica parameters are
        # detached tensors, not Parameters), so an fp32 x bf16 matmul would
        # raise. Casting to the model dtype here is a no-op under autocast
        # and does not depend on the projector's parameter storage.
        return projector(features.to(self.dtype)).to(self.device)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        labels: Optional[torch.LongTensor],
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
                and attention_mask is not None
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = attention_mask.sum(dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list) or images.ndim == 5:
            image_list = images if isinstance(images, list) else list(images)
            image_batches = [
                image.unsqueeze(0) if image.ndim == 3 else image for image in image_list
            ]
            concat_images = torch.cat(image_batches, dim=0)
            encoded = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in image_batches]
            image_features = [value.flatten(0, 1) for value in torch.split(encoded, split_sizes)]
        else:
            image_features = self.encode_images(images)

        original_labels = labels
        original_position_ids = position_ids
        original_attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        unpadded_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        unpadded_labels = [row[mask] for row, mask in zip(labels, attention_mask)]
        new_embeddings = []
        new_labels = []
        image_index = 0
        for batch_index, sample_ids in enumerate(unpadded_ids):
            image_count = int((sample_ids == IMAGE_TOKEN_INDEX).sum().item())
            sample_labels = unpadded_labels[batch_index]
            if image_count == 0:
                text_embeddings = self.get_model().embed_tokens(sample_ids)
                unused = image_features[image_index][0:0]
                new_embeddings.append(torch.cat([text_embeddings, unused], dim=0))
                new_labels.append(sample_labels)
                image_index += 1
                continue

            boundaries = [-1] + torch.where(sample_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            boundaries.append(sample_ids.shape[0])
            id_chunks = []
            label_chunks = []
            for index in range(len(boundaries) - 1):
                id_chunks.append(sample_ids[boundaries[index] + 1 : boundaries[index + 1]])
                label_chunks.append(sample_labels[boundaries[index] + 1 : boundaries[index + 1]])
            lengths = [chunk.shape[0] for chunk in id_chunks]
            flat_embeddings = self.get_model().embed_tokens(torch.cat(id_chunks))
            embedding_chunks = torch.split(flat_embeddings, lengths, dim=0)
            sample_embeddings = []
            sample_new_labels = []
            for index in range(image_count + 1):
                sample_embeddings.append(embedding_chunks[index])
                sample_new_labels.append(label_chunks[index])
                if index < image_count:
                    current_image = image_features[image_index]
                    image_index += 1
                    sample_embeddings.append(current_image)
                    sample_new_labels.append(
                        torch.full(
                            (current_image.shape[0],),
                            IGNORE_INDEX,
                            dtype=sample_labels.dtype,
                            device=sample_labels.device,
                        )
                    )
            new_embeddings.append(torch.cat(sample_embeddings))
            new_labels.append(torch.cat(sample_new_labels))

        max_model_length = getattr(self.config, "tokenizer_model_max_length", None)
        if max_model_length is not None:
            new_embeddings = [value[:max_model_length] for value in new_embeddings]
            new_labels = [value[:max_model_length] for value in new_labels]

        max_length = max(value.shape[0] for value in new_embeddings)
        batch_size = len(new_embeddings)
        padded_embeddings = []
        padded_labels = torch.full(
            (batch_size, max_length),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        new_attention_mask = torch.zeros(
            (batch_size, max_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        new_position_ids = torch.zeros(
            (batch_size, max_length), dtype=position_ids.dtype, device=position_ids.device
        )
        left_padding = getattr(self.config, "tokenizer_padding_side", "right") == "left"
        for index, (embedding, label_row) in enumerate(zip(new_embeddings, new_labels)):
            length = embedding.shape[0]
            padding = torch.zeros(
                (max_length - length, embedding.shape[1]),
                dtype=embedding.dtype,
                device=embedding.device,
            )
            padded_embeddings.append(
                torch.cat((padding, embedding), dim=0)
                if left_padding
                else torch.cat((embedding, padding), dim=0)
            )
            target_slice = slice(-length, None) if left_padding else slice(0, length)
            padded_labels[index, target_slice] = label_row
            new_attention_mask[index, target_slice] = True
            new_position_ids[index, target_slice] = torch.arange(
                length, dtype=position_ids.dtype, device=position_ids.device
            )

        return (
            None,
            None if original_position_ids is None else new_position_ids,
            None
            if original_attention_mask is None
            else new_attention_mask.to(dtype=original_attention_mask.dtype),
            past_key_values,
            torch.stack(padded_embeddings),
            None if original_labels is None else padded_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer) -> None:
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_start_end:
            count = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            if count:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data
                input_embeddings[-count:] = input_embeddings[:-count].mean(dim=0, keepdim=True)
                output_embeddings[-count:] = output_embeddings[:-count].mean(dim=0, keepdim=True)
