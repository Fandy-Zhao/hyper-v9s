from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from compose.adapters.manager import ExpertManager
from compose.adapters.types import ComposeSelection

from .metadata import ExpertMetadata, ExpertStatus


class ExpertPool:
    """Registry and fixed-selection facade for Compose experts."""

    def __init__(self, manager: ExpertManager) -> None:
        self.manager = manager
        self._metadata = OrderedDict()  # type: OrderedDict[int, ExpertMetadata]

    def register(
        self,
        expert_id: int,
        name: Optional[str] = None,
        origin_task_id: Optional[str] = None,
        source_checkpoint: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ExpertMetadata:
        expert_id = int(expert_id)
        if expert_id in self._metadata:
            raise ValueError("expert {} is already registered".format(expert_id))
        self.manager.add_expert(expert_id)
        metadata = ExpertMetadata(
            expert_id=expert_id,
            name=name or "expert-{}".format(expert_id),
            origin_task_id=origin_task_id,
            source_checkpoint=source_checkpoint,
            tags=list(tags or []),
        )
        self._metadata[expert_id] = metadata
        return metadata

    def get(self, expert_id: int) -> ExpertMetadata:
        try:
            return self._metadata[int(expert_id)]
        except KeyError:
            raise KeyError("expert {} is not in the pool".format(expert_id))

    def expert_ids(self) -> List[int]:
        return list(self._metadata.keys())

    @property
    def trainable_expert_ids(self) -> List[int]:
        """Ids of the experts currently marked TRAINING by ``train_only``."""
        return [
            metadata.expert_id
            for metadata in self._metadata.values()
            if metadata.status is ExpertStatus.TRAINING
        ]

    def make_selection(
        self,
        expert_ids: Sequence[int],
        batch_size: int,
        gates: Optional[Sequence[float]] = None,
        device: Optional[torch.device] = None,
        normalization: str = "none",
    ) -> ComposeSelection:
        for expert_id in expert_ids:
            self.get(expert_id)
        return self.manager.make_selection(
            expert_ids, batch_size, gates, device, normalization
        )

    def train_only(self, expert_ids: Iterable[int]) -> None:
        selected = set(int(value) for value in expert_ids)
        for expert_id, metadata in self._metadata.items():
            metadata.status = (
                ExpertStatus.TRAINING if expert_id in selected else ExpertStatus.FROZEN
            )
        self.manager.train_only(selected)

    def mark_steps(self, expert_id: int, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.get(expert_id).trained_steps += int(steps)

    def sync_training_step(self, global_step: int) -> None:
        """Record Trainer progress without double-counting resumed checkpoints."""

        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        for metadata in self._metadata.values():
            if metadata.status == ExpertStatus.TRAINING:
                metadata.trained_steps = max(metadata.trained_steps, int(global_step))

    def to_dict(self) -> Dict[str, object]:
        return {
            "format_version": 1,
            "experts": [metadata.to_dict() for metadata in self._metadata.values()],
        }

    def restore_metadata(self, entries: Iterable[Dict[str, object]]) -> None:
        for entry in entries:
            metadata = ExpertMetadata.from_dict(entry)
            if metadata.expert_id not in self.manager.expert_ids():
                self.manager.add_expert(metadata.expert_id)
            self._metadata[metadata.expert_id] = metadata
