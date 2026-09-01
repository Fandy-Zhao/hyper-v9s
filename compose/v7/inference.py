"""Task-agnostic inference routing over committed experts only."""

from typing import Iterable

from torch import Tensor, nn

from .pool import V7ExpertKeyPool
from .query import FixedMultimodalQuery
from .routing import GlobalTop2Result, GlobalTop2Router


class V7InferenceRouter(nn.Module):
    def __init__(self, key_pool: V7ExpertKeyPool) -> None:
        super().__init__()
        if key_pool.current_ids:
            raise ValueError("inference checkpoints cannot contain current candidates")
        if any(
            key_pool.metadata[expert_id]["lifecycle"] != "historical"
            for expert_id in key_pool.selectable_ids()
        ):
            raise ValueError("inference selectable pool must be fully committed")
        key_pool.freeze_all()
        self.query = FixedMultimodalQuery()
        self.router = GlobalTop2Router(key_pool)

    def forward(self, z_visual: Tensor, z_text: Tensor) -> GlobalTop2Result:
        # No task id, answer, oracle, clustering, router MLP or batch statistic.
        return self.router(self.query(z_visual, z_text))

    @staticmethod
    def cross_task_pairs(result: GlobalTop2Result, pool: V7ExpertKeyPool):
        rows = []
        for pair in result.expert_ids.detach().cpu().tolist():
            tasks = tuple(int(pool.metadata[int(value)]["origin_task"]) for value in pair)
            if tasks[0] != tasks[1]:
                rows.append({"expert_ids": tuple(pair), "origin_tasks": tasks})
        return rows

