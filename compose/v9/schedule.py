"""Three-stage training schedule (spec §16).

``bootstrap`` -- every candidate is held above an exposure floor so a
zero-output expert can accumulate the answer gradient that gives it capability.

``soft`` -- the co-evolution stage proper: Top-C recall, differentiable gates,
one answer forward, gate gradient, responsibility, key update.

``hard`` -- discretisation.  The forward pass uses the deployed Top-2 selection
while backward keeps the soft surrogate, so what trains and what ships are the
same routing decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import V9RoutingConfig, V9ScheduleConfig


STAGE_BOOTSTRAP = "bootstrap"
STAGE_SOFT = "soft"
STAGE_HARD = "hard"
STAGES = (STAGE_BOOTSTRAP, STAGE_SOFT, STAGE_HARD)


@dataclass(frozen=True)
class V9StageState:
    stage: str
    temperature: float
    global_step: int
    total_steps: int
    stage_start: int
    stage_end: int

    @property
    def stage_progress(self) -> float:
        span = max(self.stage_end - self.stage_start, 1)
        return min(max((self.global_step - self.stage_start) / span, 0.0), 1.0)

    @property
    def progress(self) -> float:
        return min(max(self.global_step / max(self.total_steps, 1), 0.0), 1.0)

    @property
    def is_bootstrap(self) -> bool:
        return self.stage == STAGE_BOOTSTRAP

    @property
    def is_hard(self) -> bool:
        return self.stage == STAGE_HARD


def _ramp(start: float, end: float, fraction: float) -> float:
    return float(start) + (float(end) - float(start)) * min(max(fraction, 0.0), 1.0)


class V9StageScheduler:
    """Maps an optimizer step to a stage and a routing temperature.

    Step counts are derived from the trainer's own ``max_steps``, so the ratios
    stay correct when gradient accumulation, world size or the dataloader length
    changes -- the schedule is a fraction of the run, not a fixed step list.
    """

    def __init__(
        self,
        schedule: V9ScheduleConfig,
        routing: V9RoutingConfig,
        total_steps: int,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.schedule = schedule
        self.routing = routing
        self.total_steps = int(total_steps)
        self.bootstrap_end = self._boundary(schedule.bootstrap_ratio)
        self.soft_end = self._boundary(schedule.bootstrap_ratio + schedule.soft_ratio)
        if self.soft_end < self.bootstrap_end:
            raise ValueError("stage boundaries are not monotonic")

    def _boundary(self, ratio: float) -> int:
        return int(round(float(ratio) * self.total_steps))

    def state(self, global_step: int) -> V9StageState:
        step = int(global_step)
        if step < self.bootstrap_end:
            return V9StageState(
                stage=STAGE_BOOTSTRAP,
                temperature=float(self.routing.temperature_start),
                global_step=step,
                total_steps=self.total_steps,
                stage_start=0,
                stage_end=self.bootstrap_end,
            )
        if step < self.soft_end:
            span = max(self.soft_end - self.bootstrap_end, 1)
            fraction = (step - self.bootstrap_end) / span
            return V9StageState(
                stage=STAGE_SOFT,
                temperature=_ramp(
                    self.routing.temperature_start,
                    self.routing.temperature_mid,
                    fraction,
                ),
                global_step=step,
                total_steps=self.total_steps,
                stage_start=self.bootstrap_end,
                stage_end=self.soft_end,
            )
        span = max(self.total_steps - self.soft_end, 1)
        fraction = (step - self.soft_end) / span
        return V9StageState(
            stage=STAGE_HARD,
            temperature=_ramp(
                self.routing.temperature_mid,
                self.routing.temperature_end,
                fraction,
            ),
            global_step=step,
            total_steps=self.total_steps,
            stage_start=self.soft_end,
            stage_end=self.total_steps,
        )

    def describe(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "bootstrap_steps": self.bootstrap_end,
            "soft_steps": self.soft_end - self.bootstrap_end,
            "hard_steps": self.total_steps - self.soft_end,
            "temperature_start": self.routing.temperature_start,
            "temperature_mid": self.routing.temperature_mid,
            "temperature_end": self.routing.temperature_end,
        }


__all__ = [
    "STAGES",
    "STAGE_BOOTSTRAP",
    "STAGE_HARD",
    "STAGE_SOFT",
    "V9StageScheduler",
    "V9StageState",
]
