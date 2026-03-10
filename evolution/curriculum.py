"""Auto-curriculum for scaling environment difficulty.

Automatically adjusts environment parameters based on population performance:
- Stage 0 (easy): more food sources, larger amounts, fewer obstacles
- Stage 1 (medium): default parameters
- Stage 2 (hard): scarce food, more obstacles, faster pheromone decay

Advances when foraging efficiency exceeds a threshold, retreats when it
drops below a lower threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import SimConfig


@dataclass
class CurriculumStage:
    """Parameters for a curriculum difficulty stage."""
    name: str
    num_food_sources: int
    food_amount_min: int
    food_amount_max: int
    num_obstacles: int
    food_respawn_interval: int
    advance_threshold: float    # foraging efficiency to advance
    retreat_threshold: float    # foraging efficiency to retreat


_STAGES = [
    CurriculumStage(
        name="easy",
        num_food_sources=8,
        food_amount_min=100,
        food_amount_max=800,
        num_obstacles=5,
        food_respawn_interval=2000,
        advance_threshold=0.4,
        retreat_threshold=0.0,  # can't retreat from stage 0
    ),
    CurriculumStage(
        name="medium",
        num_food_sources=5,
        food_amount_min=50,
        food_amount_max=500,
        num_obstacles=10,
        food_respawn_interval=3000,
        advance_threshold=0.6,
        retreat_threshold=0.2,
    ),
    CurriculumStage(
        name="hard",
        num_food_sources=3,
        food_amount_min=20,
        food_amount_max=200,
        num_obstacles=15,
        food_respawn_interval=5000,
        advance_threshold=1.0,  # can't advance beyond final stage
        retreat_threshold=0.3,
    ),
]


class AutoCurriculum:
    """Scales environment difficulty based on population performance."""

    def __init__(self, initial_stage: int = 0) -> None:
        self._stage_idx = max(0, min(initial_stage, len(_STAGES) - 1))
        self._efficiency_history: list[float] = []
        self._window: int = 5  # number of generations to average over

    @property
    def stage(self) -> CurriculumStage:
        return _STAGES[self._stage_idx]

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def stage_name(self) -> str:
        return self.stage.name

    def update(self, efficiency: float) -> bool:
        """Record efficiency and check for stage transitions.

        Args:
            efficiency: foraging efficiency score [0, 1]

        Returns:
            True if stage changed
        """
        self._efficiency_history.append(efficiency)

        if len(self._efficiency_history) < self._window:
            return False

        recent = self._efficiency_history[-self._window:]
        avg = sum(recent) / len(recent)

        stage = _STAGES[self._stage_idx]

        # Try to advance
        if avg >= stage.advance_threshold and self._stage_idx < len(_STAGES) - 1:
            self._stage_idx += 1
            self._efficiency_history.clear()
            return True

        # Try to retreat
        if avg < stage.retreat_threshold and self._stage_idx > 0:
            self._stage_idx -= 1
            self._efficiency_history.clear()
            return True

        return False

    def apply_to_config(self, cfg: SimConfig) -> SimConfig:
        """Apply current curriculum stage to config (modifies in place and returns)."""
        stage = self.stage
        cfg.world.num_food_sources = stage.num_food_sources
        cfg.world.food_source_amount_range = [stage.food_amount_min, stage.food_amount_max]
        cfg.world.num_obstacles = stage.num_obstacles
        cfg.world.food_respawn_interval = stage.food_respawn_interval
        return cfg
