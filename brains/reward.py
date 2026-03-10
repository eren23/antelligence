"""Sparse colony-level reward signal.

Replaces the hand-crafted shaping rewards in nn_brain.compute_reward() with a
minimal sparse signal that only fires on goal-relevant outcomes:
  - Food deposited at nest: +10.0
  - Food picked up:         +1.0
  - Death:                  -1.0
  - Otherwise:               0.0

This forces learning brains to discover their own behavioral strategies rather
than following hand-crafted gradient signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.actions import AntAction
    from agents.sensory import SensoryInput


class SparseReward:
    """Only rewards goal-relevant outcomes. No shaping."""

    def compute(
        self,
        prev: SensoryInput | None,
        curr: SensoryInput,
        action: AntAction,
        alive: bool,
    ) -> float:
        if not alive:
            return -1.0

        if prev is not None:
            # Food deposited at nest: was carrying food, now not
            if prev.carrying == "food" and curr.carrying is None:
                return 10.0

            # Picked up food: was empty, now carrying food
            if prev.carrying is None and curr.carrying == "food":
                return 1.0

        return 0.0
