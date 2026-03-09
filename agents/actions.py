"""AntAction dataclass — output of a brain's decide() call."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AntAction:
    turn: float = 0.0                       # delta heading in radians, clamped [-pi/6, pi/6]
    speed_mult: float = 1.0                 # 0.0–1.0 multiplier on base speed
    deposit_pheromone: Optional[str] = None  # channel name or None
    deposit_strength: float = 0.0           # 0.0–1.0
    pickup: bool = False                    # attempt to pick up item at current pos
    drop: bool = False                      # drop carried item
    recruit_signal: bool = False            # emit short-range recruit pulse

    def clamped(self) -> AntAction:
        """Return a copy with values clamped to valid ranges."""
        max_turn = math.pi / 6
        return AntAction(
            turn=max(-max_turn, min(max_turn, self.turn)),
            speed_mult=max(0.0, min(1.0, self.speed_mult)),
            deposit_pheromone=self.deposit_pheromone,
            deposit_strength=max(0.0, min(1.0, self.deposit_strength)),
            pickup=self.pickup,
            drop=self.drop,
            recruit_signal=self.recruit_signal,
        )
