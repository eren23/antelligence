"""FoodSource dataclass with depletion and stochastic respawning."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agents.ant import Vec2


@dataclass
class FoodSource:
    """A depletable, respawnable food source in the world."""

    pos: Vec2
    radius: float
    amount: float
    max_amount: float
    depleted: bool = False
    _ticks_since_depleted: int = field(default=0, init=False, repr=False)

    def deplete(self, requested: float) -> float:
        """Remove food and return the amount actually taken.

        Returns 0 if the source is already depleted.
        """
        if self.depleted or self.amount <= 0:
            return 0.0
        taken = min(requested, self.amount)
        self.amount -= taken
        if self.amount <= 0:
            self.amount = 0.0
            self.depleted = True
            self._ticks_since_depleted = 0
        return taken

    def contains_point(self, point: Vec2) -> bool:
        """Check if *point* falls within this source's radius."""
        return point.distance_to(self.pos) <= self.radius

    def tick_respawn(
        self, respawn_interval: int, rng: random.Random | None = None,
    ) -> bool:
        """Advance respawn timer.  Returns ``True`` if food respawned.

        After *respawn_interval* ticks of depletion, each subsequent tick
        has an increasing probability of respawning the source.
        """
        if not self.depleted:
            return False

        self._ticks_since_depleted += 1

        if self._ticks_since_depleted < respawn_interval:
            return False

        if rng is None:
            rng = random.Random()

        overshoot = self._ticks_since_depleted - respawn_interval
        probability = min(1.0, 0.01 + overshoot * 0.002)

        if rng.random() < probability:
            self.amount = self.max_amount
            self.depleted = False
            self._ticks_since_depleted = 0
            return True

        return False
