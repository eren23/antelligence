"""Colony manager — spawning, role assignment, population dynamics, and stats."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from agents.ant import Ant, Role, Vec2
from config import SimConfig


@dataclass
class ColonyStats:
    """Snapshot of colony state for HUD / metrics."""

    population: int = 0
    dead_count: int = 0
    food_stored: float = 0.0
    food_deposited: float = 0.0
    role_counts: dict[str, int] = field(default_factory=dict)
    avg_energy: float = 0.0


@dataclass
class Corpse:
    """A dead ant's body left on the ground."""

    pos: Vec2
    age: int = 0  # ticks since death


class Colony:
    """Manages ant lifecycle, role assignment, and colony economy."""

    __slots__ = (
        "_cfg",
        "_rng",
        "_ants",
        "_corpses",
        "_food_stored",
        "_food_deposited_this_tick",
        "_next_id",
        "_tick_count",
        "_dead_count",
    )

    def __init__(self, cfg: SimConfig, nest_center: Vec2, seed: int | None = None) -> None:
        self._cfg = cfg
        self._rng = random.Random(seed)
        self._ants: list[Ant] = []
        self._corpses: list[Corpse] = []
        self._food_stored: float = cfg.colony.initial_food_stored
        self._food_deposited_this_tick: float = 0.0
        self._next_id: int = 0
        self._tick_count: int = 0
        self._dead_count: int = 0

        # Initial spawn
        for _ in range(cfg.colony.initial_population):
            self._spawn_ant(nest_center)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ants(self) -> list[Ant]:
        return self._ants

    @property
    def corpses(self) -> list[Corpse]:
        return self._corpses

    @property
    def food_stored(self) -> float:
        return self._food_stored

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _assign_role(self) -> Role:
        """Pick a role according to configured distribution."""
        dist = self._cfg.roles.default_distribution
        r = self._rng.random()
        cumulative = 0.0
        for role_name, prob in dist.items():
            cumulative += prob
            if r < cumulative:
                return Role(role_name)
        return Role.IDLE  # fallback

    def _spawn_ant(self, nest_center: Vec2) -> Ant:
        """Create a new ant at the nest with a random heading."""
        # Small random offset within nest radius
        angle = self._rng.uniform(0, 2 * math.pi)
        offset = self._rng.uniform(0, 20)
        pos = nest_center + Vec2.from_angle(angle, offset)

        ant = Ant(
            id=self._next_id,
            pos=pos,
            heading=self._rng.uniform(0, 2 * math.pi),
            speed=self._cfg.ant.base_speed,
            energy=self._cfg.ant.energy_max,
            role=self._assign_role(),
        )
        self._next_id += 1
        self._ants.append(ant)
        return ant

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, nest_center: Vec2) -> None:
        """Advance colony state by one tick.

        - Remove dead ants and create corpses.
        - Spawn new ants if food stored > threshold and under max population.
        - Rebalance roles at configured interval.
        """
        self._tick_count += 1

        # Remove dead ants, create corpses
        alive: list[Ant] = []
        for ant in self._ants:
            if ant.alive:
                ant.age += 1
                alive.append(ant)
            else:
                self._corpses.append(Corpse(pos=Vec2(ant.pos.x, ant.pos.y)))
                self._dead_count += 1
        self._ants = alive

        # Spawn new ants based on food stored and spawn rate
        alive_count = len(self._ants)
        if (
            self._food_stored > 50.0
            and alive_count < self._cfg.colony.max_population
        ):
            # spawn_rate is expected number of ants to spawn per tick
            spawn_chance = self._cfg.colony.spawn_rate
            # Fractional: accumulate probability
            while spawn_chance > 0 and alive_count < self._cfg.colony.max_population:
                if spawn_chance >= 1.0 or self._rng.random() < spawn_chance:
                    self._spawn_ant(nest_center)
                    self._food_stored -= 1.0  # each spawn costs 1 food
                    alive_count += 1
                spawn_chance -= 1.0

        # Age corpses
        for corpse in self._corpses:
            corpse.age += 1

        # Role rebalancing
        if self._tick_count % self._cfg.roles.rebalance_interval == 0:
            self._rebalance_roles()

    # ------------------------------------------------------------------
    # Food economy
    # ------------------------------------------------------------------

    def deposit_food(self, amount: float) -> None:
        """Add food to colony storage (called when ant deposits at nest)."""
        self._food_stored += amount
        try:
            self._food_deposited_this_tick += amount
        except AttributeError:
            pass  # loaded from older state without this slot

    # ------------------------------------------------------------------
    # Role rebalancing
    # ------------------------------------------------------------------

    def _rebalance_roles(self) -> None:
        """Reassign roles to match the configured distribution."""
        if not self._ants:
            return

        dist = self._cfg.roles.default_distribution
        n = len(self._ants)

        # Compute target counts
        targets: dict[str, int] = {}
        remaining = n
        role_names = list(dist.keys())
        for i, role_name in enumerate(role_names):
            if i == len(role_names) - 1:
                targets[role_name] = remaining
            else:
                count = round(dist[role_name] * n)
                targets[role_name] = count
                remaining -= count

        # Count current roles
        current: dict[str, list[Ant]] = {r: [] for r in dist}
        for ant in self._ants:
            key = ant.role.value
            if key in current:
                current[key].append(ant)
            else:
                current.setdefault("idle", []).append(ant)

        # Find surplus and deficit
        surplus: list[Ant] = []
        for role_name, ants_in_role in current.items():
            target = targets.get(role_name, 0)
            if len(ants_in_role) > target:
                excess = ants_in_role[target:]
                surplus.extend(excess)

        # Assign surplus ants to deficit roles
        for role_name in role_names:
            target = targets.get(role_name, 0)
            have = len(current.get(role_name, []))
            deficit = target - have
            while deficit > 0 and surplus:
                ant = surplus.pop()
                ant.role = Role(role_name)
                deficit -= 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> ColonyStats:
        """Compute current colony statistics."""
        role_counts: dict[str, int] = {}
        total_energy = 0.0
        for ant in self._ants:
            role_counts[ant.role.value] = role_counts.get(ant.role.value, 0) + 1
            total_energy += ant.energy

        n = len(self._ants)
        deposited = getattr(self, "_food_deposited_this_tick", 0.0)
        try:
            self._food_deposited_this_tick = 0.0
        except AttributeError:
            pass  # loaded from older state without this slot
        return ColonyStats(
            population=n,
            dead_count=self._dead_count,
            food_stored=self._food_stored,
            food_deposited=deposited,
            role_counts=role_counts,
            avg_energy=total_energy / n if n > 0 else 0.0,
        )

    def remove_corpse(self, corpse: Corpse) -> None:
        """Remove a corpse from the colony's corpse list."""
        if corpse in self._corpses:
            self._corpses.remove(corpse)
