"""MetricsTracker — per-tick time series recording for the colony simulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.ant import Ant, Role
    from agents.colony import Colony
    from world.pheromone import PheromoneGrid
    from world.world import World


@dataclass
class TickSnapshot:
    """Single-tick metric snapshot."""

    tick: int = 0
    food_income: float = 0.0          # food deposited this tick
    population: int = 0
    role_distribution: dict[str, int] = field(default_factory=dict)
    pheromone_mass: float = 0.0       # total pheromone across all channels
    trail_entropy: float = 0.0        # spatial entropy of food pheromone
    avg_foraging_trip: float = 0.0    # mean ticks per completed foraging round-trip
    deaths: int = 0                   # deaths this tick
    avg_reward: float = 0.0           # mean reward across all ants this tick


class MetricsTracker:
    """Records time series of simulation metrics.

    Call :meth:`record` once per tick after colony/world updates.
    All series are stored in-memory as lists; for long runs, use
    *window* to keep only the most recent *N* snapshots.
    """

    __slots__ = (
        "_snapshots",
        "_window",
        "_prev_food_stored",
        "_trip_starts",
        "_completed_trips",
        "_deaths_prev",
        "_reward_accumulator",
        "_reward_count",
    )

    def __init__(self, window: int = 0) -> None:
        """
        Args:
            window: If > 0, only keep the latest *window* snapshots (ring buffer).
                    0 means unlimited.
        """
        self._snapshots: deque[TickSnapshot] = (
            deque(maxlen=window) if window > 0 else deque()
        )
        self._prev_food_stored: float = 0.0
        # Foraging trip tracking: ant_id -> tick when it left the nest carrying nothing
        self._trip_starts: dict[int, int] = {}
        self._completed_trips: deque[float] = deque(maxlen=500)
        self._deaths_prev: int = 0
        self._reward_accumulator: float = 0.0
        self._reward_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        tick: int,
        colony: Colony,
        world: World,
        pheromone_grid: PheromoneGrid,
    ) -> TickSnapshot:
        """Sample all metrics for the current tick and store internally."""
        stats = colony.stats()

        # Food income = gross food deposited this tick (tracked by colony)
        food_income = stats.food_deposited

        # Deaths this tick
        deaths_now = stats.dead_count - self._deaths_prev
        self._deaths_prev = stats.dead_count

        # Pheromone mass (sum of entire grid)
        pheromone_mass = float(np.sum(pheromone_grid.grid))

        # Trail entropy (spatial entropy of food channel)
        trail_entropy = self._compute_trail_entropy(pheromone_grid)

        # Foraging trip tracking
        self._update_foraging_trips(tick, colony.ants, world)
        avg_trip = self._mean_trip_duration()

        # Average reward
        avg_reward = (
            self._reward_accumulator / self._reward_count
            if self._reward_count > 0
            else 0.0
        )
        self._reward_accumulator = 0.0
        self._reward_count = 0

        snap = TickSnapshot(
            tick=tick,
            food_income=food_income,
            population=stats.population,
            role_distribution=dict(stats.role_counts),
            pheromone_mass=pheromone_mass,
            trail_entropy=trail_entropy,
            avg_foraging_trip=avg_trip,
            deaths=deaths_now,
            avg_reward=avg_reward,
        )
        self._snapshots.append(snap)
        return snap

    def accumulate_reward(self, reward: float) -> None:
        """Call once per ant per tick to feed per-tick average reward."""
        self._reward_accumulator += reward
        self._reward_count += 1

    @property
    def snapshots(self) -> list[TickSnapshot]:
        return list(self._snapshots)

    @property
    def latest(self) -> TickSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def series(self, field_name: str) -> list[float]:
        """Extract a named scalar series (e.g. ``"food_income"``)."""
        return [getattr(s, field_name) for s in self._snapshots]

    def ticks(self) -> list[int]:
        return [s.tick for s in self._snapshots]

    def aggregate_window(self, n: int = 1000) -> np.ndarray:
        """Aggregate last *n* snapshots into a 5-dim feature vector.

        Returns:
            [total_food, avg_reward, avg_population, total_deaths, avg_pheromone_mass]
        """
        snaps = list(self._snapshots)[-n:] if self._snapshots else []
        if not snaps:
            return np.zeros(5, dtype=np.float64)

        total_food = sum(s.food_income for s in snaps)
        avg_reward = sum(s.avg_reward for s in snaps) / len(snaps)
        avg_pop = sum(s.population for s in snaps) / len(snaps)
        total_deaths = sum(s.deaths for s in snaps)
        avg_phero = sum(s.pheromone_mass for s in snaps) / len(snaps)

        return np.array([
            total_food, avg_reward, avg_pop, total_deaths, avg_phero,
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trail_entropy(grid: PheromoneGrid) -> float:
        """Shannon entropy of food-channel pheromone distribution.

        Treats each cell as a probability (normalized by total mass).
        High entropy = diffuse pheromone; low entropy = concentrated trails.
        """
        from world.pheromone import Channel

        food = grid.grid[Channel.FOOD].ravel()
        total = food.sum()
        if total < 1e-12:
            return 0.0
        p = food / total
        # Filter zeros to avoid log(0)
        mask = p > 0
        return float(-np.sum(p[mask] * np.log2(p[mask])))

    def _update_foraging_trips(
        self, tick: int, ants: list[Ant], world: World
    ) -> None:
        """Track round-trip foraging durations.

        A trip starts when a forager leaves the nest without food and ends
        when it returns carrying food.
        """
        from agents.ant import Role

        nest = world.nest
        for ant in ants:
            if ant.role != Role.FORAGER:
                # Clean up if role changed
                self._trip_starts.pop(ant.id, None)
                continue

            in_nest = nest.contains(ant.pos)

            if ant.id not in self._trip_starts:
                # Start a trip when a forager leaves the nest without food
                if not in_nest and ant.carrying is None:
                    self._trip_starts[ant.id] = tick
            else:
                # Trip completes when forager returns to nest carrying food
                if in_nest and ant.carrying == "food":
                    start = self._trip_starts.pop(ant.id)
                    duration = tick - start
                    if duration > 0:
                        self._completed_trips.append(float(duration))

        # Prune dead ants from trip tracking
        alive_ids = {a.id for a in ants}
        dead_ids = [aid for aid in self._trip_starts if aid not in alive_ids]
        for aid in dead_ids:
            del self._trip_starts[aid]

    def _mean_trip_duration(self) -> float:
        if not self._completed_trips:
            return 0.0
        return sum(self._completed_trips) / len(self._completed_trips)
