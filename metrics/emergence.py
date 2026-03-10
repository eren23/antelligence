"""Emergent behavior detectors for the colony simulation.

Each detector analyses simulation state and returns a score in [0, 1]
indicating how strongly the emergent behavior is present.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.ant import Ant
    from agents.colony import Colony, Corpse
    from world.pheromone import PheromoneGrid
    from world.world import World


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EmergenceReport:
    """Scores for all emergent behaviors (each in [0, 1])."""

    trail_formation: float = 0.0
    cemetery_clustering: float = 0.0
    brood_sorting: float = 0.0
    foraging_efficiency: float = 0.0
    adaptive_rerouting: float = 0.0
    recruitment_cascade: float = 0.0
    role_rebalancing: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "trail_formation": self.trail_formation,
            "cemetery_clustering": self.cemetery_clustering,
            "brood_sorting": self.brood_sorting,
            "foraging_efficiency": self.foraging_efficiency,
            "adaptive_rerouting": self.adaptive_rerouting,
            "recruitment_cascade": self.recruitment_cascade,
            "role_rebalancing": self.role_rebalancing,
        }

    def as_vector(self) -> np.ndarray:
        """Return emergence scores as a 7-dim numpy vector for judge input."""
        return np.array([
            self.trail_formation,
            self.cemetery_clustering,
            self.brood_sorting,
            self.foraging_efficiency,
            self.adaptive_rerouting,
            self.recruitment_cascade,
            self.role_rebalancing,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Detector implementations
# ---------------------------------------------------------------------------

def detect_trail_formation(grid: PheromoneGrid) -> float:
    """Detect organized pheromone trails via corridor width analysis.

    Measures how concentrated the food pheromone is into narrow corridors
    vs. uniformly diffused. Uses the Gini coefficient of the pheromone
    distribution: high Gini = concentrated trails = strong formation.

    Returns:
        Score in [0, 1]. Higher = stronger trail formation.
    """
    from world.pheromone import Channel

    food = grid.grid[Channel.FOOD].ravel()
    total = food.sum()
    if total < 1e-9:
        return 0.0

    # Gini coefficient: measures inequality of distribution
    sorted_vals = np.sort(food)
    n = len(sorted_vals)
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = float((2.0 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)
    return max(0.0, min(1.0, gini))


def _dbscan_corpses(
    corpses: list[Corpse], eps: float = 30.0, min_samples: int = 3,
) -> tuple[np.ndarray, int]:
    """Run DBSCAN on corpse positions. Returns (labels, num_clusters)."""
    n = len(corpses)
    positions = np.array([[c.pos.x, c.pos.y] for c in corpses], dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        dists = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
        neighbors = np.where(dists <= eps)[0]

        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if labels[q] == -1 or labels[q] == cluster_id:
                labels[q] = cluster_id
                q_dists = np.sqrt(np.sum((positions - positions[q]) ** 2, axis=1))
                q_neighbors = np.where(q_dists <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)
            j += 1

        cluster_id += 1

    return labels, cluster_id


def detect_cemetery_clustering(corpses: list[Corpse]) -> float:
    """Detect cemetery clustering using density-based spatial analysis (DBSCAN-style).

    Real ants pile corpses into discrete clusters. This measures the
    fraction of corpses that belong to a dense cluster (eps=30, min_samples=3).

    Returns:
        Score in [0, 1]. Higher = stronger clustering.
    """
    if len(corpses) < 3:
        return 0.0
    labels, _ = _dbscan_corpses(corpses)
    clustered = int(np.sum(labels >= 0))
    return clustered / len(corpses)


def count_cemetery_clusters(corpses: list[Corpse]) -> int:
    """Count distinct cemetery clusters using DBSCAN (eps=30, min_samples=3)."""
    if len(corpses) < 3:
        return 0
    _, num_clusters = _dbscan_corpses(corpses)
    return num_clusters


def detect_brood_sorting(ants: list[Ant], nest_center: 'Vec2') -> float:
    """Detect brood sorting by measuring how close brood-carriers are to nest center.

    Real nurse ants sort brood by developmental stage, with younger brood
    closer to the center. We measure how concentrated brood-carrying ants
    are near the nest as a proxy.

    Returns:
        Score in [0, 1]. Higher = tighter brood sorting around nest.
    """
    from agents.ant import Vec2

    brood_carriers = [a for a in ants if a.carrying == "brood"]
    if not brood_carriers:
        # If no brood carriers, check nurse positions relative to nest
        from agents.ant import Role
        nurses = [a for a in ants if a.role == Role.NURSE and a.alive]
        if len(nurses) < 2:
            return 0.0
        dists = [a.pos.distance_to(nest_center) for a in nurses]
        mean_dist = sum(dists) / len(dists)
        # Score: nurses close to nest = higher score
        # Normalize: 0 at 200px+, 1 at 0px
        return max(0.0, min(1.0, 1.0 - mean_dist / 200.0))

    dists = [a.pos.distance_to(nest_center) for a in brood_carriers]
    mean_dist = sum(dists) / len(dists)
    return max(0.0, min(1.0, 1.0 - mean_dist / 200.0))


class ForagingEfficiencyDetector:
    """Tracks foraging efficiency as food/tick moving average.

    Maintains a sliding window of per-tick food income and computes
    a normalized efficiency score.
    """

    __slots__ = ("_window", "_income_history", "_max_observed")

    def __init__(self, window: int = 200) -> None:
        self._window = window
        self._income_history: deque[float] = deque(maxlen=window)
        self._max_observed: float = 1.0

    def update(self, food_income: float) -> float:
        """Record food income for this tick and return efficiency score.

        Returns:
            Score in [0, 1]. Higher = more efficient foraging.
        """
        self._income_history.append(food_income)
        if not self._income_history:
            return 0.0

        avg = sum(self._income_history) / len(self._income_history)
        if avg > self._max_observed:
            self._max_observed = avg

        if self._max_observed < 1e-9:
            return 0.0
        return min(1.0, avg / self._max_observed)


class AdaptiveReroutingDetector:
    """Detect adaptive re-routing after obstacle/food-source disruption.

    Monitors food income and detects recovery: a dip followed by
    return to baseline indicates the colony adapted its routes.
    """

    __slots__ = (
        "_baseline_window",
        "_income_history",
        "_baseline",
        "_dip_detected",
        "_dip_tick",
        "_recovery_score",
    )

    def __init__(self, baseline_window: int = 200) -> None:
        self._baseline_window = baseline_window
        self._income_history: deque[float] = deque(maxlen=baseline_window)
        self._baseline: float = 0.0
        self._dip_detected: bool = False
        self._dip_tick: int = 0
        self._recovery_score: float = 0.0

    def update(self, tick: int, food_income: float) -> float:
        """Record income and return re-routing score.

        Returns:
            Score in [0, 1]. Higher = stronger evidence of adaptive re-routing.
        """
        self._income_history.append(food_income)
        if len(self._income_history) < 20:
            return 0.0

        recent = list(self._income_history)
        current_avg = sum(recent[-20:]) / 20.0

        # Update baseline from older history
        if len(recent) >= self._baseline_window:
            self._baseline = sum(recent[:self._baseline_window // 2]) / (self._baseline_window // 2)
        elif len(recent) >= 40:
            half = len(recent) // 2
            self._baseline = sum(recent[:half]) / half

        if self._baseline < 1e-9:
            return 0.0

        ratio = current_avg / self._baseline

        if not self._dip_detected:
            # Detect dip: current drops below 30% of baseline
            if ratio < 0.3:
                self._dip_detected = True
                self._dip_tick = tick
                self._recovery_score = 0.0
        else:
            # After dip: measure recovery
            if ratio > 0.7:
                # Recovery achieved — score based on speed
                elapsed = tick - self._dip_tick
                # Faster recovery = higher score (normalize against 500 ticks)
                self._recovery_score = max(0.0, min(1.0, 1.0 - elapsed / 500.0))
                self._dip_detected = False
            elif tick - self._dip_tick > 1000:
                # Timeout — no recovery
                self._dip_detected = False
                self._recovery_score = 0.0

        return self._recovery_score


class RecruitmentCascadeDetector:
    """Detect recruitment cascades by measuring forager flux.

    A recruitment cascade occurs when one ant's food discovery triggers
    a wave of foragers heading to the same area. Measured by sudden
    increases in the number of foragers carrying food.
    """

    __slots__ = ("_carrying_history", "_window")

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._carrying_history: deque[int] = deque(maxlen=window)

    def update(self, ants: list[Ant]) -> float:
        """Record forager flux and return cascade score.

        Returns:
            Score in [0, 1]. Higher = stronger recruitment cascade.
        """
        from agents.ant import Role

        carrying_count = sum(
            1 for a in ants
            if a.role == Role.FORAGER and a.carrying == "food" and a.alive
        )
        self._carrying_history.append(carrying_count)

        if len(self._carrying_history) < 10:
            return 0.0

        history = list(self._carrying_history)
        # Compare recent flux to earlier baseline
        recent = sum(history[-10:]) / 10.0
        baseline = sum(history[:-10]) / max(1, len(history) - 10)

        if baseline < 0.5:
            # If baseline is near zero, any carrying is notable
            return min(1.0, recent / max(1.0, len(ants) * 0.1))

        # Score: how much recent flux exceeds baseline
        ratio = recent / baseline
        # ratio of 2.0+ maps to score 1.0
        return max(0.0, min(1.0, (ratio - 1.0)))


class RoleRebalancingDetector:
    """Detect role rebalancing correlation.

    Measures how well the current role distribution matches the target
    distribution from config. Also tracks temporal correlation: whether
    role shifts correlate with environmental changes.
    """

    __slots__ = ("_target_dist", "_score_history")

    def __init__(self, target_distribution: dict[str, float]) -> None:
        self._target_dist = target_distribution
        self._score_history: deque[float] = deque(maxlen=100)

    def update(self, role_counts: dict[str, int]) -> float:
        """Compute how well current distribution matches target.

        Returns:
            Score in [0, 1]. Higher = closer to target distribution.
        """
        total = sum(role_counts.values())
        if total == 0:
            return 0.0

        # Compute chi-squared-like similarity
        deviation = 0.0
        for role, target_frac in self._target_dist.items():
            actual_frac = role_counts.get(role, 0) / total
            deviation += abs(actual_frac - target_frac)

        # Max possible deviation is 2.0 (completely wrong distribution)
        # Score: 1.0 - normalized deviation
        score = max(0.0, 1.0 - deviation)
        self._score_history.append(score)
        return score


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class EmergenceDetector:
    """Aggregates all emergent behavior detectors.

    Call :meth:`update` once per tick to refresh all scores.
    """

    __slots__ = (
        "_foraging",
        "_rerouting",
        "_recruitment",
        "_rebalancing",
        "_report",
    )

    def __init__(self, target_role_distribution: dict[str, float]) -> None:
        self._foraging = ForagingEfficiencyDetector()
        self._rerouting = AdaptiveReroutingDetector()
        self._recruitment = RecruitmentCascadeDetector()
        self._rebalancing = RoleRebalancingDetector(target_role_distribution)
        self._report = EmergenceReport()

    @property
    def report(self) -> EmergenceReport:
        return self._report

    def update(
        self,
        tick: int,
        colony: Colony,
        world: World,
        pheromone_grid: PheromoneGrid,
        food_income: float,
    ) -> EmergenceReport:
        """Run all detectors and return updated report."""
        from agents.ant import Vec2

        stats = colony.stats()
        nest_center = world.nest.center

        self._report.trail_formation = detect_trail_formation(pheromone_grid)
        self._report.cemetery_clustering = detect_cemetery_clustering(colony.corpses)
        self._report.brood_sorting = detect_brood_sorting(colony.ants, nest_center)
        self._report.foraging_efficiency = self._foraging.update(food_income)
        self._report.adaptive_rerouting = self._rerouting.update(tick, food_income)
        self._report.recruitment_cascade = self._recruitment.update(colony.ants)
        self._report.role_rebalancing = self._rebalancing.update(stats.role_counts)

        return self._report
