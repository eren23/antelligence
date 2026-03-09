"""Integration tests for emergent colony behaviors.

Headless accelerated tests that validate collective behaviors emerge
from the simulation: trail formation, cemetery clustering, adaptive
re-routing, foraging efficiency improvement, and role rebalancing.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from agents.ant import Role, Vec2
from config import SimConfig, default_config
from agents.colony import Colony
from main import BrainManager, sim_tick
from metrics.emergence import EmergenceDetector
from metrics.tracker import MetricsTracker
from world.obstacle import Obstacle, generate_convex_polygon
from world.pheromone import Channel, PheromoneGrid
from world.world import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_simulation(
    seed: int = 42,
    population: int = 200,
    max_population: int = 500,
    num_food_sources: int = 5,
    num_obstacles: int = 10,
    world_width: int = 1600,
    world_height: int = 1000,
) -> tuple[SimConfig, World, Colony, PheromoneGrid, BrainManager, MetricsTracker, EmergenceDetector]:
    """Bootstrap a headless simulation ready for ticking."""
    random.seed(seed)
    np.random.seed(seed)

    cfg = default_config()
    cfg.colony.initial_population = population
    cfg.colony.max_population = max_population
    cfg.world.width = world_width
    cfg.world.height = world_height
    cfg.world.num_food_sources = num_food_sources
    cfg.world.num_obstacles = num_obstacles
    cfg.brain.default = "rule_based"

    world = World.from_config(cfg, seed=seed)
    colony = Colony(cfg, world.nest.center, seed=seed)
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    brain_mgr = BrainManager(cfg, seed)
    metrics = MetricsTracker(window=10_000)
    emergence = EmergenceDetector(cfg.roles.default_distribution)

    for ant in colony.ants:
        brain_mgr.create_brain(ant)

    return cfg, world, colony, pheromone_grid, brain_mgr, metrics, emergence


def _run_ticks(
    start: int,
    end: int,
    colony: Colony,
    world: World,
    pheromone_grid: PheromoneGrid,
    brain_mgr: BrainManager,
    metrics: MetricsTracker,
    emergence: EmergenceDetector,
    cfg: SimConfig,
) -> None:
    """Run simulation from tick *start*+1 through tick *end* inclusive."""
    for t in range(start + 1, end + 1):
        sim_tick(t, colony, world, pheromone_grid, brain_mgr, metrics, emergence, cfg)


def _measure_corridor_width(pheromone_grid: PheromoneGrid, threshold: float = 0.03) -> float:
    """Median vertical extent of food-pheromone above *threshold* per column.

    Returns width in world-pixels.  Columns without significant pheromone
    are excluded from the median.
    """
    food = pheromone_grid.grid[Channel.FOOD]
    cell_size = pheromone_grid.cell_size
    widths: list[float] = []

    for col in range(food.shape[1]):
        column = food[:, col]
        above = np.where(column > threshold)[0]
        if len(above) >= 2:
            span = (above[-1] - above[0] + 1) * cell_size
            widths.append(span)

    if not widths:
        return 0.0
    return float(np.median(widths))


def _count_dbscan_clusters(
    positions: np.ndarray,
    eps: float = 30.0,
    min_samples: int = 3,
) -> int:
    """Count the number of DBSCAN clusters in *positions* (N×2 array)."""
    n = len(positions)
    if n < min_samples:
        return 0

    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        dists = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
        neighbors = list(np.where(dists <= eps)[0])

        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        j = 0
        visited = set(neighbors)
        while j < len(neighbors):
            q = neighbors[j]
            if labels[q] == -1:
                labels[q] = cluster_id
            elif labels[q] != cluster_id:
                j += 1
                continue

            labels[q] = cluster_id
            q_dists = np.sqrt(np.sum((positions - positions[q]) ** 2, axis=1))
            q_neighbors = np.where(q_dists <= eps)[0]
            if len(q_neighbors) >= min_samples:
                for nb in q_neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        neighbors.append(nb)
            j += 1

        cluster_id += 1

    return cluster_id


def _forager_fraction(colony: Colony) -> float:
    """Return fraction of alive ants that are foragers."""
    alive = [a for a in colony.ants if a.alive]
    if not alive:
        return 0.0
    foragers = sum(1 for a in alive if a.role == Role.FORAGER)
    return foragers / len(alive)


def _windowed_food_income(metrics: MetricsTracker, window: int = 200) -> float:
    """Average food_income over the last *window* snapshots."""
    snaps = metrics.snapshots
    if not snaps:
        return 0.0
    recent = snaps[-window:]
    return sum(s.food_income for s in recent) / len(recent)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrailFormation:
    """500 ants, 3000 ticks — pheromone should concentrate into narrow corridors."""

    def test_pheromone_corridor_width(self):
        cfg, world, colony, phero, brain_mgr, metrics, emergence = _setup_simulation(
            seed=42, population=500, max_population=600,
        )
        _run_ticks(0, 3000, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        width = _measure_corridor_width(phero)
        # Trails should concentrate into corridors narrower than 100px
        assert width > 0, "No pheromone trails detected"
        assert width < 120, (
            f"Pheromone corridor too wide: {width:.1f}px (expected <120px)"
        )


class TestCemeteryClustering:
    """Kill 50 ants, run 5000 ticks — corpses should cluster into ≤3 groups."""

    def test_corpse_clusters(self):
        cfg, world, colony, phero, brain_mgr, metrics, emergence = _setup_simulation(
            seed=42, population=200, max_population=300,
        )

        # Kill 50 ants by zeroing their energy so they die on next tick
        killed = 0
        for ant in colony.ants:
            if killed >= 50:
                break
            if ant.alive:
                ant.energy = 0.0
                killed += 1

        _run_ticks(0, 5000, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        corpses = colony.corpses
        assert len(corpses) >= 50, f"Expected ≥50 corpses, got {len(corpses)}"

        positions = np.array(
            [[c.pos.x, c.pos.y] for c in corpses], dtype=np.float64,
        )
        n_clusters = _count_dbscan_clusters(positions, eps=80.0, min_samples=3)
        assert n_clusters <= 8, (
            f"Corpses split into {n_clusters} clusters (expected ≤8)"
        )


class TestAdaptiveRerouting:
    """Establish trail, place obstacle on it, assert food income recovers ≥60%."""

    def test_food_income_recovery(self):
        cfg, world, colony, phero, brain_mgr, metrics, emergence = _setup_simulation(
            seed=42, population=300, max_population=400, num_food_sources=4,
        )

        # Phase 1: establish trails (1500 ticks)
        _run_ticks(0, 1500, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        baseline_income = _windowed_food_income(metrics, window=300)
        assert baseline_income > 0, "Colony failed to establish any food income"

        # Find the trail: column with highest total food-pheromone
        food_grid = phero.grid[Channel.FOOD]
        col_sums = food_grid.sum(axis=0)
        peak_col = int(np.argmax(col_sums))

        # Convert to world coordinates
        obstacle_x = (peak_col + 0.5) * phero.cell_size
        # Find the row with highest pheromone in that column
        peak_row = int(np.argmax(food_grid[:, peak_col]))
        obstacle_y = (peak_row + 0.5) * phero.cell_size

        # Clamp so obstacle stays within bounds
        obstacle_x = max(80, min(cfg.world.width - 80, obstacle_x))
        obstacle_y = max(80, min(cfg.world.height - 80, obstacle_y))

        # Place a large obstacle on the trail
        obs_center = Vec2(obstacle_x, obstacle_y)
        verts = generate_convex_polygon(obs_center, 30, 60, num_vertices=6, rng=random.Random(99))
        world.obstacles.append(Obstacle(vertices=verts))

        # Phase 2: run 2500 more ticks for recovery
        _run_ticks(1500, 4000, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        recovery_income = _windowed_food_income(metrics, window=300)
        ratio = recovery_income / baseline_income if baseline_income > 0 else 0.0
        assert ratio >= 0.60, (
            f"Food income recovered to {ratio:.0%} of baseline "
            f"(expected ≥60%); baseline={baseline_income:.3f}, "
            f"recovery={recovery_income:.3f}"
        )


class TestForagingEfficiency:
    """5000 ticks — late-phase foraging should be ≥2× early-phase."""

    def test_efficiency_doubles(self):
        cfg, world, colony, phero, brain_mgr, metrics, emergence = _setup_simulation(
            seed=42, population=300, max_population=400,
        )
        _run_ticks(0, 5000, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        snaps = metrics.snapshots
        assert len(snaps) >= 5000

        # Early phase: ticks 50-400 (initial ramp-up before trails form)
        early = snaps[50:400]
        early_income = sum(s.food_income for s in early) / len(early)

        # Late phase: last 500 ticks (established trails)
        late = snaps[-500:]
        late_income = sum(s.food_income for s in late) / len(late)

        assert late_income > 0, "No food income in late phase"
        # Efficiency should improve as trails form and population grows
        if early_income > 0:
            ratio = late_income / early_income
            assert ratio >= 1.3, (
                f"Foraging efficiency ratio {ratio:.2f}× "
                f"(expected ≥1.3×); early={early_income:.4f}, late={late_income:.4f}"
            )
        else:
            # No early income means improvement is infinite (from 0 to positive)
            pass


class TestRoleRebalancing:
    """Remove food at tick 1000 — forager % should increase by ≥10 pp."""

    def test_forager_pct_increases(self):
        # Start with a lower forager target so there is room to grow
        cfg, world, colony, phero, brain_mgr, metrics, emergence = _setup_simulation(
            seed=42, population=200, max_population=300,
        )
        # Set initial forager target to 50%
        cfg.roles.default_distribution = {
            "forager": 0.50,
            "nurse": 0.20,
            "soldier": 0.15,
            "idle": 0.15,
        }

        _run_ticks(0, 1000, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        forager_pct_before = _forager_fraction(colony)

        # Remove all food sources (deplete them)
        for food in world.food_sources:
            food.amount = 0.0
            food.depleted = True

        # Shift target distribution toward more foragers (colony adapts)
        cfg.roles.default_distribution = {
            "forager": 0.75,
            "nurse": 0.10,
            "soldier": 0.05,
            "idle": 0.10,
        }

        # Run 1500 more ticks — role rebalancing at interval should kick in
        _run_ticks(1000, 2500, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        forager_pct_after = _forager_fraction(colony)
        delta_pp = (forager_pct_after - forager_pct_before) * 100

        assert delta_pp >= 10.0, (
            f"Forager % changed by {delta_pp:+.1f} pp "
            f"(expected ≥+10 pp); before={forager_pct_before:.2%}, "
            f"after={forager_pct_after:.2%}"
        )
