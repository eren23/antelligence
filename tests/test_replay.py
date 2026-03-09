"""Deterministic replay test.

Validates that:
1. Running the simulation twice from the same seed to tick N+500
   produces identical ant positions (within 0.01px), proving
   deterministic replay from the same initial state.
2. Saving state at tick N and reloading preserves positions, energy,
   food stored, and pheromone grid exactly.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from config import default_config
from agents.colony import Colony
from main import BrainManager, save_state, load_state, sim_tick
from metrics.emergence import EmergenceDetector
from metrics.tracker import MetricsTracker
from rendering.controls import ControlState
from world.pheromone import PheromoneGrid
from world.world import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap(seed: int):
    """Create a fresh simulation with deterministic seeding."""
    random.seed(seed)
    np.random.seed(seed)

    cfg = default_config()
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


def _get_positions(colony: Colony) -> dict[int, tuple[float, float]]:
    """Return ``{ant_id: (x, y)}`` for all alive ants."""
    return {
        ant.id: (ant.pos.x, ant.pos.y)
        for ant in colony.ants
        if ant.alive
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeterministicReplay:
    """Save at tick N, run to N+500, reload at N with same seed, re-run — positions match."""

    SEED = 42
    CHECKPOINT_TICK = 100
    REPLAY_TICKS = 500

    def test_replay_positions_match(self, tmp_path: Path):
        """Save at N, run to N+500; reload at N, re-run to N+500 — positions match within 0.01px."""
        N = self.CHECKPOINT_TICK
        STEPS = self.REPLAY_TICKS
        seed = self.SEED

        # =============================================================
        # Run A: seed → tick 0 → N (save) → N+STEPS (reference run)
        # =============================================================
        cfg, world, colony, phero, brain_mgr, metrics, emergence = _bootstrap(seed)

        for t in range(1, N + 1):
            sim_tick(t, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        # Save state at tick N
        save_path = tmp_path / "checkpoint.pkl"
        control = ControlState(brain_type=brain_mgr.current_type)
        save_state(save_path, N, seed, brain_mgr.current_type, colony, world, phero, control)

        # Continue to N+STEPS (reference positions)
        for t in range(N + 1, N + STEPS + 1):
            sim_tick(t, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        positions_a = _get_positions(colony)

        # =============================================================
        # Run B: reload from checkpoint at tick N, re-run to N+STEPS
        # All RNG states are restored by load_state, so the trajectory
        # must be identical to Run A from tick N onward.
        # =============================================================
        brain_mgr_b = BrainManager(cfg, seed)
        tick_b, colony_b, world_b, phero_b, control_b = load_state(
            save_path, cfg, brain_mgr_b,
        )
        assert tick_b == N

        metrics_b = MetricsTracker(window=10_000)
        emergence_b = EmergenceDetector(cfg.roles.default_distribution)

        for t in range(N + 1, N + STEPS + 1):
            sim_tick(t, colony_b, world_b, phero_b, brain_mgr_b, metrics_b, emergence_b, cfg)

        positions_b = _get_positions(colony_b)

        # =============================================================
        # Assertions: positions must match within 0.01px
        # =============================================================
        assert set(positions_a.keys()) == set(positions_b.keys()), (
            f"Alive ant sets differ: "
            f"run_a={len(positions_a)}, run_b={len(positions_b)}"
        )

        for ant_id in positions_a:
            xa, ya = positions_a[ant_id]
            xb, yb = positions_b[ant_id]
            err = max(abs(xa - xb), abs(ya - yb))
            assert err < 0.01, (
                f"Ant {ant_id} position mismatch: "
                f"run_a=({xa:.6f}, {ya:.6f}) vs "
                f"run_b=({xb:.6f}, {yb:.6f}), "
                f"delta={err:.6f}px"
            )

    def test_loaded_state_matches_checkpoint(self, tmp_path: Path):
        """Verify save/load round-trip preserves positions, energy, and pheromone."""
        seed = self.SEED
        N = self.CHECKPOINT_TICK

        cfg, world, colony, phero, brain_mgr, metrics, emergence = _bootstrap(seed)

        for t in range(1, N + 1):
            sim_tick(t, colony, world, phero, brain_mgr, metrics, emergence, cfg)

        # Snapshot state before save
        pre_positions = _get_positions(colony)
        pre_energy = {a.id: a.energy for a in colony.ants if a.alive}
        pre_food = colony.food_stored

        # Save and reload
        save_path = tmp_path / "fidelity.pkl"
        control = ControlState(brain_type=brain_mgr.current_type)
        save_state(save_path, N, seed, brain_mgr.current_type, colony, world, phero, control)

        brain_mgr_b = BrainManager(cfg, seed)
        tick_b, colony_b, world_b, phero_b, control_b = load_state(
            save_path, cfg, brain_mgr_b,
        )

        post_positions = _get_positions(colony_b)
        post_energy = {a.id: a.energy for a in colony_b.ants if a.alive}

        # Positions must be identical (no simulation step between save/load)
        for ant_id in pre_positions:
            assert ant_id in post_positions, f"Ant {ant_id} missing after load"
            xp, yp = pre_positions[ant_id]
            xl, yl = post_positions[ant_id]
            assert abs(xp - xl) < 1e-9 and abs(yp - yl) < 1e-9, (
                f"Ant {ant_id} position shifted on load: "
                f"({xp}, {yp}) -> ({xl}, {yl})"
            )

        # Energy must match
        for ant_id in pre_energy:
            assert ant_id in post_energy
            assert abs(pre_energy[ant_id] - post_energy[ant_id]) < 1e-9

        # Colony food stored must match
        assert abs(pre_food - colony_b.food_stored) < 1e-9

        # Pheromone grid must match
        np.testing.assert_array_equal(phero.grid, phero_b.grid)
