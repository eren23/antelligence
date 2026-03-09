"""Tests for world/pheromone.py — PheromoneGrid."""

from __future__ import annotations

import time

import numpy as np
import pytest

from config import PheromoneConfig, PheromoneChannelConfig
from world.pheromone import PheromoneGrid, Channel


# ---------------------------------------------------------------------------
# Helper configs for isolating behaviours
# ---------------------------------------------------------------------------

def _no_diffusion_cfg() -> PheromoneConfig:
    """sigma=0 → identity diffusion, so tick() only evaporates."""
    return PheromoneConfig(
        cell_size=4,
        channels={
            "food": PheromoneChannelConfig(decay=0.995, diffusion_sigma=0.0),
            "home": PheromoneChannelConfig(decay=0.990, diffusion_sigma=0.0),
            "danger": PheromoneChannelConfig(decay=0.980, diffusion_sigma=0.0),
            "recruit": PheromoneChannelConfig(decay=0.970, diffusion_sigma=0.0),
        },
    )


def _no_evaporation_cfg() -> PheromoneConfig:
    """decay=1.0 → no evaporation, so tick() only diffuses."""
    return PheromoneConfig(
        cell_size=4,
        channels={
            "food": PheromoneChannelConfig(decay=1.0, diffusion_sigma=0.5),
            "home": PheromoneChannelConfig(decay=1.0, diffusion_sigma=0.5),
            "danger": PheromoneChannelConfig(decay=1.0, diffusion_sigma=0.8),
            "recruit": PheromoneChannelConfig(decay=1.0, diffusion_sigma=1.0),
        },
    )


# ---------------------------------------------------------------------------
# Deposit
# ---------------------------------------------------------------------------

class TestDeposit:
    def test_deposit_and_readback(self):
        pg = PheromoneGrid(100, 100)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.5)
        # 10 // 4 = 2
        assert pg.grid[Channel.FOOD, 2, 2] == pytest.approx(0.5)

    def test_deposit_stacking(self):
        pg = PheromoneGrid(100, 100)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.3)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.4)
        assert pg.grid[Channel.FOOD, 2, 2] == pytest.approx(0.7)

    def test_deposit_clamped_to_one(self):
        pg = PheromoneGrid(100, 100)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.8)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.5)
        assert pg.grid[Channel.FOOD, 2, 2] == pytest.approx(1.0)

    def test_deposit_out_of_bounds_ignored(self):
        pg = PheromoneGrid(100, 100)
        pg.deposit(-5.0, 10.0, Channel.FOOD, 0.5)
        pg.deposit(10.0, 200.0, Channel.FOOD, 0.5)
        pg.deposit(100.0, 10.0, Channel.FOOD, 0.5)  # exactly at boundary
        assert pg.grid.sum() == 0.0


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

class TestDiffusion:
    def test_mass_conservation_interior(self):
        """Diffusion-only (decay=1) preserves mass for an interior deposit."""
        cfg = _no_evaporation_cfg()
        pg = PheromoneGrid(400, 400, cfg)
        pg.deposit(200.0, 200.0, Channel.FOOD, 1.0)

        mass_before = float(pg.grid[Channel.FOOD].sum())
        pg.tick()
        mass_after = float(pg.grid[Channel.FOOD].sum())

        assert mass_after == pytest.approx(mass_before, rel=1e-3)

    def test_diffusion_spreads_to_neighbours(self):
        cfg = _no_evaporation_cfg()
        pg = PheromoneGrid(100, 100, cfg)
        pg.deposit(50.0, 50.0, Channel.FOOD, 1.0)
        col, row = 50 // 4, 50 // 4  # (12, 12)

        assert pg.grid[Channel.FOOD, row, col + 1] == 0.0
        pg.tick()
        assert pg.grid[Channel.FOOD, row, col] < 1.0
        assert pg.grid[Channel.FOOD, row, col + 1] > 0.0
        assert pg.grid[Channel.FOOD, row + 1, col] > 0.0

    def test_uniform_grid_unchanged_by_diffusion(self):
        """A perfectly uniform grid stays uniform after diffusion (interior)."""
        cfg = _no_evaporation_cfg()
        pg = PheromoneGrid(100, 100, cfg)
        pg.grid[Channel.FOOD, :, :] = 0.5

        pg.tick()

        interior = pg.grid[Channel.FOOD, 2:-2, 2:-2]
        assert np.allclose(interior, 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# Evaporation
# ---------------------------------------------------------------------------

class TestEvaporation:
    def test_single_tick_decay(self):
        cfg = _no_diffusion_cfg()
        pg = PheromoneGrid(100, 100, cfg)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.8)
        pg.tick()
        assert pg.grid[Channel.FOOD, 2, 2] == pytest.approx(0.8 * 0.995, rel=1e-5)

    def test_multi_tick_decay(self):
        cfg = _no_diffusion_cfg()
        pg = PheromoneGrid(100, 100, cfg)
        pg.deposit(10.0, 10.0, Channel.HOME, 1.0)
        n = 100
        for _ in range(n):
            pg.tick()
        expected = 0.990 ** n
        assert pg.grid[Channel.HOME, 2, 2] == pytest.approx(expected, rel=1e-4)

    def test_per_channel_decay_rates(self):
        cfg = _no_diffusion_cfg()
        pg = PheromoneGrid(100, 100, cfg)
        for ch in Channel:
            pg.deposit(10.0, 10.0, ch, 1.0)

        pg.tick()

        expected = {
            Channel.FOOD: 0.995,
            Channel.HOME: 0.990,
            Channel.DANGER: 0.980,
            Channel.RECRUIT: 0.970,
        }
        for ch, decay in expected.items():
            assert pg.grid[ch, 2, 2] == pytest.approx(decay, rel=1e-5)


# ---------------------------------------------------------------------------
# Bilinear sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_sample_at_cell_centre(self):
        pg = PheromoneGrid(100, 100)
        pg.grid[Channel.FOOD, 5, 5] = 0.75
        # Cell (5,5) centre → world (5*4+2, 5*4+2) = (22, 22)
        assert pg.sample(22.0, 22.0, Channel.FOOD) == pytest.approx(0.75, abs=1e-6)

    def test_sample_between_two_centres_horizontal(self):
        pg = PheromoneGrid(100, 100)
        pg.grid[Channel.FOOD, 5, 5] = 0.4
        pg.grid[Channel.FOOD, 5, 6] = 0.8
        # Midpoint of centres x=22 and x=26 → x=24
        assert pg.sample(24.0, 22.0, Channel.FOOD) == pytest.approx(0.6, abs=1e-5)

    def test_sample_bilinear_four_corners(self):
        pg = PheromoneGrid(100, 100)
        pg.grid[Channel.FOOD, 5, 5] = 0.0
        pg.grid[Channel.FOOD, 5, 6] = 1.0
        pg.grid[Channel.FOOD, 6, 5] = 1.0
        pg.grid[Channel.FOOD, 6, 6] = 0.0
        # Centre of the 4 cells: world (24, 24)
        assert pg.sample(24.0, 24.0, Channel.FOOD) == pytest.approx(0.5, abs=1e-5)

    def test_sample_all_channels(self):
        pg = PheromoneGrid(100, 100)
        pg.grid[Channel.FOOD, 5, 5] = 0.1
        pg.grid[Channel.DANGER, 5, 5] = 0.9
        result = pg.sample_all(22.0, 22.0)
        assert result["food"] == pytest.approx(0.1, abs=1e-6)
        assert result["danger"] == pytest.approx(0.9, abs=1e-6)
        assert result["home"] == pytest.approx(0.0, abs=1e-6)
        assert result["recruit"] == pytest.approx(0.0, abs=1e-6)

    def test_sample_edge_clamp(self):
        """Sampling far outside the grid clamps to edge cell values."""
        pg = PheromoneGrid(100, 100)
        pg.grid[Channel.FOOD, 0, 0] = 0.33
        # World (0,0) is top-left corner — should return corner cell
        assert pg.sample(0.0, 0.0, Channel.FOOD) == pytest.approx(0.33, abs=1e-5)


# ---------------------------------------------------------------------------
# Channel independence
# ---------------------------------------------------------------------------

class TestChannelIndependence:
    def test_deposit_one_channel_others_zero(self):
        pg = PheromoneGrid(100, 100)
        pg.deposit(10.0, 10.0, Channel.FOOD, 0.5)
        assert pg.grid[Channel.FOOD, 2, 2] == pytest.approx(0.5)
        for ch in (Channel.HOME, Channel.DANGER, Channel.RECRUIT):
            assert pg.grid[ch, 2, 2] == 0.0

    def test_tick_channel_isolation(self):
        pg = PheromoneGrid(200, 200)
        pg.deposit(100.0, 100.0, Channel.DANGER, 1.0)
        pg.tick()
        for ch in (Channel.FOOD, Channel.HOME, Channel.RECRUIT):
            assert pg.grid[ch].sum() == 0.0
        assert pg.grid[Channel.DANGER].sum() > 0.0


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_tick_under_5ms(self):
        """tick() on default 400x250 grid completes in <5 ms (median of 20 runs)."""
        pg = PheromoneGrid(1600, 1000)  # 400 cols × 250 rows
        rng = np.random.default_rng(42)
        pg.grid[:] = rng.random(pg.grid.shape, dtype=np.float32) * 0.5

        # Warm-up
        for _ in range(3):
            pg.tick()

        times: list[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            pg.tick()
            times.append(time.perf_counter() - t0)

        median_ms = sorted(times)[len(times) // 2] * 1000
        assert median_ms < 5.0, f"Median tick time {median_ms:.2f} ms exceeds 5 ms target"
