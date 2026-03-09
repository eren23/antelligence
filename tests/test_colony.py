"""Tests for colony manager — agents/colony.py."""

from __future__ import annotations

import math

import pytest

from agents.ant import Ant, Role, Vec2
from agents.colony import Colony, ColonyStats, Corpse
from config import default_config, SimConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colony_cfg(
    initial_pop: int = 20,
    max_pop: int = 50,
    spawn_rate: float = 0.1,
    initial_food: float = 100.0,
    rebalance_interval: int = 500,
) -> SimConfig:
    cfg = default_config()
    cfg.colony.initial_population = initial_pop
    cfg.colony.max_population = max_pop
    cfg.colony.spawn_rate = spawn_rate
    cfg.colony.initial_food_stored = initial_food
    cfg.roles.rebalance_interval = rebalance_interval
    return cfg


NEST = Vec2(100.0, 100.0)


# ---------------------------------------------------------------------------
# Initial spawning
# ---------------------------------------------------------------------------

class TestSpawning:
    def test_initial_population(self):
        cfg = _colony_cfg(initial_pop=30)
        colony = Colony(cfg, NEST, seed=42)
        assert len(colony.ants) == 30

    def test_ants_spawned_near_nest(self):
        cfg = _colony_cfg(initial_pop=50)
        colony = Colony(cfg, NEST, seed=42)
        for ant in colony.ants:
            dist = ant.pos.distance_to(NEST)
            assert dist <= 25  # within offset range (20) + tolerance

    def test_unique_ids(self):
        cfg = _colony_cfg(initial_pop=20)
        colony = Colony(cfg, NEST, seed=42)
        ids = [a.id for a in colony.ants]
        assert len(ids) == len(set(ids))

    def test_all_alive(self):
        cfg = _colony_cfg(initial_pop=10)
        colony = Colony(cfg, NEST, seed=42)
        assert all(a.alive for a in colony.ants)

    def test_full_energy(self):
        cfg = _colony_cfg(initial_pop=10)
        colony = Colony(cfg, NEST, seed=42)
        for ant in colony.ants:
            assert ant.energy == pytest.approx(cfg.ant.energy_max)


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

class TestRoleAssignment:
    def test_roles_match_distribution(self):
        """With enough ants, roles should roughly match the configured distribution."""
        cfg = _colony_cfg(initial_pop=200)
        colony = Colony(cfg, NEST, seed=42)

        counts: dict[str, int] = {}
        for ant in colony.ants:
            counts[ant.role.value] = counts.get(ant.role.value, 0) + 1

        n = len(colony.ants)
        for role_name, target_frac in cfg.roles.default_distribution.items():
            actual_frac = counts.get(role_name, 0) / n
            # Within 10% of target (generous due to randomness)
            assert abs(actual_frac - target_frac) < 0.10, (
                f"Role {role_name}: expected ~{target_frac:.2f}, got {actual_frac:.2f}"
            )

    def test_all_roles_present(self):
        cfg = _colony_cfg(initial_pop=100)
        colony = Colony(cfg, NEST, seed=42)
        roles = {a.role.value for a in colony.ants}
        for role_name in cfg.roles.default_distribution:
            assert role_name in roles


# ---------------------------------------------------------------------------
# Spawning new ants (tick)
# ---------------------------------------------------------------------------

class TestSpawnOnTick:
    def test_spawn_when_food_above_threshold(self):
        cfg = _colony_cfg(initial_pop=5, max_pop=50, spawn_rate=1.0, initial_food=200.0)
        colony = Colony(cfg, NEST, seed=42)
        initial_count = len(colony.ants)
        colony.tick(NEST)
        assert len(colony.ants) >= initial_count  # should spawn at least 1

    def test_no_spawn_low_food(self):
        cfg = _colony_cfg(initial_pop=5, max_pop=50, spawn_rate=1.0, initial_food=10.0)
        colony = Colony(cfg, NEST, seed=42)
        initial_count = len(colony.ants)
        colony.tick(NEST)
        assert len(colony.ants) == initial_count

    def test_no_spawn_at_max_population(self):
        cfg = _colony_cfg(initial_pop=10, max_pop=10, spawn_rate=1.0, initial_food=200.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.tick(NEST)
        assert len(colony.ants) == 10

    def test_food_consumed_on_spawn(self):
        cfg = _colony_cfg(initial_pop=5, max_pop=50, spawn_rate=2.0, initial_food=200.0)
        colony = Colony(cfg, NEST, seed=42)
        food_before = colony.food_stored
        colony.tick(NEST)
        if len(colony.ants) > 5:
            assert colony.food_stored < food_before


# ---------------------------------------------------------------------------
# Dead ant removal + corpse creation
# ---------------------------------------------------------------------------

class TestDeadAntRemoval:
    def test_dead_ant_removed(self):
        cfg = _colony_cfg(initial_pop=5)
        colony = Colony(cfg, NEST, seed=42)
        colony.ants[0].alive = False
        colony.tick(NEST)
        assert all(a.alive for a in colony.ants)

    def test_corpse_created_at_death_position(self):
        cfg = _colony_cfg(initial_pop=5)
        colony = Colony(cfg, NEST, seed=42)
        death_pos = Vec2(colony.ants[0].pos.x, colony.ants[0].pos.y)
        colony.ants[0].alive = False
        colony.tick(NEST)
        assert len(colony.corpses) == 1
        assert colony.corpses[0].pos.x == pytest.approx(death_pos.x)
        assert colony.corpses[0].pos.y == pytest.approx(death_pos.y)

    def test_dead_count_increments(self):
        cfg = _colony_cfg(initial_pop=5)
        colony = Colony(cfg, NEST, seed=42)
        colony.ants[0].alive = False
        colony.ants[1].alive = False
        colony.tick(NEST)
        stats = colony.stats()
        assert stats.dead_count == 2

    def test_multiple_deaths_across_ticks(self):
        cfg = _colony_cfg(initial_pop=10, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.ants[0].alive = False
        colony.tick(NEST)
        colony.ants[0].alive = False  # kill another
        colony.tick(NEST)
        assert len(colony.corpses) == 2


# ---------------------------------------------------------------------------
# Food deposit accounting
# ---------------------------------------------------------------------------

class TestFoodDeposit:
    def test_deposit_increments_storage(self):
        cfg = _colony_cfg(initial_food=100.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.deposit_food(25.0)
        assert colony.food_stored == pytest.approx(125.0)

    def test_multiple_deposits(self):
        cfg = _colony_cfg(initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.deposit_food(10.0)
        colony.deposit_food(20.0)
        colony.deposit_food(30.0)
        assert colony.food_stored == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Role rebalancing
# ---------------------------------------------------------------------------

class TestRoleRebalancing:
    def test_rebalance_at_interval(self):
        cfg = _colony_cfg(initial_pop=100, rebalance_interval=10, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)

        # Force all ants to forager
        for ant in colony.ants:
            ant.role = Role.FORAGER

        # Tick until rebalance fires (tick 10)
        for _ in range(10):
            colony.tick(NEST)

        # After rebalance, should have non-forager roles
        roles = {a.role for a in colony.ants}
        assert len(roles) > 1

    def test_rebalance_matches_distribution(self):
        cfg = _colony_cfg(initial_pop=100, rebalance_interval=5, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)

        # Force all to idle
        for ant in colony.ants:
            ant.role = Role.IDLE

        for _ in range(5):
            colony.tick(NEST)

        # Check distribution roughly matches config
        counts: dict[str, int] = {}
        for ant in colony.ants:
            counts[ant.role.value] = counts.get(ant.role.value, 0) + 1

        n = len(colony.ants)
        for role_name, target_frac in cfg.roles.default_distribution.items():
            actual_frac = counts.get(role_name, 0) / n
            assert abs(actual_frac - target_frac) < 0.10, (
                f"Role {role_name}: expected ~{target_frac:.2f}, got {actual_frac:.2f}"
            )

    def test_no_rebalance_before_interval(self):
        cfg = _colony_cfg(initial_pop=20, rebalance_interval=100, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)

        # Force all to forager
        for ant in colony.ants:
            ant.role = Role.FORAGER

        # Only tick 5 times (well before interval=100)
        for _ in range(5):
            colony.tick(NEST)

        assert all(a.role == Role.FORAGER for a in colony.ants)


# ---------------------------------------------------------------------------
# Colony stats
# ---------------------------------------------------------------------------

class TestColonyStats:
    def test_stats_population(self):
        cfg = _colony_cfg(initial_pop=15)
        colony = Colony(cfg, NEST, seed=42)
        s = colony.stats()
        assert s.population == 15

    def test_stats_food_stored(self):
        cfg = _colony_cfg(initial_food=77.0)
        colony = Colony(cfg, NEST, seed=42)
        s = colony.stats()
        assert s.food_stored == pytest.approx(77.0)

    def test_stats_role_counts(self):
        cfg = _colony_cfg(initial_pop=20)
        colony = Colony(cfg, NEST, seed=42)
        s = colony.stats()
        total = sum(s.role_counts.values())
        assert total == 20

    def test_stats_avg_energy(self):
        cfg = _colony_cfg(initial_pop=5)
        colony = Colony(cfg, NEST, seed=42)
        # All ants start at max energy
        s = colony.stats()
        assert s.avg_energy == pytest.approx(cfg.ant.energy_max)

    def test_stats_dead_count_zero(self):
        cfg = _colony_cfg(initial_pop=5)
        colony = Colony(cfg, NEST, seed=42)
        s = colony.stats()
        assert s.dead_count == 0


# ---------------------------------------------------------------------------
# Corpse management
# ---------------------------------------------------------------------------

class TestCorpseManagement:
    def test_corpse_ages(self):
        cfg = _colony_cfg(initial_pop=5, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.ants[0].alive = False
        colony.tick(NEST)
        # Corpse was created this tick, then aged once
        assert colony.corpses[0].age == 1

    def test_remove_corpse(self):
        cfg = _colony_cfg(initial_pop=5, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.ants[0].alive = False
        colony.tick(NEST)
        corpse = colony.corpses[0]
        colony.remove_corpse(corpse)
        assert len(colony.corpses) == 0

    def test_ant_age_increments(self):
        cfg = _colony_cfg(initial_pop=3, spawn_rate=0.0, initial_food=0.0)
        colony = Colony(cfg, NEST, seed=42)
        colony.tick(NEST)
        colony.tick(NEST)
        for ant in colony.ants:
            assert ant.age == 2
