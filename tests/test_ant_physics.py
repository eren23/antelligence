"""Tests for ant physics/movement — agents/ant.py."""

from __future__ import annotations

import math

import pytest

from agents.ant import (
    Ant,
    Role,
    Vec2,
    _wrap_angle,
    update_heading,
    advance_position,
    handle_obstacle_collision,
    handle_world_bounds,
    deplete_energy,
    check_death,
    refill_at_nest,
    try_pickup,
    try_drop,
)
from agents.actions import AntAction
from config import default_config
from world.world import World, NestRegion
from world.food import FoodSource
from world.obstacle import Obstacle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_world(
    width: int = 200,
    height: int = 200,
    nest_center: Vec2 | None = None,
) -> World:
    """Create a minimal world with no obstacles or food."""
    if nest_center is None:
        nest_center = Vec2(100.0, 100.0)
    return World(
        width=width,
        height=height,
        nest=NestRegion(center=nest_center, radius=40.0),
    )


def _default_ant_cfg():
    return default_config().ant


# ---------------------------------------------------------------------------
# Heading update
# ---------------------------------------------------------------------------

class TestHeadingUpdate:
    def test_zero_turn(self):
        ant = Ant(id=0, heading=0.5)
        update_heading(ant, AntAction(turn=0.0))
        assert ant.heading == pytest.approx(0.5)

    def test_positive_turn(self):
        ant = Ant(id=0, heading=0.0)
        update_heading(ant, AntAction(turn=0.3))
        assert ant.heading == pytest.approx(0.3)

    def test_turn_clamped(self):
        """Turn greater than pi/6 is clamped."""
        ant = Ant(id=0, heading=0.0)
        update_heading(ant, AntAction(turn=1.0))  # > pi/6 ≈ 0.524
        assert ant.heading == pytest.approx(math.pi / 6, abs=1e-6)

    def test_heading_wraps_at_pi(self):
        ant = Ant(id=0, heading=math.pi - 0.1)
        update_heading(ant, AntAction(turn=math.pi / 6))
        assert -math.pi <= ant.heading <= math.pi

    def test_heading_wraps_at_negative_pi(self):
        ant = Ant(id=0, heading=-math.pi + 0.1)
        update_heading(ant, AntAction(turn=-math.pi / 6))
        assert -math.pi <= ant.heading <= math.pi


class TestWrapAngle:
    def test_within_range(self):
        assert _wrap_angle(1.0) == pytest.approx(1.0)

    def test_positive_overflow(self):
        assert _wrap_angle(math.pi + 0.5) == pytest.approx(-math.pi + 0.5, abs=1e-10)

    def test_negative_overflow(self):
        assert _wrap_angle(-math.pi - 0.5) == pytest.approx(math.pi - 0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# Position advance
# ---------------------------------------------------------------------------

class TestAdvancePosition:
    def test_straight_ahead(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), heading=0.0, speed=2.0)
        advance_position(ant, AntAction(speed_mult=1.0), world)
        # Heading 0 → moves right (+x), terrain at (100,100) is grass (1.0 mult)
        assert ant.pos.x == pytest.approx(102.0, abs=0.5)
        assert ant.pos.y == pytest.approx(100.0, abs=0.5)

    def test_speed_mult_zero(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), heading=0.0, speed=2.0)
        advance_position(ant, AntAction(speed_mult=0.0), world)
        assert ant.pos.x == pytest.approx(100.0)
        assert ant.pos.y == pytest.approx(100.0)

    def test_speed_mult_half(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), heading=0.0, speed=2.0)
        terrain = world.get_terrain(ant.pos)
        tmult = world.terrain_speed_modifier(terrain)
        advance_position(ant, AntAction(speed_mult=0.5), world)
        expected_x = 100.0 + 2.0 * 0.5 * tmult
        assert ant.pos.x == pytest.approx(expected_x, abs=1e-6)

    def test_heading_pi_half_moves_down(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, 50.0), heading=math.pi / 2, speed=2.0)
        advance_position(ant, AntAction(speed_mult=1.0), world)
        assert ant.pos.y > 50.0  # moved down (positive y)
        assert ant.pos.x == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Obstacle collision
# ---------------------------------------------------------------------------

class TestObstacleCollision:
    def _world_with_obstacle(self) -> World:
        world = _small_world()
        # Square obstacle centred at (150, 100)
        verts = [Vec2(140, 90), Vec2(160, 90), Vec2(160, 110), Vec2(140, 110)]
        world.obstacles.append(Obstacle(vertices=verts))
        return world

    def test_no_collision_outside(self):
        world = self._world_with_obstacle()
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), heading=0.0, speed=2.0)
        handle_obstacle_collision(ant, world)
        # Position unchanged
        assert ant.pos.x == pytest.approx(100.0)
        assert ant.pos.y == pytest.approx(100.0)

    def test_collision_pushes_out(self):
        world = self._world_with_obstacle()
        ant = Ant(id=0, pos=Vec2(150.0, 100.0), heading=0.0, speed=2.0)
        handle_obstacle_collision(ant, world)
        # Ant should no longer be inside the obstacle
        obs = world.obstacles[0]
        assert not obs.contains_point(ant.pos)

    def test_heading_adjusted_on_slide(self):
        world = self._world_with_obstacle()
        ant = Ant(id=0, pos=Vec2(141.0, 100.0), heading=0.0, speed=2.0)
        original_heading = ant.heading
        handle_obstacle_collision(ant, world)
        # Heading may change due to slide projection
        # Just verify it's a valid angle
        assert -math.pi <= ant.heading <= math.pi


# ---------------------------------------------------------------------------
# World-bound reflection
# ---------------------------------------------------------------------------

class TestWorldBounds:
    def test_left_boundary_reflects(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(-1.0, 100.0), heading=math.pi)
        handle_world_bounds(ant, world)
        assert ant.pos.x == 0.0
        # Heading should be reflected to face right
        dx = Vec2.from_angle(ant.heading)
        assert dx.x >= 0

    def test_right_boundary_reflects(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(201.0, 100.0), heading=0.0)
        handle_world_bounds(ant, world)
        assert ant.pos.x == 200.0
        dx = Vec2.from_angle(ant.heading)
        assert dx.x <= 0

    def test_top_boundary_reflects(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, -1.0), heading=-math.pi / 2)
        handle_world_bounds(ant, world)
        assert ant.pos.y == 0.0
        dx = Vec2.from_angle(ant.heading)
        assert dx.y >= 0

    def test_bottom_boundary_reflects(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, 201.0), heading=math.pi / 2)
        handle_world_bounds(ant, world)
        assert ant.pos.y == 200.0
        dx = Vec2.from_angle(ant.heading)
        assert dx.y <= 0

    def test_in_bounds_no_change(self):
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), heading=0.5)
        handle_world_bounds(ant, world)
        assert ant.heading == pytest.approx(0.5)

    def test_corner_reflects_both(self):
        world = _small_world()
        # Heading toward top-left corner
        ant = Ant(id=0, pos=Vec2(-1.0, -1.0), heading=math.pi + math.pi / 4)
        handle_world_bounds(ant, world)
        assert ant.pos.x == 0.0
        assert ant.pos.y == 0.0


# ---------------------------------------------------------------------------
# Energy depletion
# ---------------------------------------------------------------------------

class TestEnergyDepletion:
    def test_base_cost(self):
        cfg = _default_ant_cfg()
        ant = Ant(id=0, energy=100.0)
        deplete_energy(ant, AntAction(speed_mult=0.0), cfg)
        expected = 100.0 - cfg.energy_cost_base
        assert ant.energy == pytest.approx(expected)

    def test_speed_cost(self):
        cfg = _default_ant_cfg()
        ant = Ant(id=0, energy=100.0)
        deplete_energy(ant, AntAction(speed_mult=1.0), cfg)
        expected = 100.0 - cfg.energy_cost_base - cfg.energy_cost_speed * 1.0
        assert ant.energy == pytest.approx(expected)

    def test_carry_cost(self):
        cfg = _default_ant_cfg()
        ant = Ant(id=0, energy=100.0, carrying="food", carry_amount=0.5)
        deplete_energy(ant, AntAction(speed_mult=1.0), cfg)
        expected = 100.0 - cfg.energy_cost_base - cfg.energy_cost_speed - cfg.energy_cost_carry
        assert ant.energy == pytest.approx(expected)

    def test_energy_does_not_go_negative(self):
        cfg = _default_ant_cfg()
        ant = Ant(id=0, energy=0.001)
        deplete_energy(ant, AntAction(speed_mult=1.0), cfg)
        assert ant.energy >= 0.0

    def test_multiple_ticks_accumulate(self):
        cfg = _default_ant_cfg()
        ant = Ant(id=0, energy=100.0)
        for _ in range(100):
            deplete_energy(ant, AntAction(speed_mult=1.0), cfg)
        cost_per_tick = cfg.energy_cost_base + cfg.energy_cost_speed
        expected = max(0.0, 100.0 - cost_per_tick * 100)
        assert ant.energy == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Death / corpse conversion
# ---------------------------------------------------------------------------

class TestDeath:
    def test_alive_ant_no_death(self):
        ant = Ant(id=0, energy=50.0)
        assert not check_death(ant)
        assert ant.alive

    def test_zero_energy_dies(self):
        ant = Ant(id=0, energy=0.0)
        assert check_death(ant)
        assert not ant.alive

    def test_dead_ant_clears_carrying(self):
        ant = Ant(id=0, energy=0.0, carrying="food", carry_amount=0.5)
        check_death(ant)
        assert ant.carrying is None
        assert ant.carry_amount == 0.0

    def test_dead_ant_speed_zero(self):
        ant = Ant(id=0, energy=0.0, speed=2.0)
        check_death(ant)
        assert ant.speed == 0.0

    def test_already_dead_returns_false(self):
        ant = Ant(id=0, energy=0.0, alive=False)
        assert not check_death(ant)


# ---------------------------------------------------------------------------
# Nest energy refill
# ---------------------------------------------------------------------------

class TestNestRefill:
    def test_refill_at_nest(self):
        cfg = _default_ant_cfg()
        world = _small_world(nest_center=Vec2(100.0, 100.0))
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), energy=50.0)
        result = refill_at_nest(ant, world, cfg)
        assert ant.energy == pytest.approx(50.0 + cfg.energy_refill_rate)
        assert result is None

    def test_refill_capped_at_max(self):
        cfg = _default_ant_cfg()
        world = _small_world(nest_center=Vec2(100.0, 100.0))
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), energy=99.5)
        refill_at_nest(ant, world, cfg)
        assert ant.energy == pytest.approx(cfg.energy_max)

    def test_no_refill_outside_nest(self):
        cfg = _default_ant_cfg()
        world = _small_world(nest_center=Vec2(100.0, 100.0))
        ant = Ant(id=0, pos=Vec2(0.0, 0.0), energy=50.0)
        result = refill_at_nest(ant, world, cfg)
        assert ant.energy == pytest.approx(50.0)
        assert result is None

    def test_food_deposit_at_nest(self):
        cfg = _default_ant_cfg()
        world = _small_world(nest_center=Vec2(100.0, 100.0))
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), energy=80.0, carrying="food", carry_amount=0.5)
        result = refill_at_nest(ant, world, cfg)
        assert result == "food"
        assert ant.carrying is None
        assert ant.carry_amount == 0.0


# ---------------------------------------------------------------------------
# Pickup / drop
# ---------------------------------------------------------------------------

class TestPickupDrop:
    def test_pickup_food(self):
        cfg = _default_ant_cfg()
        world = _small_world()
        food = FoodSource(pos=Vec2(50.0, 50.0), radius=20.0, amount=100.0, max_amount=100.0)
        world.food_sources.append(food)
        ant = Ant(id=0, pos=Vec2(50.0, 50.0))
        assert try_pickup(ant, world, cfg)
        assert ant.carrying == "food"
        assert ant.carry_amount > 0.0

    def test_pickup_fails_already_carrying(self):
        cfg = _default_ant_cfg()
        world = _small_world()
        food = FoodSource(pos=Vec2(50.0, 50.0), radius=20.0, amount=100.0, max_amount=100.0)
        world.food_sources.append(food)
        ant = Ant(id=0, pos=Vec2(50.0, 50.0), carrying="brood", carry_amount=0.5)
        assert not try_pickup(ant, world, cfg)

    def test_pickup_fails_no_food(self):
        cfg = _default_ant_cfg()
        world = _small_world()
        ant = Ant(id=0, pos=Vec2(50.0, 50.0))
        assert not try_pickup(ant, world, cfg)

    def test_pickup_depletes_food_source(self):
        cfg = _default_ant_cfg()
        world = _small_world()
        food = FoodSource(pos=Vec2(50.0, 50.0), radius=20.0, amount=0.5, max_amount=100.0)
        world.food_sources.append(food)
        ant = Ant(id=0, pos=Vec2(50.0, 50.0))
        try_pickup(ant, world, cfg)
        assert food.amount < 0.5

    def test_drop(self):
        ant = Ant(id=0, carrying="food", carry_amount=0.7)
        result = try_drop(ant)
        assert result == ("food", 0.7)
        assert ant.carrying is None
        assert ant.carry_amount == 0.0

    def test_drop_nothing(self):
        ant = Ant(id=0)
        result = try_drop(ant)
        assert result is None
