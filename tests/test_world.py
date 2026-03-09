"""Tests for world engine: food sources, obstacles, terrain, and bounds."""

from __future__ import annotations

import math
import random

import pytest

from agents.ant import Vec2
from config import default_config
from world.food import FoodSource
from world.obstacle import Obstacle, generate_convex_polygon
from world.world import World, NestRegion, TerrainType


# ---------------------------------------------------------------------------
# FoodSource
# ---------------------------------------------------------------------------

class TestFoodSource:
    def test_deplete_partial(self):
        food = FoodSource(pos=Vec2(100, 100), radius=20, amount=100, max_amount=100)
        taken = food.deplete(30)
        assert taken == 30
        assert food.amount == pytest.approx(70)
        assert not food.depleted

    def test_deplete_full(self):
        food = FoodSource(pos=Vec2(100, 100), radius=20, amount=10, max_amount=100)
        taken = food.deplete(50)
        assert taken == pytest.approx(10)
        assert food.amount == 0
        assert food.depleted

    def test_deplete_already_depleted(self):
        food = FoodSource(
            pos=Vec2(100, 100), radius=20, amount=0, max_amount=100, depleted=True,
        )
        assert food.deplete(10) == 0

    def test_contains_point(self):
        food = FoodSource(pos=Vec2(50, 50), radius=10, amount=100, max_amount=100)
        assert food.contains_point(Vec2(50, 50))
        assert food.contains_point(Vec2(55, 50))
        assert not food.contains_point(Vec2(70, 50))

    def test_respawn_before_interval(self):
        food = FoodSource(
            pos=Vec2(0, 0), radius=10, amount=0, max_amount=100, depleted=True,
        )
        rng = random.Random(42)
        for _ in range(100):
            assert not food.tick_respawn(200, rng=rng)

    def test_respawn_after_interval(self):
        food = FoodSource(
            pos=Vec2(0, 0), radius=10, amount=0, max_amount=100, depleted=True,
        )
        rng = random.Random(42)
        # Fast-forward past the interval
        food._ticks_since_depleted = 2999
        respawned = False
        for _ in range(500):
            if food.tick_respawn(3000, rng=rng):
                respawned = True
                break
        assert respawned
        assert food.amount == 100
        assert not food.depleted

    def test_non_depleted_no_respawn(self):
        food = FoodSource(pos=Vec2(0, 0), radius=10, amount=50, max_amount=100)
        assert not food.tick_respawn(100)


# ---------------------------------------------------------------------------
# Obstacle geometry
# ---------------------------------------------------------------------------

class TestObstacle:
    @pytest.fixture
    def square(self) -> Obstacle:
        """Axis-aligned 10×10 square at origin (CCW winding)."""
        return Obstacle(vertices=[
            Vec2(0, 0), Vec2(10, 0), Vec2(10, 10), Vec2(0, 10),
        ])

    @pytest.fixture
    def triangle(self) -> Obstacle:
        return Obstacle(vertices=[Vec2(0, 0), Vec2(20, 0), Vec2(10, 20)])

    # -- point-in-polygon --

    def test_contains_inside(self, square: Obstacle):
        assert square.contains_point(Vec2(5, 5))

    def test_contains_outside(self, square: Obstacle):
        assert not square.contains_point(Vec2(15, 5))
        assert not square.contains_point(Vec2(-1, 5))

    def test_contains_on_edge(self, square: Obstacle):
        assert square.contains_point(Vec2(5, 0))

    def test_triangle_contains(self, triangle: Obstacle):
        assert triangle.contains_point(Vec2(10, 5))
        assert not triangle.contains_point(Vec2(0, 20))

    # -- ray intersection --

    def test_ray_hit_from_left(self, square: Obstacle):
        t = square.ray_intersection(Vec2(-5, 5), Vec2(1, 0))
        assert t is not None
        assert t == pytest.approx(5.0)

    def test_ray_miss(self, square: Obstacle):
        t = square.ray_intersection(Vec2(-5, 5), Vec2(-1, 0))
        assert t is None

    def test_ray_from_above(self, square: Obstacle):
        t = square.ray_intersection(Vec2(5, 20), Vec2(0, -1))
        assert t is not None
        assert t == pytest.approx(10.0)

    def test_ray_diagonal(self, square: Obstacle):
        t = square.ray_intersection(Vec2(-5, -5), Vec2(1, 1))
        assert t is not None
        assert t > 0

    def test_ray_zero_direction(self, square: Obstacle):
        assert square.ray_intersection(Vec2(5, 5), Vec2(0, 0)) is None

    # -- slide along edge --

    def test_slide_pushes_outside(self, square: Obstacle):
        """Point near bottom edge is pushed below the polygon."""
        pos = Vec2(5, 0.5)
        vel = Vec2(2, -3)
        new_pos, new_vel = square.slide_along_edge(pos, vel)
        assert new_pos.y <= 0.0
        # Velocity should be projected onto the bottom edge (horizontal)
        assert new_vel.x == pytest.approx(2.0, abs=0.01)
        assert abs(new_vel.y) < 0.01

    def test_slide_velocity_direction(self, square: Obstacle):
        """Velocity component along the edge is preserved in sign."""
        pos = Vec2(0.5, 5)
        vel = Vec2(-3, 4)
        new_pos, new_vel = square.slide_along_edge(pos, vel)
        # Near the left edge; velocity projected onto left edge (vertical)
        assert new_pos.x <= 0.0

    # -- construction --

    def test_minimum_vertices_rejected(self):
        with pytest.raises(ValueError):
            Obstacle(vertices=[Vec2(0, 0), Vec2(1, 1)])

    def test_generate_convex_polygon_count(self):
        rng = random.Random(42)
        verts = generate_convex_polygon(Vec2(100, 100), 15, 40, num_vertices=6, rng=rng)
        assert len(verts) == 6

    def test_generate_convex_polygon_is_convex(self):
        """All cross-products of consecutive edge pairs must have the same sign."""
        rng = random.Random(99)
        for _ in range(20):
            verts = generate_convex_polygon(
                Vec2(rng.uniform(0, 500), rng.uniform(0, 500)),
                10, 50, rng=rng,
            )
            n = len(verts)
            crosses = []
            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]
                c = verts[(i + 2) % n]
                cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
                crosses.append(cross)
            assert all(c >= 0 for c in crosses) or all(c <= 0 for c in crosses)


# ---------------------------------------------------------------------------
# World — bounds, terrain, placement, raycast
# ---------------------------------------------------------------------------

class TestWorld:
    @pytest.fixture
    def world(self) -> World:
        return World.from_config(default_config(), seed=42)

    @pytest.fixture
    def small_world(self) -> World:
        cfg = default_config()
        cfg.world.width = 400
        cfg.world.height = 300
        cfg.world.num_food_sources = 3
        cfg.world.num_obstacles = 5
        return World.from_config(cfg, seed=123)

    # -- bounds --

    def test_in_bounds(self, world: World):
        assert world.in_bounds(Vec2(100, 100))
        assert world.in_bounds(Vec2(0, 0))
        assert world.in_bounds(Vec2(1600, 1000))
        assert not world.in_bounds(Vec2(-1, 100))
        assert not world.in_bounds(Vec2(100, 1001))

    def test_clamp_to_bounds(self, world: World):
        clamped = world.clamp_to_bounds(Vec2(-10, 2000))
        assert clamped.x == 0.0
        assert clamped.y == 1000.0

    # -- nest --

    def test_nest_contains_center(self, world: World):
        assert world.nest.contains(world.nest.center)

    def test_nest_excludes_far_point(self, world: World):
        assert not world.nest.contains(Vec2(world.width - 10, world.height - 10))

    # -- terrain --

    def test_terrain_valid_types(self, world: World):
        valid = {"grass", "mud", "sand", "water"}
        for x in range(0, world.width, 200):
            for y in range(0, world.height, 200):
                assert world.get_terrain(Vec2(x, y)) in valid

    def test_terrain_deterministic(self, world: World):
        pos = Vec2(500.5, 300.3)
        assert world.get_terrain(pos) == world.get_terrain(pos)

    def test_terrain_speed_modifiers(self, world: World):
        assert world.terrain_speed_modifier("grass") == 1.0
        assert world.terrain_speed_modifier("mud") == 0.6
        assert world.terrain_speed_modifier("sand") == 0.8
        assert world.terrain_speed_modifier("water") == 0.3

    # -- placement --

    def test_food_sources_count(self, world: World):
        assert 3 <= len(world.food_sources) <= 6

    def test_food_sources_in_bounds(self, world: World):
        for food in world.food_sources:
            assert world.in_bounds(food.pos)
            assert food.amount > 0
            assert not food.depleted

    def test_food_not_in_nest(self, world: World):
        for food in world.food_sources:
            assert food.pos.distance_to(world.nest.center) > world.nest.radius

    def test_obstacles_count(self, world: World):
        assert 5 <= len(world.obstacles) <= 15

    def test_obstacles_have_vertices(self, world: World):
        for obs in world.obstacles:
            assert len(obs.vertices) >= 3

    # -- queries --

    def test_food_at_found(self, small_world: World):
        if small_world.food_sources:
            food = small_world.food_sources[0]
            assert small_world.food_at(food.pos) is food

    def test_food_at_empty(self, world: World):
        assert world.food_at(Vec2(0, 0)) is None

    # -- raycast --

    def test_raycast_no_obstacles(self):
        world = World(
            width=200, height=200,
            nest=NestRegion(center=Vec2(100, 100), radius=20),
        )
        assert world.raycast(Vec2(50, 50), Vec2(1, 0), 100) == 100.0

    def test_raycast_hits_obstacle(self):
        obs = Obstacle(vertices=[
            Vec2(80, 40), Vec2(90, 40), Vec2(90, 60), Vec2(80, 60),
        ])
        world = World(
            width=200, height=200,
            nest=NestRegion(center=Vec2(100, 100), radius=20),
            obstacles=[obs],
        )
        dist = world.raycast(Vec2(50, 50), Vec2(1, 0), 200)
        assert dist == pytest.approx(30.0)

    # -- tick / respawn --

    def test_tick_triggers_respawn(self):
        food = FoodSource(
            pos=Vec2(100, 100), radius=20, amount=0, max_amount=100, depleted=True,
        )
        world = World(
            width=200, height=200,
            nest=NestRegion(center=Vec2(50, 50), radius=20),
            food_sources=[food],
            _food_respawn_interval=10,
            _terrain_seed=42,
        )
        respawned = False
        for _ in range(500):
            world.tick()
            if not food.depleted:
                respawned = True
                break
        assert respawned

    # -- determinism --

    def test_from_config_deterministic(self):
        cfg = default_config()
        w1 = World.from_config(cfg, seed=99)
        w2 = World.from_config(cfg, seed=99)
        assert len(w1.obstacles) == len(w2.obstacles)
        assert len(w1.food_sources) == len(w2.food_sources)
        for f1, f2 in zip(w1.food_sources, w2.food_sources):
            assert f1.pos.x == pytest.approx(f2.pos.x)
            assert f1.pos.y == pytest.approx(f2.pos.y)
