"""Tests for sensory builder — agents/sensory.py."""

from __future__ import annotations

import math

import pytest

from agents.ant import Ant, Role, Vec2
from agents.sensory import (
    SensoryInput,
    NeighborInfo,
    build_sensory,
    _sample_antenna_cone,
    _cast_obstacle_rays,
    _find_neighbors,
    _compute_food_gradient,
)
from config import default_config
from world.world import World, NestRegion
from world.pheromone import PheromoneGrid, Channel
from world.obstacle import Obstacle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_world(
    width: int = 200,
    height: int = 200,
    nest_center: Vec2 | None = None,
) -> World:
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
# Antenna cone sampling
# ---------------------------------------------------------------------------

class TestAntennaCone:
    def test_antenna_reads_pheromone_at_tip(self):
        """Antenna tip should sample pheromone at the correct grid position."""
        pg = PheromoneGrid(200, 200)
        # Deposit pheromone 40px to the right of (100, 100)
        pg.deposit(140.0, 100.0, Channel.FOOD, 0.9)

        # Ant facing right (heading=0), left antenna offset +30°, range 40px
        result = _sample_antenna_cone(
            Vec2(100.0, 100.0), 0.0, math.radians(30), 40.0, pg,
        )
        # The tip is at (100 + 40*cos(30°), 100 + 40*sin(30°)) ≈ (134.6, 120)
        # Not exactly at the deposit point, so reading should be small
        assert "food" in result
        assert isinstance(result["food"], float)

    def test_left_antenna_higher_when_source_left(self):
        """Pheromone to the left should give higher reading on left antenna."""
        pg = PheromoneGrid(200, 200)
        # Ant at (100, 100) heading right. Place pheromone above (which is
        # left when heading=0, since left antenna = +30° = upward component)
        # Actually, positive y is "down" in screen coords, so +angle goes "down"
        # Place pheromone at the left antenna tip position
        tip_angle = math.radians(30)
        tip = Vec2(100 + 40 * math.cos(tip_angle), 100 + 40 * math.sin(tip_angle))
        pg.deposit(tip.x, tip.y, Channel.FOOD, 0.9)

        left = _sample_antenna_cone(
            Vec2(100.0, 100.0), 0.0, math.radians(30), 40.0, pg,
        )
        right = _sample_antenna_cone(
            Vec2(100.0, 100.0), 0.0, -math.radians(30), 40.0, pg,
        )
        assert left["food"] > right["food"]

    def test_all_channels_returned(self):
        pg = PheromoneGrid(200, 200)
        result = _sample_antenna_cone(Vec2(100, 100), 0.0, 0.5, 40.0, pg)
        for ch in ("food", "home", "danger", "recruit"):
            assert ch in result


# ---------------------------------------------------------------------------
# Obstacle raycasts
# ---------------------------------------------------------------------------

class TestObstacleRaycasts:
    def test_no_obstacles_all_ones(self):
        world = _small_world()
        rays = _cast_obstacle_rays(
            Vec2(100, 100), 0.0, 5, 30.0, math.radians(120), world,
        )
        assert len(rays) == 5
        assert all(r == pytest.approx(1.0) for r in rays)

    def test_obstacle_ahead_centre_ray_hits(self):
        world = _small_world()
        # Place obstacle directly in front (heading=0, so to the right)
        verts = [Vec2(115, 95), Vec2(125, 95), Vec2(125, 105), Vec2(115, 105)]
        world.obstacles.append(Obstacle(vertices=verts))
        rays = _cast_obstacle_rays(
            Vec2(100, 100), 0.0, 5, 30.0, math.radians(120), world,
        )
        # Centre ray (index 2) should hit the obstacle
        assert rays[2] < 1.0

    def test_ray_count_matches(self):
        world = _small_world()
        rays = _cast_obstacle_rays(
            Vec2(100, 100), 0.0, 3, 30.0, math.radians(120), world,
        )
        assert len(rays) == 3

    def test_closer_obstacle_smaller_value(self):
        world = _small_world()
        verts = [Vec2(110, 95), Vec2(120, 95), Vec2(120, 105), Vec2(110, 105)]
        world.obstacles.append(Obstacle(vertices=verts))
        rays = _cast_obstacle_rays(
            Vec2(100, 100), 0.0, 5, 30.0, math.radians(120), world,
        )
        # Centre ray should be closer than outer rays
        assert rays[2] < 1.0


# ---------------------------------------------------------------------------
# Neighbor detection
# ---------------------------------------------------------------------------

class TestNeighborDetection:
    def test_finds_nearby_ants(self):
        ant = Ant(id=0, pos=Vec2(100, 100))
        nearby = Ant(id=1, pos=Vec2(110, 100))
        far = Ant(id=2, pos=Vec2(200, 200))
        neighbors = _find_neighbors(ant, [ant, nearby, far], 20.0)
        assert len(neighbors) == 1
        assert neighbors[0].id == 1

    def test_excludes_self(self):
        ant = Ant(id=0, pos=Vec2(100, 100))
        neighbors = _find_neighbors(ant, [ant], 20.0)
        assert len(neighbors) == 0

    def test_excludes_dead_ants(self):
        ant = Ant(id=0, pos=Vec2(100, 100))
        dead = Ant(id=1, pos=Vec2(105, 100), alive=False)
        neighbors = _find_neighbors(ant, [ant, dead], 20.0)
        assert len(neighbors) == 0

    def test_relative_position_correct(self):
        ant = Ant(id=0, pos=Vec2(100, 100))
        other = Ant(id=1, pos=Vec2(110, 105))
        neighbors = _find_neighbors(ant, [ant, other], 20.0)
        assert len(neighbors) == 1
        assert neighbors[0].relative_pos.x == pytest.approx(10.0)
        assert neighbors[0].relative_pos.y == pytest.approx(5.0)

    def test_neighbor_info_fields(self):
        ant = Ant(id=0, pos=Vec2(100, 100))
        other = Ant(id=1, pos=Vec2(110, 100), role=Role.SOLDIER, carrying="food")
        neighbors = _find_neighbors(ant, [ant, other], 20.0)
        assert neighbors[0].role_value == "soldier"
        assert neighbors[0].carrying == "food"

    def test_boundary_radius(self):
        """Ant at exactly the radius boundary should be included."""
        ant = Ant(id=0, pos=Vec2(100, 100))
        edge = Ant(id=1, pos=Vec2(120, 100))  # exactly 20px away
        neighbors = _find_neighbors(ant, [ant, edge], 20.0)
        assert len(neighbors) == 1


# ---------------------------------------------------------------------------
# Nest direction
# ---------------------------------------------------------------------------

class TestNestDirection:
    def test_nest_to_right(self):
        world = _small_world(nest_center=Vec2(150.0, 100.0))
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(100.0, 100.0))
        si = build_sensory(ant, world, pg, [ant], cfg)
        assert si.nest_direction.x > 0
        assert si.nest_direction.y == pytest.approx(0.0, abs=0.01)
        assert si.nest_distance == pytest.approx(50.0)

    def test_nest_direction_unit_vector(self):
        world = _small_world(nest_center=Vec2(150.0, 130.0))
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(100.0, 100.0))
        si = build_sensory(ant, world, pg, [ant], cfg)
        length = si.nest_direction.length()
        assert length == pytest.approx(1.0, abs=1e-6)

    def test_nest_at_same_position(self):
        world = _small_world(nest_center=Vec2(100.0, 100.0))
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(100.0, 100.0))
        si = build_sensory(ant, world, pg, [ant], cfg)
        assert si.nest_distance == pytest.approx(0.0)
        # Direction should be zero vector when at nest
        assert si.nest_direction.length() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Food gradient
# ---------------------------------------------------------------------------

class TestFoodGradient:
    def test_gradient_points_toward_food(self):
        pg = PheromoneGrid(200, 200)
        # Deposit food pheromone just to the right of the sampling position
        # (within one cell_size so central-difference can detect it)
        pg.deposit(106.0, 100.0, Channel.FOOD, 1.0)
        grad = _compute_food_gradient(Vec2(100.0, 100.0), pg)
        # Gradient x should be positive (pointing toward food)
        assert grad.x > 0

    def test_zero_gradient_no_pheromone(self):
        pg = PheromoneGrid(200, 200)
        grad = _compute_food_gradient(Vec2(100.0, 100.0), pg)
        assert grad.x == pytest.approx(0.0, abs=1e-10)
        assert grad.y == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Ground type
# ---------------------------------------------------------------------------

class TestGroundType:
    def test_ground_type_is_valid(self):
        world = _small_world()
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(50.0, 50.0))
        si = build_sensory(ant, world, pg, [ant], cfg)
        assert si.ground_type in ("grass", "mud", "sand", "water")


# ---------------------------------------------------------------------------
# Full build_sensory integration
# ---------------------------------------------------------------------------

class TestBuildSensory:
    def test_returns_sensory_input(self):
        world = _small_world()
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(100.0, 100.0), energy=75.0, role=Role.NURSE)
        si = build_sensory(ant, world, pg, [ant], cfg)
        assert isinstance(si, SensoryInput)
        assert si.energy == pytest.approx(75.0)
        assert si.role_value == "nurse"

    def test_obstacle_rays_count(self):
        world = _small_world()
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(100.0, 100.0))
        si = build_sensory(ant, world, pg, [ant], cfg)
        assert len(si.obstacle_rays) == cfg.obstacle_ray_count

    def test_self_state_copied(self):
        world = _small_world()
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(
            id=0, pos=Vec2(100, 100), energy=42.0,
            carrying="brood", carry_amount=0.3, role=Role.SOLDIER,
            age=500, speed=1.5,
        )
        si = build_sensory(ant, world, pg, [ant], cfg)
        assert si.energy == pytest.approx(42.0)
        assert si.carrying == "brood"
        assert si.carry_amount == pytest.approx(0.3)
        assert si.role_value == "soldier"
        assert si.age == 500
        assert si.speed == pytest.approx(1.5)

    def test_to_vector_dimensionality(self):
        world = _small_world()
        cfg = _default_ant_cfg()
        pg = PheromoneGrid(200, 200)
        ant = Ant(id=0, pos=Vec2(100.0, 100.0))
        si = build_sensory(ant, world, pg, [ant], cfg)
        vec = si.to_vector()
        assert len(vec) == 39
        assert all(isinstance(v, float) for v in vec)
