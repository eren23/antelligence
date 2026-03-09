"""Shared test fixtures for the colony simulation."""

from __future__ import annotations

import math

import pytest

from agents.ant import Ant, Role, Vec2
from agents.actions import AntAction
from agents.sensory import SensoryInput, NeighborInfo
from config import SimConfig, default_config, load_config


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg() -> SimConfig:
    """Full default configuration (no file needed)."""
    return default_config()


@pytest.fixture
def small_world_cfg() -> SimConfig:
    """A small 200x200 world for fast tests."""
    cfg = default_config()
    cfg.world.width = 200
    cfg.world.height = 200
    cfg.colony.initial_population = 10
    cfg.colony.max_population = 20
    cfg.world.num_food_sources = 2
    cfg.world.num_obstacles = 2
    return cfg


# ---------------------------------------------------------------------------
# Vec2 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def origin() -> Vec2:
    return Vec2(0.0, 0.0)


@pytest.fixture
def nest_pos() -> Vec2:
    """Default nest position (center-left of default world)."""
    return Vec2(200.0, 500.0)


# ---------------------------------------------------------------------------
# Ant fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_ant(nest_pos: Vec2) -> Ant:
    """A single ant at the nest with full energy, no brain attached."""
    return Ant(
        id=0,
        pos=Vec2(nest_pos.x, nest_pos.y),
        heading=0.0,
        speed=2.0,
        energy=100.0,
        role=Role.FORAGER,
    )


@pytest.fixture
def carrying_ant(nest_pos: Vec2) -> Ant:
    """An ant carrying food, heading home."""
    return Ant(
        id=1,
        pos=Vec2(800.0, 500.0),
        heading=math.pi,  # facing left (toward nest)
        speed=2.0,
        energy=80.0,
        carrying="food",
        carry_amount=0.5,
        role=Role.FORAGER,
    )


@pytest.fixture
def low_energy_ant() -> Ant:
    """An ant about to die from energy depletion."""
    return Ant(
        id=2,
        pos=Vec2(400.0, 300.0),
        heading=0.0,
        speed=2.0,
        energy=0.05,
        role=Role.FORAGER,
    )


@pytest.fixture
def ant_colony(nest_pos: Vec2) -> list[Ant]:
    """A small colony of 10 ants at the nest."""
    return [
        Ant(
            id=i,
            pos=Vec2(nest_pos.x + i * 2, nest_pos.y),
            heading=i * (2 * math.pi / 10),
            speed=2.0,
            energy=100.0,
            role=Role.FORAGER,
        )
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# Sensory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_sensory() -> SensoryInput:
    """Sensory input with all defaults (no stimuli)."""
    return SensoryInput()


@pytest.fixture
def food_sensory() -> SensoryInput:
    """Sensory input with food pheromone detected to the left."""
    return SensoryInput(
        antenna_left={"food": 0.8, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        antenna_right={"food": 0.2, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        obstacle_rays=[1.0, 1.0, 1.0, 1.0, 1.0],
        nest_direction=Vec2(-1.0, 0.0),
        nest_distance=500.0,
        food_gradient=Vec2(0.5, 0.3),
        energy=90.0,
        role_value="forager",
    )


# ---------------------------------------------------------------------------
# Action fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def noop_action() -> AntAction:
    """An action that does nothing (go straight, normal speed, no interactions)."""
    return AntAction()


@pytest.fixture
def turn_left_action() -> AntAction:
    """Turn left at max rate."""
    return AntAction(turn=-math.pi / 6)


@pytest.fixture
def deposit_food_action() -> AntAction:
    """Deposit food pheromone at full strength."""
    return AntAction(
        deposit_pheromone="food",
        deposit_strength=1.0,
    )
