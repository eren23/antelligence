"""World class — terrain, bounds, nest region, food sources, and obstacles."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.ant import Vec2
from config import SimConfig
from world.food import FoodSource
from world.obstacle import Obstacle, generate_convex_polygon


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

class TerrainType(Enum):
    GRASS = "grass"
    MUD = "mud"
    SAND = "sand"
    WATER = "water"


# ---------------------------------------------------------------------------
# Nest region
# ---------------------------------------------------------------------------

@dataclass
class NestRegion:
    """Circular nest area where ants spawn and deposit food."""

    center: Vec2
    radius: float

    def contains(self, pos: Vec2) -> bool:
        return pos.distance_to(self.center) <= self.radius


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

@dataclass
class World:
    """The simulation world with terrain, food sources, and obstacles."""

    width: int
    height: int
    nest: NestRegion
    food_sources: list[FoodSource] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    _terrain_seed: int = field(default=42, repr=False)
    _food_respawn_interval: int = field(default=3000, repr=False)
    _rng: random.Random = field(default=None, init=False, repr=False)
    _tick_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self._terrain_seed)

    # --- Bounds ---------------------------------------------------------------

    def in_bounds(self, pos: Vec2) -> bool:
        return 0 <= pos.x <= self.width and 0 <= pos.y <= self.height

    def clamp_to_bounds(self, pos: Vec2) -> Vec2:
        return Vec2(
            max(0.0, min(float(self.width), pos.x)),
            max(0.0, min(float(self.height), pos.y)),
        )

    # --- Terrain --------------------------------------------------------------

    def get_terrain(self, pos: Vec2) -> str:
        """Deterministic terrain type at *pos* using layered sinusoidal noise."""
        nx = pos.x / self.width
        ny = pos.y / self.height
        val = (
            math.sin(nx * 13.7 + ny * 7.3 + self._terrain_seed * 0.1) * 0.5
            + math.sin(nx * 5.1 - ny * 11.9 + self._terrain_seed * 0.3) * 0.3
            + math.sin(nx * 19.3 + ny * 3.7) * 0.2
        )
        if val < -0.5:
            return TerrainType.WATER.value
        if val < -0.1:
            return TerrainType.MUD.value
        if val > 0.5:
            return TerrainType.SAND.value
        return TerrainType.GRASS.value

    def terrain_speed_modifier(self, terrain: str) -> float:
        """Speed multiplier for *terrain* type."""
        modifiers = {
            TerrainType.GRASS.value: 1.0,
            TerrainType.MUD.value: 0.6,
            TerrainType.SAND.value: 0.8,
            TerrainType.WATER.value: 0.3,
        }
        return modifiers.get(terrain, 1.0)

    # --- Tick -----------------------------------------------------------------

    def tick(self) -> None:
        """Advance world state by one tick (food respawning)."""
        self._tick_count += 1
        for food in self.food_sources:
            food.tick_respawn(self._food_respawn_interval, rng=self._rng)

    # --- Queries --------------------------------------------------------------

    def food_at(self, pos: Vec2) -> Optional[FoodSource]:
        """First non-depleted food source containing *pos*, or ``None``."""
        for food in self.food_sources:
            if not food.depleted and food.contains_point(pos):
                return food
        return None

    def obstacle_at(self, pos: Vec2) -> Optional[Obstacle]:
        """Obstacle containing *pos*, or ``None``."""
        for obs in self.obstacles:
            if obs.contains_point(pos):
                return obs
        return None

    def raycast(self, origin: Vec2, direction: Vec2, max_dist: float) -> float:
        """Distance to nearest obstacle along ray, or *max_dist* if none."""
        d = direction.normalized()
        closest = max_dist
        for obs in self.obstacles:
            t = obs.ray_intersection(origin, d)
            if t is not None and t < closest:
                closest = t
        return closest

    # --- Factory --------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: SimConfig, seed: int | None = None) -> World:
        """Create a fully-populated World from *cfg* with random placement."""
        wc = cfg.world
        rng = random.Random(seed)

        nest = NestRegion(
            center=Vec2(wc.width * 0.125, wc.height * 0.5),
            radius=40.0,
        )

        world = cls(
            width=wc.width,
            height=wc.height,
            nest=nest,
            _terrain_seed=seed if seed is not None else 42,
            _food_respawn_interval=wc.food_respawn_interval,
        )

        # -- obstacles (clamped to 5-15) --
        num_obs = max(5, min(15, wc.num_obstacles))
        margin = 60
        for _ in range(num_obs):
            for _attempt in range(50):
                cx = rng.uniform(margin, wc.width - margin)
                cy = rng.uniform(margin, wc.height - margin)
                center = Vec2(cx, cy)
                if center.distance_to(nest.center) < nest.radius + 80:
                    continue
                verts = generate_convex_polygon(center, 15, 40, rng=rng)
                world.obstacles.append(Obstacle(vertices=verts))
                break

        # -- food sources (clamped to 3-6) --
        num_food = max(3, min(6, wc.num_food_sources))
        for _ in range(num_food):
            for _attempt in range(50):
                fx = rng.uniform(margin, wc.width - margin)
                fy = rng.uniform(margin, wc.height - margin)
                fpos = Vec2(fx, fy)
                if fpos.distance_to(nest.center) < nest.radius + 60:
                    continue
                if any(o.contains_point(fpos) for o in world.obstacles):
                    continue
                radius = rng.uniform(
                    wc.food_source_size_range[0], wc.food_source_size_range[1],
                )
                amount = rng.uniform(
                    wc.food_source_amount_range[0], wc.food_source_amount_range[1],
                )
                world.food_sources.append(
                    FoodSource(pos=fpos, radius=radius, amount=amount, max_amount=amount),
                )
                break

        return world
