"""SensoryInput dataclass and sensory builder — what an ant perceives each tick."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from agents.ant import Vec2

if TYPE_CHECKING:
    from agents.ant import Ant
    from config import AntConfig
    from world.world import World
    from world.pheromone import PheromoneGrid


@dataclass
class NeighborInfo:
    """Compact info about a nearby ant."""
    id: int
    relative_pos: Vec2
    role_value: str          # Role.value string
    carrying: Optional[str]


@dataclass
class SensoryInput:
    # Antennae readings: per-channel pheromone concentration (left / right)
    antenna_left: dict[str, float] = field(default_factory=dict)
    antenna_right: dict[str, float] = field(default_factory=dict)

    # Obstacle raycasts — distances (normalized 0..1, 1 = max range = no hit)
    obstacle_rays: list[float] = field(default_factory=list)

    # Nest info
    nest_direction: Vec2 = field(default_factory=Vec2)   # unit vector toward nest
    nest_distance: float = 0.0                           # pixels, unnormalized
    nest_bearing: float = 0.0                            # signed angle from heading to nest (-π..π)

    # Neighbors
    neighbors: list[NeighborInfo] = field(default_factory=list)

    # Food pheromone gradient at current position
    food_gradient: Vec2 = field(default_factory=Vec2)

    # Direct food sensing (not pheromone-based)
    food_nearby: float = 0.0           # 1.0 if within pickup range, 0.0 otherwise
    nearest_food_bearing: float = 0.0  # signed angle from heading to nearest food (-1..1)
    nearest_food_distance: float = 1.0 # distance to nearest food (0..1, 1 = out of range)

    # Ground / terrain type at current position
    ground_type: str = "grass"

    # Self state (copied for convenience so brains don't need Ant reference)
    energy: float = 100.0
    carrying: Optional[str] = None
    carry_amount: float = 0.0
    role_value: str = "forager"
    age: int = 0
    speed: float = 2.0

    def to_vector(self) -> list[float]:
        """Encode sensory input as a flat float vector for neural brains.

        Layout (39 floats):
          [0..7]   antenna left/right × 4 channels (food, home, danger, recruit)
          [8..12]  obstacle raycasts (5 values)
          [13..14] nest direction (sin, cos)
          [15]     nest distance (normalized by world diagonal)
          [16]     energy (normalized 0..1)
          [17..20] carrying one-hot (none, food, brood, corpse)
          [21..24] role one-hot (forager, nurse, soldier, idle)
          [25]     neighbor count (clamped 0..10, normalized)
          [26..27] neighbor avg relative position (x, y normalized)
          [28..29] food gradient (x, y)
          [30..33] ground type one-hot (grass, mud, sand, water)
          [34]     age (normalized, /10000)
          [35]     speed (normalized, /5)
          [36]     food_nearby (binary: 1.0 if within pickup range)
          [37]     nearest_food_bearing (normalized -1..1)
          [38]     nearest_food_distance (normalized 0..1)
        """
        channels = ["food", "home", "danger", "recruit"]
        vec: list[float] = []

        # Antenna readings (8)
        for ch in channels:
            vec.append(self.antenna_left.get(ch, 0.0))
        for ch in channels:
            vec.append(self.antenna_right.get(ch, 0.0))

        # Obstacle rays (5) — pad/truncate to 5
        rays = self.obstacle_rays[:5]
        rays += [1.0] * (5 - len(rays))
        vec.extend(rays)

        # Nest direction (2)
        vec.append(self.nest_direction.x)
        vec.append(self.nest_direction.y)

        # Nest distance normalized (1)
        vec.append(min(self.nest_distance / 2000.0, 1.0))

        # Energy (1)
        vec.append(self.energy / 100.0)

        # Carrying one-hot (4)
        carry_map = {None: 0, "food": 1, "brood": 2, "corpse": 3}
        carry_idx = carry_map.get(self.carrying, 0)
        vec.extend([1.0 if i == carry_idx else 0.0 for i in range(4)])

        # Role one-hot (4)
        role_map = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}
        role_idx = role_map.get(self.role_value, 0)
        vec.extend([1.0 if i == role_idx else 0.0 for i in range(4)])

        # Neighbors (3)
        n_count = min(len(self.neighbors), 10)
        vec.append(n_count / 10.0)
        if self.neighbors:
            avg_x = sum(n.relative_pos.x for n in self.neighbors) / len(self.neighbors)
            avg_y = sum(n.relative_pos.y for n in self.neighbors) / len(self.neighbors)
            vec.append(max(-1.0, min(1.0, avg_x / 50.0)))
            vec.append(max(-1.0, min(1.0, avg_y / 50.0)))
        else:
            vec.extend([0.0, 0.0])

        # Food gradient (2)
        vec.append(self.food_gradient.x)
        vec.append(self.food_gradient.y)

        # Ground type one-hot (4)
        ground_map = {"grass": 0, "mud": 1, "sand": 2, "water": 3}
        ground_idx = ground_map.get(self.ground_type, 0)
        vec.extend([1.0 if i == ground_idx else 0.0 for i in range(4)])

        # Age and speed (2)
        vec.append(min(self.age / 10000.0, 1.0))
        vec.append(min(self.speed / 5.0, 1.0))

        # Direct food sensing (3)
        vec.append(self.food_nearby)
        vec.append(self.nearest_food_bearing)
        vec.append(self.nearest_food_distance)

        return vec


# ---------------------------------------------------------------------------
# Sensory builder — populate SensoryInput from world state
# ---------------------------------------------------------------------------

_CHANNELS = ["food", "home", "danger", "recruit"]


def _sample_antenna_cone(
    pos: Vec2,
    heading: float,
    offset_angle: float,
    antenna_range: float,
    pheromone_grid: PheromoneGrid,
) -> dict[str, float]:
    """Sample pheromone at the tip of an antenna cone.

    The antenna points at ``heading + offset_angle`` and reads from a point
    ``antenna_range`` pixels ahead using bilinear interpolation on the grid.
    """
    tip_angle = heading + offset_angle
    tip = pos + Vec2.from_angle(tip_angle, antenna_range)
    return pheromone_grid.sample_all(tip.x, tip.y)


def _cast_obstacle_rays(
    pos: Vec2,
    heading: float,
    ray_count: int,
    ray_range: float,
    ray_arc_rad: float,
    world: World,
) -> list[float]:
    """Cast *ray_count* rays in a forward arc and return normalised distances.

    Rays are evenly distributed across *ray_arc_rad* centred on *heading*.
    Return values are in [0, 1] where 1 means no obstacle within range.
    """
    rays: list[float] = []
    half_arc = ray_arc_rad / 2.0
    if ray_count <= 1:
        angles = [heading]
    else:
        angles = [
            heading - half_arc + i * ray_arc_rad / (ray_count - 1)
            for i in range(ray_count)
        ]
    for angle in angles:
        direction = Vec2.from_angle(angle)
        dist = world.raycast(pos, direction, ray_range)
        rays.append(dist / ray_range)
    return rays


def _find_neighbors(
    ant: Ant,
    all_ants: list[Ant],
    radius: float,
) -> list[NeighborInfo]:
    """Find alive ants within *radius* of *ant* (excluding self)."""
    radius_sq = radius * radius
    neighbors: list[NeighborInfo] = []
    for other in all_ants:
        if other.id == ant.id or not other.alive:
            continue
        rel = other.pos - ant.pos
        if rel.length_sq() <= radius_sq:
            neighbors.append(NeighborInfo(
                id=other.id,
                relative_pos=rel,
                role_value=other.role.value,
                carrying=other.carrying,
            ))
    return neighbors


def _compute_food_gradient(
    pos: Vec2,
    pheromone_grid: PheromoneGrid,
) -> Vec2:
    """Approximate food pheromone gradient via central differences."""
    from world.pheromone import Channel

    delta = float(pheromone_grid.cell_size)
    gx = (
        pheromone_grid.sample(pos.x + delta, pos.y, Channel.FOOD)
        - pheromone_grid.sample(pos.x - delta, pos.y, Channel.FOOD)
    ) / (2.0 * delta)
    gy = (
        pheromone_grid.sample(pos.x, pos.y + delta, Channel.FOOD)
        - pheromone_grid.sample(pos.x, pos.y - delta, Channel.FOOD)
    ) / (2.0 * delta)
    return Vec2(gx, gy)


def _find_nearest_food(
    ant: Ant,
    world: World,
    sense_range: float = 300.0,
) -> tuple[float, float, float]:
    """Find the nearest non-depleted food source within *sense_range*.

    Returns:
        (food_nearby, bearing_normalized, distance_normalized)
        - food_nearby: 1.0 if ant is inside a food source's radius, else 0.0
        - bearing_normalized: signed angle from heading to food, in [-1, 1]
        - distance_normalized: distance / sense_range, clamped to [0, 1]
    """
    best_dist = float("inf")
    best_food = None

    for food in world.food_sources:
        if food.depleted:
            continue
        d = ant.pos.distance_to(food.pos)
        if d < best_dist:
            best_dist = d
            best_food = food

    if best_food is None or best_dist > sense_range:
        return 0.0, 0.0, 1.0

    # Is the ant standing on the food?
    food_nearby = 1.0 if best_food.contains_point(ant.pos) else 0.0

    # Bearing: signed angle from ant heading to food direction
    to_food = best_food.pos - ant.pos
    food_angle = math.atan2(to_food.y, to_food.x)
    bearing = food_angle - ant.heading
    # Normalize to [-π, π]
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
    bearing_normalized = max(-1.0, min(1.0, bearing / math.pi))

    distance_normalized = min(best_dist / sense_range, 1.0)

    return food_nearby, bearing_normalized, distance_normalized


def build_sensory(
    ant: Ant,
    world: World,
    pheromone_grid: PheromoneGrid,
    all_ants: list[Ant],
    cfg: AntConfig,
) -> SensoryInput:
    """Build a complete SensoryInput for *ant* from current world state."""
    antenna_angle_rad = cfg.antenna_angle_rad

    # Antenna cone sampling (left = +offset, right = -offset)
    antenna_left = _sample_antenna_cone(
        ant.pos, ant.heading, antenna_angle_rad, cfg.antenna_range, pheromone_grid,
    )
    antenna_right = _sample_antenna_cone(
        ant.pos, ant.heading, -antenna_angle_rad, cfg.antenna_range, pheromone_grid,
    )

    # Obstacle raycasts
    obstacle_rays = _cast_obstacle_rays(
        ant.pos, ant.heading,
        cfg.obstacle_ray_count, cfg.obstacle_ray_range, cfg.obstacle_ray_arc_rad,
        world,
    )

    # Nest direction and distance
    to_nest = world.nest.center - ant.pos
    nest_distance = to_nest.length()
    nest_direction = to_nest.normalized() if nest_distance > 1e-6 else Vec2(0.0, 0.0)

    # Nest bearing: signed angle from ant heading to nest direction
    nest_angle = math.atan2(to_nest.y, to_nest.x)
    nest_bearing = nest_angle - ant.heading
    # Normalize to [-π, π]
    nest_bearing = (nest_bearing + math.pi) % (2 * math.pi) - math.pi

    # Neighbors
    neighbors = _find_neighbors(ant, all_ants, cfg.neighbor_radius)

    # Food gradient
    food_gradient = _compute_food_gradient(ant.pos, pheromone_grid)

    # Ground type
    ground_type = world.get_terrain(ant.pos)

    # Direct food sensing
    food_nearby, nearest_food_bearing, nearest_food_distance = _find_nearest_food(
        ant, world,
    )

    return SensoryInput(
        antenna_left=antenna_left,
        antenna_right=antenna_right,
        obstacle_rays=obstacle_rays,
        nest_direction=nest_direction,
        nest_distance=nest_distance,
        nest_bearing=nest_bearing,
        neighbors=neighbors,
        food_gradient=food_gradient,
        food_nearby=food_nearby,
        nearest_food_bearing=nearest_food_bearing,
        nearest_food_distance=nearest_food_distance,
        ground_type=ground_type,
        energy=ant.energy,
        carrying=ant.carrying,
        carry_amount=ant.carry_amount,
        role_value=ant.role.value,
        age=ant.age,
        speed=ant.speed,
    )
