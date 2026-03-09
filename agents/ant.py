"""Ant agent dataclass, Vec2 helper, Role enum, and physics/movement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from brains.interface import BrainBackend
    from agents.sensory import SensoryInput
    from agents.actions import AntAction
    from config import AntConfig
    from world.world import World


# ---------------------------------------------------------------------------
# Vec2 — lightweight 2D vector helper
# ---------------------------------------------------------------------------

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> Vec2:
        ln = self.length()
        if ln == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / ln, self.y / ln)

    def distance_to(self, other: Vec2) -> float:
        return (self - other).length()

    def angle(self) -> float:
        """Return angle in radians from positive x-axis."""
        return math.atan2(self.y, self.x)

    @staticmethod
    def from_angle(radians: float, magnitude: float = 1.0) -> Vec2:
        return Vec2(math.cos(radians) * magnitude, math.sin(radians) * magnitude)

    def rotated(self, radians: float) -> Vec2:
        c, s = math.cos(radians), math.sin(radians)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Vec2({self.x:.2f}, {self.y:.2f})"


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class Role(Enum):
    FORAGER = "forager"
    NURSE = "nurse"
    SOLDIER = "soldier"
    IDLE = "idle"


# ---------------------------------------------------------------------------
# Ant dataclass
# ---------------------------------------------------------------------------

@dataclass
class Ant:
    id: int
    pos: Vec2 = field(default_factory=Vec2)
    heading: float = 0.0                    # radians
    speed: float = 2.0                      # pixels/tick (base)
    energy: float = 100.0                   # 0–100
    carrying: Optional[str] = None          # None | "food" | "brood" | "corpse"
    carry_amount: float = 0.0              # 0.0–1.0
    role: Role = Role.FORAGER
    age: int = 0                            # ticks alive
    alive: bool = True
    brain: Optional[BrainBackend] = field(default=None, repr=False)
    sensory: Optional[SensoryInput] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Physics / movement helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def update_heading(ant: Ant, action: AntAction) -> None:
    """Apply clamped turn delta to ant heading."""
    clamped = action.clamped()
    ant.heading = _wrap_angle(ant.heading + clamped.turn)


def advance_position(ant: Ant, action: AntAction, world: World) -> None:
    """Move ant forward by speed * speed_mult * terrain_mult."""
    clamped = action.clamped()
    terrain = world.get_terrain(ant.pos)
    terrain_mult = world.terrain_speed_modifier(terrain)
    dist = ant.speed * clamped.speed_mult * terrain_mult
    direction = Vec2.from_angle(ant.heading, dist)
    ant.pos = ant.pos + direction


def handle_obstacle_collision(ant: Ant, world: World) -> None:
    """If ant is inside an obstacle, slide along its edge."""
    obs = world.obstacle_at(ant.pos)
    if obs is not None:
        velocity = Vec2.from_angle(ant.heading, ant.speed)
        new_pos, new_vel = obs.slide_along_edge(ant.pos, velocity)
        ant.pos = new_pos
        if new_vel.length_sq() > 1e-12:
            ant.heading = new_vel.angle()


def handle_world_bounds(ant: Ant, world: World) -> None:
    """Reflect heading when ant hits world boundaries."""
    reflected = False
    dx = Vec2.from_angle(ant.heading)

    if ant.pos.x <= 0:
        ant.pos = Vec2(0.0, ant.pos.y)
        if dx.x < 0:
            dx = Vec2(-dx.x, dx.y)
            reflected = True
    elif ant.pos.x >= world.width:
        ant.pos = Vec2(float(world.width), ant.pos.y)
        if dx.x > 0:
            dx = Vec2(-dx.x, dx.y)
            reflected = True

    if ant.pos.y <= 0:
        ant.pos = Vec2(ant.pos.x, 0.0)
        if dx.y < 0:
            dx = Vec2(dx.x, -dx.y)
            reflected = True
    elif ant.pos.y >= world.height:
        ant.pos = Vec2(ant.pos.x, float(world.height))
        if dx.y > 0:
            dx = Vec2(dx.x, -dx.y)
            reflected = True

    if reflected:
        ant.heading = dx.angle()


def deplete_energy(ant: Ant, action: AntAction, cfg: AntConfig) -> None:
    """Subtract energy: base + speed component + carry component."""
    clamped = action.clamped()
    cost = cfg.energy_cost_base + cfg.energy_cost_speed * clamped.speed_mult
    if ant.carrying is not None:
        cost += cfg.energy_cost_carry
    ant.energy = max(0.0, ant.energy - cost)


def check_death(ant: Ant) -> bool:
    """If energy <= 0 the ant dies and becomes a corpse marker.

    Returns True if the ant just died.
    """
    if ant.alive and ant.energy <= 0:
        ant.alive = False
        ant.carrying = None
        ant.carry_amount = 0.0
        ant.speed = 0.0
        return True
    return False


def refill_at_nest(ant: Ant, world: World, cfg: AntConfig) -> Optional[str]:
    """Refill energy if inside nest. Returns 'food' if food was deposited.

    If carrying food, automatically deposits it.
    """
    if not world.nest.contains(ant.pos):
        return None
    ant.energy = min(cfg.energy_max, ant.energy + cfg.energy_refill_rate)

    if ant.carrying == "food":
        deposited = ant.carrying
        ant.carrying = None
        carry = ant.carry_amount
        ant.carry_amount = 0.0
        return deposited
    return None


def try_pickup(ant: Ant, world: World, cfg: AntConfig) -> bool:
    """Attempt to pick up food at current position.

    Returns True if something was picked up.
    """
    if ant.carrying is not None:
        return False
    food = world.food_at(ant.pos)
    if food is not None:
        taken = food.deplete(cfg.max_carry)
        if taken > 0:
            ant.carrying = "food"
            ant.carry_amount = min(taken / cfg.max_carry, 1.0)
            return True
    return False


def try_drop(ant: Ant) -> Optional[tuple[str, float]]:
    """Drop whatever the ant is carrying.

    Returns (item_type, amount) if something was dropped, else None.
    """
    if ant.carrying is None:
        return None
    item = ant.carrying
    amount = ant.carry_amount
    ant.carrying = None
    ant.carry_amount = 0.0
    return (item, amount)
