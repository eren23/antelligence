"""Convex polygon obstacles with collision detection and response."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from agents.ant import Vec2


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """Convex polygon obstacle with CCW winding order."""

    vertices: list[Vec2]
    _edges: list[tuple[Vec2, Vec2]] = field(default=None, init=False, repr=False)
    _normals: list[Vec2] = field(default=None, init=False, repr=False)
    _center: Vec2 = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("Obstacle must have at least 3 vertices")
        self._compute_cache()

    def _compute_cache(self) -> None:
        n = len(self.vertices)
        self._edges = []
        self._normals = []
        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]
            self._edges.append((a, b))
            # Outward normal for CCW winding: right perpendicular of edge
            edge = b - a
            normal = Vec2(edge.y, -edge.x).normalized()
            self._normals.append(normal)
        cx = sum(v.x for v in self.vertices) / n
        cy = sum(v.y for v in self.vertices) / n
        self._center = Vec2(cx, cy)

    @property
    def center(self) -> Vec2:
        return self._center

    def contains_point(self, point: Vec2) -> bool:
        """Test if *point* is inside this convex polygon.

        Uses cross-product sign test: for CCW winding, a point is inside
        iff it is to the left of (or on) every directed edge.
        """
        n = len(self.vertices)
        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]
            cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
            if cross < 0:
                return False
        return True

    def ray_intersection(self, origin: Vec2, direction: Vec2) -> Optional[float]:
        """Closest intersection distance of a ray with this polygon.

        Args:
            origin: Ray start point.
            direction: Ray direction (need not be unit-length).

        Returns:
            Distance parameter *t* along *direction*, or ``None``.
        """
        if direction.length_sq() == 0:
            return None

        closest_t: Optional[float] = None
        for a, b in self._edges:
            t = _ray_segment_intersection(origin, direction, a, b)
            if t is not None and t >= 0:
                if closest_t is None or t < closest_t:
                    closest_t = t
        return closest_t

    def closest_edge_info(self, point: Vec2) -> tuple[int, float, Vec2]:
        """Find the edge nearest to *point*.

        Returns:
            ``(edge_index, distance, closest_point_on_edge)``
        """
        best_idx = 0
        best_dist = float("inf")
        best_point = self.vertices[0]

        for i, (a, b) in enumerate(self._edges):
            cp = _closest_point_on_segment(point, a, b)
            d = point.distance_to(cp)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_point = cp

        return best_idx, best_dist, best_point

    def slide_along_edge(self, pos: Vec2, velocity: Vec2) -> tuple[Vec2, Vec2]:
        """Collision response: push *pos* out and slide *velocity* along edge.

        Args:
            pos: Position inside or near the obstacle.
            velocity: Desired movement vector.

        Returns:
            ``(new_position, adjusted_velocity)``
        """
        edge_idx, dist, closest_pt = self.closest_edge_info(pos)
        normal = self._normals[edge_idx]

        # Push position outside along the outward normal
        push_dist = max(0.5, dist)
        new_pos = closest_pt + normal * push_dist

        # Project velocity onto edge tangent
        a, b = self._edges[edge_idx]
        edge_dir = (b - a).normalized()
        projected_speed = velocity.dot(edge_dir)
        new_vel = edge_dir * projected_speed

        return new_pos, new_vel


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ray_segment_intersection(
    origin: Vec2, direction: Vec2, seg_a: Vec2, seg_b: Vec2,
) -> Optional[float]:
    """Ray–segment intersection via parametric solve.

    Returns ray parameter *t* (distance along *direction*), or ``None``
    if the ray and segment are parallel or do not intersect.
    """
    dx, dy = direction.x, direction.y
    sx = seg_b.x - seg_a.x
    sy = seg_b.y - seg_a.y

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:
        return None  # parallel

    ox = seg_a.x - origin.x
    oy = seg_a.y - origin.y

    t = (ox * sy - oy * sx) / denom   # ray parameter
    u = (ox * dy - oy * dx) / denom   # segment parameter

    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def _closest_point_on_segment(point: Vec2, seg_a: Vec2, seg_b: Vec2) -> Vec2:
    """Closest point on segment *seg_a*–*seg_b* to *point*."""
    ab = seg_b - seg_a
    ab_sq = ab.length_sq()
    if ab_sq < 1e-12:
        return seg_a
    t = max(0.0, min(1.0, (point - seg_a).dot(ab) / ab_sq))
    return seg_a + ab * t


# ---------------------------------------------------------------------------
# Random convex polygon generator
# ---------------------------------------------------------------------------

def generate_convex_polygon(
    center: Vec2,
    min_radius: float,
    max_radius: float,
    num_vertices: int = 0,
    rng: random.Random | None = None,
) -> list[Vec2]:
    """Generate a random convex polygon with guaranteed CCW winding.

    Uses a regular polygon with random rotation and axis scaling to
    ensure convexity (affine transforms preserve convexity).
    """
    if rng is None:
        rng = random.Random()
    if num_vertices <= 0:
        num_vertices = rng.randint(4, 8)

    base_radius = rng.uniform(min_radius, max_radius)
    rotation = rng.uniform(0, 2 * math.pi)
    scale_x = rng.uniform(0.7, 1.3)
    scale_y = rng.uniform(0.7, 1.3)

    vertices: list[Vec2] = []
    for i in range(num_vertices):
        angle = rotation + (2 * math.pi * i / num_vertices)
        x = center.x + base_radius * math.cos(angle) * scale_x
        y = center.y + base_radius * math.sin(angle) * scale_y
        vertices.append(Vec2(x, y))

    return vertices
