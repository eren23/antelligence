"""Spatial grid for O(n) neighbor lookup.

Replaces the O(n^2) brute-force neighbor search in sensory.py with a uniform
grid that supports O(k) radius queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.ant import Ant, Vec2


class SpatialGrid:
    """Uniform spatial grid for fast neighbor queries.

    Parameters:
        width: World width in pixels.
        height: World height in pixels.
        cell_size: Grid cell size (should be >= neighbor_radius for correctness).
    """

    __slots__ = ("_cell_size", "_cols", "_rows", "_cells")

    def __init__(self, width: int, height: int, cell_size: int = 40) -> None:
        self._cell_size = cell_size
        self._cols = max(1, width // cell_size + 1)
        self._rows = max(1, height // cell_size + 1)
        self._cells: list[list[Ant]] = [[] for _ in range(self._cols * self._rows)]

    def _cell_index(self, x: float, y: float) -> int:
        col = max(0, min(int(x) // self._cell_size, self._cols - 1))
        row = max(0, min(int(y) // self._cell_size, self._rows - 1))
        return row * self._cols + col

    def rebuild(self, ants: list[Ant]) -> None:
        """Rebuild the grid from scratch. O(n)."""
        for cell in self._cells:
            cell.clear()
        for ant in ants:
            if ant.alive:
                idx = self._cell_index(ant.pos.x, ant.pos.y)
                self._cells[idx].append(ant)

    def query_radius(self, pos: Vec2, radius: float) -> list[Ant]:
        """Return all alive ants within *radius* of *pos*. O(k)."""
        cs = self._cell_size
        radius_sq = radius * radius

        # Compute cell range to search
        min_col = max(0, int(pos.x - radius) // cs)
        max_col = min(self._cols - 1, int(pos.x + radius) // cs)
        min_row = max(0, int(pos.y - radius) // cs)
        max_row = min(self._rows - 1, int(pos.y + radius) // cs)

        result: list[Ant] = []
        for row in range(min_row, max_row + 1):
            row_offset = row * self._cols
            for col in range(min_col, max_col + 1):
                for ant in self._cells[row_offset + col]:
                    dx = ant.pos.x - pos.x
                    dy = ant.pos.y - pos.y
                    if dx * dx + dy * dy <= radius_sq:
                        result.append(ant)
        return result
