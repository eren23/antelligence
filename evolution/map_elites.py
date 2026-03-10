"""MAP-Elites quality-diversity archive.

Maintains a 2D grid of behavioral niches. Each cell stores the best
individual found for that behavioral region. This promotes diverse
behavioral strategies rather than convergence to a single optimum.

Default behavior dimensions: (foraging_efficiency, trail_formation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolution.population import Individual


class MAPElitesArchive:
    """Quality-Diversity archive using MAP-Elites.

    2D grid indexed by behavior descriptors. Each cell holds the best
    individual discovered for that behavioral niche.
    """

    def __init__(self, dims: tuple[int, int] = (10, 10)) -> None:
        self._dims = dims
        self._grid: list[list[Individual | None]] = [
            [None for _ in range(dims[1])]
            for _ in range(dims[0])
        ]
        self._insertions: int = 0

    @property
    def dims(self) -> tuple[int, int]:
        return self._dims

    def _to_cell(self, behavior: np.ndarray) -> tuple[int, int]:
        """Map a 2D behavior descriptor ([0,1] × [0,1]) to grid cell."""
        r = max(0, min(self._dims[0] - 1, int(behavior[0] * self._dims[0])))
        c = max(0, min(self._dims[1] - 1, int(behavior[1] * self._dims[1])))
        return r, c

    def try_insert(self, individual: Individual) -> bool:
        """Try to insert individual into the archive.

        Succeeds if the cell is empty or the individual has higher fitness
        than the current occupant.

        Returns True if inserted.
        """
        r, c = self._to_cell(individual.behavior_descriptor)

        current = self._grid[r][c]
        if current is None or individual.fitness > current.fitness:
            self._grid[r][c] = individual
            self._insertions += 1
            return True
        return False

    def sample_parent(self, rng: np.random.Generator) -> Individual | None:
        """Uniformly sample an individual from occupied cells."""
        occupied = []
        for r in range(self._dims[0]):
            for c in range(self._dims[1]):
                if self._grid[r][c] is not None:
                    occupied.append(self._grid[r][c])
        if not occupied:
            return None
        return occupied[int(rng.integers(len(occupied)))]

    def coverage(self) -> float:
        """Fraction of cells that are occupied."""
        filled = sum(
            1 for r in range(self._dims[0])
            for c in range(self._dims[1])
            if self._grid[r][c] is not None
        )
        total = self._dims[0] * self._dims[1]
        return filled / total if total > 0 else 0.0

    def best_fitness(self) -> float:
        """Return the highest fitness in the archive."""
        best = float("-inf")
        for r in range(self._dims[0]):
            for c in range(self._dims[1]):
                ind = self._grid[r][c]
                if ind is not None and ind.fitness > best:
                    best = ind.fitness
        return best if best > float("-inf") else 0.0

    def all_individuals(self) -> list[Individual]:
        """Return all individuals in the archive."""
        result = []
        for r in range(self._dims[0]):
            for c in range(self._dims[1]):
                if self._grid[r][c] is not None:
                    result.append(self._grid[r][c])
        return result

    @property
    def total_insertions(self) -> int:
        return self._insertions
