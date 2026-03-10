"""Population manager for evolutionary training.

Manages a population of brain individuals, handles evaluation, selection,
and reproduction across generations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolution.genome import ArchitectureGenome
from evolution.operators import mutate_weights, crossover_weights, tournament_select


@dataclass
class Individual:
    """A single brain individual in the population."""
    id: int
    weights: list[np.ndarray]              # flattened parameter arrays
    genome: ArchitectureGenome
    fitness: float = 0.0
    behavior_descriptor: np.ndarray = field(
        default_factory=lambda: np.zeros(2),
    )  # from emergence report (e.g., foraging_efficiency, trail_formation)


class PopulationManager:
    """Manages a population of brain individuals across generations.

    Handles initialization, evaluation orchestration, selection, and
    reproduction with elitism.
    """

    def __init__(
        self,
        size: int = 20,
        eval_ticks: int = 5000,
        elitism: int = 4,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.02,
        crossover_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._size = size
        self._eval_ticks = eval_ticks
        self._elitism = min(elitism, size)
        self._tournament_size = tournament_size
        self._mutation_rate = mutation_rate
        self._mutation_scale = mutation_scale
        self._crossover_prob = crossover_prob
        self._rng = np.random.default_rng(seed)
        self._population: list[Individual] = []
        self._generation: int = 0
        self._next_id: int = 0

    @property
    def population(self) -> list[Individual]:
        return self._population

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def size(self) -> int:
        return self._size

    @property
    def eval_ticks(self) -> int:
        return self._eval_ticks

    def initialize(self, weight_factory: Any) -> None:
        """Initialize population with random weights.

        Args:
            weight_factory: callable() -> list[np.ndarray] that produces
                           a new random set of weights.
        """
        self._population = []
        for _ in range(self._size):
            weights = weight_factory()
            ind = Individual(
                id=self._next_id,
                weights=weights,
                genome=ArchitectureGenome(),
            )
            self._next_id += 1
            self._population.append(ind)

    def set_fitness(self, idx: int, fitness: float, behavior: np.ndarray | None = None) -> None:
        """Set fitness and optional behavior descriptor for individual at index."""
        self._population[idx].fitness = fitness
        if behavior is not None:
            self._population[idx].behavior_descriptor = behavior

    def select_and_reproduce(self) -> None:
        """Create next generation via tournament selection + crossover + mutation.

        Top `elitism` individuals are carried over unchanged.
        """
        self._generation += 1

        # Sort by fitness (descending)
        sorted_pop = sorted(self._population, key=lambda i: i.fitness, reverse=True)

        # Elite carry-over
        new_pop: list[Individual] = []
        for ind in sorted_pop[:self._elitism]:
            elite = Individual(
                id=self._next_id,
                weights=[w.copy() for w in ind.weights],
                genome=ind.genome,
                fitness=0.0,
            )
            self._next_id += 1
            new_pop.append(elite)

        # Fill rest via selection + reproduction
        fitnesses = [ind.fitness for ind in self._population]
        while len(new_pop) < self._size:
            parent_a_idx = tournament_select(fitnesses, self._tournament_size, self._rng)
            parent_a = self._population[parent_a_idx]

            if self._rng.random() < self._crossover_prob and self._size > 1:
                parent_b_idx = tournament_select(fitnesses, self._tournament_size, self._rng)
                parent_b = self._population[parent_b_idx]
                child_weights = crossover_weights(
                    parent_a.weights, parent_b.weights, self._rng,
                )
                child_genome = ArchitectureGenome.crossover(
                    parent_a.genome, parent_b.genome, self._rng,
                )
            else:
                child_weights = [w.copy() for w in parent_a.weights]
                child_genome = parent_a.genome

            # Mutation
            child_weights = mutate_weights(
                child_weights, self._mutation_rate, self._mutation_scale, self._rng,
            )
            child_genome = child_genome.mutate(self._rng, self._mutation_rate)

            child = Individual(
                id=self._next_id,
                weights=child_weights,
                genome=child_genome,
                fitness=0.0,
            )
            self._next_id += 1
            new_pop.append(child)

        self._population = new_pop

    def get_best(self) -> Individual:
        """Return the individual with highest fitness."""
        return max(self._population, key=lambda i: i.fitness)

    def stats(self) -> dict[str, float]:
        """Return population statistics."""
        fits = [i.fitness for i in self._population]
        return {
            "generation": self._generation,
            "best_fitness": max(fits),
            "mean_fitness": sum(fits) / len(fits),
            "min_fitness": min(fits),
            "std_fitness": float(np.std(fits)),
        }
