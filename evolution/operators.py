"""Evolutionary operators: mutation, crossover, and selection.

Operates on lists of numpy weight arrays (compatible with both MLPWeights
and TransformerWeightSet parameter lists).
"""

from __future__ import annotations

import numpy as np


def mutate_weights(
    weights: list[np.ndarray],
    rate: float = 0.1,
    scale: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Gaussian mutation on a subset of weights.

    Each weight element has `rate` probability of being perturbed by
    Gaussian noise with standard deviation `scale`.

    Returns a new list of weight arrays (copies).
    """
    if rng is None:
        rng = np.random.default_rng()

    mutated = []
    for w in weights:
        w_new = w.copy()
        mask = rng.random(w.shape) < rate
        w_new[mask] += rng.normal(0, scale, w.shape)[mask]
        mutated.append(w_new)
    return mutated


def crossover_weights(
    parent_a: list[np.ndarray],
    parent_b: list[np.ndarray],
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Uniform crossover between two weight sets.

    For each weight element, randomly selects from parent_a or parent_b.

    Returns a new list of weight arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    child = []
    for wa, wb in zip(parent_a, parent_b):
        mask = rng.random(wa.shape) < 0.5
        w_new = np.where(mask, wa, wb)
        child.append(w_new)
    return child


def tournament_select(
    fitnesses: list[float],
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> int:
    """Tournament selection. Returns index of winner.

    Randomly selects `tournament_size` individuals and returns the one
    with the highest fitness.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(fitnesses)
    tournament_size = min(tournament_size, n)
    candidates = rng.choice(n, size=tournament_size, replace=False)
    best_idx = candidates[0]
    best_fit = fitnesses[candidates[0]]
    for idx in candidates[1:]:
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_idx = idx
    return int(best_idx)
