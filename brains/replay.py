"""Prioritized experience replay buffer.

Implements TD-error prioritized sampling with importance-sampling weights
for off-policy correction. Experiences with higher TD-error are sampled
more frequently, focusing learning on surprising transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayExperience:
    """Single experience for replay."""
    state: np.ndarray
    action_idx: int         # flattened action index or action hash
    reward: float
    next_state: np.ndarray
    done: bool
    td_error: float = 1.0  # initial priority


class PrioritizedReplayBuffer:
    """TD-error prioritized experience replay.

    Uses a sum-tree internally for efficient O(log n) sampling proportional
    to priority. For simplicity, we use a flat array with renormalization
    which is O(n) but sufficient for moderate buffer sizes.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        epsilon: float = 1e-6,
    ) -> None:
        self._capacity = capacity
        self._alpha = alpha        # priority exponent (0 = uniform, 1 = full prioritization)
        self._beta = beta          # importance-sampling exponent
        self._beta_increment = beta_increment
        self._epsilon = epsilon    # small constant added to TD errors
        self._buffer: list[ReplayExperience] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self._pos: int = 0
        self._max_priority: float = 1.0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def full(self) -> bool:
        return len(self._buffer) >= self._capacity

    def add(self, exp: ReplayExperience, td_error: float | None = None) -> None:
        """Add experience with priority based on TD error."""
        priority = (abs(td_error) + self._epsilon) ** self._alpha if td_error is not None \
            else self._max_priority

        if len(self._buffer) < self._capacity:
            self._buffer.append(exp)
        else:
            self._buffer[self._pos] = exp

        self._priorities[self._pos] = priority
        self._max_priority = max(self._max_priority, priority)
        self._pos = (self._pos + 1) % self._capacity

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None,
    ) -> tuple[list[ReplayExperience], np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Returns:
            experiences: list of sampled experiences
            weights: importance-sampling weights (batch_size,)
            indices: indices into buffer for priority updates
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(self._buffer)
        if n == 0:
            return [], np.array([]), np.array([], dtype=int)

        batch_size = min(batch_size, n)

        # Compute sampling probabilities
        priorities = self._priorities[:n]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = rng.choice(n, size=batch_size, replace=False, p=probs)

        # Importance-sampling weights
        self._beta = min(1.0, self._beta + self._beta_increment)
        weights = (n * probs[indices]) ** (-self._beta)
        weights /= weights.max()  # normalize

        experiences = [self._buffer[i] for i in indices]
        return experiences, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self._epsilon) ** self._alpha
            self._priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
