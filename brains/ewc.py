"""Elastic Weight Consolidation for continual learning.

Protects important weights from catastrophic forgetting by adding a penalty
term that penalizes changes to weights that were important for previous tasks.
The importance is estimated via the diagonal of the Fisher information matrix.
"""

from __future__ import annotations

import numpy as np


class EWC:
    """Elastic Weight Consolidation.

    After a task is learned, call :meth:`consolidate` to compute Fisher
    information and snapshot the weights. During subsequent training, call
    :meth:`penalty_gradient` to get the regularization gradient that should
    be added to the main training gradient.
    """

    def __init__(self, lambda_ewc: float = 100.0) -> None:
        self._lambda = lambda_ewc
        self._star_params: list[np.ndarray] = []    # θ* snapshot
        self._fisher_diag: list[np.ndarray] = []    # F diagonal

    @property
    def has_consolidated(self) -> bool:
        return len(self._star_params) > 0

    def compute_fisher(
        self,
        weights: list[np.ndarray],
        log_prob_grads: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        """Compute diagonal Fisher information from sampled log-prob gradients.

        Args:
            weights: list of parameter arrays
            log_prob_grads: list of gradient-lists, one per sample.
                           Each inner list has same structure as weights.

        Returns:
            Fisher diagonal (list of arrays matching weights shapes)
        """
        n = len(log_prob_grads)
        if n == 0:
            return [np.zeros_like(w) for w in weights]

        fisher = [np.zeros_like(w) for w in weights]
        for grads in log_prob_grads:
            for f, g in zip(fisher, grads):
                f += g ** 2

        for f in fisher:
            f /= n

        return fisher

    def consolidate(
        self,
        weights: list[np.ndarray],
        log_prob_grads: list[list[np.ndarray]],
    ) -> None:
        """Snapshot weights and compute Fisher information.

        Call this after training on a task to protect learned knowledge.

        Args:
            weights: current parameter arrays
            log_prob_grads: sampled log-prob gradients for Fisher estimation
        """
        self._star_params = [w.copy() for w in weights]
        self._fisher_diag = self.compute_fisher(weights, log_prob_grads)

    def penalty_gradient(self, weights: list[np.ndarray]) -> list[np.ndarray]:
        """Compute EWC penalty gradient: λ * F * (θ - θ*).

        Add this to the training gradient to regularize toward previous solution.
        """
        if not self.has_consolidated:
            return [np.zeros_like(w) for w in weights]

        grads = []
        for w, w_star, f in zip(weights, self._star_params, self._fisher_diag):
            grads.append(self._lambda * f * (w - w_star))
        return grads

    def penalty_loss(self, weights: list[np.ndarray]) -> float:
        """Compute EWC penalty loss: 0.5 * λ * Σ F_i * (θ_i - θ*_i)^2."""
        if not self.has_consolidated:
            return 0.0

        loss = 0.0
        for w, w_star, f in zip(weights, self._star_params, self._fisher_diag):
            loss += float(0.5 * self._lambda * np.sum(f * (w - w_star) ** 2))
        return loss
