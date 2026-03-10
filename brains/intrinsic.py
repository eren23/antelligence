"""Intrinsic motivation modules for exploration.

Provides:
  - RNDExplorer: Random Network Distillation curiosity via prediction error
  - IntrinsicCombiner: Blends extrinsic + intrinsic rewards with decaying coefficient
"""

from __future__ import annotations

import math

import numpy as np


class RNDExplorer:
    """Random Network Distillation — curiosity via prediction error.

    A fixed random target network and a trainable predictor network are both
    simple 2-layer MLPs. The intrinsic reward is the MSE between their outputs:
    novel states produce high error, familiar states produce low error.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden: int = 32,
        output_dim: int = 16,
        lr: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / hidden)

        # Fixed target network (never trained)
        self._target_W1 = rng.normal(0, s1, (input_dim, hidden))
        self._target_b1 = np.zeros(hidden)
        self._target_W2 = rng.normal(0, s2, (hidden, output_dim))
        self._target_b2 = np.zeros(output_dim)

        # Trainable predictor network
        self._pred_W1 = rng.normal(0, s1, (input_dim, hidden))
        self._pred_b1 = np.zeros(hidden)
        self._pred_W2 = rng.normal(0, s2, (hidden, output_dim))
        self._pred_b2 = np.zeros(output_dim)

        self._lr = lr

        # Running statistics for normalizing intrinsic reward
        self._reward_mean: float = 0.0
        self._reward_var: float = 1.0
        self._reward_count: int = 0

    def _target_forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self._target_W1 + self._target_b1)
        return h @ self._target_W2 + self._target_b2

    def _pred_forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self._pred_W1 + self._pred_b1)
        return h @ self._pred_W2 + self._pred_b2

    def intrinsic_reward(self, state: np.ndarray) -> float:
        """Compute intrinsic reward for a state vector (39,)."""
        if state.ndim == 1:
            state = state[np.newaxis, :]

        target = self._target_forward(state)
        pred = self._pred_forward(state)
        mse = float(np.mean((target - pred) ** 2))

        # Update running stats for normalization
        self._reward_count += 1
        delta = mse - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = mse - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / max(self._reward_count, 2)

        # Normalize
        std = max(math.sqrt(abs(self._reward_var)), 1e-8)
        return (mse - self._reward_mean) / std

    def update(self, state: np.ndarray) -> None:
        """Train the predictor to match the target (reduces novelty for seen states)."""
        if state.ndim == 1:
            state = state[np.newaxis, :]

        # Forward predictor
        h_pred = np.maximum(0.0, state @ self._pred_W1 + self._pred_b1)
        pred = h_pred @ self._pred_W2 + self._pred_b2

        # Target (no grad)
        target = self._target_forward(state)

        # MSE gradient: d/d_pred = 2 * (pred - target) / output_dim
        output_dim = pred.shape[-1]
        dL_dpred = 2.0 * (pred - target) / output_dim
        batch = state.shape[0]

        # Backprop through predictor
        dW2 = h_pred.T @ dL_dpred / batch
        db2 = dL_dpred.mean(axis=0)
        dL_dh = dL_dpred @ self._pred_W2.T
        dL_dh *= (h_pred > 0).astype(float)  # ReLU derivative
        dW1 = state.T @ dL_dh / batch
        db1 = dL_dh.mean(axis=0)

        # SGD update
        self._pred_W1 -= self._lr * dW1
        self._pred_b1 -= self._lr * db1
        self._pred_W2 -= self._lr * dW2
        self._pred_b2 -= self._lr * db2


class IntrinsicCombiner:
    """Blends extrinsic + intrinsic rewards with decaying coefficient.

    combined = extrinsic + coef * intrinsic
    coef decays each call so exploration diminishes as training progresses.
    """

    def __init__(self, coef: float = 0.5, decay: float = 0.9999) -> None:
        self._coef = coef
        self._decay = decay

    @property
    def coef(self) -> float:
        return self._coef

    def combine(self, extrinsic: float, intrinsic: float) -> float:
        combined = extrinsic + self._coef * intrinsic
        self._coef *= self._decay
        return combined
