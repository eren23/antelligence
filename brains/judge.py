"""Learned judge model for colony episode evaluation.

Uses a Bradley-Terry model trained on pairwise comparisons of episode
summaries to learn a fitness scoring function. The judge replaces
hand-crafted fitness heuristics with a learned evaluation.

Input: 12-dim feature vector from emergence detectors + tracker metrics
Output: scalar fitness score
Training: Bradley-Terry model on episode pairs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeSummary:
    """Summary of one simulation episode for judge comparison."""
    features: np.ndarray    # 12-dim: emergence (7) + tracker (5)
    food_total: float       # total food deposited (ground-truth quality signal)


class ComparisonBuffer:
    """Stores episode summaries for generating training pairs."""

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = capacity
        self._episodes: list[EpisodeSummary] = []

    def add_episode(self, features: np.ndarray, food_total: float) -> None:
        ep = EpisodeSummary(features=features.copy(), food_total=food_total)
        if len(self._episodes) >= self._capacity:
            self._episodes.pop(0)
        self._episodes.append(ep)

    def __len__(self) -> int:
        return len(self._episodes)

    def sample_pairs(
        self, n: int = 32, rng: np.random.Generator | None = None,
    ) -> list[tuple[EpisodeSummary, EpisodeSummary, float]]:
        """Sample n pairs with Bradley-Terry labels.

        Returns list of (ep_a, ep_b, label) where label is the probability
        that ep_a is better than ep_b (1.0 if a > b, 0.0 if b > a,
        0.5 if equal).
        """
        if len(self._episodes) < 2:
            return []
        if rng is None:
            rng = np.random.default_rng()

        pairs: list[tuple[EpisodeSummary, EpisodeSummary, float]] = []
        for _ in range(n):
            i, j = rng.choice(len(self._episodes), size=2, replace=False)
            a, b = self._episodes[i], self._episodes[j]
            if a.food_total > b.food_total:
                label = 1.0
            elif b.food_total > a.food_total:
                label = 0.0
            else:
                label = 0.5
            pairs.append((a, b, label))
        return pairs


class JudgeModel:
    """Learns to score colony episodes via pairwise comparisons.

    Architecture: 2-layer MLP (input_dim → hidden → 1)
    Training: Bradley-Terry model — P(a > b) = σ(score(a) - score(b))
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden: int = 32,
        lr: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / hidden)

        self._W1 = rng.normal(0, s1, (input_dim, hidden))
        self._b1 = np.zeros(hidden)
        self._W2 = rng.normal(0, s2, (hidden, 1))
        self._b2 = np.zeros(1)
        self._lr = lr

    def _forward(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Forward pass returning (score, hidden_activations)."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        h = np.maximum(0.0, x @ self._W1 + self._b1)
        score = float((h @ self._W2 + self._b2).squeeze())
        return score, h.squeeze()

    def score(self, features: np.ndarray) -> float:
        """Score a single episode feature vector."""
        s, _ = self._forward(features)
        return s

    def train_from_comparisons(
        self,
        pairs: list[tuple[EpisodeSummary, EpisodeSummary, float]],
    ) -> float:
        """Train on pairwise comparisons. Returns mean loss."""
        if not pairs:
            return 0.0

        total_loss = 0.0
        for ep_a, ep_b, label in pairs:
            score_a, h_a = self._forward(ep_a.features)
            score_b, h_b = self._forward(ep_b.features)

            # Bradley-Terry: P(a > b) = σ(score_a - score_b)
            diff = score_a - score_b
            diff = max(-20.0, min(20.0, diff))  # numerical stability
            prob = 1.0 / (1.0 + math.exp(-diff))

            # Binary cross-entropy loss
            prob_clamped = max(1e-8, min(1.0 - 1e-8, prob))
            loss = -(label * math.log(prob_clamped) + (1 - label) * math.log(1 - prob_clamped))
            total_loss += loss

            # Gradient: dL/d_diff = prob - label
            dL_ddiff = prob - label

            # Backprop through both branches
            self._backprop_single(ep_a.features, h_a, dL_ddiff)
            self._backprop_single(ep_b.features, h_b, -dL_ddiff)

        return total_loss / len(pairs)

    def _backprop_single(
        self,
        x: np.ndarray,
        h: np.ndarray,
        dL_dscore: float,
    ) -> None:
        """Backprop gradient for a single example."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if h.ndim == 1:
            h = h[np.newaxis, :]

        # score = h @ W2 + b2
        dW2 = h.T * dL_dscore
        db2 = np.array([dL_dscore])
        dL_dh = dL_dscore * self._W2.T  # (1, hidden)

        # h = relu(x @ W1 + b1)
        dL_dh *= (h > 0).astype(float)
        dW1 = x.T @ dL_dh
        db1 = dL_dh.squeeze()

        # SGD
        self._W1 -= self._lr * dW1
        self._b1 -= self._lr * db1
        self._W2 -= self._lr * dW2
        self._b2 -= self._lr * db2

    def distribute_to_agents(
        self,
        score: float,
        contributions: dict[int, float],
    ) -> dict[int, float]:
        """Distribute colony-level score to individual agents by contribution.

        Args:
            score: judge's colony-level fitness score
            contributions: {ant_id: contribution_fraction} where fractions sum to ~1

        Returns:
            {ant_id: individual_reward}
        """
        total = sum(contributions.values())
        if total < 1e-8:
            return {k: 0.0 for k in contributions}
        return {
            ant_id: score * (frac / total)
            for ant_id, frac in contributions.items()
        }
