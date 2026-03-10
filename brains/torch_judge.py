"""PyTorch Bradley-Terry judge model for colony episode evaluation.

Replaces manual gradient computation in judge.py with PyTorch autograd.
Same interface: score() + train_from_comparisons().
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from brains.judge import EpisodeSummary
from brains.torch_utils import get_device


class TorchJudgeModel(nn.Module):
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
        super().__init__()
        self._device = get_device()

        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.to(self._device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def score(self, features: np.ndarray) -> float:
        """Score a single episode feature vector."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return float(self(x).item())

    def train_from_comparisons(
        self,
        pairs: list[tuple[EpisodeSummary, EpisodeSummary, float]],
    ) -> float:
        """Train on pairwise comparisons. Returns mean loss."""
        if not pairs:
            return 0.0

        features_a = torch.tensor(
            np.array([p[0].features for p in pairs]),
            dtype=torch.float32, device=self._device,
        )
        features_b = torch.tensor(
            np.array([p[1].features for p in pairs]),
            dtype=torch.float32, device=self._device,
        )
        labels = torch.tensor(
            [p[2] for p in pairs],
            dtype=torch.float32, device=self._device,
        )

        score_a = self(features_a)
        score_b = self(features_b)
        diff = score_a - score_b

        # BCE loss: -[y*log(σ(d)) + (1-y)*log(1-σ(d))]
        loss = nn.functional.binary_cross_entropy_with_logits(diff, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def distribute_to_agents(
        self,
        score_val: float,
        contributions: dict[int, float],
    ) -> dict[int, float]:
        """Distribute colony-level score to individual agents by contribution."""
        total = sum(contributions.values())
        if total < 1e-8:
            return {k: 0.0 for k in contributions}
        return {
            ant_id: score_val * (frac / total)
            for ant_id, frac in contributions.items()
        }
