"""PyTorch RND (Random Network Distillation) intrinsic motivation.

Replaces manual SGD in intrinsic.py with PyTorch autograd.
Same interface: intrinsic_reward() + update().
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from brains.torch_utils import get_device


class _RNDNet(nn.Module):
    """Simple 2-layer MLP for RND target/predictor."""

    def __init__(self, input_dim: int = 39, hidden: int = 32, output_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchRNDExplorer:
    """RND curiosity with autograd.

    A fixed random target network and a trainable predictor network.
    Intrinsic reward = MSE between their outputs (normalized).
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden: int = 32,
        output_dim: int = 16,
        lr: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self._device = get_device()

        if seed is not None:
            torch.manual_seed(seed)

        # Fixed target (never trained)
        self.target = _RNDNet(input_dim, hidden, output_dim).to(self._device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        # Trainable predictor
        self.predictor = _RNDNet(input_dim, hidden, output_dim).to(self._device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running stats for normalization
        self._reward_mean: float = 0.0
        self._reward_var: float = 1.0
        self._reward_count: int = 0

    def intrinsic_reward(self, state: np.ndarray) -> float:
        """Compute intrinsic reward for a state vector (39,)."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            target_out = self.target(x)
            pred_out = self.predictor(x)
            mse = float((target_out - pred_out).pow(2).mean().item())

        # Update running stats
        self._reward_count += 1
        delta = mse - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = mse - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / max(self._reward_count, 2)

        std = max(math.sqrt(abs(self._reward_var)), 1e-8)
        return (mse - self._reward_mean) / std

    def update(self, state: np.ndarray) -> None:
        """Train predictor to match target (reduces novelty for seen states)."""
        x = torch.tensor(state, dtype=torch.float32, device=self._device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            target_out = self.target(x)

        pred_out = self.predictor(x)
        loss = (pred_out - target_out).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
