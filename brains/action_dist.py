"""GPU-native batched action distribution for 11-head output.

Replaces per-sample Python loops with batched torch.distributions ops.
This is the key GPU optimization for PPO training.

Head mapping:
    [0]   turn      — Normal(tanh(z)*π/6, σ=0.1)
    [1]   speed     — Normal(sigmoid(z), σ=0.1)
    [2:7] deposit   — Categorical(softmax(z))
    [7]   strength  — Normal(sigmoid(z), σ=0.1)
    [8]   pickup    — Bernoulli(sigmoid(z))
    [9]   drop      — Bernoulli(sigmoid(z))
    [10]  recruit   — Bernoulli(sigmoid(z))
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.distributions as D

from agents.actions import AntAction

# Pheromone channel names (must stay in sync with action_utils._DEPOSIT_CHANNELS)
_DEPOSIT_CHANNELS: list[Optional[str]] = [None, "food", "home", "danger", "recruit"]

_SIGMA = 0.1
_PI_OVER_6 = math.pi / 6.0


class ActionDistribution:
    """Batched GPU-native action distribution for the 11-head output.

    All operations are fully batched — no Python loops over samples.
    """

    def __init__(self, logits: torch.Tensor):
        """Build distributions from raw logits.

        Args:
            logits: (B, 11) raw output from policy network.
        """
        assert logits.dim() == 2 and logits.shape[1] == 11, (
            f"Expected (B, 11) logits, got {logits.shape}"
        )
        self._logits = logits
        self._device = logits.device

        # Continuous heads: Normal distributions
        self._turn_mean = torch.tanh(logits[:, 0]) * _PI_OVER_6
        self._speed_mean = torch.sigmoid(logits[:, 1])
        self._strength_mean = torch.sigmoid(logits[:, 7])

        self._turn_dist = D.Normal(self._turn_mean, _SIGMA)
        self._speed_dist = D.Normal(self._speed_mean, _SIGMA)
        self._strength_dist = D.Normal(self._strength_mean, _SIGMA)

        # Categorical head: deposit channel
        self._deposit_dist = D.Categorical(logits=logits[:, 2:7])

        # Binary heads: Bernoulli
        self._pickup_dist = D.Bernoulli(logits=logits[:, 8])
        self._drop_dist = D.Bernoulli(logits=logits[:, 9])
        self._recruit_dist = D.Bernoulli(logits=logits[:, 10])

    def sample(self) -> torch.Tensor:
        """Sample actions. Returns (B, 7) tensor.

        Columns: [turn, speed, deposit_idx, strength, pickup, drop, recruit]
        """
        turn = self._turn_dist.sample()
        speed = self._speed_dist.sample()
        deposit = self._deposit_dist.sample().float()
        strength = self._strength_dist.sample()
        pickup = self._pickup_dist.sample()
        drop = self._drop_dist.sample()
        recruit = self._recruit_dist.sample()

        return torch.stack([turn, speed, deposit, strength, pickup, drop, recruit], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of actions. Returns (B,) tensor.

        Args:
            actions: (B, 7) tensor with columns
                     [turn, speed, deposit_idx, strength, pickup, drop, recruit]
        """
        lp = torch.zeros(actions.shape[0], device=self._device, dtype=self._logits.dtype)

        lp = lp + self._turn_dist.log_prob(actions[:, 0])
        lp = lp + self._speed_dist.log_prob(actions[:, 1])
        lp = lp + self._deposit_dist.log_prob(actions[:, 2].long())
        lp = lp + self._strength_dist.log_prob(actions[:, 3])
        lp = lp + self._pickup_dist.log_prob(actions[:, 4])
        lp = lp + self._drop_dist.log_prob(actions[:, 5])
        lp = lp + self._recruit_dist.log_prob(actions[:, 6])

        return lp

    def entropy(self) -> torch.Tensor:
        """Compute policy entropy. Returns (B,) tensor.

        Includes all heads (continuous + discrete).
        """
        ent = torch.zeros(self._logits.shape[0], device=self._device, dtype=self._logits.dtype)
        ent = ent + self._turn_dist.entropy()
        ent = ent + self._speed_dist.entropy()
        ent = ent + self._deposit_dist.entropy()
        ent = ent + self._strength_dist.entropy()
        ent = ent + self._pickup_dist.entropy()
        ent = ent + self._drop_dist.entropy()
        ent = ent + self._recruit_dist.entropy()
        return ent


def action_tensor_to_ant_actions(t: np.ndarray) -> list[AntAction]:
    """Convert (B, 7) numpy action tensor to list of AntAction objects.

    Used at the sim-physics boundary where AntAction dataclasses are needed.
    """
    actions = []
    for i in range(t.shape[0]):
        deposit_idx = int(t[i, 2])
        deposit_pheromone = _DEPOSIT_CHANNELS[deposit_idx]
        strength_val = float(t[i, 3])
        actions.append(AntAction(
            turn=float(t[i, 0]),
            speed_mult=float(t[i, 1]),
            deposit_pheromone=deposit_pheromone,
            deposit_strength=strength_val if deposit_pheromone is not None else 0.0,
            pickup=bool(t[i, 4] > 0.5),
            drop=bool(t[i, 5] > 0.5),
            recruit_signal=bool(t[i, 6] > 0.5),
        ))
    return actions
