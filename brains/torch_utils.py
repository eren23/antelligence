"""Shared PyTorch utilities for all torch brain backends."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from agents.actions import AntAction
from brains.action_utils import _DEPOSIT_CHANNELS


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

_DEVICE: torch.device | None = None


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")
    return _DEVICE


# ---------------------------------------------------------------------------
# Differentiable log-probability of action
# ---------------------------------------------------------------------------

def torch_log_prob_of_action(logits: torch.Tensor, action: AntAction) -> torch.Tensor:
    """Compute differentiable log-probability of a taken action.

    Mirrors the 11-head decode from nn_brain.py:
        [0]   turn      — tanh → Gaussian(σ=0.1)
        [1]   speed     — sigmoid → Gaussian(σ=0.1)
        [2:7] deposit   — softmax → Categorical
        [7]   strength  — sigmoid → Gaussian(σ=0.1)
        [8]   pickup    — sigmoid → Bernoulli
        [9]   drop      — sigmoid → Bernoulli
        [10]  recruit   — sigmoid → Bernoulli

    Args:
        logits: (11,) raw output logits (must be part of computation graph).
        action: the sampled AntAction.

    Returns:
        Scalar tensor log-probability (differentiable).
    """
    sigma = 0.1
    lp = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Turn: tanh output, Gaussian
    turn_mean = torch.tanh(logits[0]) * (math.pi / 6.0)
    lp = lp - 0.5 * ((action.turn - turn_mean) / sigma) ** 2

    # Speed: sigmoid output, Gaussian
    speed_mean = torch.sigmoid(logits[1])
    lp = lp - 0.5 * ((action.speed_mult - speed_mean) / sigma) ** 2

    # Deposit: softmax categorical
    deposit_log_probs = torch.log_softmax(logits[2:7], dim=0)
    deposit_idx = _DEPOSIT_CHANNELS.index(action.deposit_pheromone)
    lp = lp + deposit_log_probs[deposit_idx]

    # Strength: sigmoid output, Gaussian
    str_mean = torch.sigmoid(logits[7])
    lp = lp - 0.5 * ((action.deposit_strength - str_mean) / sigma) ** 2

    # Binary heads: sigmoid Bernoulli
    for j, acted in enumerate([action.pickup, action.drop, action.recruit_signal]):
        p = torch.sigmoid(logits[8 + j])
        if acted:
            lp = lp + torch.log(p.clamp(min=1e-8))
        else:
            lp = lp + torch.log((1.0 - p).clamp(min=1e-8))

    return lp


def torch_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute policy entropy from raw logits (11,) or (batch, 11).

    Only includes categorical (deposit) and Bernoulli (binary) heads.
    Continuous heads have fixed-variance entropy (constant, skipped).
    """
    squeeze = logits.dim() == 1
    if squeeze:
        logits = logits.unsqueeze(0)

    ent = torch.zeros(logits.shape[0], device=logits.device)

    # Categorical entropy for deposit (indices 2:7)
    deposit_probs = torch.softmax(logits[:, 2:7], dim=-1)
    ent = ent - (deposit_probs * torch.log(deposit_probs + 1e-8)).sum(dim=-1)

    # Bernoulli entropy for binary heads (indices 8, 9, 10)
    for j in range(3):
        p = torch.sigmoid(logits[:, 8 + j]).clamp(1e-8, 1.0 - 1e-8)
        ent = ent - (p * torch.log(p) + (1 - p) * torch.log(1 - p))

    if squeeze:
        return ent.squeeze(0)
    return ent
