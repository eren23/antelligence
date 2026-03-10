"""Shared action utilities — decode, sample, log-prob, reward for 11-head output.

Extracted from nn_brain.py. Used by torch brains, main.py, PPO, and training scripts.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from agents.actions import AntAction
from agents.sensory import SensoryInput


# Pheromone channel names for deposit head (index 0 = none)
_DEPOSIT_CHANNELS: list[Optional[str]] = [None, "food", "home", "danger", "recruit"]


# ---------------------------------------------------------------------------
# NumPy activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    x = np.clip(x, -500.0, 500.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Decode raw 11-dim logits → action components
# ---------------------------------------------------------------------------

def decode_output(raw: np.ndarray) -> dict:
    """Decode raw network output (11,) into action components.

    Returns dict with:
        turn, speed, deposit_probs, strength, pickup, drop, recruit
    """
    squeeze = raw.ndim == 1
    if squeeze:
        raw = raw[np.newaxis, :]

    turn = _tanh(raw[:, 0:1]) * (math.pi / 6.0)       # [-π/6, π/6]
    speed = _sigmoid(raw[:, 1:2])                        # [0, 1]
    deposit_probs = _softmax(raw[:, 2:7], axis=-1)       # (batch, 5)
    strength = _sigmoid(raw[:, 7:8])                     # [0, 1]
    pickup = _sigmoid(raw[:, 8:9])                       # [0, 1]
    drop = _sigmoid(raw[:, 9:10])                        # [0, 1]
    recruit = _sigmoid(raw[:, 10:11])                    # [0, 1]

    result = dict(
        turn=turn, speed=speed, deposit_probs=deposit_probs,
        strength=strength, pickup=pickup, drop=drop, recruit=recruit,
    )
    if squeeze:
        result = {k: v[0] for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Sample a concrete AntAction from decoded output
# ---------------------------------------------------------------------------

def sample_action(decoded: dict, rng: np.random.Generator) -> AntAction:
    """Sample a concrete AntAction from decoded network output (single ant)."""
    turn_val = float(decoded["turn"][0])
    speed_val = float(decoded["speed"][0])

    probs = decoded["deposit_probs"]
    deposit_idx = int(rng.choice(len(probs), p=probs))
    deposit_pheromone = _DEPOSIT_CHANNELS[deposit_idx]

    strength_val = float(decoded["strength"][0])

    pickup_val = bool(rng.random() < float(decoded["pickup"][0]))
    drop_val = bool(rng.random() < float(decoded["drop"][0]))
    recruit_val = bool(rng.random() < float(decoded["recruit"][0]))

    return AntAction(
        turn=turn_val,
        speed_mult=speed_val,
        deposit_pheromone=deposit_pheromone,
        deposit_strength=strength_val if deposit_pheromone is not None else 0.0,
        pickup=pickup_val,
        drop=drop_val,
        recruit_signal=recruit_val,
    )


# ---------------------------------------------------------------------------
# Log-probability of an action under the policy (NumPy-side)
# ---------------------------------------------------------------------------

def log_prob_of_action(decoded: dict, action: AntAction) -> float:
    """Compute log-probability of the taken action under the policy.

    For continuous outputs (turn, speed, strength), we treat the network output
    as the mean of a narrow Gaussian with fixed σ and compute log-prob.
    For discrete outputs (deposit, pickup, drop, recruit), we use the
    categorical/Bernoulli log-prob.
    """
    lp = 0.0

    sigma_turn = 0.1
    turn_diff = action.turn - float(decoded["turn"][0])
    lp += -0.5 * (turn_diff / sigma_turn) ** 2 - math.log(sigma_turn)

    sigma_speed = 0.1
    speed_diff = action.speed_mult - float(decoded["speed"][0])
    lp += -0.5 * (speed_diff / sigma_speed) ** 2 - math.log(sigma_speed)

    probs = decoded["deposit_probs"]
    deposit_idx = _DEPOSIT_CHANNELS.index(action.deposit_pheromone)
    lp += math.log(max(float(probs[deposit_idx]), 1e-8))

    sigma_str = 0.1
    str_diff = action.deposit_strength - float(decoded["strength"][0])
    lp += -0.5 * (str_diff / sigma_str) ** 2 - math.log(sigma_str)

    for key, acted in [("pickup", action.pickup), ("drop", action.drop),
                       ("recruit", action.recruit_signal)]:
        p = float(decoded[key][0])
        p = max(min(p, 1.0 - 1e-8), 1e-8)
        if acted:
            lp += math.log(p)
        else:
            lp += math.log(1.0 - p)

    return lp


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    prev_sensory: SensoryInput | None,
    curr_sensory: SensoryInput,
    action: AntAction,
    alive: bool,
) -> float:
    """Compute reward signal for a single ant transition."""
    reward = 0.0

    if not alive:
        return -0.5

    if prev_sensory is not None:
        if prev_sensory.carrying == "food" and curr_sensory.carrying is None:
            reward += 1.0
        if prev_sensory.carrying is None and curr_sensory.carrying == "food":
            reward += 0.3
        if curr_sensory.carrying is None and curr_sensory.nearest_food_distance < 1.0:
            food_dist_delta = prev_sensory.nearest_food_distance - curr_sensory.nearest_food_distance
            if food_dist_delta > 0.001:
                reward += 0.2
            elif food_dist_delta < -0.001:
                reward -= 0.05
        if curr_sensory.carrying == "food":
            nest_dist_prev = prev_sensory.nest_distance
            nest_dist_curr = curr_sensory.nest_distance
            if nest_dist_curr < nest_dist_prev - 0.5:
                reward += 0.2
            elif nest_dist_curr > nest_dist_prev + 0.5:
                reward -= 0.05

    energy_norm = curr_sensory.energy / 100.0
    if energy_norm > 0.5:
        reward += 0.1
    elif energy_norm < 0.2:
        reward -= 0.1

    if action.speed_mult > 0.3:
        reward += 0.05

    return reward
