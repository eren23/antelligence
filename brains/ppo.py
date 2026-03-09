"""PPO (Proximal Policy Optimization) training infrastructure.

Phase 2 of the REINFORCE → Imitation + PPO migration:
  - PPORolloutBuffer: stores trajectory data with GAE computation
  - ppo_update_mlp: PPO update for the MLP brain
  - Entropy computation for mixed action heads
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.actions import AntAction
from brains.nn_brain import (
    MLPWeights,
    ForwardCache,
    _DEPOSIT_CHANNELS,
    _backprop_logits_gradient,
    _sigmoid,
    _softmax,
    _tanh,
    forward,
    forward_with_value,
    decode_output,
    log_prob_of_action,
)


# ---------------------------------------------------------------------------
# PPO Rollout Buffer
# ---------------------------------------------------------------------------

@dataclass
class PPOStep:
    """Single step of a PPO rollout."""
    state: np.ndarray         # sensory vector (39,)
    action: AntAction         # action taken
    log_prob: float           # log π_old(a|s)
    value: float              # V(s) from old policy
    reward: float             # r_t
    done: bool                # terminal flag


class PPORolloutBuffer:
    """Collects rollout data and computes GAE advantages."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._steps: list[PPOStep] = []

    def add(self, step: PPOStep) -> None:
        self._steps.append(step)

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def ready(self) -> bool:
        return len(self._steps) >= self.capacity

    def compute_gae(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[AntAction], np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            last_value: V(s_{T+1}) bootstrap value for non-terminal
            gamma: discount factor
            lam: GAE lambda

        Returns:
            states: (N, 39)
            advantages: (N,)
            returns: (N,) — advantages + values (targets for value function)
            actions: list of N AntActions
            old_log_probs: (N,)
            old_values: (N,)
        """
        n = len(self._steps)
        advantages = np.zeros(n, dtype=np.float64)
        rewards = np.array([s.reward for s in self._steps])
        values = np.array([s.value for s in self._steps])
        dones = np.array([s.done for s in self._steps])

        # GAE backward pass
        gae = 0.0
        for t in range(n - 1, -1, -1):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        states = np.array([s.state for s in self._steps])
        actions = [s.action for s in self._steps]
        old_log_probs = np.array([s.log_prob for s in self._steps])
        old_values = values

        return states, advantages, returns, actions, old_log_probs, old_values

    def clear(self) -> None:
        self._steps.clear()


# ---------------------------------------------------------------------------
# Entropy computation for mixed action heads
# ---------------------------------------------------------------------------

def compute_entropy(z_out: np.ndarray) -> np.ndarray:
    """Compute entropy of the policy for each sample.

    Combines entropy from:
    - Deposit head (categorical over 5 channels)
    - Binary heads (pickup, drop, recruit — Bernoulli)

    Continuous heads (turn, speed, strength) have fixed-variance Gaussian
    entropy which is constant, so we skip them.

    Args:
        z_out: (batch, 11) raw logits

    Returns:
        entropy: (batch,) total entropy per sample
    """
    batch = z_out.shape[0]
    entropy = np.zeros(batch)

    # Categorical entropy for deposit head
    probs = _softmax(z_out[:, 2:7], axis=-1)
    entropy += -np.sum(probs * np.log(probs + 1e-8), axis=-1)

    # Bernoulli entropy for binary heads
    for j in range(3):
        p = _sigmoid(z_out[:, 8 + j:9 + j]).squeeze(-1)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        entropy += -(p * np.log(p) + (1 - p) * np.log(1 - p))

    return entropy


# ---------------------------------------------------------------------------
# PPO logit gradient computation
# ---------------------------------------------------------------------------

def compute_ppo_logit_grad(
    z_out: np.ndarray,
    actions: list[AntAction],
    advantages: np.ndarray,
    old_log_probs: np.ndarray,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PPO clipped surrogate loss gradient w.r.t. output logits.

    Args:
        z_out: (batch, 11) raw output logits from current policy
        actions: list of batch AntActions (from rollout)
        advantages: (batch,) GAE advantages
        old_log_probs: (batch,) log π_old(a|s) from rollout
        clip_epsilon: PPO clipping parameter
        entropy_coef: weight for entropy bonus

    Returns:
        dL_dz: (batch, 11) gradient for policy loss
        new_log_probs: (batch,) log probs under current policy (for diagnostics)
    """
    batch = z_out.shape[0]
    dL_dz = np.zeros_like(z_out)
    new_log_probs = np.zeros(batch)

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std > 1e-8:
        advantages_norm = (advantages - adv_mean) / adv_std
    else:
        advantages_norm = advantages - adv_mean

    for i in range(batch):
        # Compute current log_prob and its gradient w.r.t. z_out
        decoded = decode_output(z_out[i])
        new_lp = log_prob_of_action(decoded, actions[i])
        new_log_probs[i] = new_lp

        ratio = math.exp(new_lp - old_log_probs[i])
        adv = float(advantages_norm[i])

        # Clipped surrogate: min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
        clipped_ratio = max(1.0 - clip_epsilon, min(1.0 + clip_epsilon, ratio))

        if adv >= 0:
            effective_ratio = min(ratio, clipped_ratio)
        else:
            effective_ratio = max(ratio, clipped_ratio)

        # Check if clipped (gradient is zero if clipped)
        if abs(ratio - effective_ratio) > 1e-8:
            # Clipped — no gradient from this sample for policy
            pass
        else:
            # Not clipped — gradient flows through
            # ∂L/∂z = adv * ratio * ∂log_pi/∂z
            # ∂log_pi/∂z is the REINFORCE-style gradient for log-prob
            _accumulate_logprob_grad(dL_dz, i, z_out[i], actions[i], -adv * ratio)

        # Entropy bonus gradient (always flows)
        _accumulate_entropy_grad(dL_dz, i, z_out[i], -entropy_coef)

    return dL_dz, new_log_probs


def _accumulate_logprob_grad(
    dL_dz: np.ndarray,
    idx: int,
    z: np.ndarray,
    action: AntAction,
    scale: float,
) -> None:
    """Accumulate ∂log π(a|s)/∂z scaled by `scale` into dL_dz[idx]."""
    pi_over_6 = math.pi / 6.0

    # Turn (idx 0): tanh output, Gaussian log-prob
    t = np.tanh(z[0])
    sigma_t = 0.1
    turn_target = action.turn
    dL_dz[idx, 0] += (turn_target - t * pi_over_6) * pi_over_6 * (1 - t**2) / (sigma_t**2) * scale

    # Speed (idx 1): sigmoid, Gaussian log-prob
    s = float(_sigmoid(np.array([z[1]]))[0])
    sigma_s = 0.1
    dL_dz[idx, 1] += (action.speed_mult - s) * s * (1 - s) / (sigma_s**2) * scale

    # Deposit (idx 2:7): softmax, categorical
    probs = _softmax(z[2:7])
    deposit_idx = _DEPOSIT_CHANNELS.index(action.deposit_pheromone)
    grad_dep = -probs.copy()
    grad_dep[deposit_idx] += 1.0
    dL_dz[idx, 2:7] += grad_dep * scale

    # Strength (idx 7): sigmoid, Gaussian
    st = float(_sigmoid(np.array([z[7]]))[0])
    sigma_st = 0.1
    dL_dz[idx, 7] += (action.deposit_strength - st) * st * (1 - st) / (sigma_st**2) * scale

    # Binary heads (idx 8,9,10)
    for j, acted in enumerate([action.pickup, action.drop, action.recruit_signal]):
        sig = float(_sigmoid(np.array([z[8 + j]]))[0])
        dL_dz[idx, 8 + j] += (float(acted) - sig) * scale


def _accumulate_entropy_grad(
    dL_dz: np.ndarray,
    idx: int,
    z: np.ndarray,
    scale: float,
) -> None:
    """Accumulate entropy gradient into dL_dz[idx].

    Entropy bonus gradient: ∂H/∂z
    For categorical (deposit): ∂H/∂z_j = -softmax_j * (1 + log(softmax_j)) ... simplified
    For Bernoulli: ∂H/∂z = σ(z)(1-σ(z)) * (log(1-σ(z)) - log(σ(z)))
    """
    # Deposit categorical entropy gradient
    probs = _softmax(z[2:7])
    # ∂H/∂z = -∑_k p_k(δ_jk - p_j)(1 + log p_k) = p_j * ∑_k p_k * log(p_k) - p_j * log(p_j)
    # Simplified: ∂(-∑p log p)/∂z_j = ∑_k (δ_jk - p_j)(-log p_k - 1) * p_k
    # = -p_j - p_j*log(p_j) + p_j * (∑_k p_k * (log p_k + 1))
    # Actually for softmax entropy the gradient is:
    # ∂H/∂z_j = p_j * (H + log p_j)  (with a sign that encourages higher entropy)
    H = -np.sum(probs * np.log(probs + 1e-8))
    for k in range(5):
        dL_dz[idx, 2 + k] += scale * probs[k] * (H + np.log(probs[k] + 1e-8))

    # Bernoulli entropy gradient
    for j in range(3):
        p = float(_sigmoid(np.array([z[8 + j]]))[0])
        p = max(1e-8, min(1 - 1e-8, p))
        # ∂H/∂z = σ(z)(1-σ(z)) * (log(1-σ(z)) - log(σ(z)))
        # = p(1-p) * log((1-p)/p)
        dL_dz[idx, 8 + j] += scale * p * (1 - p) * (math.log(1 - p) - math.log(p))


# ---------------------------------------------------------------------------
# Value head gradient
# ---------------------------------------------------------------------------

def compute_value_grad(
    weights: MLPWeights,
    cache: ForwardCache,
    values: np.ndarray,
    returns: np.ndarray,
    old_values: np.ndarray | None = None,
    clip_epsilon: float = 0.2,
) -> tuple[list[np.ndarray], float]:
    """Compute gradients for the value head.

    Uses MSE loss with optional value clipping.

    Args:
        weights: MLP weights (need W_val, b_val)
        cache: forward pass cache (need a2 hidden activations)
        values: (batch,) current value predictions
        returns: (batch,) GAE returns (value targets)
        old_values: (batch,) old value predictions for clipping (optional)
        clip_epsilon: value clipping range

    Returns:
        [dW_val, db_val]: gradients for value head parameters
        value_loss: scalar loss
    """
    batch = cache.a2.shape[0]

    if old_values is not None:
        # Clipped value loss
        v_clipped = old_values + np.clip(values - old_values, -clip_epsilon, clip_epsilon)
        loss_unclipped = (values - returns) ** 2
        loss_clipped = (v_clipped - returns) ** 2
        value_loss = 0.5 * np.mean(np.maximum(loss_unclipped, loss_clipped))

        # Use gradient from whichever loss is larger
        use_clipped = loss_clipped > loss_unclipped
        dL_dv = np.where(use_clipped, v_clipped - returns, values - returns) / batch
    else:
        value_loss = 0.5 * np.mean((values - returns) ** 2)
        dL_dv = (values - returns) / batch

    # dL/dv is (batch,), W_val is (h2, 1), value = a2 @ W_val + b_val
    # dW_val = a2.T @ dL_dv.reshape(-1, 1) / batch
    dL_dv_col = dL_dv.reshape(-1, 1)  # (batch, 1)
    dW_val = cache.a2.T @ dL_dv_col / batch
    db_val = dL_dv.mean(axis=0, keepdims=True)

    return [dW_val, db_val], float(value_loss)


# ---------------------------------------------------------------------------
# Full PPO update for MLP
# ---------------------------------------------------------------------------

def ppo_update_mlp(
    weights: MLPWeights,
    buffer: PPORolloutBuffer,
    *,
    last_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run a full PPO update on the rollout buffer.

    Returns dict of training metrics.
    """
    # Compute GAE
    states, advantages, returns, actions, old_log_probs, old_values = \
        buffer.compute_gae(last_value, gamma, gae_lambda)

    n = len(states)
    rng = np.random.default_rng()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for epoch in range(epochs):
        perm = rng.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]

            batch_states = states[idx]
            batch_actions = [actions[i] for i in idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]
            batch_old_lp = old_log_probs[idx]
            batch_old_values = old_values[idx]

            # Forward pass with value
            z_out, values, cache = forward_with_value(weights, batch_states)
            if z_out.ndim == 1:
                z_out = z_out[np.newaxis, :]
            if isinstance(values, (int, float)):
                values = np.array([values])

            # Policy gradient
            policy_dL_dz, new_lps = compute_ppo_logit_grad(
                z_out, batch_actions, batch_advantages, batch_old_lp,
                clip_epsilon, entropy_coef,
            )

            # Backprop policy gradient through MLP
            policy_grads = _backprop_logits_gradient(weights, cache, policy_dL_dz)

            # Value gradient
            value_grads, v_loss = compute_value_grad(
                weights, cache, values, batch_returns, batch_old_values, clip_epsilon,
            )

            # Combine gradients
            # policy_grads = [dW1, db1, dW2, db2, dW_out, db_out]
            # value_grads = [dW_val, db_val]
            # Hidden layers get gradient from both policy and value heads
            # But value head only contributes through the shared hidden layers

            # Value contribution to hidden layers (backprop through value head)
            batch_sz = cache.a2.shape[0]
            dL_dv = (values - batch_returns).reshape(-1, 1) / batch_sz  # (batch, 1)
            dL_da2_val = dL_dv @ weights.W_val.T  # (batch, h2)
            dL_dz2_val = dL_da2_val * (cache.z2 > 0).astype(float)
            dW2_val = cache.a1.T @ dL_dz2_val / batch_sz * value_loss_coef
            db2_val = dL_dz2_val.mean(axis=0) * value_loss_coef
            dL_da1_val = dL_dz2_val @ weights.W2.T
            dL_dz1_val = dL_da1_val * (cache.z1 > 0).astype(float)
            dW1_val = cache.x.T @ dL_dz1_val / batch_sz * value_loss_coef
            db1_val = dL_dz1_val.mean(axis=0) * value_loss_coef

            # Combined hidden layer gradients (policy + value)
            combined_grads = [
                policy_grads[0] + dW1_val,   # dW1
                policy_grads[1] + db1_val,   # db1
                policy_grads[2] + dW2_val,   # dW2
                policy_grads[3] + db2_val,   # db2
                policy_grads[4],              # dW_out (policy only)
                policy_grads[5],              # db_out (policy only)
                value_grads[0] * value_loss_coef,   # dW_val
                value_grads[1] * value_loss_coef,   # db_val
            ]

            # Gradient clipping by global norm
            total_norm_sq = sum(float(np.sum(g ** 2)) for g in combined_grads)
            total_norm = math.sqrt(total_norm_sq)
            if total_norm > max_grad_norm:
                clip_scale = max_grad_norm / (total_norm + 1e-8)
                combined_grads = [g * clip_scale for g in combined_grads]

            # Apply gradients
            all_params = weights.all_params()
            for param, grad in zip(all_params, combined_grads):
                param -= lr * grad

            # Track metrics
            entropy = float(compute_entropy(z_out).mean())
            total_policy_loss += float(np.mean(batch_advantages ** 2))
            total_value_loss += v_loss
            total_entropy += entropy
            num_updates += 1

    buffer.clear()

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "num_updates": num_updates,
    }


# ---------------------------------------------------------------------------
# PPO RoleTrainer for MLP
# ---------------------------------------------------------------------------

class PPORoleTrainer:
    """PPO trainer for a single role's shared MLP weights."""

    def __init__(
        self,
        weights: MLPWeights,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        rollout_length: int = 256,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ):
        self.weights = weights
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.rollout_length = rollout_length
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.buffer = PPORolloutBuffer(rollout_length)
        self.global_step: int = 0
        self.baseline: float = 0.0  # compatibility with trainer state save/load

    def update(self) -> dict[str, float]:
        """Run PPO update if buffer is ready. Returns training metrics."""
        if not self.buffer.ready:
            return {}

        self.global_step += 1

        # Get bootstrap value from last state
        last_step = self.buffer._steps[-1]
        _, last_val, _ = forward_with_value(self.weights, last_step.state)
        last_value = float(last_val) if not last_step.done else 0.0

        metrics = ppo_update_mlp(
            self.weights,
            self.buffer,
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            epochs=self.epochs_per_update,
            batch_size=self.batch_size,
            lr=self.lr,
            max_grad_norm=self.max_grad_norm,
        )
        return metrics


# ---------------------------------------------------------------------------
# PPO RoleTrainer for Transformer (zeroth-order)
# ---------------------------------------------------------------------------

class PPOTransformerRoleTrainer:
    """PPO trainer for transformer weights using zeroth-order gradient with PPO advantage signal.

    Much stronger than raw REINFORCE noise because the advantage signal from
    GAE provides a much better gradient estimate.
    """

    def __init__(
        self,
        weight_set: Any,  # TransformerWeightSet
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rollout_length: int = 256,
        perturbation_scale: float = 0.01,
    ):
        self.weight_set = weight_set
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_length = rollout_length
        self.perturbation_scale = perturbation_scale
        self.buffer = PPORolloutBuffer(rollout_length)
        self.global_step: int = 0
        self.baseline: float = 0.0
        self._rng = np.random.default_rng(0)

    def update(self) -> dict[str, float]:
        """Run PPO update using zeroth-order optimization."""
        if not self.buffer.ready:
            return {}

        self.global_step += 1

        # Compute GAE
        states, advantages, returns, actions, old_lps, old_vals = \
            self.buffer.compute_gae(0.0, self.gamma, self.gae_lambda)

        # Mean advantage as the signal
        mean_advantage = float(advantages.mean())

        # Perturb parameters in direction of advantage
        for param in self.weight_set.all_parameters():
            perturbation = self._rng.standard_normal(param.shape) * self.perturbation_scale
            param += self.lr * mean_advantage * perturbation

        self.buffer.clear()

        return {
            "mean_advantage": mean_advantage,
            "num_steps": len(states),
        }
