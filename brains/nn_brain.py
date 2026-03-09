"""NNBrain — pure NumPy MLP with REINFORCE policy gradient training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.actions import AntAction
from agents.sensory import SensoryInput
from config import NNBrainConfig


# ---------------------------------------------------------------------------
# NumPy MLP building blocks
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


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


# Pheromone channel names for deposit head (index 0 = none)
_DEPOSIT_CHANNELS: list[Optional[str]] = [None, "food", "home", "danger", "recruit"]

# Role names → index
_ROLE_INDEX = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}


# ---------------------------------------------------------------------------
# Weight set for one MLP
# ---------------------------------------------------------------------------

@dataclass
class MLPWeights:
    """Weights for a 2-hidden-layer MLP with multi-head output.

    Architecture:
        Input(input_dim) → Dense(h1, ReLU) → Dense(h2, ReLU) → multi-head output

    Output heads (total 11 neurons):
        turn(1, tanh), speed(1, sigmoid), deposit(5, softmax),
        strength(1, sigmoid), pickup(1, sigmoid), drop(1, sigmoid),
        recruit(1, sigmoid)

    Value head (for PPO): shared hidden layers → Dense(1)
    """
    # Hidden layers
    W1: np.ndarray = field(repr=False)  # (input_dim, h1)
    b1: np.ndarray = field(repr=False)  # (h1,)
    W2: np.ndarray = field(repr=False)  # (h1, h2)
    b2: np.ndarray = field(repr=False)  # (h2,)
    # Output layer
    W_out: np.ndarray = field(repr=False)  # (h2, 11)
    b_out: np.ndarray = field(repr=False)  # (11,)
    # Value head (shares hidden layers)
    W_val: np.ndarray = field(repr=False)  # (h2, 1)
    b_val: np.ndarray = field(repr=False)  # (1,)

    @classmethod
    def random_init(
        cls,
        input_dim: int,
        h1: int = 64,
        h2: int = 32,
        output_dim: int = 11,
        rng: np.random.Generator | None = None,
    ) -> MLPWeights:
        """He initialization for ReLU layers, small init for output."""
        if rng is None:
            rng = np.random.default_rng()
        return cls(
            W1=rng.normal(0, math.sqrt(2.0 / input_dim), (input_dim, h1)),
            b1=np.zeros(h1),
            W2=rng.normal(0, math.sqrt(2.0 / h1), (h1, h2)),
            b2=np.zeros(h2),
            W_out=rng.normal(0, 0.01, (h2, output_dim)),
            b_out=np.zeros(output_dim),
            W_val=rng.normal(0, 0.01, (h2, 1)),
            b_val=np.zeros(1),
        )

    def params(self) -> list[np.ndarray]:
        """All parameter arrays (for iteration) — policy only."""
        return [self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out]

    def all_params(self) -> list[np.ndarray]:
        """All parameter arrays including value head."""
        return [self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out,
                self.W_val, self.b_val]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

@dataclass
class ForwardCache:
    """Intermediate values stored during forward pass for backprop."""
    x: np.ndarray       # input (batch, input_dim)
    z1: np.ndarray      # pre-activation layer 1
    a1: np.ndarray      # post-activation layer 1 (ReLU)
    z2: np.ndarray      # pre-activation layer 2
    a2: np.ndarray      # post-activation layer 2 (ReLU)
    z_out: np.ndarray   # raw output logits (batch, 11)


def forward(weights: MLPWeights, x: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
    """Forward pass through MLP.

    Args:
        weights: MLP parameters.
        x: Input array, shape (batch, input_dim) or (input_dim,).

    Returns:
        raw_out: Raw logits, shape (batch, 11) or (11,).
        cache: Cached activations for backprop.
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x[np.newaxis, :]

    z1 = x @ weights.W1 + weights.b1
    a1 = _relu(z1)
    z2 = a1 @ weights.W2 + weights.b2
    a2 = _relu(z2)
    z_out = a2 @ weights.W_out + weights.b_out

    cache = ForwardCache(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z_out=z_out)

    if squeeze:
        return z_out[0], cache
    return z_out, cache


def forward_with_value(weights: MLPWeights, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, ForwardCache]:
    """Forward pass returning both policy logits and value estimate.

    Returns:
        raw_out: Raw logits, shape (batch, 11) or (11,).
        value: Value estimate, shape (batch,) or scalar.
        cache: Cached activations for backprop.
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x[np.newaxis, :]

    z1 = x @ weights.W1 + weights.b1
    a1 = _relu(z1)
    z2 = a1 @ weights.W2 + weights.b2
    a2 = _relu(z2)
    z_out = a2 @ weights.W_out + weights.b_out
    value = (a2 @ weights.W_val + weights.b_val).squeeze(-1)  # (batch,)

    cache = ForwardCache(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z_out=z_out)

    if squeeze:
        return z_out[0], float(value[0]), cache
    return z_out, value, cache


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


def sample_action(decoded: dict, rng: np.random.Generator) -> AntAction:
    """Sample a concrete AntAction from decoded network output (single ant)."""
    # Turn: use output directly (already scaled)
    turn_val = float(decoded["turn"][0])

    # Speed: use output directly
    speed_val = float(decoded["speed"][0])

    # Deposit: sample from categorical
    probs = decoded["deposit_probs"]
    deposit_idx = int(rng.choice(len(probs), p=probs))
    deposit_pheromone = _DEPOSIT_CHANNELS[deposit_idx]

    # Strength
    strength_val = float(decoded["strength"][0])

    # Binary decisions: sample from Bernoulli
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
# Log-probability computation (for REINFORCE)
# ---------------------------------------------------------------------------

def log_prob_of_action(decoded: dict, action: AntAction) -> float:
    """Compute log-probability of the taken action under the policy.

    For continuous outputs (turn, speed, strength), we treat the network output
    as the mean of a narrow Gaussian with fixed σ and compute log-prob.
    For discrete outputs (deposit, pickup, drop, recruit), we use the
    categorical/Bernoulli log-prob.
    """
    lp = 0.0

    # Turn: Gaussian log-prob with σ=0.1
    sigma_turn = 0.1
    turn_diff = action.turn - float(decoded["turn"][0])
    lp += -0.5 * (turn_diff / sigma_turn) ** 2 - math.log(sigma_turn)

    # Speed: Gaussian log-prob with σ=0.1
    sigma_speed = 0.1
    speed_diff = action.speed_mult - float(decoded["speed"][0])
    lp += -0.5 * (speed_diff / sigma_speed) ** 2 - math.log(sigma_speed)

    # Deposit: categorical log-prob
    probs = decoded["deposit_probs"]
    deposit_idx = _DEPOSIT_CHANNELS.index(action.deposit_pheromone)
    lp += math.log(max(float(probs[deposit_idx]), 1e-8))

    # Strength: Gaussian with σ=0.1
    sigma_str = 0.1
    str_diff = action.deposit_strength - float(decoded["strength"][0])
    lp += -0.5 * (str_diff / sigma_str) ** 2 - math.log(sigma_str)

    # Binary: Bernoulli log-prob
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
# REINFORCE gradient computation (analytical)
# ---------------------------------------------------------------------------

def _backprop_logits_gradient(
    weights: MLPWeights,
    cache: ForwardCache,
    dL_dz_out: np.ndarray,
) -> list[np.ndarray]:
    """Backprop through MLP given gradient of loss w.r.t. output logits.

    Args:
        weights: MLP weights.
        cache: Forward pass cache.
        dL_dz_out: (batch, 11) gradient of loss w.r.t. raw output logits.

    Returns:
        List of gradients [dW1, db1, dW2, db2, dW_out, db_out].
    """
    batch = cache.x.shape[0]

    # Output layer
    dW_out = cache.a2.T @ dL_dz_out / batch
    db_out = dL_dz_out.mean(axis=0)

    # Backprop to layer 2
    dL_da2 = dL_dz_out @ weights.W_out.T
    dL_dz2 = dL_da2 * (cache.z2 > 0).astype(float)  # ReLU derivative

    dW2 = cache.a1.T @ dL_dz2 / batch
    db2 = dL_dz2.mean(axis=0)

    # Backprop to layer 1
    dL_da1 = dL_dz2 @ weights.W2.T
    dL_dz1 = dL_da1 * (cache.z1 > 0).astype(float)

    dW1 = cache.x.T @ dL_dz1 / batch
    db1 = dL_dz1.mean(axis=0)

    return [dW1, db1, dW2, db2, dW_out, db_out]


def compute_reinforce_logit_grad(
    weights: MLPWeights,
    cache: ForwardCache,
    actions: list[AntAction],
    advantages: np.ndarray,
) -> list[np.ndarray]:
    """Compute REINFORCE policy gradient through analytical backprop.

    For each sample, computes ∂log π(a|s) / ∂θ * advantage, then averages.
    We compute ∂log π / ∂z_out analytically and backprop through MLP.
    """
    batch = cache.z_out.shape[0] if cache.z_out.ndim > 1 else 1
    z_out = cache.z_out if cache.z_out.ndim > 1 else cache.z_out[np.newaxis, :]

    # Build dlog_pi / dz_out for each sample
    dL_dz = np.zeros_like(z_out)

    for i in range(batch):
        adv = float(advantages[i]) if advantages.ndim > 0 else float(advantages)

        # Turn head (idx 0): tanh output, Gaussian log-prob
        # d/dz[turn_logprob] = d/dz[ -(a - tanh(z)*scale)^2 / (2σ^2) ]
        #   = (a - tanh(z)*scale) * scale * (1 - tanh(z)^2) / σ^2
        t = np.tanh(z_out[i, 0])
        scale = math.pi / 6.0
        sigma_t = 0.1
        turn_target = actions[i].turn
        dL_dz[i, 0] = (turn_target - t * scale) * scale * (1 - t**2) / (sigma_t**2) * adv

        # Speed head (idx 1): sigmoid, Gaussian log-prob
        s = _sigmoid(z_out[i, 1:2])[0]
        sigma_s = 0.1
        speed_target = actions[i].speed_mult
        dL_dz[i, 1] = (speed_target - s) * s * (1 - s) / (sigma_s**2) * adv

        # Deposit head (idx 2..6): softmax, categorical log-prob
        # d/dz[log softmax_k] = 1{j==k} - softmax_j
        probs = _softmax(z_out[i, 2:7])
        deposit_idx = _DEPOSIT_CHANNELS.index(actions[i].deposit_pheromone)
        grad_deposit = -probs.copy()
        grad_deposit[deposit_idx] += 1.0
        dL_dz[i, 2:7] = grad_deposit * adv

        # Strength head (idx 7): sigmoid, Gaussian log-prob
        st = _sigmoid(z_out[i, 7:8])[0]
        sigma_st = 0.1
        str_target = actions[i].deposit_strength
        dL_dz[i, 7] = (str_target - st) * st * (1 - st) / (sigma_st**2) * adv

        # Binary heads (idx 8, 9, 10): sigmoid, Bernoulli log-prob
        # d/dz[log Bernoulli] = (a - σ(z)) where a∈{0,1}
        for j, acted in enumerate([actions[i].pickup, actions[i].drop,
                                   actions[i].recruit_signal]):
            sig_val = _sigmoid(z_out[i, 8 + j:9 + j])[0]
            dL_dz[i, 8 + j] = (float(acted) - sig_val) * adv

    return _backprop_logits_gradient(weights, cache, dL_dz)


# ---------------------------------------------------------------------------
# Reward signal computation
# ---------------------------------------------------------------------------

def compute_reward(
    prev_sensory: SensoryInput | None,
    curr_sensory: SensoryInput,
    action: AntAction,
    alive: bool,
) -> float:
    """Compute reward signal for a single ant transition.

    Rewards:
        +1.0  food deposited at nest (was carrying, now not, at nest)
        +0.3  picked up food (was not carrying, now carrying)
        -0.5  death (energy depleted)
        +0.1  energy efficiency (energy > 50)
        -0.1  low energy penalty (energy < 20)
        +0.05 exploration (moving at reasonable speed)
        +0.15 approaching food (food distance decreased while not carrying)
        +0.1  approaching nest (nest distance decreased while carrying food)
    """
    reward = 0.0

    if not alive:
        return -0.5

    if prev_sensory is not None:
        # Food deposited: was carrying food, now not
        if prev_sensory.carrying == "food" and curr_sensory.carrying is None:
            reward += 1.0

        # Picked up food: was not carrying, now carrying food
        if prev_sensory.carrying is None and curr_sensory.carrying == "food":
            reward += 0.3

        # Approaching food: strong signal for getting closer when not carrying
        if curr_sensory.carrying is None and curr_sensory.nearest_food_distance < 1.0:
            food_dist_delta = prev_sensory.nearest_food_distance - curr_sensory.nearest_food_distance
            if food_dist_delta > 0.001:
                reward += 0.2  # flat bonus for closing distance
            elif food_dist_delta < -0.001:
                reward -= 0.05  # mild penalty for moving away from food

        # Approaching nest: reward getting closer when carrying food
        if curr_sensory.carrying == "food":
            nest_dist_prev = prev_sensory.nest_distance
            nest_dist_curr = curr_sensory.nest_distance
            if nest_dist_curr < nest_dist_prev - 0.5:
                reward += 0.2  # flat bonus for heading home with food
            elif nest_dist_curr > nest_dist_prev + 0.5:
                reward -= 0.05  # mild penalty for wandering away from nest

    # Energy management
    energy_norm = curr_sensory.energy / 100.0
    if energy_norm > 0.5:
        reward += 0.1
    elif energy_norm < 0.2:
        reward -= 0.1

    # Exploration bonus
    if action.speed_mult > 0.3:
        reward += 0.05

    return reward


# ---------------------------------------------------------------------------
# Shared weight registry (one set per role)
# ---------------------------------------------------------------------------

class SharedWeightRegistry:
    """Manages shared MLPWeights for each ant role.

    All ants of the same role share a single weight set.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_sizes: list[int] | None = None,
        seed: int | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.rng = np.random.default_rng(seed)
        self._weights: dict[str, MLPWeights] = {}

        # Pre-create weights for all four roles
        for role in _ROLE_INDEX:
            self._weights[role] = MLPWeights.random_init(
                input_dim, hidden_sizes[0], hidden_sizes[1], 11, self.rng,
            )

    def get(self, role: str) -> MLPWeights:
        return self._weights[role]

    def roles(self) -> list[str]:
        return list(self._weights.keys())


# ---------------------------------------------------------------------------
# Rolling experience buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """Single step of experience for REINFORCE."""
    sensory_vec: np.ndarray     # input vector
    action: AntAction           # action taken
    log_prob: float             # log π(a|s)
    reward: float               # reward received


class RollingBuffer:
    """Fixed-size circular buffer of experience steps."""

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._buf: list[Experience] = []
        self._pos: int = 0

    def add(self, exp: Experience) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(exp)
        else:
            self._buf[self._pos] = exp
        self._pos = (self._pos + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) >= self.capacity

    def get_all(self) -> list[Experience]:
        """Return all stored experiences in insertion order."""
        if len(self._buf) < self.capacity:
            return list(self._buf)
        # Return from oldest to newest
        return self._buf[self._pos:] + self._buf[:self._pos]

    def clear(self) -> None:
        self._buf.clear()
        self._pos = 0


# ---------------------------------------------------------------------------
# REINFORCE trainer (per-role)
# ---------------------------------------------------------------------------

class RoleTrainer:
    """REINFORCE trainer for a single role's shared weights."""

    def __init__(
        self,
        weights: MLPWeights,
        lr: float = 1e-4,
        gamma: float = 0.95,
        buffer_size: int = 200,
        warmup_steps: int = 50,
    ):
        self.weights = weights
        self.lr = lr
        self.gamma = gamma
        self.buffer = RollingBuffer(buffer_size)
        self.baseline: float = 0.0       # running mean return
        self.baseline_alpha: float = 0.01  # EMA factor
        self.warmup_steps = warmup_steps
        self.global_step: int = 0

    @property
    def effective_lr(self) -> float:
        """Learning rate with linear warmup."""
        if self.global_step >= self.warmup_steps:
            return self.lr
        return self.lr * (self.global_step / max(self.warmup_steps, 1))

    def update(self) -> float:
        """Run one REINFORCE update from the buffer. Returns mean loss."""
        experiences = self.buffer.get_all()
        if len(experiences) < 2:
            return 0.0

        self.global_step += 1

        # Compute discounted returns
        rewards = np.array([e.reward for e in experiences])
        returns = _discounted_returns(rewards, self.gamma)

        # Update baseline
        mean_return = float(returns.mean())
        self.baseline = (
            self.baseline * (1 - self.baseline_alpha)
            + mean_return * self.baseline_alpha
        )

        # Advantages
        advantages = returns - self.baseline

        # Batch forward pass for gradient computation
        inputs = np.array([e.sensory_vec for e in experiences])
        actions = [e.action for e in experiences]
        _, cache = forward(self.weights, inputs)

        # Compute gradients via analytical backprop
        grads = compute_reinforce_logit_grad(
            self.weights, cache, actions, advantages,
        )

        # Apply gradient ascent (maximize expected return)
        lr = self.effective_lr
        for param, grad in zip(self.weights.params(), grads):
            param += lr * grad

        return float(-np.mean(advantages ** 2))  # proxy loss


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns for each timestep."""
    n = len(rewards)
    returns = np.zeros(n)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    # Normalize returns
    std = returns.std()
    if std > 1e-8:
        returns = (returns - returns.mean()) / std
    return returns


# ---------------------------------------------------------------------------
# Batch forward pass for all ants of a role
# ---------------------------------------------------------------------------

def batch_decide(
    weights: MLPWeights,
    sensory_inputs: list[SensoryInput],
    rng: np.random.Generator,
) -> list[tuple[AntAction, float, np.ndarray]]:
    """Batch forward pass for multiple ants sharing the same weights.

    Returns list of (action, log_prob, sensory_vec) tuples.
    """
    if not sensory_inputs:
        return []

    # Encode all sensory inputs
    vecs = np.array([s.to_vector() for s in sensory_inputs])

    # Batch forward
    raw_out, _ = forward(weights, vecs)
    if raw_out.ndim == 1:
        raw_out = raw_out[np.newaxis, :]

    results = []
    for i in range(len(sensory_inputs)):
        row = raw_out[i]
        decoded = decode_output(row)
        action = sample_action(decoded, rng)
        lp = log_prob_of_action(decoded, action)
        results.append((action, lp, vecs[i]))

    return results


# ---------------------------------------------------------------------------
# NNBrain — per-ant brain instance (thin wrapper around shared weights)
# ---------------------------------------------------------------------------

class NNBrain:
    """Neural network brain for a single ant.

    Uses shared weights from a SharedWeightRegistry (per role) and
    maintains a per-ant experience buffer for REINFORCE training.

    Satisfies the BrainBackend protocol (decide + learn).
    """

    def __init__(
        self,
        role: str = "forager",
        registry: SharedWeightRegistry | None = None,
        trainer: RoleTrainer | None = None,
        cfg: NNBrainConfig | None = None,
        seed: int | None = None,
    ):
        if cfg is None:
            cfg = NNBrainConfig()

        self.role = role
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Shared weight registry
        if registry is None:
            registry = SharedWeightRegistry(
                input_dim=39,
                hidden_sizes=cfg.hidden_sizes,
                seed=seed,
            )
        self.registry = registry
        self.weights = registry.get(role)

        # Per-role trainer (may be shared across ants of same role)
        if trainer is None:
            trainer = RoleTrainer(
                self.weights,
                lr=cfg.learning_rate,
                gamma=cfg.gamma,
                buffer_size=cfg.buffer_size,
            )
        self.trainer = trainer

        # Per-ant state
        self._prev_sensory: SensoryInput | None = None
        self._prev_action: AntAction | None = None
        self._prev_log_prob: float = 0.0
        self._prev_vec: np.ndarray | None = None
        self._step_count: int = 0

    def decide(self, sensory: SensoryInput) -> AntAction:
        """Decide an action given sensory input. Satisfies BrainBackend."""
        vec = np.array(sensory.to_vector(), dtype=np.float64)
        raw_out, _ = forward(self.weights, vec)
        decoded = decode_output(raw_out)
        action = sample_action(decoded, self.rng)
        log_prob = log_prob_of_action(decoded, action)

        # Store for learning
        self._prev_sensory = sensory
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_vec = vec

        return action

    def learn(self, reward: float) -> None:
        """Receive reward signal. Stores experience and periodically trains."""
        if self._prev_action is None or self._prev_vec is None:
            return

        exp = Experience(
            sensory_vec=self._prev_vec,
            action=self._prev_action,
            log_prob=self._prev_log_prob,
            reward=reward,
        )
        self.trainer.buffer.add(exp)
        self._step_count += 1

        # Periodically update
        if (self._step_count % self.cfg.update_interval == 0
                and self.trainer.buffer.full):
            self.trainer.update()

    def compute_reward(
        self,
        prev_sensory: SensoryInput | None,
        curr_sensory: SensoryInput,
        action: AntAction,
        alive: bool = True,
    ) -> float:
        """Compute reward for a transition (convenience wrapper)."""
        return compute_reward(prev_sensory, curr_sensory, action, alive)
