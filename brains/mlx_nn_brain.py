"""MLX NNBrain — MLX-accelerated MLP with REINFORCE policy gradient training.

Drop-in replacement for NNBrain that leverages Apple Silicon GPU via Metal.
Same architecture: Input(39) -> Dense(64, ReLU) -> Dense(32, ReLU) -> 11-head output.
Uses MLX auto-differentiation for REINFORCE gradients.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from agents.actions import AntAction
from agents.sensory import SensoryInput
from brains.nn_brain import (
    _DEPOSIT_CHANNELS,
    compute_reward,
)
from config import NNBrainConfig


# ---------------------------------------------------------------------------
# Role names
# ---------------------------------------------------------------------------

_ROLE_INDEX = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}


# ---------------------------------------------------------------------------
# MLX MLP model
# ---------------------------------------------------------------------------

class MLPModel(nn.Module):
    """Two-hidden-layer MLP with multi-head output (11 neurons)."""

    def __init__(self, input_dim: int = 39, h1: int = 64, h2: int = 32, output_dim: int = 11):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_out = nn.Linear(h2, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.fc1(x))
        h = nn.relu(self.fc2(h))
        return self.fc_out(h)


# ---------------------------------------------------------------------------
# Output decoding and action sampling
# ---------------------------------------------------------------------------

def _decode_output_mlx(raw: mx.array) -> dict:
    """Decode raw network output (11,) into action components as Python floats."""
    raw_np = raw.tolist()

    turn = math.tanh(raw_np[0]) * (math.pi / 6.0)
    speed = 1.0 / (1.0 + math.exp(-max(-500, min(500, raw_np[1]))))

    deposit_logits = raw_np[2:7]
    max_d = max(deposit_logits)
    exps = [math.exp(d - max_d) for d in deposit_logits]
    s = sum(exps)
    deposit_probs = [e / s for e in exps]

    strength = 1.0 / (1.0 + math.exp(-max(-500, min(500, raw_np[7]))))
    pickup = 1.0 / (1.0 + math.exp(-max(-500, min(500, raw_np[8]))))
    drop = 1.0 / (1.0 + math.exp(-max(-500, min(500, raw_np[9]))))
    recruit = 1.0 / (1.0 + math.exp(-max(-500, min(500, raw_np[10]))))

    return dict(
        turn=turn, speed=speed, deposit_probs=deposit_probs,
        strength=strength, pickup=pickup, drop=drop, recruit=recruit,
    )


def _sample_action(decoded: dict, rng_state: list) -> AntAction:
    """Sample a concrete AntAction from decoded output."""
    import random as _rng

    r = _rng.Random(rng_state[0])
    rng_state[0] += 1

    probs = decoded["deposit_probs"]
    deposit_idx = r.choices(range(len(probs)), weights=probs, k=1)[0]
    deposit_pheromone = _DEPOSIT_CHANNELS[deposit_idx]

    return AntAction(
        turn=decoded["turn"],
        speed_mult=decoded["speed"],
        deposit_pheromone=deposit_pheromone,
        deposit_strength=decoded["strength"] if deposit_pheromone is not None else 0.0,
        pickup=r.random() < decoded["pickup"],
        drop=r.random() < decoded["drop"],
        recruit_signal=r.random() < decoded["recruit"],
    )


def _log_prob_of_action(decoded: dict, action: AntAction) -> float:
    """Compute log-probability of the taken action under the policy."""
    lp = 0.0
    sigma = 0.1

    lp += -0.5 * ((action.turn - decoded["turn"]) / sigma) ** 2 - math.log(sigma)
    lp += -0.5 * ((action.speed_mult - decoded["speed"]) / sigma) ** 2 - math.log(sigma)

    probs = decoded["deposit_probs"]
    deposit_idx = _DEPOSIT_CHANNELS.index(action.deposit_pheromone)
    lp += math.log(max(probs[deposit_idx], 1e-8))

    lp += -0.5 * ((action.deposit_strength - decoded["strength"]) / sigma) ** 2 - math.log(sigma)

    for key, acted in [("pickup", action.pickup), ("drop", action.drop),
                       ("recruit", action.recruit_signal)]:
        p = max(min(decoded[key], 1.0 - 1e-8), 1e-8)
        lp += math.log(p) if acted else math.log(1.0 - p)

    return lp


# ---------------------------------------------------------------------------
# REINFORCE loss (for MLX auto-differentiation)
# ---------------------------------------------------------------------------

def _reinforce_loss(
    model: MLPModel,
    inputs: mx.array,
    target_logits_grad: mx.array,
) -> mx.array:
    """Pseudo-loss whose gradient equals the REINFORCE policy gradient."""
    logits = model(inputs)
    return -mx.sum(logits * mx.stop_gradient(target_logits_grad))


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

class _Experience:
    __slots__ = ("sensory_vec", "action", "log_prob", "reward")

    def __init__(self, sensory_vec: list[float], action: AntAction, log_prob: float, reward: float):
        self.sensory_vec = sensory_vec
        self.action = action
        self.log_prob = log_prob
        self.reward = reward


class _RollingBuffer:
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._buf: list[_Experience] = []
        self._pos: int = 0

    def add(self, exp: _Experience) -> None:
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

    def get_all(self) -> list[_Experience]:
        if len(self._buf) < self.capacity:
            return list(self._buf)
        return self._buf[self._pos:] + self._buf[:self._pos]


# ---------------------------------------------------------------------------
# Shared weight registry (one MLPModel per role)
# ---------------------------------------------------------------------------

class SharedWeightRegistry:
    """Manages shared MLPModels for each ant role."""

    def __init__(
        self,
        input_dim: int = 39,
        hidden_sizes: list[int] | None = None,
        seed: int | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.input_dim = input_dim
        self._models: dict[str, MLPModel] = {}

        mx.random.seed(seed if seed is not None else 42)
        for role in _ROLE_INDEX:
            self._models[role] = MLPModel(input_dim, hidden_sizes[0], hidden_sizes[1], 11)

    def get(self, role: str) -> MLPModel:
        return self._models[role]

    def roles(self) -> list[str]:
        return list(self._models.keys())


# ---------------------------------------------------------------------------
# Discounted returns
# ---------------------------------------------------------------------------

def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    n = len(rewards)
    returns = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(var) if var > 1e-16 else 1.0
    return [(r - mean) / std for r in returns]


# ---------------------------------------------------------------------------
# REINFORCE trainer (per-role)
# ---------------------------------------------------------------------------

class RoleTrainer:
    """REINFORCE trainer for a single role's shared MLX model."""

    def __init__(
        self,
        model: MLPModel,
        lr: float = 1e-4,
        gamma: float = 0.95,
        buffer_size: int = 200,
    ):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.buffer = _RollingBuffer(buffer_size)
        self.baseline: float = 0.0
        self.baseline_alpha: float = 0.01
        self.global_step: int = 0

    def update(self) -> float:
        """Run one REINFORCE update from the buffer."""
        experiences = self.buffer.get_all()
        if len(experiences) < 2:
            return 0.0

        self.global_step += 1
        batch_size = len(experiences)

        rewards = [e.reward for e in experiences]
        returns = _discounted_returns(rewards, self.gamma)
        mean_return = sum(returns) / len(returns)
        self.baseline = self.baseline * (1 - self.baseline_alpha) + mean_return * self.baseline_alpha
        advantages = [r - self.baseline for r in returns]

        inputs = mx.array([e.sensory_vec for e in experiences])
        logits = self.model(inputs)
        mx.eval(logits)  # mlx lazy eval, not python eval
        logits_list = logits.tolist()

        # Compute target logit gradient analytically
        target_grad = [[0.0] * 11 for _ in range(batch_size)]
        for i in range(batch_size):
            adv = advantages[i]
            z = logits_list[i]
            act = experiences[i].action

            t = math.tanh(z[0])
            scale = math.pi / 6.0
            sigma = 0.1
            target_grad[i][0] = (act.turn - t * scale) * scale * (1 - t**2) / (sigma**2) * adv

            sv = 1.0 / (1.0 + math.exp(-max(-500, min(500, z[1]))))
            target_grad[i][1] = (act.speed_mult - sv) * sv * (1 - sv) / (sigma**2) * adv

            dep_logits = z[2:7]
            max_d = max(dep_logits)
            exps = [math.exp(d - max_d) for d in dep_logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            deposit_idx = _DEPOSIT_CHANNELS.index(act.deposit_pheromone)
            for j in range(5):
                target_grad[i][2 + j] = (-probs[j] + (1.0 if j == deposit_idx else 0.0)) * adv

            st = 1.0 / (1.0 + math.exp(-max(-500, min(500, z[7]))))
            target_grad[i][7] = (act.deposit_strength - st) * st * (1 - st) / (sigma**2) * adv

            for j, acted in enumerate([act.pickup, act.drop, act.recruit_signal]):
                sig = 1.0 / (1.0 + math.exp(-max(-500, min(500, z[8 + j]))))
                target_grad[i][8 + j] = (float(acted) - sig) * adv

        target_grad_mx = mx.array(target_grad)

        loss_fn = nn.value_and_grad(self.model, _reinforce_loss)
        loss_val, grads = loss_fn(self.model, inputs, target_grad_mx)

        def _apply_grads(params, grads, lr):
            if isinstance(params, dict):
                return {k: _apply_grads(params[k], grads[k], lr) if k in grads else params[k]
                        for k in params}
            elif isinstance(params, list):
                return [_apply_grads(p, g, lr) for p, g in zip(params, grads)]
            else:
                return params - lr * grads

        new_params = _apply_grads(self.model.parameters(), grads, self.lr)
        self.model.update(new_params)
        mx.eval(self.model.parameters())  # mlx lazy eval

        return float(loss_val.item())


# ---------------------------------------------------------------------------
# MLXNNBrain — per-ant brain instance
# ---------------------------------------------------------------------------

class MLXNNBrain:
    """MLX-accelerated MLP brain implementing BrainBackend."""

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
        self._rng_state = [seed if seed is not None else 42]

        if registry is None:
            registry = SharedWeightRegistry(
                input_dim=39, hidden_sizes=cfg.hidden_sizes, seed=seed,
            )
        self.registry = registry
        self.model = registry.get(role)

        if trainer is None:
            trainer = RoleTrainer(
                self.model, lr=cfg.learning_rate, gamma=cfg.gamma,
                buffer_size=cfg.buffer_size,
            )
        self.trainer = trainer

        self._prev_sensory: SensoryInput | None = None
        self._prev_action: AntAction | None = None
        self._prev_log_prob: float = 0.0
        self._prev_vec: list[float] | None = None
        self._step_count: int = 0

    def decide(self, sensory: SensoryInput) -> AntAction:
        vec = sensory.to_vector()
        x = mx.array([vec])
        raw = self.model(x)
        mx.eval(raw)  # mlx lazy eval

        decoded = _decode_output_mlx(raw[0])
        action = _sample_action(decoded, self._rng_state)
        log_prob = _log_prob_of_action(decoded, action)

        self._prev_sensory = sensory
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_vec = vec

        return action

    def learn(self, reward: float) -> None:
        if self._prev_action is None or self._prev_vec is None:
            return

        exp = _Experience(
            sensory_vec=self._prev_vec,
            action=self._prev_action,
            log_prob=self._prev_log_prob,
            reward=reward,
        )
        self.trainer.buffer.add(exp)
        self._step_count += 1

        if (self._step_count % self.cfg.update_interval == 0
                and self.trainer.buffer.full):
            self.trainer.update()
