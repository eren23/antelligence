"""TorchNNBrain — PyTorch MLP with REINFORCE policy gradient training.

Drop-in replacement for NNBrain that leverages PyTorch autograd.
Same architecture: Input(39) → Dense(64, ReLU) → Dense(32, ReLU) → multi-head output.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from agents.actions import AntAction
from agents.sensory import SensoryInput
from brains.action_utils import (
    _DEPOSIT_CHANNELS,
    compute_reward,
    decode_output,
    log_prob_of_action,
    sample_action,
)
from brains.torch_utils import get_device, torch_log_prob_of_action
from config import NNBrainConfig


# ---------------------------------------------------------------------------
# Role names
# ---------------------------------------------------------------------------

_ROLE_INDEX = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}


# ---------------------------------------------------------------------------
# PyTorch MLP Model
# ---------------------------------------------------------------------------

class TorchMLPModel(nn.Module):
    """Policy + Value MLP. Same architecture as NumPy MLPWeights."""

    def __init__(
        self,
        input_dim: int = 39,
        hidden1: int = 64,
        hidden2: int = 32,
        output_dim: int = 11,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden2, output_dim)
        self.value_head = nn.Linear(hidden2, 1)

        # He initialization for hidden layers, small init for heads
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.policy_head.weight, std=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value_estimate)."""
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

class _Experience:
    __slots__ = ("sensory_vec", "action", "log_prob", "reward")

    def __init__(
        self,
        sensory_vec: np.ndarray,
        action: AntAction,
        log_prob: float,
        reward: float,
    ):
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

    def clear(self) -> None:
        self._buf.clear()
        self._pos = 0


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
# Shared weight registry (one TorchMLPModel per role)
# ---------------------------------------------------------------------------

class TorchSharedWeightRegistry:
    """Manages shared TorchMLPModels for each ant role."""

    def __init__(
        self,
        input_dim: int = 39,
        hidden_sizes: list[int] | None = None,
        seed: int | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.input_dim = input_dim
        self._device = get_device()
        self._models: dict[str, TorchMLPModel] = {}

        if seed is not None:
            torch.manual_seed(seed)
        for role in _ROLE_INDEX:
            model = TorchMLPModel(input_dim, hidden_sizes[0], hidden_sizes[1], 11)
            model.to(self._device)
            self._models[role] = model

    def get(self, role: str) -> TorchMLPModel:
        return self._models[role]

    def roles(self) -> list[str]:
        return list(self._models.keys())


# ---------------------------------------------------------------------------
# REINFORCE trainer (per-role)
# ---------------------------------------------------------------------------

class TorchRoleTrainer:
    """REINFORCE trainer for a single role's shared PyTorch model."""

    def __init__(
        self,
        model: TorchMLPModel,
        lr: float = 1e-4,
        gamma: float = 0.95,
        buffer_size: int = 200,
        warmup_steps: int = 50,
    ):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.buffer = _RollingBuffer(buffer_size)
        self.baseline: float = 0.0
        self.baseline_alpha: float = 0.01
        self.warmup_steps = warmup_steps
        self.global_step: int = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._device = next(model.parameters()).device

    @property
    def effective_lr(self) -> float:
        if self.global_step >= self.warmup_steps:
            return self.lr
        return self.lr * (self.global_step / max(self.warmup_steps, 1))

    def update(self) -> float:
        """Run one REINFORCE update from the buffer. Returns proxy loss."""
        experiences = self.buffer.get_all()
        if len(experiences) < 2:
            return 0.0

        self.global_step += 1

        # Adjust optimizer LR for warmup
        eff_lr = self.effective_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = eff_lr

        rewards = [e.reward for e in experiences]
        returns = _discounted_returns(rewards, self.gamma)
        mean_return = sum(returns) / len(returns)
        self.baseline = (
            self.baseline * (1 - self.baseline_alpha)
            + mean_return * self.baseline_alpha
        )
        advantages = [r - self.baseline for r in returns]

        # Batch forward
        inputs = torch.tensor(
            [e.sensory_vec.tolist() for e in experiences],
            dtype=torch.float32,
            device=self._device,
        )
        logits, _ = self.model(inputs)

        # Compute REINFORCE loss: -Σ advantage_i * log_prob_i
        loss = torch.tensor(0.0, device=self._device)
        for i, (exp, adv) in enumerate(zip(experiences, advantages)):
            lp = torch_log_prob_of_action(logits[i], exp.action)
            loss = loss - adv * lp

        loss = loss / len(experiences)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())


# ---------------------------------------------------------------------------
# TorchNNBrain — per-ant brain instance
# ---------------------------------------------------------------------------

class TorchNNBrain:
    """PyTorch MLP brain implementing BrainBackend protocol.

    Uses shared weights from a TorchSharedWeightRegistry (per role) and
    maintains a per-ant experience buffer for REINFORCE training.
    """

    def __init__(
        self,
        role: str = "forager",
        registry: TorchSharedWeightRegistry | None = None,
        trainer: TorchRoleTrainer | None = None,
        cfg: NNBrainConfig | None = None,
        seed: int | None = None,
    ):
        if cfg is None:
            cfg = NNBrainConfig()

        self.role = role
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        if registry is None:
            registry = TorchSharedWeightRegistry(
                input_dim=39, hidden_sizes=cfg.hidden_sizes, seed=seed,
            )
        self.registry = registry
        self.model = registry.get(role)

        if trainer is None:
            trainer = TorchRoleTrainer(
                self.model,
                lr=cfg.learning_rate,
                gamma=cfg.gamma,
                buffer_size=cfg.buffer_size,
            )
        self.trainer = trainer

        self._prev_sensory: SensoryInput | None = None
        self._prev_action: AntAction | None = None
        self._prev_log_prob: float = 0.0
        self._prev_vec: np.ndarray | None = None
        self._step_count: int = 0

    def decide(self, sensory: SensoryInput) -> AntAction:
        vec = np.array(sensory.to_vector(), dtype=np.float64)

        # Forward pass (no grad needed for action sampling)
        with torch.no_grad():
            x = torch.tensor(vec, dtype=torch.float32, device=next(self.model.parameters()).device)
            logits, _ = self.model(x.unsqueeze(0))
            raw_np = logits[0].cpu().numpy()

        # Reuse NumPy-side action decoding
        decoded = decode_output(raw_np)
        action = sample_action(decoded, self.rng)
        lp = log_prob_of_action(decoded, action)

        self._prev_sensory = sensory
        self._prev_action = action
        self._prev_log_prob = lp
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

    def compute_reward(
        self,
        prev_sensory: SensoryInput | None,
        curr_sensory: SensoryInput,
        action: AntAction,
        alive: bool = True,
    ) -> float:
        return compute_reward(prev_sensory, curr_sensory, action, alive)
