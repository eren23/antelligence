"""TorchTransformerBrain — PyTorch causal transformer with REINFORCE training.

Replaces both transformer_brain.py (forward) and transformer_backprop.py (backward)
with a single file using PyTorch autograd. Uses nn.TransformerEncoder for the
transformer stack and autograd for all gradient computation.

Same 11-head action output as NNBrain / TorchNNBrain.
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
from brains.nn_brain import (
    _DEPOSIT_CHANNELS,
    compute_reward,
    decode_output,
    log_prob_of_action,
    sample_action,
)
from brains.torch_utils import get_device, torch_log_prob_of_action
from config import TransformerBrainConfig


# ---------------------------------------------------------------------------
# Role names
# ---------------------------------------------------------------------------

_ROLE_INDEX = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}
_OUTPUT_DIM = 11


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# PyTorch Transformer Model
# ---------------------------------------------------------------------------

class TorchTransformerModel(nn.Module):
    """Causal transformer with policy + value heads.

    Architecture matches the NumPy TransformerWeightSet:
        input_proj → positional encoding → TransformerEncoder → last-token → heads
    """

    def __init__(
        self,
        input_dim: int = 39,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 64,
        output_dim: int = 11,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.policy_head = nn.Linear(d_model, output_dim)
        self.value_head = nn.Linear(d_model, 1)

        # Fixed positional encoding
        self.register_buffer("pos_enc", _sinusoidal_pe(max_seq_len, d_model))

        # Init
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.policy_head.weight, std=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) or (seq_len, input_dim)

        Returns:
            policy_logits: (batch, output_dim) or (output_dim,)
            value: (batch,) or scalar
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)

        seq_len = x.size(1)

        h = self.input_proj(x)  # (B, seq, d_model)
        h = h + self.pos_enc[:seq_len].unsqueeze(0)

        # Causal mask
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device,
            )

        h = self.transformer(h, mask=mask, is_causal=True)

        # Last token
        last = h[:, -1, :]  # (B, d_model)
        policy_logits = self.policy_head(last)  # (B, output_dim)
        value = self.value_head(last).squeeze(-1)  # (B,)

        if squeeze:
            return policy_logits.squeeze(0), value.squeeze(0)
        return policy_logits, value


# ---------------------------------------------------------------------------
# Experience buffer (reuse same pattern)
# ---------------------------------------------------------------------------

class _Experience:
    __slots__ = ("context_flat", "action", "log_prob", "reward", "seq_len")

    def __init__(
        self,
        context_flat: np.ndarray,
        action: AntAction,
        log_prob: float,
        reward: float,
        seq_len: int,
    ):
        self.context_flat = context_flat
        self.action = action
        self.log_prob = log_prob
        self.reward = reward
        self.seq_len = seq_len


class _RollingBuffer:
    def __init__(self, capacity: int = 500):
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
# Shared weight registry
# ---------------------------------------------------------------------------

class TorchTransformerSharedWeightRegistry:
    """Manages shared TorchTransformerModels for each ant role."""

    def __init__(
        self,
        input_dim: int = 39,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 64,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self._device = get_device()
        self._models: dict[str, TorchTransformerModel] = {}

        if seed is not None:
            torch.manual_seed(seed)
        for role in _ROLE_INDEX:
            model = TorchTransformerModel(
                input_dim, d_model, n_heads, n_layers, ffn_dim, _OUTPUT_DIM,
            )
            model.to(self._device)
            self._models[role] = model

    def get(self, role: str) -> TorchTransformerModel:
        return self._models[role]

    def roles(self) -> list[str]:
        return list(self._models.keys())


# ---------------------------------------------------------------------------
# REINFORCE trainer (per-role)
# ---------------------------------------------------------------------------

class TorchTransformerRoleTrainer:
    """REINFORCE trainer for a single role's shared transformer model."""

    def __init__(
        self,
        model: TorchTransformerModel,
        lr: float = 5e-5,
        gamma: float = 0.95,
        buffer_size: int = 500,
        warmup_steps: int = 50,
        context_length: int = 16,
    ):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.buffer = _RollingBuffer(buffer_size)
        self.baseline: float = 0.0
        self.baseline_alpha: float = 0.01
        self.warmup_steps = warmup_steps
        self.global_step: int = 0
        self.context_length = context_length
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._device = next(model.parameters()).device
        self._input_dim = model.input_proj.in_features

    @property
    def effective_lr(self) -> float:
        if self.global_step >= self.warmup_steps:
            return self.lr
        return self.lr * (self.global_step / max(self.warmup_steps, 1))

    def update(self) -> float:
        """Run one REINFORCE update. Returns proxy loss."""
        experiences = self.buffer.get_all()
        if len(experiences) < 2:
            return 0.0

        self.global_step += 1

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

        # Accumulate loss over experiences (each has different seq_len)
        total_loss = torch.tensor(0.0, device=self._device)
        count = 0

        for exp, adv in zip(experiences, advantages):
            # Reconstruct context sequence
            vec = exp.context_flat
            seq_len = exp.seq_len
            if len(vec) >= self._input_dim * seq_len:
                x_np = vec[: self._input_dim * seq_len].reshape(seq_len, self._input_dim)
            else:
                x_np = vec[: self._input_dim].reshape(1, self._input_dim)

            x = torch.tensor(x_np, dtype=torch.float32, device=self._device)
            logits, _ = self.model(x)  # (output_dim,)

            lp = torch_log_prob_of_action(logits, exp.action)
            total_loss = total_loss - adv * lp
            count += 1

        if count > 0:
            total_loss = total_loss / count
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return float(total_loss.item()) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# TorchTransformerBrain — per-ant brain instance
# ---------------------------------------------------------------------------

class TorchTransformerBrain:
    """PyTorch causal transformer brain implementing BrainBackend protocol.

    Maintains a sliding context window and uses a transformer encoder to
    produce actions. Same multi-head output as TorchNNBrain.
    """

    def __init__(
        self,
        role: str = "forager",
        registry: TorchTransformerSharedWeightRegistry | None = None,
        trainer: TorchTransformerRoleTrainer | None = None,
        cfg: TransformerBrainConfig | None = None,
        input_dim: int = 39,
        seed: int = 42,
    ):
        if cfg is None:
            cfg = TransformerBrainConfig()

        self.cfg = cfg
        self.role = role
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)

        if registry is None:
            registry = TorchTransformerSharedWeightRegistry(
                input_dim=input_dim,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                ffn_dim=cfg.ffn_dim,
                seed=seed,
            )
        self.registry = registry
        self.model = registry.get(role)

        if trainer is None:
            trainer = TorchTransformerRoleTrainer(
                self.model,
                lr=cfg.learning_rate,
                gamma=0.95,
                buffer_size=cfg.buffer_size,
                context_length=cfg.context_length,
            )
        self.trainer = trainer

        # Sliding context window
        self.context: deque[np.ndarray] = deque(maxlen=cfg.context_length)

        # Per-ant state
        self._prev_sensory: SensoryInput | None = None
        self._prev_action: AntAction | None = None
        self._prev_log_prob: float = 0.0
        self._prev_context_flat: np.ndarray | None = None
        self._prev_seq_len: int = 0
        self._step_count: int = 0

    def decide(self, sensory: SensoryInput) -> AntAction:
        vec = np.array(sensory.to_vector(), dtype=np.float64)
        self.context.append(vec)

        ctx = np.stack(list(self.context), axis=0)  # (seq_len, input_dim)

        with torch.no_grad():
            x = torch.tensor(ctx, dtype=torch.float32, device=next(self.model.parameters()).device)
            logits, _ = self.model(x)  # (output_dim,)
            raw_np = logits.cpu().numpy()

        decoded = decode_output(raw_np)
        action = sample_action(decoded, self.rng)
        lp = log_prob_of_action(decoded, action)

        self._prev_sensory = sensory
        self._prev_action = action
        self._prev_log_prob = lp
        self._prev_context_flat = ctx.flatten()
        self._prev_seq_len = ctx.shape[0]
        self._step_count += 1

        return action

    def learn(self, reward: float) -> None:
        if self._prev_action is None or self._prev_context_flat is None:
            return

        exp = _Experience(
            context_flat=self._prev_context_flat,
            action=self._prev_action,
            log_prob=self._prev_log_prob,
            reward=reward,
            seq_len=self._prev_seq_len,
        )
        self.trainer.buffer.add(exp)

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
