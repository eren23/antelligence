"""MLX TransformerBrain — MLX-accelerated causal transformer with REINFORCE training.

Drop-in replacement for TransformerBrain that leverages Apple Silicon GPU via Metal.
Same architecture: context_length=16, d_model=32, n_heads=4, n_layers=2, ffn_dim=64.
Uses MLX auto-differentiation for REINFORCE gradients.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from agents.actions import AntAction
from agents.sensory import SensoryInput
from brains.nn_brain import (
    _DEPOSIT_CHANNELS,
    compute_reward,
)
from brains.mlx_nn_brain import (
    _decode_output_mlx,
    _sample_action,
    _log_prob_of_action,
    _Experience,
    _RollingBuffer,
    _discounted_returns,
)
from config import TransformerBrainConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIM = 11
_ROLE_INDEX = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}


# ---------------------------------------------------------------------------
# MLX Transformer model
# ---------------------------------------------------------------------------

class CausalTransformerModel(nn.Module):
    """Causal transformer with learned input/output projections."""

    def __init__(
        self,
        input_dim: int = 39,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 64,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = [
            _TransformerBlock(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ]

        self.output_proj = nn.Linear(d_model, _OUTPUT_DIM)

        # Sinusoidal positional encoding (fixed)
        pe = _sinusoidal_pe(max_seq_len, d_model)
        self._pos_enc = mx.array(pe)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. x: (..., seq_len, input_dim) -> (..., OUTPUT_DIM)."""
        seq_len = x.shape[-2]
        h = self.input_proj(x) + self._pos_enc[:seq_len]

        mask = _create_additive_causal_mask(seq_len)

        for layer in self.layers:
            h = layer(h, mask)

        last_h = h[..., -1, :]
        return self.output_proj(last_h)


class _TransformerBlock(nn.Module):
    """MHA -> Add & LayerNorm -> FFN (GELU) -> Add & LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        attn_out = self.attn(x, x, x, mask=mask)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn2(nn.gelu(self.ffn1(x)))
        return self.ln2(x + ffn_out)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sinusoidal_pe(max_len: int, d_model: int) -> list[list[float]]:
    pe = [[0.0] * d_model for _ in range(max_len)]
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            div = math.exp(i * -(math.log(10000.0) / d_model))
            pe[pos][i] = math.sin(pos * div)
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(pos * div)
    return pe


def _create_additive_causal_mask(seq_len: int) -> mx.array:
    """Create additive causal mask for MLX MultiHeadAttention."""
    indices = mx.arange(seq_len)
    mask = indices[None, :] > indices[:, None]
    return mask * mx.array(-1e9)


# ---------------------------------------------------------------------------
# REINFORCE loss
# ---------------------------------------------------------------------------

def _reinforce_loss(
    model: CausalTransformerModel,
    inputs: mx.array,
    target_logits_grad: mx.array,
) -> mx.array:
    logits = model(inputs)
    return -mx.sum(logits * mx.stop_gradient(target_logits_grad))


# ---------------------------------------------------------------------------
# Shared weight registry
# ---------------------------------------------------------------------------

class SharedWeightRegistry:
    """Manages shared CausalTransformerModels for each ant role."""

    def __init__(
        self,
        input_dim: int = 39,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 64,
        seed: int | None = None,
    ):
        self._models: dict[str, CausalTransformerModel] = {}

        mx.random.seed(seed if seed is not None else 42)
        for role in _ROLE_INDEX:
            self._models[role] = CausalTransformerModel(
                input_dim, d_model, n_heads, n_layers, ffn_dim,
            )

    def get(self, role: str) -> CausalTransformerModel:
        return self._models[role]

    def roles(self) -> list[str]:
        return list(self._models.keys())


# ---------------------------------------------------------------------------
# REINFORCE trainer
# ---------------------------------------------------------------------------

class RoleTrainer:
    """REINFORCE trainer for a single role's shared MLX transformer model."""

    def __init__(
        self,
        model: CausalTransformerModel,
        lr: float = 5e-5,
        gamma: float = 0.95,
        buffer_size: int = 500,
        context_length: int = 16,
        input_dim: int = 39,
    ):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.buffer = _RollingBuffer(buffer_size)
        self.baseline: float = 0.0
        self.baseline_alpha: float = 0.01
        self.global_step: int = 0
        self.context_length = context_length
        self.input_dim = input_dim

    def update(self) -> float:
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

        # Rebuild context sequences from stored flat vectors
        input_dim = self.input_dim
        batch_inputs = []
        for exp in experiences:
            flat = exp.sensory_vec
            total = len(flat)
            seq_len = total // input_dim
            seq = [flat[j * input_dim:(j + 1) * input_dim] for j in range(seq_len)]
            batch_inputs.append(seq)

        max_seq = max(len(s) for s in batch_inputs)
        padded = []
        for seq in batch_inputs:
            pad_len = max_seq - len(seq)
            padded_seq = [[0.0] * input_dim] * pad_len + seq
            padded.append(padded_seq)

        inputs = mx.array(padded)
        logits = self.model(inputs)
        mx.eval(logits)  # force mlx lazy evaluation
        logits_list = logits.tolist()

        target_grad = [[0.0] * _OUTPUT_DIM for _ in range(batch_size)]
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
            exps_d = [math.exp(d - max_d) for d in dep_logits]
            s = sum(exps_d)
            probs = [e / s for e in exps_d]
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
        mx.eval(self.model.parameters())  # force mlx lazy evaluation

        return float(loss_val.item())


# ---------------------------------------------------------------------------
# MLXTransformerBrain — per-ant brain instance
# ---------------------------------------------------------------------------

class MLXTransformerBrain:
    """MLX-accelerated causal transformer brain implementing BrainBackend."""

    def __init__(
        self,
        role: str = "forager",
        registry: SharedWeightRegistry | None = None,
        trainer: RoleTrainer | None = None,
        cfg: TransformerBrainConfig | None = None,
        input_dim: int = 39,
        seed: int = 42,
    ):
        if cfg is None:
            cfg = TransformerBrainConfig()
        self.cfg = cfg
        self.role = role
        self.input_dim = input_dim
        self._rng_state = [seed]

        if registry is None:
            registry = SharedWeightRegistry(
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
            trainer = RoleTrainer(
                self.model,
                lr=cfg.learning_rate,
                gamma=0.95,
                buffer_size=cfg.buffer_size,
                context_length=cfg.context_length,
                input_dim=input_dim,
            )
        self.trainer = trainer

        self.context: deque[list[float]] = deque(maxlen=cfg.context_length)

        self._prev_sensory: SensoryInput | None = None
        self._prev_action: AntAction | None = None
        self._prev_log_prob: float = 0.0
        self._prev_context_flat: list[float] | None = None
        self._step_count: int = 0

    def decide(self, sensory: SensoryInput) -> AntAction:
        vec = sensory.to_vector()
        self.context.append(vec)

        ctx_list = list(self.context)
        ctx = mx.array([ctx_list])  # (1, seq_len, input_dim)
        logits = self.model(ctx)
        mx.eval(logits)  # force mlx lazy evaluation

        decoded = _decode_output_mlx(logits[0])
        action = _sample_action(decoded, self._rng_state)
        log_prob = _log_prob_of_action(decoded, action)

        flat: list[float] = []
        for v in ctx_list:
            flat.extend(v)

        self._prev_sensory = sensory
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_context_flat = flat
        self._step_count += 1

        return action

    def learn(self, reward: float) -> None:
        if self._prev_action is None or self._prev_context_flat is None:
            return

        exp = _Experience(
            sensory_vec=self._prev_context_flat,
            action=self._prev_action,
            log_prob=self._prev_log_prob,
            reward=reward,
        )
        self.trainer.buffer.add(exp)

        if (self._step_count % self.cfg.update_interval == 0
                and self.trainer.buffer.full):
            self.trainer.update()
