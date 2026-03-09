"""TransformerBrain — pure NumPy causal transformer with REINFORCE training.

Implements the BrainBackend protocol with:
  - Scaled dot-product attention with causal masking
  - Sinusoidal positional encoding
  - Layer normalization & GELU activation
  - Learned input / output projections
  - Multi-head self-attention (n_heads=4, d_model=32)
  - Same multi-head action output as NNBrain (11 logits)
  - REINFORCE training (500-step buffer, updates every 200 ticks, lr=5e-5)
  - Sliding context window of last 16 sensory snapshots
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from agents.actions import AntAction
from agents.sensory import SensoryInput
from brains.nn_brain import (
    Experience,
    RollingBuffer,
    _DEPOSIT_CHANNELS,
    _sigmoid,
    _softmax,
    _tanh,
    decode_output,
    sample_action,
    log_prob_of_action,
    compute_reward,
    _discounted_returns,
)
from config import TransformerBrainConfig


# ---------------------------------------------------------------------------
# Activation & utility functions
# ---------------------------------------------------------------------------

_OUTPUT_DIM = 11

_ROLE_INDEX = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (tanh approximation)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5,
) -> np.ndarray:
    """Layer normalization over the last dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding of shape ``(max_len, d_model)``."""
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular additive mask: 0 for allowed, -1e9 for masked."""
    return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention.

    Args:
        Q, K: ``(..., seq_len, d_k)``
        V:    ``(..., seq_len, d_v)``
        mask: ``(seq_len, seq_len)`` additive causal mask

    Returns:
        output:  ``(..., seq_len, d_v)``
        weights: ``(..., seq_len, seq_len)`` — each row sums to 1
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """Multi-head self-attention with learned Q/K/V/O projections."""

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        s = math.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, s, (d_model, d_model))
        self.W_k = rng.normal(0, s, (d_model, d_model))
        self.W_v = rng.normal(0, s, (d_model, d_model))
        self.W_o = rng.normal(0, s, (d_model, d_model))
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            x:    ``(..., seq_len, d_model)``
            mask: ``(seq_len, seq_len)`` additive causal mask

        Returns:
            output:       ``(..., seq_len, d_model)``
            attn_weights: ``(..., n_heads, seq_len, seq_len)``
        """
        *batch, seq_len, _ = x.shape

        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        def split(t: np.ndarray) -> np.ndarray:
            return t.reshape(*batch, seq_len, self.n_heads, self.d_k).swapaxes(-3, -2)

        attn_out, attn_w = scaled_dot_product_attention(
            split(Q), split(K), split(V), mask,
        )
        # (..., n_heads, seq_len, d_k) → (..., seq_len, d_model)
        attn_out = attn_out.swapaxes(-3, -2).reshape(*batch, seq_len, self.d_model)
        return attn_out @ self.W_o + self.b_o, attn_w

    def parameters(self) -> list[np.ndarray]:
        return [self.W_q, self.W_k, self.W_v, self.W_o,
                self.b_q, self.b_k, self.b_v, self.b_o]


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class TransformerLayer:
    """MHA → Add & LayerNorm → FFN (GELU) → Add & LayerNorm."""

    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, rng: np.random.Generator,
    ):
        self.mha = MultiHeadAttention(d_model, n_heads, rng)

        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)

        s1 = math.sqrt(2.0 / d_model)
        s2 = math.sqrt(2.0 / ffn_dim)
        self.W1 = rng.normal(0, s1, (d_model, ffn_dim))
        self.b1 = np.zeros(ffn_dim)
        self.W2 = rng.normal(0, s2, (ffn_dim, d_model))
        self.b2 = np.zeros(d_model)

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        attn_out, attn_w = self.mha.forward(x, mask)
        x = layer_norm(x + attn_out, self.ln1_g, self.ln1_b)
        ffn_out = gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        x = layer_norm(x + ffn_out, self.ln2_g, self.ln2_b)
        return x, attn_w

    def parameters(self) -> list[np.ndarray]:
        return (
            self.mha.parameters()
            + [self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
               self.W1, self.b1, self.W2, self.b2]
        )


# ---------------------------------------------------------------------------
# Shared weight registry (one set per role)
# ---------------------------------------------------------------------------

class TransformerWeightSet:
    """Holds all transformer parameters for one role."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        rng: np.random.Generator,
    ):
        self.d_model = d_model
        self.n_heads = n_heads

        s_in = math.sqrt(2.0 / input_dim)
        self.W_in = rng.normal(0, s_in, (input_dim, d_model))
        self.b_in = np.zeros(d_model)

        self.layers = [
            TransformerLayer(d_model, n_heads, ffn_dim, rng)
            for _ in range(n_layers)
        ]

        s_out = math.sqrt(2.0 / d_model)
        self.W_out = rng.normal(0, s_out, (d_model, _OUTPUT_DIM))
        self.b_out = np.zeros(_OUTPUT_DIM)

        # Value head (for PPO)
        self.W_val = rng.normal(0, 0.01, (d_model, 1))
        self.b_val = np.zeros(1)

        self.pos_enc = sinusoidal_positional_encoding(64, d_model)

    def all_parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = [self.W_in, self.b_in]
        for lyr in self.layers:
            params.extend(lyr.parameters())
        params.extend([self.W_out, self.b_out, self.W_val, self.b_val])
        return params

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Run transformer on context tensor.

        Args:
            x: ``(..., seq_len, input_dim)``

        Returns:
            logits:       ``(..., _OUTPUT_DIM)``
            attn_weights: per-layer ``(..., n_heads, seq_len, seq_len)``
        """
        seq_len = x.shape[-2]
        h = x @ self.W_in + self.b_in
        h = h + self.pos_enc[:seq_len]
        mask = causal_mask(seq_len)

        all_attn: list[np.ndarray] = []
        for lyr in self.layers:
            h, aw = lyr.forward(h, mask)
            all_attn.append(aw)

        last_hidden = h[..., -1, :]
        logits = last_hidden @ self.W_out + self.b_out
        return logits, all_attn

    def forward_with_value(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Run transformer returning both logits and value estimate.

        Returns:
            logits:       ``(..., _OUTPUT_DIM)``
            value:        ``(...,)`` scalar value estimate
            attn_weights: per-layer attention weights
        """
        seq_len = x.shape[-2]
        h = x @ self.W_in + self.b_in
        h = h + self.pos_enc[:seq_len]
        mask = causal_mask(seq_len)

        all_attn: list[np.ndarray] = []
        for lyr in self.layers:
            h, aw = lyr.forward(h, mask)
            all_attn.append(aw)

        last_hidden = h[..., -1, :]
        logits = last_hidden @ self.W_out + self.b_out
        value = (last_hidden @ self.W_val + self.b_val).squeeze(-1)
        return logits, value, all_attn


class SharedWeightRegistry:
    """Manages shared TransformerWeightSets for each ant role."""

    def __init__(
        self,
        input_dim: int = 39,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 64,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self._weights: dict[str, TransformerWeightSet] = {}
        for role in _ROLE_INDEX:
            self._weights[role] = TransformerWeightSet(
                input_dim, d_model, n_heads, n_layers, ffn_dim, self.rng,
            )

    def get(self, role: str) -> TransformerWeightSet:
        return self._weights[role]

    def roles(self) -> list[str]:
        return list(self._weights.keys())


# ---------------------------------------------------------------------------
# REINFORCE trainer (per-role)
# ---------------------------------------------------------------------------

class RoleTrainer:
    """REINFORCE trainer for a single role's shared transformer weights.

    Uses zeroth-order policy gradient estimation: computes scalar REINFORCE
    signal from discounted returns and log-probs, then perturbs all parameters
    in the gradient direction scaled by that signal.
    """

    def __init__(
        self,
        weight_set: TransformerWeightSet,
        lr: float = 5e-5,
        gamma: float = 0.95,
        buffer_size: int = 500,
        warmup_steps: int = 50,
    ):
        self.weight_set = weight_set
        self.lr = lr
        self.gamma = gamma
        self.buffer = RollingBuffer(buffer_size)
        self.baseline: float = 0.0
        self.baseline_alpha: float = 0.01
        self.warmup_steps = warmup_steps
        self.global_step: int = 0
        self._update_rng = np.random.default_rng(0)

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

        rewards = np.array([e.reward for e in experiences])
        log_probs = np.array([e.log_prob for e in experiences])

        returns = _discounted_returns(rewards, self.gamma)

        mean_return = float(returns.mean())
        self.baseline = (
            self.baseline * (1 - self.baseline_alpha)
            + mean_return * self.baseline_alpha
        )

        advantages = returns - self.baseline

        # Policy gradient signal
        pg_signal = float(-np.mean(log_probs * advantages))

        lr = self.effective_lr
        for param in self.weight_set.all_parameters():
            param += lr * pg_signal * self._update_rng.standard_normal(param.shape)

        return float(-np.mean(advantages ** 2))


# ---------------------------------------------------------------------------
# Batch forward pass for colony-wide vectorized processing
# ---------------------------------------------------------------------------

def batch_decide(
    weight_set: TransformerWeightSet,
    context_seqs: list[np.ndarray],
    rng: np.random.Generator,
) -> list[tuple[AntAction, float, np.ndarray]]:
    """Vectorized batch forward pass for multiple ants sharing the same weights.

    Args:
        weight_set: shared transformer weights
        context_seqs: list of (seq_len, input_dim) context arrays per ant
        rng: random generator for action sampling

    Returns:
        list of (action, log_prob, context_flat) tuples
    """
    if not context_seqs:
        return []

    # Pad all sequences to the same length for batching
    max_seq = max(s.shape[0] for s in context_seqs)
    input_dim = context_seqs[0].shape[1]
    batch_size = len(context_seqs)

    batch_input = np.zeros((batch_size, max_seq, input_dim))
    for i, seq in enumerate(context_seqs):
        slen = seq.shape[0]
        batch_input[i, max_seq - slen:, :] = seq  # right-align

    # Batch forward
    logits, _ = weight_set.forward(batch_input)  # (batch, 11)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]

    results = []
    for i in range(batch_size):
        raw = logits[i]
        decoded = decode_output(raw)
        action = sample_action(decoded, rng)
        lp = log_prob_of_action(decoded, action)
        results.append((action, lp, context_seqs[i].flatten()))

    return results


# ---------------------------------------------------------------------------
# TransformerBrain — per-ant brain instance
# ---------------------------------------------------------------------------

class TransformerBrain:
    """Pure NumPy causal transformer brain implementing BrainBackend.

    Maintains a sliding context window of the last ``context_length`` sensory
    snapshots and uses a stack of causal transformer layers to produce actions.
    Uses the same multi-head action output as NNBrain (11 logits → decode_output
    → sample_action). Training uses REINFORCE with a rolling reward buffer.
    """

    def __init__(
        self,
        role: str = "forager",
        registry: SharedWeightRegistry | None = None,
        trainer: RoleTrainer | None = None,
        cfg: Optional[TransformerBrainConfig] = None,
        input_dim: int = 39,
        seed: int = 42,
    ):
        if cfg is None:
            cfg = TransformerBrainConfig()
        self.cfg = cfg
        self.role = role
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)

        # Shared weight registry
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
        self.weight_set = registry.get(role)

        # Per-role trainer (may be shared across ants of same role)
        if trainer is None:
            trainer = RoleTrainer(
                self.weight_set,
                lr=cfg.learning_rate,
                gamma=0.95,
                buffer_size=cfg.buffer_size,
                warmup_steps=50,
            )
        self.trainer = trainer

        # Sliding context window
        self.context: deque[np.ndarray] = deque(maxlen=cfg.context_length)

        # Per-ant state
        self._prev_sensory: SensoryInput | None = None
        self._prev_action: AntAction | None = None
        self._prev_log_prob: float = 0.0
        self._prev_context_flat: np.ndarray | None = None
        self._step_count: int = 0

    # ------------------------------------------------------------------ forward

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Run transformer on context tensor. Delegates to weight_set.forward()."""
        return self.weight_set.forward(x)

    # ---------------------------------------------------- BrainBackend protocol

    def decide(self, sensory: SensoryInput) -> AntAction:
        """Given sensory input, return an action decision."""
        vec = np.array(sensory.to_vector(), dtype=np.float64)
        self.context.append(vec)

        ctx = np.stack(list(self.context), axis=0)  # (seq_len, input_dim)
        logits, _ = self._forward(ctx)

        # Same multi-head action output as NNBrain
        decoded = decode_output(logits)
        action = sample_action(decoded, self.rng)
        log_prob = log_prob_of_action(decoded, action)

        # Store for learning
        self._prev_sensory = sensory
        self._prev_action = action
        self._prev_log_prob = log_prob
        self._prev_context_flat = ctx.flatten()
        self._step_count += 1

        return action

    def learn(self, reward: float) -> None:
        """Receive reward signal. Stores experience and periodically trains."""
        if self._prev_action is None or self._prev_context_flat is None:
            return

        exp = Experience(
            sensory_vec=self._prev_context_flat,
            action=self._prev_action,
            log_prob=self._prev_log_prob,
            reward=reward,
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
        """Compute reward for a transition (convenience wrapper)."""
        return compute_reward(prev_sensory, curr_sensory, action, alive)
