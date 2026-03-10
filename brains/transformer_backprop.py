"""Analytical backpropagation for the pure-NumPy causal transformer.

Provides:
  - TransformerBackpropCache: stores all forward-pass intermediates
  - transformer_forward_with_cache(): forward pass saving activations
  - transformer_backward(): full analytical backprop through the transformer
  - Helper functions: gelu_backward, layer_norm_backward, attention_backward
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from brains.transformer_brain import (
        TransformerWeightSet,
        TransformerLayer,
        MultiHeadAttention,
    )


# ---------------------------------------------------------------------------
# Local copies of activation functions (avoids circular import)
# ---------------------------------------------------------------------------

def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (tanh approximation)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5,
) -> np.ndarray:
    """Layer normalization over the last dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular additive mask: 0 for allowed, -1e9 for masked."""
    return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9


# ---------------------------------------------------------------------------
# Cached intermediates for a single layer
# ---------------------------------------------------------------------------

@dataclass
class LayerCache:
    """Intermediate values for one transformer layer."""
    # Inputs
    x_in: np.ndarray                # input to this layer (seq_len, d_model)
    # Attention
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray
    Q_heads: np.ndarray             # (n_heads, seq_len, d_k)
    K_heads: np.ndarray
    V_heads: np.ndarray
    attn_weights: np.ndarray        # (n_heads, seq_len, seq_len)
    attn_out_heads: np.ndarray      # (n_heads, seq_len, d_k)
    attn_out: np.ndarray            # (seq_len, d_model) after W_o
    # Post-attention residual + layer norm
    x_after_ln1: np.ndarray         # output of first layer norm
    ln1_mean: np.ndarray
    ln1_var: np.ndarray
    residual1: np.ndarray           # x_in + attn_out (before LN1)
    # FFN
    ffn_pre: np.ndarray             # x_after_ln1 @ W1 + b1 (before GELU)
    ffn_act: np.ndarray             # gelu(ffn_pre)
    ffn_out: np.ndarray             # ffn_act @ W2 + b2
    # Post-FFN residual + layer norm
    x_after_ln2: np.ndarray
    ln2_mean: np.ndarray
    ln2_var: np.ndarray
    residual2: np.ndarray           # x_after_ln1 + ffn_out (before LN2)


@dataclass
class TransformerBackpropCache:
    """Full forward-pass cache for backpropagation."""
    x_input: np.ndarray                     # original input (seq_len, input_dim)
    h_projected: np.ndarray = field(default_factory=lambda: np.array([]))
    layer_caches: list[LayerCache] = field(default_factory=list)
    last_hidden: np.ndarray = field(default_factory=lambda: np.array([]))
    mask: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Helper backward functions
# ---------------------------------------------------------------------------

def gelu_backward(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU (tanh approximation) w.r.t. input."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(inner)
    sech2 = 1.0 - tanh_val ** 2
    inner_deriv = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_deriv


def layer_norm_backward(
    dL_dout: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward through layer normalization.

    Returns:
        dL_dx, dL_dgamma, dL_dbeta
    """
    d = x.shape[-1]
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * std_inv

    dL_dgamma = np.sum(dL_dout * x_hat, axis=tuple(range(x.ndim - 1)))
    dL_dbeta = np.sum(dL_dout, axis=tuple(range(x.ndim - 1)))

    dL_dxhat = dL_dout * gamma
    dL_dvar = np.sum(dL_dxhat * (x - mean) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
    dL_dmean = np.sum(dL_dxhat * -std_inv, axis=-1, keepdims=True) + \
               dL_dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
    dL_dx = dL_dxhat * std_inv + dL_dvar * 2.0 * (x - mean) / d + dL_dmean / d

    return dL_dx, dL_dgamma, dL_dbeta


def attention_backward(
    dL_dout: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    attn_weights: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward through scaled dot-product attention.

    All inputs have shape (..., seq_len, d_k) or (..., seq_len, seq_len).

    Returns:
        dL_dQ, dL_dK, dL_dV
    """
    d_k = Q.shape[-1]

    # dL_dV: attn_weights.T @ dL_dout
    dL_dV = attn_weights.swapaxes(-2, -1) @ dL_dout

    # dL_d_attn_weights
    dL_dw = dL_dout @ V.swapaxes(-2, -1)

    # Softmax backward: dL_d_scores
    # For row-wise softmax: dL_ds_i = w_i * (dL_dw_i - sum_j(w_ij * dL_dw_ij))
    sum_term = np.sum(dL_dw * attn_weights, axis=-1, keepdims=True)
    dL_dscores = attn_weights * (dL_dw - sum_term)

    # Scale by 1/sqrt(d_k)
    scale = 1.0 / math.sqrt(d_k)
    dL_dscores = dL_dscores * scale

    # dL_dQ = dL_dscores @ K
    dL_dQ = dL_dscores @ K
    # dL_dK = dL_dscores.T @ Q
    dL_dK = dL_dscores.swapaxes(-2, -1) @ Q

    return dL_dQ, dL_dK, dL_dV


# ---------------------------------------------------------------------------
# Forward with cache
# ---------------------------------------------------------------------------

def transformer_forward_with_cache(
    ws: TransformerWeightSet,
    x: np.ndarray,
) -> tuple[np.ndarray, TransformerBackpropCache]:
    """Forward pass through transformer saving all intermediates for backprop.

    Args:
        ws: transformer weights
        x: (seq_len, input_dim) input tensor (single sequence, no batch dim)

    Returns:
        logits: (output_dim,)
        cache: TransformerBackpropCache with all intermediates
    """
    cache = TransformerBackpropCache(x_input=x)

    seq_len = x.shape[0]
    h = x @ ws.W_in + ws.b_in
    h = h + ws.pos_enc[:seq_len]
    cache.h_projected = h.copy()
    mask = _causal_mask(seq_len)
    cache.mask = mask

    for lyr in ws.layers:
        lc = _layer_forward_with_cache(lyr, h, mask)
        cache.layer_caches.append(lc)
        h = lc.x_after_ln2

    last_hidden = h[-1, :]  # (d_model,)
    cache.last_hidden = last_hidden
    logits = last_hidden @ ws.W_out + ws.b_out
    return logits, cache


def _layer_forward_with_cache(
    lyr: TransformerLayer,
    x: np.ndarray,
    mask: Optional[np.ndarray],
) -> LayerCache:
    """Forward pass through one transformer layer with caching."""
    mha = lyr.mha
    d_model = mha.d_model
    n_heads = mha.n_heads
    d_k = mha.d_k
    seq_len = x.shape[0]

    # MHA
    Q = x @ mha.W_q + mha.b_q
    K = x @ mha.W_k + mha.b_k
    V = x @ mha.W_v + mha.b_v

    def split(t: np.ndarray) -> np.ndarray:
        return t.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    Q_heads = split(Q)
    K_heads = split(K)
    V_heads = split(V)

    # Scaled dot-product attention per head
    scores = Q_heads @ K_heads.swapaxes(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn_weights = _softmax(scores, axis=-1)
    attn_out_heads = attn_weights @ V_heads  # (n_heads, seq_len, d_k)

    # Concat and project
    attn_concat = attn_out_heads.transpose(1, 0, 2).reshape(seq_len, d_model)
    attn_out = attn_concat @ mha.W_o + mha.b_o

    # Residual + LayerNorm 1
    residual1 = x + attn_out
    ln1_mean = residual1.mean(axis=-1, keepdims=True)
    ln1_var = residual1.var(axis=-1, keepdims=True)
    x_after_ln1 = _layer_norm(residual1, lyr.ln1_g, lyr.ln1_b)

    # FFN
    ffn_pre = x_after_ln1 @ lyr.W1 + lyr.b1
    ffn_act = _gelu(ffn_pre)
    ffn_out = ffn_act @ lyr.W2 + lyr.b2

    # Residual + LayerNorm 2
    residual2 = x_after_ln1 + ffn_out
    ln2_mean = residual2.mean(axis=-1, keepdims=True)
    ln2_var = residual2.var(axis=-1, keepdims=True)
    x_after_ln2 = _layer_norm(residual2, lyr.ln2_g, lyr.ln2_b)

    return LayerCache(
        x_in=x,
        Q=Q, K=K, V=V,
        Q_heads=Q_heads, K_heads=K_heads, V_heads=V_heads,
        attn_weights=attn_weights,
        attn_out_heads=attn_out_heads,
        attn_out=attn_out,
        x_after_ln1=x_after_ln1,
        ln1_mean=ln1_mean, ln1_var=ln1_var,
        residual1=residual1,
        ffn_pre=ffn_pre, ffn_act=ffn_act, ffn_out=ffn_out,
        x_after_ln2=x_after_ln2,
        ln2_mean=ln2_mean, ln2_var=ln2_var,
        residual2=residual2,
    )


# ---------------------------------------------------------------------------
# Full backward pass
# ---------------------------------------------------------------------------

def transformer_backward(
    ws: TransformerWeightSet,
    cache: TransformerBackpropCache,
    dL_dlogits: np.ndarray,
) -> list[np.ndarray]:
    """Full analytical backprop through the transformer.

    Args:
        ws: transformer weights
        cache: TransformerBackpropCache from forward pass
        dL_dlogits: (output_dim,) gradient of loss w.r.t. output logits

    Returns:
        List of gradients in same order as ws.all_parameters():
        [dW_in, db_in, <per-layer params>, dW_out, db_out, dW_val, db_val]
    """
    # Output layer gradients
    last_hidden = cache.last_hidden  # (d_model,)
    dW_out = np.outer(last_hidden, dL_dlogits)
    db_out = dL_dlogits.copy()

    # dL/d_last_hidden
    dL_dh_last = dL_dlogits @ ws.W_out.T  # (d_model,)

    # Expand to full sequence: only last position has gradient
    n_layers = len(ws.layers)
    seq_len = cache.layer_caches[0].x_in.shape[0] if n_layers > 0 else cache.h_projected.shape[0]
    d_model = ws.d_model

    dL_dh = np.zeros((seq_len, d_model))
    dL_dh[-1, :] = dL_dh_last

    # Backward through layers (reverse order)
    layer_grads: list[list[np.ndarray]] = []
    for i in range(n_layers - 1, -1, -1):
        lyr = ws.layers[i]
        lc = cache.layer_caches[i]
        dL_dh, lgrads = _layer_backward(lyr, lc, dL_dh, cache.mask)
        layer_grads.insert(0, lgrads)

    # Input projection gradients
    dW_in = cache.x_input.T @ dL_dh
    db_in = dL_dh.sum(axis=0)

    # Assemble gradient list matching ws.all_parameters() order
    all_grads: list[np.ndarray] = [dW_in, db_in]
    for lgrads in layer_grads:
        all_grads.extend(lgrads)
    all_grads.extend([dW_out, db_out])

    # Value head: zero gradients (trained separately or not used here)
    all_grads.append(np.zeros_like(ws.W_val))
    all_grads.append(np.zeros_like(ws.b_val))

    return all_grads


def _layer_backward(
    lyr: TransformerLayer,
    lc: LayerCache,
    dL_dout: np.ndarray,
    mask: Optional[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Backward through one transformer layer.

    Returns:
        dL_dx_in: gradient w.r.t. layer input
        grads: list of parameter gradients matching lyr.parameters() order
    """
    mha = lyr.mha
    d_model = mha.d_model
    n_heads = mha.n_heads
    d_k = mha.d_k
    seq_len = lc.x_in.shape[0]

    # --- LayerNorm 2 backward ---
    dL_dresidual2, dL_dln2_g, dL_dln2_b = layer_norm_backward(
        dL_dout, lc.residual2, lyr.ln2_g, lc.ln2_mean, lc.ln2_var,
    )

    # --- FFN backward ---
    dL_dffn_out = dL_dresidual2.copy()
    # ffn_out = ffn_act @ W2 + b2
    dW2 = lc.ffn_act.T @ dL_dffn_out
    db2 = dL_dffn_out.sum(axis=0)
    dL_dffn_act = dL_dffn_out @ lyr.W2.T

    # GELU backward
    dL_dffn_pre = dL_dffn_act * gelu_backward(lc.ffn_pre)

    # ffn_pre = x_after_ln1 @ W1 + b1
    dW1 = lc.x_after_ln1.T @ dL_dffn_pre
    db1 = dL_dffn_pre.sum(axis=0)
    dL_dx_after_ln1 = dL_dffn_pre @ lyr.W1.T

    # Residual connection: x_after_ln1 = LN(x_in + attn_out)
    dL_dx_after_ln1 += dL_dresidual2  # from FFN residual skip

    # --- LayerNorm 1 backward ---
    dL_dresidual1, dL_dln1_g, dL_dln1_b = layer_norm_backward(
        dL_dx_after_ln1, lc.residual1, lyr.ln1_g, lc.ln1_mean, lc.ln1_var,
    )

    # --- MHA backward ---
    dL_dattn_out = dL_dresidual1.copy()

    # attn_concat @ W_o + b_o
    dW_o = lc.attn_out_heads.transpose(1, 0, 2).reshape(seq_len, d_model).T @ dL_dattn_out
    db_o = dL_dattn_out.sum(axis=0)
    dL_dattn_concat = dL_dattn_out @ mha.W_o.T

    # Reshape back to heads
    dL_dattn_heads = dL_dattn_concat.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Attention backward per head
    dL_dQ_heads, dL_dK_heads, dL_dV_heads = attention_backward(
        dL_dattn_heads, lc.Q_heads, lc.K_heads, lc.V_heads,
        lc.attn_weights, mask,
    )

    # Unsplit heads
    def unsplit(t: np.ndarray) -> np.ndarray:
        return t.transpose(1, 0, 2).reshape(seq_len, d_model)

    dL_dQ = unsplit(dL_dQ_heads)
    dL_dK = unsplit(dL_dK_heads)
    dL_dV = unsplit(dL_dV_heads)

    # Q = x @ W_q + b_q
    dW_q = lc.x_in.T @ dL_dQ
    db_q = dL_dQ.sum(axis=0)
    dW_k = lc.x_in.T @ dL_dK
    db_k = dL_dK.sum(axis=0)
    dW_v = lc.x_in.T @ dL_dV
    db_v = dL_dV.sum(axis=0)

    dL_dx_from_attn = dL_dQ @ mha.W_q.T + dL_dK @ mha.W_k.T + dL_dV @ mha.W_v.T

    # Residual: gradient flows through to input
    dL_dx_in = dL_dresidual1 + dL_dx_from_attn

    # Collect gradients in same order as lyr.parameters()
    # MHA params: [W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o]
    # LN + FFN: [ln1_g, ln1_b, ln2_g, ln2_b, W1, b1, W2, b2]
    grads = [
        dW_q, dW_k, dW_v, dW_o,
        db_q, db_k, db_v, db_o,
        dL_dln1_g, dL_dln1_b, dL_dln2_g, dL_dln2_b,
        dW1, db1, dW2, db2,
    ]

    return dL_dx_in, grads
