"""Tests for TransformerBrain — pure NumPy causal transformer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.actions import AntAction
from agents.sensory import SensoryInput
from agents.ant import Vec2
from brains.transformer_brain import (
    TransformerBrain,
    TransformerWeightSet,
    SharedWeightRegistry,
    RoleTrainer,
    MultiHeadAttention,
    TransformerLayer,
    batch_decide,
    causal_mask,
    gelu,
    layer_norm,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
    softmax,
    _OUTPUT_DIM,
)
from brains.nn_brain import _DEPOSIT_CHANNELS, RollingBuffer, Experience
from config import TransformerBrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def cfg() -> TransformerBrainConfig:
    return TransformerBrainConfig()


def _make_sensory(energy: float = 80.0, carrying: str | None = None,
                  role: str = "forager") -> SensoryInput:
    """Helper to build a SensoryInput with reasonable defaults."""
    return SensoryInput(
        antenna_left={"food": 0.5, "home": 0.1, "danger": 0.0, "recruit": 0.0},
        antenna_right={"food": 0.3, "home": 0.2, "danger": 0.0, "recruit": 0.0},
        obstacle_rays=[1.0, 0.8, 1.0, 0.9, 1.0],
        nest_direction=Vec2(-0.7, 0.7),
        nest_distance=300.0,
        food_gradient=Vec2(0.2, -0.1),
        energy=energy,
        carrying=carrying,
        role_value=role,
    )


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------

class TestGELU:
    def test_reference_match(self):
        """GELU must match the tanh-approximation reference formula."""
        x = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
        result = gelu(x)
        ref = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_zero(self):
        assert gelu(np.array([0.0]))[0] == 0.0

    def test_positive_for_positive_input(self):
        x = np.linspace(0.1, 5.0, 50)
        assert np.all(gelu(x) > 0)

    def test_approx_zero_for_large_negative(self):
        x = np.array([-10.0])
        assert abs(gelu(x)[0]) < 1e-6

    def test_known_values(self):
        """GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588."""
        np.testing.assert_allclose(gelu(np.array([1.0]))[0], 0.8412, atol=0.001)
        np.testing.assert_allclose(gelu(np.array([-1.0]))[0], -0.1588, atol=0.001)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        x = np.random.default_rng(0).normal(size=8)
        np.testing.assert_allclose(softmax(x).sum(), 1.0, atol=1e-12)

    def test_all_positive(self):
        x = np.random.default_rng(1).normal(size=10)
        assert np.all(softmax(x) > 0)

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)

    def test_batch(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        p = softmax(x, axis=-1)
        assert p.shape == (2, 3)
        np.testing.assert_allclose(p.sum(axis=-1), [1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Layer normalization
# ---------------------------------------------------------------------------

class TestLayerNorm:
    def test_output_mean_near_zero(self):
        x = np.random.default_rng(0).normal(5.0, 3.0, (10, 32))
        gamma, beta = np.ones(32), np.zeros(32)
        out = layer_norm(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)

    def test_output_var_near_one(self):
        x = np.random.default_rng(0).normal(5.0, 3.0, (10, 32))
        gamma, beta = np.ones(32), np.zeros(32)
        out = layer_norm(x, gamma, beta)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=0.05)

    def test_affine_shift(self):
        x = np.random.default_rng(0).normal(size=(5, 16))
        gamma = np.full(16, 2.0)
        beta = np.full(16, 3.0)
        out = layer_norm(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 3.0, atol=0.1)

    def test_3d_input(self):
        """Layer norm on (batch, seq, d_model) should normalize along last axis."""
        x = np.random.default_rng(0).normal(3.0, 2.0, (4, 8, 32))
        gamma, beta = np.ones(32), np.zeros(32)
        out = layer_norm(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_shape(self):
        pe = sinusoidal_positional_encoding(16, 32)
        assert pe.shape == (16, 32)

    def test_uniqueness(self):
        """Every position must have a unique encoding vector."""
        pe = sinusoidal_positional_encoding(16, 32)
        for i in range(16):
            for j in range(i + 1, 16):
                assert not np.allclose(pe[i], pe[j]), (
                    f"Positions {i} and {j} have identical encoding"
                )

    def test_values_bounded(self):
        pe = sinusoidal_positional_encoding(16, 32)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_sin_cos_pattern_at_origin(self):
        pe = sinusoidal_positional_encoding(16, 32)
        # pos 0: sin(0)=0 for even dims, cos(0)=1 for odd dims
        np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-12)
        np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-12)

    def test_different_d_model(self):
        pe = sinusoidal_positional_encoding(8, 64)
        assert pe.shape == (8, 64)
        # Still unique
        for i in range(8):
            for j in range(i + 1, 8):
                assert not np.allclose(pe[i], pe[j])


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:
    def test_weights_sum_to_one(self):
        """Attention weights must sum to 1 along key dimension."""
        rng = np.random.default_rng(42)
        Q = rng.normal(size=(6, 8))
        K = rng.normal(size=(6, 8))
        V = rng.normal(size=(6, 8))
        _, w = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)

    def test_weights_sum_to_one_with_causal_mask(self):
        """Even with causal masking, attention weights sum to 1."""
        rng = np.random.default_rng(42)
        seq = 8
        Q = rng.normal(size=(seq, 4))
        K = rng.normal(size=(seq, 4))
        V = rng.normal(size=(seq, 4))
        _, w = scaled_dot_product_attention(Q, K, V, causal_mask(seq))
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)

    def test_weights_sum_to_one_batched(self):
        """Batched attention weights sum to 1."""
        rng = np.random.default_rng(42)
        Q = rng.normal(size=(3, 4, 6, 8))
        K = rng.normal(size=(3, 4, 6, 8))
        V = rng.normal(size=(3, 4, 6, 8))
        _, w = scaled_dot_product_attention(Q, K, V, causal_mask(6))
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)

    def test_output_shape(self):
        Q = np.random.default_rng(0).normal(size=(4, 8))
        K = np.random.default_rng(1).normal(size=(4, 8))
        V = np.random.default_rng(2).normal(size=(4, 16))
        out, w = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (4, 16)
        assert w.shape == (4, 4)

    def test_multi_head_output_shape(self):
        rng = np.random.default_rng(42)
        mha = MultiHeadAttention(32, 4, rng)
        x = rng.normal(size=(8, 32))
        out, aw = mha.forward(x, causal_mask(8))
        assert out.shape == (8, 32)
        assert aw.shape == (4, 8, 8)

    def test_multi_head_batched_shape(self):
        rng = np.random.default_rng(42)
        mha = MultiHeadAttention(32, 4, rng)
        x = rng.normal(size=(5, 8, 32))
        out, aw = mha.forward(x, causal_mask(8))
        assert out.shape == (5, 8, 32)
        assert aw.shape == (5, 4, 8, 8)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

class TestCausalMask:
    def test_shape(self):
        assert causal_mask(8).shape == (8, 8)

    def test_diagonal_and_below_zero(self):
        m = causal_mask(8)
        for i in range(8):
            for j in range(i + 1):
                assert m[i, j] == 0.0

    def test_upper_triangle_large_negative(self):
        m = causal_mask(8)
        for i in range(8):
            for j in range(i + 1, 8):
                assert m[i, j] == -1e9

    def test_single_position(self):
        m = causal_mask(1)
        assert m.shape == (1, 1)
        assert m[0, 0] == 0.0

    def test_causal_masking_blocks_future(self):
        """Future positions should receive ~0 attention weight."""
        rng = np.random.default_rng(7)
        seq = 8
        Q = rng.normal(size=(seq, 4))
        K = rng.normal(size=(seq, 4))
        V = rng.normal(size=(seq, 4))
        _, w = scaled_dot_product_attention(Q, K, V, causal_mask(seq))
        for i in range(seq):
            for j in range(i + 1, seq):
                assert w[i, j] < 1e-6, f"w[{i},{j}]={w[i,j]} is not near zero"


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

class TestContextWindow:
    def test_sliding_window_size(self):
        cfg = TransformerBrainConfig(context_length=4)
        brain = TransformerBrain(cfg=cfg, seed=0)
        for i in range(10):
            brain.decide(SensoryInput(energy=float(i)))
        assert len(brain.context) == 4

    def test_sliding_window_content(self):
        """After 10 steps with context_length=4, window should hold last 4 snapshots."""
        cfg = TransformerBrainConfig(context_length=4)
        brain = TransformerBrain(cfg=cfg, seed=0)
        for i in range(10):
            brain.decide(SensoryInput(energy=float(i)))
        # energy is at index 16, normalised by /100
        energies = [brain.context[j][16] for j in range(4)]
        expected = [6.0 / 100, 7.0 / 100, 8.0 / 100, 9.0 / 100]
        np.testing.assert_allclose(energies, expected, atol=1e-12)

    def test_single_step_works(self):
        brain = TransformerBrain(seed=0)
        action = brain.decide(SensoryInput())
        assert action is not None
        assert -math.pi / 6 <= action.turn <= math.pi / 6

    def test_context_grows_then_caps(self):
        cfg = TransformerBrainConfig(context_length=4)
        brain = TransformerBrain(cfg=cfg, seed=0)
        for i in range(6):
            brain.decide(SensoryInput(energy=float(i * 10)))
            if i < 4:
                assert len(brain.context) == i + 1
            else:
                assert len(brain.context) == 4


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------

class TestTransformerBrainOutput:
    def test_valid_output_types(self):
        brain = TransformerBrain(seed=42)
        action = brain.decide(SensoryInput())
        assert isinstance(action.turn, float)
        assert isinstance(action.speed_mult, float)
        assert action.deposit_pheromone in (None, "food", "home", "danger", "recruit")
        assert isinstance(action.deposit_strength, float)
        assert isinstance(action.pickup, bool)
        assert isinstance(action.drop, bool)
        assert isinstance(action.recruit_signal, bool)

    def test_action_ranges_over_many_steps(self):
        brain = TransformerBrain(seed=42)
        for _ in range(30):
            action = brain.decide(SensoryInput(energy=50.0))
            assert -math.pi / 6 <= action.turn <= math.pi / 6
            assert 0.0 <= action.speed_mult <= 1.0
            assert 0.0 <= action.deposit_strength <= 1.0

    def test_forward_output_dim(self):
        brain = TransformerBrain(seed=0)
        ctx = np.random.default_rng(0).normal(size=(4, 39))
        logits, attn = brain._forward(ctx)
        assert logits.shape == (_OUTPUT_DIM,)
        assert len(attn) == brain.cfg.n_layers

    def test_forward_batched_output_dim(self):
        brain = TransformerBrain(seed=0)
        ctx = np.random.default_rng(0).normal(size=(3, 4, 39))
        logits, attn = brain._forward(ctx)
        assert logits.shape == (3, _OUTPUT_DIM)

    def test_deposit_pheromone_valid(self):
        """Deposit pheromone must be from the valid channel list."""
        brain = TransformerBrain(seed=42)
        channels_seen = set()
        for _ in range(100):
            action = brain.decide(_make_sensory())
            assert action.deposit_pheromone in _DEPOSIT_CHANNELS
            if action.deposit_pheromone is not None:
                channels_seen.add(action.deposit_pheromone)
        assert len(channels_seen) >= 1

    def test_no_strength_when_no_deposit(self):
        """If deposit_pheromone is None, strength should be 0."""
        brain = TransformerBrain(seed=42)
        for _ in range(50):
            action = brain.decide(_make_sensory())
            if action.deposit_pheromone is None:
                assert action.deposit_strength == 0.0


# ---------------------------------------------------------------------------
# Batch consistency (determinism)
# ---------------------------------------------------------------------------

class TestBatchConsistency:
    def test_same_seed_same_output(self):
        s = SensoryInput(energy=75.0, role_value="forager")
        a1 = TransformerBrain(seed=99).decide(s)
        a2 = TransformerBrain(seed=99).decide(s)
        assert a1.turn == a2.turn
        assert a1.speed_mult == a2.speed_mult
        assert a1.deposit_pheromone == a2.deposit_pheromone
        assert a1.pickup == a2.pickup

    def test_different_seed_different_output(self):
        s = SensoryInput(energy=75.0)
        a1 = TransformerBrain(seed=1).decide(s)
        a2 = TransformerBrain(seed=2).decide(s)
        differs = (
            a1.turn != a2.turn
            or a1.speed_mult != a2.speed_mult
            or a1.deposit_pheromone != a2.deposit_pheromone
        )
        assert differs

    def test_batch_forward_matches_individual(self):
        """Batch forward pass should give same results as individual passes."""
        rng = np.random.default_rng(99)
        ws = TransformerWeightSet(39, 32, 4, 2, 64, rng)
        seqs = [np.random.default_rng(i).normal(size=(4, 39)) for i in range(5)]

        # Individual
        individual_outs = []
        for seq in seqs:
            out, _ = ws.forward(seq)
            individual_outs.append(out)

        # Batch
        batch_in = np.stack(seqs, axis=0)  # (5, 4, 39)
        batch_out, _ = ws.forward(batch_in)

        for i in range(5):
            np.testing.assert_allclose(batch_out[i], individual_outs[i], atol=1e-10)


# ---------------------------------------------------------------------------
# BrainBackend protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_has_decide(self):
        brain = TransformerBrain(seed=42)
        assert hasattr(brain, "decide")
        assert callable(brain.decide)

    def test_has_learn(self):
        brain = TransformerBrain(seed=42)
        assert hasattr(brain, "learn")
        assert callable(brain.learn)

    def test_decide_returns_ant_action(self):
        brain = TransformerBrain(seed=42)
        action = brain.decide(SensoryInput())
        assert isinstance(action, AntAction)

    def test_learn_accepts_reward(self):
        brain = TransformerBrain(seed=42)
        brain.decide(SensoryInput())
        brain.learn(1.0)  # Should not raise

    def test_runtime_checkable(self):
        """TransformerBrain should satisfy BrainBackend protocol."""
        from brains.interface import BrainBackend
        brain = TransformerBrain(seed=42)
        assert isinstance(brain, BrainBackend)


# ---------------------------------------------------------------------------
# Weight sharing
# ---------------------------------------------------------------------------

class TestWeightSharing:
    def test_same_role_shares_weights(self):
        registry = SharedWeightRegistry(seed=123)
        brain1 = TransformerBrain(role="forager", registry=registry, seed=1)
        brain2 = TransformerBrain(role="forager", registry=registry, seed=2)
        assert brain1.weight_set is brain2.weight_set
        assert brain1.weight_set.W_in is brain2.weight_set.W_in

    def test_different_role_different_weights(self):
        registry = SharedWeightRegistry(seed=123)
        brain_f = TransformerBrain(role="forager", registry=registry, seed=1)
        brain_s = TransformerBrain(role="soldier", registry=registry, seed=2)
        assert brain_f.weight_set is not brain_s.weight_set

    def test_four_role_weight_sets(self):
        registry = SharedWeightRegistry(seed=123)
        roles = registry.roles()
        assert set(roles) == {"forager", "nurse", "soldier", "idle"}
        weights = [registry.get(r) for r in roles]
        ids = [id(w) for w in weights]
        assert len(set(ids)) == 4


# ---------------------------------------------------------------------------
# REINFORCE training
# ---------------------------------------------------------------------------

class TestREINFORCE:
    def test_learn_no_crash(self):
        brain = TransformerBrain(seed=0)
        for _ in range(15):
            brain.decide(SensoryInput())
            brain.learn(1.0)

    def test_experience_buffer_fills(self):
        cfg = TransformerBrainConfig(buffer_size=20, update_interval=10_000)
        brain = TransformerBrain(cfg=cfg, seed=0)
        for _ in range(25):
            brain.decide(SensoryInput())
            brain.learn(1.0)
        assert len(brain.trainer.buffer) == 20

    def test_weights_change_after_update(self):
        cfg = TransformerBrainConfig(update_interval=10, buffer_size=10)
        brain = TransformerBrain(cfg=cfg, seed=0)
        w_before = brain.weight_set.W_out.copy()
        for _ in range(10):
            brain.decide(SensoryInput())
            brain.learn(1.0)
        assert not np.array_equal(brain.weight_set.W_out, w_before)

    def test_warmup_lr(self):
        ws = TransformerWeightSet(39, 32, 4, 2, 64, np.random.default_rng(0))
        trainer = RoleTrainer(ws, lr=5e-5, warmup_steps=50)
        assert trainer.effective_lr == 0.0
        trainer.global_step = 25
        assert trainer.effective_lr == pytest.approx(2.5e-5)
        trainer.global_step = 50
        assert trainer.effective_lr == pytest.approx(5e-5)

    def test_decide_learn_loop(self):
        """Full decide→learn loop should run without errors."""
        cfg = TransformerBrainConfig(buffer_size=10, update_interval=5)
        brain = TransformerBrain(cfg=cfg, seed=42)
        for step in range(20):
            sensory = _make_sensory(energy=100 - step * 2)
            action = brain.decide(sensory)
            reward = 0.1 if action.speed_mult > 0.5 else -0.1
            brain.learn(reward)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_batch_decide_returns_correct_count(self):
        rng = np.random.default_rng(42)
        registry = SharedWeightRegistry(seed=42)
        ws = registry.get("forager")
        seqs = [rng.normal(size=(4, 39)) for _ in range(5)]
        results = batch_decide(ws, seqs, rng)
        assert len(results) == 5
        for action, lp, flat in results:
            assert isinstance(action, AntAction)
            assert math.isfinite(lp)

    def test_batch_decide_empty(self):
        rng = np.random.default_rng(42)
        registry = SharedWeightRegistry(seed=42)
        ws = registry.get("forager")
        results = batch_decide(ws, [], rng)
        assert results == []

    def test_batch_decide_different_seq_lengths(self):
        """batch_decide should handle contexts of different lengths."""
        rng = np.random.default_rng(42)
        registry = SharedWeightRegistry(seed=42)
        ws = registry.get("forager")
        seqs = [
            rng.normal(size=(2, 39)),
            rng.normal(size=(8, 39)),
            rng.normal(size=(16, 39)),
        ]
        results = batch_decide(ws, seqs, rng)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class TestTransformerLayer:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        layer = TransformerLayer(32, 4, 64, rng)
        x = rng.normal(size=(8, 32))
        out, aw = layer.forward(x, causal_mask(8))
        assert out.shape == (8, 32)
        assert aw.shape == (4, 8, 8)

    def test_batched_output_shape(self):
        rng = np.random.default_rng(0)
        layer = TransformerLayer(32, 4, 64, rng)
        x = rng.normal(size=(3, 8, 32))
        out, aw = layer.forward(x, causal_mask(8))
        assert out.shape == (3, 8, 32)
        assert aw.shape == (3, 4, 8, 8)

    def test_parameters_count(self):
        rng = np.random.default_rng(0)
        layer = TransformerLayer(32, 4, 64, rng)
        params = layer.parameters()
        # MHA: 8 (W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
        # LN1: 2, LN2: 2, FFN: 4 (W1, b1, W2, b2)
        assert len(params) == 16


# ---------------------------------------------------------------------------
# TransformerWeightSet
# ---------------------------------------------------------------------------

class TestTransformerWeightSet:
    def test_forward_single(self):
        rng = np.random.default_rng(0)
        ws = TransformerWeightSet(39, 32, 4, 2, 64, rng)
        x = rng.normal(size=(4, 39))
        logits, attn = ws.forward(x)
        assert logits.shape == (_OUTPUT_DIM,)
        assert len(attn) == 2

    def test_forward_batch(self):
        rng = np.random.default_rng(0)
        ws = TransformerWeightSet(39, 32, 4, 2, 64, rng)
        x = rng.normal(size=(3, 4, 39))
        logits, attn = ws.forward(x)
        assert logits.shape == (3, _OUTPUT_DIM)

    def test_deterministic(self):
        rng = np.random.default_rng(0)
        ws = TransformerWeightSet(39, 32, 4, 2, 64, rng)
        x = np.random.default_rng(5).normal(size=(4, 39))
        out1, _ = ws.forward(x)
        out2, _ = ws.forward(x)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Performance sanity check
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_single_decide_under_5ms(self):
        """Single decide() call should be well under 5ms for d_model=32."""
        import time
        brain = TransformerBrain(seed=0)
        # Warm up context
        for _ in range(16):
            brain.decide(SensoryInput())
        # Time a full-context forward pass
        start = time.perf_counter()
        for _ in range(10):
            brain.decide(SensoryInput())
        elapsed_ms = (time.perf_counter() - start) * 1000 / 10
        assert elapsed_ms < 5.0, f"decide() took {elapsed_ms:.2f}ms, target <5ms"
