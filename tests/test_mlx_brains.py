"""Tests for MLX-accelerated brain backends."""

from __future__ import annotations

import math

import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX not available")

from agents.actions import AntAction
from agents.sensory import SensoryInput
from brains.mlx_nn_brain import (
    MLPModel,
    MLXNNBrain,
    SharedWeightRegistry as NNRegistry,
    RoleTrainer as NNRoleTrainer,
    _decode_output_mlx,
)
from brains.mlx_transformer_brain import (
    CausalTransformerModel,
    MLXTransformerBrain,
    SharedWeightRegistry as TFRegistry,
    RoleTrainer as TFRoleTrainer,
)
from config import NNBrainConfig, TransformerBrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sensory(**kwargs) -> SensoryInput:
    defaults = dict(
        antenna_left={"food": 0.0, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        antenna_right={"food": 0.0, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        obstacle_rays=[1.0, 1.0, 1.0, 1.0, 1.0],
        nest_distance=100.0,
        energy=80.0,
        role_value="forager",
    )
    defaults.update(kwargs)
    return SensoryInput(**defaults)


def _mlx_evaluate(*arrays):
    """Force MLX lazy graph evaluation (this is NOT Python's eval)."""
    mlx.eval(*arrays)


# ---------------------------------------------------------------------------
# MLX NN Brain tests
# ---------------------------------------------------------------------------

class TestMLPModel:
    def test_forward_shape_single(self):
        model = MLPModel(39, 64, 32, 11)
        x = mlx.zeros((1, 39))
        out = model(x)
        _mlx_evaluate(out)
        assert out.shape == (1, 11)

    def test_forward_shape_batch(self):
        model = MLPModel(39, 64, 32, 11)
        x = mlx.zeros((8, 39))
        out = model(x)
        _mlx_evaluate(out)
        assert out.shape == (8, 11)

    def test_output_varies_with_input(self):
        model = MLPModel(39, 64, 32, 11)
        x1 = mlx.zeros((1, 39))
        x2 = mlx.ones((1, 39))
        out1 = model(x1)
        out2 = model(x2)
        _mlx_evaluate(out1, out2)
        assert out1.tolist() != out2.tolist()


class TestMLXNNBrain:
    def test_decide_returns_ant_action(self):
        brain = MLXNNBrain(role="forager", seed=42)
        sensory = _make_sensory()
        action = brain.decide(sensory)
        assert isinstance(action, AntAction)

    def test_output_ranges(self):
        brain = MLXNNBrain(role="forager", seed=42)
        for _ in range(10):
            action = brain.decide(_make_sensory())
            assert -math.pi / 6 - 0.01 <= action.turn <= math.pi / 6 + 0.01
            assert 0.0 <= action.speed_mult <= 1.0
            assert 0.0 <= action.deposit_strength <= 1.0

    def test_learn_does_not_crash(self):
        brain = MLXNNBrain(role="forager", seed=42)
        brain.decide(_make_sensory())
        brain.learn(0.5)
        brain.decide(_make_sensory())
        brain.learn(-0.3)

    def test_shared_weights_across_roles(self):
        registry = NNRegistry(input_dim=39, seed=42)
        b1 = MLXNNBrain(role="forager", registry=registry, seed=1)
        b2 = MLXNNBrain(role="forager", registry=registry, seed=2)
        assert b1.model is b2.model

    def test_different_roles_different_weights(self):
        registry = NNRegistry(input_dim=39, seed=42)
        b1 = MLXNNBrain(role="forager", registry=registry, seed=1)
        b2 = MLXNNBrain(role="nurse", registry=registry, seed=1)
        assert b1.model is not b2.model

    def test_compatible_action_format(self):
        """MLX brain should produce same action field types as NumPy brain."""
        brain = MLXNNBrain(role="forager", seed=42)
        action = brain.decide(_make_sensory())

        assert isinstance(action.turn, float)
        assert isinstance(action.speed_mult, float)
        assert action.deposit_pheromone is None or isinstance(action.deposit_pheromone, str)
        assert isinstance(action.deposit_strength, float)
        assert isinstance(action.pickup, bool)
        assert isinstance(action.drop, bool)
        assert isinstance(action.recruit_signal, bool)

    def test_training_convergence(self):
        """Train with constant positive reward -- pickup probability should increase."""
        cfg = NNBrainConfig(buffer_size=50, update_interval=50)
        brain = MLXNNBrain(role="forager", cfg=cfg, seed=42)

        pickup_probs = []
        for i in range(200):
            s = _make_sensory()
            action = brain.decide(s)
            reward = 1.0 if action.pickup else -0.5
            brain.learn(reward)

            if i >= 150:
                pickup_probs.append(1.0 if action.pickup else 0.0)

        avg_pickup = sum(pickup_probs) / len(pickup_probs) if pickup_probs else 0
        assert avg_pickup > 0.2, f"Pickup rate {avg_pickup:.2f} too low after training"


# ---------------------------------------------------------------------------
# MLX Transformer Brain tests
# ---------------------------------------------------------------------------

class TestCausalTransformerModel:
    def test_forward_shape_single(self):
        model = CausalTransformerModel(39, 32, 4, 2, 64)
        x = mlx.zeros((1, 4, 39))
        out = model(x)
        _mlx_evaluate(out)
        assert out.shape == (1, 11)

    def test_forward_shape_batch(self):
        model = CausalTransformerModel(39, 32, 4, 2, 64)
        x = mlx.zeros((4, 8, 39))
        out = model(x)
        _mlx_evaluate(out)
        assert out.shape == (4, 11)

    def test_different_seq_lengths(self):
        model = CausalTransformerModel(39, 32, 4, 2, 64)
        x1 = mlx.zeros((1, 1, 39))
        x2 = mlx.zeros((1, 16, 39))
        out1 = model(x1)
        out2 = model(x2)
        _mlx_evaluate(out1, out2)
        assert out1.shape == (1, 11)
        assert out2.shape == (1, 11)


class TestMLXTransformerBrain:
    def test_decide_returns_ant_action(self):
        brain = MLXTransformerBrain(role="forager", seed=42)
        action = brain.decide(_make_sensory())
        assert isinstance(action, AntAction)

    def test_context_window_fills(self):
        cfg = TransformerBrainConfig(context_length=4)
        brain = MLXTransformerBrain(role="forager", cfg=cfg, seed=42)

        for i in range(6):
            brain.decide(_make_sensory())

        assert len(brain.context) == 4

    def test_output_ranges(self):
        brain = MLXTransformerBrain(role="forager", seed=42)
        for _ in range(10):
            action = brain.decide(_make_sensory())
            assert -math.pi / 6 - 0.01 <= action.turn <= math.pi / 6 + 0.01
            assert 0.0 <= action.speed_mult <= 1.0

    def test_learn_does_not_crash(self):
        brain = MLXTransformerBrain(role="forager", seed=42)
        brain.decide(_make_sensory())
        brain.learn(0.5)
        brain.decide(_make_sensory())
        brain.learn(-0.3)

    def test_shared_weights_across_roles(self):
        registry = TFRegistry(seed=42)
        b1 = MLXTransformerBrain(role="forager", registry=registry, seed=1)
        b2 = MLXTransformerBrain(role="forager", registry=registry, seed=2)
        assert b1.model is b2.model

    def test_compatible_action_format(self):
        brain = MLXTransformerBrain(role="forager", seed=42)
        action = brain.decide(_make_sensory())

        assert isinstance(action.turn, float)
        assert isinstance(action.speed_mult, float)
        assert action.deposit_pheromone is None or isinstance(action.deposit_pheromone, str)
        assert isinstance(action.pickup, bool)
        assert isinstance(action.drop, bool)

    def test_batch_consistency(self):
        """Two brains with same seed and same input produce same output."""
        cfg = TransformerBrainConfig(context_length=4)
        b1 = MLXTransformerBrain(role="forager", cfg=cfg, seed=42)
        b2 = MLXTransformerBrain(role="forager", cfg=cfg, seed=42)

        s = _make_sensory()
        a1 = b1.decide(s)
        a2 = b2.decide(s)

        assert a1.turn == a2.turn
        assert a1.speed_mult == a2.speed_mult


# ---------------------------------------------------------------------------
# Decode output tests
# ---------------------------------------------------------------------------

class TestDecodeOutput:
    def test_decoded_keys(self):
        raw = mlx.zeros((11,))
        _mlx_evaluate(raw)
        decoded = _decode_output_mlx(raw)
        assert "turn" in decoded
        assert "speed" in decoded
        assert "deposit_probs" in decoded
        assert "strength" in decoded
        assert "pickup" in decoded
        assert "drop" in decoded
        assert "recruit" in decoded

    def test_deposit_probs_sum_to_one(self):
        raw = mlx.array([0.1, -0.5, 1.0, 0.2, -0.3, 0.7, 0.0, 0.5, -0.2, 0.3, 0.1])
        _mlx_evaluate(raw)
        decoded = _decode_output_mlx(raw)
        assert abs(sum(decoded["deposit_probs"]) - 1.0) < 1e-6

    def test_turn_range(self):
        for val in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            raw = mlx.array([val] + [0.0] * 10)
            _mlx_evaluate(raw)
            decoded = _decode_output_mlx(raw)
            assert -math.pi / 6 <= decoded["turn"] <= math.pi / 6

    def test_sigmoid_outputs_in_zero_one(self):
        raw = mlx.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 2.0, -5.0, 4.0])
        _mlx_evaluate(raw)
        decoded = _decode_output_mlx(raw)
        assert 0.0 < decoded["speed"] < 1.0
        assert 0.0 < decoded["strength"] < 1.0
        assert 0.0 < decoded["pickup"] < 1.0
        assert 0.0 < decoded["drop"] < 1.0
        assert 0.0 < decoded["recruit"] < 1.0
