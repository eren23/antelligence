"""Tests for brains/nn_brain.py — NumPy MLP with REINFORCE policy gradient."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.actions import AntAction
from agents.sensory import SensoryInput, NeighborInfo
from agents.ant import Vec2
from brains.nn_brain import (
    MLPWeights,
    SharedWeightRegistry,
    NNBrain,
    RoleTrainer,
    RollingBuffer,
    Experience,
    forward,
    decode_output,
    sample_action,
    log_prob_of_action,
    batch_decide,
    compute_reward,
    compute_reinforce_logit_grad,
    _relu,
    _sigmoid,
    _softmax,
    _tanh,
    _discounted_returns,
    _DEPOSIT_CHANNELS,
)
from config import NNBrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def cfg() -> NNBrainConfig:
    return NNBrainConfig(
        hidden_sizes=[64, 32],
        learning_rate=1e-4,
        gamma=0.95,
        buffer_size=200,
        update_interval=100,
    )


@pytest.fixture
def weights(rng: np.random.Generator) -> MLPWeights:
    return MLPWeights.random_init(39, 64, 32, 11, rng)


@pytest.fixture
def registry() -> SharedWeightRegistry:
    return SharedWeightRegistry(input_dim=39, hidden_sizes=[64, 32], seed=123)


@pytest.fixture
def empty_sensory() -> SensoryInput:
    return SensoryInput()


@pytest.fixture
def food_sensory() -> SensoryInput:
    return SensoryInput(
        antenna_left={"food": 0.8, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        antenna_right={"food": 0.2, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        obstacle_rays=[1.0, 1.0, 1.0, 1.0, 1.0],
        nest_direction=Vec2(-1.0, 0.0),
        nest_distance=500.0,
        food_gradient=Vec2(0.5, 0.3),
        energy=90.0,
        role_value="forager",
    )


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
# Activation function tests
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_relu(x), x)

    def test_relu_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(_relu(x), np.zeros(3))

    def test_relu_mixed(self):
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(_relu(x), np.array([0.0, 0.0, 1.0]))

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 100)
        y = _sigmoid(x)
        assert np.all(y >= 0.0) and np.all(y <= 1.0)

    def test_sigmoid_midpoint(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_extreme_values(self):
        assert _sigmoid(np.array([500.0]))[0] == pytest.approx(1.0)
        assert _sigmoid(np.array([-500.0]))[0] == pytest.approx(0.0)

    def test_tanh_range(self):
        x = np.linspace(-5, 5, 100)
        y = _tanh(x)
        assert np.all(y >= -1.0) and np.all(y <= 1.0)

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = _softmax(x)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p > 0)

    def test_softmax_batch(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        p = _softmax(x, axis=-1)
        assert p.shape == (2, 3)
        np.testing.assert_allclose(p.sum(axis=-1), [1.0, 1.0])


# ---------------------------------------------------------------------------
# MLPWeights tests
# ---------------------------------------------------------------------------

class TestMLPWeights:
    def test_random_init_shapes(self, rng):
        w = MLPWeights.random_init(39, 64, 32, 11, rng)
        assert w.W1.shape == (39, 64)
        assert w.b1.shape == (64,)
        assert w.W2.shape == (64, 32)
        assert w.b2.shape == (32,)
        assert w.W_out.shape == (32, 11)
        assert w.b_out.shape == (11,)

    def test_params_returns_all(self, weights):
        params = weights.params()
        assert len(params) == 6
        shapes = [(39, 64), (64,), (64, 32), (32,), (32, 11), (11,)]
        for p, s in zip(params, shapes):
            assert p.shape == s

    def test_he_init_scale(self, rng):
        w = MLPWeights.random_init(39, 64, 32, 11, rng)
        # He init std for first layer should be ~ sqrt(2/39) ≈ 0.236
        expected_std = math.sqrt(2.0 / 39)
        actual_std = w.W1.std()
        assert actual_std == pytest.approx(expected_std, rel=0.3)

    def test_biases_start_zero(self, weights):
        np.testing.assert_array_equal(weights.b1, np.zeros(64))
        np.testing.assert_array_equal(weights.b2, np.zeros(32))
        np.testing.assert_array_equal(weights.b_out, np.zeros(11))


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_single_output_shape(self, weights):
        x = np.zeros(39)
        out, cache = forward(weights, x)
        assert out.shape == (11,)

    def test_batch_output_shape(self, weights):
        x = np.zeros((5, 39))
        out, cache = forward(weights, x)
        assert out.shape == (5, 11)

    def test_cache_shapes_single(self, weights):
        x = np.zeros(39)
        _, cache = forward(weights, x)
        assert cache.x.shape == (1, 39)
        assert cache.z1.shape == (1, 64)
        assert cache.a1.shape == (1, 64)
        assert cache.z2.shape == (1, 32)
        assert cache.a2.shape == (1, 32)
        assert cache.z_out.shape == (1, 11)

    def test_cache_shapes_batch(self, weights):
        x = np.random.randn(10, 39)
        _, cache = forward(weights, x)
        assert cache.x.shape == (10, 39)
        assert cache.z1.shape == (10, 64)
        assert cache.a1.shape == (10, 64)

    def test_deterministic(self, weights):
        x = np.random.randn(39)
        out1, _ = forward(weights, x)
        out2, _ = forward(weights, x)
        np.testing.assert_array_equal(out1, out2)

    def test_different_inputs_different_outputs(self, weights):
        x1 = np.random.randn(39)
        x2 = np.random.randn(39)
        out1, _ = forward(weights, x1)
        out2, _ = forward(weights, x2)
        assert not np.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Output decoding tests
# ---------------------------------------------------------------------------

class TestDecodeOutput:
    def test_single_decode_keys(self, weights):
        x = np.random.randn(39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)
        expected_keys = {"turn", "speed", "deposit_probs", "strength",
                         "pickup", "drop", "recruit"}
        assert set(decoded.keys()) == expected_keys

    def test_turn_range(self, weights, rng):
        """Turn output should be in [-π/6, π/6]."""
        for _ in range(20):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            turn = float(decoded["turn"][0])
            assert -math.pi / 6 <= turn <= math.pi / 6

    def test_speed_range(self, weights, rng):
        """Speed output should be in [0, 1]."""
        for _ in range(20):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            speed = float(decoded["speed"][0])
            assert 0.0 <= speed <= 1.0

    def test_deposit_probs_valid(self, weights, rng):
        """Deposit probabilities should sum to 1 and be non-negative."""
        for _ in range(20):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            probs = decoded["deposit_probs"]
            assert len(probs) == 5
            assert all(p >= 0 for p in probs)
            assert sum(probs) == pytest.approx(1.0)

    def test_sigmoid_outputs_range(self, weights, rng):
        """Strength, pickup, drop, recruit should be in [0, 1]."""
        for _ in range(20):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            for key in ("strength", "pickup", "drop", "recruit"):
                val = float(decoded[key][0])
                assert 0.0 <= val <= 1.0

    def test_batch_decode(self, weights):
        x = np.random.randn(5, 39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)
        assert decoded["turn"].shape == (5, 1)
        assert decoded["deposit_probs"].shape == (5, 5)


# ---------------------------------------------------------------------------
# Action sampling tests
# ---------------------------------------------------------------------------

class TestSampleAction:
    def test_produces_ant_action(self, weights, rng):
        x = rng.standard_normal(39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)
        action = sample_action(decoded, rng)
        assert isinstance(action, AntAction)

    def test_turn_within_bounds(self, weights, rng):
        for _ in range(50):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            action = sample_action(decoded, rng)
            assert -math.pi / 6 <= action.turn <= math.pi / 6

    def test_speed_within_bounds(self, weights, rng):
        for _ in range(50):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            action = sample_action(decoded, rng)
            assert 0.0 <= action.speed_mult <= 1.0

    def test_deposit_pheromone_valid(self, weights, rng):
        channels_seen = set()
        for _ in range(200):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            action = sample_action(decoded, rng)
            assert action.deposit_pheromone in _DEPOSIT_CHANNELS
            if action.deposit_pheromone is not None:
                channels_seen.add(action.deposit_pheromone)
        # Should see at least some non-None channels over 200 samples
        assert len(channels_seen) >= 1

    def test_boolean_fields_are_bool(self, weights, rng):
        x = rng.standard_normal(39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)
        action = sample_action(decoded, rng)
        assert isinstance(action.pickup, bool)
        assert isinstance(action.drop, bool)
        assert isinstance(action.recruit_signal, bool)

    def test_no_strength_when_no_deposit(self, weights, rng):
        """If deposit_pheromone is None, strength should be 0."""
        for _ in range(100):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            action = sample_action(decoded, rng)
            if action.deposit_pheromone is None:
                assert action.deposit_strength == 0.0


# ---------------------------------------------------------------------------
# Log-probability tests
# ---------------------------------------------------------------------------

class TestLogProb:
    def test_finite(self, weights, rng):
        x = rng.standard_normal(39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)
        action = sample_action(decoded, rng)
        lp = log_prob_of_action(decoded, action)
        assert math.isfinite(lp)

    def test_negative(self, weights, rng):
        """Log-prob should be negative (prob < 1)."""
        x = rng.standard_normal(39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)
        action = sample_action(decoded, rng)
        lp = log_prob_of_action(decoded, action)
        assert lp < 0

    def test_higher_prob_for_modal_action(self, weights, rng):
        """The most probable deposit channel should have higher log-prob."""
        x = rng.standard_normal(39)
        raw, _ = forward(weights, x)
        decoded = decode_output(raw)

        # Find the most likely deposit channel
        probs = decoded["deposit_probs"]
        best_idx = int(np.argmax(probs))

        action_best = sample_action(decoded, rng)
        action_best.deposit_pheromone = _DEPOSIT_CHANNELS[best_idx]

        worst_idx = int(np.argmin(probs))
        action_worst = sample_action(decoded, rng)
        action_worst.deposit_pheromone = _DEPOSIT_CHANNELS[worst_idx]

        # Keep everything else the same
        action_worst.turn = action_best.turn
        action_worst.speed_mult = action_best.speed_mult
        action_worst.deposit_strength = action_best.deposit_strength
        action_worst.pickup = action_best.pickup
        action_worst.drop = action_best.drop
        action_worst.recruit_signal = action_best.recruit_signal

        lp_best = log_prob_of_action(decoded, action_best)
        lp_worst = log_prob_of_action(decoded, action_worst)
        assert lp_best >= lp_worst


# ---------------------------------------------------------------------------
# Sensory encoding tests
# ---------------------------------------------------------------------------

class TestSensoryEncoding:
    def test_vector_length(self, empty_sensory):
        vec = empty_sensory.to_vector()
        assert len(vec) == 39

    def test_vector_all_finite(self, food_sensory):
        vec = food_sensory.to_vector()
        assert all(math.isfinite(v) for v in vec)

    def test_vector_different_for_different_inputs(self, empty_sensory, food_sensory):
        v1 = empty_sensory.to_vector()
        v2 = food_sensory.to_vector()
        assert v1 != v2


# ---------------------------------------------------------------------------
# NNBrain protocol compliance tests
# ---------------------------------------------------------------------------

class TestNNBrainProtocol:
    def test_has_decide(self):
        brain = NNBrain(seed=42)
        assert hasattr(brain, "decide")
        assert callable(brain.decide)

    def test_has_learn(self):
        brain = NNBrain(seed=42)
        assert hasattr(brain, "learn")
        assert callable(brain.learn)

    def test_decide_returns_ant_action(self, empty_sensory):
        brain = NNBrain(seed=42)
        action = brain.decide(empty_sensory)
        assert isinstance(action, AntAction)

    def test_learn_accepts_reward(self, empty_sensory):
        brain = NNBrain(seed=42)
        brain.decide(empty_sensory)
        brain.learn(1.0)  # Should not raise

    def test_decide_output_valid_ranges(self, food_sensory):
        brain = NNBrain(seed=42)
        action = brain.decide(food_sensory)
        assert -math.pi / 6 <= action.turn <= math.pi / 6
        assert 0.0 <= action.speed_mult <= 1.0
        assert action.deposit_pheromone in _DEPOSIT_CHANNELS
        assert 0.0 <= action.deposit_strength <= 1.0
        assert isinstance(action.pickup, bool)
        assert isinstance(action.drop, bool)
        assert isinstance(action.recruit_signal, bool)

    def test_runtime_checkable(self):
        """NNBrain should satisfy BrainBackend protocol."""
        from brains.interface import BrainBackend
        brain = NNBrain(seed=42)
        assert isinstance(brain, BrainBackend)


# ---------------------------------------------------------------------------
# Weight sharing tests
# ---------------------------------------------------------------------------

class TestWeightSharing:
    def test_same_role_shares_weights(self, registry):
        brain1 = NNBrain(role="forager", registry=registry, seed=1)
        brain2 = NNBrain(role="forager", registry=registry, seed=2)
        # Both should reference the same weight arrays
        assert brain1.weights is brain2.weights
        assert brain1.weights.W1 is brain2.weights.W1

    def test_different_role_different_weights(self, registry):
        brain_f = NNBrain(role="forager", registry=registry, seed=1)
        brain_s = NNBrain(role="soldier", registry=registry, seed=2)
        assert brain_f.weights is not brain_s.weights

    def test_four_role_weight_sets(self, registry):
        roles = registry.roles()
        assert set(roles) == {"forager", "nurse", "soldier", "idle"}
        weights_list = [registry.get(r) for r in roles]
        # All different objects
        ids = [id(w) for w in weights_list]
        assert len(set(ids)) == 4

    def test_weight_modification_propagates(self, registry):
        """Modifying weights through one brain should affect another of same role."""
        brain1 = NNBrain(role="nurse", registry=registry, seed=1)
        brain2 = NNBrain(role="nurse", registry=registry, seed=2)
        old_val = brain1.weights.W1[0, 0]
        brain1.weights.W1[0, 0] = 999.0
        assert brain2.weights.W1[0, 0] == 999.0
        brain1.weights.W1[0, 0] = old_val  # Restore


# ---------------------------------------------------------------------------
# Batch vs individual consistency tests
# ---------------------------------------------------------------------------

class TestBatchConsistency:
    def test_batch_forward_matches_individual(self, weights):
        """Batch forward pass should give same results as individual passes."""
        rng = np.random.default_rng(99)
        inputs = [rng.standard_normal(39) for _ in range(5)]

        # Individual
        individual_outs = []
        for x in inputs:
            out, _ = forward(weights, x)
            individual_outs.append(out)

        # Batch
        batch_in = np.array(inputs)
        batch_out, _ = forward(weights, batch_in)

        for i in range(5):
            np.testing.assert_allclose(batch_out[i], individual_outs[i], atol=1e-12)

    def test_batch_decide_same_network_output(self, registry):
        """batch_decide should produce actions from same forward pass."""
        rng = np.random.default_rng(77)
        weights = registry.get("forager")
        sensory_list = [_make_sensory(energy=50 + i * 10) for i in range(5)]
        results = batch_decide(weights, sensory_list, rng)
        assert len(results) == 5
        for action, lp, vec in results:
            assert isinstance(action, AntAction)
            assert math.isfinite(lp)
            assert len(vec) == 39

    def test_batch_decide_empty_list(self, registry):
        rng = np.random.default_rng(1)
        results = batch_decide(registry.get("forager"), [], rng)
        assert results == []


# ---------------------------------------------------------------------------
# Rolling buffer tests
# ---------------------------------------------------------------------------

class TestRollingBuffer:
    def _dummy_exp(self, reward: float = 0.0) -> Experience:
        return Experience(
            sensory_vec=np.zeros(39),
            action=AntAction(),
            log_prob=-1.0,
            reward=reward,
        )

    def test_add_and_len(self):
        buf = RollingBuffer(capacity=5)
        assert len(buf) == 0
        buf.add(self._dummy_exp())
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = RollingBuffer(capacity=3)
        for i in range(10):
            buf.add(self._dummy_exp(float(i)))
        assert len(buf) == 3

    def test_full_property(self):
        buf = RollingBuffer(capacity=3)
        assert not buf.full
        for _ in range(3):
            buf.add(self._dummy_exp())
        assert buf.full

    def test_circular_overwrites_oldest(self):
        buf = RollingBuffer(capacity=3)
        for i in range(5):
            buf.add(self._dummy_exp(float(i)))
        all_exp = buf.get_all()
        rewards = [e.reward for e in all_exp]
        # Should contain the last 3: 2.0, 3.0, 4.0
        assert rewards == [2.0, 3.0, 4.0]

    def test_clear(self):
        buf = RollingBuffer(capacity=5)
        for _ in range(5):
            buf.add(self._dummy_exp())
        buf.clear()
        assert len(buf) == 0
        assert not buf.full


# ---------------------------------------------------------------------------
# Discounted returns tests
# ---------------------------------------------------------------------------

class TestDiscountedReturns:
    def test_single_reward(self):
        rewards = np.array([1.0])
        returns = _discounted_returns(rewards, 0.95)
        assert len(returns) == 1
        # Single reward: normalized to 0 (mean=itself, std=0, so raw returned)
        assert math.isfinite(returns[0])

    def test_monotone_discount(self):
        rewards = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        returns = _discounted_returns(rewards, 0.95)
        # Later rewards should be discounted less for earlier timesteps
        # After normalization, the last return should be highest
        assert returns[-1] > returns[0]

    def test_gamma_zero_means_no_future(self):
        rewards = np.array([1.0, 2.0, 3.0])
        returns = _discounted_returns(rewards, 0.0)
        # With γ=0, each return equals just its own reward (then normalized)
        assert len(returns) == 3

    def test_length_preserved(self):
        rewards = np.random.randn(50)
        returns = _discounted_returns(rewards, 0.95)
        assert len(returns) == 50

    def test_normalized(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = _discounted_returns(rewards, 0.95)
        # Should be approximately zero mean unit variance
        assert abs(returns.mean()) < 0.1
        assert abs(returns.std() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Reward computation tests
# ---------------------------------------------------------------------------

class TestRewardComputation:
    def test_food_deposit_reward(self):
        prev = _make_sensory(carrying="food")
        curr = _make_sensory(carrying=None)
        r = compute_reward(prev, curr, AntAction(speed_mult=0.5), alive=True)
        assert r >= 1.0  # +1.0 for food deposit

    def test_food_pickup_reward(self):
        prev = _make_sensory(carrying=None)
        curr = _make_sensory(carrying="food")
        r = compute_reward(prev, curr, AntAction(speed_mult=0.5), alive=True)
        assert r >= 0.3  # +0.3 for pickup

    def test_death_penalty(self):
        curr = _make_sensory(energy=0)
        r = compute_reward(None, curr, AntAction(), alive=False)
        assert r == -0.5

    def test_high_energy_bonus(self):
        curr = _make_sensory(energy=80)
        r = compute_reward(None, curr, AntAction(speed_mult=0.5), alive=True)
        assert r > 0  # Should have energy bonus + exploration bonus

    def test_low_energy_penalty(self):
        curr = _make_sensory(energy=10)
        r = compute_reward(None, curr, AntAction(speed_mult=0.0), alive=True)
        assert r < 0  # Low energy penalty

    def test_exploration_bonus(self):
        curr = _make_sensory(energy=80)
        r_fast = compute_reward(None, curr, AntAction(speed_mult=0.5), alive=True)
        r_slow = compute_reward(None, curr, AntAction(speed_mult=0.1), alive=True)
        assert r_fast > r_slow  # Moving faster gets bonus


# ---------------------------------------------------------------------------
# Gradient / backprop tests
# ---------------------------------------------------------------------------

class TestGradients:
    def test_gradients_non_zero(self, weights, rng):
        """REINFORCE gradients should be non-zero for non-zero advantages."""
        x = rng.standard_normal((10, 39))
        raw, cache = forward(weights, x)
        actions = []
        for i in range(10):
            decoded = decode_output(raw[i])
            actions.append(sample_action(decoded, rng))
        advantages = np.ones(10)

        grads = compute_reinforce_logit_grad(weights, cache, actions, advantages)
        assert len(grads) == 6
        # At least some gradients should be non-zero
        for g in grads:
            assert np.any(g != 0), f"Gradient of shape {g.shape} is all zeros"

    def test_gradient_shapes_match_params(self, weights, rng):
        x = rng.standard_normal((5, 39))
        raw, cache = forward(weights, x)
        actions = []
        for i in range(5):
            decoded = decode_output(raw[i])
            actions.append(sample_action(decoded, rng))
        advantages = np.ones(5)
        grads = compute_reinforce_logit_grad(weights, cache, actions, advantages)
        for g, p in zip(grads, weights.params()):
            assert g.shape == p.shape

    def test_zero_advantage_zero_gradient(self, weights, rng):
        """Zero advantage should produce zero gradients."""
        x = rng.standard_normal((5, 39))
        raw, cache = forward(weights, x)
        actions = []
        for i in range(5):
            decoded = decode_output(raw[i])
            actions.append(sample_action(decoded, rng))
        advantages = np.zeros(5)
        grads = compute_reinforce_logit_grad(weights, cache, actions, advantages)
        for g in grads:
            np.testing.assert_allclose(g, 0.0, atol=1e-15)

    def test_gradient_scales_with_advantage(self, weights, rng):
        """Doubling advantage should double the gradient."""
        x = rng.standard_normal((5, 39))
        raw, cache = forward(weights, x)
        actions = []
        for i in range(5):
            decoded = decode_output(raw[i])
            actions.append(sample_action(decoded, rng))

        adv1 = np.ones(5)
        adv2 = np.ones(5) * 2.0
        grads1 = compute_reinforce_logit_grad(weights, cache, actions, adv1)
        grads2 = compute_reinforce_logit_grad(weights, cache, actions, adv2)

        for g1, g2 in zip(grads1, grads2):
            np.testing.assert_allclose(g2, g1 * 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# RoleTrainer tests
# ---------------------------------------------------------------------------

class TestRoleTrainer:
    def test_warmup_lr(self, weights):
        trainer = RoleTrainer(weights, lr=1e-4, warmup_steps=50)
        assert trainer.effective_lr == 0.0  # step 0
        trainer.global_step = 25
        assert trainer.effective_lr == pytest.approx(5e-5)  # half warmup
        trainer.global_step = 50
        assert trainer.effective_lr == pytest.approx(1e-4)  # full lr

    def test_update_changes_weights(self, rng):
        weights = MLPWeights.random_init(39, 64, 32, 11, rng)
        trainer = RoleTrainer(weights, lr=1e-3, buffer_size=20, warmup_steps=0)

        # Fill buffer with diverse experiences
        for i in range(20):
            x = rng.standard_normal(39)
            raw, _ = forward(weights, x)
            decoded = decode_output(raw)
            action = sample_action(decoded, rng)
            exp = Experience(
                sensory_vec=x,
                action=action,
                log_prob=log_prob_of_action(decoded, action),
                reward=float(rng.standard_normal()),
            )
            trainer.buffer.add(exp)

        old_w1 = weights.W1.copy()
        trainer.update()
        # Weights should have changed
        assert not np.allclose(weights.W1, old_w1)


# ---------------------------------------------------------------------------
# Learning convergence test
# ---------------------------------------------------------------------------

class TestLearningConvergence:
    def test_learns_to_prefer_high_reward_action(self):
        """Train on a simple scenario: high reward for pickup=True.

        After training, the network should increase pickup probability.
        """
        rng = np.random.default_rng(42)
        weights = MLPWeights.random_init(39, 64, 32, 11, rng)
        trainer = RoleTrainer(weights, lr=5e-3, buffer_size=50, warmup_steps=0)

        fixed_input = rng.standard_normal(39)

        # Measure initial pickup probability
        raw_init, _ = forward(weights, fixed_input)
        decoded_init = decode_output(raw_init)
        initial_pickup_prob = float(decoded_init["pickup"][0])

        # Train: give high reward when pickup=True, low when pickup=False
        for epoch in range(30):
            for _ in range(50):
                raw, _ = forward(weights, fixed_input)
                decoded = decode_output(raw)
                action = sample_action(decoded, rng)

                # Reward proportional to pickup probability
                reward = 2.0 if action.pickup else -1.0
                exp = Experience(
                    sensory_vec=fixed_input,
                    action=action,
                    log_prob=log_prob_of_action(decoded, action),
                    reward=reward,
                )
                trainer.buffer.add(exp)
            trainer.update()

        # After training, pickup prob should increase
        raw_final, _ = forward(weights, fixed_input)
        decoded_final = decode_output(raw_final)
        final_pickup_prob = float(decoded_final["pickup"][0])

        assert final_pickup_prob > initial_pickup_prob, (
            f"Expected pickup prob to increase: {initial_pickup_prob:.4f} -> {final_pickup_prob:.4f}"
        )

    def test_learns_deposit_channel_preference(self):
        """Train to prefer depositing 'food' pheromone."""
        rng = np.random.default_rng(123)
        weights = MLPWeights.random_init(39, 64, 32, 11, rng)
        trainer = RoleTrainer(weights, lr=5e-3, buffer_size=50, warmup_steps=0)

        fixed_input = rng.standard_normal(39)

        raw_init, _ = forward(weights, fixed_input)
        decoded_init = decode_output(raw_init)
        initial_food_prob = float(decoded_init["deposit_probs"][1])  # food is index 1

        for epoch in range(30):
            for _ in range(50):
                raw, _ = forward(weights, fixed_input)
                decoded = decode_output(raw)
                action = sample_action(decoded, rng)
                reward = 2.0 if action.deposit_pheromone == "food" else -0.5
                exp = Experience(
                    sensory_vec=fixed_input,
                    action=action,
                    log_prob=log_prob_of_action(decoded, action),
                    reward=reward,
                )
                trainer.buffer.add(exp)
            trainer.update()

        raw_final, _ = forward(weights, fixed_input)
        decoded_final = decode_output(raw_final)
        final_food_prob = float(decoded_final["deposit_probs"][1])

        assert final_food_prob > initial_food_prob, (
            f"Expected food deposit prob to increase: {initial_food_prob:.4f} -> {final_food_prob:.4f}"
        )


# ---------------------------------------------------------------------------
# Integration: full NNBrain decide→learn loop
# ---------------------------------------------------------------------------

class TestNNBrainIntegration:
    def test_decide_learn_loop(self):
        """NNBrain should handle a full decide→learn loop without errors."""
        brain = NNBrain(role="forager", seed=42,
                        cfg=NNBrainConfig(buffer_size=10, update_interval=5))
        for step in range(20):
            sensory = _make_sensory(energy=100 - step * 2)
            action = brain.decide(sensory)
            reward = 0.1 if action.speed_mult > 0.5 else -0.1
            brain.learn(reward)

    def test_different_roles_diverge(self):
        """Different roles should produce different actions after some training."""
        registry = SharedWeightRegistry(seed=42)
        cfg = NNBrainConfig(buffer_size=10, update_interval=5)

        brain_f = NNBrain(role="forager", registry=registry, seed=1, cfg=cfg)
        brain_s = NNBrain(role="soldier", registry=registry, seed=2, cfg=cfg)

        sensory = _make_sensory()
        action_f = brain_f.decide(sensory)
        action_s = brain_s.decide(sensory)

        # Different weights → different actions (at least turn or speed differ)
        assert (action_f.turn != action_s.turn or
                action_f.speed_mult != action_s.speed_mult)

    def test_compute_reward_convenience(self):
        brain = NNBrain(seed=42)
        prev = _make_sensory(carrying="food")
        curr = _make_sensory(carrying=None)
        r = brain.compute_reward(prev, curr, AntAction(speed_mult=0.5), alive=True)
        assert r >= 1.0
