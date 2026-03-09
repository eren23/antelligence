"""Imitation learning — behavioral cloning from the rule-based brain.

Phase 1 of the REINFORCE → Imitation + PPO migration:
  1. DemonstrationCollector: runs headless sim with RuleBasedBrain, records
     (sensory_vector, action_target) pairs from forager ants.
  2. ImitationTrainer: supervised training on demonstrations using mixed loss
     (MSE for continuous heads, cross-entropy for categorical/binary heads).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from agents.actions import AntAction
from agents.sensory import SensoryInput
from brains.nn_brain import (
    MLPWeights,
    ForwardCache,
    _DEPOSIT_CHANNELS,
    _backprop_logits_gradient,
    _sigmoid,
    _softmax,
    _tanh,
    forward,
)


# ---------------------------------------------------------------------------
# Action → target vector encoding
# ---------------------------------------------------------------------------

def encode_action_target(action: AntAction) -> np.ndarray:
    """Encode an AntAction as a target vector (11 floats) matching network output.

    Layout:
        [0]     turn value (raw, pre-tanh — we invert tanh to get logit)
        [1]     speed_mult (raw, pre-sigmoid — we invert sigmoid to get logit)
        [2:7]   one-hot for deposit channel (5 values: none, food, home, danger, recruit)
        [7]     deposit_strength (raw, pre-sigmoid logit)
        [8:11]  pickup, drop, recruit as 0.0/1.0
    """
    target = np.zeros(11, dtype=np.float64)

    # Turn: invert tanh*scale to get logit
    scale = math.pi / 6.0
    t_normalized = max(-0.999, min(0.999, action.turn / scale))
    target[0] = np.arctanh(t_normalized)

    # Speed: invert sigmoid to get logit
    s = max(1e-6, min(1.0 - 1e-6, action.speed_mult))
    target[1] = np.log(s / (1.0 - s))

    # Deposit channel: one-hot
    deposit_idx = _DEPOSIT_CHANNELS.index(action.deposit_pheromone)
    target[2 + deposit_idx] = 1.0

    # Deposit strength: invert sigmoid
    st = max(1e-6, min(1.0 - 1e-6, action.deposit_strength))
    target[7] = np.log(st / (1.0 - st))

    # Binary decisions
    target[8] = 1.0 if action.pickup else 0.0
    target[9] = 1.0 if action.drop else 0.0
    target[10] = 1.0 if action.recruit_signal else 0.0

    return target


# ---------------------------------------------------------------------------
# Demonstration collector
# ---------------------------------------------------------------------------

class _RecordingBrain:
    """Wrapper around RuleBasedBrain that records (sensory, action) pairs."""

    def __init__(self, inner, role: str, store: list):
        self._inner = inner
        self._role = role
        self._store = store

    def decide(self, sensory: SensoryInput) -> AntAction:
        action = self._inner.decide(sensory)
        self._store.append((sensory, action))
        return action

    def learn(self, reward: float) -> None:
        self._inner.learn(reward)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class DemonstrationCollector:
    """Runs headless simulation with RuleBasedBrain, collecting (state, action) pairs."""

    def __init__(
        self,
        ticks: int = 5000,
        num_ants: int = 200,
        seed: int = 42,
        config_path: str = "colony_config.yaml",
        roles: list[str] | None = None,
    ):
        self.ticks = ticks
        self.num_ants = num_ants
        self.seed = seed
        self.config_path = config_path
        self.roles = roles or ["forager"]

    def collect(self, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Run sim and return (states, targets) arrays.

        Returns:
            states: (N, 39) float64 array of sensory vectors
            targets: (N, 11) float64 array of action targets
        """
        from brains.rule_based import RuleBasedBrain
        from config import load_config, default_config
        from agents.colony import Colony
        from world.world import World
        from world.pheromone import PheromoneGrid

        # Setup
        random.seed(self.seed)
        np.random.seed(self.seed)

        cfg_path = Path(self.config_path)
        cfg = load_config(cfg_path) if cfg_path.exists() else default_config()
        cfg.brain.default = "rule_based"
        cfg.colony.initial_population = self.num_ants

        world = World.from_config(cfg, seed=self.seed)
        pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
        colony = Colony(cfg, world.nest.center, seed=self.seed)

        # Recording store — collecting (sensory, action) pairs per tick
        demo_store: list[tuple[SensoryInput, AntAction]] = []

        # Assign recording-wrapped rule-based brains
        for ant in colony.ants:
            inner = RuleBasedBrain(
                world_width=cfg.world.width,
                world_height=cfg.world.height,
                rng_seed=ant.id + self.seed,
            )
            if ant.role.value in self.roles:
                ant.brain = _RecordingBrain(inner, ant.role.value, demo_store)
            else:
                ant.brain = inner

        from main import sim_tick, BrainManager
        from metrics.tracker import MetricsTracker
        from metrics.emergence import EmergenceDetector

        brain_mgr = BrainManager(cfg, self.seed)

        metrics = MetricsTracker(window=0)
        emergence = EmergenceDetector(cfg.roles.default_distribution)

        states_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        for tick in range(1, self.ticks + 1):
            demo_store.clear()

            # sim_tick calls brain.decide() which triggers recording
            sim_tick(
                tick, colony, world, pheromone_grid,
                brain_mgr, metrics, emergence, cfg,
                learn=False,
            )

            # Convert recorded pairs
            for sensory, action in demo_store:
                vec = np.array(sensory.to_vector(), dtype=np.float64)
                target = encode_action_target(action)
                states_list.append(vec)
                targets_list.append(target)

            # Newly spawned ants need recording brains too
            for ant in colony.ants:
                if ant.alive and ant.brain is not None and not isinstance(ant.brain, (_RecordingBrain, RuleBasedBrain)):
                    inner = RuleBasedBrain(
                        world_width=cfg.world.width,
                        world_height=cfg.world.height,
                        rng_seed=ant.id + self.seed,
                    )
                    if ant.role.value in self.roles:
                        ant.brain = _RecordingBrain(inner, ant.role.value, demo_store)
                    else:
                        ant.brain = inner

            if verbose and tick % 1000 == 0:
                print(f"  [demo] Tick {tick:>5}/{self.ticks}  |  samples: {len(states_list):>8,}")

        states = np.array(states_list, dtype=np.float64)
        targets = np.array(targets_list, dtype=np.float64)

        if verbose:
            print(f"  [demo] Collected {len(states):,} demonstrations from {self.ticks} ticks")

        return states, targets


# ---------------------------------------------------------------------------
# Imitation gradient computation (analytical)
# ---------------------------------------------------------------------------

def compute_imitation_logit_grad(
    z_out: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compute gradient of imitation loss w.r.t. raw output logits.

    Mixed loss:
        - Turn (idx 0): MSE on tanh(z)*scale vs target turn logit → tanh(target)*scale
        - Speed (idx 1): MSE on sigmoid(z) vs sigmoid(target)
        - Deposit (idx 2:7): cross-entropy on softmax logits vs one-hot target
        - Strength (idx 7): MSE on sigmoid(z) vs sigmoid(target)
        - Pickup/drop/recruit (idx 8:11): binary cross-entropy on sigmoid vs target

    Args:
        z_out: (batch, 11) raw output logits
        targets: (batch, 11) target vectors from encode_action_target

    Returns:
        dL_dz: (batch, 11) gradient of loss w.r.t. z_out
    """
    batch = z_out.shape[0]
    dL_dz = np.zeros_like(z_out)
    scale = math.pi / 6.0

    # Turn head (idx 0): MSE on post-activation
    # Loss = 0.5 * (tanh(z)*scale - tanh(t)*scale)^2
    # dL/dz = (tanh(z)*scale - tanh(t)*scale) * scale * (1 - tanh(z)^2)
    z_tanh = np.tanh(z_out[:, 0])
    t_tanh = np.tanh(targets[:, 0])
    dL_dz[:, 0] = (z_tanh * scale - t_tanh * scale) * scale * (1.0 - z_tanh ** 2)

    # Speed head (idx 1): MSE on sigmoid output
    # Loss = 0.5 * (sigmoid(z) - sigmoid(t))^2
    # dL/dz = (sigmoid(z) - sigmoid(t)) * sigmoid(z) * (1 - sigmoid(z))
    z_sig = _sigmoid(z_out[:, 1:2]).squeeze(-1)
    t_sig = _sigmoid(targets[:, 1:2]).squeeze(-1)
    dL_dz[:, 1] = (z_sig - t_sig) * z_sig * (1.0 - z_sig)

    # Deposit head (idx 2:7): cross-entropy on softmax
    # Loss = -sum(target * log(softmax(z)))
    # dL/dz = softmax(z) - target
    probs = _softmax(z_out[:, 2:7], axis=-1)
    dL_dz[:, 2:7] = probs - targets[:, 2:7]

    # Strength head (idx 7): MSE on sigmoid
    z_str = _sigmoid(z_out[:, 7:8]).squeeze(-1)
    t_str = _sigmoid(targets[:, 7:8]).squeeze(-1)
    dL_dz[:, 7] = (z_str - t_str) * z_str * (1.0 - z_str)

    # Binary heads (idx 8:11): binary cross-entropy
    # Loss = -[t*log(sigmoid(z)) + (1-t)*log(1-sigmoid(z))]
    # dL/dz = sigmoid(z) - t
    for j in range(3):
        sig = _sigmoid(z_out[:, 8 + j:9 + j]).squeeze(-1)
        dL_dz[:, 8 + j] = sig - targets[:, 8 + j]

    return dL_dz


# ---------------------------------------------------------------------------
# Imitation trainer
# ---------------------------------------------------------------------------

class ImitationTrainer:
    """Supervised behavioral cloning trainer for MLPWeights."""

    def __init__(
        self,
        weights: MLPWeights,
        lr: float = 1e-3,
        lr_decay: float = 0.95,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        self.weights = weights
        self.lr = lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.batch_size = batch_size

    def train(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True,
    ) -> list[float]:
        """Train weights on demonstration data.

        Args:
            states: (N, 39) sensory vectors
            targets: (N, 11) action targets

        Returns:
            List of per-epoch average losses.
        """
        n = len(states)
        epoch_losses: list[float] = []
        lr = self.lr
        rng = np.random.default_rng(0)

        for epoch in range(self.epochs):
            # Shuffle
            perm = rng.permutation(n)
            states_shuffled = states[perm]
            targets_shuffled = targets[perm]

            batch_losses = []
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_s = states_shuffled[start:end]
                batch_t = targets_shuffled[start:end]

                # Forward
                z_out, cache = forward(self.weights, batch_s)
                if z_out.ndim == 1:
                    z_out = z_out[np.newaxis, :]

                # Compute loss for logging
                loss = self._compute_loss(z_out, batch_t)
                batch_losses.append(loss)

                # Gradient
                dL_dz = compute_imitation_logit_grad(z_out, batch_t)
                grads = _backprop_logits_gradient(self.weights, cache, dL_dz)

                # SGD update
                for param, grad in zip(self.weights.params(), grads):
                    param -= lr * grad

            avg_loss = float(np.mean(batch_losses))
            epoch_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [imitation] Epoch {epoch + 1:>3}/{self.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}")

            lr *= self.lr_decay

        return epoch_losses

    @staticmethod
    def _compute_loss(z_out: np.ndarray, targets: np.ndarray) -> float:
        """Compute mixed loss for monitoring."""
        scale = math.pi / 6.0
        batch = z_out.shape[0]

        # Turn MSE
        turn_pred = np.tanh(z_out[:, 0]) * scale
        turn_target = np.tanh(targets[:, 0]) * scale
        loss_turn = np.mean((turn_pred - turn_target) ** 2)

        # Speed MSE
        speed_pred = _sigmoid(z_out[:, 1:2]).squeeze(-1)
        speed_target = _sigmoid(targets[:, 1:2]).squeeze(-1)
        loss_speed = np.mean((speed_pred - speed_target) ** 2)

        # Deposit cross-entropy
        probs = _softmax(z_out[:, 2:7], axis=-1)
        loss_deposit = -np.mean(np.sum(targets[:, 2:7] * np.log(probs + 1e-8), axis=-1))

        # Strength MSE
        str_pred = _sigmoid(z_out[:, 7:8]).squeeze(-1)
        str_target = _sigmoid(targets[:, 7:8]).squeeze(-1)
        loss_str = np.mean((str_pred - str_target) ** 2)

        # Binary CE
        loss_binary = 0.0
        for j in range(3):
            sig = _sigmoid(z_out[:, 8 + j:9 + j]).squeeze(-1)
            t = targets[:, 8 + j]
            loss_binary += -np.mean(t * np.log(sig + 1e-8) + (1 - t) * np.log(1 - sig + 1e-8))

        return float(loss_turn + loss_speed + loss_deposit + loss_str + loss_binary)


# ---------------------------------------------------------------------------
# Transformer imitation trainer
# ---------------------------------------------------------------------------

class TransformerImitationTrainer:
    """Behavioral cloning for TransformerWeightSet using zeroth-order updates.

    Since we don't have analytical backprop through the transformer, we use
    finite-difference parameter perturbation guided by the imitation loss.
    For single-step context (seq_len=1), this works reasonably well.
    """

    def __init__(
        self,
        weight_set: Any,  # TransformerWeightSet
        lr: float = 1e-3,
        lr_decay: float = 0.95,
        epochs: int = 50,
        batch_size: int = 64,
        perturbation_scale: float = 0.01,
    ):
        self.weight_set = weight_set
        self.lr = lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.perturbation_scale = perturbation_scale

    def train(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True,
    ) -> list[float]:
        """Train via zeroth-order optimization on single-step contexts."""
        n = len(states)
        epoch_losses: list[float] = []
        lr = self.lr
        rng = np.random.default_rng(0)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            states_shuffled = states[perm]
            targets_shuffled = targets[perm]

            batch_losses = []
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_s = states_shuffled[start:end]
                batch_t = targets_shuffled[start:end]

                # Forward with seq_len=1
                ctx = batch_s[:, np.newaxis, :]  # (batch, 1, 39)
                logits, _ = self.weight_set.forward(ctx)
                if logits.ndim == 1:
                    logits = logits[np.newaxis, :]

                loss = ImitationTrainer._compute_loss(logits, batch_t)
                batch_losses.append(loss)

                # Zeroth-order update: perturb each parameter, measure loss change
                for param in self.weight_set.all_parameters():
                    perturbation = rng.standard_normal(param.shape) * self.perturbation_scale
                    param += perturbation

                    logits_p, _ = self.weight_set.forward(ctx)
                    if logits_p.ndim == 1:
                        logits_p = logits_p[np.newaxis, :]
                    loss_p = ImitationTrainer._compute_loss(logits_p, batch_t)

                    # Approximate gradient direction
                    param -= perturbation  # restore
                    grad_estimate = (loss_p - loss) / self.perturbation_scale
                    param -= lr * grad_estimate * perturbation / self.perturbation_scale

            avg_loss = float(np.mean(batch_losses))
            epoch_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [imitation-tf] Epoch {epoch + 1:>3}/{self.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}")

            lr *= self.lr_decay

        return epoch_losses
