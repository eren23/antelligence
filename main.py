"""Entry point — integrates all systems into the core simulation loop.

Per-tick sequence:
  1. Build sensory inputs for all ants
  2. Compute rewards & train (from previous tick's transition)
  3. Run brain.decide() for each ant
  4. Apply actions: movement, pheromone deposit, pickup/drop
  5. Update pheromone engine: deposit, diffuse, evaporate
  6. Update world: food respawn
  7. Update colony: spawning, death, role rebalance
  8. Assign brains to newly spawned ants
  9. Record metrics & emergence

Features:
  - Brain hot-swap (R/N/T keys) preserving ant positions and state
  - Simulation speed control (0.5x – 10x)
  - Deterministic RNG seeding for reproducibility
  - State save/load for replay support
"""

from __future__ import annotations

import argparse
import json
import math
import pickle  # noqa: S403 — used for internal state serialisation only
import random
import sys
import time as _time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agents.actions import AntAction
from agents.ant import (
    Ant,
    Role,
    Vec2,
    advance_position,
    check_death,
    deplete_energy,
    handle_obstacle_collision,
    handle_world_bounds,
    refill_at_nest,
    try_drop,
    try_pickup,
    update_heading,
)
from agents.colony import Colony, Corpse
from agents.sensory import build_sensory
from world.spatial import SpatialGrid
from brains.action_utils import compute_reward
from brains.reward import SparseReward
from brains.rule_based import RuleBasedBrain

# PyTorch backends (optional)
try:
    from brains.torch_nn_brain import (
        TorchNNBrain,
        TorchSharedWeightRegistry as TorchNNWeightRegistry,
        TorchRoleTrainer as TorchNNRoleTrainer,
    )
    from brains.torch_transformer_brain import (
        TorchTransformerBrain,
        TorchTransformerSharedWeightRegistry as TorchTFWeightRegistry,
        TorchTransformerRoleTrainer as TorchTFRoleTrainer,
    )
    from brains.action_dist import ActionDistribution, action_tensor_to_ant_actions
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
from config import SimConfig, default_config, load_config
from metrics.emergence import EmergenceDetector
from metrics.reporter import write_report
from metrics.tracker import MetricsTracker
from rendering.controls import ControlState
from rendering.renderer import Renderer
from world.pheromone import Channel, PheromoneGrid
from world.world import World

# ---------------------------------------------------------------------------
# Channel name → enum mapping
# ---------------------------------------------------------------------------

_CHANNEL_MAP: dict[str, Channel] = {
    "food": Channel.FOOD,
    "home": Channel.HOME,
    "danger": Channel.DANGER,
    "recruit": Channel.RECRUIT,
}


# ---------------------------------------------------------------------------
# Weight persistence helpers
# ---------------------------------------------------------------------------

def _save_trainer_state(path: Path, trainer: object) -> None:
    """Save trainer baseline & global_step as JSON sidecar."""
    state = {
        "baseline": getattr(trainer, "baseline", 0.0),
        "global_step": getattr(trainer, "global_step", 0),
        "lr": getattr(trainer, "lr", 0.0),
        "gamma": getattr(trainer, "gamma", 0.0),
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _load_trainer_state(path: Path, trainer: object) -> None:
    """Restore trainer baseline & global_step from JSON sidecar."""
    if not path.exists():
        return
    with open(path) as f:
        state = json.load(f)
    if "baseline" in state:
        trainer.baseline = state["baseline"]
    if "global_step" in state:
        trainer.global_step = state["global_step"]


# ---------------------------------------------------------------------------
# Brain factory — creates and manages brain instances per backend type
# ---------------------------------------------------------------------------

class BrainManager:
    """Creates and hot-swaps brain backends for all ants.

    Shared weight registries and trainers for NN / transformer are lazily
    initialised on first use and reused across hot-swaps so that learned
    weights survive a round-trip (e.g. rule_based → nn → rule_based → nn).
    """

    def __init__(self, cfg: SimConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self._current_type: str = cfg.brain.default

        # Shared registries (lazy)
        self._torch_nn_registry = None
        self._torch_nn_trainers: dict = {}
        self._torch_tf_registry = None
        self._torch_tf_trainers: dict = {}

    @property
    def current_type(self) -> str:
        return self._current_type

    # -- weight persistence --------------------------------------------------

    def save_weights(self, path: Path) -> None:
        """Save all initialized brain weights to a directory."""
        path.mkdir(parents=True, exist_ok=True)

        # PyTorch NN
        if self._torch_nn_registry is not None and _TORCH_AVAILABLE:
            import torch as _torch
            for role in self._torch_nn_registry.roles():
                model = self._torch_nn_registry.get(role)
                sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
                np.savez(path / f"torch_nn_{role}.npz", **sd)
                trainer = self._torch_nn_trainers.get(role)
                if trainer:
                    _save_trainer_state(path / f"trainer_torch_nn_{role}.json", trainer)

        # PyTorch Transformer
        if self._torch_tf_registry is not None and _TORCH_AVAILABLE:
            import torch as _torch
            for role in self._torch_tf_registry.roles():
                model = self._torch_tf_registry.get(role)
                sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
                np.savez(path / f"torch_transformer_{role}.npz", **sd)
                trainer = self._torch_tf_trainers.get(role)
                if trainer:
                    _save_trainer_state(path / f"trainer_torch_transformer_{role}.json", trainer)

    def load_weights(self, path: Path) -> None:
        """Load weights from directory into registries."""
        if not path.is_dir():
            print(f"Warning: weights directory '{path}' not found, skipping load.")
            return

        loaded_any = False

        # PyTorch NN
        if _TORCH_AVAILABLE:
            import torch as _torch
            for npz in sorted(path.glob("torch_nn_*.npz")):
                role = npz.stem.removeprefix("torch_nn_")
                self._ensure_torch_nn()
                model = self._torch_nn_registry.get(role)
                d = np.load(npz)
                sd = {k: _torch.tensor(d[k]) for k in d.files}
                model.load_state_dict(sd)
                model.to(self._torch_nn_registry._device)
                trainer = self._torch_nn_trainers.get(role)
                if trainer:
                    _load_trainer_state(path / f"trainer_torch_nn_{role}.json", trainer)
                loaded_any = True

        # PyTorch Transformer
        if _TORCH_AVAILABLE:
            import torch as _torch
            for npz in sorted(path.glob("torch_transformer_*.npz")):
                role = npz.stem.removeprefix("torch_transformer_")
                self._ensure_torch_tf()
                model = self._torch_tf_registry.get(role)
                d = np.load(npz)
                sd = {k: _torch.tensor(d[k]) for k in d.files}
                model.load_state_dict(sd)
                model.to(self._torch_tf_registry._device)
                trainer = self._torch_tf_trainers.get(role)
                if trainer:
                    _load_trainer_state(path / f"trainer_torch_transformer_{role}.json", trainer)
                loaded_any = True

        if loaded_any:
            print(f"Weights loaded from '{path}'.")
        else:
            print(f"No weight files found in '{path}'.")

    # -- lazy init helpers ---------------------------------------------------

    def _ensure_torch_nn(self) -> None:
        if self._torch_nn_registry is not None or not _TORCH_AVAILABLE:
            return
        nn_cfg = self.cfg.brain.nn
        self._torch_nn_registry = TorchNNWeightRegistry(
            input_dim=39, hidden_sizes=nn_cfg.hidden_sizes, seed=self.seed,
        )
        for role in self._torch_nn_registry.roles():
            self._torch_nn_trainers[role] = TorchNNRoleTrainer(
                self._torch_nn_registry.get(role),
                lr=nn_cfg.learning_rate, gamma=nn_cfg.gamma,
                buffer_size=nn_cfg.buffer_size,
            )

    def _ensure_torch_tf(self) -> None:
        if self._torch_tf_registry is not None or not _TORCH_AVAILABLE:
            return
        tf_cfg = self.cfg.brain.transformer
        self._torch_tf_registry = TorchTFWeightRegistry(
            input_dim=39,
            d_model=tf_cfg.d_model, n_heads=tf_cfg.n_heads,
            n_layers=tf_cfg.n_layers, ffn_dim=tf_cfg.ffn_dim,
            seed=self.seed,
        )
        for role in self._torch_tf_registry.roles():
            self._torch_tf_trainers[role] = TorchTFRoleTrainer(
                self._torch_tf_registry.get(role),
                lr=tf_cfg.learning_rate, gamma=0.95,
                buffer_size=tf_cfg.buffer_size,
                context_length=tf_cfg.context_length,
            )

    # -- brain creation ------------------------------------------------------

    def create_brain(self, ant: Ant, brain_type: str | None = None) -> None:
        """Assign a fresh brain to *ant*."""
        bt = brain_type or self._current_type
        role = ant.role.value
        ant_seed = ant.id + self.seed

        if bt == "rule_based":
            ant.brain = RuleBasedBrain(
                world_width=self.cfg.world.width,
                world_height=self.cfg.world.height,
                rng_seed=ant_seed,
            )
        elif bt == "torch_nn" and _TORCH_AVAILABLE:
            self._ensure_torch_nn()
            ant.brain = TorchNNBrain(
                role=role,
                registry=self._torch_nn_registry,
                trainer=self._torch_nn_trainers[role],
                cfg=self.cfg.brain.nn,
                seed=ant_seed,
            )
        elif bt == "torch_transformer" and _TORCH_AVAILABLE:
            self._ensure_torch_tf()
            ant.brain = TorchTransformerBrain(
                role=role,
                registry=self._torch_tf_registry,
                trainer=self._torch_tf_trainers[role],
                cfg=self.cfg.brain.transformer,
                seed=ant_seed,
            )

    def hot_swap(self, ants: list[Ant], new_type: str) -> None:
        """Switch every ant's brain while preserving positions and state."""
        if new_type == self._current_type:
            return
        self._current_type = new_type
        for ant in ants:
            if ant.alive:
                self.create_brain(ant, new_type)


# ---------------------------------------------------------------------------
# Save / Load — pickle-based state serialisation
#
# Only used for internal replay support; files are produced and consumed
# by the same application.  Never load untrusted pickle files.
# ---------------------------------------------------------------------------

def _save_brain_state(ant: Ant) -> dict | None:
    """Extract serialisable internal state from an ant's brain (if any)."""
    brain = ant.brain
    if brain is None:
        return None
    if isinstance(brain, RuleBasedBrain):
        return {
            "type": "rule_based",
            "rng_state": brain._rng.getstate(),
            "forager_state": brain._forager_state.value,
            "soldier_state": brain._soldier_state.value,
            "nurse_state": brain._nurse_state.value,
            "ticks": brain._ticks,
            "patrol_angle": brain._patrol_angle,
            "wander_bias": brain._wander_bias,
        }
    return None


def _restore_brain_state(ant: Ant, state: dict | None) -> None:
    """Restore brain internal state saved by ``_save_brain_state``."""
    if state is None or ant.brain is None:
        return
    from brains.rule_based import ForagerState, SoldierState, NurseState

    brain = ant.brain
    if state.get("type") == "rule_based" and isinstance(brain, RuleBasedBrain):
        brain._rng.setstate(state["rng_state"])
        brain._forager_state = ForagerState(state["forager_state"])
        brain._soldier_state = SoldierState(state["soldier_state"])
        brain._nurse_state = NurseState(state["nurse_state"])
        brain._ticks = state["ticks"]
        brain._patrol_angle = state["patrol_angle"]
        brain._wander_bias = state["wander_bias"]


def save_state(
    path: Path,
    tick: int,
    seed: int,
    brain_type: str,
    colony: Colony,
    world: World,
    pheromone_grid: PheromoneGrid,
    control: ControlState,
) -> None:
    """Persist the full simulation state to *path*.

    Includes RNG states for deterministic replay support.
    Only used for internal state serialisation — never load untrusted files.
    """
    ant_records = [
        {
            "id": a.id, "px": a.pos.x, "py": a.pos.y,
            "heading": a.heading, "speed": a.speed, "energy": a.energy,
            "carrying": a.carrying, "carry_amount": a.carry_amount,
            "role": a.role.value, "age": a.age, "alive": a.alive,
            "brain_state": _save_brain_state(a),
        }
        for a in colony.ants
    ]
    corpse_records = [
        {"px": c.pos.x, "py": c.pos.y, "age": c.age}
        for c in colony.corpses
    ]
    food_records = [
        {
            "px": f.pos.x, "py": f.pos.y,
            "radius": f.radius, "amount": f.amount, "max_amount": f.max_amount,
        }
        for f in world.food_sources
    ]

    blob = {
        "tick": tick,
        "seed": seed,
        "brain_type": brain_type,
        "ants": ant_records,
        "corpses": corpse_records,
        "food_stored": colony.food_stored,
        "next_id": colony._next_id,
        "tick_count": colony.tick_count,
        "dead_count": colony._dead_count,
        "food_sources": food_records,
        "pheromone_grid": pheromone_grid.grid.copy(),
        "speed_index": control.speed_index,
        "paused": control.paused,
        # RNG states for deterministic replay
        "colony_rng_state": colony._rng.getstate(),
        "world_rng_state": world._rng.getstate(),
        "global_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301


def load_state(
    path: Path,
    cfg: SimConfig,
    brain_mgr: BrainManager,
) -> tuple[int, Colony, World, PheromoneGrid, ControlState]:
    """Restore simulation state from *path*.

    Returns ``(tick, colony, world, pheromone_grid, control)``.

    Restores RNG states when available for deterministic replay.

    .. warning:: Only load files you created yourself — pickle can execute
       arbitrary code on untrusted input.
    """
    from world.food import FoodSource

    with open(path, "rb") as f:
        blob: dict = pickle.load(f)  # noqa: S301

    seed: int = blob["seed"]

    # Rebuild world (deterministic from seed for terrain / obstacles)
    world = World.from_config(cfg, seed=seed)
    world.food_sources.clear()
    for fd in blob["food_sources"]:
        world.food_sources.append(FoodSource(
            pos=Vec2(fd["px"], fd["py"]),
            radius=fd["radius"],
            amount=fd["amount"],
            max_amount=fd["max_amount"],
        ))

    # Restore world RNG state if available
    if "world_rng_state" in blob:
        world._rng.setstate(blob["world_rng_state"])

    # Pheromone grid
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    pheromone_grid.grid[:] = blob["pheromone_grid"]

    # Colony — bypass __init__ to avoid initial spawning
    colony = Colony.__new__(Colony)
    colony._cfg = cfg
    colony._rng = random.Random(seed)
    colony._ants = []
    colony._corpses = []
    colony._food_stored = blob["food_stored"]
    colony._next_id = blob["next_id"]
    colony._tick_count = blob["tick_count"]
    colony._dead_count = blob["dead_count"]

    # Restore colony RNG state if available
    if "colony_rng_state" in blob:
        colony._rng.setstate(blob["colony_rng_state"])

    brain_type: str = blob["brain_type"]
    brain_mgr._current_type = brain_type

    for ad in blob["ants"]:
        ant = Ant(
            id=ad["id"],
            pos=Vec2(ad["px"], ad["py"]),
            heading=ad["heading"],
            speed=ad["speed"],
            energy=ad["energy"],
            carrying=ad["carrying"],
            carry_amount=ad["carry_amount"],
            role=Role(ad["role"]),
            age=ad["age"],
            alive=ad["alive"],
        )
        if ant.alive:
            brain_mgr.create_brain(ant, brain_type)
            _restore_brain_state(ant, ad.get("brain_state"))
        colony._ants.append(ant)

    for cd in blob["corpses"]:
        colony._corpses.append(Corpse(pos=Vec2(cd["px"], cd["py"]), age=cd["age"]))

    # Restore global RNG states if available
    if "global_rng_state" in blob:
        random.setstate(blob["global_rng_state"])
    if "np_rng_state" in blob:
        np.random.set_state(blob["np_rng_state"])

    control = ControlState(
        brain_type=brain_type,
        speed_index=blob.get("speed_index", 1),
        paused=blob.get("paused", False),
    )

    return blob["tick"], colony, world, pheromone_grid, control


# ---------------------------------------------------------------------------
# Simulation tick
# ---------------------------------------------------------------------------

def sim_tick(
    tick: int,
    colony: Colony,
    world: World,
    pheromone_grid: PheromoneGrid,
    brain_mgr: BrainManager,
    metrics: MetricsTracker,
    emergence: EmergenceDetector,
    cfg: SimConfig,
    *,
    learn: bool = True,
    torch_batch_policy: str = "auto",
    perf_stats: dict[str, float] | None = None,
) -> None:
    """Execute one simulation tick (the full per-tick pipeline)."""
    t_tick_start = _time.perf_counter() if perf_stats is not None else 0.0
    ants = colony.ants

    # Build spatial grid for O(n) neighbor lookups
    spatial_grid = SpatialGrid(cfg.world.width, cfg.world.height, cell_size=40)
    spatial_grid.rebuild(ants)

    # 1. Build sensory inputs for all alive ants
    for ant in ants:
        if ant.alive:
            ant.sensory = build_sensory(ant, world, pheromone_grid, ants, cfg.ant,
                                        spatial_grid=spatial_grid)

    # 2. Reward & learn from the *previous* tick's transition
    #    (must happen before decide() overwrites brain internals)
    sparse_reward = SparseReward()
    if learn:
        for ant in ants:
            if not ant.alive or ant.brain is None or ant.sensory is None:
                continue
            prev_act: AntAction | None = getattr(ant.brain, "_prev_action", None)
            if prev_act is None:
                continue
            prev_si = getattr(ant.brain, "_prev_sensory", None)
            is_learning_brain = not isinstance(ant.brain, RuleBasedBrain)
            if is_learning_brain:
                reward = sparse_reward.compute(prev_si, ant.sensory, prev_act, True)
            else:
                reward = compute_reward(prev_si, ant.sensory, prev_act, True)
            ant.brain.learn(reward)
            metrics.accumulate_reward(reward)

    # 3. Brain decides
    t_policy_start = _time.perf_counter() if perf_stats is not None else 0.0
    actions: dict[int, AntAction] = {}

    # Phase 1 GPU path: batch torch policy inference by role
    use_batched_torch = (
        _TORCH_AVAILABLE
        and torch_batch_policy in ("auto", "on")
    )
    if use_batched_torch:
        try:
            import torch as _torch

            nn_groups: dict[str, list[Ant]] = {}
            tf_groups: dict[str, list[Ant]] = {}
            for ant in ants:
                if not ant.alive or ant.brain is None or ant.sensory is None:
                    continue
                if isinstance(ant.brain, TorchNNBrain):
                    nn_groups.setdefault(ant.role.value, []).append(ant)
                elif isinstance(ant.brain, TorchTransformerBrain):
                    tf_groups.setdefault(ant.role.value, []).append(ant)

            # Batched torch NN forward pass per role
            for role, group in nn_groups.items():
                if not group:
                    continue
                model = group[0].brain.model
                device = next(model.parameters()).device
                vecs64 = [np.array(ant.sensory.to_vector(), dtype=np.float64) for ant in group]
                batch = np.stack([v.astype(np.float32, copy=False) for v in vecs64], axis=0)
                with _torch.no_grad():
                    x = _torch.tensor(batch, dtype=_torch.float32, device=device)
                    logits, _ = model(x)
                    dist = ActionDistribution(logits)
                    action_tensors = dist.sample()
                    log_probs = dist.log_prob(action_tensors)
                    action_np = action_tensors.cpu().numpy()
                    lp_np = log_probs.cpu().numpy()
                ant_actions = action_tensor_to_ant_actions(action_np)
                for ant, vec64, action, lp in zip(group, vecs64, ant_actions, lp_np):
                    ant.brain._prev_sensory = ant.sensory
                    ant.brain._prev_action = action
                    ant.brain._prev_log_prob = float(lp)
                    ant.brain._prev_vec = vec64
                    ant.brain._step_count += 1
                    actions[ant.id] = action

            # Batched torch transformer forward pass by role + seq_len
            for role, group in tf_groups.items():
                if not group:
                    continue
                model = group[0].brain.model
                device = next(model.parameters()).device
                seq_buckets: dict[int, list[tuple[Ant, np.ndarray, np.ndarray]]] = {}
                for ant in group:
                    vec64 = np.array(ant.sensory.to_vector(), dtype=np.float64)
                    ant.brain.context.append(vec64)
                    ctx = np.stack(list(ant.brain.context), axis=0)
                    seq_buckets.setdefault(ctx.shape[0], []).append((ant, vec64, ctx))

                for seq_len, items in seq_buckets.items():
                    x_np = np.stack(
                        [ctx.astype(np.float32, copy=False) for _, _, ctx in items],
                        axis=0,
                    )
                    with _torch.no_grad():
                        x = _torch.tensor(x_np, dtype=_torch.float32, device=device)
                        logits, _ = model(x)
                        dist = ActionDistribution(logits)
                        action_tensors = dist.sample()
                        log_probs = dist.log_prob(action_tensors)
                        action_np = action_tensors.cpu().numpy()
                        lp_np = log_probs.cpu().numpy()
                    ant_actions = action_tensor_to_ant_actions(action_np)
                    for (ant, _vec64, ctx), action, lp in zip(items, ant_actions, lp_np):
                        ant.brain._prev_sensory = ant.sensory
                        ant.brain._prev_action = action
                        ant.brain._prev_log_prob = float(lp)
                        ant.brain._prev_context_flat = ctx.flatten()
                        ant.brain._prev_seq_len = ctx.shape[0]
                        ant.brain._step_count += 1
                        actions[ant.id] = action
        except Exception:
            # Safety fallback: keep simulation running if batching path fails.
            if torch_batch_policy == "on":
                raise

    for ant in ants:
        if ant.id in actions:
            continue
        if ant.alive and ant.brain is not None and ant.sensory is not None:
            actions[ant.id] = ant.brain.decide(ant.sensory)
        else:
            actions[ant.id] = AntAction()
    if perf_stats is not None:
        perf_stats["policy_s"] = perf_stats.get("policy_s", 0.0) + (_time.perf_counter() - t_policy_start)

    # 4. Apply actions — physics, movement, interactions
    for ant in ants:
        if not ant.alive:
            continue
        action = actions[ant.id]

        # Survival instinct: when energy is critically low, head toward nest
        # Gated: skip for learning brains unless learning_brain_patches is True
        is_learning_brain = not isinstance(ant.brain, RuleBasedBrain)
        if (cfg.brain.patches.survival_homing
                and ant.energy < cfg.ant.energy_max * 0.3
                and (cfg.brain.patches.learning_brain_patches or not is_learning_brain)):
            to_nest = world.nest.center - ant.pos
            nest_angle = math.atan2(to_nest.y, to_nest.x)
            turn_to_nest = nest_angle - ant.heading
            turn_to_nest = (turn_to_nest + math.pi) % (2 * math.pi) - math.pi
            action = AntAction(
                turn=turn_to_nest * 0.5,  # strong bias toward nest
                speed_mult=1.0,
                deposit_pheromone=action.deposit_pheromone,
                deposit_strength=action.deposit_strength,
                pickup=action.pickup,
                drop=action.drop,
                recruit_signal=action.recruit_signal,
            )

        # Movement
        update_heading(ant, action)
        advance_position(ant, action, world)
        handle_obstacle_collision(ant, world)
        handle_world_bounds(ant, world)

        # Energy
        deplete_energy(ant, action, cfg.ant)
        died = check_death(ant)
        if died:
            # Immediate death penalty so learning gets the signal
            # (the ant will be removed from colony.ants by colony.tick)
            if learn and ant.brain is not None:
                death_penalty = -1.0 if is_learning_brain else -0.5
                ant.brain.learn(death_penalty)
                metrics.accumulate_reward(death_penalty)
            continue

        # Nest: refill energy & auto-deposit food
        carry_before = ant.carry_amount
        deposited = refill_at_nest(ant, world, cfg.ant)
        if deposited == "food":
            colony.deposit_food(carry_before)

        # Item interactions — auto-pickup if enabled, otherwise respect brain decision
        # Gated: skip auto-pickup for learning brains unless learning_brain_patches is True
        was_carrying = ant.carrying
        if action.pickup or (cfg.brain.patches.auto_pickup
                and (cfg.brain.patches.learning_brain_patches or not is_learning_brain)
                and ant.carrying is None):
            try_pickup(ant, world, cfg.ant)
            # Nurses also try to pick up nearby corpses
            if ant.carrying is None and ant.role == Role.NURSE:
                for corpse in list(colony.corpses):
                    if ant.pos.distance_to(corpse.pos) <= cfg.ant.neighbor_radius:
                        ant.carrying = "corpse"
                        ant.carry_amount = 0.0
                        colony.remove_corpse(corpse)
                        break
        # Food drop guard: prevent dropping food (it's auto-deposited at nest).
        if action.drop and ant.carrying is not None and (
            not cfg.brain.patches.food_drop_guard or ant.carrying != "food"
        ):
            dropped = try_drop(ant)
            if dropped is not None and dropped[0] == "corpse":
                colony.corpses.append(Corpse(pos=Vec2(ant.pos.x, ant.pos.y)))

    # 5. Pheromone deposition from actions
    for ant in ants:
        if not ant.alive:
            continue
        action = actions[ant.id]
        if action.deposit_pheromone and action.deposit_strength > 0:
            ch = _CHANNEL_MAP.get(action.deposit_pheromone)
            if ch is not None:
                pheromone_grid.deposit(
                    ant.pos.x, ant.pos.y, ch, action.deposit_strength,
                )
        # Recruit signal → recruit pheromone pulse
        if action.recruit_signal:
            pheromone_grid.deposit(ant.pos.x, ant.pos.y, Channel.RECRUIT, 0.6)

    # 6. Pheromone engine tick (diffuse + evaporate)
    pheromone_grid.tick()

    # 7. World tick (food respawn)
    world.tick()

    # 8. Colony tick (remove dead → corpses, spawn new ants, role rebalance)
    colony.tick(world.nest.center)

    # 9. Assign brains to newly spawned ants (they arrive with brain=None)
    for ant in colony.ants:
        if ant.brain is None and ant.alive:
            brain_mgr.create_brain(ant)

    # 10. Record metrics
    snapshot = metrics.record(tick, colony, world, pheromone_grid)

    # 11. Update emergence detectors
    emergence.update(tick, colony, world, pheromone_grid, snapshot.food_income)
    if perf_stats is not None:
        perf_stats["tick_s"] = perf_stats.get("tick_s", 0.0) + (_time.perf_counter() - t_tick_start)


# ---------------------------------------------------------------------------
# Headless batch runner — importable by compare_brains.py
# ---------------------------------------------------------------------------

def run_headless(
    *,
    brain: str = "rule_based",
    ticks: int = 5000,
    seed: int = 42,
    ants: int | None = None,
    config_path: str = "colony_config.yaml",
    verbose: bool = True,
    torch_batch_policy: str = "auto",
) -> dict[str, Any]:
    """Run a headless simulation and return a JSON-serializable report dict.

    This is the primary programmatic entry point for batch experiments.
    """
    from metrics.emergence import count_cemetery_clusters

    # ---- Deterministic seeding ----
    random.seed(seed)
    np.random.seed(seed)

    # ---- Configuration ----
    cfg_path = Path(config_path)
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config()
    cfg.brain.default = brain
    if ants is not None:
        cfg.colony.initial_population = ants

    # ---- Initialise state ----
    brain_mgr = BrainManager(cfg, seed)
    world = World.from_config(cfg, seed=seed)
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    colony = Colony(cfg, world.nest.center, seed=seed)

    for ant in colony.ants:
        brain_mgr.create_brain(ant)

    metrics = MetricsTracker(window=0)  # unlimited — keep all snapshots
    emergence = EmergenceDetector(cfg.roles.default_distribution)

    # ---- Run simulation ----
    t_start = _time.perf_counter()
    perf_stats: dict[str, float] = {"tick_s": 0.0, "policy_s": 0.0}

    for tick in range(1, ticks + 1):
        sim_tick(
            tick,
            colony,
            world,
            pheromone_grid,
            brain_mgr,
            metrics,
            emergence,
            cfg,
            torch_batch_policy=torch_batch_policy,
            perf_stats=perf_stats,
        )

        if verbose and tick % 1000 == 0:
            stats = colony.stats()
            elapsed = _time.perf_counter() - t_start
            avg_tick_ms = (perf_stats["tick_s"] / tick) * 1000.0 if tick > 0 else 0.0
            avg_policy_ms = (perf_stats["policy_s"] / tick) * 1000.0 if tick > 0 else 0.0
            tps = tick / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{brain}] Tick {tick:>8,}  |  Pop {stats.population:>4}"
                f"  |  Food {stats.food_stored:>8.1f}"
                f"  |  Deaths {stats.dead_count:>5}"
                f"  |  {tps:>6.1f} t/s"
                f"  |  sim {avg_tick_ms:>6.2f} ms"
                f"  |  policy {avg_policy_ms:>6.2f} ms"
            )

    t_elapsed = _time.perf_counter() - t_start

    # ---- Build report ----
    snapshots = metrics.snapshots
    stats = colony.stats()

    total_food = sum(s.food_income for s in snapshots)
    total_deaths = sum(s.deaths for s in snapshots)

    # Trail entropy time series (downsampled to ~500 points max)
    step = max(1, len(snapshots) // 500)
    trail_entropy_series = [round(s.trail_entropy, 4) for s in snapshots[::step]]

    # Final food income rate (rolling average over last 100 ticks)
    tail = snapshots[-100:] if len(snapshots) >= 100 else snapshots
    food_income_final = sum(s.food_income for s in tail) / len(tail) if tail else 0.0

    # Foraging trip average
    avg_trip = metrics._mean_trip_duration()

    # Cemetery clusters
    cluster_count = count_cemetery_clusters(colony.corpses)

    # Brain-specific metrics
    brain_metrics: dict[str, Any] = {"type": brain}
    if snapshots:
        brain_metrics["avg_reward"] = round(
            sum(s.avg_reward for s in snapshots) / len(snapshots), 6,
        )

    return {
        "config": {
            "brain": brain,
            "seed": seed,
            "ticks": ticks,
            "initial_population": cfg.colony.initial_population,
            "max_population": cfg.colony.max_population,
        },
        "food_collected_total": round(total_food, 2),
        "food_income_rate_final": round(food_income_final, 4),
        "population_final": stats.population,
        "deaths": total_deaths,
        "trail_entropy_over_time": trail_entropy_series,
        "cemetery_cluster_count": cluster_count,
        "avg_foraging_trip_ticks": round(avg_trip, 2),
        "role_distribution_final": stats.role_counts,
        "brain_specific": brain_metrics,
        "performance": {
            "wall_time_seconds": round(t_elapsed, 2),
            "ticks_per_second": round(ticks / t_elapsed, 1) if t_elapsed > 0 else 0,
        },
        "emergence": emergence.report.as_dict(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swarm Colony Simulation")
    p.add_argument("--config", "-c", default="colony_config.yaml",
                   help="YAML config path (default: colony_config.yaml)")
    p.add_argument("--seed", "-s", type=int, default=42,
                   help="RNG seed for deterministic runs (default: 42)")
    p.add_argument("--headless", action="store_true",
                   help="Run without opening a window")
    p.add_argument("--ticks", "-t", type=int, default=0,
                   help="Max ticks (0 = unlimited)")
    p.add_argument("--brain", choices=["rule_based", "torch_nn", "torch_transformer"],
                   default=None, help="Override initial brain backend")
    p.add_argument("--ants", type=int, default=None,
                   help="Override initial ant population count")
    p.add_argument("--load", type=str, default=None,
                   help="Resume from a saved .pkl state file")
    p.add_argument("--report", type=str, default=None,
                   help="Write JSON metrics report on exit")
    p.add_argument("--save-weights", type=str, default=None,
                   help="Save trained weights on exit to DIR (default: weights/)")
    p.add_argument("--load-weights", type=str, default=None,
                   help="Load pre-trained weights from DIR on startup")
    p.add_argument("--autosave-interval", type=int, default=0,
                   help="Auto-save weights every N ticks (0 = disabled)")
    p.add_argument(
        "--torch-batch-policy",
        choices=["auto", "on", "off"],
        default="auto",
        help="Torch policy batching mode for action inference (default: auto)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed: int = args.seed

    # ------------------------------------------------------------------
    # Fast path: headless + ticks → use run_headless for richer reports
    # ------------------------------------------------------------------
    if args.headless and args.ticks > 0 and not args.load and not args.load_weights:
        report = run_headless(
            brain=args.brain or "rule_based",
            ticks=args.ticks,
            seed=seed,
            ants=args.ants,
            config_path=args.config,
            verbose=True,
            torch_batch_policy=args.torch_batch_policy,
        )
        if args.report:
            rpath = Path(args.report)
            rpath.parent.mkdir(parents=True, exist_ok=True)
            with open(rpath, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report → {rpath}")
        return

    # ------------------------------------------------------------------
    # Interactive / legacy headless path
    # ------------------------------------------------------------------

    # ---- Deterministic seeding ----
    random.seed(seed)
    np.random.seed(seed)

    # ---- Configuration ----
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config()

    if args.brain:
        cfg.brain.default = args.brain
    if args.ants is not None:
        cfg.colony.initial_population = args.ants

    # ---- Brain manager ----
    brain_mgr = BrainManager(cfg, seed)

    # ---- Initialise or load state ----
    if args.load:
        load_path = Path(args.load)
        tick, colony, world, pheromone_grid, control = load_state(
            load_path, cfg, brain_mgr,
        )
        print(f"Resumed from '{load_path}' at tick {tick}.")
    else:
        tick = 0
        world = World.from_config(cfg, seed=seed)
        pheromone_grid = PheromoneGrid(
            cfg.world.width, cfg.world.height, cfg.pheromone,
        )
        colony = Colony(cfg, world.nest.center, seed=seed)
        control = ControlState(brain_type=cfg.brain.default)

        # Assign initial brains
        for ant in colony.ants:
            brain_mgr.create_brain(ant)

    # ---- Load pre-trained weights ----
    if args.load_weights:
        brain_mgr.load_weights(Path(args.load_weights))

    # ---- Metrics & emergence ----
    metrics = MetricsTracker(window=10_000)
    emergence = EmergenceDetector(cfg.roles.default_distribution)

    # ---- Renderer (unless headless) ----
    renderer: Renderer | None = None
    if not args.headless:
        renderer = Renderer(world, pheromone_grid, cfg)

    max_ticks: int = args.ticks
    tick_accumulator: float = 0.0
    autosave_interval: int = args.autosave_interval
    weights_dir = Path(args.save_weights) if args.save_weights else None

    # ===================================================================
    # Main loop
    # ===================================================================
    try:
        running = True
        while running:
            # -- Brain hot-swap (R / N / T keys update control.brain_type) --
            if control.brain_type != brain_mgr.current_type:
                brain_mgr.hot_swap(colony.ants, control.brain_type)

            # -- Simulation ticks (speed-aware accumulator) --
            if not control.paused:
                tick_accumulator += control.speed_multiplier
                while tick_accumulator >= 1.0:
                    tick += 1
                    sim_tick(
                        tick, colony, world, pheromone_grid,
                        brain_mgr, metrics, emergence, cfg,
                        torch_batch_policy=args.torch_batch_policy,
                    )
                    tick_accumulator -= 1.0

                    # Periodic weight autosave
                    if (autosave_interval > 0
                            and weights_dir is not None
                            and tick % autosave_interval == 0):
                        brain_mgr.save_weights(weights_dir)

                    if max_ticks > 0 and tick >= max_ticks:
                        running = False
                        break

            # -- Render (or headless progress) --
            if renderer is not None:
                snap = metrics.latest
                if snap is not None:
                    renderer.record_reward(snap.avg_reward)

                renderer.render(world, colony, pheromone_grid, control, tick)

                if control.quit_requested:
                    running = False
            else:
                # Headless: no frame cap — tick as fast as possible
                if tick > 0 and tick % 1000 == 0:
                    stats = colony.stats()
                    print(
                        f"Tick {tick:>8,}  |  Pop {stats.population:>4}"
                        f"  |  Food {stats.food_stored:>8.1f}"
                        f"  |  Deaths {stats.dead_count:>5}"
                    )

                if max_ticks <= 0 and tick == 0:
                    print("Headless mode requires --ticks or an initial state.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # ===================================================================
    # Cleanup
    # ===================================================================

    # Auto-save state on exit
    save_dir = Path("saves")
    state_file = save_dir / f"state_tick{tick}.pkl"
    save_state(
        state_file, tick, seed, brain_mgr.current_type,
        colony, world, pheromone_grid, control,
    )
    print(f"State saved → {state_file}")

    # Save weights on exit
    if weights_dir is not None:
        brain_mgr.save_weights(weights_dir)
        print(f"Weights saved → {weights_dir}")

    # Optional JSON report
    if args.report:
        rpath = write_report(
            args.report, metrics, emergence,
            include_time_series=True, downsample=10,
        )
        print(f"Report → {rpath}")

    if renderer is not None:
        renderer.quit()


if __name__ == "__main__":
    main()
