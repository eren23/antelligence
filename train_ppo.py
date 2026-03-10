#!/usr/bin/env python3
"""Fine-tune an imitation-trained brain with PPO.

Usage:
    python train_ppo.py --brain nn --load-imitation weights/imitation/ --ticks 50000
    python train_ppo.py --brain transformer --load-imitation weights/imitation/ --ticks 50000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.actions import AntAction
from agents.sensory import SensoryInput, build_sensory
from brains.nn_brain import (
    MLPWeights,
    NNBrain,
    SharedWeightRegistry as NNWeightRegistry,
    RoleTrainer as NNRoleTrainer,
    compute_reward,
    forward_with_value,
    decode_output,
    sample_action,
    log_prob_of_action,
)
from brains.reward import SparseReward
from brains.ppo import PPORoleTrainer, PPOStep, PPOTransformerRoleTrainer
from brains.transformer_brain import (
    TransformerBrain,
    SharedWeightRegistry as TFWeightRegistry,
    RoleTrainer as TFRoleTrainer,
)
from config import SimConfig, load_config, default_config, PPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO fine-tuning after imitation learning")
    p.add_argument("--brain", choices=["nn", "transformer", "torch_nn", "torch_transformer"], default="nn")
    p.add_argument("--load-imitation", type=str, default="weights/imitation",
                    help="Load imitation-trained weights from DIR")
    p.add_argument("--ticks", type=int, default=50000,
                    help="Total training ticks (default: 50000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="weights/ppo",
                    help="Output directory for PPO weights")
    p.add_argument("--config", type=str, default="colony_config.yaml")
    p.add_argument("--validate-ticks", type=int, default=5000,
                    help="Ticks for validation run")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout-length", type=int, default=256)
    p.add_argument("--save-interval", type=int, default=10000,
                    help="Save weights every N ticks")
    p.add_argument("--ants", type=int, default=200)
    return p.parse_args()


def compute_reward_ppo(
    prev_sensory: SensoryInput | None,
    curr_sensory: SensoryInput,
    action: AntAction,
    alive: bool,
) -> float:
    """PPO reward — delegates to SparseReward for genuine learning."""
    return SparseReward().compute(prev_sensory, curr_sensory, action, alive)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PPO Fine-Tuning: {args.brain} ===")
    print(f"  ticks: {args.ticks}, lr: {args.lr}, seed: {args.seed}")
    print(f"  loading imitation weights from: {args.load_imitation}")
    print()

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config()
    cfg.brain.default = args.brain
    cfg.colony.initial_population = args.ants

    from agents.colony import Colony
    from world.world import World
    from world.pheromone import PheromoneGrid
    from main import BrainManager, sim_tick
    from metrics.tracker import MetricsTracker
    from metrics.emergence import EmergenceDetector

    # Create brain manager and load imitation weights
    brain_mgr = BrainManager(cfg, args.seed)
    imitation_path = Path(args.load_imitation)
    if imitation_path.is_dir():
        brain_mgr.load_weights(imitation_path)
    else:
        print(f"Warning: imitation weights not found at {imitation_path}")

    # Create PPO trainers alongside the existing REINFORCE trainers
    ppo_trainers: dict[str, Any] = {}

    if args.brain == "nn":
        brain_mgr._ensure_nn()
        for role in brain_mgr._nn_registry.roles():
            weights = brain_mgr._nn_registry.get(role)
            ppo_trainers[role] = PPORoleTrainer(
                weights,
                lr=args.lr,
                gamma=cfg.brain.ppo.gamma,
                gae_lambda=cfg.brain.ppo.gae_lambda,
                clip_epsilon=cfg.brain.ppo.clip_epsilon,
                value_loss_coef=cfg.brain.ppo.value_loss_coef,
                entropy_coef=cfg.brain.ppo.entropy_coef,
                rollout_length=args.rollout_length,
                epochs_per_update=cfg.brain.ppo.epochs_per_update,
                batch_size=cfg.brain.ppo.batch_size,
                max_grad_norm=cfg.brain.ppo.max_grad_norm,
            )
    elif args.brain == "transformer":
        brain_mgr._ensure_tf()
        for role in brain_mgr._tf_registry.roles():
            ws = brain_mgr._tf_registry.get(role)
            ppo_trainers[role] = PPOTransformerRoleTrainer(
                ws,
                lr=args.lr,
                gamma=cfg.brain.ppo.gamma,
                gae_lambda=cfg.brain.ppo.gae_lambda,
                rollout_length=args.rollout_length,
            )
    elif args.brain.startswith("torch_"):
        from brains.torch_ppo import TorchPPOTrainer
        if args.brain == "torch_nn":
            brain_mgr._ensure_torch_nn()
            for role in brain_mgr._torch_nn_registry.roles():
                model = brain_mgr._torch_nn_registry.get(role)
                ppo_trainers[role] = TorchPPOTrainer.from_config(
                    model, cfg.brain.ppo, lr=args.lr,
                    rollout_length=args.rollout_length,
                )
        elif args.brain == "torch_transformer":
            brain_mgr._ensure_torch_tf()
            for role in brain_mgr._torch_tf_registry.roles():
                model = brain_mgr._torch_tf_registry.get(role)
                ppo_trainers[role] = TorchPPOTrainer.from_config(
                    model, cfg.brain.ppo, lr=args.lr,
                    rollout_length=args.rollout_length,
                )

    # Initialize simulation
    world = World.from_config(cfg, seed=args.seed)
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    colony = Colony(cfg, world.nest.center, seed=args.seed)

    for ant in colony.ants:
        brain_mgr.create_brain(ant)

    metrics = MetricsTracker(window=0)
    emergence = EmergenceDetector(cfg.roles.default_distribution)

    def _ppo_state_from_brain(ant_brain: Any) -> np.ndarray | None:
        """Extract PPO state in model-ready shape.

        - torch_nn / nn: returns flat sensory vector (39,)
        - torch_transformer: returns fixed-shape context (context_len, 39), zero-padded on the left
        """
        prev_vec = getattr(ant_brain, "_prev_vec", None)
        if prev_vec is not None:
            return np.asarray(prev_vec, dtype=np.float32)

        prev_ctx_flat = getattr(ant_brain, "_prev_context_flat", None)
        if prev_ctx_flat is None:
            return None

        # Transformer context path.
        if args.brain == "torch_transformer":
            flat = np.asarray(prev_ctx_flat, dtype=np.float32).reshape(-1)
            if flat.size % 39 != 0:
                return None

            seq_len = int(getattr(ant_brain, "_prev_seq_len", 0))
            if seq_len <= 0:
                seq_len = flat.size // 39
            seq_len = min(seq_len, flat.size // 39)
            if seq_len <= 0:
                return None

            ctx = flat[: seq_len * 39].reshape(seq_len, 39)
            target_len = int(cfg.brain.transformer.context_length)
            if target_len > seq_len:
                pad = np.zeros((target_len - seq_len, 39), dtype=np.float32)
                ctx = np.concatenate([pad, ctx], axis=0)
            elif target_len < seq_len:
                ctx = ctx[-target_len:]
            return ctx.astype(np.float32, copy=False)

        return np.asarray(prev_ctx_flat, dtype=np.float32)

    # Training loop with PPO
    t_start = time.perf_counter()
    best_food = 0.0
    food_window: list[float] = []  # rolling food tracking

    for tick in range(1, args.ticks + 1):
        # Build sensory
        for ant in colony.ants:
            if ant.alive:
                ant.sensory = build_sensory(ant, world, pheromone_grid, colony.ants, cfg.ant)

        # PPO reward & buffer collection
        for ant in colony.ants:
            if not ant.alive or ant.brain is None or ant.sensory is None:
                continue

            prev_act: AntAction | None = getattr(ant.brain, "_prev_action", None)
            if prev_act is None:
                continue

            prev_si = getattr(ant.brain, "_prev_sensory", None)
            reward = compute_reward_ppo(prev_si, ant.sensory, prev_act, True)
            metrics.accumulate_reward(reward)

            # Add to PPO buffer for this role
            role = ant.role.value
            if role in ppo_trainers:
                trainer = ppo_trainers[role]
                prev_vec = _ppo_state_from_brain(ant.brain)
                if prev_vec is not None:
                    prev_lp = getattr(ant.brain, "_prev_log_prob", 0.0)

                    # Get value estimate
                    if args.brain == "nn":
                        _, value, _ = forward_with_value(
                            brain_mgr._nn_registry.get(role), prev_vec
                        )
                        value = float(value) if not isinstance(value, float) else value
                    elif args.brain.startswith("torch_"):
                        import torch as _torch
                        with _torch.no_grad():
                            _model = ppo_trainers[role].model
                            _dev = next(_model.parameters()).device
                            _x = _torch.tensor(prev_vec, dtype=_torch.float32, device=_dev)
                            if args.brain == "torch_transformer":
                                # Transformer expects (B, seq, input_dim=39). We store flattened context.
                                if _x.dim() == 1:
                                    if (_x.numel() % 39) != 0:
                                        raise RuntimeError(
                                            f"torch_transformer prev_vec has invalid size {_x.numel()} (not divisible by 39)",
                                        )
                                    _x = _x.view(1, -1, 39)
                                elif _x.dim() == 2:
                                    # Could be (seq, 39) or (B, flat_context)
                                    if _x.shape[-1] == 39:
                                        _x = _x.unsqueeze(0)
                                    elif (_x.shape[-1] % 39) == 0:
                                        _x = _x.view(_x.shape[0], -1, 39)
                                    else:
                                        raise RuntimeError(
                                            f"torch_transformer prev_vec has invalid shape {tuple(_x.shape)}",
                                        )
                                else:
                                    raise RuntimeError(
                                        f"torch_transformer prev_vec has unsupported rank {_x.dim()}",
                                    )
                            else:
                                if _x.dim() == 1:
                                    _x = _x.unsqueeze(0)
                            _, _val = _model(_x)
                            value = float(_val.item())
                    else:
                        value = 0.0  # transformer doesn't have analytical value yet

                    step = PPOStep(
                        state=prev_vec,
                        action=prev_act,
                        log_prob=prev_lp,
                        value=value,
                        reward=reward,
                        done=False,
                    )
                    trainer.buffer.add(step)

                    # Check if buffer is ready for update
                    if trainer.buffer.ready:
                        update_metrics = trainer.update()
                        if update_metrics and tick % 5000 == 0:
                            print(f"    [PPO {role}] {update_metrics}")

        # Brain decides
        actions: dict[int, AntAction] = {}
        for ant in colony.ants:
            if ant.alive and ant.brain is not None and ant.sensory is not None:
                actions[ant.id] = ant.brain.decide(ant.sensory)
            else:
                actions[ant.id] = AntAction()

        # Apply actions (same as sim_tick but with PPO reward function)
        from agents.ant import (
            update_heading, advance_position, handle_obstacle_collision,
            handle_world_bounds, deplete_energy, check_death, refill_at_nest,
            try_pickup, try_drop, Role,
        )
        from agents.colony import Corpse
        from agents.ant import Vec2

        for ant in colony.ants:
            if not ant.alive:
                continue
            action = actions[ant.id]

            # Survival homing removed — PPO training should never override the policy
            # The policy must learn survival behavior on its own.

            update_heading(ant, action)
            advance_position(ant, action, world)
            handle_obstacle_collision(ant, world)
            handle_world_bounds(ant, world)

            deplete_energy(ant, action, cfg.ant)
            died = check_death(ant)
            if died:
                # Death penalty to PPO buffer
                role = ant.role.value
                if role in ppo_trainers:
                    prev_vec = _ppo_state_from_brain(ant.brain)
                    if prev_vec is not None:
                        ppo_trainers[role].buffer.add(PPOStep(
                            state=prev_vec,
                            action=action,
                            log_prob=getattr(ant.brain, "_prev_log_prob", 0.0),
                            value=0.0,
                            reward=-0.5,
                            done=True,
                        ))
                continue

            carry_before = ant.carry_amount
            deposited = refill_at_nest(ant, world, cfg.ant)
            if deposited == "food":
                colony.deposit_food(carry_before)

            was_carrying = ant.carrying
            if action.pickup or ant.carrying is None:
                try_pickup(ant, world, cfg.ant)
                if ant.carrying is None and ant.role == Role.NURSE:
                    for corpse in list(colony.corpses):
                        if ant.pos.distance_to(corpse.pos) <= cfg.ant.neighbor_radius:
                            ant.carrying = "corpse"
                            ant.carry_amount = 0.0
                            colony.remove_corpse(corpse)
                            break
            if action.drop and ant.carrying is not None and ant.carrying != "food":
                dropped = try_drop(ant)
                if dropped is not None and dropped[0] == "corpse":
                    colony.corpses.append(Corpse(pos=Vec2(ant.pos.x, ant.pos.y)))

        # Pheromone deposition
        from world.pheromone import Channel
        _CHANNEL_MAP = {
            "food": Channel.FOOD, "home": Channel.HOME,
            "danger": Channel.DANGER, "recruit": Channel.RECRUIT,
        }
        for ant in colony.ants:
            if not ant.alive:
                continue
            action = actions[ant.id]
            if action.deposit_pheromone and action.deposit_strength > 0:
                ch = _CHANNEL_MAP.get(action.deposit_pheromone)
                if ch is not None:
                    pheromone_grid.deposit(ant.pos.x, ant.pos.y, ch, action.deposit_strength)

        pheromone_grid.tick()
        world.tick()
        colony.tick(world.nest.center)

        for ant in colony.ants:
            if ant.brain is None and ant.alive:
                brain_mgr.create_brain(ant)

        snapshot = metrics.record(tick, colony, world, pheromone_grid)
        emergence.update(tick, colony, world, pheromone_grid, snapshot.food_income)

        # Progress reporting
        if tick % 5000 == 0:
            stats = colony.stats()
            print(
                f"  [{args.brain}] Tick {tick:>8,}  |  Pop {stats.population:>4}"
                f"  |  Food {stats.food_stored:>8.1f}"
                f"  |  Deaths {stats.dead_count:>5}"
            )

        # Periodic save
        if args.save_interval > 0 and tick % args.save_interval == 0:
            brain_mgr.save_weights(output_dir)

    t_elapsed = time.perf_counter() - t_start

    # Final save
    brain_mgr.save_weights(output_dir)
    print(f"\nTraining complete in {t_elapsed:.1f}s. Weights saved to {output_dir}")

    # Validation
    print(f"\nValidation ({args.validate_ticks} ticks)...")
    from main import run_headless

    report = run_headless(
        brain=args.brain,
        ticks=args.validate_ticks,
        seed=args.seed + 1,
        ants=args.ants,
        config_path=args.config,
        verbose=True,
    )

    report_rule = run_headless(
        brain="rule_based",
        ticks=args.validate_ticks,
        seed=args.seed + 1,
        ants=args.ants,
        config_path=args.config,
        verbose=True,
    )

    food_ppo = report["food_collected_total"]
    food_rule = report_rule["food_collected_total"]
    ratio = food_ppo / max(food_rule, 1)

    print()
    print("=" * 60)
    print(f"  {args.brain} (PPO):       {food_ppo:.1f} food collected")
    print(f"  rule_based (baseline): {food_rule:.1f} food collected")
    print(f"  ratio: {ratio:.2%}")
    print("=" * 60)

    summary = {
        "brain": args.brain,
        "ticks": args.ticks,
        "lr": args.lr,
        "seed": args.seed,
        "food_ppo": food_ppo,
        "food_rule_based": food_rule,
        "ratio": ratio,
        "training_time_s": round(t_elapsed, 1),
    }
    with open(output_dir / "ppo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
