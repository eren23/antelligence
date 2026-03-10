#!/usr/bin/env python3
"""Fine-tune a brain with PPO using GPU-batched ActionDistribution.

Two-phase collect-train loop:
  Phase 1 (COLLECT): CPU sim + GPU inference per tick
  Phase 2 (TRAIN):   All GPU — batched PPO updates via ActionDistribution

Usage:
    python3 train_ppo.py --brain torch_nn --ticks 50000
    python3 train_ppo.py --brain torch_transformer --ticks 50000 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.actions import AntAction
from agents.sensory import SensoryInput, build_sensory
from brains.action_utils import compute_reward
from brains.reward import SparseReward
from brains.torch_ppo import TorchPPOTrainer
from brains.rollout_storage import RolloutStorage
from config import SimConfig, load_config, default_config, PPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training with GPU-batched actions")
    p.add_argument("--brain", choices=["torch_nn", "torch_transformer"], default="torch_nn")
    p.add_argument("--load-weights", type=str, default=None,
                    help="Load pre-trained weights from DIR")
    p.add_argument("--ticks", type=int, default=50000,
                    help="Total training ticks (default: 50000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="weights/ppo",
                    help="Output directory for PPO weights")
    p.add_argument("--config", type=str, default="colony_config.yaml")
    p.add_argument("--validate-ticks", type=int, default=5000,
                    help="Ticks for validation run")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout-length", type=int, default=1024)
    p.add_argument("--save-interval", type=int, default=10000,
                    help="Save weights every N ticks")
    p.add_argument("--ants", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PPO Training: {args.brain} ===")
    print(f"  ticks: {args.ticks}, lr: {args.lr}, seed: {args.seed}")
    if args.load_weights:
        print(f"  loading weights from: {args.load_weights}")
    print()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    brain_mgr = BrainManager(cfg, args.seed)
    if args.load_weights:
        load_path = Path(args.load_weights)
        if load_path.is_dir():
            brain_mgr.load_weights(load_path)

    # Create PPO trainers per role
    ppo_trainers: dict[str, TorchPPOTrainer] = {}

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
        """Extract PPO state in model-ready shape."""
        prev_vec = getattr(ant_brain, "_prev_vec", None)
        if prev_vec is not None:
            return np.asarray(prev_vec, dtype=np.float32)

        prev_ctx_flat = getattr(ant_brain, "_prev_context_flat", None)
        if prev_ctx_flat is None:
            return None

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

    # Training loop
    t_start = time.perf_counter()
    sparse_reward = SparseReward()

    from brains.action_dist import ActionDistribution
    from brains.rollout_storage import RolloutStorage as RS

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
            reward = sparse_reward.compute(prev_si, ant.sensory, prev_act, True)
            metrics.accumulate_reward(reward)

            role = ant.role.value
            if role in ppo_trainers:
                trainer = ppo_trainers[role]
                prev_vec = _ppo_state_from_brain(ant.brain)
                if prev_vec is not None:
                    prev_lp = getattr(ant.brain, "_prev_log_prob", 0.0)

                    # Get value estimate
                    with torch.no_grad():
                        _model = trainer.model
                        _dev = trainer._device
                        _x = torch.tensor(prev_vec, dtype=torch.float32, device=_dev)
                        if args.brain == "torch_transformer":
                            if _x.dim() == 1:
                                _x = _x.view(1, -1, 39)
                            elif _x.dim() == 2:
                                _x = _x.unsqueeze(0)
                        else:
                            if _x.dim() == 1:
                                _x = _x.unsqueeze(0)
                        _, _val = _model(_x)
                        value = float(_val.item())

                    # Store as tensors in RolloutStorage
                    state_t = torch.tensor(prev_vec.flatten(), dtype=torch.float32, device=_dev)
                    # Build action tensor from AntAction
                    from brains.action_utils import _DEPOSIT_CHANNELS
                    dep_idx = _DEPOSIT_CHANNELS.index(prev_act.deposit_pheromone)
                    act_t = torch.tensor([
                        prev_act.turn, prev_act.speed_mult, float(dep_idx),
                        prev_act.deposit_strength,
                        float(prev_act.pickup), float(prev_act.drop),
                        float(prev_act.recruit_signal),
                    ], dtype=torch.float32, device=_dev)
                    lp_t = torch.tensor(prev_lp, dtype=torch.float32, device=_dev)
                    val_t = torch.tensor(value, dtype=torch.float32, device=_dev)

                    trainer.buffer.insert(state_t, act_t, lp_t, val_t, reward, False)

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

        # Apply actions
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

            update_heading(ant, action)
            advance_position(ant, action, world)
            handle_obstacle_collision(ant, world)
            handle_world_bounds(ant, world)

            deplete_energy(ant, action, cfg.ant)
            died = check_death(ant)
            if died:
                role = ant.role.value
                if role in ppo_trainers:
                    trainer = ppo_trainers[role]
                    prev_vec = _ppo_state_from_brain(ant.brain)
                    if prev_vec is not None:
                        _dev = trainer._device
                        state_t = torch.tensor(prev_vec.flatten(), dtype=torch.float32, device=_dev)
                        act_t = torch.zeros(7, dtype=torch.float32, device=_dev)
                        lp_t = torch.tensor(0.0, dtype=torch.float32, device=_dev)
                        val_t = torch.tensor(0.0, dtype=torch.float32, device=_dev)
                        trainer.buffer.insert(state_t, act_t, lp_t, val_t, -0.5, True)
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
