"""Overnight training script — runs headless simulation with periodic checkpointing.

Usage:
    python3 train_overnight.py --brain nn --ticks 500000 --ants 300 --save-every 10000
    python3 train_overnight.py --brain transformer --ticks 200000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import sys
import time as _time
from pathlib import Path

import numpy as np

from agents.colony import Colony
from config import SimConfig, default_config, load_config
from main import BrainManager, sim_tick
from metrics.emergence import EmergenceDetector
from metrics.tracker import MetricsTracker
from world.pheromone import PheromoneGrid
from world.world import World


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overnight headless training")
    p.add_argument("--brain", choices=["rule_based", "torch_nn", "torch_transformer"],
                   default="torch_nn", help="Brain backend to train (default: torch_nn)")
    p.add_argument("--ticks", "-t", type=int, default=100_000,
                   help="Total training ticks (default: 100000)")
    p.add_argument("--ants", type=int, default=None,
                   help="Override initial ant population")
    p.add_argument("--seed", "-s", type=int, default=42,
                   help="RNG seed (default: 42)")
    p.add_argument("--config", "-c", default="colony_config.yaml",
                   help="YAML config path")
    p.add_argument("--save-every", type=int, default=5000,
                   help="Auto-save weights every N ticks (default: 5000)")
    p.add_argument("--output-dir", type=str, default="weights",
                   help="Directory for weight checkpoints (default: weights/)")
    p.add_argument("--log-every", type=int, default=1000,
                   help="Print progress every N ticks (default: 1000)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed = args.seed

    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)

    # Configuration
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config()
    cfg.brain.default = args.brain
    if args.ants is not None:
        cfg.colony.initial_population = args.ants

    # Initialise
    brain_mgr = BrainManager(cfg, seed)
    world = World.from_config(cfg, seed=seed)
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    colony = Colony(cfg, world.nest.center, seed=seed)

    for ant in colony.ants:
        brain_mgr.create_brain(ant)

    metrics = MetricsTracker(window=0)
    emergence = EmergenceDetector(cfg.roles.default_distribution)

    output_dir = Path(args.output_dir)
    best_dir = output_dir / "best"
    best_food_income: float = -1.0

    # Graceful shutdown on Ctrl+C
    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nGraceful shutdown requested...")

    signal.signal(signal.SIGINT, _sigint_handler)

    print(f"Training: brain={args.brain}, ticks={args.ticks:,}, "
          f"ants={cfg.colony.initial_population}, seed={seed}")
    print(f"Checkpoints → {output_dir}/  |  Best → {best_dir}/")
    print("-" * 70)

    t_start = _time.perf_counter()
    total_reward = 0.0
    reward_count = 0

    for tick in range(1, args.ticks + 1):
        if interrupted:
            break

        sim_tick(tick, colony, world, pheromone_grid, brain_mgr, metrics, emergence, cfg)

        # Track rewards
        snap = metrics.latest
        if snap is not None:
            total_reward += snap.avg_reward
            reward_count += 1

        # Progress logging
        if tick % args.log_every == 0:
            stats = colony.stats()
            elapsed = _time.perf_counter() - t_start
            tps = tick / elapsed if elapsed > 0 else 0
            avg_r = total_reward / reward_count if reward_count > 0 else 0.0

            # Food income over last 100 ticks
            snapshots = metrics.snapshots
            tail = snapshots[-100:] if len(snapshots) >= 100 else snapshots
            food_rate = sum(s.food_income for s in tail) / len(tail) if tail else 0.0

            print(
                f"Tick {tick:>8,}/{args.ticks:,}  |  Pop {stats.population:>4}"
                f"  |  Food {stats.food_stored:>8.1f}"
                f"  |  Income {food_rate:>6.3f}"
                f"  |  Avg R {avg_r:>+7.4f}"
                f"  |  {tps:>6.0f} t/s"
            )

        # Periodic checkpoint
        if tick % args.save_every == 0:
            brain_mgr.save_weights(output_dir)

            # Track best checkpoint by food income rate
            snapshots = metrics.snapshots
            tail = snapshots[-100:] if len(snapshots) >= 100 else snapshots
            food_rate = sum(s.food_income for s in tail) / len(tail) if tail else 0.0

            if food_rate > best_food_income:
                best_food_income = food_rate
                brain_mgr.save_weights(best_dir)
                print(f"  ★ New best checkpoint (income={food_rate:.4f})")

    # Final save
    elapsed = _time.perf_counter() - t_start
    brain_mgr.save_weights(output_dir)

    # Check if this final state is the best
    snapshots = metrics.snapshots
    tail = snapshots[-100:] if len(snapshots) >= 100 else snapshots
    food_rate = sum(s.food_income for s in tail) / len(tail) if tail else 0.0
    if food_rate > best_food_income:
        best_food_income = food_rate
        brain_mgr.save_weights(best_dir)

    # Summary
    stats = colony.stats()
    summary = {
        "brain": args.brain,
        "seed": seed,
        "ticks_completed": tick,
        "ticks_target": args.ticks,
        "wall_time_seconds": round(elapsed, 2),
        "ticks_per_second": round(tick / elapsed, 1) if elapsed > 0 else 0,
        "final_population": stats.population,
        "final_food_stored": round(stats.food_stored, 2),
        "best_food_income_rate": round(best_food_income, 4),
        "final_food_income_rate": round(food_rate, 4),
        "avg_reward": round(total_reward / reward_count, 6) if reward_count > 0 else 0.0,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 70)
    print(f"Training complete: {tick:,} ticks in {elapsed:.1f}s ({tick/elapsed:.0f} t/s)")
    print(f"Final food income: {food_rate:.4f} | Best: {best_food_income:.4f}")
    print(f"Weights → {output_dir}/  |  Best → {best_dir}/")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
