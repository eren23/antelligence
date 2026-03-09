#!/usr/bin/env python3
"""Train a neural brain via behavioral cloning from the rule-based brain.

Usage:
    python train_imitation.py --brain nn --demo-ticks 5000 --epochs 50
    python train_imitation.py --brain transformer --demo-ticks 5000 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from brains.imitation import DemonstrationCollector, ImitationTrainer, TransformerImitationTrainer
from brains.nn_brain import MLPWeights, SharedWeightRegistry as NNWeightRegistry
from brains.transformer_brain import (
    SharedWeightRegistry as TFWeightRegistry,
    TransformerWeightSet,
)
from config import load_config, default_config, ImitationConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Imitation learning from rule-based brain")
    p.add_argument("--brain", choices=["nn", "transformer"], default="nn",
                    help="Brain type to train (default: nn)")
    p.add_argument("--demo-ticks", type=int, default=5000,
                    help="Ticks for demonstration collection (default: 5000)")
    p.add_argument("--demo-ants", type=int, default=200,
                    help="Ant count for demonstrations (default: 200)")
    p.add_argument("--epochs", type=int, default=50,
                    help="Training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=64,
                    help="Mini-batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
    p.add_argument("--seed", type=int, default=42,
                    help="RNG seed (default: 42)")
    p.add_argument("--output-dir", type=str, default="weights/imitation",
                    help="Output directory for weights (default: weights/imitation)")
    p.add_argument("--config", type=str, default="colony_config.yaml",
                    help="Config file path")
    p.add_argument("--validate-ticks", type=int, default=5000,
                    help="Ticks for validation run (default: 5000)")
    p.add_argument("--roles", nargs="+", default=["forager"],
                    help="Roles to collect demonstrations for (default: forager)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Imitation Learning: {args.brain} ===")
    print(f"  demo-ticks: {args.demo_ticks}, epochs: {args.epochs}, seed: {args.seed}")
    print()

    # ---- Phase 1: Collect demonstrations ----
    print("Phase 1: Collecting demonstrations from rule-based brain...")
    t0 = time.perf_counter()

    collector = DemonstrationCollector(
        ticks=args.demo_ticks,
        num_ants=args.demo_ants,
        seed=args.seed,
        config_path=args.config,
        roles=args.roles,
    )
    states, targets = collector.collect(verbose=True)
    t_collect = time.perf_counter() - t0
    print(f"  Collection time: {t_collect:.1f}s")
    print()

    # ---- Phase 2: Train ----
    print(f"Phase 2: Training {args.brain} on {len(states):,} demonstrations...")
    t0 = time.perf_counter()

    roles = ["forager", "nurse", "soldier", "idle"]

    if args.brain == "nn":
        registry = NNWeightRegistry(input_dim=39, hidden_sizes=[64, 32], seed=args.seed)

        for role in roles:
            print(f"\n  Training role: {role}")
            weights = registry.get(role)
            trainer = ImitationTrainer(
                weights,
                lr=args.lr,
                lr_decay=0.95,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            # Filter demos for this role if we have role info, otherwise use all
            # Since role is encoded in sensory vector positions 21-24,
            # we can filter by role index
            role_idx_map = {"forager": 0, "nurse": 1, "soldier": 2, "idle": 3}
            if role in args.roles:
                # Use all data for roles we collected
                losses = trainer.train(states, targets, verbose=True)
            else:
                # For roles we didn't specifically collect, still train on all data
                # (cross-role imitation can still help)
                losses = trainer.train(states, targets, verbose=True)

            # Save weights
            w = weights
            np.savez(
                output_dir / f"nn_{role}.npz",
                W1=w.W1, b1=w.b1, W2=w.W2, b2=w.b2,
                W_out=w.W_out, b_out=w.b_out,
                W_val=w.W_val, b_val=w.b_val,
            )

    elif args.brain == "transformer":
        registry = TFWeightRegistry(
            input_dim=39, d_model=32, n_heads=4, n_layers=2, ffn_dim=64,
            seed=args.seed,
        )

        for role in roles:
            print(f"\n  Training role: {role}")
            ws = registry.get(role)
            trainer = TransformerImitationTrainer(
                ws,
                lr=args.lr,
                lr_decay=0.95,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            losses = trainer.train(states, targets, verbose=True)

            # Save weights
            params = ws.all_parameters()
            np.savez(
                output_dir / f"transformer_{role}.npz",
                **{f"p{i}": p for i, p in enumerate(params)},
            )

    t_train = time.perf_counter() - t0
    print(f"\n  Training time: {t_train:.1f}s")
    print()

    # ---- Phase 3: Validate ----
    print(f"Phase 3: Validation run ({args.validate_ticks} ticks)...")
    t0 = time.perf_counter()

    from main import run_headless

    # Validate imitation-trained brain
    report_imitation = run_headless(
        brain=args.brain,
        ticks=args.validate_ticks,
        seed=args.seed + 1,  # different seed for validation
        ants=args.demo_ants,
        config_path=args.config,
        verbose=True,
    )

    # Also run rule-based for comparison
    print("\n  Rule-based baseline:")
    report_rule = run_headless(
        brain="rule_based",
        ticks=args.validate_ticks,
        seed=args.seed + 1,
        ants=args.demo_ants,
        config_path=args.config,
        verbose=True,
    )

    t_validate = time.perf_counter() - t0

    # ---- Summary ----
    food_imitation = report_imitation["food_collected_total"]
    food_rule = report_rule["food_collected_total"]
    ratio = food_imitation / max(food_rule, 1)

    print()
    print("=" * 60)
    print(f"  {args.brain} (imitation):  {food_imitation:.1f} food collected")
    print(f"  rule_based (baseline):  {food_rule:.1f} food collected")
    print(f"  ratio: {ratio:.2%}")
    print(f"  weights saved to: {output_dir}")
    print("=" * 60)

    # Save training summary
    summary = {
        "brain": args.brain,
        "demo_ticks": args.demo_ticks,
        "epochs": args.epochs,
        "seed": args.seed,
        "num_demonstrations": len(states),
        "food_imitation": food_imitation,
        "food_rule_based": food_rule,
        "ratio": ratio,
        "collection_time_s": round(t_collect, 1),
        "training_time_s": round(t_train, 1),
        "validation_time_s": round(t_validate, 1),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
