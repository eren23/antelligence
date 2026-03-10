#!/usr/bin/env python3
"""Evolutionary training orchestrator for self-evolving ant colony brains.

Combines:
  - Population-based evolution with MAP-Elites diversity
  - Learned judge model for fitness evaluation
  - Auto-curriculum for difficulty scaling
  - Intrinsic motivation (RND curiosity)
  - Sparse reward signal (no hand-crafted shaping)

Usage:
    python train_evolve.py --brain nn --population 10 --generations 20
    python train_evolve.py --brain transformer --population 20 --generations 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from agents.colony import Colony
from agents.sensory import build_sensory
from brains.intrinsic import IntrinsicCombiner, RNDExplorer
from brains.judge import ComparisonBuffer, JudgeModel
from brains.nn_brain import (
    MLPWeights,
    NNBrain,
    SharedWeightRegistry as NNWeightRegistry,
    RoleTrainer as NNRoleTrainer,
)
from brains.reward import SparseReward
from brains.transformer_brain import (
    TransformerBrain,
    SharedWeightRegistry as TFWeightRegistry,
    RoleTrainer as TFRoleTrainer,
)
from config import SimConfig, default_config, load_config
from evolution.curriculum import AutoCurriculum
from evolution.genome import ArchitectureGenome
from evolution.map_elites import MAPElitesArchive
from evolution.population import Individual, PopulationManager
from metrics.emergence import EmergenceDetector
from metrics.tracker import MetricsTracker
from world.pheromone import PheromoneGrid
from world.world import World


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolutionary training for ant colony brains")
    p.add_argument("--brain", choices=["nn", "transformer", "torch_nn", "torch_transformer"], default="nn")
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--eval-ticks", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="weights/evolution")
    p.add_argument("--config", type=str, default="colony_config.yaml")
    p.add_argument("--ants", type=int, default=100,
                    help="Ants per evaluation (lower for speed)")
    p.add_argument("--elitism", type=int, default=4)
    p.add_argument("--map-elites-dims", type=int, nargs=2, default=[10, 10])
    p.add_argument("--curriculum", action="store_true", default=False,
                    help="Enable auto-curriculum difficulty scaling")
    return p.parse_args()


def _make_nn_weights(cfg: SimConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Create a fresh set of NN weights as flat parameter list."""
    nn_cfg = cfg.brain.nn
    w = MLPWeights.random_init(39, nn_cfg.hidden_sizes[0], nn_cfg.hidden_sizes[1], 11, rng)
    return w.all_params()


def _make_tf_weights(cfg: SimConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Create a fresh set of transformer weights as flat parameter list."""
    from brains.transformer_brain import TransformerWeightSet
    tf_cfg = cfg.brain.transformer
    ws = TransformerWeightSet(39, tf_cfg.d_model, tf_cfg.n_heads, tf_cfg.n_layers,
                              tf_cfg.ffn_dim, rng)
    return [p.copy() for p in ws.all_parameters()]


def _make_torch_nn_weights(cfg: SimConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Create torch NN weights as flat numpy parameter list (for evolution)."""
    import torch
    from brains.torch_nn_brain import TorchMLPModel
    nn_cfg = cfg.brain.nn
    torch.manual_seed(int(rng.integers(2**31)))
    model = TorchMLPModel(39, nn_cfg.hidden_sizes[0], nn_cfg.hidden_sizes[1], 11)
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _make_torch_tf_weights(cfg: SimConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Create torch transformer weights as flat numpy parameter list."""
    import torch
    from brains.torch_transformer_brain import TorchTransformerModel
    tf_cfg = cfg.brain.transformer
    torch.manual_seed(int(rng.integers(2**31)))
    model = TorchTransformerModel(39, tf_cfg.d_model, tf_cfg.n_heads, tf_cfg.n_layers,
                                   tf_cfg.ffn_dim, 11)
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _load_weights_into_registry(
    weights: list[np.ndarray],
    brain_type: str,
    cfg: SimConfig,
    seed: int,
):
    """Load weight arrays into a fresh registry (all roles get same weights)."""
    if brain_type == "nn":
        nn_cfg = cfg.brain.nn
        registry = NNWeightRegistry(39, nn_cfg.hidden_sizes, seed)
        for role in registry.roles():
            w = registry.get(role)
            params = w.all_params()
            for p, src in zip(params, weights):
                if p.shape == src.shape:
                    p[:] = src
        return registry
    elif brain_type == "torch_nn":
        import torch
        from brains.torch_nn_brain import TorchSharedWeightRegistry
        nn_cfg = cfg.brain.nn
        registry = TorchSharedWeightRegistry(39, nn_cfg.hidden_sizes, seed)
        for role in registry.roles():
            model = registry.get(role)
            sd = model.state_dict()
            param_list = list(sd.keys())
            for key, src in zip(param_list, weights):
                if sd[key].shape == torch.Size(src.shape):
                    sd[key] = torch.tensor(src, dtype=torch.float32)
            model.load_state_dict(sd)
        return registry
    elif brain_type == "torch_transformer":
        import torch
        from brains.torch_transformer_brain import TorchTransformerSharedWeightRegistry
        tf_cfg = cfg.brain.transformer
        registry = TorchTransformerSharedWeightRegistry(
            39, tf_cfg.d_model, tf_cfg.n_heads, tf_cfg.n_layers, tf_cfg.ffn_dim, seed,
        )
        for role in registry.roles():
            model = registry.get(role)
            sd = model.state_dict()
            param_list = list(sd.keys())
            for key, src in zip(param_list, weights):
                if sd[key].shape == torch.Size(src.shape):
                    sd[key] = torch.tensor(src, dtype=torch.float32)
            model.load_state_dict(sd)
        return registry
    else:
        from brains.transformer_brain import TransformerWeightSet
        tf_cfg = cfg.brain.transformer
        registry = TFWeightRegistry(39, tf_cfg.d_model, tf_cfg.n_heads,
                                     tf_cfg.n_layers, tf_cfg.ffn_dim, seed)
        for role in registry.roles():
            ws = registry.get(role)
            params = ws.all_parameters()
            for p, src in zip(params, weights):
                if p.shape == src.shape:
                    p[:] = src
        return registry


def evaluate_individual(
    individual: Individual,
    brain_type: str,
    cfg: SimConfig,
    eval_ticks: int,
    seed: int,
    n_ants: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run simulation with individual's weights and return fitness metrics.

    Returns:
        (food_total, emergence_vector, tracker_vector, behavior_descriptor)
    """
    rng_seed = seed + individual.id

    # Create world
    world = World.from_config(cfg, seed=rng_seed)
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    colony = Colony(cfg, world.nest.center, seed=rng_seed)

    # Load weights into registry
    registry = _load_weights_into_registry(individual.weights, brain_type, cfg, rng_seed)

    # Create RND explorer shared across ants
    rnd = RNDExplorer(input_dim=39, seed=rng_seed)
    combiner = IntrinsicCombiner(coef=individual.genome.intrinsic_coef)

    # Create brains for all ants
    sparse_reward = SparseReward()
    if brain_type == "nn":
        trainers: dict[str, NNRoleTrainer] = {}
        for role in registry.roles():
            trainers[role] = NNRoleTrainer(
                registry.get(role),
                lr=individual.genome.learning_rate,
                gamma=cfg.brain.nn.gamma,
                buffer_size=cfg.brain.nn.buffer_size,
            )
        for ant in colony.ants:
            ant.brain = NNBrain(
                role=ant.role.value,
                registry=registry,
                trainer=trainers[ant.role.value],
                cfg=cfg.brain.nn,
                seed=rng_seed + ant.id,
                intrinsic_combiner=combiner,
                rnd_explorer=rnd,
            )
    elif brain_type == "torch_nn":
        from brains.torch_nn_brain import TorchNNBrain, TorchRoleTrainer as TorchNNRT
        torch_trainers: dict[str, TorchNNRT] = {}
        for role in registry.roles():
            torch_trainers[role] = TorchNNRT(
                registry.get(role),
                lr=individual.genome.learning_rate,
                gamma=cfg.brain.nn.gamma,
                buffer_size=cfg.brain.nn.buffer_size,
            )
        for ant in colony.ants:
            ant.brain = TorchNNBrain(
                role=ant.role.value,
                registry=registry,
                trainer=torch_trainers[ant.role.value],
                cfg=cfg.brain.nn,
                seed=rng_seed + ant.id,
            )
    elif brain_type == "torch_transformer":
        from brains.torch_transformer_brain import (
            TorchTransformerBrain, TorchTransformerRoleTrainer as TorchTFRT,
        )
        torch_tf_trainers: dict[str, TorchTFRT] = {}
        for role in registry.roles():
            torch_tf_trainers[role] = TorchTFRT(
                registry.get(role),
                lr=individual.genome.learning_rate,
                gamma=0.95,
                buffer_size=cfg.brain.transformer.buffer_size,
                context_length=cfg.brain.transformer.context_length,
            )
        for ant in colony.ants:
            ant.brain = TorchTransformerBrain(
                role=ant.role.value,
                registry=registry,
                trainer=torch_tf_trainers[ant.role.value],
                cfg=cfg.brain.transformer,
                seed=rng_seed + ant.id,
            )
    else:
        tf_trainers: dict[str, TFRoleTrainer] = {}
        for role in registry.roles():
            tf_trainers[role] = TFRoleTrainer(
                registry.get(role),
                lr=individual.genome.learning_rate,
                gamma=0.95,
                buffer_size=cfg.brain.transformer.buffer_size,
            )
        for ant in colony.ants:
            ant.brain = TransformerBrain(
                role=ant.role.value,
                registry=registry,
                trainer=tf_trainers[ant.role.value],
                cfg=cfg.brain.transformer,
                seed=rng_seed + ant.id,
                intrinsic_combiner=combiner,
                rnd_explorer=rnd,
            )

    # Metrics
    metrics = MetricsTracker(window=0)
    emergence = EmergenceDetector(cfg.roles.default_distribution)

    # Run simulation
    from agents.ant import (
        update_heading, advance_position, handle_obstacle_collision,
        handle_world_bounds, deplete_energy, check_death, refill_at_nest,
        try_pickup, try_drop, Role,
    )
    from agents.actions import AntAction
    from world.spatial import SpatialGrid
    from world.pheromone import Channel

    _CHANNEL_MAP = {"food": Channel.FOOD, "home": Channel.HOME,
                     "danger": Channel.DANGER, "recruit": Channel.RECRUIT}

    for tick in range(1, eval_ticks + 1):
        ants = colony.ants

        # Spatial grid for neighbor lookup
        spatial_grid = SpatialGrid(cfg.world.width, cfg.world.height, cell_size=40)
        spatial_grid.rebuild(ants)

        # Build sensory
        for ant in ants:
            if ant.alive:
                ant.sensory = build_sensory(ant, world, pheromone_grid, ants, cfg.ant,
                                            spatial_grid=spatial_grid)

        # Reward & learn
        for ant in ants:
            if not ant.alive or ant.brain is None or ant.sensory is None:
                continue
            prev_act = getattr(ant.brain, "_prev_action", None)
            if prev_act is None:
                continue
            prev_si = getattr(ant.brain, "_prev_sensory", None)
            reward = sparse_reward.compute(prev_si, ant.sensory, prev_act, True)
            ant.brain.learn(reward)
            metrics.accumulate_reward(reward)

        # Decide
        actions: dict[int, AntAction] = {}
        for ant in ants:
            if ant.alive and ant.brain is not None and ant.sensory is not None:
                actions[ant.id] = ant.brain.decide(ant.sensory)
            else:
                actions[ant.id] = AntAction()

        # Apply actions (no patches for learning brains)
        for ant in ants:
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
                if ant.brain is not None:
                    ant.brain.learn(-1.0)
                    metrics.accumulate_reward(-1.0)
                continue

            carry_before = ant.carry_amount
            deposited = refill_at_nest(ant, world, cfg.ant)
            if deposited == "food":
                colony.deposit_food(carry_before)

            if action.pickup:
                try_pickup(ant, world, cfg.ant)
            if action.drop and ant.carrying is not None and ant.carrying != "food":
                try_drop(ant)

        # Pheromone
        for ant in ants:
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

        # Assign brains to newly spawned ants
        for ant in colony.ants:
            if ant.brain is None and ant.alive:
                if brain_type == "nn":
                    ant.brain = NNBrain(
                        role=ant.role.value,
                        registry=registry,
                        trainer=trainers[ant.role.value],
                        cfg=cfg.brain.nn,
                        seed=rng_seed + ant.id,
                        intrinsic_combiner=combiner,
                        rnd_explorer=rnd,
                    )
                elif brain_type == "torch_nn":
                    from brains.torch_nn_brain import TorchNNBrain
                    ant.brain = TorchNNBrain(
                        role=ant.role.value,
                        registry=registry,
                        trainer=torch_trainers[ant.role.value],
                        cfg=cfg.brain.nn,
                        seed=rng_seed + ant.id,
                    )
                elif brain_type == "torch_transformer":
                    from brains.torch_transformer_brain import TorchTransformerBrain
                    ant.brain = TorchTransformerBrain(
                        role=ant.role.value,
                        registry=registry,
                        trainer=torch_tf_trainers[ant.role.value],
                        cfg=cfg.brain.transformer,
                        seed=rng_seed + ant.id,
                    )
                else:
                    ant.brain = TransformerBrain(
                        role=ant.role.value,
                        registry=registry,
                        trainer=tf_trainers[ant.role.value],
                        cfg=cfg.brain.transformer,
                        seed=rng_seed + ant.id,
                        intrinsic_combiner=combiner,
                        rnd_explorer=rnd,
                    )

        snapshot = metrics.record(tick, colony, world, pheromone_grid)
        emergence.update(tick, colony, world, pheromone_grid, snapshot.food_income)

    # Compute fitness
    stats = colony.stats()
    food_total = stats.food_stored

    emergence_vec = emergence.report.as_vector()
    tracker_vec = metrics.aggregate_window(eval_ticks)

    # Behavior descriptor: (foraging_efficiency, trail_formation)
    behavior = np.array([
        emergence.report.foraging_efficiency,
        emergence.report.trail_formation,
    ])

    return food_total, emergence_vec, tracker_vec, behavior


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Evolutionary Training: {args.brain} ===")
    print(f"  population: {args.population}, generations: {args.generations}")
    print(f"  eval_ticks: {args.eval_ticks}, ants: {args.ants}, seed: {args.seed}")
    print()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else default_config()
    cfg.brain.default = args.brain
    cfg.colony.initial_population = args.ants

    # Initialize population
    pop_mgr = PopulationManager(
        size=args.population,
        eval_ticks=args.eval_ticks,
        elitism=args.elitism,
        seed=args.seed,
    )

    def weight_factory():
        if args.brain == "nn":
            return _make_nn_weights(cfg, rng)
        elif args.brain == "torch_nn":
            return _make_torch_nn_weights(cfg, rng)
        elif args.brain == "torch_transformer":
            return _make_torch_tf_weights(cfg, rng)
        else:
            return _make_tf_weights(cfg, rng)

    pop_mgr.initialize(weight_factory)

    # MAP-Elites archive
    archive = MAPElitesArchive(dims=tuple(args.map_elites_dims))

    # Judge model
    judge = JudgeModel(input_dim=12, hidden=32, seed=args.seed)
    comparison_buffer = ComparisonBuffer(capacity=200)

    # Auto-curriculum
    curriculum = AutoCurriculum() if args.curriculum else None

    t_start = time.perf_counter()

    for gen in range(1, args.generations + 1):
        gen_start = time.perf_counter()

        # Apply curriculum to config
        if curriculum is not None:
            curriculum.apply_to_config(cfg)

        # Evaluate each individual
        for idx, individual in enumerate(pop_mgr.population):
            food_total, emergence_vec, tracker_vec, behavior = evaluate_individual(
                individual, args.brain, cfg, args.eval_ticks,
                args.seed + gen * 1000, args.ants,
            )

            # Compute fitness: judge score + raw food as baseline
            judge_features = np.concatenate([emergence_vec, tracker_vec])
            judge_score = judge.score(judge_features)
            fitness = food_total + judge_score * 10.0  # blend

            pop_mgr.set_fitness(idx, fitness, behavior)

            # Record for judge training
            comparison_buffer.add_episode(judge_features, food_total)

            # Try to insert into MAP-Elites
            archive.try_insert(individual)

        # Train judge from comparisons
        if len(comparison_buffer) >= 4:
            pairs = comparison_buffer.sample_pairs(n=32, rng=rng)
            judge_loss = judge.train_from_comparisons(pairs)
        else:
            judge_loss = 0.0

        # Update curriculum
        best = pop_mgr.get_best()
        if curriculum is not None:
            best_eff = best.behavior_descriptor[0] if len(best.behavior_descriptor) > 0 else 0.0
            stage_changed = curriculum.update(best_eff)
            if stage_changed:
                print(f"  [Curriculum] Stage changed to: {curriculum.stage_name}")

        # Report
        stats = pop_mgr.stats()
        gen_time = time.perf_counter() - gen_start
        print(
            f"  Gen {gen:>3}/{args.generations}  |"
            f"  Best {stats['best_fitness']:>8.1f}  |"
            f"  Mean {stats['mean_fitness']:>8.1f}  |"
            f"  Std {stats['std_fitness']:>6.2f}  |"
            f"  Archive {archive.coverage()*100:.0f}%  |"
            f"  Judge loss {judge_loss:.4f}  |"
            f"  {gen_time:.1f}s"
        )

        # Select and reproduce
        pop_mgr.select_and_reproduce()

    t_elapsed = time.perf_counter() - t_start

    # Save best individual
    best = pop_mgr.get_best()
    np.savez(
        output_dir / f"best_{args.brain}.npz",
        **{f"p{i}": w for i, w in enumerate(best.weights)},
    )

    # Save archive summary
    archive_inds = archive.all_individuals()
    archive_summary = {
        "coverage": archive.coverage(),
        "total_insertions": archive.total_insertions,
        "num_occupied": len(archive_inds),
        "best_fitness": archive.best_fitness(),
    }
    with open(output_dir / "archive_summary.json", "w") as f:
        json.dump(archive_summary, f, indent=2)

    # Save training summary
    summary = {
        "brain": args.brain,
        "population": args.population,
        "generations": args.generations,
        "eval_ticks": args.eval_ticks,
        "seed": args.seed,
        "best_fitness": best.fitness,
        "final_archive_coverage": archive.coverage(),
        "training_time_s": round(t_elapsed, 1),
    }
    with open(output_dir / "evolution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete in {t_elapsed:.1f}s")
    print(f"  Best fitness: {best.fitness:.1f}")
    print(f"  Archive coverage: {archive.coverage()*100:.0f}%")
    print(f"  Saved to {output_dir}")


if __name__ == "__main__":
    main()
