"""Compare all three brain backends across multiple seeds.

Usage:
    python compare_brains.py                        # defaults: 3 seeds, 5000 ticks
    python compare_brains.py --ticks 10000 --seeds 1 2 3 4 5
    python compare_brains.py --ants 100 --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from main import run_headless

BRAINS = ["rule_based", "nn", "transformer"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare brain backends")
    p.add_argument("--ticks", "-t", type=int, default=5000,
                   help="Ticks per simulation (default: 5000)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7],
                   help="RNG seeds to test (default: 42 123 7)")
    p.add_argument("--ants", type=int, default=None,
                   help="Override initial ant population")
    p.add_argument("--config", "-c", default="colony_config.yaml",
                   help="YAML config path")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Save full results to JSON file")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress per-tick progress output")
    return p.parse_args(argv)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_comparison(
    *,
    ticks: int,
    seeds: list[int],
    ants: int | None,
    config_path: str,
    verbose: bool,
) -> dict[str, Any]:
    """Run all brains across all seeds. Returns full results dict."""
    all_results: dict[str, list[dict[str, Any]]] = {b: [] for b in BRAINS}
    total_runs = len(BRAINS) * len(seeds)
    run_num = 0

    for brain in BRAINS:
        for seed in seeds:
            run_num += 1
            print(f"\n{'='*60}")
            print(f"Run {run_num}/{total_runs}: brain={brain}, seed={seed}")
            print(f"{'='*60}")

            report = run_headless(
                brain=brain,
                ticks=ticks,
                seed=seed,
                ants=ants,
                config_path=config_path,
                verbose=verbose,
            )
            all_results[brain].append(report)

    return all_results


def _trail_formation_speed(report: dict[str, Any]) -> float:
    """Estimate how quickly trails form: tick at which entropy first drops
    below 50% of peak. Lower = faster trail formation."""
    entropy = report.get("trail_entropy_over_time", [])
    if not entropy:
        return float("inf")
    peak = max(entropy)
    if peak < 1e-6:
        return float("inf")
    threshold = peak * 0.5
    total_ticks = report["config"]["ticks"]
    step = max(1, total_ticks // len(entropy))
    for i, val in enumerate(entropy):
        if val >= threshold:
            return float(i * step)
    return float("inf")


def print_comparison_table(results: dict[str, list[dict[str, Any]]]) -> None:
    """Print a formatted comparison table to stdout."""
    print(f"\n{'='*80}")
    print("BRAIN COMPARISON RESULTS")
    print(f"{'='*80}\n")

    # Column headers
    header = f"{'Metric':<30}"
    for brain in BRAINS:
        header += f"  {brain:>14}"
    print(header)
    print("-" * len(header))

    # Aggregate metrics per brain
    rows: list[tuple[str, list[str]]] = []

    for label, key, fmt in [
        ("Food collected (total)", "food_collected_total", ".1f"),
        ("Food income rate (final)", "food_income_rate_final", ".4f"),
        ("Population (final)", "population_final", ".0f"),
        ("Deaths (total)", "deaths", ".0f"),
        ("Avg foraging trip (ticks)", "avg_foraging_trip_ticks", ".1f"),
        ("Cemetery clusters", "cemetery_cluster_count", ".1f"),
    ]:
        vals = []
        for brain in BRAINS:
            reports = results[brain]
            mean = _avg([r[key] for r in reports])
            vals.append(f"{mean:{fmt}}")
        rows.append((label, vals))

    # Trail formation speed (derived metric)
    trail_vals = []
    for brain in BRAINS:
        reports = results[brain]
        speeds = [_trail_formation_speed(r) for r in reports]
        finite = [s for s in speeds if s != float("inf")]
        if finite:
            trail_vals.append(f"{_avg(finite):.0f}")
        else:
            trail_vals.append("N/A")
    rows.append(("Trail formation speed (tick)", trail_vals))

    # Foraging efficiency from emergence
    eff_vals = []
    for brain in BRAINS:
        reports = results[brain]
        mean = _avg([r["emergence"]["foraging_efficiency"] for r in reports])
        eff_vals.append(f"{mean:.3f}")
    rows.append(("Foraging efficiency (score)", eff_vals))

    # Avg reward (brain_specific)
    reward_vals = []
    for brain in BRAINS:
        reports = results[brain]
        mean = _avg([
            r["brain_specific"].get("avg_reward", 0.0) for r in reports
        ])
        reward_vals.append(f"{mean:.4f}")
    rows.append(("Avg reward", reward_vals))

    # Wall time
    time_vals = []
    for brain in BRAINS:
        reports = results[brain]
        mean = _avg([r["performance"]["wall_time_seconds"] for r in reports])
        time_vals.append(f"{mean:.1f}s")
    rows.append(("Wall time (avg)", time_vals))

    # Print rows
    for label, vals in rows:
        line = f"{label:<30}"
        for v in vals:
            line += f"  {v:>14}"
        print(line)

    print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    results = run_comparison(
        ticks=args.ticks,
        seeds=args.seeds,
        ants=args.ants,
        config_path=args.config,
        verbose=not args.quiet,
    )

    print_comparison_table(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full results saved → {out_path}")


if __name__ == "__main__":
    main()
