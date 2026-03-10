#!/usr/bin/env python3
"""Check torch-vs-numpy migration quality from compare_brains.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAIRS = [
    ("nn", "torch_nn"),
    ("transformer", "torch_transformer"),
]


def _avg_metric(reports: list[dict[str, Any]], key: str) -> float:
    if not reports:
        return 0.0
    return sum(float(r.get(key, 0.0)) for r in reports) / len(reports)


def _pair_check(
    results: dict[str, list[dict[str, Any]]],
    base: str,
    torch_name: str,
    max_food_drop: float,
    max_death_increase: float,
) -> dict[str, Any]:
    base_reports = results.get(base, [])
    torch_reports = results.get(torch_name, [])

    if not base_reports or not torch_reports:
        return {
            "base": base,
            "torch": torch_name,
            "present": False,
            "pass": False,
            "reason": "missing reports for one or both backends",
        }

    base_food = _avg_metric(base_reports, "food_collected_total")
    torch_food = _avg_metric(torch_reports, "food_collected_total")
    base_deaths = _avg_metric(base_reports, "deaths")
    torch_deaths = _avg_metric(torch_reports, "deaths")

    food_drop_ratio = 0.0 if base_food <= 1e-9 else max(0.0, (base_food - torch_food) / base_food)
    death_increase_ratio = 0.0 if base_deaths <= 1e-9 else max(0.0, (torch_deaths - base_deaths) / base_deaths)

    ok_food = food_drop_ratio <= max_food_drop
    ok_deaths = death_increase_ratio <= max_death_increase

    return {
        "base": base,
        "torch": torch_name,
        "present": True,
        "pass": bool(ok_food and ok_deaths),
        "metrics": {
            "food_avg_base": round(base_food, 4),
            "food_avg_torch": round(torch_food, 4),
            "deaths_avg_base": round(base_deaths, 4),
            "deaths_avg_torch": round(torch_deaths, 4),
            "food_drop_ratio": round(food_drop_ratio, 6),
            "death_increase_ratio": round(death_increase_ratio, 6),
        },
        "thresholds": {
            "max_food_drop": max_food_drop,
            "max_death_increase": max_death_increase,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check torch migration quality against numpy baselines")
    p.add_argument("--input", required=True, help="compare_brains.py JSON file")
    p.add_argument("--output", default=None, help="Optional JSON output path")
    p.add_argument(
        "--pairs",
        nargs="*",
        default=[],
        help="Optional pair list in base:torch format (default: nn:torch_nn transformer:torch_transformer)",
    )
    p.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="Treat missing pair data as skipped instead of failed",
    )
    p.add_argument("--max-food-drop", type=float, default=0.20,
                   help="Maximum allowed relative food drop (default: 0.20)")
    p.add_argument("--max-death-increase", type=float, default=0.20,
                   help="Maximum allowed relative death increase (default: 0.20)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    with open(in_path) as f:
        results: dict[str, list[dict[str, Any]]] = json.load(f)

    pairs: list[tuple[str, str]] = []
    if args.pairs:
        for pair in args.pairs:
            if ":" not in pair:
                raise SystemExit(f"invalid --pairs entry '{pair}', expected base:torch")
            base, torch_name = pair.split(":", 1)
            base = base.strip()
            torch_name = torch_name.strip()
            if not base or not torch_name:
                raise SystemExit(f"invalid --pairs entry '{pair}', expected base:torch")
            pairs.append((base, torch_name))
    else:
        pairs = list(PAIRS)

    checks = [
        _pair_check(
            results,
            base,
            torch_name,
            max_food_drop=args.max_food_drop,
            max_death_increase=args.max_death_increase,
        )
        for base, torch_name in pairs
    ]

    if args.allow_missing_pairs:
        overall_pass = all((not c.get("present", False)) or c.get("pass", False) for c in checks)
    else:
        overall_pass = all(c.get("pass", False) for c in checks)
    summary = {
        "overall_pass": overall_pass,
        "checks": checks,
        "input": str(in_path),
        "allow_missing_pairs": bool(args.allow_missing_pairs),
        "pairs": [f"{base}:{torch_name}" for base, torch_name in pairs],
    }

    print("Torch Migration Check")
    print("=" * 60)
    for c in checks:
        pair = f"{c['base']} -> {c['torch']}"
        if not c.get("present", False):
            print(f"[{'SKIP' if args.allow_missing_pairs else 'FAIL'}] {pair}: missing data")
            continue
        metrics = c["metrics"]
        print(
            f"[{'PASS' if c['pass'] else 'FAIL'}] {pair} | "
            f"food_drop={metrics['food_drop_ratio']:.2%} | "
            f"death_increase={metrics['death_increase_ratio']:.2%}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary -> {out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
