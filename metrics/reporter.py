"""JSON report generation for headless simulation mode."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metrics.emergence import EmergenceDetector
    from metrics.tracker import MetricsTracker, TickSnapshot


def _snapshot_to_dict(snap: TickSnapshot) -> dict[str, Any]:
    """Convert a TickSnapshot to a JSON-serializable dict."""
    return {
        "tick": snap.tick,
        "food_income": round(snap.food_income, 4),
        "population": snap.population,
        "role_distribution": snap.role_distribution,
        "pheromone_mass": round(snap.pheromone_mass, 4),
        "trail_entropy": round(snap.trail_entropy, 4),
        "avg_foraging_trip": round(snap.avg_foraging_trip, 2),
        "deaths": snap.deaths,
        "avg_reward": round(snap.avg_reward, 6),
    }


def _compute_summary(snapshots: list[TickSnapshot]) -> dict[str, Any]:
    """Compute aggregate summary statistics from all snapshots."""
    if not snapshots:
        return {}

    n = len(snapshots)
    total_food = sum(s.food_income for s in snapshots)
    total_deaths = sum(s.deaths for s in snapshots)
    avg_pop = sum(s.population for s in snapshots) / n
    avg_pheromone = sum(s.pheromone_mass for s in snapshots) / n
    avg_entropy = sum(s.trail_entropy for s in snapshots) / n
    avg_reward = sum(s.avg_reward for s in snapshots) / n

    # Peak and final values
    peak_pop = max(s.population for s in snapshots)
    final_pop = snapshots[-1].population
    final_food_income = snapshots[-1].food_income

    # Foraging trip stats (from non-zero entries)
    trip_vals = [s.avg_foraging_trip for s in snapshots if s.avg_foraging_trip > 0]
    avg_trip = sum(trip_vals) / len(trip_vals) if trip_vals else 0.0

    return {
        "total_ticks": snapshots[-1].tick,
        "total_food_collected": round(total_food, 2),
        "total_deaths": total_deaths,
        "avg_population": round(avg_pop, 1),
        "peak_population": peak_pop,
        "final_population": final_pop,
        "avg_pheromone_mass": round(avg_pheromone, 2),
        "avg_trail_entropy": round(avg_entropy, 4),
        "avg_foraging_trip_duration": round(avg_trip, 2),
        "avg_reward": round(avg_reward, 6),
        "final_food_income": round(final_food_income, 4),
    }


def generate_report(
    tracker: MetricsTracker,
    emergence: EmergenceDetector | None = None,
    *,
    include_time_series: bool = True,
    downsample: int = 1,
) -> dict[str, Any]:
    """Build a complete JSON-serializable report dict.

    Args:
        tracker: The MetricsTracker with recorded snapshots.
        emergence: Optional EmergenceDetector for behavior scores.
        include_time_series: If True, include per-tick data.
        downsample: Include every Nth snapshot in time series (1 = all).

    Returns:
        A dict ready for ``json.dumps()``.
    """
    snapshots = tracker.snapshots
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": _compute_summary(snapshots),
    }

    if emergence is not None:
        report["emergence"] = emergence.report.as_dict()

    if include_time_series and snapshots:
        step = max(1, downsample)
        report["time_series"] = [
            _snapshot_to_dict(s) for s in snapshots[::step]
        ]

    return report


def write_report(
    path: str | Path,
    tracker: MetricsTracker,
    emergence: EmergenceDetector | None = None,
    *,
    include_time_series: bool = True,
    downsample: int = 1,
    indent: int = 2,
) -> Path:
    """Generate and write the JSON report to *path*.

    Args:
        path: Output file path.
        tracker: The MetricsTracker with recorded snapshots.
        emergence: Optional EmergenceDetector for behavior scores.
        include_time_series: If True, include per-tick data.
        downsample: Include every Nth snapshot in time series.
        indent: JSON indentation level (0 for compact).

    Returns:
        The resolved Path that was written.
    """
    path = Path(path)
    report = generate_report(
        tracker,
        emergence,
        include_time_series=include_time_series,
        downsample=downsample,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=indent if indent > 0 else None)
    return path
