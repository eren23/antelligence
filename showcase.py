"""Showcase video script — records a demo as MP4 via ffmpeg pipe.

Usage:
    python3 showcase.py --load-weights weights/best/ --brain nn --output showcase.mp4
    python3 showcase.py --load-weights weights/best/ --brain nn --ticks 3500 --no-annotations
    python3 showcase.py --brain rule_based --ticks 9000 --freeform --output ants_long.mp4
"""

from __future__ import annotations

import argparse
import random
import signal
import subprocess
import sys
import time as _time
from pathlib import Path

import numpy as np
import pygame

from agents.ant import Vec2
from agents.colony import Colony
from config import SimConfig, default_config, load_config
from main import BrainManager, sim_tick
from metrics.emergence import EmergenceDetector
from metrics.tracker import MetricsTracker
from rendering.controls import ControlState
from rendering.renderer import Renderer
from world.food import FoodSource
from world.obstacle import Obstacle
from world.pheromone import PheromoneGrid
from world.world import World


# ---------------------------------------------------------------------------
# Choreographed events
# ---------------------------------------------------------------------------

def _default_events(world: World, colony: Colony) -> list[dict]:
    """Return the default choreography event list."""
    cx = world.width / 2
    cy = world.height / 2
    return [
        {"tick": 500, "action": "pheromone_overlay", "channel": 0,
         "annotation": "Enabling food pheromone overlay"},
        {"tick": 800, "action": "add_food", "positions": [(cx - 200, cy - 100), (cx + 200, cy + 100)],
         "annotation": "Placing two new food sources"},
        {"tick": 1200, "action": "add_obstacle",
         "vertices": [(cx - 150, cy - 30), (cx + 150, cy - 30),
                      (cx + 150, cy + 10), (cx - 150, cy + 10)],
         "annotation": "Obstacle wall blocks trail"},
        {"tick": 1500, "action": "density_heatmap", "enable": True,
         "annotation": "Density heatmap view"},
        {"tick": 1650, "action": "density_heatmap", "enable": False,
         "annotation": ""},
        {"tick": 1800, "action": "food_crisis",
         "annotation": "Food crisis triggered — all sources removed"},
        {"tick": 2200, "action": "kill_ants", "fraction": 0.15,
         "annotation": "15% population killed"},
        {"tick": 2600, "action": "remove_obstacles",
         "annotation": "Obstacles removed — watch re-routing"},
        {"tick": 3000, "action": "pheromone_overlay", "channel": 1,
         "annotation": "Switching to home pheromone overlay"},
    ]


def _execute_event(
    event: dict,
    state: ControlState,
    world: World,
    colony: Colony,
) -> None:
    """Execute a single choreography event."""
    action = event["action"]

    if action == "pheromone_overlay":
        # Reset all, enable specified channel
        state.pheromone_visible = [False, False, False, False]
        ch = event.get("channel")
        if ch is not None and 0 <= ch < 4:
            state.pheromone_visible[ch] = True

    elif action == "add_food":
        for px, py in event.get("positions", []):
            world.food_sources.append(FoodSource(
                pos=Vec2(float(px), float(py)),
                radius=25.0, amount=500.0, max_amount=500.0,
            ))

    elif action == "add_obstacle":
        verts = event.get("vertices", [])
        if len(verts) >= 3:
            poly = [Vec2(float(x), float(y)) for x, y in verts]
            world.obstacles.append(Obstacle(vertices=poly))

    elif action == "food_crisis":
        world.food_sources.clear()

    elif action == "kill_ants":
        fraction = event.get("fraction", 0.10)
        kill_count = max(1, int(len(colony.ants) * fraction))
        victims = random.sample(colony.ants, min(kill_count, len(colony.ants)))
        for ant in victims:
            ant.alive = False
            ant.energy = 0.0

    elif action == "remove_obstacles":
        world.obstacles.clear()

    elif action == "density_heatmap":
        state.show_density_heatmap = event.get("enable", False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record showcase demo video")
    p.add_argument("--load-weights", type=str, default=None,
                   help="Directory with trained weights")
    p.add_argument("--brain", choices=["torch_nn", "torch_transformer", "rule_based"],
                   default="torch_nn", help="Brain backend (default: torch_nn)")
    p.add_argument("--output", "-o", type=str, default="showcase.mp4",
                   help="Output video path (default: showcase.mp4)")
    p.add_argument("--fps", type=int, default=30,
                   help="Output video FPS (default: 30)")
    p.add_argument("--ticks", "-t", type=int, default=3500,
                   help="Total ticks to record (default: 3500)")
    p.add_argument("--seed", "-s", type=int, default=42,
                   help="World seed (default: 42)")
    p.add_argument("--ants", type=int, default=300,
                   help="Initial population (default: 300)")
    p.add_argument("--config", "-c", default="colony_config.yaml",
                   help="YAML config path")
    p.add_argument("--no-annotations", action="store_true",
                   help="Disable text overlays on video")
    p.add_argument("--freeform", action="store_true",
                   help="Skip choreographed events, just record the colony running")
    p.add_argument("--learn", action="store_true",
                   help="Enable brain training during recording (off by default)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    cfg.colony.initial_population = args.ants

    # Initialise simulation
    brain_mgr = BrainManager(cfg, seed)
    world = World.from_config(cfg, seed=seed)
    pheromone_grid = PheromoneGrid(cfg.world.width, cfg.world.height, cfg.pheromone)
    colony = Colony(cfg, world.nest.center, seed=seed)

    for ant in colony.ants:
        brain_mgr.create_brain(ant)

    # Load trained weights
    if args.load_weights:
        brain_mgr.load_weights(Path(args.load_weights))

    metrics = MetricsTracker(window=10_000)
    emergence = EmergenceDetector(cfg.roles.default_distribution)
    control = ControlState(brain_type=args.brain)

    # Enable all visual overlays for showcase recordings
    control.pheromone_visible = [True, True, True, True]
    control.show_trail_analysis = True
    control.show_hud = True
    control.show_density_heatmap = False  # too noisy layered with pheromones
    if args.brain == "torch_transformer":
        control.show_attention = True

    # Renderer (needed for visual output)
    renderer = Renderer(world, pheromone_grid, cfg)
    renderer.set_recording(True)

    width, height = cfg.world.width, cfg.world.height

    # FFmpeg pipe for video encoding
    output_path = args.output
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium",
        "-crf", "23", "-pix_fmt", "yuv420p",
        output_path,
    ]

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install it with: brew install ffmpeg")
        renderer.quit()
        sys.exit(1)

    # Choreography
    events = [] if args.freeform else _default_events(world, colony)
    event_map: dict[int, dict] = {e["tick"]: e for e in events}
    current_annotation = ""
    annotation_end_tick = 0

    print(f"Recording: {args.ticks} ticks → {output_path} @ {args.fps} FPS")
    print(f"Resolution: {width}x{height}, Brain: {args.brain}")
    print("-" * 50)

    # Graceful shutdown on SIGTERM/SIGINT — finalize video instead of corrupting
    _stop_requested = False

    def _handle_signal(signum, frame):
        nonlocal _stop_requested
        _stop_requested = True
        print(f"\nSignal {signum} received — finishing video…")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    t_start = _time.perf_counter()

    for tick in range(1, args.ticks + 1):
        if _stop_requested:
            print(f"  Stopped early at tick {tick}.")
            break

        # Execute choreography event if scheduled
        if tick in event_map:
            ev = event_map[tick]
            _execute_event(ev, control, world, colony)
            ann = ev.get("annotation", "")
            if ann:
                current_annotation = ann
                annotation_end_tick = tick + 200  # show annotation for 200 ticks
                print(f"  Tick {tick:>5}: {ann}")

        if tick > annotation_end_tick:
            current_annotation = ""

        # Simulation tick
        sim_tick(tick, colony, world, pheromone_grid, brain_mgr, metrics, emergence, cfg,
                 learn=args.learn)

        # Render frame
        snap = metrics.latest
        if snap is not None:
            renderer.record_reward(snap.avg_reward)

        renderer.render(world, colony, pheromone_grid, control, tick)

        # Draw annotations on top of the rendered frame
        if not args.no_annotations and current_annotation:
            renderer.draw_text_overlay(f"Tick {tick} — {current_annotation}")
            pygame.display.flip()

        # Capture and pipe frame
        frame = renderer.capture_frame()
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("Error: ffmpeg pipe broken.")
            break

        # Drain pygame events to prevent freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _stop_requested = True

        # Progress
        if tick % 500 == 0:
            elapsed = _time.perf_counter() - t_start
            fps_actual = tick / elapsed if elapsed > 0 else 0
            print(f"  Progress: {tick:>5}/{args.ticks} ({fps_actual:.0f} frames/s)")

    # Finalize — always close ffmpeg cleanly so the mp4 is valid
    proc.stdin.close()
    proc.wait()

    elapsed = _time.perf_counter() - t_start
    duration = tick / args.fps

    renderer.quit()

    if proc.returncode == 0:
        print("-" * 50)
        print(f"Video saved → {output_path}")
        print(f"Duration: {duration:.1f}s @ {args.fps} FPS | Rendered in {elapsed:.1f}s")
    else:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"ffmpeg exited with code {proc.returncode}")
        if stderr:
            print(stderr[-500:])


if __name__ == "__main__":
    main()
