"""Input handling — keyboard shortcuts, mouse interactions, simulation controls."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import pygame

if TYPE_CHECKING:
    from agents.ant import Ant, Vec2
    from agents.colony import Colony
    from world.world import World
    from world.obstacle import Obstacle

# ---------------------------------------------------------------------------
# Simulation speed presets
# ---------------------------------------------------------------------------

SPEED_PRESETS: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# ControlState — all mutable UI / interaction state
# ---------------------------------------------------------------------------

@dataclass
class ControlState:
    """Mutable state driven by user input, consumed by renderer and sim loop."""

    # Simulation
    paused: bool = False
    speed_index: int = 1                          # index into SPEED_PRESETS
    brain_type: str = "rule_based"                # "rule_based" | "nn" | "transformer" | "torch_nn" | "torch_transformer"
    quit_requested: bool = False

    # Pheromone overlay toggles (one per channel, index 0-3)
    pheromone_visible: list[bool] = field(
        default_factory=lambda: [False, False, False, False],
    )

    # Overlay modes
    show_hud: bool = True
    show_density_heatmap: bool = False            # H key
    show_trail_analysis: bool = False             # A key
    show_weight_heatmap: bool = False             # W key
    show_action_distribution: bool = False        # D key
    show_attention: bool = False                  # V key (transformer only)

    # Inspector
    selected_ant_id: Optional[int] = None

    # Obstacle drawing (right-drag)
    drawing_obstacle: bool = False
    obstacle_vertices: list[tuple[float, float]] = field(default_factory=list)

    @property
    def speed_multiplier(self) -> float:
        return SPEED_PRESETS[self.speed_index]


# ---------------------------------------------------------------------------
# Event processing
# ---------------------------------------------------------------------------

def _find_ant_at(
    pos: tuple[int, int],
    colony: Colony,
    click_radius: float = 8.0,
) -> Optional[int]:
    """Return the id of the ant closest to screen *pos*, or None."""
    from agents.ant import Vec2

    mx, my = pos
    best_id: Optional[int] = None
    best_dist_sq = click_radius * click_radius
    for ant in colony.ants:
        dx = ant.pos.x - mx
        dy = ant.pos.y - my
        d2 = dx * dx + dy * dy
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_id = ant.id
    return best_id


def process_events(
    state: ControlState,
    colony: Colony,
    world: World,
) -> None:
    """Poll all Pygame events and update *state* / *world* / *colony* accordingly."""
    from agents.ant import Vec2
    from world.food import FoodSource
    from world.obstacle import Obstacle

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state.quit_requested = True
            return

        # ---- Key presses ----
        if event.type == pygame.KEYDOWN:
            _handle_keydown(event, state, colony, world)

        # ---- Mouse ----
        elif event.type == pygame.MOUSEBUTTONDOWN:
            _handle_mouse_down(event, state, colony, world)

        elif event.type == pygame.MOUSEBUTTONUP:
            _handle_mouse_up(event, state, world)

        elif event.type == pygame.MOUSEMOTION:
            _handle_mouse_motion(event, state)


# ---------------------------------------------------------------------------
# Key handlers
# ---------------------------------------------------------------------------

def _handle_keydown(
    event: pygame.event.Event,
    state: ControlState,
    colony: Colony,
    world: World,
) -> None:
    key = event.key
    mods = pygame.key.get_mods()

    # Pause / resume
    if key == pygame.K_SPACE:
        state.paused = not state.paused

    # Speed control
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        state.speed_index = min(state.speed_index + 1, len(SPEED_PRESETS) - 1)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        state.speed_index = max(state.speed_index - 1, 0)

    # Pheromone overlays  (1-4 toggle individual, 0 = all off)
    elif key == pygame.K_1:
        state.pheromone_visible[0] = not state.pheromone_visible[0]
    elif key == pygame.K_2:
        state.pheromone_visible[1] = not state.pheromone_visible[1]
    elif key == pygame.K_3:
        state.pheromone_visible[2] = not state.pheromone_visible[2]
    elif key == pygame.K_4:
        state.pheromone_visible[3] = not state.pheromone_visible[3]
    elif key == pygame.K_0:
        state.pheromone_visible = [False, False, False, False]

    # Brain switch
    elif key == pygame.K_r:
        state.brain_type = "rule_based"
    elif key == pygame.K_n:
        state.brain_type = "nn"
    elif key == pygame.K_t:
        if mods & pygame.KMOD_SHIFT:
            state.brain_type = "torch_transformer"
        else:
            state.brain_type = "transformer"
    elif key == pygame.K_m:
        state.brain_type = "torch_nn"

    # Overlays
    elif key == pygame.K_s:
        state.show_hud = not state.show_hud
    elif key == pygame.K_h:
        state.show_density_heatmap = not state.show_density_heatmap
    elif key == pygame.K_a:
        state.show_trail_analysis = not state.show_trail_analysis
    elif key == pygame.K_w:
        state.show_weight_heatmap = not state.show_weight_heatmap
    elif key == pygame.K_d:
        state.show_action_distribution = not state.show_action_distribution
    elif key == pygame.K_v:
        state.show_attention = not state.show_attention

    # Kill 10 % of colony
    elif key == pygame.K_k:
        _kill_random_ants(colony, fraction=0.10)

    # Food crisis — remove all food sources
    elif key == pygame.K_f:
        world.food_sources.clear()

    # Deselect ant
    elif key == pygame.K_ESCAPE:
        state.selected_ant_id = None


# ---------------------------------------------------------------------------
# Mouse handlers
# ---------------------------------------------------------------------------

def _handle_mouse_down(
    event: pygame.event.Event,
    state: ControlState,
    colony: Colony,
    world: World,
) -> None:
    from agents.ant import Vec2
    from world.food import FoodSource
    from world.obstacle import Obstacle

    mods = pygame.key.get_mods()
    pos = event.pos  # (x, y) in screen / world coords (1:1 mapping)

    # Left-click → select / deselect ant
    if event.button == 1:
        ant_id = _find_ant_at(pos, colony)
        state.selected_ant_id = ant_id  # None deselects

    # Middle-click → place food source (500 units, radius 25)
    elif event.button == 2:
        fs = FoodSource(
            pos=Vec2(float(pos[0]), float(pos[1])),
            radius=25.0,
            amount=500.0,
            max_amount=500.0,
        )
        world.food_sources.append(fs)

    # Right-click
    elif event.button == 3:
        if mods & pygame.KMOD_SHIFT:
            # Shift+right-click → remove obstacle under cursor
            _remove_obstacle_at(Vec2(float(pos[0]), float(pos[1])), world)
        else:
            # Right-click drag → start drawing obstacle
            state.drawing_obstacle = True
            state.obstacle_vertices = [pos]


def _handle_mouse_up(
    event: pygame.event.Event,
    state: ControlState,
    world: World,
) -> None:
    from agents.ant import Vec2
    from world.obstacle import Obstacle

    if event.button == 3 and state.drawing_obstacle:
        state.drawing_obstacle = False
        verts = state.obstacle_vertices
        if len(verts) >= 3:
            poly_verts = [Vec2(float(x), float(y)) for x, y in verts]
            try:
                obs = Obstacle(vertices=poly_verts)
                world.obstacles.append(obs)
            except ValueError:
                pass  # degenerate polygon
        state.obstacle_vertices = []


def _handle_mouse_motion(
    event: pygame.event.Event,
    state: ControlState,
) -> None:
    if state.drawing_obstacle:
        # Add vertex if far enough from last point (>10 px)
        last = state.obstacle_vertices[-1]
        dx = event.pos[0] - last[0]
        dy = event.pos[1] - last[1]
        if dx * dx + dy * dy > 100:  # 10 px squared
            state.obstacle_vertices.append(event.pos)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kill_random_ants(colony: Colony, fraction: float = 0.10) -> None:
    """Kill a random fraction of the colony."""
    import random

    ants = colony.ants
    kill_count = max(1, int(len(ants) * fraction))
    victims = random.sample(ants, min(kill_count, len(ants)))
    for ant in victims:
        ant.alive = False
        ant.energy = 0.0


def _remove_obstacle_at(pos: Vec2, world: World) -> None:
    """Remove the first obstacle containing *pos*."""
    for i, obs in enumerate(world.obstacles):
        if obs.contains_point(pos):
            world.obstacles.pop(i)
            return
