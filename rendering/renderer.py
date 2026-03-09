"""Main Pygame renderer — 60 FPS target, composites world, ants, overlays, HUD."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pygame

from rendering.controls import ControlState, process_events
from rendering.hud import HUD
from rendering.pheromone_overlay import PheromoneOverlay

if TYPE_CHECKING:
    from agents.ant import Ant
    from agents.colony import Colony
    from config import SimConfig
    from world.pheromone import PheromoneGrid
    from world.world import World

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BG_COLOR = (30, 30, 35)

_ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    "forager": (50, 200, 50),
    "nurse": (200, 100, 200),
    "soldier": (200, 50, 50),
    "idle": (150, 150, 150),
}

_CARRYING_TINT: dict[str, tuple[int, int, int]] = {
    "food": (255, 220, 50),
    "brood": (255, 180, 120),
    "corpse": (120, 80, 60),
}

_NEST_COLOR = (80, 60, 40)
_NEST_RING = (160, 120, 60)
_FOOD_COLOR = (30, 200, 30)
_FOOD_DEPLETED_COLOR = (80, 80, 80)
_OBSTACLE_COLOR = (90, 90, 100)
_OBSTACLE_EDGE = (130, 130, 140)
_CORPSE_COLOR = (100, 60, 40)

_TERRAIN_COLORS: dict[str, tuple[int, int, int]] = {
    "grass": (28, 32, 25),
    "mud": (40, 30, 22),
    "sand": (45, 42, 30),
    "water": (20, 25, 45),
}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    """Top-level Pygame renderer.  Create once, call :meth:`render` each frame."""

    __slots__ = (
        "_screen",
        "_clock",
        "_hud",
        "_pheromone_overlay",
        "_world_w",
        "_world_h",
        "_terrain_surface",
        "_recording",
    )

    def __init__(
        self,
        world: World,
        pheromone_grid: PheromoneGrid,
        cfg: SimConfig,
    ) -> None:
        self._world_w = world.width
        self._world_h = world.height

        pygame.init()
        self._screen = pygame.display.set_mode(
            (self._world_w, self._world_h),
        )
        pygame.display.set_caption("Swarm Colony Simulation")
        self._clock = pygame.time.Clock()

        self._hud = HUD()
        self._pheromone_overlay = PheromoneOverlay(
            pheromone_grid, self._world_w, self._world_h,
        )

        self._recording = False

        # Pre-render static terrain background
        self._terrain_surface = self._build_terrain(world)

    # ------------------------------------------------------------------
    # Terrain (rendered once)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_terrain(world: World) -> pygame.Surface:
        """Build a static terrain surface. Samples every 8 pixels for speed."""
        surf = pygame.Surface((world.width, world.height))
        surf.fill(_BG_COLOR)
        step = 8
        for ty in range(0, world.height, step):
            for tx in range(0, world.width, step):
                from agents.ant import Vec2
                terrain = world.get_terrain(Vec2(float(tx), float(ty)))
                color = _TERRAIN_COLORS.get(terrain, _BG_COLOR)
                pygame.draw.rect(surf, color, (tx, ty, step, step))
        return surf

    # ------------------------------------------------------------------
    # Main render call
    # ------------------------------------------------------------------

    def render(
        self,
        world: World,
        colony: Colony,
        pheromone_grid: PheromoneGrid,
        state: ControlState,
        tick: int,
    ) -> None:
        """Draw one frame.  Call at 60 FPS from the main loop."""

        # Process input
        process_events(state, colony, world)

        # Background — terrain
        self._screen.blit(self._terrain_surface, (0, 0))

        # Pheromone overlays (toggleable per channel)
        if any(state.pheromone_visible):
            self._pheromone_overlay.draw_pheromone(
                self._screen, pheromone_grid, state.pheromone_visible,
            )

        # Trail analysis overlay
        if state.show_trail_analysis:
            self._pheromone_overlay.draw_trail_analysis(
                self._screen, pheromone_grid,
            )

        # Density heatmap (replaces individual ant rendering)
        if state.show_density_heatmap:
            self._pheromone_overlay.draw_density(self._screen, colony)
        else:
            # Draw world objects
            self._draw_obstacles(self._screen, world)
            self._draw_food_sources(self._screen, world)
            self._draw_nest(self._screen, world)
            self._draw_corpses(self._screen, colony)
            self._draw_ants(self._screen, colony)

            # Obstacle being drawn by user
            if state.drawing_obstacle and len(state.obstacle_vertices) >= 2:
                pygame.draw.lines(
                    self._screen, (255, 255, 0), False,
                    state.obstacle_vertices, 2,
                )

        # HUD overlays
        stats = colony.stats()

        if state.show_hud:
            self._hud.draw_stats(
                self._screen, stats, state.brain_type,
                pheromone_grid, state.speed_multiplier,
                state.paused, tick,
            )
            self._hud.draw_world_legend(self._screen)
            self._hud.draw_controls_legend(self._screen)

        if state.selected_ant_id is not None:
            self._hud.draw_inspector(self._screen, colony, state.selected_ant_id)

        # Learning visualizations
        self._hud.draw_reward_graph(self._screen)

        if state.show_weight_heatmap:
            self._hud.draw_weight_heatmap(self._screen, colony)

        if state.show_action_distribution:
            self._hud.draw_action_distribution(self._screen, colony)

        if state.show_attention:
            self._hud.draw_attention(
                self._screen, colony, state.selected_ant_id,
            )

        # Flip & tick clock
        pygame.display.flip()
        if not self._recording:
            self._clock.tick(60)

    # ------------------------------------------------------------------
    # Record reward (delegated to HUD)
    # ------------------------------------------------------------------

    def record_reward(self, reward: float) -> None:
        """Pass per-tick average reward to HUD for graphing."""
        self._hud.record_reward(reward)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_nest(self, surface: pygame.Surface, world: World) -> None:
        cx, cy = int(world.nest.center.x), int(world.nest.center.y)
        r = int(world.nest.radius)
        pygame.draw.circle(surface, _NEST_COLOR, (cx, cy), r)
        pygame.draw.circle(surface, _NEST_RING, (cx, cy), r, 2)

    def _draw_food_sources(self, surface: pygame.Surface, world: World) -> None:
        for food in world.food_sources:
            cx, cy = int(food.pos.x), int(food.pos.y)
            r = max(3, int(food.radius))
            if food.depleted:
                pygame.draw.circle(surface, _FOOD_DEPLETED_COLOR, (cx, cy), r, 1)
            else:
                # Alpha proportional to remaining food
                frac = food.amount / max(food.max_amount, 1.0)
                green = int(100 + 155 * frac)
                color = (30, green, 30)
                pygame.draw.circle(surface, color, (cx, cy), r)
                pygame.draw.circle(surface, _FOOD_COLOR, (cx, cy), r, 1)

    def _draw_obstacles(self, surface: pygame.Surface, world: World) -> None:
        for obs in world.obstacles:
            points = [(int(v.x), int(v.y)) for v in obs.vertices]
            if len(points) >= 3:
                pygame.draw.polygon(surface, _OBSTACLE_COLOR, points)
                pygame.draw.polygon(surface, _OBSTACLE_EDGE, points, 1)

    def _draw_corpses(self, surface: pygame.Surface, colony: Colony) -> None:
        for corpse in colony.corpses:
            cx, cy = int(corpse.pos.x), int(corpse.pos.y)
            pygame.draw.circle(surface, _CORPSE_COLOR, (cx, cy), 2)

    def _draw_ants(self, surface: pygame.Surface, colony: Colony) -> None:
        """Draw each ant as an oriented triangle colour-coded by role/carrying."""
        for ant in colony.ants:
            if not ant.alive:
                continue
            self._draw_ant(surface, ant)

    @staticmethod
    def _draw_ant(surface: pygame.Surface, ant: Ant) -> None:
        """Render a single ant as a small oriented triangle."""
        # Colour: carrying item overrides role colour
        if ant.carrying is not None:
            color = _CARRYING_TINT.get(ant.carrying, _ROLE_COLORS["idle"])
        else:
            color = _ROLE_COLORS.get(ant.role.value, (150, 150, 150))

        # Low energy → darken
        if ant.energy < 30:
            dim = max(0.3, ant.energy / 30.0)
            color = (int(color[0] * dim), int(color[1] * dim), int(color[2] * dim))

        # Triangle pointing in heading direction
        size = 4.0
        h = ant.heading
        # Tip (front)
        tip_x = ant.pos.x + math.cos(h) * size
        tip_y = ant.pos.y + math.sin(h) * size
        # Left wing
        left_x = ant.pos.x + math.cos(h + 2.5) * size * 0.6
        left_y = ant.pos.y + math.sin(h + 2.5) * size * 0.6
        # Right wing
        right_x = ant.pos.x + math.cos(h - 2.5) * size * 0.6
        right_y = ant.pos.y + math.sin(h - 2.5) * size * 0.6

        points = [
            (int(tip_x), int(tip_y)),
            (int(left_x), int(left_y)),
            (int(right_x), int(right_y)),
        ]
        pygame.draw.polygon(surface, color, points)

    # ------------------------------------------------------------------
    # Video capture support
    # ------------------------------------------------------------------

    def set_recording(self, recording: bool) -> None:
        """Enable/disable recording mode (skips frame-rate cap)."""
        self._recording = recording

    def capture_frame(self) -> np.ndarray:
        """Return current screen as (H, W, 3) uint8 RGB array."""
        arr = pygame.surfarray.array3d(self._screen)  # (W, H, 3)
        return arr.transpose(1, 0, 2)  # → (H, W, 3)

    def draw_text_overlay(self, text: str, y: int = 20) -> None:
        """Draw annotation text on the screen surface."""
        font = pygame.font.SysFont("monospace", 18, bold=True)
        shadow = font.render(text, True, (0, 0, 0))
        surface = font.render(text, True, (255, 255, 255))
        self._screen.blit(shadow, (12, y + 1))
        self._screen.blit(surface, (10, y))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def quit(self) -> None:
        """Shut down Pygame."""
        pygame.quit()

    @property
    def screen(self) -> pygame.Surface:
        return self._screen
