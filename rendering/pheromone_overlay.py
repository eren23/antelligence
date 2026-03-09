"""Pheromone heatmap overlay and ant density heatmap rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

if TYPE_CHECKING:
    from agents.colony import Colony
    from world.pheromone import PheromoneGrid

# Channel colours (R, G, B) matching config defaults
_CHANNEL_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),      # FOOD   — green
    (0, 100, 255),     # HOME   — blue
    (255, 0, 0),       # DANGER — red
    (255, 200, 0),     # RECRUIT — yellow
]

_CHANNEL_NAMES: list[str] = ["Food", "Home", "Danger", "Recruit"]


class PheromoneOverlay:
    """Renders translucent pheromone heatmaps and ant density heatmaps.

    Uses a pre-allocated RGBA surface that is scaled from grid resolution
    to world resolution once per frame, keeping the hot path in NumPy.
    """

    __slots__ = (
        "_grid_w",
        "_grid_h",
        "_cell_size",
        "_world_w",
        "_world_h",
        "_overlay_surface",
        "_density_surface",
        "_pixel_buf",
    )

    def __init__(self, pheromone_grid: PheromoneGrid, world_w: int, world_h: int) -> None:
        self._grid_w = pheromone_grid.cols
        self._grid_h = pheromone_grid.rows
        self._cell_size = pheromone_grid.cell_size
        self._world_w = world_w
        self._world_h = world_h

        # Small RGBA surface at grid resolution; will be scaled up on blit
        self._overlay_surface = pygame.Surface(
            (self._grid_w, self._grid_h), pygame.SRCALPHA,
        )
        self._density_surface = pygame.Surface(
            (self._grid_w, self._grid_h), pygame.SRCALPHA,
        )
        # Reusable pixel buffer (rows, cols, 4) — RGBA
        self._pixel_buf = np.zeros(
            (self._grid_h, self._grid_w, 4), dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # Pheromone channel overlay
    # ------------------------------------------------------------------

    def draw_pheromone(
        self,
        target: pygame.Surface,
        pheromone_grid: PheromoneGrid,
        visible_channels: list[bool],
        alpha: int = 120,
    ) -> None:
        """Composite enabled pheromone channels onto *target* as a translucent heatmap."""
        buf = self._pixel_buf
        buf[:] = 0  # clear

        any_visible = False
        for ch_idx in range(4):
            if not visible_channels[ch_idx]:
                continue
            any_visible = True
            # grid shape: (4, rows, cols)
            channel_data = pheromone_grid.grid[ch_idx]  # (rows, cols) float32
            intensity = np.clip(channel_data, 0.0, 1.0)

            r, g, b = _CHANNEL_COLORS[ch_idx]
            # Additive blend into buffer (clamp later)
            buf[:, :, 0] = np.clip(
                buf[:, :, 0].astype(np.int16) + (intensity * r).astype(np.int16),
                0, 255,
            ).astype(np.uint8)
            buf[:, :, 1] = np.clip(
                buf[:, :, 1].astype(np.int16) + (intensity * g).astype(np.int16),
                0, 255,
            ).astype(np.uint8)
            buf[:, :, 2] = np.clip(
                buf[:, :, 2].astype(np.int16) + (intensity * b).astype(np.int16),
                0, 255,
            ).astype(np.uint8)
            # Alpha: max of existing and new channel
            new_alpha = (intensity * alpha).astype(np.uint8)
            buf[:, :, 3] = np.maximum(buf[:, :, 3], new_alpha)

        if not any_visible:
            return

        # Transpose to (cols, rows, 4) because pygame pixel arrays are (w, h)
        pixels_transposed = np.transpose(buf, (1, 0, 2))
        pygame.pixelcopy.array_to_surface(self._overlay_surface, pixels_transposed[:, :, :3])
        # Apply alpha channel manually via a per-pixel alpha surface
        alpha_arr = pixels_transposed[:, :, 3]
        alpha_surface = pygame.Surface(
            (self._grid_w, self._grid_h), pygame.SRCALPHA,
        )
        # Blit the RGB surface, then set per-pixel alpha
        alpha_surface.blit(self._overlay_surface, (0, 0))
        pygame.surfarray.pixels_alpha(alpha_surface)[:] = alpha_arr

        # Scale up to world size and blit
        scaled = pygame.transform.scale(alpha_surface, (self._world_w, self._world_h))
        target.blit(scaled, (0, 0))

    # ------------------------------------------------------------------
    # Ant density heatmap
    # ------------------------------------------------------------------

    def draw_density(
        self,
        target: pygame.Surface,
        colony: Colony,
        alpha: int = 150,
    ) -> None:
        """Render an ant density heatmap (hot = many ants in a cell)."""
        # Accumulate ant counts per grid cell
        density = np.zeros((self._grid_h, self._grid_w), dtype=np.float32)
        cs = self._cell_size
        for ant in colony.ants:
            col = int(ant.pos.x // cs)
            row = int(ant.pos.y // cs)
            if 0 <= col < self._grid_w and 0 <= row < self._grid_h:
                density[row, col] += 1.0

        # Normalize to [0, 1] (clamp at ~10 ants per cell for visual range)
        max_val = max(density.max(), 1.0)
        norm = np.clip(density / min(max_val, 10.0), 0.0, 1.0)

        # Colour map: black → blue → cyan → yellow → red (heat)
        buf = self._pixel_buf
        buf[:] = 0
        # Simple 3-stop gradient: 0=black, 0.5=cyan, 1.0=red
        r = np.clip((norm - 0.5) * 2.0, 0.0, 1.0)
        g = np.where(norm < 0.5, norm * 2.0, (1.0 - norm) * 2.0)
        b = np.clip((0.5 - norm) * 2.0, 0.0, 1.0)

        buf[:, :, 0] = (r * 255).astype(np.uint8)
        buf[:, :, 1] = (g * 255).astype(np.uint8)
        buf[:, :, 2] = (b * 255).astype(np.uint8)
        buf[:, :, 3] = (norm * alpha).astype(np.uint8)

        pixels_transposed = np.transpose(buf, (1, 0, 2))
        alpha_surface = pygame.Surface(
            (self._grid_w, self._grid_h), pygame.SRCALPHA,
        )
        rgb_surface = pygame.Surface((self._grid_w, self._grid_h))
        pygame.pixelcopy.array_to_surface(rgb_surface, pixels_transposed[:, :, :3])
        alpha_surface.blit(rgb_surface, (0, 0))
        pygame.surfarray.pixels_alpha(alpha_surface)[:] = pixels_transposed[:, :, 3]

        scaled = pygame.transform.scale(alpha_surface, (self._world_w, self._world_h))
        target.blit(scaled, (0, 0))

    # ------------------------------------------------------------------
    # Trail analysis (top-3 strongest pheromone corridors)
    # ------------------------------------------------------------------

    def draw_trail_analysis(
        self,
        target: pygame.Surface,
        pheromone_grid: PheromoneGrid,
        alpha: int = 180,
    ) -> None:
        """Highlight the strongest FOOD pheromone trails (threshold top ~5%)."""
        food_grid = pheromone_grid.grid[0]  # Channel.FOOD
        if food_grid.max() < 1e-6:
            return

        threshold = np.percentile(food_grid, 95)
        mask = food_grid >= threshold

        buf = np.zeros((self._grid_h, self._grid_w, 4), dtype=np.uint8)
        intensity = np.clip(food_grid / max(food_grid.max(), 1e-6), 0.0, 1.0)
        buf[mask, 0] = 255
        buf[mask, 1] = 255
        buf[mask, 2] = 0
        buf[mask, 3] = (intensity[mask] * alpha).astype(np.uint8)

        pixels_transposed = np.transpose(buf, (1, 0, 2))
        alpha_surface = pygame.Surface(
            (self._grid_w, self._grid_h), pygame.SRCALPHA,
        )
        rgb_surface = pygame.Surface((self._grid_w, self._grid_h))
        pygame.pixelcopy.array_to_surface(rgb_surface, pixels_transposed[:, :, :3])
        alpha_surface.blit(rgb_surface, (0, 0))
        pygame.surfarray.pixels_alpha(alpha_surface)[:] = pixels_transposed[:, :, 3]

        scaled = pygame.transform.scale(alpha_surface, (self._world_w, self._world_h))
        target.blit(scaled, (0, 0))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def channel_names() -> list[str]:
        return list(_CHANNEL_NAMES)

    @staticmethod
    def channel_colors() -> list[tuple[int, int, int]]:
        return list(_CHANNEL_COLORS)
