"""HUD overlays — stats panel, ant inspector, reward graph, weight heatmap,
action distribution histogram, and attention visualization."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pygame

if TYPE_CHECKING:
    from agents.ant import Ant
    from agents.colony import Colony, ColonyStats
    from rendering.controls import ControlState
    from world.pheromone import PheromoneGrid

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_BG = (20, 20, 30, 200)
_BG_SOLID = (20, 20, 30)
_TEXT = (220, 220, 220)
_TEXT_DIM = (140, 140, 160)
_ACCENT = (80, 200, 255)
_BAR_COLORS: dict[str, tuple[int, int, int]] = {
    "forager": (50, 200, 50),
    "nurse": (200, 100, 200),
    "soldier": (200, 50, 50),
    "idle": (150, 150, 150),
}
_GRAPH_LINE = (80, 255, 120)
_GRAPH_BG = (30, 30, 40)

# ---------------------------------------------------------------------------
# HUD class
# ---------------------------------------------------------------------------


class HUD:
    """Heads-up display: colony stats, ant inspector, learning visualizations."""

    __slots__ = (
        "_font",
        "_font_sm",
        "_font_lg",
        "_reward_history",
        "_food_income_history",
        "_last_food_stored",
    )

    def __init__(self) -> None:
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 14)
        self._font_sm = pygame.font.SysFont("monospace", 11)
        self._font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self._reward_history: deque[float] = deque(maxlen=200)
        self._food_income_history: deque[float] = deque(maxlen=200)
        self._last_food_stored: float = 0.0

    # ------------------------------------------------------------------
    # Main stats overlay (top-left)
    # ------------------------------------------------------------------

    def draw_stats(
        self,
        target: pygame.Surface,
        stats: ColonyStats,
        brain_type: str,
        pheromone_grid: PheromoneGrid,
        speed_mult: float,
        paused: bool,
        tick: int,
    ) -> None:
        """Draw the colony stats panel in the top-left corner."""
        # Track food income
        income = stats.food_stored - self._last_food_stored
        self._last_food_stored = stats.food_stored
        self._food_income_history.append(max(income, 0.0))
        avg_income = (
            sum(self._food_income_history) / len(self._food_income_history)
            if self._food_income_history
            else 0.0
        )

        panel_w, panel_h = 260, 280
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(_BG)

        x, y = 8, 6
        lh = 17  # line height

        # Title
        self._text(panel, "COLONY STATUS", x, y, self._font_lg, _ACCENT)
        y += lh + 6

        # Status line
        status = "PAUSED" if paused else f"{speed_mult:.1f}x"
        self._text(panel, f"Tick: {tick:,}  [{status}]", x, y, self._font, _TEXT)
        y += lh

        self._text(panel, f"Brain: {brain_type}", x, y, self._font, _TEXT)
        y += lh

        self._text(panel, f"Population: {stats.population}  (died: {stats.dead_count})", x, y, self._font, _TEXT)
        y += lh

        self._text(panel, f"Food stored: {stats.food_stored:.1f}", x, y, self._font, _TEXT)
        y += lh

        self._text(panel, f"Income rate: {avg_income:.3f}/tick", x, y, self._font, _TEXT)
        y += lh

        self._text(panel, f"Avg energy: {stats.avg_energy:.1f}", x, y, self._font, _TEXT)
        y += lh + 4

        # Pheromone mass
        masses = pheromone_grid.grid.sum(axis=(1, 2))  # per channel
        ch_names = ["Food", "Home", "Danger", "Recruit"]
        self._text(panel, "Pheromone mass:", x, y, self._font, _TEXT_DIM)
        y += lh
        for i, name in enumerate(ch_names):
            self._text(panel, f"  {name}: {masses[i]:.1f}", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 3

        y += 6

        # Role distribution bar chart
        self._text(panel, "Roles:", x, y, self._font, _TEXT_DIM)
        y += lh
        bar_x = x + 4
        bar_w = panel_w - 20
        bar_h = 14
        total = max(stats.population, 1)
        for role_name in ("forager", "nurse", "soldier", "idle"):
            count = stats.role_counts.get(role_name, 0)
            frac = count / total
            color = _BAR_COLORS.get(role_name, (100, 100, 100))
            w = int(frac * bar_w)
            pygame.draw.rect(panel, color, (bar_x, y, max(w, 1), bar_h))
            label = f"{role_name}: {count} ({frac:.0%})"
            self._text(panel, label, bar_x + 2, y + 1, self._font_sm, (255, 255, 255))
            y += bar_h + 2

        target.blit(panel, (6, 6))

    # ------------------------------------------------------------------
    # Ant inspector panel (right side)
    # ------------------------------------------------------------------

    def draw_inspector(
        self,
        target: pygame.Surface,
        colony: Colony,
        selected_id: int,
    ) -> None:
        """Draw detailed ant inspector panel on the right side of the screen."""
        ant = _find_ant(colony, selected_id)
        if ant is None:
            return

        panel_w = 280
        panel_h = 400
        sw = target.get_width()
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(_BG)

        x, y = 8, 6
        lh = 16

        self._text(panel, f"ANT #{ant.id}", x, y, self._font_lg, _ACCENT)
        y += lh + 4

        self._text(panel, f"Pos: ({ant.pos.x:.1f}, {ant.pos.y:.1f})", x, y, self._font, _TEXT)
        y += lh
        heading_deg = math.degrees(ant.heading)
        self._text(panel, f"Heading: {heading_deg:.1f} deg", x, y, self._font, _TEXT)
        y += lh
        self._text(panel, f"Speed: {ant.speed:.2f}", x, y, self._font, _TEXT)
        y += lh
        self._text(panel, f"Energy: {ant.energy:.1f}", x, y, self._font, _TEXT)
        y += lh
        self._text(panel, f"Role: {ant.role.value}", x, y, self._font, _TEXT)
        y += lh
        carry_str = f"{ant.carrying} ({ant.carry_amount:.2f})" if ant.carrying else "nothing"
        self._text(panel, f"Carrying: {carry_str}", x, y, self._font, _TEXT)
        y += lh
        self._text(panel, f"Age: {ant.age} ticks", x, y, self._font, _TEXT)
        y += lh
        self._text(panel, f"Alive: {ant.alive}", x, y, self._font, _TEXT)
        y += lh + 6

        # Sensory data (if available)
        si = ant.sensory
        if si is not None:
            self._text(panel, "SENSORY INPUT", x, y, self._font, _ACCENT)
            y += lh + 2

            self._text(panel, f"Nest dir: ({si.nest_direction.x:.2f}, {si.nest_direction.y:.2f})", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 2
            self._text(panel, f"Nest dist: {si.nest_distance:.1f}", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 2
            self._text(panel, f"Ground: {si.ground_type}", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 2
            self._text(panel, f"Neighbors: {len(si.neighbors)}", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 2
            self._text(panel, f"Food grad: ({si.food_gradient.x:.3f}, {si.food_gradient.y:.3f})", x, y, self._font_sm, _TEXT_DIM)
            y += lh

            # Antenna readings
            self._text(panel, "Antenna L:", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 3
            for ch, val in si.antenna_left.items():
                self._text(panel, f"  {ch}: {val:.4f}", x, y, self._font_sm, _TEXT_DIM)
                y += lh - 4
            self._text(panel, "Antenna R:", x, y, self._font_sm, _TEXT_DIM)
            y += lh - 3
            for ch, val in si.antenna_right.items():
                self._text(panel, f"  {ch}: {val:.4f}", x, y, self._font_sm, _TEXT_DIM)
                y += lh - 4

            y += 4
            # Obstacle rays
            rays_str = ", ".join(f"{r:.2f}" for r in si.obstacle_rays[:5])
            self._text(panel, f"Rays: [{rays_str}]", x, y, self._font_sm, _TEXT_DIM)
            y += lh

        # Highlight the selected ant on the world surface
        cx, cy = int(ant.pos.x), int(ant.pos.y)
        pygame.draw.circle(target, _ACCENT, (cx, cy), 12, 2)

        target.blit(panel, (sw - panel_w - 6, 6))

    # ------------------------------------------------------------------
    # Reward graph (live line chart at bottom)
    # ------------------------------------------------------------------

    def record_reward(self, reward: float) -> None:
        """Append a per-tick average reward value for the graph."""
        self._reward_history.append(reward)

    def draw_reward_graph(
        self,
        target: pygame.Surface,
        graph_w: int = 400,
        graph_h: int = 100,
    ) -> None:
        """Draw a rolling reward line chart at the bottom-left."""
        if len(self._reward_history) < 2:
            return

        sw, sh = target.get_size()
        gx = 6
        gy = sh - graph_h - 6

        panel = pygame.Surface((graph_w, graph_h), pygame.SRCALPHA)
        panel.fill((*_GRAPH_BG, 200))

        self._text(panel, "Avg Reward", 4, 2, self._font_sm, _TEXT_DIM)

        data = list(self._reward_history)
        n = len(data)
        min_v = min(data)
        max_v = max(data)
        val_range = max(max_v - min_v, 1e-6)

        # Draw zero line
        if min_v < 0 < max_v:
            zy = int(graph_h - 14 - (0 - min_v) / val_range * (graph_h - 20))
            zy = max(14, min(graph_h - 6, zy))
            pygame.draw.line(panel, (80, 80, 80), (0, zy), (graph_w, zy), 1)

        points: list[tuple[int, int]] = []
        for i, v in enumerate(data):
            px = int(i / max(n - 1, 1) * (graph_w - 8)) + 4
            py = int(graph_h - 6 - (v - min_v) / val_range * (graph_h - 20))
            py = max(14, min(graph_h - 6, py))
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(panel, _GRAPH_LINE, False, points, 2)

        # Labels
        self._text(panel, f"{max_v:.3f}", graph_w - 60, 2, self._font_sm, _TEXT_DIM)
        self._text(panel, f"{min_v:.3f}", graph_w - 60, graph_h - 14, self._font_sm, _TEXT_DIM)

        target.blit(panel, (gx, gy))

    # ------------------------------------------------------------------
    # Weight heatmap (W key)
    # ------------------------------------------------------------------

    def draw_weight_heatmap(
        self,
        target: pygame.Surface,
        colony: Colony,
        heatmap_w: int = 200,
        heatmap_h: int = 150,
    ) -> None:
        """Draw a heatmap of the first-layer weights of the brain (if NN/transformer)."""
        # Try to get weights from first ant's brain
        if not colony.ants:
            return
        brain = colony.ants[0].brain
        if brain is None:
            return

        weights: Optional[np.ndarray] = None
        if hasattr(brain, "weights") and brain.weights:
            weights = brain.weights[0]  # first layer
        elif hasattr(brain, "w_proj"):
            weights = brain.w_proj  # transformer input projection

        if weights is None or not isinstance(weights, np.ndarray):
            return

        sw, sh = target.get_size()
        px = sw - heatmap_w - 6
        py = sh - heatmap_h - 6

        panel = pygame.Surface((heatmap_w, heatmap_h), pygame.SRCALPHA)
        panel.fill((*_GRAPH_BG, 200))
        self._text(panel, "Weights L0", 4, 2, self._font_sm, _TEXT_DIM)

        # Normalise weights for colour mapping
        w = weights.astype(np.float64)
        abs_max = max(np.abs(w).max(), 1e-6)
        normed = w / abs_max  # in [-1, 1]

        # Subsample if too large
        rows, cols = normed.shape[:2] if normed.ndim >= 2 else (1, normed.shape[0])
        if normed.ndim == 1:
            normed = normed.reshape(1, -1)

        display_rows = min(rows, heatmap_h - 20)
        display_cols = min(cols, heatmap_w - 8)
        step_r = max(1, rows // display_rows)
        step_c = max(1, cols // display_cols)
        sub = normed[::step_r, ::step_c]
        dr, dc = sub.shape[:2]

        cell_w = max(1, (heatmap_w - 8) // dc)
        cell_h = max(1, (heatmap_h - 20) // dr)

        for r in range(dr):
            for c in range(dc):
                v = sub[r, c]
                if v > 0:
                    color = (int(v * 255), 50, 50)
                else:
                    color = (50, 50, int(-v * 255))
                rect = (4 + c * cell_w, 16 + r * cell_h, cell_w, cell_h)
                pygame.draw.rect(panel, color, rect)

        target.blit(panel, (px, py))

    # ------------------------------------------------------------------
    # Action distribution histogram (D key)
    # ------------------------------------------------------------------

    def draw_action_distribution(
        self,
        target: pygame.Surface,
        colony: Colony,
        hist_w: int = 300,
        hist_h: int = 140,
    ) -> None:
        """Draw histogram of current tick's action outputs across all ants."""
        if not colony.ants:
            return

        # Collect turn values, speed values, deposit channels
        turns: list[float] = []
        speeds: list[float] = []
        deposit_counts: dict[str, int] = {
            "none": 0, "food": 0, "home": 0, "danger": 0, "recruit": 0,
        }

        for ant in colony.ants:
            si = ant.sensory
            if si is None:
                continue
            # We can infer last action from ant state indirectly;
            # for now just show current heading distribution
            turns.append(ant.heading)
            speeds.append(ant.speed)

        sw, sh = target.get_size()
        px = (sw - hist_w) // 2
        py = sh - hist_h - 6

        panel = pygame.Surface((hist_w, hist_h), pygame.SRCALPHA)
        panel.fill((*_GRAPH_BG, 200))
        self._text(panel, "Action Distribution", 4, 2, self._font_sm, _TEXT_DIM)

        if turns:
            # Heading histogram (16 bins over [-pi, pi])
            n_bins = 16
            bin_w = (hist_w - 16) // n_bins
            counts, _ = np.histogram(turns, bins=n_bins, range=(-math.pi, math.pi))
            max_count = max(counts.max(), 1)
            bar_area_h = hist_h - 30

            for i, c in enumerate(counts):
                bh = int(c / max_count * bar_area_h)
                bx = 8 + i * bin_w
                by = hist_h - 8 - bh
                color = (100, 180, 255)
                pygame.draw.rect(panel, color, (bx, by, bin_w - 1, bh))

            self._text(panel, "Heading distribution", 8, hist_h - 16, self._font_sm, _TEXT_DIM)

        target.blit(panel, (px, py))

    # ------------------------------------------------------------------
    # Attention visualization (V key, transformer only)
    # ------------------------------------------------------------------

    def draw_attention(
        self,
        target: pygame.Surface,
        colony: Colony,
        selected_id: Optional[int],
        vis_w: int = 260,
        vis_h: int = 160,
    ) -> None:
        """Visualize attention weights for a selected ant's transformer brain."""
        if selected_id is None:
            return
        ant = _find_ant(colony, selected_id)
        if ant is None or ant.brain is None:
            return
        if not hasattr(ant.brain, "last_attention"):
            return

        attn = ant.brain.last_attention  # expected shape: (n_heads, ctx_len, ctx_len) or (n_heads, ctx_len)
        if attn is None or not isinstance(attn, np.ndarray):
            return

        sw, sh = target.get_size()
        px = sw - vis_w - 290  # left of inspector panel
        py = 6

        panel = pygame.Surface((vis_w, vis_h), pygame.SRCALPHA)
        panel.fill((*_GRAPH_BG, 200))
        self._text(panel, "Attention Weights", 4, 2, self._font_sm, _ACCENT)

        # If (n_heads, ctx, ctx) show last row's attention per head
        if attn.ndim == 3:
            n_heads = attn.shape[0]
            ctx_len = attn.shape[2]
            head_h = max(1, (vis_h - 24) // n_heads)
            cell_w = max(1, (vis_w - 12) // ctx_len)

            for h in range(n_heads):
                row = attn[h, -1, :]  # last query attending to all keys
                row_max = max(row.max(), 1e-6)
                for t in range(ctx_len):
                    intensity = min(int(row[t] / row_max * 255), 255)
                    color = (intensity, intensity // 2, 255 - intensity)
                    rect = (6 + t * cell_w, 18 + h * head_h, cell_w, head_h - 1)
                    pygame.draw.rect(panel, color, rect)
                self._text(
                    panel, f"H{h}", vis_w - 20, 18 + h * head_h, self._font_sm, _TEXT_DIM,
                )
        elif attn.ndim == 2:
            # (n_heads, ctx_len)
            n_heads, ctx_len = attn.shape
            head_h = max(1, (vis_h - 24) // n_heads)
            cell_w = max(1, (vis_w - 12) // ctx_len)
            for h in range(n_heads):
                row = attn[h]
                row_max = max(row.max(), 1e-6)
                for t in range(ctx_len):
                    intensity = min(int(row[t] / row_max * 255), 255)
                    color = (intensity, intensity // 2, 255 - intensity)
                    rect = (6 + t * cell_w, 18 + h * head_h, cell_w, head_h - 1)
                    pygame.draw.rect(panel, color, rect)

        target.blit(panel, (px, py))

    # ------------------------------------------------------------------
    # World legend (bottom-left, above reward graph)
    # ------------------------------------------------------------------

    def draw_world_legend(self, target: pygame.Surface) -> None:
        """Draw a legend identifying visual elements on-screen."""
        entries: list[tuple[str, tuple[int, int, int], str]] = [
            # (shape, color, label)
            ("circle", (30, 200, 30), "Food source"),
            ("circle", (80, 60, 40), "Nest"),
            ("rect", (90, 90, 100), "Obstacle"),
            ("tri", (50, 200, 50), "Forager"),
            ("tri", (200, 100, 200), "Nurse"),
            ("tri", (200, 50, 50), "Soldier"),
            ("tri", (150, 150, 150), "Idle"),
            ("tri", (255, 220, 50), "Carrying food"),
            ("dot", (100, 60, 40), "Corpse"),
        ]

        lh = 14
        panel_w = 150
        panel_h = len(entries) * lh + 22

        sw, sh = target.get_size()
        # Position above the reward graph (graph is at y = sh - 106)
        px = 6
        py = sh - 106 - panel_h - 6

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 180))

        self._text(panel, "LEGEND", 6, 3, self._font_sm, _ACCENT)
        y = 18

        for shape, color, label in entries:
            cx, cy = 14, y + lh // 2
            if shape == "circle":
                pygame.draw.circle(panel, color, (cx, cy), 5)
                pygame.draw.circle(panel, (200, 200, 200), (cx, cy), 5, 1)
            elif shape == "rect":
                pygame.draw.rect(panel, color, (cx - 5, cy - 4, 10, 8))
                pygame.draw.rect(panel, (130, 130, 140), (cx - 5, cy - 4, 10, 8), 1)
            elif shape == "tri":
                pts = [(cx + 5, cy), (cx - 4, cy - 4), (cx - 4, cy + 4)]
                pygame.draw.polygon(panel, color, pts)
            elif shape == "dot":
                pygame.draw.circle(panel, color, (cx, cy), 3)
            self._text(panel, label, 26, y, self._font_sm, _TEXT)
            y += lh

        target.blit(panel, (px, py))

    # ------------------------------------------------------------------
    # Controls legend (bottom-right corner)
    # ------------------------------------------------------------------

    def draw_controls_legend(self, target: pygame.Surface) -> None:
        """Small overlay listing key bindings."""
        lines = [
            "Space: Pause   +/-: Speed",
            "1-4: Pheromone  0: All off",
            "R/N/T: Brain  S: Stats  H: Density",
            "W: Weights  D: Actions  V: Attention",
            "A: Trails  K: Kill  F: Food crisis",
            "LClick: Inspect  RDrag: Obstacle",
            "MClick: Food  Shift+R: Remove obs",
        ]
        lh = 14
        panel_w = 290
        panel_h = len(lines) * lh + 8

        sw, sh = target.get_size()
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 160))

        y = 4
        for line in lines:
            self._text(panel, line, 6, y, self._font_sm, _TEXT_DIM)
            y += lh

        target.blit(panel, (sw - panel_w - 6, sh - panel_h - 6))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _text(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, ...],
    ) -> None:
        rendered = font.render(text, True, color[:3])
        surface.blit(rendered, (x, y))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_ant(colony: Colony, ant_id: int) -> Optional[Ant]:
    """Find an ant by id in the colony."""
    for ant in colony.ants:
        if ant.id == ant_id:
            return ant
    return None
