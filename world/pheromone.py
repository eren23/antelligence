"""Pheromone grid engine — 4 channels with diffusion, evaporation, and bilinear sampling."""

from __future__ import annotations

import enum
import math

import numpy as np

from config import PheromoneConfig, PheromoneChannelConfig


class Channel(enum.IntEnum):
    """Pheromone channel identifiers (used as grid axis-0 indices)."""

    FOOD = 0
    HOME = 1
    DANGER = 2
    RECRUIT = 3


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalized 3-element symmetric 1D Gaussian kernel."""
    if sigma <= 0.0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    w = math.exp(-0.5 / (sigma * sigma))
    k = np.array([w, 1.0, w], dtype=np.float32)
    k /= k.sum()
    return k


class PheromoneGrid:
    """Discrete 2D pheromone grid with 4 channels.

    Grid shape is ``(4, rows, cols)`` where ``rows = world_height // cell_size``
    and ``cols = world_width // cell_size``.

    Each tick applies separable 3x3 Gaussian diffusion (zero-padded boundaries)
    followed by per-channel exponential decay.
    """

    __slots__ = ("cell_size", "cols", "rows", "grid", "_decay", "_k0", "_k1", "_tmp")

    def __init__(
        self,
        world_width: int,
        world_height: int,
        cfg: PheromoneConfig | None = None,
    ) -> None:
        if cfg is None:
            cfg = PheromoneConfig()
        self.cell_size: int = cfg.cell_size
        self.cols: int = world_width // self.cell_size
        self.rows: int = world_height // self.cell_size

        n = len(Channel)
        self.grid = np.zeros((n, self.rows, self.cols), dtype=np.float32)

        decay = np.empty(n, dtype=np.float32)
        k0 = np.empty(n, dtype=np.float32)
        k1 = np.empty(n, dtype=np.float32)
        for ch in Channel:
            ch_cfg = cfg.channels.get(ch.name.lower(), PheromoneChannelConfig())
            decay[ch] = ch_cfg.decay
            kern = _gaussian_kernel_1d(ch_cfg.diffusion_sigma)
            k0[ch] = kern[0]  # == kern[2] (symmetric)
            k1[ch] = kern[1]

        self._decay = decay.reshape(n, 1, 1)
        self._k0 = k0.reshape(n, 1, 1)
        self._k1 = k1.reshape(n, 1, 1)
        self._tmp = np.empty_like(self.grid)

    # ------------------------------------------------------------------
    # Deposit
    # ------------------------------------------------------------------

    def deposit(
        self, world_x: float, world_y: float, channel: Channel, amount: float
    ) -> None:
        """Add *amount* pheromone at world position. Additive, clamped to 1.0."""
        col = int(world_x // self.cell_size)
        row = int(world_y // self.cell_size)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.grid[channel, row, col] = min(
                self.grid[channel, row, col] + amount, 1.0
            )

    # ------------------------------------------------------------------
    # Bilinear sampling
    # ------------------------------------------------------------------

    def sample(self, world_x: float, world_y: float, channel: Channel) -> float:
        """Sample pheromone via bilinear interpolation at world coordinates.

        Cell centres are at ``((col + 0.5) * cell_size, (row + 0.5) * cell_size)``.
        Out-of-bounds coordinates are clamped to the nearest edge cell.
        """
        cx = world_x / self.cell_size - 0.5
        cy = world_y / self.cell_size - 0.5
        ix = int(math.floor(cx))
        iy = int(math.floor(cy))
        fx = cx - ix
        fy = cy - iy

        g = self.grid[channel]
        mr, mc = self.rows - 1, self.cols - 1

        r0 = min(max(iy, 0), mr)
        r1 = min(max(iy + 1, 0), mr)
        c0 = min(max(ix, 0), mc)
        c1 = min(max(ix + 1, 0), mc)

        return float(
            (1.0 - fx) * (1.0 - fy) * g[r0, c0]
            + fx * (1.0 - fy) * g[r0, c1]
            + (1.0 - fx) * fy * g[r1, c0]
            + fx * fy * g[r1, c1]
        )

    def sample_all(self, world_x: float, world_y: float) -> dict[str, float]:
        """Sample all channels, returning ``{channel_name: value}``."""
        return {ch.name.lower(): self.sample(world_x, world_y, ch) for ch in Channel}

    # ------------------------------------------------------------------
    # Tick (diffusion + evaporation)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance one step: diffuse then evaporate all channels."""
        self._diffuse()
        self._evaporate()

    def _diffuse(self) -> None:
        """Separable 3x3 Gaussian blur across all channels (vectorised)."""
        if self.cols < 2 or self.rows < 2:
            return
        g = self.grid
        t = self._tmp
        k0, k1 = self._k0, self._k1

        # Horizontal pass  (axis=2) → t
        if self.cols >= 3:
            t[:, :, 1:-1] = k0 * (g[:, :, :-2] + g[:, :, 2:]) + k1 * g[:, :, 1:-1]
        t[:, :, :1] = k1 * g[:, :, :1] + k0 * g[:, :, 1:2]
        t[:, :, -1:] = k0 * g[:, :, -2:-1] + k1 * g[:, :, -1:]

        # Vertical pass   (axis=1) → g
        if self.rows >= 3:
            g[:, 1:-1, :] = k0 * (t[:, :-2, :] + t[:, 2:, :]) + k1 * t[:, 1:-1, :]
        g[:, :1, :] = k1 * t[:, :1, :] + k0 * t[:, 1:2, :]
        g[:, -1:, :] = k0 * t[:, -2:-1, :] + k1 * t[:, -1:, :]

    def _evaporate(self) -> None:
        """Per-channel exponential decay (broadcast multiply)."""
        self.grid *= self._decay

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self, channel: Channel | None = None) -> None:
        """Zero out one or all channels."""
        if channel is not None:
            self.grid[channel] = 0.0
        else:
            self.grid[:] = 0.0
