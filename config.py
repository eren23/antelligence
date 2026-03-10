"""Configuration loader and validator for the colony simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclass hierarchy mirroring colony_config.yaml
# ---------------------------------------------------------------------------

@dataclass
class ColonyConfig:
    initial_population: int = 200
    max_population: int = 500
    spawn_rate: float = 0.1
    initial_food_stored: float = 100


@dataclass
class AntConfig:
    base_speed: float = 2.0
    max_carry: float = 1.0
    energy_max: float = 100
    energy_cost_base: float = 0.02
    energy_cost_speed: float = 0.01
    energy_cost_carry: float = 0.03
    energy_refill_rate: float = 2.0
    antenna_angle: float = 30       # degrees
    antenna_range: float = 40
    neighbor_radius: float = 20
    obstacle_ray_count: int = 5
    obstacle_ray_range: float = 30
    obstacle_ray_arc: float = 120   # degrees

    @property
    def antenna_angle_rad(self) -> float:
        return math.radians(self.antenna_angle)

    @property
    def obstacle_ray_arc_rad(self) -> float:
        return math.radians(self.obstacle_ray_arc)


@dataclass
class PheromoneChannelConfig:
    decay: float = 0.995
    diffusion_sigma: float = 0.5
    color: list[int] = field(default_factory=lambda: [0, 255, 0])


@dataclass
class PheromoneConfig:
    cell_size: int = 4
    channels: dict[str, PheromoneChannelConfig] = field(default_factory=lambda: {
        "food": PheromoneChannelConfig(0.995, 0.5, [0, 255, 0]),
        "home": PheromoneChannelConfig(0.997, 0.5, [0, 100, 255]),
        "danger": PheromoneChannelConfig(0.980, 0.8, [255, 0, 0]),
        "recruit": PheromoneChannelConfig(0.970, 1.0, [255, 200, 0]),
    })


@dataclass
class RolesConfig:
    default_distribution: dict[str, float] = field(default_factory=lambda: {
        "forager": 0.60,
        "nurse": 0.15,
        "soldier": 0.10,
        "idle": 0.15,
    })
    rebalance_interval: int = 500


@dataclass
class NNBrainConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.0001
    gamma: float = 0.95
    buffer_size: int = 200
    update_interval: int = 100


@dataclass
class TransformerBrainConfig:
    context_length: int = 16
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 64
    learning_rate: float = 0.00005
    buffer_size: int = 500
    update_interval: int = 200


@dataclass
class ImitationConfig:
    demo_ticks: int = 5000
    demo_ants: int = 200
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    lr_decay: float = 0.95


@dataclass
class PPOConfig:
    rollout_length: int = 1024
    epochs_per_update: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-4


@dataclass
class PatchConfig:
    survival_homing: bool = True
    auto_pickup: bool = True
    food_drop_guard: bool = True
    auto_deposit_at_nest: bool = True  # keep as physics rule
    learning_brain_patches: bool = False  # if False, patches skip learning brains


@dataclass
class EvolutionConfig:
    population_size: int = 20
    eval_ticks: int = 5000
    elitism: int = 4
    tournament_size: int = 3
    mutation_rate: float = 0.1
    mutation_scale: float = 0.02
    crossover_prob: float = 0.5
    generations: int = 50
    map_elites_dims: list[int] = field(default_factory=lambda: [10, 10])


@dataclass
class BrainConfig:
    default: str = "rule_based"
    nn: NNBrainConfig = field(default_factory=NNBrainConfig)
    transformer: TransformerBrainConfig = field(default_factory=TransformerBrainConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)


@dataclass
class WorldConfig:
    width: int = 1600
    height: int = 1000
    num_food_sources: int = 5
    food_source_size_range: list[int] = field(default_factory=lambda: [20, 50])
    food_source_amount_range: list[int] = field(default_factory=lambda: [50, 500])
    food_respawn_interval: int = 3000
    num_obstacles: int = 10


@dataclass
class SimConfig:
    """Top-level configuration container."""
    colony: ColonyConfig = field(default_factory=ColonyConfig)
    ant: AntConfig = field(default_factory=AntConfig)
    pheromone: PheromoneConfig = field(default_factory=PheromoneConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    world: WorldConfig = field(default_factory=WorldConfig)


# ---------------------------------------------------------------------------
# YAML → dataclass mapping helpers
# ---------------------------------------------------------------------------

def _map_dict(cls: type, data: dict[str, Any] | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if data is None:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in known})


def _parse_pheromone(data: dict[str, Any] | None) -> PheromoneConfig:
    if data is None:
        return PheromoneConfig()
    channels_raw = data.get("channels", {})
    channels = {
        name: _map_dict(PheromoneChannelConfig, ch)
        for name, ch in channels_raw.items()
    }
    return PheromoneConfig(
        cell_size=data.get("cell_size", 4),
        channels=channels if channels else PheromoneConfig().channels,
    )


def _parse_brain(data: dict[str, Any] | None) -> BrainConfig:
    if data is None:
        return BrainConfig()
    return BrainConfig(
        default=data.get("default", "rule_based"),
        nn=_map_dict(NNBrainConfig, data.get("nn")),
        transformer=_map_dict(TransformerBrainConfig, data.get("transformer")),
        imitation=_map_dict(ImitationConfig, data.get("imitation")),
        ppo=_map_dict(PPOConfig, data.get("ppo")),
        patches=_map_dict(PatchConfig, data.get("patches")),
        evolution=_map_dict(EvolutionConfig, data.get("evolution")),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when configuration is invalid."""


def _validate(cfg: SimConfig) -> None:
    """Validate configuration values. Raises ConfigError on failure."""
    if cfg.colony.initial_population < 1:
        raise ConfigError("colony.initial_population must be >= 1")
    if cfg.colony.max_population < cfg.colony.initial_population:
        raise ConfigError("colony.max_population must be >= initial_population")
    if not (0 <= cfg.colony.spawn_rate <= 10):
        raise ConfigError("colony.spawn_rate must be in [0, 10]")

    if cfg.ant.base_speed <= 0:
        raise ConfigError("ant.base_speed must be > 0")
    if cfg.ant.energy_max <= 0:
        raise ConfigError("ant.energy_max must be > 0")
    if cfg.ant.antenna_angle <= 0 or cfg.ant.antenna_angle > 180:
        raise ConfigError("ant.antenna_angle must be in (0, 180]")

    for name, ch in cfg.pheromone.channels.items():
        if not (0 < ch.decay <= 1):
            raise ConfigError(f"pheromone.channels.{name}.decay must be in (0, 1]")
        if ch.diffusion_sigma < 0:
            raise ConfigError(f"pheromone.channels.{name}.diffusion_sigma must be >= 0")
        if len(ch.color) != 3 or not all(0 <= c <= 255 for c in ch.color):
            raise ConfigError(f"pheromone.channels.{name}.color must be [R, G, B] in 0..255")

    dist = cfg.roles.default_distribution
    total = sum(dist.values())
    if abs(total - 1.0) > 0.01:
        raise ConfigError(f"roles.default_distribution must sum to 1.0, got {total:.3f}")

    valid_brains = ("rule_based", "nn", "transformer", "torch_nn", "torch_transformer")
    if cfg.brain.default not in valid_brains:
        raise ConfigError(f"brain.default must be one of {valid_brains}, got {cfg.brain.default!r}")

    if cfg.world.width < 100 or cfg.world.height < 100:
        raise ConfigError("world dimensions must be >= 100")
    sr = cfg.world.food_source_size_range
    if len(sr) != 2 or sr[0] > sr[1]:
        raise ConfigError("world.food_source_size_range must be [min, max] with min <= max")
    ar = cfg.world.food_source_amount_range
    if len(ar) != 2 or ar[0] > ar[1]:
        raise ConfigError("world.food_source_amount_range must be [min, max] with min <= max")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "colony_config.yaml") -> SimConfig:
    """Load and validate a SimConfig from a YAML file.

    Falls back to defaults for any missing section.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    cfg = SimConfig(
        colony=_map_dict(ColonyConfig, raw.get("colony")),
        ant=_map_dict(AntConfig, raw.get("ant")),
        pheromone=_parse_pheromone(raw.get("pheromone")),
        roles=_map_dict(RolesConfig, raw.get("roles")),
        brain=_parse_brain(raw.get("brain")),
        world=_map_dict(WorldConfig, raw.get("world")),
    )
    _validate(cfg)
    return cfg


def default_config() -> SimConfig:
    """Return a SimConfig with all defaults (no file needed)."""
    cfg = SimConfig()
    _validate(cfg)
    return cfg
