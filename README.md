# Ant Colony Simulation

A biologically-inspired ant colony simulation featuring emergent collective behaviors, multiple brain backends (rule-based, NumPy and torch neural networks/transformers), and real-time visualization.

## Architecture Overview

The simulation follows a per-tick pipeline:

```
Build Sensory Inputs
        |
  Reward & Learn (from previous tick)
        |
  Brain.decide() -> AntAction
        |
  Apply Actions (movement, pheromone, pickup/drop)
        |
  Pheromone Engine (deposit, diffuse, evaporate)
        |
  World Tick (food respawn)
        |
  Colony Tick (death, spawning, role rebalance)
        |
  Record Metrics & Emergence
```

**Key abstractions:**

- **BrainBackend protocol** — All brain types implement `decide(sensory) -> AntAction` and `learn(reward)`. Brains are hot-swappable at runtime.
- **SensoryInput** — What each ant perceives: antenna pheromone readings (left/right x 4 channels), obstacle raycasts, nest direction/distance/bearing, neighbors, food gradient, energy, carrying state.
- **AntAction** — Brain output: turn angle, speed multiplier, pheromone deposit (channel + strength), pickup, drop, recruit signal.
- **PheromoneGrid** — 4-channel grid (food, home, danger, recruit) with per-channel Gaussian diffusion and exponential decay.

## Installation

```bash
pip install -r requirements.txt

# Dev dependencies (tests)
pip install -r requirements-dev.txt
```

Requires Python 3.9+.

## Running

### GUI Mode

```bash
python3 main.py                          # Default: rule-based brain, seed 42
python3 main.py --brain nn               # Neural network brain
python3 main.py --brain transformer      # Transformer brain
python3 main.py --brain torch_nn         # PyTorch neural network brain
python3 main.py --brain torch_transformer  # PyTorch transformer brain
python3 main.py --seed 123 --ants 500    # Custom seed and population
python3 main.py --config my_config.yaml  # Custom configuration
```

### Headless Mode

```bash
python3 main.py --headless --ticks 5000 --brain rule_based
python3 main.py --headless --ticks 5000 --report results.json
```

### Brain Comparison

```bash
python3 compare_brains.py --brains nn torch_nn transformer torch_transformer
```

### Google Colab Worker

Use Colab as a remote worker for torch benchmarks:

1. Open [colab/torch_benchmark_worker.ipynb](colab/torch_benchmark_worker.ipynb) in Google Colab.
2. Set parameters in the first code cell (`REPO_URL`, `BRANCH`, `MODE`, `TICKS`, `WARMUP_TICKS`, `SEEDS`, `ANTS`).
3. Run all cells. The notebook mounts Google Drive, syncs the repo branch, and runs:

```bash
bash scripts/colab_benchmark.sh --mode torch_only --ticks ... --warmup-ticks ... --seeds ... --ants ... --out-dir ...
```

Artifacts are written to Drive (timestamped run folder):
- `migration.json` (full compare output)
- `migration_check.json` (torch migration summary; NumPy baseline pairs are `SKIP` in `torch_only` mode)
- `run_meta.json` (commit, device, thresholds, run parameters)
- `perf_summary.json` (per-brain avg ticks/sec + wall time)

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config`, `-c` | YAML config path | `colony_config.yaml` |
| `--seed`, `-s` | RNG seed | 42 |
| `--headless` | Run without GUI | off |
| `--ticks`, `-t` | Max ticks (0 = unlimited) | 0 |
| `--brain` | Brain backend | `rule_based` |
| `--ants` | Override population | config value |
| `--load` | Resume from saved state (.pkl) | — |
| `--report` | Write JSON metrics on exit | — |

## Controls

### Simulation

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| +/= | Increase speed (up to 10x) |
| - | Decrease speed (down to 0.5x) |

### Brain Hot-Swap

| Key | Brain |
|-----|-------|
| R | Rule-based |
| N | Neural network (NumPy) |
| T | Transformer (NumPy) |
| M | Neural network (PyTorch) |
| Shift+T | Transformer (PyTorch) |

### Pheromone Overlays

| Key | Channel |
|-----|---------|
| 1 | Food (green) |
| 2 | Home (blue) |
| 3 | Danger (red) |
| 4 | Recruit (yellow) |
| 0 | All off |

### Visualization

| Key | Overlay |
|-----|---------|
| S | HUD stats panel |
| H | Ant density heatmap |
| A | Trail analysis |
| W | Weight heatmap (NN/transformer) |
| D | Action distribution histogram |
| V | Attention visualization (transformer) |

### Mouse

| Action | Effect |
|--------|--------|
| Left-click | Select/deselect ant |
| Middle-click | Place food source |
| Right-drag | Draw obstacle polygon |
| Shift+Right-click | Remove obstacle |

### Stress Testing

| Key | Effect |
|-----|--------|
| K | Kill 10% of colony |
| F | Remove all food sources |

## Training Pipeline

### Imitation Learning (behavioral cloning from rule-based brain)

```bash
# NN brain — collect demos then train
python3 train_imitation.py --brain nn --demo-ticks 5000 --epochs 50

# Transformer brain
python3 train_imitation.py --brain transformer --demo-ticks 5000 --epochs 50

# Larger run (better results, slower)
python3 train_imitation.py --brain nn --demo-ticks 20000 --epochs 200 --demo-ants 500
```

### PPO Fine-Tuning (requires imitation-pretrained weights)

```bash
# Fine-tune NN from imitation weights
python3 train_ppo.py --brain nn --load-imitation weights/imitation/ --ticks 50000

# With custom learning rate
python3 train_ppo.py --brain nn --load-imitation weights/imitation/ --ticks 50000 --lr 1e-4
```

### Weight Persistence

```bash
# Save weights after training
python3 main.py --brain nn --ticks 5000 --headless --save-weights weights/nn_trained/

# Load pre-trained weights
python3 main.py --brain nn --load-weights weights/nn_trained/

# Auto-save every N ticks during interactive runs
python3 main.py --brain nn --save-weights weights/nn_live/ --autosave-interval 5000
```

## Hardcoded Patches

Neural brains use three configurable patches (scaffolding) to keep ants alive while learning. These compensate for behaviors the untrained policy can't yet produce:

| Patch | Effect | Why needed |
|-------|--------|------------|
| `survival_homing` | Overrides brain when energy < 30%, forcing nest-return | Without this, neural ants die within ~200 ticks |
| `auto_pickup` | Forces item pickup whenever hands are empty | Without this, neural ants walk over food without picking it up |
| `food_drop_guard` | Prevents dropping food items (auto-deposited at nest) | Without this, neural ants randomly drop food mid-trip |

Toggle in `colony_config.yaml`:

```yaml
brain:
  patches:
    survival_homing: true
    auto_pickup: true
    food_drop_guard: true
```

Or in code via `PatchConfig`. These should be progressively disabled as training improves.

## Brain Backends

### Rule-Based (`rule_based`)

Hand-crafted state machines with Braitenberg-style pheromone steering:

- **Forager**: SEARCHING → HARVESTING → RETURNING → DEPOSITING. Uses antenna differentials for food following, nest bearing for return navigation. Deposits food and home pheromone trails.
- **Soldier**: PATROLLING ↔ RESPONDING. Orbits nest at configurable radius, responds to danger pheromone.
- **Nurse**: TENDING ↔ CLEANING. Carries brood to nest, removes corpses to cemetery area (~150px from nest with clockwise clustering bias).
- **Idle**: Wanders near nest, conserving energy.

### Neural Network (`nn`)

Pure NumPy MLP with REINFORCE policy gradient training:
- Architecture: 36 inputs → 64 → 32 → 11 outputs (multi-head: turn, speed, deposit channel probabilities, deposit strength, pickup, drop, recruit)
- Per-role shared weights via `SharedWeightRegistry`
- Experience replay buffer with configurable update interval

### Transformer (`transformer`)

Pure NumPy causal transformer with temporal context:
- Architecture: 36-dim input → d_model=32, 4 heads, 2 layers, FFN=64 → 11 outputs
- Sliding context window (last 16 observations)
- Sinusoidal positional encoding, causal attention masking
- REINFORCE training via zeroth-order perturbation

### PyTorch Neural Network (`torch_nn`)

PyTorch implementation of the NN brain:
- Same action head layout as `nn`
- Auto-differentiation and optimizer support via torch
- Device auto-detection (CUDA > MPS > CPU)

### PyTorch Transformer (`torch_transformer`)

PyTorch implementation of the transformer brain:
- Same high-level architecture as `transformer`
- Causal attention with torch modules
- Device auto-detection (CUDA > MPS > CPU)

## Emergent Behaviors

The simulation produces several emergent collective behaviors:

1. **Trail Formation** — Foragers deposit food pheromone on return trips, creating concentrated corridors that attract other foragers. Positive feedback loop: more ants on trail → stronger pheromone → more ants attracted.

2. **Cemetery Clustering** — Nurses carry corpses away from the nest with a consistent directional bias, creating clustered cemetery zones rather than scattered corpses.

3. **Adaptive Rerouting** — When obstacles block established trails, pheromone decays on blocked paths while new paths form around obstacles.

4. **Foraging Efficiency** — Colony food income increases over time as trails become established and foragers exploit known food sources more efficiently.

5. **Recruitment Cascades** — Foragers finding rich food sources emit recruit pheromone, attracting nearby foragers to concentrate on productive areas.

6. **Role Rebalancing** — Colony dynamically adjusts role distribution based on food income, brood count, and threat level. Low food income → more foragers; high threats → more soldiers.

## Test Suite

```bash
# All tests (excluding long-running integration tests)
python3 -m pytest tests/ -q --ignore=tests/test_emergence.py --ignore=tests/test_replay.py

# Integration tests (emergence behaviors, ~25 min)
python3 -m pytest tests/test_emergence.py -v

# Replay determinism tests
python3 -m pytest tests/test_replay.py -v

# Specific test file
python3 -m pytest tests/test_rule_brain.py -v
```

Test files:

| File | Coverage |
|------|----------|
| `test_ant_physics.py` | Movement, collision, bounds, energy |
| `test_colony.py` | Spawning, death, roles, stats |
| `test_pheromone.py` | Diffusion, evaporation, sampling |
| `test_world.py` | World creation, food, obstacles |
| `test_sensory.py` | Sensory input building and encoding |
| `test_rule_brain.py` | State machines, steering, role rebalancing |
| `test_nn_brain.py` | NN forward pass, learning, weight sharing |
| `test_transformer_brain.py` | Attention, context window, training |
| `test_emergence.py` | Emergent behavior integration tests |
| `test_replay.py` | Save/load, deterministic replay |

## Configuration

All settings live in `colony_config.yaml`. See `config.py` for the full dataclass hierarchy and validation rules. Key sections:

```yaml
colony:
  initial_population: 200
  max_population: 500
  spawn_rate: 0.1          # ants per tick when food > 50
  initial_food_stored: 100

ant:
  base_speed: 2.0
  energy_max: 100
  antenna_angle: 30        # degrees, half-angle per cone
  antenna_range: 40        # pixels

pheromone:
  cell_size: 4
  channels:
    food:  { decay: 0.995, diffusion_sigma: 0.5 }
    home:  { decay: 0.997, diffusion_sigma: 0.5 }
    danger: { decay: 0.980, diffusion_sigma: 0.8 }
    recruit: { decay: 0.970, diffusion_sigma: 1.0 }

roles:
  default_distribution:
    forager: 0.60
    nurse: 0.15
    soldier: 0.10
    idle: 0.15
  rebalance_interval: 500

brain:
  default: rule_based
  nn:
    hidden_sizes: [64, 32]
    learning_rate: 0.0001
  transformer:
    context_length: 16
    d_model: 32
    n_heads: 4
    n_layers: 2

world:
  width: 1600
  height: 1000
  num_food_sources: 5
  num_obstacles: 10
```

## Performance

- **Rule-based**: ~1000+ ants at 60 FPS (real-time)
- **NN (NumPy)**: ~300 ants at real-time; shared weights amortize memory
- **Transformer (NumPy)**: ~200 ants at real-time; context window limits throughput
- **Torch NN/Transformer**: Hardware-accelerated where CUDA/MPS is available

Headless mode runs significantly faster (no rendering overhead). Use `--headless` for experiments.

## Known Limitations

### NN Brain (REINFORCE)

- Collects **zero food** in short experiments (5k ticks) with default REINFORCE training
- Sparse reward problem: the multi-step sequence (find food → pick up → navigate home → deposit) is never triggered by a random policy
- No value function baseline → extremely noisy gradient estimates
- Hardcoded patches keep ants alive but the brain itself doesn't learn pickup/homing — these are scaffolding, not genuine learning
- Imitation learning from rule-based brain provides a viable starting policy; PPO fine-tuning can then improve beyond the teacher

### Transformer Brain (REINFORCE + zeroth-order)

- Same zero-food problem as NN, compounded by zeroth-order gradient estimation (random parameter perturbation)
- Zeroth-order methods scale poorly: each update perturbs all ~5,000 parameters with random noise, making learning extremely sample-inefficient
- Context window (16 timesteps) provides temporal information, but the policy can't exploit it without better gradients
- Single-step imitation (seq_len=1) partially mitigates this for behavioral cloning

### General

- Pheromone grid resolution (4px cells) limits fine-grained trail formation
- No inter-colony competition or predator agents
- Transformer context window is fixed-length (no variable attention span)

See [ROADMAP.md](ROADMAP.md) for the GPU training plan and future phases.
