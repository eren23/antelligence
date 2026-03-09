# Swarm Goal

Emergent Colony Simulation — Pygame + Learnable Swarm Intelligence

Build a real-time ant colony simulation in Pygame where hundreds of agents
follow simple local rules that produce complex emergent behavior: foraging,
trail formation, bridge building, brood sorting, cemetery clustering, and
cooperative transport. The system supports three "brain" backends that can
be hot-swapped at runtime: rule-based (classic), small neural network, and
a micro-transformer. A full test suite validates every layer from individual
agent physics up to emergent population-level metrics.

---

## 0) World

A continuous 2D world (default 1600×1000 pixels) with:

- **Nest**: a circular region (radius ~40px) at center-left. Ants spawn
  here. Brood and food storage tracked internally.
- **Food sources**: 3–6 circular patches of varying richness (50–500 units)
  scattered on the map. Food depletes as ants harvest. New sources spawn
  stochastically to keep the simulation alive.
- **Obstacles**: randomly placed convex polygons (5–15) and walls. Ants
  cannot walk through them but can path around them.
- **Pheromone grid**: a discrete grid overlay (cell size ~4px) with
  multiple pheromone channels:
  - `FOOD` — "I found food, follow me home"
  - `HOME` — "I'm heading out from the nest"
  - `DANGER` — "threat detected, avoid this area"
  - `RECRUIT` — "I need help carrying something big"
- Each channel has independent **diffusion** (Gaussian blur per tick) and
  **evaporation** (exponential decay, configurable half-life per channel).
- **Terrain types** (optional stretch): mud (0.5× speed), sand (0.8×),
  grass (1.0×), water (impassable).

Rendering: Pygame at 60 FPS target. Pheromone overlay is a translucent
heatmap (toggle per channel with keys 1–4). Ants rendered as small
oriented triangles color-coded by role/state.

---

## 1) Ant Agent

Each ant is an autonomous agent with:

### State
```python
@dataclass
class Ant:
    id: int
    pos: Vec2              # continuous position
    heading: float         # radians
    speed: float           # pixels/tick (base ~2.0)
    energy: float          # 0–100, depletes over time, refills at nest
    carrying: Optional[str]  # None | "food" | "brood" | "corpse"
    carry_amount: float    # 0.0–1.0 (fraction of max carry capacity)
    role: Role             # FORAGER | NURSE | SOLDIER | IDLE
    age: int               # ticks alive
    alive: bool
    brain: BrainBackend    # swappable decision engine
    sensory: SensoryInput  # populated each tick before decision
```

### Sensory Input (what the ant perceives each tick)
- **Antennae cone**: two overlapping sensor cones (left/right, ±30° from
  heading, range 40px). Each cone samples the pheromone grid and returns
  average concentration per channel.
- **Local neighbors**: list of ants within 20px radius (id, relative
  position, role, carrying status).
- **Obstacle proximity**: raycasts (5 rays in a 120° forward arc, range
  30px) returning distance to nearest obstacle per ray.
- **Nest direction + distance**: always available (proprioceptive compass).
- **Food scent gradient**: local food-channel pheromone gradient vector.
- **Ground type** at current position.
- **Energy level**.
- **Currently carrying** item type + amount.

### Actions (output of brain each tick)
```python
@dataclass
class AntAction:
    turn: float          # delta heading in radians, clamped [-π/6, π/6]
    speed_mult: float    # 0.0–1.0 multiplier on base speed
    deposit_pheromone: Optional[str]  # channel name or None
    deposit_strength: float           # 0.0–1.0
    pickup: bool         # attempt to pick up item at current pos
    drop: bool           # drop carried item
    recruit_signal: bool # emit short-range recruit pulse
```

### Physics / Movement
- Heading updated by `turn`, then position advanced by
  `speed * speed_mult * terrain_mult` in heading direction.
- Collision with obstacles: slide along obstacle edge (no teleporting).
- Collision with world bounds: reflect heading.
- Energy cost: `0.02/tick` base + `0.01 * speed_mult` + `0.03` if carrying.
- Energy ≤ 0 → ant dies, becomes a corpse object on the ground.
- At nest: energy refills at 2.0/tick. If carrying food, deposit it.

---

## 2) Brain Backends

All three backends implement the same interface:

```python
class BrainBackend(Protocol):
    def decide(self, sensory: SensoryInput) -> AntAction: ...
    def learn(self, reward: float) -> None: ...  # no-op for rule-based
```

### 2a) Rule-Based Brain (`RuleBasedBrain`)

Classic ant colony optimization rules:

**Forager behavior (state machine):**
1. `SEARCHING` — wander with slight randomness. If food pheromone
   detected, bias turn toward stronger antenna reading (Braitenberg-style
   gradient following). Deposit `HOME` pheromone.
2. `HARVESTING` — at food source: pick up food, switch to `RETURNING`.
3. `RETURNING` — follow `HOME` pheromone gradient back to nest. Deposit
   `FOOD` pheromone (strength proportional to food quality found).
4. `DEPOSITING` — at nest: drop food, switch to `SEARCHING`.

**Obstacle avoidance:** If any forward raycast < 10px, turn away from
closest obstacle (proportional to inverse distance).

**Recruitment:** If food source is rich (>200 units remaining), set
`recruit_signal = True` for 50 ticks after finding it. Nearby idle ants
switch to forager role.

**Soldier behavior:** Patrol nest perimeter. If `DANGER` pheromone
detected, move toward source. Deposit `DANGER` pheromone near threats.

**Nurse behavior:** Stay near nest. Sort brood items (move scattered brood
toward nest center). Pick up corpses and carry them to a "cemetery" area
(far corner — this produces emergent cemetery clustering).

**Role assignment (colony-level):**
- Default: 60% forager, 15% nurse, 10% soldier, 15% idle.
- Dynamic rebalancing every 500 ticks based on food income rate, brood
  count, and threat level.

### 2b) Small Neural Network Brain (`NNBrain`)

A lightweight MLP that learns via a simple policy-gradient / REINFORCE
approach:

**Architecture:**
```
Input (sensory vector, ~40 floats) →
  Dense(64, ReLU) →
  Dense(32, ReLU) →
  Output heads:
    turn:     Dense(1, tanh) → scaled to [-π/6, π/6]
    speed:    Dense(1, sigmoid)
    deposit:  Dense(5, softmax) → [none, food, home, danger, recruit]
    strength: Dense(1, sigmoid)
    pickup:   Dense(1, sigmoid) → threshold 0.5
    drop:     Dense(1, sigmoid) → threshold 0.5
    recruit:  Dense(1, sigmoid) → threshold 0.5
```

**Sensory encoding** (fixed-size vector):
- Left/right antenna pheromone readings (4 channels × 2 = 8)
- Obstacle raycasts (5 distances, normalized)
- Nest direction (sin, cos = 2)
- Nest distance (1, normalized)
- Energy (1)
- Carrying one-hot (4: none, food, brood, corpse)
- Role one-hot (4)
- Neighbor count (1)
- Neighbor avg relative position (2)
- Food gradient vector (2)
- Ground type one-hot (4)
- Age (1, normalized)
- Speed (1)
- **Total: ~36–40 inputs**

**Reward signal** (computed per tick):
- `+1.0` for depositing food at nest
- `+0.3` for picking up food at source
- `+0.1` for following food pheromone gradient when searching
- `-0.1` for being idle (not moving) when energy > 50
- `-0.5` for dying (energy depletion)
- `-0.05` per tick (time pressure to do something useful)
- `+0.2` for successful corpse disposal (nurse)
- `+0.2` for brood sorting (nurse)

**Training:**
- Online learning: accumulate (state, action, reward) tuples in a
  rolling buffer (last 200 steps per ant).
- Every 100 ticks, compute discounted returns (γ = 0.95) and do a
  single REINFORCE update with baseline subtraction (mean return).
- Shared weights across all ants of the same role (4 weight sets).
- Learning rate: 1e-4 with linear warmup over first 2000 ticks.
- Implemented in **pure NumPy** (no PyTorch/TF dependency for the
  simulation — keep it self-contained).

### 2c) Micro-Transformer Brain (`TransformerBrain`)

A tiny causal transformer that conditions on the ant's recent history
of observations, enabling **temporal reasoning** (e.g., "I've been
going in circles" or "pheromone was stronger 5 steps ago").

**Architecture:**
```
context_length: 16 (last 16 sensory snapshots)
d_model: 32
n_heads: 4
n_layers: 2
ffn_dim: 64
```

**Input:** sequence of 16 sensory vectors (each ~40 floats),
projected to d_model=32 via a learned linear layer + sinusoidal
positional encoding.

**Output:** same action heads as NNBrain, but applied to the last
token's output embedding.

**Training:** same REINFORCE approach as NNBrain, but with a longer
buffer (last 500 steps) and updates every 200 ticks. Learning rate 5e-5.

**Implementation:** pure NumPy. Attention is standard scaled dot-product.
Softmax, layer norm, GELU activations all hand-implemented. This is the
"from scratch" flex — no ML framework.

**Performance budget:** the transformer brain must run at < 0.5ms per ant
per tick on a modern CPU for a colony of 200 ants. Use vectorized NumPy
ops across the colony (batch all 200 forward passes into matrix ops).

---

## 3) Emergent Behaviors to Verify

The simulation should produce these classic emergent phenomena (validated
by automated metrics, not just eyeballing):

| Behavior | Description | Metric |
|---|---|---|
| **Trail formation** | Ants converge on shortest paths between nest and food | Path entropy decreases over time; pheromone corridor width narrows |
| **Adaptive re-routing** | When an obstacle is placed on an established trail, ants find alternative path within 500 ticks | Trail re-convergence time |
| **Bridge building** | At narrow gaps (2–3 ant widths), ants slow/stop to let others climb over (simulated as density-aware slowdown) | Throughput at bottleneck vs. open field |
| **Cemetery clustering** | Corpses end up in 1–3 clusters in corners (nurse behavior) | Spatial clustering coefficient (DBSCAN) |
| **Brood sorting** | Brood items in nest get sorted toward center | Mean brood distance to nest center decreases over time |
| **Foraging efficiency** | Colony food income rate improves over the first 2000 ticks as trails form | Food/tick moving average (window=200) |
| **Dynamic role switching** | Starving colony → more foragers; lots of corpses → more nurses | Role distribution correlation with colony needs |
| **Recruitment cascades** | Rich food source → burst of foragers heading that direction within 200 ticks | Forager flux toward source after discovery |

---

## 4) Interactive Controls (Pygame UI)

- **Pause/Resume**: Space
- **Speed**: `+`/`-` to adjust simulation speed (0.5×, 1×, 2×, 5×, 10×)
- **Pheromone overlay toggle**: keys `1`–`4` for each channel, `0` for all off
- **Brain switch**: `R` = rule-based, `N` = neural net, `T` = transformer
  (hot-swap — all ants switch immediately, keeping positions/state)
- **Place obstacle**: right-click drag to draw a polygon obstacle
- **Place food**: middle-click to drop a food source (500 units)
- **Remove obstacle**: Shift+right-click on obstacle to remove it
- **Kill random ants**: `K` to kill 10% of colony (test nurse corpse behavior)
- **Spawn food crisis**: `F` to remove all food sources (test starvation response)
- **Colony stats overlay**: `S` to toggle stats HUD showing:
  - Population (alive / dead)
  - Food stored in nest
  - Food income rate (last 200 ticks)
  - Role distribution bar chart
  - Current brain type
  - Avg ant energy
  - Pheromone total mass per channel
- **Individual ant inspector**: left-click an ant to see its full state,
  sensory input, and brain outputs in a side panel
- **Heatmap mode**: `H` to show ant density heatmap instead of individual ants
- **Trail analysis**: `A` to highlight the top-3 most-used paths (by
  cumulative pheromone deposit)

---

## 5) Pheromone Engine (Performance Critical)

The pheromone grid is the most performance-sensitive component:

- Grid resolution: world_size / 4px = 400×250 cells per channel
- 4 channels → 4 grids of 400×250 floats
- Each tick:
  1. **Deposit**: ants write pheromone into grid cells (additive, clamped
     to max 1.0)
  2. **Diffusion**: 3×3 Gaussian blur (σ=0.5) per channel. Use
     `scipy.ndimage.gaussian_filter` or a hand-rolled separable convolution
     in NumPy.
  3. **Evaporation**: multiply each channel by its decay factor (e.g.
     `FOOD: 0.995`, `HOME: 0.990`, `DANGER: 0.98`, `RECRUIT: 0.97` per tick).
  4. **Sampling**: ants read pheromone values at their antenna positions
     (bilinear interpolation on grid).

**Target**: pheromone engine must process all 4 channels in < 5ms per tick
for the default world size. Profile and optimize.

---

## 6) Colony Configuration

All colony parameters are loaded from a `colony_config.yaml`:

```yaml
colony:
  initial_population: 200
  max_population: 500
  spawn_rate: 0.1            # new ants per tick when food stored > 50
  initial_food_stored: 100

ant:
  base_speed: 2.0
  max_carry: 1.0
  energy_max: 100
  energy_cost_base: 0.02
  energy_cost_speed: 0.01
  energy_cost_carry: 0.03
  energy_refill_rate: 2.0
  antenna_angle: 30          # degrees, half-angle of each cone
  antenna_range: 40          # pixels
  neighbor_radius: 20
  obstacle_ray_count: 5
  obstacle_ray_range: 30
  obstacle_ray_arc: 120      # degrees

pheromone:
  cell_size: 4
  channels:
    food:
      decay: 0.995
      diffusion_sigma: 0.5
      color: [0, 255, 0]     # green overlay
    home:
      decay: 0.990
      diffusion_sigma: 0.5
      color: [0, 100, 255]   # blue overlay
    danger:
      decay: 0.980
      diffusion_sigma: 0.8
      color: [255, 0, 0]     # red overlay
    recruit:
      decay: 0.970
      diffusion_sigma: 1.0
      color: [255, 200, 0]   # yellow overlay

roles:
  default_distribution:
    forager: 0.60
    nurse: 0.15
    soldier: 0.10
    idle: 0.15
  rebalance_interval: 500    # ticks

brain:
  default: rule_based        # rule_based | nn | transformer
  nn:
    hidden_sizes: [64, 32]
    learning_rate: 0.0001
    gamma: 0.95
    buffer_size: 200
    update_interval: 100
  transformer:
    context_length: 16
    d_model: 32
    n_heads: 4
    n_layers: 2
    ffn_dim: 64
    learning_rate: 0.00005
    buffer_size: 500
    update_interval: 200

world:
  width: 1600
  height: 1000
  num_food_sources: 5
  food_source_size_range: [20, 50]   # radius
  food_source_amount_range: [50, 500]
  food_respawn_interval: 3000        # ticks
  num_obstacles: 10
```

---

## 7) Project Structure

```
swarm-colony/
├── main.py                      # entry point, Pygame loop
├── colony_config.yaml
├── config.py                    # config loader + validation
├── world/
│   ├── __init__.py
│   ├── world.py                 # World class: food, obstacles, bounds
│   ├── pheromone.py             # PheromoneGrid: deposit, diffuse, evaporate, sample
│   ├── food.py                  # FoodSource dataclass + spawning logic
│   └── obstacle.py              # Obstacle geometry + collision
├── agents/
│   ├── __init__.py
│   ├── ant.py                   # Ant dataclass + physics/movement
│   ├── colony.py                # Colony manager: spawning, role assignment, stats
│   ├── sensory.py               # SensoryInput builder (antenna, raycasts, neighbors)
│   └── actions.py               # AntAction dataclass + action application
├── brains/
│   ├── __init__.py
│   ├── interface.py             # BrainBackend Protocol
│   ├── rule_based.py            # RuleBasedBrain (state machine)
│   ├── nn_brain.py              # NNBrain (NumPy MLP + REINFORCE)
│   └── transformer_brain.py     # TransformerBrain (NumPy micro-transformer)
├── rendering/
│   ├── __init__.py
│   ├── renderer.py              # Main Pygame renderer
│   ├── hud.py                   # Stats overlay, inspector panel
│   ├── pheromone_overlay.py     # Pheromone heatmap rendering
│   └── controls.py              # Input handling, interactive controls
├── metrics/
│   ├── __init__.py
│   ├── tracker.py               # MetricsTracker: records all time series
│   ├── emergence.py             # Emergent behavior detectors (trail entropy, clustering)
│   └── reporter.py              # Summary report generation
├── tests/
│   ├── __init__.py
│   ├── test_ant_physics.py      # Movement, collision, energy
│   ├── test_pheromone.py        # Deposit, diffusion, evaporation, sampling
│   ├── test_sensory.py          # Antenna cone, raycasts, neighbor detection
│   ├── test_rule_brain.py       # State transitions, pheromone following
│   ├── test_nn_brain.py         # Forward pass shapes, gradient flow, reward
│   ├── test_transformer_brain.py # Attention, positional encoding, output shapes
│   ├── test_colony.py           # Spawning, role assignment, food economy
│   ├── test_world.py            # Food depletion, obstacle placement, bounds
│   ├── test_emergence.py        # Statistical tests for emergent behaviors
│   ├── test_replay.py           # Deterministic replay from saved state
│   └── conftest.py              # Shared fixtures (mini worlds, single ants, etc.)
└── README.md
```

---

## 8) Test Suite (pytest)

### Unit Tests

**test_ant_physics.py**
- Ant moves forward by `speed * speed_mult * terrain_mult` per tick
- Heading wraps correctly at ±π
- Obstacle collision: ant slides along edge, never penetrates
- World-bound reflection: heading reverses component on contact
- Energy depletes at correct rate (base + speed + carry components)
- Ant dies when energy ≤ 0, becomes corpse
- Ant energy refills at nest at correct rate
- Carrying flag set/cleared correctly on pickup/drop

**test_pheromone.py**
- Single deposit at (x,y) creates correct cell value
- Multiple deposits stack additively, clamped at 1.0
- Diffusion spreads a point source into expected Gaussian pattern (check
  that total mass is conserved ±1%)
- Evaporation: after N ticks, value ≈ initial × decay^N (within 0.1%)
- Bilinear sampling between grid cells returns correct interpolation
- Each channel is independent (depositing on FOOD doesn't affect HOME)
- Performance: 1000 ticks of diffuse+evaporate on default grid < 5s

**test_sensory.py**
- Antenna cone samples correct grid region (geometry test with known
  pheromone placement)
- Left antenna reads higher value when pheromone source is to the left
- Obstacle raycasts return correct distances for known obstacle positions
- Neighbor detection finds correct ants within radius
- Nest direction vector points toward nest from any position
- Sensory input vector has correct dimensionality for each brain backend

**test_rule_brain.py**
- Forager in SEARCHING state with food pheromone left > right → turns left
- Forager at food source → switches to HARVESTING → picks up food
- Forager carrying food → in RETURNING state → deposits HOME pheromone
- Forager at nest with food → drops food → switches to SEARCHING
- Obstacle within 10px → ant turns away
- Nurse picks up corpse → carries toward cemetery corner
- Soldier patrols within nest perimeter range
- Role rebalancing: starving colony shifts distribution toward more foragers

**test_nn_brain.py**
- Forward pass produces correctly shaped output for each action head
- All action values within valid ranges (turn clamped, speed in [0,1], etc.)
- Gradient computation is non-zero for non-zero reward
- After N training steps with constant positive reward for a specific action,
  that action's probability increases
- Weight sharing: two ants with same role reference same weight array
- Batch forward pass (200 ants) produces same results as individual passes

**test_transformer_brain.py**
- Attention weights sum to 1.0 along key dimension
- Positional encoding produces unique patterns for each position
- Layer norm output has mean ≈ 0 and var ≈ 1
- GELU activation matches reference implementation values
- Context window correctly slides (oldest observation dropped, newest added)
- Full forward pass produces valid action shapes
- Batch processing matches individual processing
- Attention patterns are causal (no future token attention)

**test_colony.py**
- Colony spawns correct initial population
- New ants spawn at nest when food stored > threshold and spawn rate
- Ants are assigned roles matching configured distribution (within 5%)
- Dead ants are removed from active list, corpse created at death position
- Food deposit at nest increments colony food storage
- Role rebalancing triggers at configured interval
- Colony stats (population, food rate, role counts) computed correctly

**test_world.py**
- Food source depletes as ants harvest (amount decreases correctly)
- Depleted food source is marked inactive
- New food sources spawn after configured interval
- Obstacles have correct collision geometry (point-in-polygon test)
- World bounds check works for all edges
- Multiple food sources can coexist at different positions

### Integration Tests

**test_emergence.py** (these run headless, accelerated simulations)

- **Trail formation test**: place 1 food source directly east of nest, run
  500 ants with rule brain for 3000 ticks. Assert: pheromone FOOD channel
  shows a corridor (>0.1 concentration) connecting nest to food, with
  corridor width < 100px at midpoint. Uses spatial analysis on the
  pheromone grid.
- **Cemetery clustering test**: kill 50 ants, run nurses for 5000 ticks.
  Assert: corpse positions form ≤ 3 clusters (DBSCAN, eps=40, min_samples=3).
- **Adaptive re-routing test**: establish trail (2000 ticks), place obstacle
  on trail midpoint, run 1000 more ticks. Assert: food income rate recovers
  to ≥ 60% of pre-obstacle rate.
- **Foraging efficiency test**: run for 5000 ticks. Assert: food income rate
  at tick 4000–5000 is ≥ 2× rate at tick 0–1000.
- **Role rebalancing test**: remove all food sources at tick 1000. Assert:
  forager percentage increases by ≥ 10 percentage points within 1000 ticks.

**test_replay.py**
- Save full simulation state at tick N, continue to tick N+500, reload
  state at tick N with same RNG seed, re-run to tick N+500 → ant positions
  match within 0.01px. Verifies determinism.

### Performance Benchmarks (not pass/fail, but tracked)

- Pheromone engine: ms per tick for default grid (target < 5ms)
- Colony update: ms per tick for 200 ants with rule brain (target < 10ms)
- Colony update: ms per tick for 200 ants with NN brain (target < 20ms)
- Colony update: ms per tick for 200 ants with transformer brain (target < 50ms)
- Full frame (update + render): ms at 200 ants (target < 16ms = 60 FPS)

---

## 9) Experiment Mode (Headless)

A CLI mode for running experiments without Pygame rendering:

```bash
python main.py --headless --brain rule_based --ticks 10000 --ants 300 --seed 42
python main.py --headless --brain nn --ticks 20000 --ants 200 --seed 42
python main.py --headless --brain transformer --ticks 20000 --ants 200 --seed 42
```

Outputs a JSON metrics report:

```json
{
  "config": { "brain": "rule_based", "ticks": 10000, "ants": 300 },
  "food_collected_total": 1847,
  "food_income_rate_final": 0.42,
  "population_final": 287,
  "deaths": 13,
  "trail_entropy_over_time": [3.2, 2.8, 2.1, ...],
  "cemetery_cluster_count": 2,
  "avg_foraging_trip_ticks": 340,
  "role_distribution_final": {"forager": 0.65, "nurse": 0.12, "soldier": 0.10, "idle": 0.13},
  "brain_specific": {
    "avg_reward_per_tick": 0.015,
    "loss_curve": [...]
  },
  "performance": {
    "avg_tick_ms": 8.3,
    "p99_tick_ms": 14.1
  }
}
```

Compare brains:
```bash
python compare_brains.py --ticks 10000 --ants 200 --seeds 42,43,44,45,46
```
Runs all 3 brains across 5 seeds and produces a comparison table:
food collected, foraging efficiency, deaths, trail formation speed.

---

## 10) Learning Visualization

When running NN or transformer brains in GUI mode:

- **Reward graph**: live line chart (bottom of screen) showing per-tick
  average reward across colony (rolling window = 100 ticks)
- **Weight heatmap**: press `W` to show a small heatmap of first-layer
  weights, updated every 100 ticks
- **Action distribution**: press `D` to show histogram of current tick's
  action outputs across all ants (are they all turning left? all depositing
  food pheromone?)
- **Attention visualization** (transformer only): press `V` on a selected
  ant to see its attention weights across the 16-step context window
  (which past observations is it attending to?)

---

## 11) Deliverables

Source code for:
- World engine (pheromone grid, food sources, obstacles, terrain)
- Ant agent (physics, sensory input, action execution)
- Colony manager (spawning, roles, economy, stats)
- Rule-based brain (state machine with all forager/nurse/soldier behaviors)
- Neural network brain (NumPy MLP + REINFORCE)
- Micro-transformer brain (NumPy, from scratch)
- Pygame renderer (ants, pheromones, HUD, inspector, controls)
- Metrics tracker + emergent behavior detectors
- Headless experiment runner + brain comparison tool
- Full test suite (unit + integration + emergence + replay + perf)
- Colony config YAML with sensible defaults

README.md including:
- Architecture overview (agent loop, brain protocol, pheromone engine)
- How to run (GUI mode, headless mode, experiments)
- Controls reference
- Brain backends explained (rule-based, NN, transformer)
- Emergent behaviors guide (what to look for)
- Test suite description + how to run
- Performance tuning notes
- Config reference (all YAML fields documented)
- Known limitations / future ideas