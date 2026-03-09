"""Rule-based brain with state machines for forager, soldier, and nurse roles."""

from __future__ import annotations

import math
import random
from enum import Enum

from agents.actions import AntAction
from agents.sensory import SensoryInput


# ---------------------------------------------------------------------------
# State enums
# ---------------------------------------------------------------------------

class ForagerState(Enum):
    SEARCHING = "searching"
    HARVESTING = "harvesting"
    RETURNING = "returning"
    DEPOSITING = "depositing"


class SoldierState(Enum):
    PATROLLING = "patrolling"
    RESPONDING = "responding"


class NurseState(Enum):
    TENDING = "tending"
    CLEANING = "cleaning"


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_FOOD_SIGNAL_THRESHOLD = 0.3       # antenna food reading to trigger HARVESTING
_FOOD_LOST_THRESHOLD = 0.1         # below this → back to SEARCHING
_DANGER_THRESHOLD = 0.1            # antenna danger reading to trigger RESPONDING
_NEST_ARRIVE_DISTANCE = 35.0       # pixels; must be < nest radius (40)
_RICH_FOOD_THRESHOLD = 0.5         # recruit signal if food signal > this
_OBSTACLE_AVOID_THRESHOLD = 0.6    # normalised ray distance; 1.0 = clear
_PATROL_RADIUS = 80.0              # desired orbit distance from nest
_CEMETERY_DROP_DISTANCE = 150.0     # pixels from nest to drop corpse
_BRAITENBERG_GAIN = 2.0            # steering gain for antenna differential
_MAX_TURN = math.pi / 6            # matches AntAction clamp


def _clamp_turn(t: float) -> float:
    return max(-_MAX_TURN, min(_MAX_TURN, t))


# ---------------------------------------------------------------------------
# RuleBasedBrain
# ---------------------------------------------------------------------------

class RuleBasedBrain:
    """Hand-crafted decision engine with per-role state machines.

    Satisfies the ``BrainBackend`` protocol.
    """

    def __init__(
        self,
        world_width: float = 1600.0,
        world_height: float = 1000.0,
        rng_seed: int | None = None,
    ) -> None:
        self._world_width = world_width
        self._world_height = world_height
        self._world_diag = math.hypot(world_width, world_height)
        self._rng = random.Random(rng_seed)

        # Per-role state machines
        self._forager_state = ForagerState.SEARCHING
        self._soldier_state = SoldierState.PATROLLING
        self._nurse_state = NurseState.TENDING

        # Internal bookkeeping
        self._ticks: int = 0
        self._patrol_angle: float = self._rng.uniform(0, 2 * math.pi)
        self._wander_bias: float = 0.0
        self._search_boost_ticks: int = 0  # post-harvest vicinity search countdown

    # -- public properties for inspection / testing -------------------------

    @property
    def forager_state(self) -> ForagerState:
        return self._forager_state

    @forager_state.setter
    def forager_state(self, value: ForagerState) -> None:
        self._forager_state = value

    @property
    def soldier_state(self) -> SoldierState:
        return self._soldier_state

    @soldier_state.setter
    def soldier_state(self, value: SoldierState) -> None:
        self._soldier_state = value

    @property
    def nurse_state(self) -> NurseState:
        return self._nurse_state

    @nurse_state.setter
    def nurse_state(self, value: NurseState) -> None:
        self._nurse_state = value

    # ======================================================================
    # BrainBackend protocol
    # ======================================================================

    def decide(self, sensory: SensoryInput) -> AntAction:
        """Dispatch to the role-specific state machine."""
        self._ticks += 1
        role = sensory.role_value
        if role == "forager":
            return self._decide_forager(sensory)
        if role == "soldier":
            return self._decide_soldier(sensory)
        if role == "nurse":
            return self._decide_nurse(sensory)
        return self._decide_idle(sensory)

    def learn(self, reward: float) -> None:  # noqa: D401
        """No-op — rule brains don't learn."""

    # ======================================================================
    # Shared helpers
    # ======================================================================

    def _obstacle_avoidance_turn(self, s: SensoryInput) -> float:
        """Proportional turn away from close obstacles.

        Each ray contributes inversely to its distance.  Rays on the left
        push right (+turn) and vice-versa.
        """
        rays = s.obstacle_rays
        if not rays:
            return 0.0

        n = len(rays)
        total_turn = 0.0
        total_urgency = 0.0

        for i, dist in enumerate(rays):
            if dist >= _OBSTACLE_AVOID_THRESHOLD:
                continue
            closeness = 1.0 - dist / _OBSTACLE_AVOID_THRESHOLD
            # position: -1 (leftmost) … +1 (rightmost)
            position = (i / (n - 1)) * 2.0 - 1.0 if n > 1 else 0.0
            # turn *away*: obstacle on left (position<0) → turn right (+)
            total_turn += -position * closeness
            total_urgency += closeness

        if total_urgency < 1e-6:
            return 0.0

        n = len(rays)
        # Direction: which way to turn (-1..+1)
        direction = total_turn / total_urgency
        # Magnitude: how hard to turn (scales with closeness)
        magnitude = min(1.0, total_urgency / max(n * 0.3, 1.0))

        # Centre obstacle (direction≈0) still needs a nudge
        if abs(direction) < 0.05 and total_urgency > 0.3:
            direction = 0.3  # default nudge right
        return _clamp_turn(direction * magnitude * _MAX_TURN)

    def _braitenberg_turn(self, s: SensoryInput, channel: str) -> float:
        """Antenna-differential steering.  Stronger right → turn right (+)."""
        left = s.antenna_left.get(channel, 0.0)
        right = s.antenna_right.get(channel, 0.0)
        return _clamp_turn((right - left) * _BRAITENBERG_GAIN)

    def _wander_turn(self) -> float:
        """Small correlated random walk for exploration."""
        self._wander_bias += self._rng.gauss(0, 0.1)
        self._wander_bias *= 0.9
        return _clamp_turn(self._wander_bias)

    @staticmethod
    def _max_signal(s: SensoryInput, channel: str) -> float:
        return max(
            s.antenna_left.get(channel, 0.0),
            s.antenna_right.get(channel, 0.0),
        )

    # ======================================================================
    # Forager  SEARCHING → HARVESTING → RETURNING → DEPOSITING → …
    # ======================================================================

    def _decide_forager(self, s: SensoryInput) -> AntAction:
        self._update_forager_state(s)
        state = self._forager_state
        if state is ForagerState.SEARCHING:
            return self._forager_searching(s)
        if state is ForagerState.HARVESTING:
            return self._forager_harvesting(s)
        if state is ForagerState.RETURNING:
            return self._forager_returning(s)
        return self._forager_depositing(s)

    def _update_forager_state(self, s: SensoryInput) -> None:
        st = self._forager_state

        if st is ForagerState.SEARCHING:
            if s.carrying == "food":
                self._forager_state = ForagerState.RETURNING
            elif self._max_signal(s, "food") > _FOOD_SIGNAL_THRESHOLD:
                self._forager_state = ForagerState.HARVESTING

        elif st is ForagerState.HARVESTING:
            if s.carrying == "food":
                self._forager_state = ForagerState.RETURNING
            elif self._max_signal(s, "food") < _FOOD_LOST_THRESHOLD:
                self._forager_state = ForagerState.SEARCHING
                self._search_boost_ticks = 30  # vicinity search after food exhaustion

        elif st is ForagerState.RETURNING:
            if s.nest_distance < _NEST_ARRIVE_DISTANCE:
                self._forager_state = ForagerState.DEPOSITING
            elif s.carrying != "food":
                # food was auto-deposited by physics on nest entry
                self._forager_state = ForagerState.DEPOSITING

        elif st is ForagerState.DEPOSITING:
            if s.carrying is None:
                self._forager_state = ForagerState.SEARCHING

    # -- Forager sub-behaviours ---------------------------------------------

    def _forager_searching(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(
                turn=avoid, speed_mult=0.7,
                pickup=True,
                deposit_pheromone="home", deposit_strength=0.5,
            )

        food_turn = self._braitenberg_turn(s, "food")
        recruit_turn = self._braitenberg_turn(s, "recruit")
        wander = self._wander_turn()

        food_str = self._max_signal(s, "food")
        recruit_str = self._max_signal(s, "recruit")

        # Post-harvest vicinity search: boost food pheromone weight
        if self._search_boost_ticks > 0:
            self._search_boost_ticks -= 1
            food_boost = 3.0
        else:
            food_boost = 1.0

        w_wander = 0.15
        w_food = food_str * 2.0 * food_boost
        w_recruit = recruit_str * 2.5
        total_w = w_wander + w_food + w_recruit
        turn = (wander * w_wander + food_turn * w_food + recruit_turn * w_recruit) / total_w

        return AntAction(
            turn=_clamp_turn(turn), speed_mult=1.0,
            pickup=True,
            deposit_pheromone="home", deposit_strength=0.5,
        )

    def _forager_harvesting(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        food_turn = self._braitenberg_turn(s, "food")
        turn = avoid if abs(avoid) > 0.01 else food_turn

        recruit = self._max_signal(s, "food") > _RICH_FOOD_THRESHOLD
        return AntAction(
            turn=turn, speed_mult=0.5,
            pickup=True, recruit_signal=recruit,
            deposit_pheromone="home", deposit_strength=0.8,
        )

    def _forager_returning(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(
                turn=avoid, speed_mult=0.7,
                deposit_pheromone="food", deposit_strength=0.8,
            )

        # Direct navigation using nest bearing (path integration)
        turn = _clamp_turn(s.nest_bearing)

        recruit = s.carry_amount > _RICH_FOOD_THRESHOLD
        return AntAction(
            turn=turn, speed_mult=0.8,
            deposit_pheromone="food", deposit_strength=min(1.0, s.carry_amount + 0.3),
            recruit_signal=recruit,
        )

    def _forager_depositing(self, s: SensoryInput) -> AntAction:
        if s.carrying is not None:
            # Navigate closer to nest center so refill_at_nest auto-deposits
            turn = _clamp_turn(s.nest_bearing)
            return AntAction(turn=turn, speed_mult=0.5, drop=True)
        return AntAction(turn=self._wander_turn(), speed_mult=0.5)

    # ======================================================================
    # Soldier  PATROLLING ↔ RESPONDING
    # ======================================================================

    def _decide_soldier(self, s: SensoryInput) -> AntAction:
        self._update_soldier_state(s)
        if self._soldier_state is SoldierState.PATROLLING:
            return self._soldier_patrolling(s)
        return self._soldier_responding(s)

    def _update_soldier_state(self, s: SensoryInput) -> None:
        danger = self._max_signal(s, "danger")
        if self._soldier_state is SoldierState.PATROLLING:
            if danger > _DANGER_THRESHOLD:
                self._soldier_state = SoldierState.RESPONDING
        else:
            if danger < _DANGER_THRESHOLD * 0.5:
                self._soldier_state = SoldierState.PATROLLING

    def _soldier_patrolling(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(turn=avoid, speed_mult=0.6)

        self._patrol_angle += 0.05

        dist = s.nest_distance
        if dist > _PATROL_RADIUS * 1.3:
            turn = _clamp_turn(s.nest_bearing)
        elif dist < _PATROL_RADIUS * 0.5:
            turn = _clamp_turn(self._wander_turn() + 0.15)
        else:
            turn = 0.15  # gentle orbit
        return AntAction(turn=_clamp_turn(turn), speed_mult=0.8)

    def _soldier_responding(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(
                turn=avoid, speed_mult=1.0,
                deposit_pheromone="danger", deposit_strength=0.5,
            )
        turn = self._braitenberg_turn(s, "danger")
        return AntAction(
            turn=turn, speed_mult=1.0,
            deposit_pheromone="danger", deposit_strength=0.5,
        )

    # ======================================================================
    # Nurse  TENDING ↔ CLEANING
    # ======================================================================

    def _decide_nurse(self, s: SensoryInput) -> AntAction:
        self._update_nurse_state(s)
        if self._nurse_state is NurseState.TENDING:
            return self._nurse_tending(s)
        return self._nurse_cleaning(s)

    def _update_nurse_state(self, s: SensoryInput) -> None:
        if self._nurse_state is NurseState.TENDING:
            if s.carrying == "corpse":
                self._nurse_state = NurseState.CLEANING
        else:
            if s.carrying == "brood":
                self._nurse_state = NurseState.TENDING
            elif s.carrying is None:
                self._nurse_state = NurseState.TENDING

    def _nurse_tending(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(turn=avoid, speed_mult=0.5)

        if s.carrying == "brood":
            turn = _clamp_turn(s.nest_bearing)
            if s.nest_distance < _NEST_ARRIVE_DISTANCE * 0.5:
                return AntAction(turn=0.0, speed_mult=0.3, drop=True)
            return AntAction(turn=turn, speed_mult=0.6)

        # Not carrying — wander near nest, try to pick up brood
        if s.nest_distance > _NEST_ARRIVE_DISTANCE * 2:
            turn = _clamp_turn(s.nest_bearing)
        else:
            turn = self._wander_turn()
        return AntAction(turn=turn, speed_mult=0.5, pickup=True)

    def _nurse_cleaning(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(turn=avoid, speed_mult=0.5)

        if s.carrying == "corpse":
            # Move AWAY from nest using nest_bearing, with clockwise bias for clustering
            away_bearing = s.nest_bearing + math.pi  # opposite of nest direction
            away_bearing = (away_bearing + math.pi) % (2 * math.pi) - math.pi
            turn = _clamp_turn(away_bearing + 0.15)
            if s.nest_distance > _CEMETERY_DROP_DISTANCE:
                return AntAction(turn=0.0, speed_mult=0.3, drop=True)
            return AntAction(turn=turn, speed_mult=0.7)

        # Wander looking for corpses
        return AntAction(turn=self._wander_turn(), speed_mult=0.5, pickup=True)

    # ======================================================================
    # Idle
    # ======================================================================

    def _decide_idle(self, s: SensoryInput) -> AntAction:
        avoid = self._obstacle_avoidance_turn(s)
        if abs(avoid) > 0.01:
            return AntAction(turn=avoid, speed_mult=0.3)

        if s.nest_distance > _NEST_ARRIVE_DISTANCE * 3:
            turn = _clamp_turn(s.nest_bearing)
            return AntAction(turn=turn, speed_mult=0.5)
        return AntAction(turn=self._wander_turn(), speed_mult=0.2)

    # ======================================================================
    # Role rebalancing (static utility)
    # ======================================================================

    @staticmethod
    def suggest_role_distribution(
        food_income_rate: float,
        brood_count: int,
        threat_level: float,
        current_distribution: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Suggest role fractions based on colony needs.

        Parameters
        ----------
        food_income_rate:
            Food deposited per tick (rolling average).
        brood_count:
            Number of brood items in the colony.
        threat_level:
            0.0–1.0 average danger pheromone near nest.
        current_distribution:
            If given, the result is smoothed toward it (70 % old, 30 % new).

        Returns
        -------
        dict mapping role name → fraction (sums to 1.0).
        """
        forager = 0.60
        nurse = 0.15
        soldier = 0.10
        idle = 0.15

        # --- food income adjustments ---
        if food_income_rate < 0.5:
            forager += 0.15
            idle -= 0.10
            nurse -= 0.05
        elif food_income_rate > 2.0:
            forager -= 0.10
            nurse += 0.05
            soldier += 0.05

        # --- brood adjustments ---
        if brood_count > 10:
            nurse += 0.10
            forager -= 0.05
            idle -= 0.05
        elif brood_count == 0:
            nurse -= 0.05
            forager += 0.05

        # --- threat adjustments ---
        if threat_level > 0.5:
            soldier += 0.15
            idle -= 0.10
            forager -= 0.05
        elif threat_level > 0.2:
            soldier += 0.05
            idle -= 0.05

        # Clamp & normalise
        roles = {"forager": forager, "nurse": nurse, "soldier": soldier, "idle": idle}
        roles = {k: max(0.05, min(0.85, v)) for k, v in roles.items()}
        total = sum(roles.values())
        roles = {k: v / total for k, v in roles.items()}

        # Exponential smoothing toward current distribution
        if current_distribution is not None:
            alpha = 0.3
            for role in roles:
                if role in current_distribution:
                    roles[role] = (
                        alpha * roles[role]
                        + (1 - alpha) * current_distribution[role]
                    )
            total = sum(roles.values())
            roles = {k: v / total for k, v in roles.items()}

        return roles
