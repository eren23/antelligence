"""Tests for the rule-based brain: state machines, pheromone following,
obstacle avoidance, nurse/soldier behaviours, and role rebalancing."""

from __future__ import annotations

import math

import pytest

from agents.ant import Vec2
from agents.sensory import SensoryInput, NeighborInfo
from brains.rule_based import (
    RuleBasedBrain,
    ForagerState,
    SoldierState,
    NurseState,
    _FOOD_SIGNAL_THRESHOLD,
    _FOOD_LOST_THRESHOLD,
    _DANGER_THRESHOLD,
    _NEST_ARRIVE_DISTANCE,
    _RICH_FOOD_THRESHOLD,
    _OBSTACLE_AVOID_THRESHOLD,
    _PATROL_RADIUS,
    _CEMETERY_DROP_DISTANCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sensory(**overrides) -> SensoryInput:
    """Create a SensoryInput with sensible defaults and any overrides."""
    defaults = dict(
        antenna_left={"food": 0.0, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        antenna_right={"food": 0.0, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        obstacle_rays=[1.0, 1.0, 1.0, 1.0, 1.0],
        nest_direction=Vec2(-1.0, 0.0),
        nest_distance=300.0,
        neighbors=[],
        food_gradient=Vec2(0.0, 0.0),
        ground_type="grass",
        energy=80.0,
        carrying=None,
        carry_amount=0.0,
        role_value="forager",
        age=100,
        speed=2.0,
    )
    defaults.update(overrides)
    return SensoryInput(**defaults)


def _brain(**kwargs) -> RuleBasedBrain:
    return RuleBasedBrain(rng_seed=42, **kwargs)


# =========================================================================
# 1. Forager state transitions
# =========================================================================

class TestForagerStateTransitions:
    """SEARCHING → HARVESTING → RETURNING → DEPOSITING → SEARCHING."""

    def test_searching_to_harvesting_on_strong_food(self):
        brain = _brain()
        assert brain.forager_state is ForagerState.SEARCHING

        s = _make_sensory(
            antenna_left={"food": _FOOD_SIGNAL_THRESHOLD + 0.1, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.forager_state is ForagerState.HARVESTING

    def test_searching_stays_when_weak_food(self):
        brain = _brain()
        s = _make_sensory(
            antenna_left={"food": _FOOD_SIGNAL_THRESHOLD - 0.1, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.forager_state is ForagerState.SEARCHING

    def test_harvesting_to_returning_when_carrying_food(self):
        brain = _brain()
        brain.forager_state = ForagerState.HARVESTING

        s = _make_sensory(carrying="food", carry_amount=0.5)
        brain.decide(s)
        assert brain.forager_state is ForagerState.RETURNING

    def test_harvesting_back_to_searching_when_food_lost(self):
        brain = _brain()
        brain.forager_state = ForagerState.HARVESTING

        s = _make_sensory(
            antenna_left={"food": _FOOD_LOST_THRESHOLD * 0.4, "home": 0.0, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": _FOOD_LOST_THRESHOLD * 0.4, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.forager_state is ForagerState.SEARCHING

    def test_returning_to_depositing_at_nest(self):
        brain = _brain()
        brain.forager_state = ForagerState.RETURNING

        s = _make_sensory(
            carrying="food", carry_amount=0.5,
            nest_distance=_NEST_ARRIVE_DISTANCE - 5.0,
        )
        brain.decide(s)
        assert brain.forager_state is ForagerState.DEPOSITING

    def test_returning_to_depositing_when_food_auto_deposited(self):
        brain = _brain()
        brain.forager_state = ForagerState.RETURNING

        # Food was auto-deposited by physics (carrying is now None)
        s = _make_sensory(carrying=None, nest_distance=30.0)
        brain.decide(s)
        assert brain.forager_state is ForagerState.DEPOSITING

    def test_depositing_to_searching_after_drop(self):
        brain = _brain()
        brain.forager_state = ForagerState.DEPOSITING

        s = _make_sensory(carrying=None)
        brain.decide(s)
        assert brain.forager_state is ForagerState.SEARCHING

    def test_full_cycle(self):
        """Walk through the complete forager state cycle."""
        brain = _brain()

        # SEARCHING → detect food → HARVESTING
        s = _make_sensory(
            antenna_left={"food": 0.5, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.forager_state is ForagerState.HARVESTING

        # HARVESTING → pick up food → RETURNING
        s = _make_sensory(carrying="food", carry_amount=0.8)
        brain.decide(s)
        assert brain.forager_state is ForagerState.RETURNING

        # RETURNING → arrive at nest → DEPOSITING
        s = _make_sensory(
            carrying="food", carry_amount=0.8,
            nest_distance=_NEST_ARRIVE_DISTANCE - 1.0,
        )
        brain.decide(s)
        assert brain.forager_state is ForagerState.DEPOSITING

        # DEPOSITING → food dropped → SEARCHING
        s = _make_sensory(carrying=None)
        brain.decide(s)
        assert brain.forager_state is ForagerState.SEARCHING


# =========================================================================
# 2. Forager action outputs
# =========================================================================

class TestForagerActions:

    def test_searching_deposits_home_pheromone(self):
        brain = _brain()
        s = _make_sensory()
        action = brain.decide(s)
        assert action.deposit_pheromone == "home"
        assert action.deposit_strength > 0

    def test_returning_deposits_food_pheromone(self):
        brain = _brain()
        brain.forager_state = ForagerState.RETURNING

        s = _make_sensory(
            carrying="food", carry_amount=0.6,
            nest_distance=200.0,
            antenna_left={"food": 0.0, "home": 0.3, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.deposit_pheromone == "food"
        # deposit_strength = min(1.0, carry_amount + 0.3) = 0.9
        assert action.deposit_strength == pytest.approx(0.9, abs=0.01)

    def test_harvesting_tries_pickup(self):
        brain = _brain()
        brain.forager_state = ForagerState.HARVESTING

        s = _make_sensory(
            antenna_left={"food": 0.5, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.pickup is True

    def test_depositing_drops_cargo(self):
        brain = _brain()
        brain.forager_state = ForagerState.DEPOSITING

        s = _make_sensory(carrying="food", carry_amount=0.5)
        action = brain.decide(s)
        assert action.drop is True

    def test_harvesting_recruitment_on_rich_food(self):
        brain = _brain()
        brain.forager_state = ForagerState.HARVESTING

        s = _make_sensory(
            antenna_left={"food": _RICH_FOOD_THRESHOLD + 0.1, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.recruit_signal is True

    def test_harvesting_no_recruit_on_weak_food(self):
        brain = _brain()
        brain.forager_state = ForagerState.HARVESTING

        s = _make_sensory(
            antenna_left={"food": _RICH_FOOD_THRESHOLD - 0.2, "home": 0.0, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": _RICH_FOOD_THRESHOLD - 0.2, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.recruit_signal is False

    def test_returning_recruitment_on_heavy_load(self):
        brain = _brain()
        brain.forager_state = ForagerState.RETURNING

        s = _make_sensory(
            carrying="food", carry_amount=_RICH_FOOD_THRESHOLD + 0.1,
            nest_distance=200.0,
            antenna_left={"food": 0.0, "home": 0.3, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.recruit_signal is True


# =========================================================================
# 3. Braitenberg pheromone following
# =========================================================================

class TestBraitenbergSteering:

    def test_food_stronger_right_turns_right(self):
        brain = _brain()
        s = _make_sensory(
            antenna_left={"food": 0.1, "home": 0.0, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.5, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.turn > 0, "Should turn right toward stronger food on right"

    def test_food_stronger_left_turns_left(self):
        brain = _brain()
        s = _make_sensory(
            antenna_left={"food": 0.5, "home": 0.0, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.1, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.turn < 0, "Should turn left toward stronger food on left"

    def test_equal_antennae_approximately_straight(self):
        brain = _brain()
        # Force wander bias to zero for determinism
        brain._wander_bias = 0.0
        brain._rng = __import__("random").Random(0)

        s = _make_sensory(
            antenna_left={"food": 0.3, "home": 0.0, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.3, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        # With equal signals, turn should be small (only wander noise)
        assert abs(action.turn) < math.pi / 6

    def test_home_following_while_returning(self):
        brain = _brain()
        brain.forager_state = ForagerState.RETURNING

        s = _make_sensory(
            carrying="food", carry_amount=0.5,
            nest_distance=200.0,
            nest_bearing=0.3,
            antenna_left={"food": 0.0, "home": 0.1, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.0, "home": 0.6, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.turn > 0, "Should turn toward nest (nest_bearing > 0)"

    def test_danger_following_for_soldier(self):
        brain = _brain()
        brain.soldier_state = SoldierState.RESPONDING

        s = _make_sensory(
            role_value="soldier",
            antenna_left={"food": 0.0, "home": 0.0, "danger": 0.6, "recruit": 0.0},
            antenna_right={"food": 0.0, "home": 0.0, "danger": 0.1, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.turn < 0, "Should turn left toward stronger danger on left"

    def test_turn_clamped_to_max(self):
        brain = _brain()
        s = _make_sensory(
            antenna_left={"food": 0.0, "home": 0.0, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 1.0, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert abs(action.turn) <= math.pi / 6 + 1e-9


# =========================================================================
# 4. Obstacle avoidance
# =========================================================================

class TestObstacleAvoidance:

    def test_no_avoidance_when_clear(self):
        brain = _brain()
        turn = brain._obstacle_avoidance_turn(
            _make_sensory(obstacle_rays=[1.0, 1.0, 1.0, 1.0, 1.0]),
        )
        assert turn == pytest.approx(0.0)

    def test_obstacle_left_turns_right(self):
        brain = _brain()
        # Obstacle in leftmost ray (index 0)
        turn = brain._obstacle_avoidance_turn(
            _make_sensory(obstacle_rays=[0.2, 0.9, 1.0, 1.0, 1.0]),
        )
        assert turn > 0, "Should turn right to avoid obstacle on left"

    def test_obstacle_right_turns_left(self):
        brain = _brain()
        # Obstacle in rightmost ray (index 4)
        turn = brain._obstacle_avoidance_turn(
            _make_sensory(obstacle_rays=[1.0, 1.0, 1.0, 0.9, 0.2]),
        )
        assert turn < 0, "Should turn left to avoid obstacle on right"

    def test_closer_obstacle_stronger_turn(self):
        brain = _brain()
        # Use values close to threshold so turns don't saturate at max
        mild = brain._obstacle_avoidance_turn(
            _make_sensory(obstacle_rays=[0.55, 1.0, 1.0, 1.0, 1.0]),
        )
        strong = brain._obstacle_avoidance_turn(
            _make_sensory(obstacle_rays=[0.4, 1.0, 1.0, 1.0, 1.0]),
        )
        assert abs(strong) > abs(mild)

    def test_centre_obstacle_produces_nonzero_turn(self):
        brain = _brain()
        turn = brain._obstacle_avoidance_turn(
            _make_sensory(obstacle_rays=[1.0, 1.0, 0.2, 1.0, 1.0]),
        )
        assert abs(turn) > 0.01, "Centre obstacle should produce a nudge turn"

    def test_obstacle_avoidance_overrides_pheromone(self):
        """When obstacles are close, avoidance dominates over food following."""
        brain = _brain()
        s = _make_sensory(
            obstacle_rays=[0.1, 0.2, 0.3, 1.0, 1.0],
            antenna_left={"food": 0.8, "home": 0.0, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        # The obstacle is on the left, so avoidance should push right (+)
        assert action.turn > 0

    def test_empty_rays_no_crash(self):
        brain = _brain()
        turn = brain._obstacle_avoidance_turn(_make_sensory(obstacle_rays=[]))
        assert turn == 0.0


# =========================================================================
# 5. Soldier behaviour
# =========================================================================

class TestSoldierBehaviour:

    def test_starts_patrolling(self):
        brain = _brain()
        assert brain.soldier_state is SoldierState.PATROLLING

    def test_transitions_to_responding_on_danger(self):
        brain = _brain()
        s = _make_sensory(
            role_value="soldier",
            antenna_left={"food": 0.0, "home": 0.0, "danger": _DANGER_THRESHOLD + 0.1, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.soldier_state is SoldierState.RESPONDING

    def test_stays_patrolling_below_threshold(self):
        brain = _brain()
        s = _make_sensory(
            role_value="soldier",
            antenna_left={"food": 0.0, "home": 0.0, "danger": _DANGER_THRESHOLD * 0.3, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.soldier_state is SoldierState.PATROLLING

    def test_returns_to_patrolling_when_danger_subsides(self):
        brain = _brain()
        brain.soldier_state = SoldierState.RESPONDING

        s = _make_sensory(
            role_value="soldier",
            antenna_left={"food": 0.0, "home": 0.0, "danger": _DANGER_THRESHOLD * 0.3, "recruit": 0.0},
        )
        brain.decide(s)
        assert brain.soldier_state is SoldierState.PATROLLING

    def test_responding_deposits_danger_pheromone(self):
        brain = _brain()
        brain.soldier_state = SoldierState.RESPONDING

        s = _make_sensory(
            role_value="soldier",
            antenna_left={"food": 0.0, "home": 0.0, "danger": 0.5, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.deposit_pheromone == "danger"
        assert action.deposit_strength > 0

    def test_responding_moves_at_full_speed(self):
        brain = _brain()
        brain.soldier_state = SoldierState.RESPONDING

        s = _make_sensory(
            role_value="soldier",
            antenna_left={"food": 0.0, "home": 0.0, "danger": 0.5, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.speed_mult == 1.0

    def test_patrolling_orbits_at_moderate_speed(self):
        brain = _brain()
        s = _make_sensory(
            role_value="soldier",
            nest_distance=_PATROL_RADIUS,
        )
        action = brain.decide(s)
        assert action.speed_mult == pytest.approx(0.8)

    def test_patrolling_turns_inward_when_far(self):
        brain = _brain()
        s = _make_sensory(
            role_value="soldier",
            nest_distance=_PATROL_RADIUS * 2.0,
            nest_bearing=-0.5,
            antenna_left={"food": 0.0, "home": 0.4, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.0, "home": 0.1, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        # Should turn toward nest (nest_bearing < 0 → turn left)
        assert action.turn < 0


# =========================================================================
# 6. Nurse behaviour
# =========================================================================

class TestNurseBehaviour:

    def test_starts_tending(self):
        brain = _brain()
        assert brain.nurse_state is NurseState.TENDING

    def test_transitions_to_cleaning_on_corpse(self):
        brain = _brain()
        s = _make_sensory(role_value="nurse", carrying="corpse")
        brain.decide(s)
        assert brain.nurse_state is NurseState.CLEANING

    def test_transitions_to_tending_on_brood(self):
        brain = _brain()
        brain.nurse_state = NurseState.CLEANING

        s = _make_sensory(role_value="nurse", carrying="brood")
        brain.decide(s)
        assert brain.nurse_state is NurseState.TENDING

    def test_transitions_to_tending_when_empty_near_nest(self):
        brain = _brain()
        brain.nurse_state = NurseState.CLEANING

        s = _make_sensory(
            role_value="nurse",
            carrying=None,
            nest_distance=_NEST_ARRIVE_DISTANCE,
        )
        brain.decide(s)
        assert brain.nurse_state is NurseState.TENDING

    def test_tending_drops_brood_at_nest_center(self):
        brain = _brain()
        s = _make_sensory(
            role_value="nurse",
            carrying="brood",
            nest_distance=_NEST_ARRIVE_DISTANCE * 0.3,
        )
        action = brain.decide(s)
        assert action.drop is True

    def test_tending_carries_brood_toward_nest(self):
        brain = _brain()
        s = _make_sensory(
            role_value="nurse",
            carrying="brood",
            nest_distance=200.0,
            nest_bearing=0.5,
            antenna_left={"food": 0.0, "home": 0.1, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.0, "home": 0.5, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.turn > 0, "Should turn toward nest (nest_bearing > 0)"
        assert action.drop is False

    def test_tending_tries_pickup_when_not_carrying(self):
        brain = _brain()
        s = _make_sensory(
            role_value="nurse",
            carrying=None,
            nest_distance=30.0,
        )
        action = brain.decide(s)
        assert action.pickup is True

    def test_cleaning_drops_corpse_far_from_nest(self):
        brain = _brain()
        brain.nurse_state = NurseState.CLEANING

        diag = math.hypot(1600, 1000)
        s = _make_sensory(
            role_value="nurse",
            carrying="corpse",
            nest_distance=_CEMETERY_DROP_DISTANCE + 10,
        )
        action = brain.decide(s)
        assert action.drop is True

    def test_cleaning_moves_away_from_nest(self):
        brain = _brain()
        brain.nurse_state = NurseState.CLEANING

        s = _make_sensory(
            role_value="nurse",
            carrying="corpse",
            nest_distance=100.0,
            nest_bearing=-0.5,
            antenna_left={"food": 0.0, "home": 0.5, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.0, "home": 0.1, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        # nest_bearing=-0.5, away_bearing = -0.5+π ≈ 2.64 → clamped positive turn
        assert action.turn > 0, "Should move away from nest"

    def test_cleaning_tries_pickup_when_not_carrying(self):
        brain = _brain()
        brain.nurse_state = NurseState.CLEANING

        s = _make_sensory(role_value="nurse", carrying=None, nest_distance=200.0)
        action = brain.decide(s)
        assert action.pickup is True


# =========================================================================
# 7. Role rebalancing
# =========================================================================

class TestRoleRebalancing:

    def test_baseline_distribution(self):
        dist = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
        )
        assert set(dist.keys()) == {"forager", "nurse", "soldier", "idle"}
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)

    def test_low_food_boosts_foragers(self):
        low = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=0.1, brood_count=5, threat_level=0.0,
        )
        normal = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
        )
        assert low["forager"] > normal["forager"]

    def test_high_food_reduces_foragers(self):
        high = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=3.0, brood_count=5, threat_level=0.0,
        )
        normal = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
        )
        assert high["forager"] < normal["forager"]

    def test_high_brood_boosts_nurses(self):
        high_brood = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=20, threat_level=0.0,
        )
        no_brood = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=0, threat_level=0.0,
        )
        assert high_brood["nurse"] > no_brood["nurse"]

    def test_high_threat_boosts_soldiers(self):
        threat = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.8,
        )
        calm = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
        )
        assert threat["soldier"] > calm["soldier"]

    def test_moderate_threat_boosts_soldiers(self):
        moderate = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.3,
        )
        calm = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
        )
        assert moderate["soldier"] > calm["soldier"]

    def test_all_fractions_at_least_5_percent(self):
        dist = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=0.0, brood_count=0, threat_level=1.0,
        )
        for role, frac in dist.items():
            assert frac >= 0.04, f"{role} fraction {frac} is below floor"

    def test_smoothing_toward_current(self):
        current = {"forager": 0.7, "nurse": 0.1, "soldier": 0.1, "idle": 0.1}
        dist = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
            current_distribution=current,
        )
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
        # Smoothing should pull forager toward the current high value
        raw = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=1.0, brood_count=5, threat_level=0.0,
        )
        assert dist["forager"] > raw["forager"]

    def test_extreme_inputs_still_normalised(self):
        dist = RuleBasedBrain.suggest_role_distribution(
            food_income_rate=0.0, brood_count=100, threat_level=1.0,
        )
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)


# =========================================================================
# 8. Idle behaviour
# =========================================================================

class TestIdleBehaviour:

    def test_idle_slow_near_nest(self):
        brain = _brain()
        s = _make_sensory(role_value="idle", nest_distance=20.0)
        action = brain.decide(s)
        assert action.speed_mult <= 0.3

    def test_idle_moves_toward_nest_when_far(self):
        brain = _brain()
        s = _make_sensory(
            role_value="idle",
            nest_distance=_NEST_ARRIVE_DISTANCE * 5,
            nest_bearing=-0.5,
            antenna_left={"food": 0.0, "home": 0.5, "danger": 0.0, "recruit": 0.0},
            antenna_right={"food": 0.0, "home": 0.1, "danger": 0.0, "recruit": 0.0},
        )
        action = brain.decide(s)
        assert action.turn < 0, "Should turn toward nest (nest_bearing < 0)"


# =========================================================================
# 9. Protocol compliance
# =========================================================================

class TestProtocol:

    def test_satisfies_brain_backend(self):
        from brains.interface import BrainBackend
        brain = _brain()
        assert isinstance(brain, BrainBackend)

    def test_learn_is_noop(self):
        brain = _brain()
        brain.learn(1.0)
        brain.learn(-1.0)
        # Should not raise

    def test_action_values_within_bounds(self):
        brain = _brain()
        for role in ("forager", "nurse", "soldier", "idle"):
            s = _make_sensory(role_value=role)
            action = brain.decide(s)
            clamped = action.clamped()
            assert -math.pi / 6 - 1e-9 <= clamped.turn <= math.pi / 6 + 1e-9
            assert 0.0 <= clamped.speed_mult <= 1.0
            assert 0.0 <= clamped.deposit_strength <= 1.0


# =========================================================================
# 10. Edge cases
# =========================================================================

class TestEdgeCases:

    def test_missing_antenna_channels(self):
        brain = _brain()
        s = _make_sensory(antenna_left={}, antenna_right={})
        action = brain.decide(s)
        assert action is not None  # should not crash

    def test_single_obstacle_ray(self):
        brain = _brain()
        s = _make_sensory(obstacle_rays=[0.3])
        action = brain.decide(s)
        assert action is not None

    def test_zero_energy(self):
        brain = _brain()
        s = _make_sensory(energy=0.0)
        action = brain.decide(s)
        assert action is not None

    def test_role_dispatch_unknown_role(self):
        brain = _brain()
        s = _make_sensory(role_value="queen")
        action = brain.decide(s)
        # Falls through to idle
        assert action is not None
