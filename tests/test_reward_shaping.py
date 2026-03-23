from __future__ import annotations

import pytest

from moral_harvest.rewards.gini import gini_coefficient
from moral_harvest.rewards.shaping import (
    RewardShaper,
    RewardShapingConfig,
    compute_effective_alpha,
)


def test_gini_coefficient_basic_cases() -> None:
    assert gini_coefficient([1.0, 1.0, 1.0]) == 0.0
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0
    assert gini_coefficient([0.0, 0.0, 10.0]) > 0.6


def test_utilitarian_shaping_uses_mean_reward() -> None:
    shaper = RewardShaper(RewardShapingConfig(reward_type="utilitarian", alpha=0.25))
    own_rewards = {"player_0": 1.0, "player_1": 3.0}

    shaped_rewards, metrics = shaper.shape_step(own_rewards)

    # mean own = 2.0; shaped = 0.25*own + 0.75*2.0
    assert shaped_rewards["player_0"] == 1.75
    assert shaped_rewards["player_1"] == 2.25
    assert metrics["utilitarian_mean_reward"] == 2.0


def test_deontological_neighbor_bins_follow_spec() -> None:
    shaper = RewardShaper(
        RewardShapingConfig(
            reward_type="deontological",
            alpha=0.0,
            deontological_max_bonus=10.0,
        )
    )
    own_rewards = {
        "player_0": 0.0,
        "player_1": 0.0,
        "player_2": 0.0,
        "player_3": 0.0,
    }
    infos = {
        "player_0": {"nearby_apples": 0},
        "player_1": {"nearby_apples": 1},
        "player_2": {"nearby_apples": 2},
        "player_3": {"nearby_apples": 5},
    }

    shaped_rewards, _ = shaper.shape_step(own_rewards, infos)

    assert shaped_rewards["player_0"] == 0.0
    assert shaped_rewards["player_1"] == 1.0
    assert shaped_rewards["player_2"] == 2.0
    assert shaped_rewards["player_3"] == 10.0


def test_virtue_bonus_positive_when_gini_drops() -> None:
    shaper = RewardShaper(RewardShapingConfig(reward_type="virtue", alpha=0.0, virtue_scale=1.0))

    # First call seeds previous gini and returns zero delta bonus.
    first_shaped, first_metrics = shaper.shape_step({"player_0": 0.0, "player_1": 10.0})
    assert first_shaped["player_0"] == 0.0
    assert first_metrics["virtue_delta_gini"] == 0.0

    # More equal rewards -> lower gini -> negative delta -> positive bonus.
    second_shaped, second_metrics = shaper.shape_step({"player_0": 5.0, "player_1": 5.0})
    assert second_metrics["virtue_delta_gini"] is not None
    assert second_metrics["virtue_delta_gini"] < 0.0
    assert second_shaped["player_0"] > 0.0
    assert second_shaped["player_1"] > 0.0


def test_alpha_override_changes_blend_for_step() -> None:
    shaper = RewardShaper(RewardShapingConfig(reward_type="utilitarian", alpha=0.25))
    own_rewards = {"player_0": 1.0, "player_1": 3.0}

    shaped_own, _ = shaper.shape_step(own_rewards, alpha_override=1.0)
    shaped_collective, _ = shaper.shape_step(own_rewards, alpha_override=0.0)

    assert shaped_own == own_rewards
    assert shaped_collective["player_0"] == 2.0
    assert shaped_collective["player_1"] == 2.0


def test_alpha_override_validation() -> None:
    shaper = RewardShaper(RewardShapingConfig(reward_type="utilitarian", alpha=0.5))

    with pytest.raises(ValueError, match="alpha must be in \[0, 1\]"):
        shaper.shape_step({"player_0": 1.0}, alpha_override=1.5)


def test_compute_effective_alpha_schedule_boundaries() -> None:
    assert compute_effective_alpha(
        base_alpha=0.5,
        global_step=39,
        shaping_begin=40,
        rew_shaping_horizon=10,
    ) == 1.0
    assert compute_effective_alpha(
        base_alpha=0.5,
        global_step=40,
        shaping_begin=40,
        rew_shaping_horizon=10,
    ) == 1.0
    assert compute_effective_alpha(
        base_alpha=0.5,
        global_step=45,
        shaping_begin=40,
        rew_shaping_horizon=10,
    ) == 0.5
    assert compute_effective_alpha(
        base_alpha=0.5,
        global_step=50,
        shaping_begin=40,
        rew_shaping_horizon=10,
    ) == 0.0


def test_compute_effective_alpha_uses_base_when_schedule_disabled() -> None:
    assert compute_effective_alpha(
        base_alpha=0.3,
        global_step=100,
        shaping_begin=None,
        rew_shaping_horizon=None,
    ) == 0.3


def test_compute_effective_alpha_requires_both_schedule_params() -> None:
    with pytest.raises(ValueError, match="Both shaping_begin and rew_shaping_horizon"):
        compute_effective_alpha(
            base_alpha=0.5,
            global_step=0,
            shaping_begin=10,
            rew_shaping_horizon=None,
        )
