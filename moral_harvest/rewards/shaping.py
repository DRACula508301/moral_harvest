from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from moral_harvest.rewards.gini import gini_coefficient, gini_delta


RewardType = Literal["selfish", "utilitarian", "deontological", "virtue"]


@dataclass(slots=True)
class RewardShapingConfig:
    reward_type: RewardType = "utilitarian"
    alpha: float = 0.5
    deontological_max_bonus: float = 1.0
    virtue_scale: float = 1.0


# Compute reward-type-specific shaping signals and convexly blend with own rewards.
class RewardShaper:
    def __init__(self, config: RewardShapingConfig):
        self.config = config
        if not (0.0 <= self.config.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if self.config.deontological_max_bonus < 0.0:
            raise ValueError("deontological_max_bonus must be non-negative.")
        if self.config.virtue_scale < 0.0:
            raise ValueError("virtue_scale must be non-negative.")
        self._previous_gini: float | None = None

    # Reset episode-level shaping state.
    def reset_episode(self) -> None:
        self._previous_gini = None

    # Blend own reward and shaping reward.
    def combine_rewards(
        self,
        own_rewards: dict[str, float],
        shaping_rewards: dict[str, float],
    ) -> dict[str, float]:
        alpha = self.config.alpha
        return {
            agent_id: alpha * float(own_rewards[agent_id]) + (1.0 - alpha) * float(shaping_rewards[agent_id])
            for agent_id in own_rewards
        }

    # Compute shaping rewards and step metrics.
    def compute_shaping_rewards(
        self,
        own_rewards: dict[str, float],
        infos: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, float], dict[str, float | None]]:
        infos = infos or {}
        reward_type = self.config.reward_type

        if reward_type == "selfish":
            shaping_rewards = {agent_id: 0.0 for agent_id in own_rewards}
            step_metrics = {
                "shaping_reward_mean": 0.0,
                "utilitarian_mean_reward": None,
                "deontological_bonus_mean": None,
                "virtue_current_gini": None,
                "virtue_delta_gini": None,
            }
            return shaping_rewards, step_metrics

        if reward_type == "utilitarian":
            mean_reward = float(sum(own_rewards.values()) / max(len(own_rewards), 1))
            shaping_rewards = {agent_id: mean_reward for agent_id in own_rewards}
            step_metrics = {
                "shaping_reward_mean": mean_reward,
                "utilitarian_mean_reward": mean_reward,
                "deontological_bonus_mean": None,
                "virtue_current_gini": None,
                "virtue_delta_gini": None,
            }
            return shaping_rewards, step_metrics

        if reward_type == "deontological":
            shaping_rewards = {
                agent_id: self._deontological_bonus(agent_id, infos.get(agent_id, {}))
                for agent_id in own_rewards
            }
            mean_bonus = float(sum(shaping_rewards.values()) / max(len(shaping_rewards), 1))
            step_metrics = {
                "shaping_reward_mean": mean_bonus,
                "utilitarian_mean_reward": None,
                "deontological_bonus_mean": mean_bonus,
                "virtue_current_gini": None,
                "virtue_delta_gini": None,
            }
            return shaping_rewards, step_metrics

        if reward_type == "virtue":
            current_gini = gini_coefficient(own_rewards)
            if self._previous_gini is None:
                delta = 0.0
            else:
                delta = gini_delta(self._previous_gini, current_gini)
            self._previous_gini = current_gini

            bonus = -delta * self.config.virtue_scale
            shaping_rewards = {agent_id: bonus for agent_id in own_rewards}
            step_metrics = {
                "shaping_reward_mean": bonus,
                "utilitarian_mean_reward": None,
                "deontological_bonus_mean": None,
                "virtue_current_gini": current_gini,
                "virtue_delta_gini": delta,
            }
            return shaping_rewards, step_metrics

        raise ValueError(f"Unsupported reward_type='{reward_type}'.")

    # Compute full shaped reward output and diagnostic metrics for one environment step.
    def shape_step(
        self,
        own_rewards: dict[str, float],
        infos: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, float], dict[str, float | None]]:
        shaping_rewards, step_metrics = self.compute_shaping_rewards(own_rewards=own_rewards, infos=infos)
        shaped_rewards = self.combine_rewards(own_rewards=own_rewards, shaping_rewards=shaping_rewards)

        own_reward_mean = float(sum(own_rewards.values()) / max(len(own_rewards), 1))
        shaped_reward_mean = float(sum(shaped_rewards.values()) / max(len(shaped_rewards), 1))
        step_metrics["own_reward_mean"] = own_reward_mean
        step_metrics["shaped_reward_mean"] = shaped_reward_mean

        return shaped_rewards, step_metrics

    # Apply requested deontological neighbor-count scaling.
    def _deontological_bonus(self, agent_id: str, info: dict[str, Any]) -> float:
        apples_nearby = self._extract_nearby_apple_count(agent_id=agent_id, info=info)
        if apples_nearby <= 0:
            scale = 0.0
        elif apples_nearby == 1:
            scale = 0.1
        elif apples_nearby == 2:
            scale = 0.2
        else:
            scale = 1.0
        return scale * self.config.deontological_max_bonus

    # Read nearby-apple signal from known info keys.
    def _extract_nearby_apple_count(self, agent_id: str, info: dict[str, Any]) -> int:
        for key in ("nearby_apples", "num_nearby_apples", "apples_nearby"):
            if key in info:
                try:
                    return max(0, int(info[key]))
                except (TypeError, ValueError):
                    return 0
        return 0


