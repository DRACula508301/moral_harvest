from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks


# Collect reward-shaping diagnostics from infos and expose aggregated custom metrics.
class MoralHarvestMetricsCallbacks(DefaultCallbacks):
    METRIC_KEYS = (
        "own_reward_mean",
        "shaped_reward_mean",
        "shaping_reward_mean",
        "utilitarian_mean_reward",
        "deontological_bonus_mean",
        "virtue_current_gini",
        "virtue_delta_gini",
    )

    def on_episode_start(self, *, episode, **kwargs) -> None:  # noqa: ANN001
        for key in self.METRIC_KEYS:
            episode.user_data[f"reward_shaping_{key}"] = []

    def on_episode_step(self, *, episode, **kwargs) -> None:  # noqa: ANN001
        infos = self._safe_get_infos(episode)
        if not infos:
            return

        for agent_info in infos.values():
            if not isinstance(agent_info, Mapping):
                continue
            shaping = agent_info.get("reward_shaping")
            if not isinstance(shaping, Mapping):
                continue

            for key in self.METRIC_KEYS:
                value = shaping.get(key)
                if isinstance(value, (float, int)):
                    episode.user_data[f"reward_shaping_{key}"].append(float(value))

    def on_episode_end(self, *, episode, **kwargs) -> None:  # noqa: ANN001
        for key in self.METRIC_KEYS:
            values = episode.user_data.get(f"reward_shaping_{key}", [])
            if values:
                episode.custom_metrics[f"reward_shaping/{key}"] = float(mean(values))

    # Safely fetch per-agent info dict across RLlib episode variants.
    def _safe_get_infos(self, episode) -> dict[str, dict[str, Any]]:  # noqa: ANN001
        get_infos = getattr(episode, "get_infos", None)
        if callable(get_infos):
            infos = get_infos()
            if isinstance(infos, Mapping):
                return {
                    str(agent_id): dict(info) if isinstance(info, Mapping) else {}
                    for agent_id, info in infos.items()
                }
        return {}
