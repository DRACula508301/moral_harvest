from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


# Convert mapping/sequence rewards to a dense float array.
def _to_reward_array(rewards: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(rewards, Mapping):
        values = list(rewards.values())
    else:
        values = list(rewards)

    if not values:
        return np.zeros(0, dtype=np.float64)

    reward_array = np.asarray(values, dtype=np.float64)
    if np.any(reward_array < 0.0):
        raise ValueError("Gini is defined for non-negative reward vectors.")
    return reward_array


# Compute Gini coefficient for non-negative rewards.
def gini_coefficient(rewards: Mapping[str, float] | Sequence[float]) -> float:
    reward_array = _to_reward_array(rewards)
    if reward_array.size == 0:
        return 0.0

    total = float(np.sum(reward_array))
    if total <= 0.0:
        return 0.0

    sorted_rewards = np.sort(reward_array)
    n = sorted_rewards.size
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(index * sorted_rewards)) / (n * total) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


# Compute per-step Gini delta as current minus previous.
def gini_delta(previous_gini: float, current_gini: float) -> float:
    return float(current_gini - previous_gini)
