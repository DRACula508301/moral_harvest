from __future__ import annotations

from typing import Any

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from moral_harvest.rewards.shaping import RewardShaper, RewardShapingConfig


# RLlib MultiAgentEnv wrapper that replaces raw rewards with shaped rewards.
class RewardShapingMultiAgentWrapper(MultiAgentEnv):
    def __init__(
        self,
        env: MultiAgentEnv,
        shaping_config: RewardShapingConfig,
    ):
        super().__init__()
        self._env = env
        self._shaper = RewardShaper(shaping_config)
        self.last_step_metrics: dict[str, float | None] = {}

        if hasattr(self._env, "possible_agents"):
            self.possible_agents = self._env.possible_agents  # type: ignore[attr-defined]
        if hasattr(self._env, "agents"):
            self.agents = self._env.agents  # type: ignore[attr-defined]

    def get_observation_space(self, agent_id: str):
        if hasattr(self._env, "get_observation_space"):
            return self._env.get_observation_space(agent_id)
        return self._env.observation_space

    def get_action_space(self, agent_id: str):
        if hasattr(self._env, "get_action_space"):
            return self._env.get_action_space(agent_id)
        return self._env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._shaper.reset_episode()
        self.last_step_metrics = {}
        return self._env.reset(seed=seed, options=options)

    def step(self, action_dict: dict[str, int]):
        observations, rewards, terminations, truncations, infos = self._env.step(action_dict)

        env_reward_keys = {"__all__"}
        own_rewards = {
            agent_id: float(value)
            for agent_id, value in rewards.items()
            if agent_id not in env_reward_keys
        }
        if not own_rewards:
            return observations, rewards, terminations, truncations, infos

        shaped_rewards, step_metrics = self._shaper.shape_step(
            own_rewards=own_rewards,
            infos={agent_id: infos.get(agent_id, {}) for agent_id in own_rewards},
        )
        self.last_step_metrics = step_metrics

        out_rewards = dict(rewards)
        for agent_id, shaped_reward in shaped_rewards.items():
            out_rewards[agent_id] = float(shaped_reward)

        out_infos = dict(infos)
        for agent_id in own_rewards:
            agent_info = dict(out_infos.get(agent_id, {}))
            agent_info["reward_shaping"] = dict(step_metrics)
            out_infos[agent_id] = agent_info

        if hasattr(self._env, "agents"):
            self.agents = self._env.agents  # type: ignore[attr-defined]

        return observations, out_rewards, terminations, truncations, out_infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()