from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from shimmy import MeltingPotCompatibilityV0


# Multi-agent Melting Pot Harvest adapter for RLlib IPPO training.
class HarvestMultiAgentEnv(MultiAgentEnv):
    # Initialize the multi-agent wrapper and per-agent spaces.
    def __init__(self, config: dict[str, Any] | None = None):
        # Read wrapper config with defaults.
        config = config or {}
        self.substrate_name = config.get("substrate_name", "commons_harvest__open")
        self.agent_count = int(config.get("num_agents", 10))
        self.no_op_action = int(config.get("no_op_action", 0))
        self.include_ready_to_shoot = bool(config.get("include_ready_to_shoot", False))
        self.render_mode = config.get("render_mode", None)

        # Build canonical agent ID list expected by this environment.
        self.possible_agents = [f"player_{index}" for index in range(self.agent_count)]
        self.agents: list[str] = []

        # Construct underlying Melting Pot environment.
        self._env = MeltingPotCompatibilityV0(
            substrate_name=self.substrate_name,
            render_mode=self.render_mode,
        )

        # Infer single-agent spaces from the first configured player.
        base_obs_space = self._env.observation_space(self.possible_agents[0])
        base_action_space = self._env.action_space(self.possible_agents[0])

        # Expose normalized float RGB observations to policy networks.
        rgb_space = base_obs_space["RGB"]
        normalized_rgb_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=rgb_space.shape,
            dtype=np.float32,
        )

        if self.include_ready_to_shoot:
            self._single_observation_space = gym.spaces.Dict(
                {
                    "RGB": normalized_rgb_space,
                    "READY_TO_SHOOT": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self._single_observation_space = normalized_rgb_space

        self._single_action_space = base_action_space

    # Return the observation space for a specific agent ID.
    def get_observation_space(self, agent_id: str) -> gym.Space:
        return self._single_observation_space

    # Return the action space for a specific agent ID.
    def get_action_space(self, agent_id: str) -> gym.Space:
        return self._single_action_space

    # Filter and normalize a raw single-agent observation.
    def _filter_observation(self, agent_observation: dict[str, Any] | None):
        # If observation is missing, emit zero-valued placeholders.
        if agent_observation is None:
            rgb_shape = (
                self._single_observation_space["RGB"].shape
                if self.include_ready_to_shoot
                else self._single_observation_space.shape
            )
            zero_rgb = np.zeros(rgb_shape, dtype=np.float32)
            if self.include_ready_to_shoot:
                return {
                    "RGB": zero_rgb,
                    "READY_TO_SHOOT": np.array(0.0, dtype=np.float32),
                }
            return zero_rgb

        # Keep only local observation fields needed by policies.
        if self.include_ready_to_shoot:
            return {
                "RGB": agent_observation["RGB"].astype(np.float32) / 255.0,
                "READY_TO_SHOOT": np.array(agent_observation["READY_TO_SHOOT"], dtype=np.float32),
            }

        return agent_observation["RGB"].astype(np.float32) / 255.0

    # Reset and return observations for currently active agents.
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Support deterministic seeding when requested.
        if seed is not None:
            self._env.reset(seed=seed)

        observations, infos = self._env.reset(options=options)

        # Keep only configured players that are currently active.
        self.agents = [agent_id for agent_id in self.possible_agents if agent_id in observations]

        obs_out = {
            agent_id: self._filter_observation(observations.get(agent_id))
            for agent_id in self.agents
        }
        infos_out = {agent_id: infos.get(agent_id, {}) for agent_id in self.agents}
        return obs_out, infos_out

    # Advance one environment step with per-agent actions.
    def step(self, action_dict: dict[str, int]):
        # Fill in no-op for any active agent without provided action.
        joint_actions = {
            agent_id: int(action_dict.get(agent_id, self.no_op_action))
            for agent_id in self.agents
        }

        observations, rewards, terminations, truncations, infos = self._env.step(joint_actions)

        # Determine active agent IDs after transition.
        next_agents = [agent_id for agent_id in self.possible_agents if agent_id in observations]
        all_known_agents = sorted(set(self.agents) | set(next_agents))

        obs_out = {
            agent_id: self._filter_observation(observations.get(agent_id))
            for agent_id in next_agents
        }
        rewards_out = {
            agent_id: float(rewards.get(agent_id, 0.0))
            for agent_id in all_known_agents
        }
        terminations_out = {
            agent_id: bool(terminations.get(agent_id, False))
            for agent_id in all_known_agents
        }
        truncations_out = {
            agent_id: bool(truncations.get(agent_id, False))
            for agent_id in all_known_agents
        }
        infos_out = {
            agent_id: infos.get(agent_id, {})
            for agent_id in all_known_agents
        }

        # Add multi-agent episode completion signals.
        terminations_out["__all__"] = bool(all(terminations_out.values())) if all_known_agents else True
        truncations_out["__all__"] = bool(all(truncations_out.values())) if all_known_agents else False

        self.agents = next_agents
        return obs_out, rewards_out, terminations_out, truncations_out, infos_out

    # Render a frame for optional video recording.
    def render(self):
        return self._env.render()

    # Close the underlying environment.
    def close(self):
        self._env.close()
