from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from shimmy import MeltingPotCompatibilityV0


class HarvestSingleAgentEnv(gym.Env):
    """Single-agent adapter for Melting Pot Commons Harvest.

    The focal agent is controlled by PPO while all other agents take a fixed no-op
    action (0). This wrapper removes global observations by default and only keeps
    the focal agent's RGB observation.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    # Initialize the single-agent adapter and configure observation filtering.
    def __init__(self, config: dict[str, Any] | None = None):
        # Read wrapper config with safe defaults.
        config = config or {}
        self.substrate_name = config.get("substrate_name", "commons_harvest__open")
        self.focal_agent = config.get("focal_agent", "player_0")
        self.no_op_action = int(config.get("no_op_action", 0))
        self.include_ready_to_shoot = bool(config.get("include_ready_to_shoot", False))
        self.render_mode = config.get("render_mode", None)

        # Construct the underlying PettingZoo-compatible Melting Pot environment.
        self._env = MeltingPotCompatibilityV0(
            substrate_name=self.substrate_name,
            render_mode=self.render_mode,
        )
        self._last_obs: dict[str, Any] | None = None
        self._zero_rgb: np.ndarray | None = None

        # Derive action/observation spaces from the focal agent view.
        base_obs_space = self._env.observation_space(self.focal_agent)
        rgb_space = base_obs_space["RGB"]
        self.action_space = self._env.action_space(self.focal_agent)

        # Expose normalized float32 RGB observations for Torch CNN input.
        normalized_rgb_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=rgb_space.shape,
            dtype=np.float32,
        )

        if self.include_ready_to_shoot:
            self.observation_space = gym.spaces.Dict(
                {
                    "RGB": normalized_rgb_space,
                    "READY_TO_SHOOT": base_obs_space["READY_TO_SHOOT"],
                }
            )
        else:
            self.observation_space = normalized_rgb_space

    # Filter raw multi-field observations to the selected local signal(s).
    def _filter_observation(self, agent_observation: dict[str, Any] | None):
        # If the focal agent is missing (e.g., after termination), emit zero-like obs.
        if agent_observation is None:
            if self._zero_rgb is None:
                rgb_shape = self.observation_space.shape if not self.include_ready_to_shoot else self.observation_space["RGB"].shape
                self._zero_rgb = np.zeros(rgb_shape, dtype=np.float32)
            if self.include_ready_to_shoot:
                return {"RGB": self._zero_rgb, "READY_TO_SHOOT": np.array(0.0, dtype=np.float32)}
            return self._zero_rgb

        # Keep only local fields used by the policy.
        if self.include_ready_to_shoot:
            return {
                "RGB": agent_observation["RGB"].astype(np.float32) / 255.0,
                "READY_TO_SHOOT": np.array(agent_observation["READY_TO_SHOOT"], dtype=np.float32),
            }

        return agent_observation["RGB"].astype(np.float32) / 255.0

    # Reset the environment and return the filtered focal-agent observation.
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Support deterministic resets when a seed is provided.
        if seed is not None:
            self._env.reset(seed=seed)

        # Reset and cache latest full-agent observation dictionary.
        observations, infos = self._env.reset(options=options)
        self._last_obs = observations
        focal_obs = self._filter_observation(observations.get(self.focal_agent))
        info = infos.get(self.focal_agent, {})
        return focal_obs, info

    # Step with focal action while non-focal agents take no-op actions.
    def step(self, action):
        # Ensure reset has been called before the first step.
        if self._last_obs is None:
            raise RuntimeError("Environment must be reset before step().")

        # Build a joint action dict: no-op for others, learned action for focal agent.
        active_agents = list(self._last_obs.keys())
        joint_actions = {agent_id: self.no_op_action for agent_id in active_agents}
        if self.focal_agent in joint_actions:
            joint_actions[self.focal_agent] = int(action)

        # Advance the environment and cache the next full observation map.
        observations, rewards, terminations, truncations, infos = self._env.step(joint_actions)
        self._last_obs = observations

        # Extract focal-agent transition fields.
        focal_obs = self._filter_observation(observations.get(self.focal_agent))
        reward = float(rewards.get(self.focal_agent, 0.0))

        terminated = bool(terminations.get(self.focal_agent, False))
        truncated = bool(truncations.get(self.focal_agent, False))

        # Fallback to global done when focal flags are unavailable.
        if not terminated and not truncated and terminations:
            terminated = all(bool(v) for v in terminations.values())
        if not truncated and truncations:
            truncated = all(bool(v) for v in truncations.values())

        # Attach additional diagnostics for downstream logging.
        info = infos.get(self.focal_agent, {})
        info["collective_reward"] = rewards.get("COLLECTIVE_REWARD", None)
        return focal_obs, reward, terminated, truncated, info

    # Cleanly close underlying environment resources.
    def close(self):
        self._env.close()

    # Render an RGB frame for optional video capture.
    def render(self):
        return self._env.render()
