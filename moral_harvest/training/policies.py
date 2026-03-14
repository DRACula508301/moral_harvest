from __future__ import annotations

from collections.abc import Callable

from gymnasium import Space
from ray.rllib.policy.policy import PolicySpec


# Build one distinct policy per agent for vanilla selfish IPPO.
def build_distinct_policies(
    num_agents: int,
    observation_space: Space,
    action_space: Space,
) -> tuple[dict[str, PolicySpec], Callable[..., str]]:
    # Create a policy specification per player agent.
    policies: dict[str, PolicySpec] = {}
    for index in range(num_agents):
        agent_id = f"player_{index}"
        policy_id = f"policy_{agent_id}"
        policies[policy_id] = PolicySpec(
            observation_space=observation_space,
            action_space=action_space,
            config={},
        )

    # Map each environment agent to its own dedicated policy.
    def policy_mapping_fn(agent_id: str, *args, **kwargs) -> str:
        return f"policy_{agent_id}"

    return policies, policy_mapping_fn
