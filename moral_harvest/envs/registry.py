from __future__ import annotations

from ray.tune.registry import register_env

from moral_harvest.envs.meltingpot_multiagent_env import HarvestMultiAgentEnv
from moral_harvest.envs.meltingpot_env import HarvestSingleAgentEnv


HARVEST_SINGLE_AGENT_ENV_ID = "moral_harvest_single_agent"
HARVEST_MULTI_AGENT_ENV_ID = "moral_harvest_multi_agent"


# Register all custom environments used by training/evaluation scripts.
def register_environments() -> None:
    # Bind the RLlib env id to the single-agent Harvest wrapper factory.
    register_env(HARVEST_SINGLE_AGENT_ENV_ID, lambda config: HarvestSingleAgentEnv(config))

    # Bind the RLlib env id to the multi-agent Harvest wrapper factory.
    register_env(HARVEST_MULTI_AGENT_ENV_ID, lambda config: HarvestMultiAgentEnv(config))
