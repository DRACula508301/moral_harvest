from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig

from moral_harvest.envs.meltingpot_multiagent_env import HarvestMultiAgentEnv
from moral_harvest.envs.registry import HARVEST_MULTI_AGENT_ENV_ID, register_environments
from moral_harvest.training.config import SingleAgentTrainConfig
from moral_harvest.training.policies import build_distinct_policies
from moral_harvest.training.results_logger import IterationResultsWriter


# Determine RLlib GPU resource count from availability and optional override.
def _resolve_num_gpus(num_gpus_override: int | None) -> int:
    # Auto mode uses one GPU when CUDA is available.
    if num_gpus_override is None:
        return 1 if torch.cuda.is_available() else 0

    # Explicit non-positive override disables GPU.
    if num_gpus_override <= 0:
        return 0

    # Positive override is respected only if CUDA exists.
    if torch.cuda.is_available():
        return num_gpus_override
    return 0


# Safely read nested dictionary values with a default fallback.
def _lookup(d: dict[str, Any], keys: list[str], default: float | None = None):
    # Walk through nested keys and return default on any missing path segment.
    value: Any = d
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


# Extract high-level metrics from RLlib multi-agent training result payload.
def _extract_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    # Resolve reward metric across possible RLlib schema variants.
    episode_reward = (
        _lookup(result, ["env_runners", "episode_return_mean"])
        or _lookup(result, ["episode_reward_mean"])
        or _lookup(result, ["evaluation", "env_runners", "episode_return_mean"])
    )

    return {
        "episode_reward_mean": float(episode_reward) if episode_reward is not None else None,
    }


# Train vanilla selfish IPPO: each agent uses its own policy.
def run_vanilla_selfish_ippo(cfg: SingleAgentTrainConfig) -> dict[str, Any]:
    # Ensure custom environments are available before building the algorithm.
    register_environments()

    # Start Ray runtime for RLlib execution.
    ray.init(ignore_reinit_error=True)

    # Build common env config for policy and trainer setup.
    env_config = {
        "substrate_name": cfg.substrate_name,
        "num_agents": cfg.num_agents,
        "no_op_action": cfg.no_op_action,
        "include_ready_to_shoot": cfg.include_ready_to_shoot,
    }

    # Build temporary env to infer per-agent spaces for policy specs.
    probe_env = HarvestMultiAgentEnv(env_config)
    observation_space = probe_env.get_observation_space("player_0")
    action_space = probe_env.get_action_space("player_0")
    probe_env.close()

    # Create one policy per agent and deterministic mapping.
    policies, policy_mapping_fn = build_distinct_policies(
        num_agents=cfg.num_agents,
        observation_space=observation_space,
        action_space=action_space,
    )

    # Resolve compute resources: GPU when available, else CPU.
    resolved_num_gpus = _resolve_num_gpus(cfg.num_gpus)
    print(
        f"backend=rllib | mode=multi-agent-selfish | framework={cfg.framework} | num_gpus={resolved_num_gpus}"
    )

    # Build PPOConfig for multi-agent selfish IPPO.
    ppo_config = (
        PPOConfig()
        .framework(cfg.framework)
        .environment(
            env=HARVEST_MULTI_AGENT_ENV_ID,
            env_config=env_config,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(num_env_runners=cfg.num_env_runners)
        .training(
            train_batch_size=cfg.train_batch_size,
            minibatch_size=cfg.minibatch_size,
            num_epochs=cfg.num_epochs,
            lr=cfg.lr,
            gamma=cfg.gamma,
        )
        .rl_module(
            model_config={
                "conv_filters": cfg.conv_filters,
                "conv_activation": cfg.conv_activation,
                "fcnet_hiddens": cfg.fcnet_hiddens,
                "fcnet_activation": cfg.fcnet_activation,
            }
        )
        .resources(num_gpus=resolved_num_gpus)
    )

    # Optionally set deterministic seed handling for reproducibility.
    if cfg.seed is not None:
        ppo_config = ppo_config.debugging(seed=cfg.seed)

    # Instantiate the algorithm and create checkpoint output directory.
    algo = ppo_config.build_algo()

    checkpoint_root = Path(cfg.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Build deterministic results artifact directory for per-iteration metrics.
    run_name = cfg.run_name or f"rllib_multi_agent_selfish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(cfg.results_root).resolve() / "rllib" / run_name
    results_writer = IterationResultsWriter(results_dir)

    training_history: list[dict[str, Any]] = []

    try:
        # Main training loop with per-iteration metric extraction and logging.
        for iteration in range(1, cfg.stop_iters + 1):
            result = algo.train()
            metrics = _extract_metrics(result)
            metrics["iteration"] = iteration
            metrics["backend"] = "rllib"
            metrics["mode"] = "multi-agent-selfish"
            metrics["run_name"] = run_name
            training_history.append(metrics)
            results_writer.write(metrics)

            print(
                " | ".join(
                    [
                        f"iter={iteration}",
                        f"reward={metrics['episode_reward_mean']}",
                    ]
                )
            )

            # Persist checkpoints at the configured interval.
            if iteration % cfg.checkpoint_every == 0:
                checkpoint_result = algo.save(str(checkpoint_root))
                checkpoint_path = getattr(checkpoint_result, "checkpoint", checkpoint_result)
                print(f"checkpoint_saved={checkpoint_path}")

        # Save one final checkpoint and return structured training output.
        final_checkpoint_result = algo.save(str(checkpoint_root))
        final_checkpoint_path = getattr(final_checkpoint_result, "checkpoint", final_checkpoint_result)

        return {
            "status": "completed",
            "backend": "rllib",
            "mode": "multi-agent-selfish",
            "num_agents": cfg.num_agents,
            "run_name": run_name,
            "results_dir": str(results_dir),
            "iterations": cfg.stop_iters,
            "final_checkpoint": str(final_checkpoint_path),
            "history": training_history,
        }
    finally:
        # Ensure results files are closed before shutdown.
        results_writer.close()

        # Always release algorithm and Ray runtime resources.
        algo.stop()
        ray.shutdown()
