from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig

from moral_harvest.envs.registry import HARVEST_SINGLE_AGENT_ENV_ID, register_environments
from moral_harvest.training.config import SingleAgentTrainConfig
from moral_harvest.training.results_logger import IterationResultsWriter


# Safely read nested dictionary values with a default fallback.
def _lookup(d: dict[str, Any], keys: list[str], default: float | None = None):
    # Walk through nested keys and return default on any missing path segment.
    value: Any = d
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


# Normalize RLlib result payloads into a compact metric dictionary.
def _extract_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    # Pull learner-specific stats for optimization diagnostics.
    learner_stats = (
        _lookup(result, ["learners", "default_policy"], default={})
        or _lookup(result, ["info", "learner", "default_policy", "learner_stats"], default={})
        or {}
    )

    # Resolve reward metric across possible RLlib schema variants.
    episode_reward = (
        _lookup(result, ["env_runners", "episode_return_mean"])
        or _lookup(result, ["episode_reward_mean"])
        or _lookup(result, ["evaluation", "env_runners", "episode_return_mean"])
    )

    # Resolve key optimization losses and exploration entropy.
    policy_loss = (
        learner_stats.get("policy_loss")
        if isinstance(learner_stats, dict)
        else None
    )
    vf_loss = (
        learner_stats.get("vf_loss")
        if isinstance(learner_stats, dict)
        else None
    )
    entropy = (
        learner_stats.get("entropy")
        if isinstance(learner_stats, dict)
        else None
    )

    # Return metrics in a stable shape for JSON logging.
    return {
        "episode_reward_mean": float(episode_reward) if episode_reward is not None else None,
        "policy_loss": float(policy_loss) if policy_loss is not None else None,
        "value_loss": float(vf_loss) if vf_loss is not None else None,
        "entropy": float(entropy) if entropy is not None else None,
    }


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


# Run single-agent PPO training and checkpoint at a fixed iteration cadence.
def run_single_agent_ppo(cfg: SingleAgentTrainConfig) -> dict[str, Any]:
    # Ensure custom RLlib environments are available before building the algorithm.
    register_environments()

    # Start Ray runtime for RLlib execution.
    ray.init(ignore_reinit_error=True)

    # Resolve compute resources: GPU when available, else CPU.
    resolved_num_gpus = _resolve_num_gpus(cfg.num_gpus)
    print(
        f"backend=rllib | framework={cfg.framework} | num_gpus={resolved_num_gpus}"
    )

    # Build PPOConfig with env, rollout, optimization, and resource settings.
    ppo_config = (
        PPOConfig()
        .framework(cfg.framework)
        .environment(
            env=HARVEST_SINGLE_AGENT_ENV_ID,
            env_config={
                "substrate_name": cfg.substrate_name,
                "focal_agent": cfg.focal_agent,
                "no_op_action": cfg.no_op_action,
                "include_ready_to_shoot": cfg.include_ready_to_shoot,
            },
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
    run_name = cfg.run_name or f"{cfg.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(cfg.results_root).resolve() / cfg.backend / run_name
    results_writer = IterationResultsWriter(results_dir)

    training_history: list[dict[str, Any]] = []

    try:
        # Main training loop with per-iteration metric extraction and logging.
        for iteration in range(1, cfg.stop_iters + 1):
            result = algo.train()
            metrics = _extract_metrics(result)
            metrics["iteration"] = iteration
            metrics["backend"] = cfg.backend
            metrics["run_name"] = run_name
            training_history.append(metrics)
            results_writer.write(metrics)

            print(
                " | ".join(
                    [
                        f"iter={iteration}",
                        f"reward={metrics['episode_reward_mean']}",
                        f"policy_loss={metrics['policy_loss']}",
                        f"value_loss={metrics['value_loss']}",
                        f"entropy={metrics['entropy']}",
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
