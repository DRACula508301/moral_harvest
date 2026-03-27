from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core import Columns
from torch.distributions.categorical import Categorical

from moral_harvest.envs.meltingpot_env import HarvestSingleAgentEnv
from moral_harvest.envs.registry import register_environments
from moral_harvest.training.cnn_actor_critic import CleanRLCNNActorCritic


def _fmt_metric_4dp(value: float) -> str:
    return f"{float(value):.4f}"


# Parse CLI arguments for single-agent checkpoint rollout.
def parse_args() -> argparse.Namespace:
    # Build parser for checkpoint path and rollout controls.
    parser = argparse.ArgumentParser(description="Roll out a single-agent Harvest PPO checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path to checkpoint directory (RLlib) or file (CleanRL .pt)")
    parser.add_argument("--backend", choices=["auto", "rllib", "cleanrl"], default="auto")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--substrate", default="commons_harvest__open")
    parser.add_argument("--focal-agent", default="player_0")
    parser.add_argument("--include-ready-to-shoot", action="store_true")
    parser.add_argument("--no-op-action", type=int, default=0)
    parser.add_argument("--explore", action="store_true", help="Enable exploration during rollout")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-video", action="store_true", help="Record rollout videos")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for videos/<run_name>")
    return parser.parse_args()


# Infer backend from CLI value and checkpoint shape.
def resolve_backend(backend: str, checkpoint_path: Path) -> str:
    # Auto-detect based on checkpoint extension.
    if backend != "auto":
        return backend
    if checkpoint_path.suffix == ".pt":
        return "cleanrl"
    return "rllib"


# Resolve inference device for rollout.
def resolve_device() -> torch.device:
    # Prefer CUDA if available, otherwise use CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build an action function from an RLlib checkpoint.
def build_rllib_policy(checkpoint_path: Path, explore: bool) -> tuple[Callable[[object], int], Algorithm]:
    # Initialize Ray and restore the RLlib algorithm.
    register_environments()
    ray.init(ignore_reinit_error=True)
    algo = Algorithm.from_checkpoint(str(checkpoint_path.resolve()))
    module = algo.get_module("default_policy")
    device = resolve_device()
    module.to(device)

    # Return callable that maps observation to action.
    def action_fn(obs: object) -> int:
        obs_t = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: obs_t})
            logits = out["action_dist_inputs"]
            dist = Categorical(logits=logits)
            if explore:
                action = dist.sample()
            else:
                action = torch.argmax(logits, dim=-1)
        return int(action.item())

    return action_fn, algo


# Build an action function from a CleanRL checkpoint.
def build_cleanrl_policy(
    checkpoint_path: Path,
    envs: HarvestSingleAgentEnv,
    explore: bool,
) -> Callable[[object], int]:
    # Select rollout inference device.
    device = resolve_device()

    # Restore checkpoint and rebuild the Torch policy architecture.
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    cfg = checkpoint.get("config", {})
    conv_filters = cfg.get("conv_filters", [[16, [8, 8], 4], [32, [4, 4], 2], [64, [3, 3], 1]])
    fcnet_hiddens = cfg.get("fcnet_hiddens", [256, 256])

    model = CleanRLCNNActorCritic(
        obs_shape=envs.observation_space.shape,
        action_dim=int(envs.action_space.n),
        conv_filters=conv_filters,
        fcnet_hiddens=fcnet_hiddens,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Return callable that maps observation to action.
    def action_fn(obs: object) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model.forward(obs_t)
            dist = Categorical(logits=logits)
            if explore:
                action = dist.sample()
            else:
                action = torch.argmax(logits, dim=-1)
        return int(action.item())

    return action_fn


# Roll out one episode and return scalar diagnostics.
def run_episode(action_fn: Callable[[object], int], env: HarvestSingleAgentEnv) -> dict[str, float | int]:
    # Reset environment and initialize accumulators.
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    # Step until terminal or truncated signal is reached.
    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = bool(terminated or truncated)

    # Return episode-level metrics.
    return {
        "steps": steps,
        "total_reward": total_reward,
    }


# Entrypoint for evaluating a saved single-agent PPO policy.
def main() -> None:
    # Parse input arguments and validate checkpoint path.
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    # Resolve rollout backend (RLlib or CleanRL).
    backend = resolve_backend(args.backend, checkpoint_path)

    # Build rollout environment with the same wrapper options used in training.
    envs = HarvestSingleAgentEnv(
        {
            "substrate_name": args.substrate,
            "focal_agent": args.focal_agent,
            "include_ready_to_shoot": args.include_ready_to_shoot,
            "no_op_action": args.no_op_action,
            "render_mode": "rgb_array" if args.record_video else None,
        }
    )

    # Optionally wrap the env with gym video recording at videos/<run_name>.
    if args.record_video:
        run_name = args.run_name or f"single-agent-{checkpoint_path.name}"
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    if args.seed is not None:
        envs.reset(seed=args.seed)

    # Build backend-specific policy callable.
    algo: Algorithm | None = None
    if backend == "rllib":
        action_fn, algo = build_rllib_policy(checkpoint_path, explore=args.explore)
    else:
        action_fn = build_cleanrl_policy(checkpoint_path, envs, explore=args.explore)

    episode_results: list[dict[str, float | int]] = []

    try:
        # Run configured number of episodes and print per-episode stats.
        for episode_idx in range(args.episodes):
            result = run_episode(action_fn, envs)
            result["episode"] = episode_idx + 1
            episode_results.append(result)
            print(
                f"episode={result['episode']} steps={result['steps']} total_reward={_fmt_metric_4dp(float(result['total_reward']))}"
            )

        # Emit JSON summary for easy post-processing.
        total_rewards = [float(item["total_reward"]) for item in episode_results]
        summary = {
            "episodes": args.episodes,
            "mean_reward": sum(total_rewards) / max(len(total_rewards), 1),
            "min_reward": min(total_rewards) if total_rewards else None,
            "max_reward": max(total_rewards) if total_rewards else None,
            "results": episode_results,
        }
        print(json.dumps(summary, indent=2))
    finally:
        # Always release resources.
        envs.close()
        if algo is not None:
            algo.stop()
            ray.shutdown()


if __name__ == "__main__":
    main()
