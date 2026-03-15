from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from moral_harvest.envs.meltingpot_multiagent_env import HarvestMultiAgentEnv
from moral_harvest.experiments.single_agent_ppo_cleanrl import (
    CleanRLCNNActorCritic,
    _compute_gae,
    _resolve_device,
)
from moral_harvest.training.config import SingleAgentTrainConfig
from moral_harvest.training.results_logger import IterationResultsWriter


# Build a zero observation tensor matching the configured observation space.
def _zero_observation(obs_shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(obs_shape, dtype=np.float32)


# Train vanilla selfish IPPO with one CleanRL PPO policy per agent.
def run_multi_agent_selfish_cleanrl(cfg: SingleAgentTrainConfig) -> dict[str, Any]:
    if cfg.include_ready_to_shoot:
        raise ValueError(
            "CleanRL backend currently supports RGB-only observations. Omit --include-ready-to-shoot."
        )

    env = HarvestMultiAgentEnv(
        {
            "substrate_name": cfg.substrate_name,
            "num_agents": cfg.num_agents,
            "no_op_action": cfg.no_op_action,
            "include_ready_to_shoot": cfg.include_ready_to_shoot,
        }
    )

    # Infer shared observation/action spaces for each per-agent policy.
    obs_shape = env.get_observation_space("player_0").shape
    action_dim = int(env.get_action_space("player_0").n)
    agent_ids = [f"player_{index}" for index in range(cfg.num_agents)]

    # Build one model and optimizer per agent for independent PPO updates.
    device = _resolve_device(cfg.num_gpus)
    print(f"backend=cleanrl | mode=multi-agent-selfish | device={device}")

    models = {
        agent_id: CleanRLCNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            conv_filters=cfg.conv_filters,
            fcnet_hiddens=cfg.fcnet_hiddens,
        ).to(device)
        for agent_id in agent_ids
    }
    optimizers = {
        agent_id: optim.Adam(models[agent_id].parameters(), lr=cfg.lr)
        for agent_id in agent_ids
    }

    observations, _ = env.reset(seed=cfg.seed)
    checkpoint_root = Path(cfg.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    run_name = cfg.run_name or f"cleanrl_multi_agent_selfish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(cfg.results_root).resolve() / "cleanrl" / run_name
    results_writer = IterationResultsWriter(results_dir)

    rollout_steps = int(cfg.train_batch_size)
    obs_buffer = {
        agent_id: torch.zeros((rollout_steps, *obs_shape), dtype=torch.float32, device=device)
        for agent_id in agent_ids
    }
    actions_buffer = {
        agent_id: torch.zeros(rollout_steps, dtype=torch.long, device=device)
        for agent_id in agent_ids
    }
    logprobs_buffer = {
        agent_id: torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        for agent_id in agent_ids
    }
    rewards_buffer = {
        agent_id: torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        for agent_id in agent_ids
    }
    dones_buffer = {
        agent_id: torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        for agent_id in agent_ids
    }
    values_buffer = {
        agent_id: torch.zeros(rollout_steps, dtype=torch.float32, device=device)
        for agent_id in agent_ids
    }

    zero_obs = _zero_observation(obs_shape)
    episode_returns = {agent_id: 0.0 for agent_id in agent_ids}
    training_history: list[dict[str, Any]] = []

    try:
        for iteration in range(1, cfg.stop_iters + 1):
            iteration_episode_returns: list[float] = []

            for step in range(rollout_steps):
                action_dict: dict[str, int] = {}

                # Sample one action per agent and populate rollout buffers.
                for agent_id in agent_ids:
                    obs_np = observations.get(agent_id, zero_obs)
                    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                    obs_buffer[agent_id][step] = obs_tensor

                    with torch.no_grad():
                        action, log_prob, _, value = models[agent_id].get_action_and_value(obs_tensor.unsqueeze(0))

                    action_dict[agent_id] = int(action.item())
                    actions_buffer[agent_id][step] = action
                    logprobs_buffer[agent_id][step] = log_prob
                    values_buffer[agent_id][step] = value.squeeze(0)

                next_obs, rewards, terminations, truncations, _ = env.step(action_dict)

                # Store per-agent rewards/dones and track episode returns.
                for agent_id in agent_ids:
                    reward = float(rewards.get(agent_id, 0.0))
                    done = bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                    rewards_buffer[agent_id][step] = reward
                    dones_buffer[agent_id][step] = float(done)

                    episode_returns[agent_id] += reward

                done_all = bool(terminations.get("__all__", False) or truncations.get("__all__", False))
                if done_all:
                    iteration_episode_returns.append(float(np.mean(list(episode_returns.values()))))
                    episode_returns = {agent_id: 0.0 for agent_id in agent_ids}
                    observations, _ = env.reset()
                else:
                    observations = next_obs

            # Compute GAE/returns for each agent.
            advantages = {}
            returns = {}
            for agent_id in agent_ids:
                with torch.no_grad():
                    next_obs_np = observations.get(agent_id, zero_obs)
                    next_obs_tensor = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                    _, next_value = models[agent_id].forward(next_obs_tensor.unsqueeze(0))

                adv, ret = _compute_gae(
                    rewards=rewards_buffer[agent_id],
                    dones=dones_buffer[agent_id],
                    values=values_buffer[agent_id],
                    next_value=next_value.squeeze(0),
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                )
                advantages[agent_id] = adv
                returns[agent_id] = ret

            # Optimize each agent policy independently.
            b_inds = np.arange(rollout_steps)
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []

            for agent_id in agent_ids:
                last_policy_loss = 0.0
                last_value_loss = 0.0
                last_entropy = 0.0

                for _ in range(cfg.num_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, rollout_steps, cfg.minibatch_size):
                        end = start + cfg.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, new_logprob, entropy, new_value = models[agent_id].get_action_and_value(
                            obs_buffer[agent_id][mb_inds],
                            actions_buffer[agent_id][mb_inds],
                        )

                        log_ratio = new_logprob - logprobs_buffer[agent_id][mb_inds]
                        ratio = torch.exp(log_ratio)

                        mb_advantages = advantages[agent_id][mb_inds]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std(unbiased=False) + 1e-8
                        )

                        policy_loss_1 = -mb_advantages * ratio
                        policy_loss_2 = -mb_advantages * torch.clamp(
                            ratio,
                            1.0 - cfg.clip_coef,
                            1.0 + cfg.clip_coef,
                        )
                        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                        value_loss = 0.5 * ((new_value - returns[agent_id][mb_inds]) ** 2).mean()
                        entropy_loss = entropy.mean()

                        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                        optimizers[agent_id].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(models[agent_id].parameters(), cfg.max_grad_norm)
                        optimizers[agent_id].step()

                        last_policy_loss = float(policy_loss.item())
                        last_value_loss = float(value_loss.item())
                        last_entropy = float(entropy_loss.item())

                policy_losses.append(last_policy_loss)
                value_losses.append(last_value_loss)
                entropies.append(last_entropy)

            # Aggregate and persist metrics.
            episode_reward_mean = (
                float(np.mean(iteration_episode_returns)) if iteration_episode_returns else None
            )
            metrics = {
                "iteration": iteration,
                "backend": "cleanrl",
                "mode": "multi-agent-selfish",
                "run_name": run_name,
                "episode_reward_mean": episode_reward_mean,
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
                "value_loss": float(np.mean(value_losses)) if value_losses else None,
                "entropy": float(np.mean(entropies)) if entropies else None,
            }
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

            if iteration % cfg.checkpoint_every == 0:
                checkpoint_path = checkpoint_root / f"cleanrl_multi_agent_iter_{iteration:06d}.pt"
                torch.save(
                    {
                        "model_state_dicts": {agent_id: models[agent_id].state_dict() for agent_id in agent_ids},
                        "optimizer_state_dicts": {
                            agent_id: optimizers[agent_id].state_dict() for agent_id in agent_ids
                        },
                        "config": asdict(cfg),
                        "iteration": iteration,
                    },
                    checkpoint_path,
                )
                print(f"checkpoint_saved={checkpoint_path}")

        final_checkpoint_path = checkpoint_root / f"cleanrl_multi_agent_final_iter_{cfg.stop_iters:06d}.pt"
        torch.save(
            {
                "model_state_dicts": {agent_id: models[agent_id].state_dict() for agent_id in agent_ids},
                "optimizer_state_dicts": {
                    agent_id: optimizers[agent_id].state_dict() for agent_id in agent_ids
                },
                "config": asdict(cfg),
                "iteration": cfg.stop_iters,
            },
            final_checkpoint_path,
        )

        return {
            "status": "completed",
            "backend": "cleanrl",
            "mode": "multi-agent-selfish",
            "device": str(device),
            "num_agents": cfg.num_agents,
            "run_name": run_name,
            "results_dir": str(results_dir),
            "iterations": cfg.stop_iters,
            "final_checkpoint": str(final_checkpoint_path),
            "history": training_history,
        }
    finally:
        results_writer.close()
        env.close()
