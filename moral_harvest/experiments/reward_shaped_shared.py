from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from shimmy import MeltingPotCompatibilityV0

from moral_harvest.experiments.multi_agent_selfish_cleanrl import (
    MultiAgentCleanRLCNN,
    _normalize_rgb,
    _reshape_flat_by_agent,
)
from moral_harvest.experiments.single_agent_ppo_cleanrl import _resolve_device
from moral_harvest.rewards.shaping import RewardShaper, RewardShapingConfig
from moral_harvest.training.config import SingleAgentTrainConfig
from moral_harvest.training.env_metrics import (
    count_active_berries_from_world_frame,
    extract_world_rgb_frames,
)
from moral_harvest.training.results_logger import IterationResultsWriter


SUPPORTED_REWARD_TYPES = ("selfish", "utilitarian", "deontological", "virtue")


# Return selected reward types from CLI/config selection.
def _resolve_reward_types(reward_type: str) -> list[str]:
    if reward_type == "all":
        return ["utilitarian", "deontological", "virtue"]
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type '{reward_type}'. Expected one of {SUPPORTED_REWARD_TYPES} or 'all'."
        )
    return [reward_type]


# Reshape vector-env infos into [num_envs][num_agents] dictionaries.
def _reshape_infos_by_agent(
    info_raw: Any,
    num_envs: int,
    num_agents: int,
) -> list[list[dict[str, Any]]]:
    reshaped = [[{} for _ in range(num_agents)] for _ in range(num_envs)]

    if isinstance(info_raw, (list, tuple)) and len(info_raw) == num_envs * num_agents:
        for flat_index, info in enumerate(info_raw):
            env_index = flat_index // num_agents
            agent_index = flat_index % num_agents
            reshaped[env_index][agent_index] = info if isinstance(info, dict) else {}
        return reshaped

    if isinstance(info_raw, dict):
        return reshaped

    return reshaped


# Run reward-shaped multi-agent CleanRL training for one reward type.
def _run_single_reward_type(
    cfg: SingleAgentTrainConfig,
    reward_type: str,
) -> dict[str, Any]:
    if cfg.include_ready_to_shoot:
        raise ValueError(
            "CleanRL backend currently supports RGB-only observations. Omit --include-ready-to-shoot."
        )

    if cfg.num_envs <= 0:
        raise ValueError("--num-envs must be a positive integer.")

    try:
        import supersuit as ss
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SuperSuit is required for multi-agent CleanRL vectorization. Install dependency 'supersuit'."
        ) from exc

    base_env = MeltingPotCompatibilityV0(substrate_name=cfg.substrate_name, render_mode="rgb_array")
    env = ss.pettingzoo_env_to_vec_env_v1(base_env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=cfg.num_envs,
        num_cpus=0,
        base_class="gymnasium",
    )

    agent_ids = list(getattr(base_env, "possible_agents", [f"player_{index}" for index in range(cfg.num_agents)]))
    num_agents = len(agent_ids)

    obs_shape = env.observation_space["RGB"].shape
    action_dim = int(env.action_space.n)

    device = _resolve_device(cfg.num_gpus)
    print(
        " | ".join(
            [
                "backend=cleanrl",
                "mode=multi-agent-reward-shaped",
                f"reward_type={reward_type}",
                f"device={device}",
                f"num_envs={cfg.num_envs}",
                f"alpha={cfg.reward_alpha}",
            ]
        )
    )

    model = MultiAgentCleanRLCNN(
        num_agents=num_agents,
        obs_shape=obs_shape,
        action_dim=action_dim,
        conv_filters=cfg.conv_filters,
        fcnet_hiddens=cfg.fcnet_hiddens,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    shapers = [
        RewardShaper(
            RewardShapingConfig(
                reward_type=reward_type,
                alpha=cfg.reward_alpha,
                deontological_max_bonus=cfg.deontological_max_bonus,
                virtue_scale=cfg.virtue_scale,
            )
        )
        for _ in range(cfg.num_envs)
    ]

    reset_obs, _ = env.reset(seed=cfg.seed)
    next_obs = torch.tensor(
        _reshape_flat_by_agent(_normalize_rgb(reset_obs["RGB"], cfg.normalize_rgb), cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )
    next_terminations = torch.zeros((cfg.num_envs, num_agents), dtype=torch.float32, device=device)
    next_truncations = torch.zeros((cfg.num_envs, num_agents), dtype=torch.float32, device=device)

    checkpoint_root = Path(cfg.checkpoint_root).resolve() / reward_type
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    base_run_name = cfg.run_name or f"cleanrl_multi_agent_reward_shaped_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = f"{base_run_name}_{reward_type}"
    results_dir = Path(cfg.results_root).resolve() / "cleanrl" / run_name
    results_writer = IterationResultsWriter(results_dir)

    rollout_steps = int(cfg.train_batch_size)
    obs_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents, *obs_shape),
        dtype=torch.float32,
        device=device,
    )
    actions_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents),
        dtype=torch.long,
        device=device,
    )
    logprobs_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )
    rewards_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )
    terminations_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )
    truncations_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )
    values_buffer = torch.zeros(
        (rollout_steps, cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )

    episode_returns = torch.zeros((cfg.num_envs, num_agents), dtype=torch.float32, device=device)
    training_history: list[dict[str, Any]] = []

    try:
        for iteration in range(1, cfg.stop_iters + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / float(cfg.stop_iters)
                optimizer.param_groups[0]["lr"] = frac * cfg.lr

            iteration_episode_returns: list[float] = []
            shaping_metric_values: dict[str, list[float]] = {
                "own_reward_mean": [],
                "shaped_reward_mean": [],
                "shaping_reward_mean": [],
                "utilitarian_mean_reward": [],
                "deontological_bonus_mean": [],
                "virtue_current_gini": [],
                "virtue_delta_gini": [],
            }
            apple_reward_total = 0.0
            shaping_reward_total = 0.0
            total_reward_total = 0.0
            berry_observation_steps = 0
            active_berries_total = 0.0
            berries_end_by_env = [0 for _ in range(cfg.num_envs)]

            for step in range(rollout_steps):
                obs_buffer[step] = next_obs
                terminations_buffer[step] = next_terminations
                truncations_buffer[step] = next_truncations

                with torch.no_grad():
                    action, logprob, _, value = model.get_actions_and_values(next_obs)

                actions_buffer[step] = action
                logprobs_buffer[step] = logprob
                values_buffer[step] = value

                flat_actions = action.reshape(-1).cpu().numpy()
                next_obs_raw, reward_raw, termination_raw, truncation_raw, info_raw = env.step(flat_actions)

                own_rewards_np = _reshape_flat_by_agent(
                    np.asarray(reward_raw, dtype=np.float32),
                    cfg.num_envs,
                    num_agents,
                )
                apple_reward_total += float(np.sum(own_rewards_np))
                infos_by_env = _reshape_infos_by_agent(info_raw, cfg.num_envs, num_agents)

                shaped_rewards_np = np.zeros_like(own_rewards_np, dtype=np.float32)
                for env_index in range(cfg.num_envs):
                    own_rewards_dict = {
                        agent_ids[agent_index]: float(own_rewards_np[env_index, agent_index])
                        for agent_index in range(num_agents)
                    }
                    env_infos = {
                        agent_ids[agent_index]: infos_by_env[env_index][agent_index]
                        for agent_index in range(num_agents)
                    }
                    shaped_rewards_dict, step_metrics = shapers[env_index].shape_step(
                        own_rewards=own_rewards_dict,
                        infos=env_infos,
                    )
                    for agent_index, agent_id in enumerate(agent_ids):
                        shaped_rewards_np[env_index, agent_index] = float(shaped_rewards_dict[agent_id])

                    for metric_name, metric_value in step_metrics.items():
                        if metric_name in shaping_metric_values and metric_value is not None:
                            shaping_metric_values[metric_name].append(float(metric_value))

                    shaping_reward_total += float(step_metrics.get("shaping_reward_sum", 0.0) or 0.0)
                    total_reward_total += float(step_metrics.get("shaped_reward_sum", 0.0) or 0.0)

                world_rgb_frames = extract_world_rgb_frames(
                    next_obs_raw=next_obs_raw,
                    num_envs=cfg.num_envs,
                    num_agents=num_agents,
                )
                if not world_rgb_frames and cfg.num_envs == 1:
                    rendered_frame = base_env.render()
                    if isinstance(rendered_frame, np.ndarray):
                        world_rgb_frames = [rendered_frame]
                for env_index, world_frame in enumerate(world_rgb_frames):
                    berry_count = count_active_berries_from_world_frame(world_frame, sprite_size=8)
                    berry_observation_steps += 1
                    active_berries_total += float(berry_count)
                    if env_index < cfg.num_envs:
                        berries_end_by_env[env_index] = berry_count

                next_rewards = torch.tensor(
                    shaped_rewards_np,
                    dtype=torch.float32,
                    device=device,
                )
                next_terminations = torch.tensor(
                    _reshape_flat_by_agent(
                        np.asarray(termination_raw, dtype=np.float32), cfg.num_envs, num_agents
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                next_truncations = torch.tensor(
                    _reshape_flat_by_agent(np.asarray(truncation_raw, dtype=np.float32), cfg.num_envs, num_agents),
                    dtype=torch.float32,
                    device=device,
                )
                next_obs = torch.tensor(
                    _reshape_flat_by_agent(_normalize_rgb(next_obs_raw["RGB"], cfg.normalize_rgb), cfg.num_envs, num_agents),
                    dtype=torch.float32,
                    device=device,
                )

                rewards_buffer[step] = next_rewards

                episode_returns = episode_returns + next_rewards
                done_env = torch.all((next_terminations > 0.0) | (next_truncations > 0.0), dim=1)
                done_env_indices = torch.where(done_env)[0]
                for env_index in done_env_indices.tolist():
                    iteration_episode_returns.append(float(episode_returns[env_index].mean().item()))
                    episode_returns[env_index] = 0.0
                    shapers[env_index].reset_episode()

            with torch.no_grad():
                next_values = model.get_values(next_obs)
                advantages = torch.zeros_like(rewards_buffer)
                dones_buffer = torch.maximum(terminations_buffer, truncations_buffer)
                next_done = torch.maximum(next_terminations, next_truncations)

                last_gae = torch.zeros((cfg.num_envs, num_agents), dtype=torch.float32, device=device)
                for step in reversed(range(rollout_steps)):
                    if step == rollout_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_value_step = next_values
                    else:
                        next_non_terminal = 1.0 - dones_buffer[step + 1]
                        next_value_step = values_buffer[step + 1]

                    delta = rewards_buffer[step] + cfg.gamma * next_value_step * next_non_terminal - values_buffer[step]
                    last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
                    advantages[step] = last_gae

                returns = advantages + values_buffer

            batch_size = rollout_steps * cfg.num_envs
            b_obs = obs_buffer.reshape(batch_size, num_agents, *obs_shape)
            b_actions = actions_buffer.reshape(batch_size, num_agents)
            b_logprobs = logprobs_buffer.reshape(batch_size, num_agents)
            b_advantages = advantages.reshape(batch_size, num_agents)
            b_returns = returns.reshape(batch_size, num_agents)
            b_values = values_buffer.reshape(batch_size, num_agents)

            b_inds = np.arange(batch_size)
            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0
            last_approx_kl = 0.0
            kl_stopped = False

            for _ in range(cfg.num_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, new_logprob, entropy, new_value = model.get_actions_and_values(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                    )

                    log_ratio = new_logprob - b_logprobs[mb_inds]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl_per_agent = ((ratio - 1.0) - log_ratio).mean(dim=0)

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean(dim=0, keepdim=True)) / (
                        mb_advantages.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                    )

                    policy_loss_1 = -mb_advantages * ratio
                    policy_loss_2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - cfg.clip_coef,
                        1.0 + cfg.clip_coef,
                    )
                    policy_loss_per_agent = torch.max(policy_loss_1, policy_loss_2).mean(dim=0)

                    if cfg.clip_vloss:
                        value_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                        value_pred_clipped = b_values[mb_inds] + torch.clamp(
                            new_value - b_values[mb_inds],
                            -cfg.clip_coef,
                            cfg.clip_coef,
                        )
                        value_loss_clipped = (value_pred_clipped - b_returns[mb_inds]) ** 2
                        value_loss_per_agent = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean(dim=0)
                    else:
                        value_loss_per_agent = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean(dim=0)
                    entropy_per_agent = entropy.mean(dim=0)

                    total_loss = (
                        policy_loss_per_agent
                        + cfg.vf_coef * value_loss_per_agent
                        - cfg.ent_coef * entropy_per_agent
                    ).mean()

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    last_policy_loss = float(policy_loss_per_agent.mean().item())
                    last_value_loss = float(value_loss_per_agent.mean().item())
                    last_entropy = float(entropy_per_agent.mean().item())
                    last_approx_kl = float(approx_kl_per_agent.mean().item())

                    if cfg.target_kl is not None and last_approx_kl > cfg.target_kl:
                        kl_stopped = True
                        break

                if kl_stopped:
                    break

            episode_reward_mean = (
                float(np.mean(iteration_episode_returns)) if iteration_episode_returns else None
            )

            shaping_metrics_mean = {
                metric_name: (float(np.mean(values)) if values else None)
                for metric_name, values in shaping_metric_values.items()
            }

            rollout_env_steps = float(rollout_steps * cfg.num_envs)
            apple_reward_mean_per_env_step = apple_reward_total / rollout_env_steps
            shaping_reward_mean_per_env_step = shaping_reward_total / rollout_env_steps
            total_reward_mean_per_env_step = total_reward_total / rollout_env_steps

            total_berries_end = int(sum(berries_end_by_env)) if berry_observation_steps > 0 else None
            total_berries_end_mean_env = (
                float(total_berries_end / max(cfg.num_envs, 1))
                if total_berries_end is not None
                else None
            )
            avg_active_berries_per_env_step = (
                float(active_berries_total / berry_observation_steps)
                if berry_observation_steps > 0
                else None
            )
            apples_eaten_per_env_step = (
                float(apple_reward_total / berry_observation_steps)
                if berry_observation_steps > 0
                else None
            )
            berry_lifetime_steps_estimate = (
                float(avg_active_berries_per_env_step / apples_eaten_per_env_step)
                if (
                    avg_active_berries_per_env_step is not None
                    and apples_eaten_per_env_step is not None
                    and apples_eaten_per_env_step > 0.0
                )
                else None
            )

            metrics = {
                "iteration": iteration,
                "backend": "cleanrl",
                "mode": "multi-agent-reward-shaped",
                "reward_type": reward_type,
                "run_name": run_name,
                "episode_reward_mean": episode_reward_mean,
                "policy_loss": last_policy_loss,
                "value_loss": last_value_loss,
                "entropy": last_entropy,
                "approx_kl": last_approx_kl,
                "kl_stopped": kl_stopped,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "normalize_rgb": cfg.normalize_rgb,
                "apple_reward_total": apple_reward_total,
                "shaping_reward_total": shaping_reward_total,
                "total_reward_total": total_reward_total,
                "apple_reward_mean_per_env_step": apple_reward_mean_per_env_step,
                "shaping_reward_mean_per_env_step": shaping_reward_mean_per_env_step,
                "total_reward_mean_per_env_step": total_reward_mean_per_env_step,
                "total_berries_end": total_berries_end,
                "total_berries_end_mean_env": total_berries_end_mean_env,
                "avg_active_berries_per_env_step": avg_active_berries_per_env_step,
                "berry_lifetime_steps_estimate": berry_lifetime_steps_estimate,
                "berry_observation_steps": berry_observation_steps,
            }
            metrics.update(shaping_metrics_mean)
            training_history.append(metrics)
            results_writer.write(metrics)

            print(
                " | ".join(
                    [
                        f"iter={iteration}",
                        f"reward_type={reward_type}",
                        f"reward={metrics['episode_reward_mean']}",
                        f"policy_loss={metrics['policy_loss']}",
                        f"value_loss={metrics['value_loss']}",
                        f"entropy={metrics['entropy']}",
                    ]
                )
            )

            if iteration % cfg.checkpoint_every == 0:
                checkpoint_path = checkpoint_root / f"cleanrl_multi_agent_{reward_type}_iter_{iteration:06d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "iteration": iteration,
                        "reward_type": reward_type,
                    },
                    checkpoint_path,
                )
                print(f"checkpoint_saved={checkpoint_path}")

        final_checkpoint_path = checkpoint_root / f"cleanrl_multi_agent_{reward_type}_final_iter_{cfg.stop_iters:06d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "iteration": cfg.stop_iters,
                "reward_type": reward_type,
            },
            final_checkpoint_path,
        )

        return {
            "status": "completed",
            "backend": "cleanrl",
            "mode": "multi-agent-reward-shaped",
            "reward_type": reward_type,
            "device": str(device),
            "num_agents": num_agents,
            "num_envs": cfg.num_envs,
            "run_name": run_name,
            "results_dir": str(results_dir),
            "iterations": cfg.stop_iters,
            "final_checkpoint": str(final_checkpoint_path),
            "history": training_history,
        }
    finally:
        results_writer.close()
        env.close()


# Run reward-shaped training for one reward type or all configured types.
def run_reward_shaped_shared_cleanrl(cfg: SingleAgentTrainConfig) -> dict[str, Any]:
    reward_types = _resolve_reward_types(cfg.reward_type)
    outputs = [_run_single_reward_type(cfg, reward_type=reward_type) for reward_type in reward_types]

    if len(outputs) == 1:
        return outputs[0]

    return {
        "status": "completed",
        "backend": "cleanrl",
        "mode": "multi-agent-reward-shaped",
        "reward_types": reward_types,
        "runs": outputs,
    }