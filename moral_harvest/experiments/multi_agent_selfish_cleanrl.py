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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from moral_harvest.experiments.single_agent_ppo_cleanrl import _resolve_device
from moral_harvest.training.cnn_actor_critic import CleanRLCNNActorCritic
from moral_harvest.training.config import SingleAgentTrainConfig
from moral_harvest.training.env_metrics import (
    count_active_berries_from_world_frame,
    extract_world_rgb_frames,
)
from moral_harvest.training.results_logger import IterationResultsWriter


def _fmt_metric_4dp(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _round_metric_4dp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating, int, np.integer)):
        return round(float(value), 4)
    return None


# Stack one independent policy/value submodule per agent under one nn.Module.
class MultiAgentCleanRLCNN(nn.Module):
    # Build per-agent actor-critic modules with shared architecture.
    def __init__(
        self,
        num_agents: int,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        conv_filters: list[list[int | list[int]]],
        fcnet_hiddens: list[int],
    ):
        super().__init__()
        self.num_agents = num_agents
        self.agents = nn.ModuleList(
            [
                CleanRLCNNActorCritic(
                    obs_shape=obs_shape,
                    action_dim=action_dim,
                    conv_filters=conv_filters,
                    fcnet_hiddens=fcnet_hiddens,
                )
                for _ in range(num_agents)
            ]
        )

    # Predict values for all agents for a batch of observations.
    def get_values(self, obs: torch.Tensor) -> torch.Tensor:
        values = []
        for agent_index in range(self.num_agents):
            _, value = self.agents[agent_index].forward(obs[:, agent_index])
            values.append(value)
        return torch.stack(values, dim=1)

    # Sample/evaluate actions and values for all agents.
    def get_actions_and_values(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_actions = []
        out_logprobs = []
        out_entropies = []
        out_values = []

        for agent_index in range(self.num_agents):
            agent_obs = obs[:, agent_index]
            agent_actions = actions[:, agent_index] if actions is not None else None
            sampled_action, logprob, entropy, value = self.agents[agent_index].get_action_and_value(
                agent_obs,
                agent_actions,
            )
            out_actions.append(sampled_action)
            out_logprobs.append(logprob)
            out_entropies.append(entropy)
            out_values.append(value)

        return (
            torch.stack(out_actions, dim=1),
            torch.stack(out_logprobs, dim=1),
            torch.stack(out_entropies, dim=1),
            torch.stack(out_values, dim=1),
        )


# Ensure observation tensors are float32 and optionally normalized to [0, 1].
def _normalize_rgb(observations: np.ndarray, enabled: bool) -> np.ndarray:
    observations_f32 = observations.astype(np.float32)
    if enabled:
        return observations_f32 / 255.0
    return observations_f32


# Reshape flattened [num_envs * num_agents, ...] arrays to [num_envs, num_agents, ...].
def _reshape_flat_by_agent(
    values: np.ndarray,
    num_envs: int,
    num_agents: int,
) -> np.ndarray:
    return values.reshape(num_envs, num_agents, *values.shape[1:])


# Train vanilla selfish IPPO with one optimizer and summed per-agent losses.
def run_multi_agent_selfish_cleanrl(cfg: SingleAgentTrainConfig) -> dict[str, Any]:
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

    # Build vectorized PettingZoo-compatible environment through SuperSuit wrappers.
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

    # Infer spaces from vectorized env.
    obs_shape = env.observation_space["RGB"].shape
    action_dim = int(env.action_space.n)

    device = _resolve_device(cfg.num_gpus)
    print(
        f"backend=cleanrl | mode=multi-agent-selfish | device={device} | num_envs={cfg.num_envs} | anneal_lr={cfg.anneal_lr} | normalize_rgb={cfg.normalize_rgb} | clip_vloss={cfg.clip_vloss} | target_kl={cfg.target_kl}"
    )

    model = MultiAgentCleanRLCNN(
        num_agents=num_agents,
        obs_shape=obs_shape,
        action_dim=action_dim,
        conv_filters=cfg.conv_filters,
        fcnet_hiddens=cfg.fcnet_hiddens,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    reset_obs, _ = env.reset(seed=cfg.seed)
    next_obs = torch.tensor(
        _reshape_flat_by_agent(_normalize_rgb(reset_obs["RGB"], cfg.normalize_rgb), cfg.num_envs, num_agents),
        dtype=torch.float32,
        device=device,
    )
    next_terminations = torch.zeros((cfg.num_envs, num_agents), dtype=torch.float32, device=device)
    next_truncations = torch.zeros((cfg.num_envs, num_agents), dtype=torch.float32, device=device)

    checkpoint_root = Path(cfg.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    run_name = cfg.run_name or f"cleanrl_multi_agent_selfish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        iteration_range = range(1, cfg.stop_iters + 1)
        progress = (
            tqdm(iteration_range, total=cfg.stop_iters, desc="multi-agent-selfish", dynamic_ncols=True)
            if tqdm is not None
            else iteration_range
        )
        for iteration in progress:
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / float(cfg.stop_iters)
                optimizer.param_groups[0]["lr"] = frac * cfg.lr

            iteration_episode_returns: list[float] = []
            apple_reward_total = 0.0
            shaping_reward_total = 0.0
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
                next_obs_raw, reward_raw, termination_raw, truncation_raw, _ = env.step(flat_actions)

                next_rewards = torch.tensor(
                    _reshape_flat_by_agent(np.asarray(reward_raw, dtype=np.float32), cfg.num_envs, num_agents),
                    dtype=torch.float32,
                    device=device,
                )
                apple_reward_total += float(next_rewards.sum().item())
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

                world_rgb_frames = extract_world_rgb_frames(
                    next_obs_raw=next_obs_raw,
                    num_envs=cfg.num_envs,
                    num_agents=num_agents,
                )
                if not world_rgb_frames and cfg.num_envs == 1:
                    try:
                        rendered_frame = base_env.render()
                        if isinstance(rendered_frame, np.ndarray):
                            world_rgb_frames = [rendered_frame]
                    except Exception:
                        world_rgb_frames = []
                for env_index, world_frame in enumerate(world_rgb_frames):
                    berry_count = count_active_berries_from_world_frame(world_frame, sprite_size=8)
                    berry_observation_steps += 1
                    active_berries_total += float(berry_count)
                    if env_index < cfg.num_envs:
                        berries_end_by_env[env_index] = berry_count

                rewards_buffer[step] = next_rewards

                episode_returns = episode_returns + next_rewards
                done_env = torch.all((next_terminations > 0.0) | (next_truncations > 0.0), dim=1)
                done_env_indices = torch.where(done_env)[0]
                for env_index in done_env_indices.tolist():
                    iteration_episode_returns.append(float(episode_returns[env_index].mean().item()))
                    episode_returns[env_index] = 0.0

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
            total_reward_total = apple_reward_total
            rollout_env_steps = float(rollout_steps * cfg.num_envs)
            apple_reward_mean_per_env_step = apple_reward_total / rollout_env_steps
            shaping_reward_mean_per_env_step = 0.0
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
                "mode": "multi-agent-selfish",
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
            training_history.append(metrics)
            results_writer.write(metrics)

            if tqdm is not None:
                progress.set_postfix(
                    {
                        "reward": _round_metric_4dp(metrics["episode_reward_mean"]),
                        "policy": _round_metric_4dp(metrics["policy_loss"]),
                        "value": _round_metric_4dp(metrics["value_loss"]),
                        "lr": _round_metric_4dp(metrics["learning_rate"]),
                    }
                )

            iteration_summary = " | ".join(
                [
                    f"iter={iteration}",
                    f"reward={_fmt_metric_4dp(metrics['episode_reward_mean'])}",
                    f"policy_loss={_fmt_metric_4dp(metrics['policy_loss'])}",
                    f"value_loss={_fmt_metric_4dp(metrics['value_loss'])}",
                    f"entropy={_fmt_metric_4dp(metrics['entropy'])}",
                    f"lr={_fmt_metric_4dp(metrics['learning_rate'])}",
                    f"apple_reward_total={_fmt_metric_4dp(metrics['apple_reward_total'])}",
                    f"shaping_reward_total={_fmt_metric_4dp(metrics['shaping_reward_total'])}",
                    f"total_reward_total={_fmt_metric_4dp(metrics['total_reward_total'])}",
                    f"berries_end={metrics['total_berries_end']}",
                    f"berry_lifetime={_fmt_metric_4dp(metrics['berry_lifetime_steps_estimate'])}",
                ]
            )
            if tqdm is not None:
                tqdm.write(iteration_summary)
            else:
                print(iteration_summary)

            if iteration % cfg.checkpoint_every == 0:
                checkpoint_path = checkpoint_root / f"cleanrl_multi_agent_iter_{iteration:06d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "iteration": iteration,
                    },
                    checkpoint_path,
                )
                checkpoint_summary = f"checkpoint_saved={checkpoint_path}"
                if tqdm is not None:
                    tqdm.write(checkpoint_summary)
                else:
                    print(checkpoint_summary)

        final_checkpoint_path = checkpoint_root / f"cleanrl_multi_agent_final_iter_{cfg.stop_iters:06d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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
            "num_agents": num_agents,
            "num_envs": cfg.num_envs,
            "anneal_lr": cfg.anneal_lr,
            "normalize_rgb": cfg.normalize_rgb,
            "run_name": run_name,
            "results_dir": str(results_dir),
            "iterations": cfg.stop_iters,
            "final_checkpoint": str(final_checkpoint_path),
            "history": training_history,
        }
    finally:
        results_writer.close()
        env.close()
