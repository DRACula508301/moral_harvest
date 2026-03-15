from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from moral_harvest.envs.meltingpot_env import HarvestSingleAgentEnv
from moral_harvest.training.config import SingleAgentTrainConfig
from moral_harvest.training.results_logger import IterationResultsWriter


# Build a Torch CNN+MLP actor-critic for RGB observations.
class CleanRLCNNActorCritic(nn.Module):
    # Initialize convolutional encoder and actor/critic heads.
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        conv_filters: list[list[int | list[int]]],
        fcnet_hiddens: list[int],
    ):
        super().__init__()

        # Build convolutional stack from config filters.
        conv_layers: list[nn.Module] = []
        in_channels = obs_shape[2]
        for out_channels, kernel, stride in conv_filters:
            kernel_size = tuple(kernel)
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    kernel_size=kernel_size,
                    stride=int(stride),
                )
            )
            conv_layers.append(nn.ReLU())
            in_channels = int(out_channels)
        self.conv = nn.Sequential(*conv_layers)

        # Infer flattened conv feature size using a dummy input.
        with torch.no_grad():
            dummy = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1], dtype=torch.float32)
            conv_out_size = int(np.prod(self.conv(dummy).shape[1:]))

        # Build shared MLP trunk.
        mlp_layers: list[nn.Module] = []
        current_size = conv_out_size
        for hidden_size in fcnet_hiddens:
            mlp_layers.append(nn.Linear(current_size, int(hidden_size)))
            mlp_layers.append(nn.ReLU())
            current_size = int(hidden_size)
        self.trunk = nn.Sequential(*mlp_layers)

        # Build separate policy and value heads.
        self.actor = nn.Linear(current_size, action_dim)
        self.critic = nn.Linear(current_size, 1)

    # Encode observations and produce action logits plus state value.
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert NHWC env observations to NCHW for convolution.
        x = obs.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.trunk(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    # Sample action and return PPO-relevant statistics.
    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# Compute generalized advantage estimates for one rollout batch.
def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Run reversed-time GAE accumulation.
    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, dtype=torch.float32)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    # Returns are value targets for the critic.
    returns = advantages + values
    return advantages, returns


# Resolve torch device for CleanRL training.
def _resolve_device(num_gpus_override: int | None) -> torch.device:
    # Auto mode uses CUDA if available.
    if num_gpus_override is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explicit non-positive override forces CPU.
    if num_gpus_override <= 0:
        return torch.device("cpu")

    # Positive override uses CUDA when available; otherwise fallback to CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train a single-agent PPO policy using a CleanRL-style loop.
def run_single_agent_cleanrl(cfg: SingleAgentTrainConfig) -> dict[str, Any]:
    # Create env and infer observation/action dimensions.
    env = HarvestSingleAgentEnv(
        {
            "substrate_name": cfg.substrate_name,
            "focal_agent": cfg.focal_agent,
            "no_op_action": cfg.no_op_action,
            "include_ready_to_shoot": cfg.include_ready_to_shoot,
        }
    )

    if cfg.include_ready_to_shoot:
        raise ValueError("CleanRL backend currently supports RGB-only observations. Omit --include-ready-to-shoot.")

    obs_shape = env.observation_space.shape
    action_dim = int(env.action_space.n)

    # Build model and optimizer.
    device = _resolve_device(cfg.num_gpus)
    print(f"backend=cleanrl | device={device}")
    model = CleanRLCNNActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        conv_filters=cfg.conv_filters,
        fcnet_hiddens=cfg.fcnet_hiddens,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Initialize env state and checkpoint folder.
    obs, _ = env.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    checkpoint_root = Path(cfg.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Build deterministic results artifact directory for per-iteration metrics.
    run_name = cfg.run_name or f"{cfg.backend}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(cfg.results_root).resolve() / cfg.backend / run_name
    results_writer = IterationResultsWriter(results_dir)

    # Allocate rollout buffers.
    rollout_steps = int(cfg.train_batch_size)
    obs_buffer = torch.zeros((rollout_steps, *obs_shape), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros(rollout_steps, dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    dones_buffer = torch.zeros(rollout_steps, dtype=torch.float32, device=device)
    values_buffer = torch.zeros(rollout_steps, dtype=torch.float32, device=device)

    training_history: list[dict[str, Any]] = []
    episode_return = 0.0

    try:
        # Main update loop: collect rollout -> optimize PPO -> checkpoint.
        for iteration in range(1, cfg.stop_iters + 1):
            iteration_episode_returns: list[float] = []

            # Collect one rollout batch.
            for step in range(rollout_steps):
                obs_buffer[step] = obs_t

                with torch.no_grad():
                    action, log_prob, _, value = model.get_action_and_value(obs_t.unsqueeze(0))

                action_i = int(action.item())
                next_obs, reward, terminated, truncated, _ = env.step(action_i)
                done = bool(terminated or truncated)

                actions_buffer[step] = action
                logprobs_buffer[step] = log_prob
                rewards_buffer[step] = float(reward)
                dones_buffer[step] = float(done)
                values_buffer[step] = value.squeeze(0)

                episode_return += float(reward)
                if done:
                    iteration_episode_returns.append(episode_return)
                    episode_return = 0.0
                    next_obs, _ = env.reset()

                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # Compute bootstrap value and GAE/returns.
            with torch.no_grad():
                _, next_value = model.forward(obs_t.unsqueeze(0))
                next_value = next_value.squeeze(0)

            advantages, returns = _compute_gae(
                rewards=rewards_buffer,
                dones=dones_buffer,
                values=values_buffer,
                next_value=next_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            # Run PPO optimization across multiple epochs/minibatches.
            b_inds = np.arange(rollout_steps)
            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0

            for _ in range(cfg.num_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, rollout_steps, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, new_logprob, entropy, new_value = model.get_action_and_value(
                        obs_buffer[mb_inds], actions_buffer[mb_inds]
                    )

                    log_ratio = new_logprob - logprobs_buffer[mb_inds]
                    ratio = torch.exp(log_ratio)

                    mb_advantages = advantages[mb_inds]
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

                    value_loss = 0.5 * ((new_value - returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    last_policy_loss = float(policy_loss.item())
                    last_value_loss = float(value_loss.item())
                    last_entropy = float(entropy_loss.item())

            # Aggregate and print training diagnostics.
            episode_reward_mean = (
                float(np.mean(iteration_episode_returns)) if iteration_episode_returns else None
            )
            metrics = {
                "iteration": iteration,
                "backend": cfg.backend,
                "run_name": run_name,
                "episode_reward_mean": episode_reward_mean,
                "policy_loss": last_policy_loss,
                "value_loss": last_value_loss,
                "entropy": last_entropy,
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

            # Save periodic checkpoints for later rollout/evaluation.
            if iteration % cfg.checkpoint_every == 0:
                checkpoint_path = checkpoint_root / f"cleanrl_iter_{iteration:06d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "iteration": iteration,
                    },
                    checkpoint_path,
                )
                print(f"checkpoint_saved={checkpoint_path}")

        # Save one final checkpoint when training is complete.
        final_checkpoint_path = checkpoint_root / f"cleanrl_final_iter_{cfg.stop_iters:06d}.pt"
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
            "device": str(device),
            "run_name": run_name,
            "results_dir": str(results_dir),
            "iterations": cfg.stop_iters,
            "final_checkpoint": str(final_checkpoint_path),
            "history": training_history,
        }
    finally:
        # Ensure results files are closed before shutdown.
        results_writer.close()
        # Release environment resources.
        env.close()
