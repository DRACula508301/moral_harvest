from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from shimmy import MeltingPotCompatibilityV0
from torch.distributions.categorical import Categorical

from moral_harvest.experiments.multi_agent_selfish_cleanrl import MultiAgentCleanRLCNN


# Parse CLI args for one-episode multi-agent checkpoint rollout.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out one episode with a multi-agent CleanRL checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help=(
            "Checkpoint file path or '<reward_type>/<checkpoint_name>.pt'. "
            "Relative values are resolved under repo/checkpoints/."
        ),
    )
    parser.add_argument("--substrate", default="commons_harvest__open")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--explore", action="store_true", help="Sample actions stochastically instead of argmax")
    parser.add_argument("--max-steps", type=int, default=5000, help="Safety cap for rollout length")
    parser.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        help="Record rollout video (enabled by default)",
    )
    parser.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Disable rollout video recording",
    )
    parser.set_defaults(record_video=True)
    parser.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Stop after the first completed episode instead of auto-resetting until max steps",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name for videos/<run_name>")
    return parser.parse_args()


# Resolve inference device from checkpoint and hardware.
def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Resolve checkpoint path, defaulting to repo_root/checkpoints for relative refs.
def _resolve_checkpoint_path(checkpoint_ref: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoints_root = repo_root / "checkpoints"

    normalized_ref = checkpoint_ref.replace("\\", "/")
    raw_path = Path(normalized_ref)

    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    if raw_path.exists():
        return raw_path.resolve()

    parts = list(raw_path.parts)
    if parts and parts[0] in {".", ".."}:
        raw_path = Path(*parts[1:]) if len(parts) > 1 else Path()

    if raw_path.parts and raw_path.parts[0] == "checkpoints":
        raw_path = Path(*raw_path.parts[1:])

    candidate = checkpoints_root / raw_path
    return candidate


# Convert raw agent observations to RGB tensor input expected by the model.
def _build_obs_batch(
    observations: dict[str, dict[str, Any]],
    ordered_agent_ids: list[str],
    rgb_shape: tuple[int, int, int],
    normalize_rgb: bool,
) -> np.ndarray:
    batch = np.zeros((1, len(ordered_agent_ids), *rgb_shape), dtype=np.float32)
    for agent_index, agent_id in enumerate(ordered_agent_ids):
        agent_obs = observations.get(agent_id)
        if agent_obs is None:
            continue
        rgb = np.asarray(agent_obs["RGB"], dtype=np.float32)
        if normalize_rgb:
            rgb = rgb / 255.0
        batch[0, agent_index] = rgb
    return batch


# Compute one action per model-agent index.
def _compute_actions(
    model: MultiAgentCleanRLCNN,
    obs_batch: torch.Tensor,
    explore: bool,
) -> np.ndarray:
    actions: list[int] = []
    with torch.no_grad():
        for agent_index in range(model.num_agents):
            logits, _ = model.agents[agent_index].forward(obs_batch[:, agent_index])
            dist = Categorical(logits=logits)
            if explore:
                action = int(dist.sample().item())
            else:
                action = int(torch.argmax(logits, dim=-1).item())
            actions.append(action)
    return np.asarray(actions, dtype=np.int64)


# Persist rollout frames to videos/<run_name>/ as MP4.
def _save_rollout_video(frames: list[np.ndarray], run_name: str, fps: int = 15) -> Path | None:
    if not frames:
        return None

    output_dir = Path("videos") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rollout_multi_agent.mp4"

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(output_path), codec="libx264", audio=False, logger=None)
    clip.close()
    return output_path


# Roll out exactly one episode and print summary metrics.
def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    device = _resolve_device()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    cfg = checkpoint.get("config", {})

    normalize_rgb = bool(cfg.get("normalize_rgb", True))
    conv_filters = cfg.get("conv_filters", [[32, [8, 8], 4], [32, [4, 4], 2], [64, [3, 3], 1]])
    fcnet_hiddens = cfg.get("fcnet_hiddens", [512])

    env = MeltingPotCompatibilityV0(
        substrate_name=args.substrate,
        render_mode="rgb_array" if args.record_video else None,
    )
    observations, _ = env.reset(seed=args.seed)

    ordered_agent_ids = list(getattr(env, "possible_agents", sorted(observations.keys())))
    if not ordered_agent_ids:
        raise RuntimeError("Environment returned no possible agents.")

    first_agent = ordered_agent_ids[0]
    rgb_shape = tuple(env.observation_space(first_agent)["RGB"].shape)
    action_dim = int(env.action_space(first_agent).n)

    model = MultiAgentCleanRLCNN(
        num_agents=len(ordered_agent_ids),
        obs_shape=rgb_shape,
        action_dim=action_dim,
        conv_filters=conv_filters,
        fcnet_hiddens=fcnet_hiddens,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cumulative_rewards = {agent_id: 0.0 for agent_id in ordered_agent_ids}
    steps = 0
    video_frames: list[np.ndarray] = []

    try:
        while True:
            if args.max_steps > 0 and steps >= args.max_steps:
                break
            if not observations:
                break

            obs_batch_np = _build_obs_batch(
                observations=observations,
                ordered_agent_ids=ordered_agent_ids,
                rgb_shape=rgb_shape,
                normalize_rgb=normalize_rgb,
            )
            obs_batch_t = torch.tensor(obs_batch_np, dtype=torch.float32, device=device)
            actions_by_index = _compute_actions(model=model, obs_batch=obs_batch_t, explore=args.explore)

            action_dict = {
                agent_id: int(actions_by_index[agent_index])
                for agent_index, agent_id in enumerate(ordered_agent_ids)
                if agent_id in observations
            }

            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            steps += 1

            if args.record_video:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    video_frames.append(frame)

            for agent_id, reward in rewards.items():
                if agent_id in cumulative_rewards:
                    cumulative_rewards[agent_id] += float(reward)

            all_agents = [agent for agent in ordered_agent_ids if agent in terminations or agent in truncations]
            if all_agents:
                all_done = all(bool(terminations.get(agent, False) or truncations.get(agent, False)) for agent in all_agents)
                if all_done:
                    break

        total_reward = float(sum(cumulative_rewards.values()))
        mean_reward = total_reward / max(len(cumulative_rewards), 1)

        summary = {
            "checkpoint_path": str(checkpoint_path),
            "substrate": args.substrate,
            "steps": steps,
            "total_reward": total_reward,
            "mean_reward_per_agent": mean_reward,
            "reward_per_agent": cumulative_rewards,
        }

        if args.record_video:
            run_name = args.run_name or f"multi-agent-{checkpoint_path.stem}"
            video_path = _save_rollout_video(video_frames, run_name=run_name)
            summary["video_path"] = str(video_path) if video_path is not None else None

        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
