from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from moral_harvest.experiments.multi_agent_selfish_cleanrl import run_multi_agent_selfish_cleanrl
from moral_harvest.experiments.reward_shaped_shared import run_reward_shaped_shared_cleanrl
from moral_harvest.experiments.single_agent_ppo_cleanrl import run_single_agent_cleanrl
from moral_harvest.training.config import SingleAgentTrainConfig


# Parse CLI arguments for single-agent PPO training.
def parse_args() -> argparse.Namespace:
    # Build the top-level parser and expose only the current training mode.
    parser = argparse.ArgumentParser(description="Moral Harvest training CLI")

    # Environment and stopping controls.
    parser.add_argument(
        "--mode",
        choices=["single-agent", "multi-agent-selfish", "multi-agent-reward-shaped"],
        default="single-agent",
    )
    parser.add_argument("--backend", choices=["rllib", "cleanrl"], default="cleanrl")
    parser.add_argument("--substrate", default="commons_harvest__open")
    parser.add_argument("--focal-agent", default="player_0")
    parser.add_argument("--num-agents", type=int, default=7)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--stop-iters", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--reward-type",
        choices=["selfish", "utilitarian", "deontological", "virtue", "all"],
        default="utilitarian",
    )
    parser.add_argument("--reward-alpha", type=float, default=0.5)
    parser.add_argument(
        "--shaping-begin",
        type=int,
        default=None,
        help="Global env-step index where shaping schedule begins (alpha forced to 1.0 before this step).",
    )
    parser.add_argument(
        "--rew-shaping-horizon",
        type=int,
        default=None,
        help="Number of global env steps to linearly ramp shaping weight from 0 to 1.",
    )
    parser.add_argument("--deontological-max-bonus", type=float, default=1.0)
    parser.add_argument("--virtue-scale", type=float, default=1.0)

    # PPO optimization hyperparameters.
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-sgd-iter", type=int, default=10, help="Deprecated alias for --num-epochs")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", dest="clip_vloss", action="store_true")
    parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    parser.set_defaults(clip_vloss=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anneal-lr", dest="anneal_lr", action="store_true")
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.set_defaults(anneal_lr=True)
    parser.add_argument("--normalize-rgb", dest="normalize_rgb", action="store_true")
    parser.add_argument("--no-normalize-rgb", dest="normalize_rgb", action="store_false")
    parser.set_defaults(normalize_rgb=True)
    parser.add_argument("--num-gpus", type=int, default=None, help="Optional override; default auto-detects GPU")
    parser.add_argument("--seed", type=int, default=None)

    # Observation/action wrapper controls.
    parser.add_argument("--include-ready-to-shoot", action="store_true")
    parser.add_argument("--no-op-action", type=int, default=0)

    # Return parsed CLI values for downstream config construction.
    return parser.parse_args()


# Entry point for running configured training modes.
def _auto_plot_training_curves(output: dict[str, Any]) -> dict[str, Any]:
    # Use training output directory to find persisted metrics.
    results_dir_raw = output.get("results_dir")
    if not isinstance(results_dir_raw, str) or not results_dir_raw:
        output["plot_path"] = None
        return output

    results_dir = Path(results_dir_raw)
    metrics_jsonl = results_dir / "metrics.jsonl"
    metrics_csv = results_dir / "metrics.csv"
    metrics_path = metrics_jsonl if metrics_jsonl.exists() else metrics_csv

    if not metrics_path.exists():
        output["plot_path"] = None
        output["plot_error"] = f"Metrics file not found in {results_dir}"
        return output

    # Lazily import plotting utilities to keep training resilient.
    try:
        from moral_harvest.analysis.plot_training_curves import load_metrics, plot_curves
    except Exception as exc:  # pragma: no cover
        output["plot_path"] = None
        output["plot_error"] = f"Plot utilities unavailable: {exc}"
        return output

    try:
        rows = load_metrics(metrics_path)
        if not rows:
            output["plot_path"] = None
            output["plot_error"] = "Metrics file contains no rows."
            return output

        metric_priority = ["episode_reward_mean", "policy_loss", "value_loss", "entropy"]
        y_keys = [metric for metric in metric_priority if metric in rows[0]]
        if not y_keys:
            output["plot_path"] = None
            output["plot_error"] = "No plottable metric keys found in metrics rows."
            return output

        plot_path = results_dir / "training_curves.png"
        title_mode = output.get("mode", "training")
        title_backend = output.get("backend", "backend")
        title_run = output.get("run_name", "run")
        plot_curves(
            rows=rows,
            x_key="iteration",
            y_keys=y_keys,
            output_path=plot_path,
            title=f"{title_mode} | {title_backend} | {title_run}",
        )
        output["plot_path"] = str(plot_path)
        print(f"plot_saved={plot_path}")
        return output
    except Exception as exc:  # pragma: no cover
        output["plot_path"] = None
        output["plot_error"] = f"Failed to generate plot: {exc}"
        print(f"plot_skipped={exc}")
        return output


def main() -> None:
    # Parse arguments once and dispatch to the selected mode.
    args = parse_args()

    checkpoint_root = args.checkpoint_root
    if checkpoint_root is None:
        checkpoint_root = (
            "checkpoints/multi_agent/selfish"
            if args.mode == "multi-agent-selfish"
            else "checkpoints"
            if args.mode == "multi-agent-reward-shaped"
            else "checkpoints/single_agent"
        )

    results_root = args.results_root
    if results_root is None:
        results_root = (
            "results/multi_agent"
            if args.mode in {"multi-agent-selfish", "multi-agent-reward-shaped"}
            else "results/single_agent"
        )

    if args.mode in {"single-agent", "multi-agent-selfish", "multi-agent-reward-shaped"}:
        # Translate CLI args into a strongly-typed training config.
        cfg = SingleAgentTrainConfig(
            backend=args.backend,
            substrate_name=args.substrate,
            focal_agent=args.focal_agent,
            num_agents=args.num_agents,
            num_envs=args.num_envs,
            stop_iters=args.stop_iters,
            checkpoint_every=args.checkpoint_every,
            checkpoint_root=checkpoint_root,
            results_root=results_root,
            run_name=args.run_name,
            num_env_runners=args.num_env_runners,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs if args.num_epochs is not None else args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            clip_vloss=args.clip_vloss,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            anneal_lr=args.anneal_lr,
            normalize_rgb=args.normalize_rgb,
            num_gpus=args.num_gpus,
            seed=args.seed,
            include_ready_to_shoot=args.include_ready_to_shoot,
            no_op_action=args.no_op_action,
            reward_type=args.reward_type,
            reward_alpha=args.reward_alpha,
            shaping_begin=args.shaping_begin,
            rew_shaping_horizon=args.rew_shaping_horizon,
            deontological_max_bonus=args.deontological_max_bonus,
            virtue_scale=args.virtue_scale,
        )
        # Log backend selection for traceability.
        print(f"mode={args.mode} | backend={cfg.backend}")

        # Execute training and print structured output for later scripting.
        if args.mode == "single-agent" and cfg.backend == "rllib":
            output = run_single_agent_ppo(cfg)
        elif args.mode == "single-agent" and cfg.backend == "cleanrl":
            output = run_single_agent_cleanrl(cfg)
        elif args.mode == "multi-agent-selfish" and cfg.backend == "cleanrl":
            output = run_multi_agent_selfish_cleanrl(cfg)
        elif args.mode == "multi-agent-reward-shaped" and cfg.backend == "cleanrl":
            output = run_reward_shaped_shared_cleanrl(cfg)
        elif args.mode == "multi-agent-selfish" and cfg.backend == "rllib":
            raise ValueError("multi-agent-selfish with backend=rllib is deprecated. Use backend=cleanrl.")
        elif args.mode == "multi-agent-reward-shaped" and cfg.backend == "rllib":
            raise ValueError("multi-agent-reward-shaped with backend=rllib is deprecated. Use backend=cleanrl.")
        else:
            raise ValueError(f"Unsupported mode/backend combination: mode={args.mode}, backend={cfg.backend}")
        output = _auto_plot_training_curves(output)
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
