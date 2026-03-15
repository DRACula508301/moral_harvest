from __future__ import annotations

import argparse
import json

from moral_harvest.experiments.multi_agent_selfish_cleanrl import run_multi_agent_selfish_cleanrl
from moral_harvest.experiments.single_agent_cleanrl import run_single_agent_cleanrl
from moral_harvest.experiments.single_agent_ppo import run_single_agent_ppo
from moral_harvest.experiments.vanilla_selfish import run_vanilla_selfish_ippo
from moral_harvest.training.config import SingleAgentTrainConfig


# Parse CLI arguments for single-agent PPO training.
def parse_args() -> argparse.Namespace:
    # Build the top-level parser and expose only the current training mode.
    parser = argparse.ArgumentParser(description="Moral Harvest training CLI")

    # Environment and stopping controls.
    parser.add_argument("--mode", choices=["single-agent", "multi-agent-selfish"], default="single-agent")
    parser.add_argument("--backend", choices=["rllib", "cleanrl"], default="rllib")
    parser.add_argument("--substrate", default="commons_harvest__open")
    parser.add_argument("--focal-agent", default="player_0")
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--stop-iters", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--run-name", default=None)

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
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--num-gpus", type=int, default=None, help="Optional override; default auto-detects GPU")
    parser.add_argument("--seed", type=int, default=None)

    # Observation/action wrapper controls.
    parser.add_argument("--include-ready-to-shoot", action="store_true")
    parser.add_argument("--no-op-action", type=int, default=0)

    # Return parsed CLI values for downstream config construction.
    return parser.parse_args()


# Entry point for running configured training modes.
def main() -> None:
    # Parse arguments once and dispatch to the selected mode.
    args = parse_args()

    checkpoint_root = args.checkpoint_root
    if checkpoint_root is None:
        checkpoint_root = (
            "checkpoints/multi_agent/selfish"
            if args.mode == "multi-agent-selfish"
            else "checkpoints/single_agent"
        )

    results_root = args.results_root
    if results_root is None:
        results_root = "results/multi_agent" if args.mode == "multi-agent-selfish" else "results/single_agent"

    if args.mode in {"single-agent", "multi-agent-selfish"}:
        # Translate CLI args into a strongly-typed training config.
        cfg = SingleAgentTrainConfig(
            backend=args.backend,
            substrate_name=args.substrate,
            focal_agent=args.focal_agent,
            num_agents=args.num_agents,
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
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            num_gpus=args.num_gpus,
            seed=args.seed,
            include_ready_to_shoot=args.include_ready_to_shoot,
            no_op_action=args.no_op_action,
        )
        # Log backend selection for traceability.
        print(f"mode={args.mode} | backend={cfg.backend}")

        # Execute training and print structured output for later scripting.
        if args.mode == "single-agent" and cfg.backend == "rllib":
            output = run_single_agent_ppo(cfg)
        elif args.mode == "single-agent" and cfg.backend == "cleanrl":
            output = run_single_agent_cleanrl(cfg)
        elif args.mode == "multi-agent-selfish" and cfg.backend == "rllib":
            output = run_vanilla_selfish_ippo(cfg)
        elif args.mode == "multi-agent-selfish" and cfg.backend == "cleanrl":
            output = run_multi_agent_selfish_cleanrl(cfg)
        else:
            raise ValueError(f"Unsupported mode/backend combination: mode={args.mode}, backend={cfg.backend}")
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
