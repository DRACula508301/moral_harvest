from __future__ import annotations

from dataclasses import dataclass, field


# Dataclass container for single-agent PPO training parameters.
@dataclass(slots=True)
class SingleAgentTrainConfig:
    # Training backend selection.
    backend: str = "rllib"

    # Environment selection.
    substrate_name: str = "commons_harvest__open"
    focal_agent: str = "player_0"
    num_agents: int = 10

    # Training horizon and checkpoint cadence.
    stop_iters: int = 1000
    checkpoint_every: int = 100
    checkpoint_root: str = "checkpoints/single_agent"
    results_root: str = "results/single_agent"
    run_name: str | None = None

    # Rollout/optimizer settings.
    num_env_runners: int = 0
    train_batch_size: int = 4000
    minibatch_size: int = 256
    num_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # CNN/MLP model settings for 88x88x3 Harvest observations.
    conv_filters: list[list[int | list[int]]] = field(
        default_factory=lambda: [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [64, [3, 3], 1],
        ]
    )
    conv_activation: str = "relu"
    fcnet_hiddens: list[int] = field(default_factory=lambda: [256, 256])
    fcnet_activation: str = "relu"

    # Runtime framework/resources.
    framework: str = "torch"
    num_gpus: int | None = None
    seed: int | None = None

    # Wrapper-specific controls.
    no_op_action: int = 0
    include_ready_to_shoot: bool = False
