from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# Apply orthogonal initialization to Conv/Linear layers.
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


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
                layer_init(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=int(out_channels),
                        kernel_size=kernel_size,
                        stride=int(stride),
                    )
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
            mlp_layers.append(layer_init(nn.Linear(current_size, int(hidden_size))))
            mlp_layers.append(nn.ReLU())
            current_size = int(hidden_size)
        self.trunk = nn.Sequential(*mlp_layers)

        # Build separate policy and value heads.
        self.actor = layer_init(nn.Linear(current_size, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(current_size, 1), std=1.0)

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
