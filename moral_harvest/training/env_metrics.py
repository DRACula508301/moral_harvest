from __future__ import annotations

from typing import Any

import numpy as np


# Extract one WORLD.RGB frame per vectorized environment (when available).
def extract_world_rgb_frames(
    next_obs_raw: Any,
    num_envs: int,
    num_agents: int,
) -> list[np.ndarray]:
    if not isinstance(next_obs_raw, dict) or "WORLD.RGB" not in next_obs_raw:
        return []

    world_rgb = np.asarray(next_obs_raw["WORLD.RGB"])

    if world_rgb.ndim == 3 and world_rgb.shape[-1] >= 3:
        return [world_rgb[..., :3]]

    if world_rgb.ndim == 4 and world_rgb.shape[-1] >= 3:
        if world_rgb.shape[0] == num_envs:
            return [world_rgb[env_index, ..., :3] for env_index in range(num_envs)]
        if world_rgb.shape[0] == num_envs * num_agents:
            return [world_rgb[env_index * num_agents, ..., :3] for env_index in range(num_envs)]
        if num_envs == 1:
            return [world_rgb[0, ..., :3]]
        return [world_rgb[index, ..., :3] for index in range(min(num_envs, world_rgb.shape[0]))]

    if world_rgb.ndim == 5 and world_rgb.shape[-1] >= 3:
        if world_rgb.shape[0] == num_envs and world_rgb.shape[1] == num_agents:
            return [world_rgb[env_index, 0, ..., :3] for env_index in range(num_envs)]
        if num_envs == 1:
            return [world_rgb[0, 0, ..., :3]]

    return []


# Estimate active berries by counting red apple-like tiles in WORLD.RGB.
def count_active_berries_from_world_frame(world_frame: np.ndarray, sprite_size: int = 8) -> int:
    frame = np.asarray(world_frame)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return 0

    rgb = frame[..., :3].astype(np.float32)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    red_mask = (red >= 130.0) & (red >= 1.35 * green) & (red >= 1.25 * blue) & (green <= 190.0)

    height, width = red_mask.shape
    tile_rows = height // sprite_size
    tile_cols = width // sprite_size
    if tile_rows <= 0 or tile_cols <= 0:
        return 0

    cropped = red_mask[: tile_rows * sprite_size, : tile_cols * sprite_size]
    red_pixels_per_tile = cropped.reshape(tile_rows, sprite_size, tile_cols, sprite_size).sum(axis=(1, 3))

    active_apple_tiles = red_pixels_per_tile >= 4
    return int(active_apple_tiles.sum())
