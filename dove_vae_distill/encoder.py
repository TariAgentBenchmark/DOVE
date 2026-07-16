from collections.abc import Sequence

import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXEncoder3D


def normalize_encoder_profile(
    block_count: int,
    layers_per_block: int = 0,
    layers_per_down_block: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if layers_per_down_block is None:
        if layers_per_block <= 0:
            raise ValueError("layers_per_block must be positive when no block profile is provided")
        profile = (layers_per_block,) * block_count
    else:
        profile = tuple(int(value) for value in layers_per_down_block)
        if len(profile) != block_count:
            raise ValueError(f"Expected {block_count} down-block layer counts, got {len(profile)}")
        if layers_per_block > 0 and any(value != layers_per_block for value in profile):
            raise ValueError("layers_per_block conflicts with layers_per_down_block")
    if any(value <= 0 for value in profile):
        raise ValueError(f"Every down-block layer count must be positive, got {profile}")
    return profile


def build_student_encoder(
    vae,
    layers_per_block: int = 0,
    layers_per_down_block: Sequence[int] | None = None,
    mid_block_layers: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
):
    if mid_block_layers < 0 or mid_block_layers > 2:
        raise ValueError(f"mid_block_layers must be between 0 and 2, got {mid_block_layers}")
    profile = normalize_encoder_profile(
        len(vae.config.down_block_types),
        layers_per_block=layers_per_block,
        layers_per_down_block=layers_per_down_block,
    )
    encoder = CogVideoXEncoder3D(
        in_channels=vae.config.in_channels,
        out_channels=vae.config.latent_channels,
        down_block_types=tuple(vae.config.down_block_types),
        block_out_channels=tuple(vae.config.block_out_channels),
        layers_per_block=max(profile),
        act_fn=vae.config.act_fn,
        norm_eps=vae.config.norm_eps,
        norm_num_groups=vae.config.norm_num_groups,
        temporal_compression_ratio=vae.config.temporal_compression_ratio,
    )
    for down_block, layer_count in zip(encoder.down_blocks, profile, strict=True):
        down_block.resnets = torch.nn.ModuleList(list(down_block.resnets[:layer_count]))
    encoder.mid_block.resnets = torch.nn.ModuleList(
        list(encoder.mid_block.resnets[:mid_block_layers])
    )
    if device is not None or dtype is not None:
        encoder = encoder.to(device=device, dtype=dtype)
    load_result = encoder.load_state_dict(vae.encoder.state_dict(), strict=False)
    return encoder, profile, load_result


def encoder_profile_from_checkpoint(checkpoint: dict) -> tuple[tuple[int, ...], int]:
    config = checkpoint.get("config", {})
    profile = config.get("layers_per_down_block")
    if profile is None:
        layers_per_block = int(config.get("layers_per_block", 0))
        if layers_per_block <= 0:
            raise ValueError("Encoder checkpoint has no valid layer configuration")
        profile = (layers_per_block,) * 4
    return tuple(int(value) for value in profile), int(config.get("mid_block_layers", 1))
