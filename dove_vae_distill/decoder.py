from collections.abc import Sequence

import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXDecoder3D


def normalize_decoder_profile(
    block_count: int,
    layers_per_block: int = 0,
    layers_per_up_block: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if layers_per_up_block is None:
        if layers_per_block <= 0:
            raise ValueError("layers_per_block must be positive when no block profile is provided")
        profile = (layers_per_block,) * block_count
    else:
        profile = tuple(int(value) for value in layers_per_up_block)
        if len(profile) != block_count:
            raise ValueError(f"Expected {block_count} up-block layer counts, got {len(profile)}")
        if layers_per_block > 0 and any(value != layers_per_block for value in profile):
            raise ValueError("layers_per_block conflicts with layers_per_up_block")
    if any(value <= 0 for value in profile):
        raise ValueError(f"Every up-block layer count must be positive, got {profile}")
    return profile


def build_student_decoder(
    vae,
    layers_per_block: int = 0,
    layers_per_up_block: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
):
    profile = normalize_decoder_profile(
        len(vae.config.up_block_types),
        layers_per_block=layers_per_block,
        layers_per_up_block=layers_per_up_block,
    )
    decoder = CogVideoXDecoder3D(
        in_channels=vae.config.latent_channels,
        out_channels=vae.config.out_channels,
        up_block_types=tuple(vae.config.up_block_types),
        block_out_channels=tuple(vae.config.block_out_channels),
        layers_per_block=max(profile),
        act_fn=vae.config.act_fn,
        norm_eps=vae.config.norm_eps,
        norm_num_groups=vae.config.norm_num_groups,
        temporal_compression_ratio=vae.config.temporal_compression_ratio,
    )
    for up_block, layer_count in zip(decoder.up_blocks, profile, strict=True):
        # Diffusers constructs layers_per_block + 1 residual layers per up block.
        up_block.resnets = torch.nn.ModuleList(list(up_block.resnets[: layer_count + 1]))
    if device is not None or dtype is not None:
        decoder = decoder.to(device=device, dtype=dtype)
    load_result = decoder.load_state_dict(vae.decoder.state_dict(), strict=False)
    return decoder, profile, load_result


def decoder_profile_from_checkpoint(checkpoint: dict, block_count: int = 4) -> tuple[int, ...]:
    config = checkpoint.get("config", {})
    profile = config.get("layers_per_up_block")
    if profile is not None:
        return tuple(int(value) for value in profile)
    layers_per_block = int(config.get("layers_per_block", 0))
    if layers_per_block > 0:
        return (layers_per_block,) * block_count
    raise ValueError("Decoder checkpoint has no valid layer configuration")
