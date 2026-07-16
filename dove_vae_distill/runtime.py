from pathlib import Path
from collections.abc import Sequence

import torch

from .decoder import build_student_decoder, decoder_profile_from_checkpoint
from .encoder import build_student_encoder, encoder_profile_from_checkpoint


def _load_checkpoint(path: str | Path | None):
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def install_student_vae(
    pipe,
    *,
    decoder_profile: Sequence[int] | None = None,
    decoder_checkpoint_path: str | Path | None = None,
    encoder_profile: Sequence[int] | None = None,
    encoder_mid_block_layers: int = 1,
    encoder_checkpoint_path: str | Path | None = None,
) -> dict:
    encoder_checkpoint = _load_checkpoint(encoder_checkpoint_path)
    if encoder_checkpoint is not None:
        checkpoint_profile, checkpoint_mid_layers = encoder_profile_from_checkpoint(
            encoder_checkpoint
        )
        if encoder_profile is not None and tuple(encoder_profile) != checkpoint_profile:
            raise ValueError(
                "Encoder block profile does not match checkpoint: "
                f"args={tuple(encoder_profile)}, checkpoint={checkpoint_profile}"
            )
        if encoder_mid_block_layers != checkpoint_mid_layers:
            raise ValueError(
                "Encoder mid-block depth does not match checkpoint: "
                f"args={encoder_mid_block_layers}, checkpoint={checkpoint_mid_layers}"
            )
        encoder_profile = checkpoint_profile

    installed_encoder_profile = None
    if encoder_profile is not None:
        teacher_encoder = pipe.vae.encoder
        student_encoder, installed_encoder_profile, load_result = build_student_encoder(
            pipe.vae,
            layers_per_down_block=encoder_profile,
            mid_block_layers=encoder_mid_block_layers,
            device=pipe.device,
            dtype=next(teacher_encoder.parameters()).dtype,
        )
        if encoder_checkpoint is not None:
            student_encoder.load_state_dict(encoder_checkpoint["encoder"], strict=True)
        pipe.vae.encoder = student_encoder.eval()
        del teacher_encoder
        torch.cuda.empty_cache()
        print(
            "Installed layer-pruned VAE encoder: "
            f"layers_per_down_block={installed_encoder_profile}, "
            f"mid_block_layers={encoder_mid_block_layers}, "
            f"checkpoint={encoder_checkpoint_path}, "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )

    decoder_checkpoint = _load_checkpoint(decoder_checkpoint_path)
    if decoder_checkpoint is not None:
        checkpoint_profile = decoder_profile_from_checkpoint(decoder_checkpoint)
        if decoder_profile is not None and tuple(decoder_profile) != checkpoint_profile:
            raise ValueError(
                "Decoder block profile does not match checkpoint: "
                f"args={tuple(decoder_profile)}, checkpoint={checkpoint_profile}"
            )
        decoder_profile = checkpoint_profile

    installed_decoder_profile = None
    if decoder_profile is not None:
        teacher_decoder = pipe.vae.decoder
        student_decoder, installed_decoder_profile, load_result = build_student_decoder(
            pipe.vae,
            layers_per_up_block=decoder_profile,
            device=pipe.device,
            dtype=next(teacher_decoder.parameters()).dtype,
        )
        if decoder_checkpoint is not None:
            student_decoder.load_state_dict(decoder_checkpoint["decoder"], strict=True)
        pipe.vae.decoder = student_decoder.eval()
        del teacher_decoder
        torch.cuda.empty_cache()
        print(
            "Installed layer-pruned VAE decoder: "
            f"layers_per_up_block={installed_decoder_profile}, "
            f"checkpoint={decoder_checkpoint_path}, "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )

    return {
        "encoder_profile": installed_encoder_profile,
        "encoder_mid_block_layers": encoder_mid_block_layers,
        "decoder_profile": installed_decoder_profile,
    }
