import pytest

from dove_vae_distill.decoder import decoder_profile_from_checkpoint, normalize_decoder_profile
from dove_vae_distill.encoder import encoder_profile_from_checkpoint, normalize_encoder_profile


def test_normalize_encoder_profile():
    assert normalize_encoder_profile(4, layers_per_block=1) == (1, 1, 1, 1)
    assert normalize_encoder_profile(4, layers_per_down_block=(1, 1, 2, 2)) == (1, 1, 2, 2)


def test_normalize_encoder_profile_rejects_invalid_depth():
    with pytest.raises(ValueError):
        normalize_encoder_profile(4, layers_per_down_block=(1, 1, 1))
    with pytest.raises(ValueError):
        normalize_encoder_profile(4, layers_per_down_block=(1, 0, 1, 1))


def test_encoder_profile_from_checkpoint():
    checkpoint = {
        "config": {
            "layers_per_down_block": [1, 1, 1, 2],
            "mid_block_layers": 1,
        }
    }
    assert encoder_profile_from_checkpoint(checkpoint) == ((1, 1, 1, 2), 1)


def test_existing_decoder_profile_behavior_is_unchanged():
    assert normalize_decoder_profile(4, layers_per_up_block=(1, 1, 1, 2)) == (1, 1, 1, 2)


def test_decoder_profile_from_checkpoint_supports_uniform_and_nonuniform_profiles():
    assert decoder_profile_from_checkpoint(
        {"config": {"layers_per_up_block": [1, 1, 1, 2]}}
    ) == (1, 1, 1, 2)
    assert decoder_profile_from_checkpoint(
        {"config": {"layers_per_block": 2}}
    ) == (2, 2, 2, 2)
