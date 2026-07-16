import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

from finetune.datasets import RealSRDataset


class _FakeAccelerator:
    process_index = 0


class _DatasetTrainerStub:
    def __init__(self, args):
        self.args = args
        self.accelerator = _FakeAccelerator()

    def encode_video(self, video):
        raise RuntimeError("VAE distillation cache building must not encode through the dataset")

    def encode_text(self, prompt):
        raise RuntimeError("Prompt embeddings are produced by the DOVE teacher pipeline")


def parse_resolution(value: str) -> tuple[int, int, int]:
    try:
        frames, height, width = value.lower().split("x")
        return int(frames), int(height), int(width)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "train_resolution must use FxHxW format, for example 17x256x384"
        ) from exc


def set_sample_seed(seed: int, index: int) -> None:
    sample_seed = seed + index * 1009
    random.seed(sample_seed)
    np.random.seed(sample_seed % (2**32 - 1))
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)


def frames_255_to_model_range(frames: torch.Tensor) -> torch.Tensor:
    """Convert [F,C,H,W] values in 0..255 to [C,F,H,W] values in -1..1."""
    frames = frames.float().clamp(0.0, 255.0) / 255.0 * 2.0 - 1.0
    return frames.permute(1, 0, 2, 3).contiguous()


def pad_temporal_for_dove(video: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad [B,C,F,H,W] so that DOVE's `(F - 1) % 8 == 0` invariant holds."""
    frames = video.shape[2]
    remainder = (frames - 1) % 8
    if remainder == 0:
        return video, 0
    pad_frames = 8 - remainder
    last_frame = video[:, :, -1:]
    return torch.cat([video, last_frame.repeat(1, 1, pad_frames, 1, 1)], dim=2), pad_frames


@torch.inference_mode()
def encode_prompt(pipe, prompt: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    token_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    embedding = pipe.text_encoder(token_ids)[0].to(device=device, dtype=dtype)
    return embedding[0].detach().cpu()


def build_training_dataset(args, frames: int, height: int, width: int) -> RealSRDataset:
    dataset_args = SimpleNamespace(
        data_root=Path(args.data_root),
        video_column=Path(args.video_column),
        caption_column=None,
        is_cache=False,
        is_latent=False,
        crop_mode="random_crop",
        empty_prompt=args.empty_prompt,
        empty_ratio=0.0,
        prompt_cache=args.prompt_cache,
        degradation_config=args.degradation_config,
        model_name="dove-vae-distill-cache",
        train_resolution=(frames, height, width),
        arbitrary_scale=False,
    )
    return RealSRDataset(
        data_root=args.data_root,
        video_column=args.video_column,
        caption_column=None,
        max_num_frames=frames,
        height=height,
        width=width,
        trainer=_DatasetTrainerStub(dataset_args),
        prompt_cache=args.prompt_cache,
    )


def load_teacher_pipeline(args, dtype: torch.dtype, device: torch.device):
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipe.to(device)
    pipe.vae.eval()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    if args.enable_vae_slicing_tiling:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    return pipe
