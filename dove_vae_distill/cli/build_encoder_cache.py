import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
from transformers import set_seed

from dove_vae_distill.data import (
    build_training_dataset,
    frames_255_to_model_range,
    parse_resolution,
    set_sample_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build degraded RGB/moments cache for VAE encoder distillation"
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--video_column", type=str, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_resolution", type=parse_resolution, default="17x256x384")
    parser.add_argument(
        "--degradation_config",
        type=str,
        default="finetune/configs/degradation_image_video.yaml",
    )
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale_mode", type=str, default="bilinear")
    parser.add_argument("--prompt_cache", type=str, default="prompt_embeddings")
    parser.add_argument("--empty_prompt", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    frames, height, width = args.train_resolution
    if (frames - 1) % 4 != 0:
        raise ValueError("train_resolution frames must satisfy (frames - 1) % 4 == 0")
    if height % 16 or width % 16:
        raise ValueError("train_resolution height and width must be divisible by 16")

    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    set_seed(args.seed)
    dataset = build_training_dataset(args, frames=frames, height=height, width=width)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device).eval()

    records = []
    failures = []
    for sample_idx in range(args.num_samples):
        dataset_index = args.start_index + sample_idx * args.stride
        if dataset_index >= len(dataset):
            break
        set_sample_seed(args.seed, dataset_index)
        video_path = dataset.videos[dataset_index]
        try:
            hq_raw, lq_raw = dataset.preprocess(video_path)[:2]
            if hq_raw.shape[0] != frames or hq_raw.shape[-2:] != (height, width):
                raise ValueError(f"Unexpected HQ shape: {tuple(hq_raw.shape)}")
            lq_up_raw = F.interpolate(
                lq_raw,
                size=(height, width),
                mode=args.upscale_mode,
                align_corners=False,
            )
            condition = frames_255_to_model_range(lq_up_raw).unsqueeze(0)
            condition = condition.to(device=device, dtype=dtype)
            moments = vae.encode(condition).latent_dist.parameters
            sample_name = f"{len(records):05d}.pt"
            torch.save(
                {
                    "video": lq_up_raw.round().clamp(0, 255).to(torch.uint8).unsqueeze(0),
                    "teacher_moments": moments.cpu().to(torch.float16),
                    "source_video": str(video_path),
                    "dataset_index": dataset_index,
                    "seed": args.seed + dataset_index * 1009,
                },
                samples_dir / sample_name,
            )
            records.append(
                {
                    "sample": sample_name,
                    "source_video": str(video_path),
                    "dataset_index": dataset_index,
                }
            )
            print(f"[{len(records)}/{args.num_samples}] cached {video_path.name}", flush=True)
            del condition, moments
            torch.cuda.empty_cache()
        except Exception as exc:
            failures.append({"source_video": str(video_path), "error": repr(exc)})
            print(f"Failed {video_path}: {exc!r}", flush=True)

    manifest = {
        "format": "dove_vae_encoder_distill_cache_v2_degraded_lq",
        "count": len(records),
        "requested_count": args.num_samples,
        "train_resolution": {"frames": frames, "height": height, "width": width},
        "seed": args.seed,
        "model_path": str(args.model_path),
        "video_column": args.video_column,
        "degradation_config": args.degradation_config,
        "samples": records,
        "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"count": len(records), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
