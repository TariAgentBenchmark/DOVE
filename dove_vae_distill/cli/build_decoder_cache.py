import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import set_seed

from dove_vae_distill.data import (
    build_training_dataset,
    encode_prompt,
    frames_255_to_model_range,
    load_teacher_pipeline,
    pad_temporal_for_dove,
    parse_resolution,
    set_sample_seed,
)
from inference_script import process_video


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Cache real DOVE output latents and original-decoder RGB targets"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--video_column", type=str, required=True)
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
    parser.add_argument("--sr_noise_step", type=int, default=399)
    parser.add_argument("--upscale_mode", type=str, default="bilinear")
    parser.add_argument("--prompt_cache", type=str, default="prompt_embeddings")
    parser.add_argument("--empty_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable_vae_slicing_tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frames, height, width = args.train_resolution
    if (frames - 1) % 8 != 0:
        raise ValueError("train_resolution frames must satisfy (frames - 1) % 8 == 0")
    if height % 16 or width % 16:
        raise ValueError("train_resolution height and width must be divisible by 16")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Cache already exists: {args.output_dir}")
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    set_seed(args.seed)
    dataset = build_training_dataset(args, frames=frames, height=height, width=width)
    pipe = load_teacher_pipeline(args, dtype=dtype, device=device)
    prompt_embedding_cache = {}
    records = []
    failures = []

    for sample_idx in range(args.num_samples):
        dataset_index = args.start_index + sample_idx * args.stride
        if dataset_index >= len(dataset):
            break
        set_sample_seed(args.seed, dataset_index)
        video_path = dataset.videos[dataset_index]
        prompt = "" if args.empty_prompt else dataset.prompts[dataset_index]
        try:
            preprocessed = dataset.preprocess(video_path)
            hq_raw, lq_raw = preprocessed[:2]
            if hq_raw.shape[0] != frames or hq_raw.shape[-2:] != (height, width):
                raise ValueError(
                    f"Unexpected HQ shape for {video_path}: {tuple(hq_raw.shape)}"
                )
            lq_up_raw = F.interpolate(
                lq_raw,
                size=(height, width),
                mode=args.upscale_mode,
                align_corners=False,
            )
            lq_video = frames_255_to_model_range(lq_up_raw)
            condition = lq_video.unsqueeze(0).to(device=device, dtype=dtype)
            condition, pad_f = pad_temporal_for_dove(condition)
            if pad_f:
                raise ValueError("Trajectory decoder cache currently requires an unpadded frame count")

            if prompt not in prompt_embedding_cache:
                prompt_embedding_cache[prompt] = encode_prompt(
                    pipe,
                    prompt,
                    dtype=dtype,
                    device=device,
                )
            prompt_embedding = prompt_embedding_cache[prompt]
            teacher_video_01, decoder_input = process_video(
                pipe=pipe,
                video=condition,
                prompt=prompt,
                sr_noise_step=args.sr_noise_step,
                empty_prompt_embedding=prompt_embedding.unsqueeze(0),
                return_decoder_input=True,
            )
            teacher_video = (
                teacher_video_01.detach().cpu().float().clamp(0.0, 1.0) * 255.0
            ).round().to(torch.uint8)
            sample_name = f"{len(records):05d}.pt"
            torch.save(
                {
                    "latent": decoder_input.detach().cpu().to(torch.float16),
                    "teacher_video": teacher_video,
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
            del condition, decoder_input, teacher_video_01, teacher_video
            torch.cuda.empty_cache()
        except Exception as exc:
            failures.append({"source_video": str(video_path), "error": repr(exc)})
            print(f"Failed {video_path}: {exc!r}", flush=True)

    manifest = {
        "format": "dove_decoder_trajectory_cache_v1",
        "model_path": args.model_path,
        "num_entries": len(records),
        "train_resolution": {"frames": frames, "height": height, "width": width},
        "sr_noise_step": args.sr_noise_step,
        "records": records,
        "failures": failures,
    }
    with open(args.output_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"count": len(records), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
