import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from dove_vae_distill.encoder import build_student_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Distill a layer-pruned DOVE VAE encoder")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layers_per_block", type=int, default=0)
    parser.add_argument("--layers_per_down_block", type=int, nargs=4, default=None)
    parser.add_argument("--mid_block_layers", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--mean_mse_weight", type=float, default=0.5)
    parser.add_argument("--logvar_l1_weight", type=float, default=0.1)
    parser.add_argument("--logvar_mse_weight", type=float, default=0.05)
    parser.add_argument("--sample_weight", type=float, default=0.25)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class EncoderCacheDataset(Dataset):
    def __init__(self, cache_dir):
        self.samples = sorted((cache_dir / "samples").glob("*.pt"))
        if not self.samples:
            raise ValueError(f"No cache samples found in {cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = torch.load(self.samples[index], map_location="cpu", weights_only=False)
        video = sample["video"]
        teacher_moments = sample["teacher_moments"]
        if video.ndim != 5 or video.shape[0] != 1:
            raise ValueError(
                f"Expected cached video with shape [1,T,C,H,W] or [1,C,T,H,W], "
                f"got {tuple(video.shape)}"
            )
        video = video[0]
        if video.shape[0] == 3:
            pass
        elif video.shape[1] == 3:
            video = video.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError(f"Could not identify RGB channel axis in {tuple(video.shape)}")
        if teacher_moments.ndim != 5 or teacher_moments.shape[0] != 1:
            raise ValueError(
                f"Expected teacher moments with shape [1,C,T,H,W], "
                f"got {tuple(teacher_moments.shape)}"
            )
        return video, teacher_moments[0]


def encode_temporal_batches(encoder, video, frame_batch_size=8):
    num_frames = video.shape[2]
    num_batches = max(num_frames // frame_batch_size, 1)
    remaining_frames = num_frames % frame_batch_size
    conv_cache = None
    outputs = []
    for index in range(num_batches):
        start = frame_batch_size * index + (0 if index == 0 else remaining_frames)
        end = frame_batch_size * (index + 1) + remaining_frames
        output, conv_cache = encoder(video[:, :, start:end], conv_cache=conv_cache)
        outputs.append(output)
    return torch.cat(outputs, dim=2)


def save_checkpoint(student, optimizer, step, args, encoder_profile):
    checkpoint_dir = args.output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": {key: value.detach().cpu() for key, value in student.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": {
                "layers_per_block": args.layers_per_block
                if args.layers_per_down_block is None
                else 0,
                "layers_per_down_block": list(encoder_profile),
                "mid_block_layers": args.mid_block_layers,
            },
        },
        checkpoint_dir / "encoder.pt",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    teacher_vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    student, encoder_profile, load_result = build_student_encoder(
        teacher_vae,
        layers_per_block=args.layers_per_block,
        layers_per_down_block=args.layers_per_down_block,
        mid_block_layers=args.mid_block_layers,
        device=device,
        dtype=torch.bfloat16,
    )
    frame_batch_size = int(teacher_vae.num_sample_frames_batch_size)
    print(
        f"Initialized student: profile={encoder_profile}, mid={args.mid_block_layers}, "
        f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
    )
    del teacher_vae
    torch.cuda.empty_cache()

    dataset = EncoderCacheDataset(args.cache_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["encoder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        start_step = int(checkpoint["step"])

    log_path = args.output_dir / "train_log.jsonl"
    student.train()
    for step in trange(start_step + 1, args.max_steps + 1, desc="Encoder distillation"):
        try:
            video_u8, target_moments = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            video_u8, target_moments = next(iterator)

        video = video_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 127.5 - 1.0
        video = video.to(dtype=torch.bfloat16)
        target_moments = target_moments.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = encode_temporal_batches(student, video, frame_batch_size).float()
            pred_mean, pred_logvar = prediction.chunk(2, dim=1)
            target_mean, target_logvar = target_moments.chunk(2, dim=1)
            target_logvar = target_logvar.clamp(-30.0, 20.0)
            pred_logvar = pred_logvar.clamp(-30.0, 20.0)
            mean_l1_loss = F.l1_loss(pred_mean, target_mean)
            mean_mse_loss = F.mse_loss(pred_mean, target_mean) * args.mean_mse_weight
            logvar_l1_loss = (
                F.l1_loss(pred_logvar, target_logvar) * args.logvar_l1_weight
            )
            logvar_mse_loss = (
                F.mse_loss(pred_logvar, target_logvar) * args.logvar_mse_weight
            )
            noise = torch.randn_like(pred_mean)
            pred_sample = pred_mean + torch.exp(0.5 * pred_logvar) * noise
            target_sample = target_mean + torch.exp(0.5 * target_logvar) * noise
            sample_loss = F.l1_loss(pred_sample, target_sample) * args.sample_weight
            loss = (
                mean_l1_loss
                + mean_mse_loss
                + logvar_l1_loss
                + logvar_mse_loss
                + sample_loss
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": loss.item(),
            "mean_l1_loss": mean_l1_loss.item(),
            "mean_mse_loss": mean_mse_loss.item(),
            "logvar_l1_loss": logvar_l1_loss.item(),
            "logvar_mse_loss": logvar_mse_loss.item(),
            "sample_loss": sample_loss.item(),
            "grad_norm": float(grad_norm),
        }
        with open(log_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(student, optimizer, step, args, encoder_profile)


if __name__ == "__main__":
    main()
