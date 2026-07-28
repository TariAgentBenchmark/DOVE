import argparse
import json
import random
from pathlib import Path

import pyiqa
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from dove_vae_distill.decoder import build_student_decoder


def parse_args():
    parser = argparse.ArgumentParser(description="Distill a lightweight DOVE VAE decoder")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layers_per_block", type=int, default=0)
    parser.add_argument("--layers_per_up_block", type=int, nargs=4, default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--teacher_l1_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--gt_l1_weight", type=float, default=0.0)
    parser.add_argument("--frame_diff_weight", type=float, default=0.2)
    parser.add_argument("--dists_weight", type=float, default=0.01)
    parser.add_argument("--dists_every", type=int, default=4)
    parser.add_argument("--lpips_weight", type=float, default=0.0)
    parser.add_argument("--lpips_every", type=int, default=4)
    parser.add_argument(
        "--perceptual_target",
        choices=("teacher", "gt"),
        default="teacher",
        help="Reference used by DISTS, LPIPS, and high-frequency losses",
    )
    parser.add_argument("--high_frequency_weight", type=float, default=0.0)
    parser.add_argument("--clipiqa_weight", type=float, default=0.0)
    parser.add_argument("--clipiqa_every", type=int, default=1)
    parser.add_argument("--clipiqa_target", type=float, default=0.0)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class DecoderCacheDataset(Dataset):
    def __init__(self, cache_dir):
        self.samples = sorted((cache_dir / "samples").glob("*.pt"))
        if not self.samples:
            raise ValueError(f"No cache samples found in {cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = torch.load(self.samples[index], map_location="cpu", weights_only=False)
        teacher = sample["teacher_video"][0]
        ground_truth = sample.get("gt_video", sample["teacher_video"])[0]
        return sample["latent"][0], teacher, ground_truth


def decode_temporal_batches(decoder, latent, frame_batch_size=2):
    num_frames = latent.shape[2]
    num_batches = max(num_frames // frame_batch_size, 1)
    remaining_frames = num_frames % frame_batch_size
    conv_cache = None
    outputs = []
    for index in range(num_batches):
        start = frame_batch_size * index + (0 if index == 0 else remaining_frames)
        end = frame_batch_size * (index + 1) + remaining_frames
        output, conv_cache = decoder(latent[:, :, start:end], conv_cache=conv_cache)
        outputs.append(output)
    return torch.cat(outputs, dim=2)


def build_student(vae, layers_per_block, layers_per_up_block, device):
    student, profile, load_result = build_student_decoder(
        vae,
        layers_per_block=layers_per_block,
        layers_per_up_block=layers_per_up_block,
        device=device,
        dtype=torch.bfloat16,
    )
    print(
        f"Initialized student: missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    return student, profile


def save_checkpoint(student, optimizer, step, args, decoder_profile):
    checkpoint_dir = args.output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": {key: value.detach().cpu() for key, value in student.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": {
                "layers_per_block": args.layers_per_block if args.layers_per_up_block is None else 0,
                "layers_per_up_block": list(decoder_profile),
                "objective": {
                    "teacher_l1_weight": args.teacher_l1_weight,
                    "mse_weight": args.mse_weight,
                    "gt_l1_weight": args.gt_l1_weight,
                    "frame_diff_weight": args.frame_diff_weight,
                    "dists_weight": args.dists_weight,
                    "lpips_weight": args.lpips_weight,
                    "perceptual_target": args.perceptual_target,
                    "high_frequency_weight": args.high_frequency_weight,
                    "clipiqa_weight": args.clipiqa_weight,
                    "clipiqa_target": args.clipiqa_target,
                },
            },
        },
        checkpoint_dir / "decoder.pt",
    )


def laplacian_loss(prediction, target):
    kernel = prediction.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(prediction.shape[1], 1, 1, 1)
    pred_edges = F.conv2d(prediction, kernel, padding=1, groups=prediction.shape[1])
    target_edges = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return F.l1_loss(pred_edges.float(), target_edges.float())


def freeze_metric(metric):
    if metric is None:
        return None
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


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
    student, decoder_profile = build_student(
        teacher_vae,
        args.layers_per_block,
        args.layers_per_up_block,
        device,
    )
    del teacher_vae
    torch.cuda.empty_cache()

    dataset = DecoderCacheDataset(args.cache_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["decoder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        start_step = int(checkpoint["step"])
        print(
            f"Resumed decoder distillation from step {start_step} "
            f"with learning_rate={args.learning_rate}"
        )
    dists = freeze_metric(
        pyiqa.create_metric("dists", as_loss=True).to(device)
        if args.dists_weight > 0
        else None
    )
    lpips = freeze_metric(
        pyiqa.create_metric("lpips", as_loss=True).to(device)
        if args.lpips_weight > 0
        else None
    )
    clipiqa = freeze_metric(
        pyiqa.create_metric("clipiqa", as_loss=True).to(device)
        if args.clipiqa_weight > 0
        else None
    )
    log_path = args.output_dir / "train_log.jsonl"

    student.train()
    for step in trange(start_step + 1, args.max_steps + 1, desc="Decoder distillation"):
        try:
            latent, teacher_u8, gt_u8 = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            latent, teacher_u8, gt_u8 = next(iterator)

        latent = latent.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        teacher = (
            teacher_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 127.5
            - 1.0
        )
        ground_truth = (
            gt_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 127.5
            - 1.0
        )
        perceptual_target = ground_truth if args.perceptual_target == "gt" else teacher
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = decode_temporal_batches(student, latent)
            l1_loss = (
                F.l1_loss(prediction.float(), teacher) * args.teacher_l1_weight
            )
            mse_loss = F.mse_loss(prediction.float(), teacher) * args.mse_weight
            gt_l1_loss = F.l1_loss(prediction.float(), ground_truth) * args.gt_l1_weight
            pred_diff = prediction[:, :, 1:] - prediction[:, :, :-1]
            target_diff = teacher[:, :, 1:] - teacher[:, :, :-1]
            frame_diff_loss = F.l1_loss(pred_diff.float(), target_diff) * args.frame_diff_weight
            perceptual_loss = torch.zeros((), device=device)
            lpips_loss = torch.zeros((), device=device)
            high_frequency_loss = torch.zeros((), device=device)
            clipiqa_loss = torch.zeros((), device=device)
            student_clipiqa = torch.zeros((), device=device)
            teacher_clipiqa = torch.zeros((), device=device)
            target_clipiqa = torch.zeros((), device=device)
            use_clipiqa = clipiqa is not None and step % args.clipiqa_every == 0
            if (
                (dists is not None and step % args.dists_every == 0)
                or (lpips is not None and step % args.lpips_every == 0)
                or args.high_frequency_weight > 0
                or use_clipiqa
            ):
                frame_index = random.randrange(prediction.shape[2])
                pred_frame = (prediction[:, :, frame_index].float() * 0.5 + 0.5).clamp(0.0, 1.0)
                target_frame = (
                    perceptual_target[:, :, frame_index] * 0.5 + 0.5
                ).clamp(0.0, 1.0)
                teacher_frame = (teacher[:, :, frame_index] * 0.5 + 0.5).clamp(0.0, 1.0)
                if dists is not None and step % args.dists_every == 0:
                    perceptual_loss = dists(pred_frame, target_frame) * args.dists_weight
                if lpips is not None and step % args.lpips_every == 0:
                    lpips_loss = lpips(pred_frame, target_frame) * args.lpips_weight
                if args.high_frequency_weight > 0:
                    high_frequency_loss = (
                        laplacian_loss(pred_frame, target_frame) * args.high_frequency_weight
                    )
                if use_clipiqa:
                    student_clipiqa = clipiqa(pred_frame).mean()
                    teacher_clipiqa = clipiqa(teacher_frame).detach().mean()
                    target_clipiqa = torch.maximum(
                        teacher_clipiqa,
                        teacher_clipiqa.new_tensor(args.clipiqa_target),
                    )
                    clipiqa_loss = (
                        F.relu(target_clipiqa - student_clipiqa) * args.clipiqa_weight
                    )
            loss = (
                l1_loss
                + mse_loss
                + gt_l1_loss
                + frame_diff_loss
                + perceptual_loss
                + lpips_loss
                + high_frequency_loss
                + clipiqa_loss
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": loss.item(),
            "l1_loss": l1_loss.item(),
            "mse_loss": mse_loss.item(),
            "gt_l1_loss": gt_l1_loss.item(),
            "frame_diff_loss": frame_diff_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
            "lpips_loss": lpips_loss.item(),
            "high_frequency_loss": high_frequency_loss.item(),
            "clipiqa_loss": clipiqa_loss.item(),
            "student_clipiqa": student_clipiqa.item(),
            "teacher_clipiqa": teacher_clipiqa.item(),
            "target_clipiqa": target_clipiqa.item(),
            "grad_norm": float(grad_norm),
        }
        with open(log_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(student, optimizer, step, args, decoder_profile)


if __name__ == "__main__":
    main()
