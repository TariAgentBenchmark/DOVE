import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def read_video(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return torch.stack(frames)


def collect_videos(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def match_sequences(gt: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frames = min(gt.shape[0], pred.shape[0])
    height = min(gt.shape[-2], pred.shape[-2])
    width = min(gt.shape[-1], pred.shape[-1])
    return gt[:frames, :, :height, :width], pred[:frames, :, :height, :width]


def resize_for_flow(frames: torch.Tensor, max_side: int) -> torch.Tensor:
    height, width = frames.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    target_height = max(8, int(height * scale) // 8 * 8)
    target_width = max(8, int(width * scale) // 8 * 8)
    if (target_height, target_width) == (height, width):
        return frames
    return F.interpolate(frames, size=(target_height, target_width), mode="bilinear", align_corners=False)


def flow_warp(source: torch.Tensor, backward_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = source.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0).expand(batch, -1, -1) + backward_flow[:, 0]
    y = y.unsqueeze(0).expand(batch, -1, -1) + backward_flow[:, 1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = torch.stack(
        [2 * x / max(width - 1, 1) - 1, 2 * y / max(height - 1, 1) - 1],
        dim=-1,
    )
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped, valid.unsqueeze(1)


def consistency_mask(forward_flow: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
    warped_forward, valid = flow_warp(forward_flow, backward_flow)
    residual = backward_flow + warped_forward
    residual_sq = residual.square().sum(dim=1, keepdim=True)
    magnitude_sq = backward_flow.square().sum(dim=1, keepdim=True)
    magnitude_sq = magnitude_sq + warped_forward.square().sum(dim=1, keepdim=True)
    return valid & (residual_sq <= 0.01 * magnitude_sq + 0.5)


@torch.inference_mode()
def evaluate_sequence(
    raft,
    transforms,
    gt: torch.Tensor,
    pred: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    warp_errors = []
    frame_diff_errors = []
    valid_ratios = []

    for start in range(0, gt.shape[0] - 1, batch_size):
        end = min(start + batch_size, gt.shape[0] - 1)
        gt_prev = gt[start:end].to(device)
        gt_next = gt[start + 1:end + 1].to(device)
        pred_prev = pred[start:end].to(device)
        pred_next = pred[start + 1:end + 1].to(device)

        raft_prev, raft_next = transforms(gt_prev, gt_next)
        forward_flow = raft(raft_prev, raft_next)[-1]
        backward_flow = raft(raft_next, raft_prev)[-1]
        mask = consistency_mask(forward_flow, backward_flow)
        warped_pred, valid = flow_warp(pred_prev, backward_flow)
        mask = mask & valid

        pixel_error = (warped_pred - pred_next).abs().mean(dim=1, keepdim=True)
        for index in range(pixel_error.shape[0]):
            sample_mask = mask[index]
            if sample_mask.any():
                warp_errors.append(pixel_error[index][sample_mask].mean().item())
                valid_ratios.append(sample_mask.float().mean().item())

        frame_diff = (pred_next - pred_prev) - (gt_next - gt_prev)
        frame_diff_errors.extend(frame_diff.abs().mean(dim=(1, 2, 3)).cpu().tolist())

    if not warp_errors:
        raise ValueError("No valid temporal frame pairs")
    return {
        "raft_warp_l1": float(np.mean(warp_errors)),
        "frame_diff_l1": float(np.mean(frame_diff_errors)),
        "flow_valid_ratio": float(np.mean(valid_ratios)),
        "frame_pairs": len(warp_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAFT-based temporal consistency evaluation")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--flow_max_side", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms()

    gt_files = collect_videos(args.gt)
    pred_files = collect_videos(args.pred)
    common_names = sorted(set(gt_files) & set(pred_files))
    if not common_names:
        raise ValueError("No matching GT/pred video names")

    per_sample = {}
    for name in tqdm(common_names, desc="Temporal evaluation"):
        gt, pred = match_sequences(read_video(gt_files[name]), read_video(pred_files[name]))
        if gt.shape[0] < 2:
            continue
        gt = resize_for_flow(gt, args.flow_max_side)
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        per_sample[name] = evaluate_sequence(raft, transforms, gt, pred, device, args.batch_size)

    if not per_sample:
        raise ValueError("No video had at least two matching frames")
    metric_names = ["raft_warp_l1", "frame_diff_l1", "flow_valid_ratio"]
    average = {
        metric: float(np.mean([sample[metric] for sample in per_sample.values()]))
        for metric in metric_names
    }
    average["frame_pairs"] = int(sum(sample["frame_pairs"] for sample in per_sample.values()))

    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "metrics_temporal_raft.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "definition": {
                    "raft_warp_l1": "L1(pred_t, warp(pred_t-1, GT backward RAFT flow)) on forward-backward-consistent pixels",
                    "frame_diff_l1": "L1 temporal frame-difference error against GT",
                    "flow_max_side": args.flow_max_side,
                },
                "per_sample": per_sample,
                "average": average,
                "count": len(per_sample),
            },
            output_file,
            indent=2,
        )
    print(json.dumps(average, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
