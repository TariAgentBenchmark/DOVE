#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

try:
    from scripts.summarize_pure_algorithm_run import steady_seconds
except ModuleNotFoundError:
    from summarize_pure_algorithm_run import steady_seconds


METRICS_FILE = "metrics_psnr_ssim_lpips_dists_clipiqa.json"
DEFAULT_VARIANTS = ("conservative", "perceptual", "clip_margin")
BASELINE = {
    "steady_seconds": 435.91,
    "psnr": 26.5338,
    "ssim": 0.7694,
    "lpips": 0.2707,
    "dists": 0.1520,
    "clipiqa": 0.5013,
    "raft_warp_l1": 0.01037,
    "frame_diff_l1": 0.02412,
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_variant(root: Path, variant: str, psnr_floor: float) -> dict:
    evaluation = root / "evaluation" / variant
    metrics = load_json(evaluation / "metrics" / METRICS_FILE)["average"]
    temporal = load_json(evaluation / "metrics" / "metrics_temporal_raft.json")["average"]
    profile = load_json(evaluation / "stage_profile.json")
    candidate_steady = steady_seconds(profile)
    checks = {
        "lpips_not_worse_than_dove": metrics["lpips"] <= BASELINE["lpips"],
        "dists_not_worse_than_dove": metrics["dists"] <= BASELINE["dists"],
        "clipiqa_not_worse_than_dove": metrics["clipiqa"] >= BASELINE["clipiqa"],
        "psnr_above_floor": metrics["psnr"] >= psnr_floor,
        "raft_within_5_percent": (
            temporal["raft_warp_l1"] <= BASELINE["raft_warp_l1"] * 1.05
        ),
        "frame_diff_within_5_percent": (
            temporal["frame_diff_l1"] <= BASELINE["frame_diff_l1"] * 1.05
        ),
        "speedup_at_least_1_5x": BASELINE["steady_seconds"] / candidate_steady >= 1.5,
    }
    return {
        "metrics": metrics,
        "temporal": temporal,
        "steady_seconds": candidate_steady,
        "steady_speedup": BASELINE["steady_seconds"] / candidate_steady,
        "checks": checks,
        "passes_all": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize the perceptual decoder sweep")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--psnr_floor", type=float, default=25.5)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--output_name", default="perceptual_summary.json")
    args = parser.parse_args()

    summary = {
        "baseline": BASELINE,
        "psnr_floor": args.psnr_floor,
        "variants": {
            variant: summarize_variant(args.run_root, variant, args.psnr_floor)
            for variant in args.variants
        },
    }
    output_path = args.run_root / "evaluation" / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
