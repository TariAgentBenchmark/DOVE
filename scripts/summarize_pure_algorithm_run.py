#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


METRICS_FILE = "metrics_psnr_ssim_lpips_dists_clipiqa.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def steady_seconds(profile: dict) -> float:
    excluded = sum(
        profile.get("stages", {}).get(name, {}).get("seconds", 0.0)
        for name in ("model_load", "model_to_device")
    )
    return profile["run_wall_seconds"] - excluded


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a completed DOVE pure-algorithm run")
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main():
    run_root = parse_args().run_root.expanduser().resolve()
    evaluation = run_root / "evaluation"
    baseline_profile = load_json(evaluation / "baseline_stage_profile.json")
    candidate_profile = load_json(evaluation / "stage_profile.json")
    baseline_metrics = load_json(evaluation / "baseline_metrics" / METRICS_FILE)["average"]
    candidate_metrics = load_json(evaluation / "candidate_metrics" / METRICS_FILE)["average"]
    baseline_temporal = load_json(
        evaluation / "baseline_metrics" / "metrics_temporal_raft.json"
    )["average"]
    candidate_temporal = load_json(
        evaluation / "candidate_metrics" / "metrics_temporal_raft.json"
    )["average"]
    gpu_info_path = evaluation / "gpu_info.txt"
    gpu_info = (
        [
            line.strip()
            for line in gpu_info_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if gpu_info_path.is_file()
        else []
    )

    baseline_steady = steady_seconds(baseline_profile)
    candidate_steady = steady_seconds(candidate_profile)
    summary = {
        "baseline_steady_seconds": baseline_steady,
        "candidate_steady_seconds": candidate_steady,
        "steady_speedup": baseline_steady / candidate_steady,
        "baseline_core_seconds": baseline_profile["profiled_inference_seconds"],
        "candidate_core_seconds": candidate_profile["profiled_inference_seconds"],
        "core_speedup": (
            baseline_profile["profiled_inference_seconds"]
            / candidate_profile["profiled_inference_seconds"]
        ),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "baseline_temporal": baseline_temporal,
        "candidate_temporal": candidate_temporal,
        "gpu_info": gpu_info,
        "passes_1_5x": baseline_steady / candidate_steady >= 1.5,
    }
    output_path = evaluation / "summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
