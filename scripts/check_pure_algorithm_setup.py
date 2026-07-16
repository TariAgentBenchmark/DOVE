#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def existing_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DOVE pure-algorithm reproduction inputs")
    parser.add_argument("--model_path", type=existing_dir, required=True)
    parser.add_argument("--hqvsr_root", type=existing_dir, required=True)
    parser.add_argument("--hqvsr_list", type=existing_file, required=True)
    parser.add_argument("--udm10_lq", type=existing_dir, required=True)
    parser.add_argument("--udm10_gt", type=existing_dir, required=True)
    return parser.parse_args()


def count_videos(root: Path) -> int:
    suffixes = {".mp4", ".mkv", ".mov", ".avi"}
    return sum(path.suffix.lower() in suffixes for path in root.iterdir() if path.is_file())


def main():
    args = parse_args()
    required_model_parts = ["vae", "transformer", "scheduler", "tokenizer", "text_encoder"]
    missing_model_parts = [name for name in required_model_parts if not (args.model_path / name).exists()]
    with args.hqvsr_list.open(encoding="utf-8") as handle:
        listed_samples = sum(bool(line.strip()) for line in handle)

    report = {
        "model_path": str(args.model_path),
        "missing_model_parts": missing_model_parts,
        "hqvsr_root": str(args.hqvsr_root),
        "hqvsr_list_entries": listed_samples,
        "udm10_lq_videos": count_videos(args.udm10_lq),
        "udm10_gt_videos": count_videos(args.udm10_gt),
    }
    print(json.dumps(report, indent=2))

    errors = []
    if missing_model_parts:
        errors.append(f"model is missing: {', '.join(missing_model_parts)}")
    if listed_samples < 1024:
        errors.append(f"HQ-VSR list has only {listed_samples} entries; 1024 are required")
    if report["udm10_lq_videos"] != 10 or report["udm10_gt_videos"] != 10:
        errors.append("UDM10 must contain 10 LQ videos and 10 GT videos")
    if errors:
        raise SystemExit("Setup validation failed: " + "; ".join(errors))

    print("Setup validation passed.")


if __name__ == "__main__":
    main()
