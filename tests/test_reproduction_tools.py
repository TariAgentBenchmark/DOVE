import json

import pytest

from scripts.check_pure_algorithm_setup import count_videos
from scripts.summarize_perceptual_sweep import summarize_variant
from scripts.summarize_pure_algorithm_run import steady_seconds


def test_steady_seconds_excludes_only_model_setup():
    profile = {
        "run_wall_seconds": 500.0,
        "stages": {
            "model_load": {"seconds": 50.0},
            "model_to_device": {"seconds": 14.09},
            "save_output": {"seconds": 20.0},
        },
    }
    assert steady_seconds(profile) == pytest.approx(435.91)


def test_count_videos_ignores_metric_json(tmp_path):
    for name in ("000.mkv", "001.mp4", "002.mov", "003.avi"):
        (tmp_path / name).touch()
    (tmp_path / "stage_profile.json").touch()
    (tmp_path / "notes.txt").touch()
    assert count_videos(tmp_path) == 4


def test_perceptual_summary_requires_every_quality_and_speed_gate(tmp_path):
    evaluation = tmp_path / "evaluation" / "conservative"
    metrics_dir = evaluation / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics_psnr_ssim_lpips_dists_clipiqa.json").write_text(
        json.dumps(
            {
                "average": {
                    "psnr": 25.8,
                    "ssim": 0.76,
                    "lpips": 0.2700,
                    "dists": 0.1510,
                    "clipiqa": 0.5020,
                }
            }
        )
    )
    (metrics_dir / "metrics_temporal_raft.json").write_text(
        json.dumps(
            {
                "average": {
                    "raft_warp_l1": 0.0104,
                    "frame_diff_l1": 0.0242,
                }
            }
        )
    )
    (evaluation / "stage_profile.json").write_text(
        json.dumps(
            {
                "run_wall_seconds": 300.0,
                "stages": {
                    "model_load": {"seconds": 5.0},
                    "model_to_device": {"seconds": 5.0},
                },
            }
        )
    )

    result = summarize_variant(tmp_path, "conservative", psnr_floor=25.5)
    assert result["passes_all"] is True
    assert result["steady_speedup"] == pytest.approx(435.91 / 290.0)
