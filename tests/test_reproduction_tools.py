import pytest

from scripts.check_pure_algorithm_setup import count_videos
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
