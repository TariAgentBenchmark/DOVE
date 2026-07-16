from contextlib import contextmanager
import time

import torch


class StageProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def measure(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.calls[name] = self.calls.get(name, 0) + 1

    def build_report(self, run_wall_seconds: float, metadata: dict) -> dict:
        stages = {}
        for name, total in sorted(self.totals.items()):
            calls = self.calls[name]
            stages[name] = {
                "seconds": total,
                "calls": calls,
                "average_seconds": total / calls,
            }

        inference_stages = ("vae_encode", "transformer", "scheduler_update", "vae_decode")
        inference_total = sum(self.totals.get(name, 0.0) for name in inference_stages)
        for name in inference_stages:
            if name in stages:
                stages[name]["inference_fraction"] = (
                    self.totals[name] / inference_total if inference_total > 0 else 0.0
                )

        return {
            "run_wall_seconds": run_wall_seconds,
            "profiled_inference_seconds": inference_total,
            "stages": stages,
            "metadata": metadata,
        }
