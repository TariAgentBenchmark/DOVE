# DOVE Pure-Algorithm Perceptual Recovery Results

Date: 2026-07-29

## Configuration

- Base model: original DOVE Stage-2 model.
- Decoder: nonuniform profile `[1,1,1,2]`, continued to checkpoint 4500 with GT-targeted perceptual recovery.
- Encoder: degraded-LQ posterior distillation, profile `[1,1,1,1]`, one mid-block layer, checkpoint 3000.
- Cache: 1024 original decoder trajectories, 1024 encoder samples, and 1024 encoder-matched decoder trajectories; zero recorded failures.
- Evaluation: UDM10 fullclip, 10 videos, 4x output, bfloat16, one NVIDIA L40S.
- Excluded: `torch.compile`, TensorRT, CUDA Graphs, multi-GPU inference, reduced frame count, and reduced output resolution.

## Checkpoints

Final experiment artifacts:

```text
/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/perceptual_finish/
  perceptual-finish-finish-normalh100-v2_20260730/models/
  lpips_continue/checkpoint-4500/decoder.pt

/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/vae_encoder_distill/
  encoder_1111_m1_s3000_2828484/checkpoint-3000/encoder.pt
```

Reproduction runs created by `scripts/slurm/submit_pure_algorithm_pipeline.sh` use deterministic final paths below their `RUN_ROOT`:

```text
models/encoder/checkpoint-3000/encoder.pt
finish/models/lpips_continue/checkpoint-4500/decoder.pt
```

## Speed

Steady end-to-end time is `run_wall - model_load - model_to_device`. It includes preprocessing, model inference, output assembly, and video writing.

Official job `2975014` ran the baseline and final candidate sequentially on the same NVIDIA L40S with both models running without compilation:

| Single-L40S, no compile | Steady end-to-end | Speedup | Core inference | Core speedup |
|---|---:|---:|---:|---:|
| Original DOVE | 416.59 s | 1.000x | 403.49 s | 1.000x |
| Final lightweight VAE | **273.00 s** | **1.526x** | **248.56 s** | **1.623x** |

The job allocated `gres/gpu:l40s=1`; its log records `NVIDIA L40S, 46068 MiB, driver 590.48.01`. Both stage profiles record `compile_vae=false`. The result passes the `>=1.5x` customer gate with a 2.6 percentage-point margin.

## Quality

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE, same evaluation | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 |
| Decoder-only checkpoint 4100 | **26.6488** | **0.7748** | **0.2673** | **0.1439** | **0.5108** |
| Previous lightweight VAE | 26.0109 | 0.7675 | 0.2818 | 0.1553 | 0.4988 |
| Final perceptual checkpoint 4500 | 26.0497 | 0.7638 | **0.2700** | **0.1485** | **0.5509** |

Relative to the previous lightweight candidate, LPIPS improves by `4.19%`, DISTS by `4.38%`, and CLIPIQA by `10.45%`. Against original DOVE, LPIPS improves by `0.26%`, DISTS by `2.30%`, and CLIPIQA by `9.92%`; PSNR and SSIM are lower by `1.82%` and `0.73%`.

## Temporal Metrics

| Model | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|
| Original DOVE, same evaluation | 0.010368 | 0.024123 |
| Decoder-only checkpoint 4100 | 0.01062 | **0.02375** |
| Previous lightweight VAE | 0.01052 | 0.02433 |
| Final perceptual checkpoint 4500 | 0.010530 | 0.024242 |

The final candidate increases RAFT warp and frame-difference errors by `1.55%` and `0.50%` relative to the same-run original DOVE baseline, both below the allowed `5%`. Four videos at frames 5, 15, and 25 were visually audited; no new flicker, structural jump, material color shift, or sharpening halo was found.

## Perceptual Recovery Sequence

The accepted continuation is deterministic:

1. `perceptual`, checkpoint 4300: 200 steps from decoder checkpoint 4100.
2. `lpips_focus`, checkpoint 4400: 100 steps from the selected checkpoint 4300.
3. `lpips_continue`, checkpoint 4500: 100 steps from the selected checkpoint 4400.

The final objective uses teacher L1 `0.15`, teacher MSE `0.03`, GT L1 `0.08`, frame-difference `0.10`, DISTS `0.15`, LPIPS `0.25`, and CLIPIQA hinge `0.01`, with a learning rate of `1e-5`.

## Decision

Quality and temporal gates pass:

- Single-L40S no-compile speedup `1.526x >= 1.5x`.
- PSNR `26.0497 >= 25.5`.
- LPIPS `0.2700 <= 0.2707`.
- DISTS `0.1485 <= 0.1520`.
- CLIPIQA `0.5509 >= 0.5012`.
- RAFT warp and frame-difference degradation remain below `5%`.

The customer gate is fully satisfied. The supported statement is: **on one L40S, with both models running without `torch.compile`, the final lightweight VAE reaches 1.526x steady end-to-end acceleration while matching or improving original DOVE on LPIPS, DISTS, and CLIPIQA; PSNR/SSIM decrease modestly and temporal errors remain within 5%.**

The full sequence of rejected and intermediate experiments is retained in [iteration_history.md](iteration_history.md).
