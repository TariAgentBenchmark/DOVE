# DOVE Pure-Algorithm Final Results

Date: 2026-07-14

## Configuration

- Base model: original DOVE Stage-2 model.
- Decoder: trajectory-distilled nonuniform profile `[1,1,1,2]`, checkpoint 4100.
- Encoder: degraded-LQ posterior distillation, profile `[1,1,1,1]`, one mid-block layer, checkpoint 3000.
- Cache: 1024 decoder trajectory samples and 1024 encoder samples, zero recorded failures.
- Evaluation: UDM10 fullclip, 10 videos, 4x output, bfloat16, one NVIDIA L40S.
- Excluded: `torch.compile`, TensorRT, CUDA Graphs, multi-GPU inference, reduced frame count, and reduced output resolution.

## Checkpoints

Original experiment artifacts:

```text
/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/vae_decoder_distill/
  decoder_1112_s4100_2812937/checkpoint-4100/decoder.pt

/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/vae_encoder_distill/
  encoder_1111_m1_s3000_2828484/checkpoint-3000/encoder.pt
```

Reproduction runs created by `scripts/slurm/submit_pure_algorithm_pipeline.sh` use deterministic paths below their `RUN_ROOT`:

```text
models/decoder_final/checkpoint-4100/decoder.pt
models/encoder/checkpoint-3000/encoder.pt
```

## Speed

Steady end-to-end time is `run_wall - model_load - model_to_device`. It includes preprocessing, model inference, output assembly, and video writing.

| Single-L40S, no compile | Steady end-to-end | Speedup | Core inference | Core speedup |
|---|---:|---:|---:|---:|
| Original DOVE | 435.91 s | 1.000x | 404.57 s | 1.000x |
| Lightweight encoder + decoder | **289.55 s** | **1.505x** | **248.82 s** | **1.626x** |

The result clears the 1.5x threshold by about 1.06 seconds. It must be rechecked on the deployment machine before being used as an SLA.

## Quality

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5013 |
| Decoder-only checkpoint 4100 | **26.6488** | **0.7748** | **0.2673** | **0.1439** | **0.5108** |
| Lightweight encoder + decoder | 26.0109 | 0.7675 | 0.2818 | 0.1553 | 0.4988 |

Relative to original DOVE, the final pure-algorithm model changes PSNR by `-1.97%`, SSIM by `-0.25%`, LPIPS by `+4.10%`, DISTS by `+2.17%`, and CLIPIQA by `-0.50%`.

## Temporal Metrics

| Model | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|
| Original DOVE | 0.01037 | 0.02412 |
| Decoder-only checkpoint 4100 | 0.01062 | **0.02375** |
| Lightweight encoder + decoder | 0.01052 | 0.02433 |

The pure-algorithm model increases RAFT warp and frame-difference errors by `1.48%` and `0.85%`. Three-frame visual audits did not show obvious flicker or structural jumps.

## Decision

The final model passes the declared acceptance thresholds:

- Single-L40S no-compile steady speedup: `1.505x`.
- PSNR/SSIM degradation: below the allowed `8%`.
- CLIPIQA: `0.4988`, above the approximate `0.45` target.
- Temporal metrics: small regressions without obvious sampled instability.

The supported statement is: **the pure-algorithm route reaches 1.5x with a small quality tradeoff**. It is not VAE-free and is not a lossless optimization.

The full sequence of rejected and intermediate experiments is retained in [iteration_history.md](iteration_history.md).
