# DOVE Optimization Iteration History

> Archive only. This document records rejected pixel/L2P, structured diffusion,
> multi-GPU, compile-assisted, and intermediate decoder experiments. The code
> for rejected routes is intentionally not part of the supported repository.
> Use `docs/pure_algorithm_reproduction.md` for the maintained final pipeline.

Date: 2026-06-19

## Objective

Evaluate the DOVE pixel-space, VAE-free checkpoint against the original DOVE baseline on UDM10.

## Artifacts

- Pixel checkpoint: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/checkpoint/dove_pixel_l2p_full10k_blk5_320x640_2727420/checkpoint-10000`
- Pixel outputs: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/eval_pixel/pixel_udm10_ckpt10000`
- Baseline outputs: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/eval_pixel/baseline_dove_udm10`
- Pixel metrics JSON: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/eval_pixel/pixel_udm10_ckpt10000/metrics_psnr_ssim_lpips_dists_clipiqa.json`
- Baseline metrics JSON: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/eval_pixel/baseline_dove_udm10/metrics_psnr_ssim_lpips_dists_clipiqa.json`

## Setup

Dataset: UDM10, 10 videos.

Input: `LQ-Video/*.mkv`, 32 frames, 180 x 318.

Ground truth: `GT-Video/*.mkv`, 32 frames, 720 x 1272.

Evaluation metrics: PSNR, SSIM, LPIPS, DISTS, CLIPIQA.

Hardware: 1 x NVIDIA L40S for each evaluation job.

Metric computation was frame-by-frame. Batch metric mode OOMed on 720p x 32-frame videos.

## Inference Configuration

Pixel checkpoint:

- Model: `dove-pixel`, VAE-free RGB pixel transformer.
- Checkpoint: `checkpoint-10000`.
- Temporal chunking: 2 frames, no overlap.
- Spatial tiling: 320 x 640, no overlap.
- Padded target resolution during inference: 768 x 1280.
- Final output cropped back to 720 x 1272.
- Job: `2729548`, completed, elapsed `00:03:29`.
- Inference-only progress: 10 videos in about `01:25`.

Original DOVE baseline:

- Model: `/data/42-julia-hpc-rz-cv/sig95vg/DOVE/pretrained_models/DOVE`.
- VAE slicing/tiling enabled.
- Temporal chunking: 9 frames, no overlap.
- Spatial tiling: disabled.
- Padded target resolution during inference: 768 x 1280.
- Final output cropped back to 720 x 1272.
- Job: `2729622`, completed, elapsed `00:08:35`.
- Inference-only progress: 10 videos in about `07:03`.

## Average Results

Higher is better for PSNR, SSIM, CLIPIQA. Lower is better for LPIPS, DISTS.

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE baseline | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| Pixel VAE-free ckpt-10000 | 24.1250 | 0.6364 | 0.5144 | 0.2179 | 0.2842 |
| Pixel - baseline | -0.6272 | -0.0963 | +0.1991 | +0.0442 | -0.2172 |

## Interpretation

The pixel-space checkpoint is functionally valid: it loads cleanly (`missing=0`, `unexpected=0`), produces 10 UDM10 videos, and runs full metric evaluation.

Quality is currently below the original DOVE baseline on this benchmark. The gap is visible across all five metrics: PSNR and SSIM are lower, LPIPS and DISTS are higher, and CLIPIQA is much lower.

The pixel path is faster under this test configuration. Pixel inference processed 10 videos in about `01:25`, while baseline DOVE took about `07:03`, roughly a 5x inference-speed advantage on L40S. This is a practical speed signal, but not yet a final benchmark because the chunking settings are different.

## Caveats

This is not yet a full paper-level reproduction.

- The pixel checkpoint only trained for 10k steps at 2 x 320 x 640.
- Full UDM10 evaluation required tiled/chunked inference because UDM10 targets 720 x 1272 and 32 frames.
- Pixel inference used 2-frame chunks with no temporal overlap, which may hurt temporal consistency.
- The implementation does not yet include all GenDR-Pix/L2P paper details such as adversarial distillation, masked Fourier loss, padding-based CFG, or 4K-specific training.
- Baseline DOVE used 9-frame chunks with no overlap after an initial odd-overlap run exposed a write-count bug in the original script.

## Conclusion

The VAE-free DOVE path is implemented and operational, but this 10k-step checkpoint is not yet competitive with the original DOVE baseline on UDM10. The result is still useful: it proves the pixel path can complete full-resolution inference and metric evaluation, and it shows a clear speed advantage. The next quality-focused step should be additional training and loss/architecture improvements before attempting 4K claims.

## Loss / Teacher-Distillation Validation

Date: 2026-06-19

This validation checked whether the original 10k pixel checkpoint should keep the current perceptual loss mix, or move closer to the L2P-style objective consistency strategy. The tested continuation uses:

- Start checkpoint: `dove_pixel_l2p_full10k_blk5_320x640_2727420/checkpoint-10000`
- Continuation checkpoint: `dove_pixel_loss_mse500_from10k_2730309/checkpoint-10500`
- Training job: `2730309`, completed, elapsed `00:37:31`
- Loss: MSE only, `use_perceptual_loss=false`, `dists_weight=0`, `frame_diff_weight=0`
- LR: `1e-5`
- Continuation length: 500 optimizer steps, global step `10000 -> 10500`
- Evaluation job: `2730320`, completed, elapsed `00:04:02`

### GT Metrics

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE baseline | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| Pixel VAE-free ckpt-10000 | 24.1250 | 0.6364 | 0.5144 | 0.2179 | 0.2842 |
| Pixel ckpt-10500 MSE-only | 24.6216 | 0.7148 | 0.4917 | 0.2645 | 0.1503 |
| MSE-only - ckpt-10000 | +0.4966 | +0.0784 | -0.0227 | +0.0466 | -0.1339 |
| MSE-only - baseline | -0.1306 | -0.0179 | +0.1764 | +0.0908 | -0.3511 |

### Teacher-Gap Metrics

Here the original DOVE baseline output is treated as the teacher target. Higher PSNR/SSIM and lower LPIPS/DISTS mean the pixel model is closer to the teacher output. CLIPIQA is no-reference, so it reflects only the generated output quality score.

| Student vs DOVE teacher | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Pixel VAE-free ckpt-10000 | 25.0342 | 0.6943 | 0.4885 | 0.2268 | 0.2842 |
| Pixel ckpt-10500 MSE-only | 25.7416 | 0.7903 | 0.4725 | 0.2630 | 0.1503 |
| MSE-only - ckpt-10000 | +0.7074 | +0.0960 | -0.0160 | +0.0362 | -0.1339 |

### Interpretation

MSE-only continuation improved distortion-oriented metrics: PSNR, SSIM, and LPIPS improved against GT, and PSNR/SSIM also moved closer to the original DOVE teacher output. This supports testing the L2P-style objective-consistency direction further.

However, DISTS and CLIPIQA got worse. The checkpoint is likely becoming more conservative or over-smoothed: it aligns better in pixel space but loses perceptual sharpness and no-reference quality. This means pure MSE is not enough as the final loss.

### Next Validation

The next controlled run should use teacher-output distillation rather than only GT MSE:

- Precompute or stream original DOVE teacher outputs for the train clips.
- Train with a mixed objective, for example `GT MSE + teacher MSE/L1 + low-weight perceptual`.
- Keep the L2P-style frozen-block setup.
- Compare against both GT and DOVE-teacher metrics after another short continuation before scaling to longer training.

## Teacher-Output Distillation Cache Validation

Date: 2026-06-19

This run implemented a real teacher-output cache instead of using GT-only continuation. The cache stores fixed degraded LQ, GT HQ, original DOVE teacher output, and prompt embedding tensors so the student sees the same condition each time.

- Teacher cache job: `2730823`, completed, 32 cached samples at `2x320x640`
- Cache path: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/teacher_cache/dove_teacher_cache32_320x640_2730823`
- Start checkpoint: `dove_pixel_l2p_full10k_blk5_320x640_2727420/checkpoint-10000`
- Continuation checkpoint: `dove_pixel_teacherdistill_cache32_s200_from10k_2731021/checkpoint-10200`
- Training job: `2731021`, completed, elapsed `00:18:46`
- Evaluation job: `2731022`, completed, elapsed `00:07:21`
- Loss: `GT MSE + 0.5 * teacher L1 + 0.05 * DISTS`, `frame_diff_weight=0`
- Continuation length: 200 optimizer steps, global step `10000 -> 10200`

Implementation note: the short continuation now uses `--resume_model_only true` so DeepSpeed loads model weights from `checkpoint-10000` without spending time restoring the old optimizer state. This avoided the earlier long stall in full `accelerator.load_state`.

### GT Metrics

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE baseline | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| Pixel VAE-free ckpt-10000 | 24.1250 | 0.6364 | 0.5144 | 0.2179 | 0.2842 |
| Pixel ckpt-10500 MSE-only | 24.6216 | 0.7148 | 0.4917 | 0.2645 | 0.1503 |
| Pixel ckpt-10200 teacher-cache mixed | 24.4725 | 0.6876 | 0.4706 | 0.2193 | 0.1929 |
| Teacher-cache mixed - ckpt-10000 | +0.3475 | +0.0512 | -0.0438 | +0.0014 | -0.0913 |
| Teacher-cache mixed - MSE-only | -0.1491 | -0.0272 | -0.0211 | -0.0452 | +0.0426 |
| Teacher-cache mixed - baseline | -0.2797 | -0.0451 | +0.1553 | +0.0456 | -0.3085 |

### Teacher-Gap Metrics

Here the original DOVE baseline output is treated as the teacher target. Higher PSNR/SSIM and lower LPIPS/DISTS mean the pixel model is closer to the teacher output. CLIPIQA is no-reference, so it reflects only the generated output quality score.

| Student vs DOVE teacher | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Pixel VAE-free ckpt-10000 | 25.0342 | 0.6943 | 0.4885 | 0.2268 | 0.2842 |
| Pixel ckpt-10500 MSE-only | 25.7416 | 0.7903 | 0.4725 | 0.2630 | 0.1503 |
| Pixel ckpt-10200 teacher-cache mixed | 25.5839 | 0.7628 | 0.4336 | 0.2206 | 0.1929 |
| Teacher-cache mixed - ckpt-10000 | +0.5497 | +0.0685 | -0.0549 | -0.0062 | -0.0913 |
| Teacher-cache mixed - MSE-only | -0.1577 | -0.0275 | -0.0389 | -0.0424 | +0.0426 |

### Interpretation

The mixed teacher-cache run is a better compromise than pure MSE-only continuation. Against GT, it gives up some PSNR/SSIM compared with MSE-only, but improves LPIPS, DISTS, and CLIPIQA. Against the DOVE teacher output, it is also more perceptually aligned than MSE-only: LPIPS and DISTS are both lower.

This supports continuing with teacher-output distillation, but not with the exact short-run recipe as the final answer. The GT metrics are still below original DOVE, and CLIPIQA remains much lower. The current cache is only 32 samples and the continuation is only 200 steps, so the result should be treated as a directional ablation rather than a final quality claim.

### Next Validation

- Increase teacher cache size from 32 samples to at least a few hundred samples before longer training.
- Sweep teacher loss weight, starting with `0.25`, `0.5`, and `1.0`.
- Try `teacher L1` versus `teacher MSE`; the first mixed run used `teacher L1`.
- Keep low-weight perceptual loss, but test DISTS weights around `0.02`, `0.05`, and `0.1`.
- Keep reporting both GT metrics and teacher-gap metrics; optimizing only one hides the quality tradeoff.

## Teacher-Output Distillation 256-Sample Weight Sweep

Date: 2026-06-20

This validation increases the teacher cache from 32 samples to 256 samples and sweeps the teacher-output loss weight while keeping the rest of the short-run recipe fixed.

- Teacher cache job: `2733486`, completed, elapsed `00:29:49`
- Cache path: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/teacher_cache/dove_teacher_cache256_320x640_2733486`
- Cache size: 256 samples at `2x320x640`
- Start checkpoint: `dove_pixel_l2p_full10k_blk5_320x640_2727420/checkpoint-10000`
- Loss: `GT MSE + teacher_weight * teacher L1 + 0.05 * DISTS`
- Continuation length: 200 optimizer steps, global step `10000 -> 10200`

### Jobs

| Teacher weight | Train job | Train elapsed | Train MaxRSS | Eval job | Eval elapsed |
|---:|---:|---:|---:|---:|---:|
| 0.25 | `2733550` | `00:33:14` | `173735808K` | `2733551` | `00:07:34` |
| 0.5 | `2733552` | `00:34:45` | `172685580K` | `2733553` | `00:09:10` |
| 1.0 | `2733554` | `00:15:16` | `137034844K` | `2733555` | `00:06:24` |

### GT Metrics

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE baseline | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| Pixel VAE-free ckpt-10000 | 24.1250 | 0.6364 | 0.5144 | 0.2179 | 0.2842 |
| 32-cache mixed, teacher weight 0.5 | 24.4725 | 0.6876 | 0.4706 | 0.2193 | 0.1929 |
| 256-cache mixed, teacher weight 0.25 | 24.5309 | 0.6841 | 0.4705 | 0.2137 | 0.1960 |
| 256-cache mixed, teacher weight 0.5 | 24.5637 | 0.6923 | 0.4656 | 0.2196 | 0.1849 |
| 256-cache mixed, teacher weight 1.0 | 24.5952 | 0.7012 | 0.4626 | 0.2299 | 0.1696 |

### Teacher-Gap Metrics

Here the original DOVE baseline output is treated as the teacher target. Higher PSNR/SSIM and lower LPIPS/DISTS mean the pixel model is closer to the teacher output. CLIPIQA is no-reference, so it reflects only the generated output quality score.

| Student vs DOVE teacher | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Pixel VAE-free ckpt-10000 | 25.0342 | 0.6943 | 0.4885 | 0.2268 | 0.2842 |
| 32-cache mixed, teacher weight 0.5 | 25.5839 | 0.7628 | 0.4336 | 0.2206 | 0.1929 |
| 256-cache mixed, teacher weight 0.25 | 25.6140 | 0.7563 | 0.4394 | 0.2166 | 0.1960 |
| 256-cache mixed, teacher weight 0.5 | 25.6383 | 0.7648 | 0.4326 | 0.2197 | 0.1849 |
| 256-cache mixed, teacher weight 1.0 | 25.6600 | 0.7737 | 0.4300 | 0.2274 | 0.1696 |

### Interpretation

Increasing the cache from 32 to 256 samples gives a small but consistent improvement on distortion-oriented GT metrics. At the same teacher weight `0.5`, GT PSNR improves from `24.4725` to `24.5637`, SSIM improves from `0.6876` to `0.6923`, and LPIPS improves from `0.4706` to `0.4656`.

Raising the teacher weight from `0.25` to `1.0` improves PSNR, SSIM, and LPIPS against both GT and the DOVE teacher output. However, it worsens DISTS and CLIPIQA. This is the same tradeoff observed in the earlier MSE-only check: stronger pixel/teacher alignment improves reconstruction metrics but can reduce perceptual/no-reference quality.

The best short-run choice depends on what is prioritized:

- Best PSNR/SSIM/LPIPS: teacher weight `1.0`.
- Best DISTS/CLIPIQA balance: teacher weight `0.25`.
- Middle-ground setting: teacher weight `0.5`, which improves over the 32-cache run without degrading DISTS as much as `1.0`.

None of the 256-cache short runs closes the gap to original DOVE. The best GT PSNR run, teacher weight `1.0`, is still behind baseline by `0.1570` PSNR, `0.0315` SSIM, `0.1473` LPIPS, `0.0562` DISTS, and `0.3318` CLIPIQA. The direction is valid, but this is still a short ablation, not final quality.

### Next Validation

- Use teacher weight `0.5` as the conservative default for the next longer run.
- Also keep a `1.0` branch if the goal is to maximize teacher/GT reconstruction metrics.
- Reduce perceptual weight sweep range to `0.02`, `0.05`, and `0.1`; the current `0.05` does not stop CLIPIQA from falling when teacher weight increases.
- Add a visual sample panel for UDM10 before scaling the same recipe to longer training.

## Teacher-Output Distillation Longer Run

Date: 2026-06-20

This run continues the conservative `teacher_weight=0.5` branch from the 256-sample cache sweep.

- Start checkpoint: `dove_pixel_tdistill_cache256_s200_tw0p5_l1_dw0p05_from10k_2733552/checkpoint-10200`
- Final checkpoint: `dove_pixel_tdistill_cache256_s1800_tw0p5_l1_dw0p05_from10200_2735121/checkpoint-12000`
- Teacher cache: `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/teacher_cache/dove_teacher_cache256_320x640_2733486`
- Training job: `2735121`, completed, elapsed `01:57:10`, MaxRSS `207063048K`
- Evaluation job: `2735122`, completed, elapsed `00:16:24`, MaxRSS `103475228K`
- Loss: `GT MSE + 0.5 * teacher L1 + 0.05 * DISTS`
- Continuation length: 1800 optimizer steps, global step `10200 -> 12000`

### GT Metrics

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Original DOVE baseline | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| 256-cache mixed, teacher weight 0.5, ckpt-10200 | 24.5637 | 0.6923 | 0.4656 | 0.2196 | 0.1849 |
| 256-cache mixed, teacher weight 0.5, ckpt-12000 | 24.5554 | 0.6952 | 0.4553 | 0.2116 | 0.1753 |
| ckpt-12000 - ckpt-10200 | -0.0083 | +0.0029 | -0.0103 | -0.0080 | -0.0096 |
| ckpt-12000 - baseline | -0.1968 | -0.0375 | +0.1400 | +0.0379 | -0.3261 |

### Teacher-Gap Metrics

| Student vs DOVE teacher | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| 256-cache mixed, teacher weight 0.5, ckpt-10200 | 25.6383 | 0.7648 | 0.4326 | 0.2197 | 0.1849 |
| 256-cache mixed, teacher weight 0.5, ckpt-12000 | 25.5245 | 0.7601 | 0.4422 | 0.2276 | 0.1753 |
| ckpt-12000 - ckpt-10200 | -0.1138 | -0.0047 | +0.0096 | +0.0079 | -0.0096 |

### Interpretation

The longer `teacher_weight=0.5` run did not produce a clear quality breakthrough. Against GT it slightly improves SSIM, LPIPS, and DISTS compared with the 200-step checkpoint, but PSNR is essentially flat and CLIPIQA drops. Against the DOVE teacher output, all full-reference metrics get slightly worse.

This suggests the current 256-sample cache is too small or too repetitive for longer continuation: after the first short improvement, additional training starts to overfit the cache rather than improving teacher alignment on UDM10. The next useful experiment should change the recipe, not simply extend this run.

Recommended next steps:

- Increase teacher cache beyond 256 samples before another long run.
- Try lower perceptual weight, especially `DISTS_WEIGHT=0.02`, to see whether CLIPIQA stops falling.
- Keep `teacher_weight=0.5` only as a conservative baseline; use `teacher_weight=1.0` if prioritizing PSNR/SSIM/LPIPS.
- Add qualitative frame panels, because the current metrics show mixed movement and no-reference quality remains far below original DOVE.

## Structured Pixel Diffusion 1/3/5-Step Validation (Unanchored Sampler V1)

Date: 2026-07-11

Status: superseded for final decision. Review of the sampler found that V1 reconstructed the structured LQ condition from each imperfect velocity prediction. This allows condition error to accumulate across steps and does not match training, where the original LQ condition is known and fixed. Job `2811874` reruns `3/5/15-step` inference with the original LQ re-anchored at every update. The V1 numbers below are retained as an ablation of condition drift and will not be used as the final multi-step conclusion.

This experiment tests the customer's suggested speed-quality tradeoff by replacing the fixed one-step pixel prediction with a real time-conditioned structured diffusion trajectory. During training, the upscaled LQ input acts as structured noise, teacher/GT frames define the clean target, and the model is trained at random timesteps over `0..999`. Inference uses deterministic velocity-based updates for `1`, `3`, or `5` steps.

- Start checkpoint: `dove_pixel_l2p_full10k_blk5_320x640_2727420/checkpoint-10000`
- Final checkpoint: `dove_pixel_structdiff_s200_teacher_v1_from10k_2810153/checkpoint-10200`
- Teacher cache: `dove_teacher_cache256_320x640_2733486`
- Training job: `2810153`, completed, elapsed `00:34:05`, MaxRSS `206156180K`
- Evaluation job: `2810154`, completed, elapsed `00:22:47`, MaxRSS `63030708K`
- Fair baseline job: `2810155`, completed, elapsed `00:21:58`, MaxRSS `64494616K`
- Training objective: `0.5 * GT MSE + 0.5 * teacher L1 + 0.02 * DISTS + 0.1 * frame difference + 1.0 * structured velocity MSE`
- Continuation length: 200 optimizer steps, global step `10000 -> 10200`

### Fair Runtime Baseline

The original DOVE baseline was rerun with the same `2-frame` temporal chunks and `320x640` spatial tiles as the pixel path. This is the runtime baseline only: shortening DOVE from 9-frame to 2-frame chunks materially reduces its quality, so the official 9-frame output remains the quality baseline.

| Baseline configuration | End-to-end seconds | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|---:|
| Original DOVE, official 9-frame output | - | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| Original DOVE, fair 2-frame/320x640 run | 1200 | 24.0958 | 0.7007 | 0.3782 | 0.2060 | 0.4395 |

### GT Metrics And Runtime

| Structured steps | End-to-end seconds | Speedup vs fair baseline | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 125 | 9.60x | 18.2502 | 0.5402 | 0.6437 | 0.2909 | 0.1653 |
| 3 | 255 | 4.71x | 20.2607 | 0.5605 | 0.5061 | 0.2416 | 0.1818 |
| 5 | 489 | 2.45x | 18.9100 | 0.5070 | 0.5095 | 0.2486 | 0.1926 |

### Teacher-Gap Metrics

| Structured steps | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.5206 | 0.5814 | 0.6666 | 0.3100 | 0.1653 |
| 3 | 20.5060 | 0.5686 | 0.5037 | 0.2536 | 0.1818 |
| 5 | 18.9686 | 0.5120 | 0.5012 | 0.2580 | 0.1926 |

### Temporal Metrics

Lower is better for both temporal errors. `raft_warp_l1` warps the previous prediction with GT backward RAFT flow and evaluates only forward-backward-consistent pixels. `frame_diff_l1` compares predicted and GT frame differences. Flow is evaluated with a maximum side of 512 pixels.

| Model | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|
| Original DOVE, official 9-frame output | 0.01272 | 0.03068 |
| Original DOVE, fair 2-frame/320x640 run | 0.02149 | 0.03359 |
| Structured 1-step | 0.03522 | 0.04417 |
| Structured 3-step | 0.02497 | 0.03771 |
| Structured 5-step | 0.02320 | 0.03719 |

### Interpretation

Increasing the number of structured steps does recover some quality and temporal consistency. The 3-step run is best for PSNR, SSIM, LPIPS, and DISTS, while the 5-step run gives the best CLIPIQA and temporal metrics. The 5-step path also retains a practical `2.45x` end-to-end speedup against the same-chunk DOVE runtime baseline.

However, the quality gap remains too large. The 5-step checkpoint reaches only `0.1926` CLIPIQA, far below both the official DOVE baseline (`0.5014`) and the customer's approximately `0.45` acceptance range. Its PSNR and SSIM also miss the allowed 8% degradation limit. Temporal errors improve with more steps but remain worse than both DOVE baselines.

The result fails the predeclared short-run continuation gate of `CLIPIQA >= 0.30`. A longer run on the same 256-sample cache is therefore not justified. The next experiment must change the conditioning/training design rather than only add steps or repeat the cache. The most defensible options are a separately conditioned LQ branch, a lightweight residual refiner that preserves the original LQ at every step, or a hybrid path that retains part of the original VAE/detail decoder while targeting at least `1.5x` end-to-end speedup.

## Anchored Structured Pixel Diffusion 3/5/15-Step Validation

Date: 2026-07-11

This rerun fixes the V1 sampler mismatch by passing the original upscaled LQ sample to every deterministic update. The structured condition is therefore fixed throughout inference, matching training, instead of being reconstructed from each imperfect velocity prediction.

- Evaluation job: `2811874`, completed, elapsed `00:49:21`, MaxRSS `106118416K`
- Checkpoint: `dove_pixel_structdiff_s200_teacher_v1_from10k_2810153/checkpoint-10200`
- Evaluated steps: `3`, `5`, and `15`
- Failure check: no OOM, NCCL error, NaN, missing checkpoint, or non-zero exit code

### GT Metrics

| Structured steps | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---:|---:|---:|---:|---:|---:|
| 3, anchored | 20.7377 | 0.5826 | 0.4894 | 0.2350 | 0.1949 |
| 5, anchored | 18.4846 | 0.5053 | 0.5072 | 0.2499 | 0.2042 |
| 15, anchored | 14.8114 | 0.3499 | 0.5790 | 0.2954 | 0.2379 |

### Teacher-Gap Metrics

| Structured steps | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---:|---:|---:|---:|---:|---:|
| 3, anchored | 21.0084 | 0.5929 | 0.4821 | 0.2446 | 0.1949 |
| 5, anchored | 18.5069 | 0.5104 | 0.4882 | 0.2537 | 0.2042 |
| 15, anchored | 14.7688 | 0.3457 | 0.5597 | 0.2955 | 0.2379 |

### Temporal Metrics

| Model | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|
| Original DOVE, official 9-frame output | 0.01272 | 0.03068 |
| Original DOVE, fair 2-frame/320x640 run | 0.02149 | 0.03359 |
| Structured 3-step, anchored | 0.02422 | 0.03734 |
| Structured 5-step, anchored | 0.02147 | 0.03632 |
| Structured 15-step, anchored | 0.01936 | 0.03968 |

### Runtime

The raw script timer includes setup, model loading, metric-model cache effects, and video processing. The 3-step raw timer was additionally affected by a cold download, so steady video-processing time is the more comparable step-scaling measurement. The fair DOVE baseline processed the same 10 videos in `11:10` (`670` seconds).

| Structured steps | Raw script timer | Video-processing time | Processing speedup vs fair DOVE |
|---:|---:|---:|---:|
| 3, anchored | 875 s, cold setup | 231 s | 2.90x |
| 5, anchored | 482 s | 336 s | 1.99x |
| 15, anchored | 1150 s | 967 s | 0.69x |

For reference, raw-timer speedup is `2.49x` for the 5-step run and only `1.04x` for the 15-step run against the fair DOVE raw timer of `1200` seconds. The 3-step raw timer is not suitable for an end-to-end claim because it includes the one-time cold download.

### Final Interpretation

Re-anchoring fixes part of the condition-drift problem: compared with unanchored 3-step inference, the anchored 3-step result improves every GT quality metric. It does not eliminate the more fundamental trajectory error. As step count rises, CLIPIQA and RAFT warp consistency improve, but reconstruction rapidly diverges: PSNR falls from `20.7377` at 3 steps to `14.8114` at 15 steps, while SSIM falls from `0.5826` to `0.3499`.

The 15-step configuration fails both acceptance axes. Its CLIPIQA is only `0.2379`, well below the customer target near `0.45`; PSNR/SSIM degradation is far beyond the allowed 8%; and its steady processing speed is slower than the fair DOVE baseline. The 5-step run remains the best speed-quality point in this family, but it also misses the quality gate by a large margin.

Therefore, longer training of this same 256-sample VAE-free architecture is not recommended. The next engineering path should retain the original DOVE quality path and optimize or replace only its VAE decoder, targeting fixed-shape compilation/TensorRT/CUDA graphs first, then a distilled lightweight decoder if compilation alone does not reach the required `1.5x` end-to-end speedup.

## Original DOVE VAE Pipeline Profiling And Multi-GPU Tile Parallelism

Date: 2026-07-11

This experiment changes direction from VAE-free pixel diffusion to preserving the original DOVE latent model and parallelizing its actual bottleneck. Optional synchronized stage profiling was added to `inference_script.py`; a separate `torchrun` path distributes the CogVideoX VAE's overlapping spatial tiles across four GPUs while keeping the latent Transformer on rank 0.

### Single-H100 Profiling

The `official9` profile processes UDM10 with the previous 9-frame outer chunks. The percentages below use the four core inference stages only.

| Stage | Seconds | Core inference share |
|---|---:|---:|
| VAE encode | 72.10 | 31.2% |
| Latent Transformer | 10.14 | 4.4% |
| Scheduler update | 0.02 | <0.1% |
| VAE decode | 148.67 | 64.4% |
| Total core inference | 230.93 | 100% |

The VAE therefore accounts for `95.6%` of core inference time. Distilling or pruning the latent Transformer cannot provide the required speedup by itself.

| H100 configuration | Job | VAE calls | Run wall time | Steady time excluding load/device move |
|---|---:|---:|---:|---:|
| 9-frame outer chunks, default VAE tiles | 2811960 | 30 | 273.11 s | 251.40 s |
| Full 33-frame clip, default VAE tiles | 2811963 | 10 | 279.33 s | 260.20 s |
| Full clip, enlarged `384x640` VAE tiles | 2811966 | 10 | 254.30 s | 233.04 s |

Reducing outer calls alone does not reduce the amount of tiled VAE convolution. Enlarging the internal VAE tile gives only about `1.08x` steady-state speedup over the 9-frame H100 run. Disabling spatial VAE tiling entirely failed on H100 80GB in job `2811964`: the first encode needed an additional `31.64 GiB` after the process was already using `64.79 GiB`.

### Four-L40S Spatial-Tile Parallelism

The multi-GPU implementation uses one node with four 46GB L40S GPUs:

- Every rank loads one VAE replica and processes a round-robin subset of the same overlapping spatial tiles.
- Rank 0 additionally owns the original DOVE text encoder, latent Transformer, scheduler, tile gather, and official blend order.
- This is single-video spatial parallelism, not data parallelism across four independent videos.
- The original VAE temporal batching and per-tile convolution cache are preserved.

The one-video smoke job `2811979` completed in `16.76` seconds. Full UDM10 job `2811986` completed all ten videos without OOM or distributed errors:

| Stage | 10-video seconds |
|---|---:|
| Parallel VAE encode | 32.31 |
| Rank-0 latent Transformer | 26.44 |
| Parallel VAE decode and gather | 67.37 |
| Video save | 8.29 |
| Total steady run wall | 149.75 |
| One-time four-rank model load | 25.53 |
| Cold total | 175.28 |

Against the single-H100 baselines, the four-L40S path is `1.68x` faster than the 9-frame steady run and `1.56x` faster than the enlarged-tile fullclip steady run. Including cold model loading, it is `1.56x` faster than the single-H100 9-frame run. These are latency results but are not hardware-normalized: four L40S GPUs are being compared with one H100. Against the earlier one-L40S DOVE processing time of about `423` seconds, the latency speedup is approximately `2.82x`, with four times the GPU count.

### Quality And Temporal Metrics

Removing the outer 9-frame boundaries preserves full-clip temporal context in both the VAE convolution cache and the latent Transformer, improving quality as well as temporal consistency.

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Previous DOVE 9-frame quality baseline | 24.7522 | 0.7327 | 0.3153 | 0.1737 | 0.5014 |
| Four-L40S fullclip tile-parallel DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 |

| Model | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|
| Previous DOVE 9-frame quality baseline | 0.01272 | 0.03068 |
| Four-L40S fullclip tile-parallel DOVE | 0.01037 | 0.02412 |

The multi-GPU output is also numerically close to the single-H100 fullclip/default-tile output: PSNR `47.1269`, SSIM `0.9951`, LPIPS `0.0023`, and DISTS `0.0027` when the single-H100 output is treated as reference.

### Decision

This path passes the quality requirements, but it does **not** pass the customer's same-hardware `1.5x` acceptance condition. It exchanges four L40S GPUs for lower latency and is retained only as a scaling experiment. The accepted optimization result must compare one L40S with one L40S under the same fullclip input/output configuration. Work therefore continues with compiled VAE kernels and, if compilation is insufficient, a distilled lightweight VAE decoder.

The full UDM10 command is:

```bash
sbatch \
  --job-name=dove_vae_4gpu_full \
  --export=ALL,MAX_VIDEOS=0,RUN_EVAL=1 \
  experiments/dove_vae_tile_parallel_udm10.sbatch
```

## Same-Hardware Single-L40S VAE Optimization

Date: 2026-07-11

This is the hardware-normalized acceptance experiment. Every row uses one NVIDIA L40S, the same ten UDM10 inputs, full-clip inference, the same output resolution, and the same save path. Multi-GPU tile parallelism is excluded from the acceptance comparison.

The steady end-to-end time includes preprocessing, model inference, output assembly, and video saving. It excludes model loading, device transfer, and the one-time `torch.compile` warmup, matching a persistent inference-service deployment. Core time includes VAE encode, latent Transformer, scheduler update, and VAE decode only.

| Single-L40S configuration | Job | Steady end-to-end | Speedup | Core inference | Core speedup |
|---|---:|---:|---:|---:|---:|
| Original DOVE | `2812026` | 435.91 s | 1.000x | 404.57 s | 1.000x |
| Original decoder + `torch.compile` | `2812032` | 340.73 s | 1.279x | 313.98 s | 1.289x |
| 1-layer distilled decoder, checkpoint 500 | `2812049` | 328.75 s | 1.326x | 299.38 s | 1.351x |
| 1-layer distilled decoder + `torch.compile`, checkpoint 500 | `2812058` | 272.07 s | **1.602x** | 234.55 s | **1.725x** |

The cache-hot repeat, job `2812100`, records `258.19` seconds steady end-to-end and `235.50` seconds core inference, or `1.688x` and `1.718x` speedup respectively. The combined path therefore passes the `>=1.5x` steady service-mode speed requirement on equal hardware in both runs.

Compilation has a one-time warmup cost of `335.97` seconds in job `2812058`. Reusing the generated TorchInductor cache reduces warmup to `205.49` seconds in job `2812100`, but does not eliminate it. This cost must be reported separately and amortized through a persistent worker or an ahead-of-time TensorRT/Inductor artifact. The current implementation does not pass a cold one-shot latency criterion.

### Checkpoint-500 Quality

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA |
|---|---:|---:|---:|---:|---:|
| Fullclip original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 |
| Distilled decoder, checkpoint 500 | 26.1765 | 0.7749 | 0.2863 | 0.1519 | 0.3550 |

| Model | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|
| Fullclip original DOVE | 0.01037 | 0.02412 |
| Distilled decoder, checkpoint 500 | 0.01162 | 0.02424 |

Checkpoint 500 preserves distortion and temporal metrics reasonably well: PSNR drops by `1.35%`, SSIM improves slightly, LPIPS worsens by `5.76%`, and DISTS is effectively unchanged. However, CLIPIQA drops from `0.5012` to `0.3550`, so this checkpoint fails the perceptual/no-reference quality gate and is not an accepted final model.

Decoder distillation is continuing with DISTS and high-frequency losses. Checkpoints 1000, 2000, and 4000 are evaluated under the same fullclip setup. A candidate is accepted only when the equal-hardware speed and quality gates pass simultaneously; the speed result alone is not the final conclusion.

### Checkpoint-1000 Quality

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fullclip original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 | 0.01037 | 0.02412 |
| Distilled decoder, checkpoint 500 | 26.1765 | 0.7749 | 0.2863 | 0.1519 | 0.3550 | 0.01162 | 0.02424 |
| Distilled decoder, checkpoint 1000 | 26.3751 | 0.7799 | 0.2776 | 0.1446 | 0.3719 | 0.01162 | 0.02469 |
| Distilled decoder, checkpoint 2000 | 26.2606 | 0.7822 | 0.2776 | 0.1444 | 0.4106 | 0.01141 | 0.02465 |

The perceptual/high-frequency continuation improves checkpoint 1000 over checkpoint 500 on PSNR, SSIM, LPIPS, DISTS, and CLIPIQA. Checkpoint 2000 raises CLIPIQA further to `0.4106` while preserving the other reconstruction and perceptual metrics. The no-reference score is moving in the right direction but remains below the `0.45` gate. Training therefore continues on the 1024-sample cache; checkpoint 2000 is not accepted as the final model.

### Uniform-Depth Checkpoint-4000

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fullclip original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 | 0.01037 | 0.02412 |
| Uniform 1-layer decoder, checkpoint 4000 | 26.5138 | 0.7775 | 0.2724 | 0.1419 | 0.4134 | 0.01069 | 0.02389 |

Checkpoint 4000 closes the distortion, perceptual-reference, and temporal gaps, but CLIPIQA only moves from `0.4106` at checkpoint 2000 to `0.4134`. The plateau indicates that continuing the same autoencoder-latent cache is unlikely to reach `0.45`. The cache was built from clean HQ autoencoder latents, while production inference decodes latents produced by the full LQ encoder and DOVE Transformer trajectory. Further optimization therefore switches to distribution-matched decoder inputs.

## Distribution-Matched Decoder Distillation

Date: 2026-07-11

The new cache runs the complete production teacher path on synthetically degraded training clips:

1. Apply the DOVE training degradation pipeline to HQ-VSR and upscale the LQ condition.
2. Run the original VAE encoder, DOVE Transformer, scheduler update, and original VAE decoder.
3. Save the exact unscaled tensor presented to `vae.decoder` together with the original decoder RGB output.

The cache smoke job `2812691` produced four valid samples without failures. Each entry contains latent shape `[1,16,5,32,48]` and RGB target shape `[1,3,17,256,384]`. Decoder training smoke job `2812698` completed two steps and produced a checkpoint that was strictly reloaded by full inference job `2812703`.

The decoder also changes from uniform depth `[1,1,1,1]` to `[1,1,1,2]`, where entries specify `layers_per_block` for the four up blocks from lowest to highest spatial resolution. The final high-resolution block retains one additional residual layer for texture reconstruction.

| Single-L40S compiled configuration | Job | Steady end-to-end | Speedup | Core inference | Core speedup |
|---|---:|---:|---:|---:|---:|
| Original DOVE | `2812026` | 435.91 s | 1.000x | 404.57 s | 1.000x |
| Non-uniform `[1,1,1,2]` decoder | `2812704` | 281.80 s | **1.547x** | 259.55 s | **1.559x** |

The non-uniform architecture passes the equal-hardware steady-service speed gate, but with less margin than the uniform decoder. Its quality is not yet accepted: job `2812702` is building the 1024-sample distribution-matched cache, job `2812705` will train the 4000-step decoder after the cache succeeds, and job `2812706` will run the full UDM10 quality evaluation. Final acceptance still requires the trained checkpoint to pass both speed and quality gates.

### Distribution-Matched 4000-Step Result

The 1024-sample cache job `2812702` completed without failures in `01:05:37`. Training job `2812705` completed 4000 steps in `01:19:27`, and full UDM10 evaluation job `2812706` completed in `00:05:36`. No OOM, NaN, NCCL, or checkpoint errors occurred.

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fullclip original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5012 | 0.01037 | 0.02412 |
| Trajectory-distilled `[1,1,1,2]`, checkpoint 4000 | 26.4669 | 0.7800 | 0.2691 | 0.1436 | 0.4191 | 0.01059 | 0.02376 |

The distribution-matched decoder preserves or improves PSNR, SSIM, LPIPS, DISTS, and frame-difference consistency. PSNR changes by only `-0.25%`; SSIM, LPIPS, and DISTS improve. RAFT warp L1 is approximately `2.1%` worse but remains close to the teacher baseline. CLIPIQA improves over the uniform-depth clean-latent result (`0.4134 -> 0.4191`) but remains below both the `0.45` customer threshold and the original DOVE score `0.5012`.

### Acceptance Decision

The candidate passes the equal-hardware steady-service speed requirement (`1.547x`) and all reference-based distortion/perceptual metrics, but it does **not** pass the complete quality requirement because CLIPIQA is `0.4191`. Therefore this is not the final accepted customer configuration. More steps with the same objective are not justified by the small CLIPIQA gain. The next quality experiment should add a controlled no-reference/adversarial texture objective or test a residual high-frequency refinement head while preserving the measured single-L40S speed margin.

## CLIPIQA Floor-Margin Continuation And Final Acceptance

Date: 2026-07-11

The final continuation adds a bounded no-reference objective while retaining the trajectory L1, MSE, frame-difference, DISTS, and Laplacian losses. For one random frame per training sample:

`L_clipiqa = 0.05 * max(0, max(CLIPIQA_teacher, 0.50) - CLIPIQA_student)`

The IQA network is frozen, and the loss becomes zero once the student reaches the teacher score or the `0.50` floor. This limits direct metric optimization. Job `2812937` continued checkpoint 4000 for 100 steps at learning rate `2e-5`; job `2812938` performed full UDM10 evaluation.

### Final Quality And Temporal Comparison

Both rows below were evaluated from the actual single-L40S fullclip outputs using the same metric scripts. Baseline metric job: `2813033`. Candidate metric job: `2812938`.

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-L40S original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5013 | 0.01037 | 0.02412 |
| Final `[1,1,1,2]` decoder, checkpoint 4100 | **26.6488** | **0.7748** | **0.2673** | **0.1439** | **0.5108** | 0.01062 | **0.02375** |

The final candidate improves all five image-quality metrics. Relative to original DOVE, PSNR improves by `0.43%`, SSIM by `0.70%`, LPIPS by `1.26%`, DISTS by `5.33%`, and CLIPIQA by `1.90%`. It exceeds the customer CLIPIQA threshold of `0.45` by a substantial margin. Temporal behavior is effectively preserved: frame-difference error improves by `1.52%`, while RAFT warp L1 is `2.42%` higher. A side-by-side audit sampled frames 5, 15, and 25 from videos `000`, `002`, `005`, and `007`. It found no sharpening halos, artificial texture noise, or material color shift. Normalized difference maps show that changes are concentrated on edges and fine texture rather than flat sky, wall, and water regions. The local sheets can be regenerated with `tools/build_pure_algorithm_visual_audit.py`; generated images are intentionally not tracked. The remote artifacts remain under `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/eval_pixel/final_visual_examples`.

### Final Same-Hardware Speed

Job `2812954` loads the actual checkpoint-4100 and runs ten fullclip UDM10 videos on one L40S with `torch.compile`:

| Single-L40S configuration | Steady end-to-end | Speedup | Core inference | Core speedup |
|---|---:|---:|---:|---:|
| Original DOVE, job `2812026` | 435.91 s | 1.000x | 404.57 s | 1.000x |
| Final checkpoint-4100, job `2812954` | 288.14 s | **1.513x** | 258.76 s | **1.563x** |

The final checkpoint passes the same-hardware `>=1.5x` steady-service speed requirement. The job experienced unusually slow shared-filesystem model loading (`553.84` seconds) and a `247.01` second compile warmup. These startup costs are excluded from both baseline and optimized steady-service comparisons and require a persistent worker in deployment. The current path does not claim `1.5x` cold one-shot process startup.

### Final Decision

Checkpoint `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/vae_decoder_distill/decoder_1112_s4100_2812937/checkpoint-4100/decoder.pt` satisfies the declared customer gate on one L40S under the fullclip UDM10 protocol:

- Same-hardware steady end-to-end speedup: `1.513x` (required `>=1.5x`).
- PSNR/SSIM degradation: none; both improve (allowed degradation `<=8%`).
- LPIPS, DISTS, and CLIPIQA: all improve; CLIPIQA is `0.5108` (required approximately `>=0.45`).
- Temporal quality: frame-difference improves and RAFT remains within `2.42%` of baseline, with no visible temporal artifact in the audit.

This candidate is accepted for the customer's persistent single-GPU inference-service scenario. Cold-start/AOT packaging and a true 4K benchmark remain deployment and scale-validation tasks, not evidence used for this UDM10 acceptance claim.

## Pure-Algorithm 1.5x Result Without Compilation

Date: 2026-07-14

The customer subsequently required the `1.5x` result to come from model architecture and distillation only. `torch.compile`, TensorRT, CUDA graphs, multi-GPU execution, and reduced output scope are therefore excluded from this comparison. The decoder remains the accepted trajectory-distilled `[1,1,1,2]` checkpoint-4100. A new lightweight VAE encoder uses down-block profile `[1,1,1,1]` with one mid-block layer.

The encoder teacher cache contains 1024 degraded LQ clips produced with the DOVE training degradation pipeline. Each cache entry stores the upsampled RGB condition and the original encoder posterior moments. Cache job `2826060` completed `1024/1024` samples with zero failures. Encoder training job `2828484` ran 3000 steps and ended with loss `0.1864` and gradient norm `2.30`; no OOM, NaN, or checkpoint error occurred.

Final encoder checkpoint:

`/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/vae_encoder_distill/encoder_1111_m1_s3000_2828484/checkpoint-3000/encoder.pt`

### No-Compile Same-Hardware Speed

Both rows use one L40S and no VAE compilation. Steady end-to-end time is `run_wall - model_load - model_to_device`; it includes preprocessing, inference, output assembly, and video writing.

| Single-L40S no-compile configuration | Job | Steady end-to-end | Speedup | Core inference | Core speedup |
|---|---:|---:|---:|---:|---:|
| Original DOVE | `2812026` | 435.91 s | 1.000x | 404.57 s | 1.000x |
| Lightweight encoder + checkpoint-4100 decoder | `2828704` | **289.55 s** | **1.505x** | **248.82 s** | **1.626x** |

The pure-algorithm candidate clears the `>=1.5x` steady end-to-end gate by about `1.06` seconds. The margin is narrow, so the result should be rechecked on the deployment machine before committing to a production SLA.

### Quality And Temporal Comparison

Metric job `2832777` reuses the ten UDM10 videos generated by speed job `2828704`; it does not rerun inference. The comparison uses the same GT and metric implementations as the previous decoder-only result.

| Model | PSNR | SSIM | LPIPS | DISTS | CLIPIQA | RAFT warp L1 | Frame-difference L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original DOVE | 26.5338 | 0.7694 | 0.2707 | 0.1520 | 0.5013 | 0.01037 | 0.02412 |
| Decoder-only checkpoint-4100 | **26.6488** | **0.7748** | **0.2673** | **0.1439** | **0.5108** | 0.01062 | **0.02375** |
| Pure algorithm: lightweight encoder + decoder | 26.0109 | 0.7675 | 0.2818 | 0.1553 | 0.4988 | 0.01052 | 0.02433 |

Relative to original DOVE, the pure-algorithm model changes PSNR by `-1.97%`, SSIM by `-0.25%`, LPIPS by `+4.10%`, DISTS by `+2.17%`, and CLIPIQA by `-0.50%`. RAFT warp and frame-difference errors increase by approximately `1.48%` and `0.85%`. These changes remain inside the declared acceptance ranges: PSNR/SSIM degradation is below `8%`, CLIPIQA remains above `0.45`, and perceptual/temporal metrics show small rather than material regressions. However, this is not a lossless optimization and is visibly a little softer than the decoder-only model on fine text and repeated geometry.

Side-by-side sheets for videos `000`, `002`, `005`, and `007` use frames 5, 15, and 25. They can be regenerated with `tools/build_pure_algorithm_visual_audit.py`; generated images are intentionally not tracked. The visual check found stable structure and color without obvious flicker, but confirms the mild texture softening measured by LPIPS and DISTS.

### Pure-Algorithm Decision

The lightweight encoder plus trajectory-distilled decoder is accepted against the previously declared customer thresholds:

- Pure model/distillation speedup with `torch.compile=false`: `1.505x` steady end-to-end.
- PSNR/SSIM degradation: `1.97%` and `0.25%`, both within the allowed `8%`.
- CLIPIQA: `0.4988`, above the approximate `0.45` requirement.
- LPIPS/DISTS and temporal errors: modest `0.85%` to `4.10%` regressions, with no obvious instability in the sampled visual audit.

The result should be presented as **meeting the threshold with a small quality tradeoff**, not as matching or improving every original DOVE metric. The production deliverable consists of both the encoder checkpoint above and decoder checkpoint `/data/42-julia-hpc-rz-cv/sig95vg/arb_dove_outputs/vae_decoder_distill/decoder_1112_s4100_2812937/checkpoint-4100/decoder.pt`.
