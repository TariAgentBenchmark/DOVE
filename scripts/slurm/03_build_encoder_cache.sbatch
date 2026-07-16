#!/usr/bin/env bash
#SBATCH --job-name=dove-enc-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env HQVSR_ROOT HQVSR_LIST
prepare_job

mkdir -p "${RUN_ROOT}/cache/encoder"
"${PYTHON_BIN}" -m dove_vae_distill.cli.build_encoder_cache \
  --model_path "${MODEL_PATH}" \
  --data_root "${HQVSR_ROOT}" \
  --video_column "${HQVSR_LIST}" \
  --output_dir "${RUN_ROOT}/cache/encoder" \
  --train_resolution "${TRAIN_RESOLUTION:-17x256x384}" \
  --degradation_config "${DEGRADATION_CONFIG:-finetune/configs/degradation_image_video.yaml}" \
  --num_samples "${NUM_SAMPLES:-1024}" \
  --upscale_mode bilinear
