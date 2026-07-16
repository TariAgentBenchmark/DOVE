#!/usr/bin/env bash
#SBATCH --job-name=dove-dec-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=08:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env HQVSR_ROOT HQVSR_LIST
prepare_job

mkdir -p "${RUN_ROOT}/cache/decoder"
"${PYTHON_BIN}" -m dove_vae_distill.cli.build_decoder_cache \
  --model_path "${MODEL_PATH}" \
  --data_root "${HQVSR_ROOT}" \
  --video_column "${HQVSR_LIST}" \
  --output_dir "${RUN_ROOT}/cache/decoder" \
  --train_resolution "${TRAIN_RESOLUTION:-17x256x384}" \
  --degradation_config "${DEGRADATION_CONFIG:-finetune/configs/degradation_image_video.yaml}" \
  --num_samples "${NUM_SAMPLES:-1024}" \
  --sr_noise_step "${SR_NOISE_STEP:-399}"
