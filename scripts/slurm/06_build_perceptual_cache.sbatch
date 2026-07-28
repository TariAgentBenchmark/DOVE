#!/usr/bin/env bash
#SBATCH --job-name=dove-perc-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=10:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env HQVSR_ROOT HQVSR_LIST BASE_DECODER_CACHE ENCODER_CHECKPOINT
prepare_job

cache_dir="${PERCEPTUAL_RUN_ROOT:?Set PERCEPTUAL_RUN_ROOT}/cache/decoder_encoder_matched"
mkdir -p "${cache_dir}"
"${PYTHON_BIN}" -m dove_vae_distill.cli.build_decoder_cache \
  --model_path "${MODEL_PATH}" \
  --data_root "${HQVSR_ROOT}" \
  --video_column "${HQVSR_LIST}" \
  --output_dir "${cache_dir}" \
  --train_resolution "${TRAIN_RESOLUTION:-17x256x384}" \
  --degradation_config "${DEGRADATION_CONFIG:-finetune/configs/degradation_image_video.yaml}" \
  --num_samples "${NUM_SAMPLES:-1024}" \
  --sr_noise_step "${SR_NOISE_STEP:-399}" \
  --trajectory_encoder_checkpoint "${ENCODER_CHECKPOINT}" \
  --trajectory_encoder_layers_per_down_block 1 1 1 1 \
  --trajectory_encoder_mid_block_layers 1 \
  --teacher_cache_dir "${BASE_DECODER_CACHE}" \
  --store_ground_truth

echo "PERCEPTUAL_CACHE=${cache_dir}"
