#!/usr/bin/env bash
#SBATCH --job-name=dove-perc-finish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=02:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env SOURCE_PERCEPTUAL_RUN_ROOT FINISH_RUN_ROOT BASE_FINISH_CHECKPOINT
export RUN_ROOT="${FINISH_RUN_ROOT}"
prepare_job

variant_id=${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-1}
variants=(lpips_continue lpips_push)
teacher_l1=(0.15 0.12)
mse=(0.03 0.02)
gt_l1=(0.08 0.10)
dists=(0.15 0.18)
lpips=(0.25 0.40)

variant=${variants[variant_id]}
output_dir="${FINISH_RUN_ROOT}/models/${variant}"
mkdir -p "${output_dir}"

"${PYTHON_BIN}" -m dove_vae_distill.cli.train_decoder \
  --cache_dir "${SOURCE_PERCEPTUAL_RUN_ROOT}/cache/decoder_encoder_matched" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${output_dir}" \
  --layers_per_up_block 1 1 1 2 \
  --resume_checkpoint "${BASE_FINISH_CHECKPOINT}" \
  --max_steps 4500 \
  --learning_rate 1e-5 \
  --teacher_l1_weight "${teacher_l1[variant_id]}" \
  --mse_weight "${mse[variant_id]}" \
  --gt_l1_weight "${gt_l1[variant_id]}" \
  --frame_diff_weight 0.10 \
  --dists_weight "${dists[variant_id]}" \
  --dists_every 1 \
  --lpips_weight "${lpips[variant_id]}" \
  --lpips_every 1 \
  --perceptual_target gt \
  --high_frequency_weight 0.00 \
  --clipiqa_weight 0.01 \
  --clipiqa_target 0.505 \
  --clipiqa_every 1 \
  --save_every 50 \
  --seed "${SEED:-42}"

echo "VARIANT=${variant}"
echo "CHECKPOINT_4450=${output_dir}/checkpoint-4450/decoder.pt"
echo "CHECKPOINT_4500=${output_dir}/checkpoint-4500/decoder.pt"
