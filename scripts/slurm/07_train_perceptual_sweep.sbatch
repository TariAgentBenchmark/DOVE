#!/usr/bin/env bash
#SBATCH --job-name=dove-perc-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=06:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env BASE_DECODER_CHECKPOINT PERCEPTUAL_RUN_ROOT
prepare_job

variant_id=${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-2}
variants=(conservative perceptual clip_margin)
teacher_l1=(0.50 0.25 0.25)
mse=(0.20 0.10 0.05)
gt_l1=(0.05 0.05 0.05)
dists=(0.05 0.10 0.05)
lpips=(0.05 0.10 0.10)
high_frequency=(0.10 0.10 0.20)
clipiqa=(0.03 0.03 0.06)
clipiqa_target=(0.505 0.505 0.510)

variant=${variants[variant_id]}
output_dir="${PERCEPTUAL_RUN_ROOT}/models/${variant}"
end_step=${PERCEPTUAL_MAX_STEPS:-4300}
mkdir -p "${output_dir}"

"${PYTHON_BIN}" -m dove_vae_distill.cli.train_decoder \
  --cache_dir "${PERCEPTUAL_RUN_ROOT}/cache/decoder_encoder_matched" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${output_dir}" \
  --layers_per_up_block 1 1 1 2 \
  --resume_checkpoint "${BASE_DECODER_CHECKPOINT}" \
  --max_steps "${end_step}" \
  --learning_rate 2e-5 \
  --teacher_l1_weight "${teacher_l1[variant_id]}" \
  --mse_weight "${mse[variant_id]}" \
  --gt_l1_weight "${gt_l1[variant_id]}" \
  --frame_diff_weight 0.10 \
  --dists_weight "${dists[variant_id]}" \
  --dists_every "${PERCEPTUAL_METRIC_EVERY:-4}" \
  --lpips_weight "${lpips[variant_id]}" \
  --lpips_every "${PERCEPTUAL_METRIC_EVERY:-4}" \
  --perceptual_target gt \
  --high_frequency_weight "${high_frequency[variant_id]}" \
  --clipiqa_weight "${clipiqa[variant_id]}" \
  --clipiqa_target "${clipiqa_target[variant_id]}" \
  --clipiqa_every 1 \
  --save_every "${PERCEPTUAL_SAVE_EVERY:-100}" \
  --seed "${SEED:-42}"

echo "VARIANT=${variant}"
echo "DECODER_CHECKPOINT=${output_dir}/checkpoint-${end_step}/decoder.pt"
