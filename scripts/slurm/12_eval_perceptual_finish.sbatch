#!/usr/bin/env bash
#SBATCH --job-name=dove-perc-finish-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env UDM10_LQ UDM10_GT ENCODER_CHECKPOINT FINISH_RUN_ROOT
export RUN_ROOT="${FINISH_RUN_ROOT}"
prepare_job

eval_id=${SLURM_ARRAY_TASK_ID:?Submit this script with --array=0-3}
variants=(lpips_continue lpips_push)
steps=(4450 4500)
variant=${variants[eval_id / 2]}
step=${steps[eval_id % 2]}
name="${variant}_s${step}"
checkpoint="${FINISH_RUN_ROOT}/models/${variant}/checkpoint-${step}/decoder.pt"
output_dir="${FINISH_RUN_ROOT}/evaluation/${name}"
pred_dir="${output_dir}/pred"
metrics_dir="${output_dir}/metrics"
mkdir -p "${pred_dir}" "${metrics_dir}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" inference_script.py \
  --input_dir "${UDM10_LQ}" \
  --model_path "${MODEL_PATH}" \
  --output_path "${pred_dir}" \
  --dtype bfloat16 \
  --fps 8 \
  --save_format yuv444p \
  --chunk_len 0 \
  --is_vae_st \
  --profile_stages \
  --profile_output "${output_dir}/stage_profile.json" \
  --vae_decoder_layers_per_up_block 1 1 1 2 \
  --vae_decoder_checkpoint "${checkpoint}" \
  --vae_encoder_layers_per_down_block 1 1 1 1 \
  --vae_encoder_mid_block_layers 1 \
  --vae_encoder_checkpoint "${ENCODER_CHECKPOINT}"

"${PYTHON_BIN}" eval_metrics.py \
  --gt "${UDM10_GT}" \
  --pred "${pred_dir}" \
  --out "${metrics_dir}" \
  --metrics psnr,ssim,lpips,dists,clipiqa

"${PYTHON_BIN}" eval_temporal_metrics.py \
  --gt "${UDM10_GT}" \
  --pred "${pred_dir}" \
  --out "${metrics_dir}" \
  --batch_size 2 \
  --flow_max_side 512

echo "VARIANT=${variant}"
echo "STEP=${step}"
echo "EVALUATION_DIR=${output_dir}"
