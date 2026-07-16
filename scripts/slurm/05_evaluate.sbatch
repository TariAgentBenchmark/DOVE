#!/usr/bin/env bash
#SBATCH --job-name=dove-pure-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
require_env UDM10_LQ UDM10_GT
prepare_job

output_dir="${RUN_ROOT}/evaluation"
baseline_dir="${output_dir}/baseline"
pred_dir="${output_dir}/pred"
mkdir -p "${baseline_dir}" "${pred_dir}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" inference_script.py \
  --input_dir "${UDM10_LQ}" \
  --model_path "${MODEL_PATH}" \
  --output_path "${baseline_dir}" \
  --dtype bfloat16 \
  --fps 8 \
  --save_format yuv444p \
  --chunk_len 0 \
  --is_vae_st \
  --profile_stages \
  --profile_output "${output_dir}/baseline_stage_profile.json"

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
  --vae_decoder_checkpoint "${RUN_ROOT}/models/decoder_final/checkpoint-4100/decoder.pt" \
  --vae_encoder_layers_per_down_block 1 1 1 1 \
  --vae_encoder_mid_block_layers 1 \
  --vae_encoder_checkpoint "${RUN_ROOT}/models/encoder/checkpoint-3000/encoder.pt"

mkdir -p "${output_dir}/baseline_metrics" "${output_dir}/candidate_metrics"
"${PYTHON_BIN}" eval_metrics.py \
  --gt "${UDM10_GT}" \
  --pred "${baseline_dir}" \
  --out "${output_dir}/baseline_metrics" \
  --metrics psnr,ssim,lpips,dists,clipiqa

"${PYTHON_BIN}" eval_temporal_metrics.py \
  --gt "${UDM10_GT}" \
  --pred "${baseline_dir}" \
  --out "${output_dir}/baseline_metrics" \
  --batch_size 2 \
  --flow_max_side 512

"${PYTHON_BIN}" eval_metrics.py \
  --gt "${UDM10_GT}" \
  --pred "${pred_dir}" \
  --out "${output_dir}/candidate_metrics" \
  --metrics psnr,ssim,lpips,dists,clipiqa

"${PYTHON_BIN}" eval_temporal_metrics.py \
  --gt "${UDM10_GT}" \
  --pred "${pred_dir}" \
  --out "${output_dir}/candidate_metrics" \
  --batch_size 2 \
  --flow_max_side 512

"${PYTHON_BIN}" scripts/summarize_pure_algorithm_run.py "${RUN_ROOT}"

echo "EVALUATION_DIR=${output_dir}"
