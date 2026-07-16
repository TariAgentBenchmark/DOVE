#!/usr/bin/env bash
#SBATCH --job-name=dove-enc-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
prepare_job

output_dir="${RUN_ROOT}/models/encoder"
mkdir -p "${output_dir}"
"${PYTHON_BIN}" -m dove_vae_distill.cli.train_encoder \
  --cache_dir "${RUN_ROOT}/cache/encoder" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${output_dir}" \
  --layers_per_down_block 1 1 1 1 \
  --mid_block_layers 1 \
  --max_steps 3000 \
  --learning_rate 1e-4 \
  --mean_mse_weight 0.5 \
  --logvar_l1_weight 0.1 \
  --logvar_mse_weight 0.05 \
  --sample_weight 0.25 \
  --save_every 250 \
  --seed "${SEED:-42}"

echo "ENCODER_CHECKPOINT=${output_dir}/checkpoint-3000/encoder.pt"
