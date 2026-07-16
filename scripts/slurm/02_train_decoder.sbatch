#!/usr/bin/env bash
#SBATCH --job-name=dove-dec-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=14:00:00

source "${DOVE_ROOT:?Set DOVE_ROOT}/scripts/slurm/common.sh"
prepare_job

main_dir="${RUN_ROOT}/models/decoder_main"
final_dir="${RUN_ROOT}/models/decoder_final"
mkdir -p "${main_dir}" "${final_dir}"

common_args=(
  --cache_dir "${RUN_ROOT}/cache/decoder"
  --model_path "${MODEL_PATH}"
  --layers_per_up_block 1 1 1 2
  --mse_weight 0.5
  --frame_diff_weight 0.2
  --dists_weight 0.05
  --dists_every 4
  --high_frequency_weight 0.2
  --seed "${SEED:-42}"
)

"${PYTHON_BIN}" -m dove_vae_distill.cli.train_decoder \
  "${common_args[@]}" \
  --output_dir "${main_dir}" \
  --max_steps 4000 \
  --learning_rate 1e-4 \
  --save_every 250

"${PYTHON_BIN}" -m dove_vae_distill.cli.train_decoder \
  "${common_args[@]}" \
  --output_dir "${final_dir}" \
  --resume_checkpoint "${main_dir}/checkpoint-4000/decoder.pt" \
  --max_steps 4100 \
  --learning_rate 2e-5 \
  --clipiqa_weight 0.05 \
  --clipiqa_target 0.50 \
  --clipiqa_every 1 \
  --save_every 100

echo "DECODER_CHECKPOINT=${final_dir}/checkpoint-4100/decoder.pt"
