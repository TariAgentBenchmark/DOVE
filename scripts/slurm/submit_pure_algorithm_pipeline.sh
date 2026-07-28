#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
export DOVE_ROOT=${DOVE_ROOT:-${repo_root}}

required=(
  MODEL_PATH HQVSR_ROOT HQVSR_LIST UDM10_LQ UDM10_GT OUTPUT_ROOT
  DECODER_PARTITION ENCODER_PARTITION EVAL_PARTITION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: ${name}" >&2
    exit 2
  fi
done

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export RUN_ROOT="${OUTPUT_ROOT%/}/pure-algorithm-${run_id}"
export PERCEPTUAL_RUN_ROOT="${RUN_ROOT}"
export SOURCE_PERCEPTUAL_RUN_ROOT="${RUN_ROOT}"
export REFINE_RUN_ROOT="${RUN_ROOT}/refine"
export FINISH_RUN_ROOT="${RUN_ROOT}/finish"
export BASE_DECODER_CACHE="${RUN_ROOT}/cache/decoder"
export BASE_DECODER_CHECKPOINT="${RUN_ROOT}/models/decoder_final/checkpoint-4100/decoder.pt"
export ENCODER_CHECKPOINT="${RUN_ROOT}/models/encoder/checkpoint-3000/encoder.pt"
export BASE_REFINE_CHECKPOINT="${RUN_ROOT}/models/perceptual/checkpoint-4300/decoder.pt"
export BASE_FINISH_CHECKPOINT="${REFINE_RUN_ROOT}/models/lpips_focus/checkpoint-4400/decoder.pt"
export FINAL_DECODER_CHECKPOINT="${FINISH_RUN_ROOT}/models/lpips_continue/checkpoint-4500/decoder.pt"
mkdir -p "${RUN_ROOT}/logs"
{
  echo "DOVE_ROOT=${DOVE_ROOT}"
  git -C "${DOVE_ROOT}" rev-parse HEAD
  git -C "${DOVE_ROOT}" status --short
} >"${RUN_ROOT}/source-state.txt"
git -C "${DOVE_ROOT}" diff >"${RUN_ROOT}/source.patch"

sbatch_args=(--parsable --export=ALL --qos="${SLURM_QOS:-normal}")

submit() {
  local partition=$1
  local script=$2
  shift 2
  sbatch "${sbatch_args[@]}" "$@" \
    --partition "${partition}" \
    --output "${RUN_ROOT}/logs/%x_%j.out" \
    --error "${RUN_ROOT}/logs/%x_%j.err" \
    "${script_dir}/${script}"
}

decoder_cache_job=$(submit "${DECODER_PARTITION}" 01_build_decoder_cache.sbatch)
encoder_cache_job=$(submit "${ENCODER_PARTITION}" 03_build_encoder_cache.sbatch)
decoder_train_job=$(submit "${DECODER_PARTITION}" 02_train_decoder.sbatch --dependency="afterok:${decoder_cache_job}")
encoder_train_job=$(submit "${ENCODER_PARTITION}" 04_train_encoder.sbatch --dependency="afterok:${encoder_cache_job}")
perceptual_cache_job=$(
  submit "${DECODER_PARTITION}" 06_build_perceptual_cache.sbatch \
    --dependency="afterok:${decoder_cache_job}:${encoder_train_job}"
)
perceptual_train_job=$(
  submit "${DECODER_PARTITION}" 07_train_perceptual_sweep.sbatch \
    --array=1 \
    --dependency="afterok:${decoder_train_job}:${perceptual_cache_job}"
)
refine_train_job=$(
  submit "${DECODER_PARTITION}" 09_train_perceptual_refine.sbatch \
    --array=2 \
    --dependency="afterok:${perceptual_train_job}"
)
finish_train_job=$(
  submit "${DECODER_PARTITION}" 11_train_perceptual_finish.sbatch \
    --array=0 \
    --dependency="afterok:${refine_train_job}"
)

if [[ -n "${EVAL_GRES:-}" ]]; then
  evaluation_job=$(
    submit "${EVAL_PARTITION}" 13_eval_perceptual_final.sbatch \
      --gres "${EVAL_GRES}" \
      --dependency="afterok:${finish_train_job}"
  )
else
  evaluation_job=$(
    submit "${EVAL_PARTITION}" 13_eval_perceptual_final.sbatch \
      --dependency="afterok:${finish_train_job}"
  )
fi

cat <<EOF
RUN_ROOT=${RUN_ROOT}
DECODER_CACHE_JOB=${decoder_cache_job}
DECODER_TRAIN_JOB=${decoder_train_job}
ENCODER_CACHE_JOB=${encoder_cache_job}
ENCODER_TRAIN_JOB=${encoder_train_job}
PERCEPTUAL_CACHE_JOB=${perceptual_cache_job}
PERCEPTUAL_TRAIN_JOB=${perceptual_train_job}
REFINE_TRAIN_JOB=${refine_train_job}
FINISH_TRAIN_JOB=${finish_train_job}
EVALUATION_JOB=${evaluation_job}
FINAL_DECODER_CHECKPOINT=${FINAL_DECODER_CHECKPOINT}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT}
EOF
