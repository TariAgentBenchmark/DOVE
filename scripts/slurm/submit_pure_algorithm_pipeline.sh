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
mkdir -p "${RUN_ROOT}/logs"
{
  echo "DOVE_ROOT=${DOVE_ROOT}"
  git -C "${DOVE_ROOT}" rev-parse HEAD
  git -C "${DOVE_ROOT}" status --short
} >"${RUN_ROOT}/source-state.txt"
git -C "${DOVE_ROOT}" diff >"${RUN_ROOT}/source.patch"

sbatch_args=(--parsable --export=ALL)

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
evaluation_job=$(
  submit "${EVAL_PARTITION}" 05_evaluate.sbatch \
    --dependency="afterok:${decoder_train_job}:${encoder_train_job}"
)

cat <<EOF
RUN_ROOT=${RUN_ROOT}
DECODER_CACHE_JOB=${decoder_cache_job}
DECODER_TRAIN_JOB=${decoder_train_job}
ENCODER_CACHE_JOB=${encoder_cache_job}
ENCODER_TRAIN_JOB=${encoder_train_job}
EVALUATION_JOB=${evaluation_job}
EOF
