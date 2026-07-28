#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
export DOVE_ROOT=${DOVE_ROOT:-${repo_root}}

required=(
  MODEL_PATH HQVSR_ROOT HQVSR_LIST UDM10_LQ UDM10_GT OUTPUT_ROOT
  BASE_DECODER_CACHE BASE_DECODER_CHECKPOINT ENCODER_CHECKPOINT
  DECODER_PARTITION EVAL_PARTITION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: ${name}" >&2
    exit 2
  fi
done

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export PERCEPTUAL_RUN_ROOT="${OUTPUT_ROOT%/}/perceptual-sweep-${run_id}"
export RUN_ROOT="${PERCEPTUAL_RUN_ROOT}"
mkdir -p "${PERCEPTUAL_RUN_ROOT}/logs"

sbatch_common=(
  --parsable
  --export=ALL
  --qos="${SLURM_QOS:-normal}"
  --output="${PERCEPTUAL_RUN_ROOT}/logs/%x_%A_%a.out"
  --error="${PERCEPTUAL_RUN_ROOT}/logs/%x_%A_%a.err"
)

cache_job=$(sbatch "${sbatch_common[@]}" --partition "${DECODER_PARTITION}" \
  "${script_dir}/06_build_perceptual_cache.sbatch")
train_job=$(sbatch "${sbatch_common[@]}" --partition "${DECODER_PARTITION}" \
  --array=0-2 --dependency="afterok:${cache_job}" \
  "${script_dir}/07_train_perceptual_sweep.sbatch")
eval_job=$(sbatch "${sbatch_common[@]}" --partition "${EVAL_PARTITION}" \
  --array=0-2 --dependency="afterok:${train_job}" \
  "${script_dir}/08_eval_perceptual_sweep.sbatch")

cat <<EOF
PERCEPTUAL_RUN_ROOT=${PERCEPTUAL_RUN_ROOT}
CACHE_JOB=${cache_job}
TRAIN_ARRAY_JOB=${train_job}
EVAL_ARRAY_JOB=${eval_job}
EOF
