#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
export DOVE_ROOT=${DOVE_ROOT:-${repo_root}}

required=(
  MODEL_PATH UDM10_LQ UDM10_GT OUTPUT_ROOT SOURCE_PERCEPTUAL_RUN_ROOT
  BASE_REFINE_CHECKPOINT ENCODER_CHECKPOINT DECODER_PARTITION EVAL_PARTITION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: ${name}" >&2
    exit 2
  fi
done

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export REFINE_RUN_ROOT="${OUTPUT_ROOT%/}/perceptual-refine-${run_id}"
export RUN_ROOT="${REFINE_RUN_ROOT}"
mkdir -p "${REFINE_RUN_ROOT}/logs"

common=(
  --parsable
  --export=ALL
  --qos="${SLURM_QOS:-normal}"
  --output="${REFINE_RUN_ROOT}/logs/%x_%A_%a.out"
  --error="${REFINE_RUN_ROOT}/logs/%x_%A_%a.err"
)

train_job=$(sbatch "${common[@]}" --partition "${DECODER_PARTITION}" --array=0-2 \
  "${script_dir}/09_train_perceptual_refine.sbatch")
if [[ -n "${EVAL_GRES:-}" ]]; then
  eval_job=$(sbatch "${common[@]}" --partition "${EVAL_PARTITION}" \
    --gres "${EVAL_GRES}" --array=0-5 --dependency="afterok:${train_job}" \
    "${script_dir}/10_eval_perceptual_refine.sbatch")
else
  eval_job=$(sbatch "${common[@]}" --partition "${EVAL_PARTITION}" \
    --array=0-5 --dependency="afterok:${train_job}" \
    "${script_dir}/10_eval_perceptual_refine.sbatch")
fi

cat <<EOF
REFINE_RUN_ROOT=${REFINE_RUN_ROOT}
TRAIN_ARRAY_JOB=${train_job}
EVAL_ARRAY_JOB=${eval_job}
EOF
