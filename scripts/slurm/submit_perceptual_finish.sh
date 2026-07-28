#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
export DOVE_ROOT=${DOVE_ROOT:-${repo_root}}

required=(
  PYTHON_BIN MODEL_PATH UDM10_LQ UDM10_GT OUTPUT_ROOT SOURCE_PERCEPTUAL_RUN_ROOT
  BASE_FINISH_CHECKPOINT ENCODER_CHECKPOINT DECODER_PARTITION EVAL_PARTITION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: ${name}" >&2
    exit 2
  fi
done

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export FINISH_RUN_ROOT="${OUTPUT_ROOT%/}/perceptual-finish-${run_id}"
export RUN_ROOT="${FINISH_RUN_ROOT}"
mkdir -p "${FINISH_RUN_ROOT}/logs"

common=(
  --parsable
  --export=ALL
  --qos="${SLURM_QOS:-normal}"
  --output="${FINISH_RUN_ROOT}/logs/%x_%A_%a.out"
  --error="${FINISH_RUN_ROOT}/logs/%x_%A_%a.err"
)

train_job=$(sbatch "${common[@]}" --partition "${DECODER_PARTITION}" --array=0-1 \
  "${script_dir}/11_train_perceptual_finish.sbatch")
if [[ -n "${EVAL_GRES:-}" ]]; then
  eval_job=$(sbatch "${common[@]}" --partition "${EVAL_PARTITION}" \
    --gres "${EVAL_GRES}" --array=0-3 --dependency="afterok:${train_job}" \
    "${script_dir}/12_eval_perceptual_finish.sbatch")
else
  eval_job=$(sbatch "${common[@]}" --partition "${EVAL_PARTITION}" \
    --array=0-3 --dependency="afterok:${train_job}" \
    "${script_dir}/12_eval_perceptual_finish.sbatch")
fi

cat <<EOF
FINISH_RUN_ROOT=${FINISH_RUN_ROOT}
TRAIN_ARRAY_JOB=${train_job}
EVAL_ARRAY_JOB=${eval_job}
EOF
