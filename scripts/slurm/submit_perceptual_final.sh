#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
export DOVE_ROOT=${DOVE_ROOT:-${repo_root}}

required=(
  PYTHON_BIN MODEL_PATH UDM10_LQ UDM10_GT OUTPUT_ROOT
  FINAL_DECODER_CHECKPOINT ENCODER_CHECKPOINT EVAL_PARTITION
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: ${name}" >&2
    exit 2
  fi
done

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export RUN_ROOT="${OUTPUT_ROOT%/}/perceptual-final-${run_id}"
mkdir -p "${RUN_ROOT}/logs"

if [[ -n "${EVAL_GRES:-}" ]]; then
  job=$(sbatch \
    --parsable \
    --export=ALL \
    --qos="${SLURM_QOS:-normal}" \
    --partition="${EVAL_PARTITION}" \
    --gres "${EVAL_GRES}" \
    --output="${RUN_ROOT}/logs/%x_%j.out" \
    --error="${RUN_ROOT}/logs/%x_%j.err" \
    "${script_dir}/13_eval_perceptual_final.sbatch")
else
  job=$(sbatch \
    --parsable \
    --export=ALL \
    --qos="${SLURM_QOS:-normal}" \
    --partition="${EVAL_PARTITION}" \
    --output="${RUN_ROOT}/logs/%x_%j.out" \
    --error="${RUN_ROOT}/logs/%x_%j.err" \
    "${script_dir}/13_eval_perceptual_final.sbatch")
fi

cat <<EOF
RUN_ROOT=${RUN_ROOT}
FINAL_EVAL_JOB=${job}
EOF
