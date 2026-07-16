#!/usr/bin/env bash

set -euo pipefail

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "Required environment variable is not set: ${name}" >&2
      exit 2
    fi
  done
}

prepare_job() {
  require_env DOVE_ROOT RUN_ROOT MODEL_PATH
  PYTHON_BIN=${PYTHON_BIN:-python}
  local job_id=${SLURM_JOB_ID:-local}
  local cache_root="${RUN_ROOT}/runtime_cache/${job_id}"
  export TORCH_HOME=${TORCH_HOME:-${RUN_ROOT}/shared_cache/torch}

  mkdir -p "${cache_root}"/{triton,torch_extensions,xdg,hf,wandb} "${TORCH_HOME}"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  export TORCH_EXTENSIONS_DIR="${cache_root}/torch_extensions"
  export XDG_CACHE_HOME="${cache_root}/xdg"
  export HF_HOME="${cache_root}/hf"
  export WANDB_DIR="${cache_root}/wandb"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1

  cd "${DOVE_ROOT}"
  mkdir -p "${RUN_ROOT}/environment"
  {
    "${PYTHON_BIN}" --version
    "${PYTHON_BIN}" -m pip freeze
  } >"${RUN_ROOT}/environment/python-${job_id}.txt"
}
