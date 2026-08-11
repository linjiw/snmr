#!/usr/bin/env bash
# Source this file from any directory to activate the local SNMR environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Source this script: source scripts/activate_snmr.sh\n' >&2
    exit 1
fi

_snmr_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$_snmr_root/.venv/bin/python" ]]; then
    printf 'SNMR environment is missing: %s/.venv\n' "$_snmr_root" >&2
    return 1
fi

source "$_snmr_root/.venv/bin/activate"

# Prefer the isolated, repository-pinned teacher clone provisioned on the data disk.
if [[ -z "${SNMR_GMR_ROOT:-}" ]]; then
    if [[ -d /data/robotixx/snmr-externals/GMR ]]; then
        export SNMR_GMR_ROOT=/data/robotixx/snmr-externals/GMR
    else
        export SNMR_GMR_ROOT="$_snmr_root/../GMR"
    fi
fi
export SNMR_HOLOSOMA_ROOT="${SNMR_HOLOSOMA_ROOT:-$_snmr_root/../holosoma}"
export SNMR_DATA_ROOT="${SNMR_DATA_ROOT:-$_snmr_root/../data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Prefer portable research tools installed on the data disk (for example,
# Tectonic) without requiring host-level packages.
export SNMR_TOOLS_ROOT="${SNMR_TOOLS_ROOT:-/data/robotixx/snmr-tools}"
if [[ -d "$SNMR_TOOLS_ROOT/bin" ]]; then
    export PATH="$SNMR_TOOLS_ROOT/bin:$PATH"
fi
export SNMR_CACHE_ROOT="${SNMR_CACHE_ROOT:-/data/robotixx/snmr-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SNMR_CACHE_ROOT}"
export TORCH_HOME="${TORCH_HOME:-$SNMR_CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SNMR_CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$SNMR_CACHE_ROOT/cuda}"
if [[ -d "$SNMR_CACHE_ROOT/tmp" ]]; then
    export TMPDIR="${TMPDIR:-$SNMR_CACHE_ROOT/tmp}"
fi

# Keep globally installed ROS/Python packages out of both isolated research environments.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

# Holosoma/MJWarp intentionally stays in its own environment.
export WBT_PYTHON="${WBT_PYTHON:-$SNMR_HOLOSOMA_ROOT/.venv/hsmujoco/bin/python}"

unset _snmr_root
