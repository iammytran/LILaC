# Shared preamble for every shell script in `scripts/`.
#
# Sourced — not executed — by the wrapper scripts. Two responsibilities:
#
#   1. Activate the conda env required by the calling script.
#   2. Pick CUDA_VISIBLE_DEVICES dynamically (skipped if the caller already
#      set CUDA_VISIBLE_DEVICES, or if min_free_mib==0).
#
# Usage from a wrapper script:
#
#     REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
#     cd "$REPO_ROOT"
#     source "$REPO_ROOT/scripts/_lilac_preamble.sh" lilac-qwen 18000
#
# Args:
#   $1  conda env name (lilac-baseline | lilac-qwen | lilac-mmembed | …)
#   $2  minimum free GPU memory in MiB. Pass 0 for CPU-only scripts.

_LILAC_ENV_NAME="${1:?env name (\$1) is required}"
_LILAC_MIN_FREE_MIB="${2:-0}"

_LILAC_CONDA_HOME="${LILAC_CONDA_HOME:-/opt/miniconda3}"

# ─── 1) Conda env activation ──────────────────────────────────────────────────
if [[ -f "$_LILAC_CONDA_HOME/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$_LILAC_CONDA_HOME/etc/profile.d/conda.sh"
    conda activate "$_LILAC_ENV_NAME"
else
    echo "[preamble] WARNING: $_LILAC_CONDA_HOME/etc/profile.d/conda.sh not found —" \
         "running with whatever Python is on PATH." >&2
fi

# ─── 2) GPU selection (dynamic) ───────────────────────────────────────────────
if [[ "$_LILAC_MIN_FREE_MIB" -gt 0 && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        _LILAC_FREE_GPUS="$(
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
            awk -F',' -v need="$_LILAC_MIN_FREE_MIB" '$2+0 >= need {print $1}' |
            paste -sd, -
        )"
        if [[ -n "$_LILAC_FREE_GPUS" ]]; then
            export CUDA_VISIBLE_DEVICES="$_LILAC_FREE_GPUS"
            echo "[preamble] auto-picked CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
                 "(need ≥ ${_LILAC_MIN_FREE_MIB} MiB free per GPU)"
        else
            echo "[preamble] WARNING: no GPU has ≥ ${_LILAC_MIN_FREE_MIB} MiB free." \
                 "Falling back to CUDA_VISIBLE_DEVICES=0; consider re-trying later." >&2
            export CUDA_VISIBLE_DEVICES=0
        fi
    fi
fi
