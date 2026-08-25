#!/usr/bin/env bash
#
# Step 5 — extract subimages from page/inline images.
#
# Dispatches per dataset type (retriever_metadata.yaml):
#   multimodalqa → Qwen-VL bbox detect-and-crop + caption.text fill
#   vqa          → MinerU (default) layout analysis + adapter
#
# Override the type-based default with `--analyzer doclayout_yolo|mineru|qwen_bbox`.
# Enable adaptive doclayout tiling with:
#   --adaptive_tiling --tiling_min_subcomponents 18 --tiling_min_aspect_ratio 1.6
#
# Usage:
#   ./step5_add_subimages.sh                                 # all benchmarks
#   ./step5_add_subimages.sh -b "MMCoQA"                     # auto -> qwen_bbox
#   ./step5_add_subimages.sh -b "MP-DocVQA"                   # auto -> mineru
#   ./step5_add_subimages.sh -b "MP-DocVQA" -a doclayout_yolo
#   ./step5_add_subimages.sh -b "MMCoQA" --start_idx 0 --end_idx 200
#
# Conda env: lilac-qwen (Qwen-VL + dispatcher). The layout-analyzer
# subprocess is launched in its own env by the dispatcher.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/_lilac_preamble.sh" lilac-qwen 18000

ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
BENCHMARKS=("${ALL_BENCHMARKS[@]}")
MODE="all"
ANALYZER=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; shift 2;;
        -a|--analyzer)   ANALYZER="$2"; shift 2;;
        -m|--mode)       MODE="$2"; shift 2;;
        --start_idx)     EXTRA+=(--start_idx "$2"); shift 2;;
        --end_idx)       EXTRA+=(--end_idx "$2"); shift 2;;
        --num_gpus)      EXTRA+=(--num_gpus "$2"); shift 2;;
        --adaptive_tiling) EXTRA+=(--adaptive_tiling); shift 1;;
        --tiling_min_subcomponents) EXTRA+=(--tiling_min_subcomponents "$2"); shift 2;;
        --tiling_min_aspect_ratio) EXTRA+=(--tiling_min_aspect_ratio "$2"); shift 2;;
        --tiling_overlap) EXTRA+=(--tiling_overlap "$2"); shift 2;;
        --tiling_max_tiles) EXTRA+=(--tiling_max_tiles "$2"); shift 2;;
        -h|--help) sed -n '2,24p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

ANALYZER_ARGS=()
if [[ -n "$ANALYZER" ]]; then
    ANALYZER_ARGS=(--analyzer "$ANALYZER")
fi

echo "[step5] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits | sed 's/^/[step5][nvidia-smi] /'

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step5 $DS [$MODE]"
    python3 -m src.lilac.lcg_constructor.step5_add_subimages \
        --target_data "$DS" --mode "$MODE" \
        "${ANALYZER_ARGS[@]}" "${EXTRA[@]}"
done
