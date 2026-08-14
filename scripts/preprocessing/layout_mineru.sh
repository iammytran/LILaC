#!/usr/bin/env bash
#
# Preprocessing — MinerU layout analysis (CLI wrapper) on a folder of images / PDFs.
#
# Usage:
#   ./scripts/preprocessing/layout_mineru.sh \
#         --input  path/to/raw_pdfs_or_images \
#         --output path/to/mineru_outputs
#
#   ./scripts/preprocessing/layout_mineru.sh \
#         --input  datasets/MyDataset/raw_pdfs \
#         --output artifacts/MyDataset/mineru/dev \
#         --backend pipeline --method auto
#
# Run `./models/download_layout_analyzers.sh --only mineru` once beforehand
# to install the mineru CLI + fetch its model weights.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT=""
OUTPUT=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)  INPUT="$2"; shift 2;;
        -o|--output) OUTPUT="$2"; shift 2;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0;;
        *) EXTRA+=("$1"); shift;;
    esac
done
if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Both --input and --output are required." >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python3 \
    src/lilac/lcg_constructor/preprocessing/layout_mineru.py \
    --input_dir "$INPUT" --output_dir "$OUTPUT" "${EXTRA[@]}" \
    --backend pipeline
