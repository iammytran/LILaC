#!/usr/bin/env bash
#
# Download embedder checkpoints into models/.
# Target directory names must match the *_PATH constants in
# src/models/embedder/{mmembed,unime,mme5}.py.
#
# Usage:
#   ./models/download_embedders.sh                # MM-Embed only (default)
#   ./models/download_embedders.sh --all          # MM-Embed + UniME + mmE5
#   ./models/download_embedders.sh --only unime   # one of {mmembed,unime,mme5}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=1

ALL=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)  ALL=1; shift;;
        --only) ONLY="$2"; shift 2;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

want()  { [[ "$ALL" -eq 1 || -z "$ONLY" && "$1" == "mmembed" || "$ONLY" == "$1" ]]; }

# MM-Embed  (src/models/embedder/mmembed.py → models/MM-Embed)
if want mmembed; then
    hf download \
        nvidia/MM-Embed \
        --local-dir "$REPO_ROOT/models/MM-Embed"
fi

# # UniME     (src/models/embedder/unime.py  → models/UniME-LLaVA-OneVision-7B)
# if want unime; then
#     hf download \
#         DeepGlint-AI/UniME-LLaVA-OneVision-7B \
#         --local-dir "$REPO_ROOT/models/UniME-LLaVA-OneVision-7B"
# fi

# # mmE5      (src/models/embedder/mme5.py   → models/mmE5-mllama-11b-instruct)
# if want mme5; then
#     hf download \
#         intfloat/mmE5-mllama-11b-instruct \
#         --local-dir "$REPO_ROOT/models/mmE5-mllama-11b-instruct"
# fi
