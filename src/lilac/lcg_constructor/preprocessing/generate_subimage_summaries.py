#!/usr/bin/env python3
"""Generate a per-subimage VLM scene description for VQA-type benchmarks.

Writes a `summaries_subimage/dev/<doc>___i_<N>.txt` sidecar per figure
crop, consumed by `serializations/subimage+image+summary.json`. Only
applies to vqa-type datasets (InfoVQA, MP-DocVQA, SlideVQA); MMCoQA /
MultimodalQA reuse the parent page's summary instead.

Inputs:
  - `parsed_documents/dev/<doc>.json` for the figure-subimage list
  - `datasets/<DS>/image_components_sub/<doc>/<doc>___i_<N>.jpg` for the crop

Output:
  - `artifacts/<DS>/image_summaries_sub/dev/<doc>___i_<N>.txt`
    (flat dir; matches `image_subsummaries_dirname` alias).

Usage:
  python -m src.lilac.lcg_constructor.preprocessing.generate_subimage_summaries \\
      --target_data InfoVQA \\
      --num_gpus 3

Idempotent: skips entries whose output .txt already exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.utils import (
    read_yaml, REPO_ROOT, artifact_subpath, parsed_documents_path,
)
from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import (
    caption_images,
)


SUBIMAGE_PROMPT = (
    "Describe the image in a factual paragraph: list every visible object, "
    "person, and text element, and explain the spatial layout and what the "
    "image is depicting. Be concrete and specific — no preamble, no commentary."
)


def collect_subimage_jobs(config, dataset_name: str):
    """Walk parsed_documents and return parallel (crop_path, out_path) lists."""
    pd_dir = Path(parsed_documents_path(config, dataset_name))
    # Per-subimage crops live under datasets/<DS>/image_components_sub/<doc>/.
    root = Path(f"{REPO_ROOT}/datasets/{dataset_name}/image_components_sub")
    out_dir = Path(artifact_subpath(
        config, dataset_name, "image_subsummaries_dirname", "dev",
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    crops, outs = [], []
    missing_crops = 0
    for pd_path in sorted(pd_dir.glob("*.json")):
        doc_id = pd_path.stem
        with pd_path.open() as f:
            pd = json.load(f)
        subimages = pd.get("subimage", {})
        for entry in subimages.values():
            fn = entry.get("filename")  # e.g. "10840___i_3.jpg"
            if not fn:
                continue
            crop = root / doc_id / fn
            if not crop.exists():
                missing_crops += 1
                continue
            out_txt = out_dir / f"{Path(fn).stem}.txt"
            crops.append(str(crop))
            outs.append(str(out_txt))
    print(f"[generate_subimage_summaries] {len(crops)} crops to consider "
          f"({missing_crops} missing-on-disk skipped)")
    return crops, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_data", required=True,
                    help="Dataset name (e.g. InfoVQA)")
    ap.add_argument("--config", default=f"{REPO_ROOT}/config/lcg_constructor/a.yaml")
    ap.add_argument("--max_tokens", type=int, default=512,
                    help="Per-crop description max tokens")
    ap.add_argument("--num_gpus", type=int, default=0,
                    help="Cap on Qwen-VL workers (0 = all visible)")
    args = ap.parse_args()

    config = read_yaml(args.config)
    if args.target_data not in config["dataset_metadata"]:
        raise SystemExit(f"unknown dataset: {args.target_data}")
    dtype = config["dataset_metadata"][args.target_data].get("type", "")
    if dtype != "vqa":
        raise SystemExit(
            f"dataset {args.target_data} type={dtype!r}; this helper is only "
            f"for vqa-type benchmarks (multimodalqa-type reuses page summary)."
        )

    crops, outs = collect_subimage_jobs(config, args.target_data)
    if not crops:
        print("[generate_subimage_summaries] nothing to do.")
        return

    caption_images(
        image_paths=crops,
        output_paths=outs,
        prompt=SUBIMAGE_PROMPT,
        max_tokens=args.max_tokens,
        num_gpus=(args.num_gpus or None),
    )


if __name__ == "__main__":
    main()
