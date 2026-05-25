#!/usr/bin/env python3
"""
Generate a descriptive title for every page image in a parsed_documents
directory and write it into parsed_documents[*].title.

Shipped vqa-type benchmarks (MP-DocVQA, InfoVQA, SlideVQA) have
descriptive titles (e.g. "COVID-19-Related Threats in Q1 2020") that
flow into every component's embedding text via Image/Text/Table
serialization. Our adapters previously just stored the doc id stem
("33159") which gave the embedder no semantic anchor. This script
generates titles via Qwen2.5-VL on the page image, with a per-page
.title sidecar for resumability, and merges into parsed_documents.

Usage:
  python -m src.lilac.lcg_constructor.preprocessing.generate_titles \\
      --parsed_documents <datasets/<DS>/parsed_documents/dev> \\
      --images <datasets/<DS>/image_components/dev> \\
      --titles_out <datasets/<DS>/page_titles> \\
      --num_gpus 4

Idempotent: skips pages whose .title sidecar already exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import caption_images


TITLE_PROMPT = (
    "Generate a concise descriptive title (under 20 words) for this document "
    "page. Capture the main subject. Output only the title text — no quotes, "
    "no preamble, no commentary."
)


def collect_pages(parsed_documents_dir: Path, images_dir: Path):
    """Return parallel lists of (page image path, title sidecar path, doc_id)."""
    page_images = []
    title_paths = []
    doc_ids = []
    for pd_path in sorted(parsed_documents_dir.glob("*.json")):
        doc_id = pd_path.stem
        # Try jpg / jpeg / png variations
        for ext in (".jpg", ".jpeg", ".png"):
            img = images_dir / f"{doc_id}{ext}"
            if img.exists():
                page_images.append(str(img))
                doc_ids.append(doc_id)
                break
        else:
            continue
        # Title sidecar lives next to images_components_sub by default —
        # but we keep them in a dedicated dir so they don't get confused with
        # subimage crop captions.
        # (Filled in by caller — see main().)
    return page_images, doc_ids


def merge_titles(parsed_documents_dir: Path, title_paths: List[Path],
                doc_ids: List[str]) -> None:
    """Read each .title sidecar and write into parsed_documents[*].title."""
    updated, missing = 0, 0
    for doc_id, title_path in zip(doc_ids, title_paths):
        pd_path = parsed_documents_dir / f"{doc_id}.json"
        if not pd_path.exists() or not title_path.exists():
            missing += 1
            continue
        title_text = title_path.read_text(encoding="utf-8").strip()
        if not title_text:
            missing += 1
            continue
        with open(pd_path) as f:
            pd = json.load(f)
        pd["title"] = title_text
        with open(pd_path, "w") as f:
            json.dump(pd, f, indent=4)
        updated += 1
    print(f"[generate_titles] updated {updated} parsed_documents; {missing} skipped (no sidecar)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_documents", required=True,
                    help="parsed_documents/dev directory")
    ap.add_argument("--images", required=True,
                    help="image_components/dev directory (page images)")
    ap.add_argument("--titles_out", required=True,
                    help="Directory for per-page .title sidecars (resumable)")
    ap.add_argument("--max_tokens", type=int, default=64,
                    help="Title max tokens — short by design")
    ap.add_argument("--num_gpus", type=int, default=0,
                    help="Cap on Qwen-VL workers (0 = all visible)")
    args = ap.parse_args()

    pd_dir = Path(args.parsed_documents)
    img_dir = Path(args.images)
    out_dir = Path(args.titles_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_images, doc_ids = collect_pages(pd_dir, img_dir)
    title_paths = [out_dir / f"{d}.title" for d in doc_ids]

    print(f"[generate_titles] {len(page_images)} pages, "
          f"{sum(1 for p in title_paths if p.exists())} already have titles")

    caption_images(
        image_paths=page_images,
        output_paths=[str(p) for p in title_paths],
        prompt=TITLE_PROMPT,
        max_tokens=args.max_tokens,
        num_gpus=(args.num_gpus or None),
    )

    merge_titles(pd_dir, title_paths, doc_ids)


if __name__ == "__main__":
    main()
