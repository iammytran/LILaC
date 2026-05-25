#!/usr/bin/env python3
"""
Convert the Hugging Face MMCoQA-Doc release (parquet) → the LILaC
``datasets/MMCoQA/parsed_documents/dev/*.json`` + ``image_components/dev/*.png``
layout the LILaC pipeline expects.

The HF release ships ``text.parquet``, ``table.parquet``, ``image.parquet`` and
``image_dump.parquet``. Its built-in ``load.py`` already restores per-document
JSON files plus PNGs, but with a slimmer top-level schema than LILaC needs:

* HF schema:           ``{title, text, table, image}``  per component:
                       ``{text/table/image_name, heading_path, hyperlinks, label_id}``
* LILaC schema (used by step1–7):
                       ``{title, hierarchy, id_sequence, header,
                          text, table, image, sentence, proposition,
                          table_segment, subimage, id_to_html}``

This script bridges the two. The flat ``heading_path`` per component is folded
into the nested ``hierarchy`` tree (using the ``"COMPONENTS"`` key constant
that ``src/lilac/basic_class/component.py`` walks). Empty containers are added
for the layers the pipeline fills in later (``sentence``, ``table_segment``,
``subimage``, ``proposition``). Image entries get a ``filename`` field
(matching the on-disk name LILaC produces in ``image_components/dev/``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
except ImportError:
    print("pandas is required. `pip install pandas pyarrow` in the conda env first.", file=sys.stderr)
    sys.exit(1)


COMPONENTS_KEY = "COMPONENTS"   # matches MMDocConstants.COMPONENTS.value


def _safe_filename(name: str, max_len: int = 50) -> str:
    import re
    if not name:
        return "unknown"
    name = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', name).strip()
    if len(name) > max_len:
        name = name[:max_len]
    return name


def _heading_path_of(comp_dict: Dict[str, Any]) -> List[str]:
    hp = comp_dict.get("heading_path") or []
    if isinstance(hp, str):
        try:
            hp = json.loads(hp)
        except json.JSONDecodeError:
            hp = []
    return [str(s) for s in hp]


def _build_hierarchy(text: dict, table: dict, image: dict) -> dict:
    """Fold each component's flat heading_path into a nested hierarchy tree."""
    root: Dict[str, Any] = {COMPONENTS_KEY: []}
    for container in (text, table, image):
        for cid, comp in container.items():
            path = _heading_path_of(comp)
            node = root
            for seg in path:
                node = node.setdefault(seg, {COMPONENTS_KEY: []})
            node.setdefault(COMPONENTS_KEY, []).append(cid)
    return root


def _adapt_document(hf_doc: Dict[str, Any]) -> Dict[str, Any]:
    """One restored HF JSON → one LILaC parsed_document JSON."""
    text  = hf_doc.get("text", {})  or {}
    table = hf_doc.get("table", {}) or {}
    image = hf_doc.get("image", {}) or {}

    # Patch image entries: LILaC pipeline expects:
    #   - `filename`  (not `image_name`)
    #   - `caption`   as a dict {text, hyperlinks: []}, not a raw string
    for cid, comp in image.items():
        if "filename" not in comp:
            comp["filename"] = comp.get("image_name") or ""
        comp.setdefault("edges", [])
        # Normalize caption to LILaC's expected shape.
        cap = comp.get("caption")
        if isinstance(cap, str):
            comp["caption"] = {"text": cap, "hyperlinks": []}
        elif cap is None:
            comp["caption"] = {"text": "", "hyperlinks": []}
        elif isinstance(cap, dict):
            cap.setdefault("text", "")
            cap.setdefault("hyperlinks", [])

    # text / table entries: drop in an empty edges field for downstream steps.
    for cid, comp in text.items():
        comp.setdefault("edges", [])
    for cid, comp in table.items():
        comp.setdefault("edges", [])

    return {
        "title":         hf_doc.get("title", ""),
        "hierarchy":     _build_hierarchy(text, table, image),
        "id_sequence":   list(text.keys()) + list(table.keys()) + list(image.keys()),
        "header":        {},
        "text":          text,
        "table":         table,
        "image":         image,
        # Containers the pipeline fills in:
        "sentence":      {},
        "proposition":   {},
        "table_segment": {},
        "subimage":      {},
        "id_to_html":    {},
    }


def main():
    ap = argparse.ArgumentParser(description="HF MMCoQA-Doc → LILaC parsed_documents")
    ap.add_argument("--hf_dir",  default="datasets/MMCoQA",
                    help="Directory holding the HF release (with *.parquet + load.py).")
    ap.add_argument("--out_doc_dir", default="datasets/MMCoQA/parsed_documents/dev",
                    help="Target dir for LILaC-format JSON files.")
    ap.add_argument("--out_img_dir", default="datasets/MMCoQA/image_components/dev",
                    help="Target dir for image files (.png).")
    ap.add_argument("--workdir", default="/tmp/lilac_mmcoqa_hf_restore",
                    help="Scratch dir used by HF load.py; cleared on each run.")
    args = ap.parse_args()

    hf_dir   = Path(args.hf_dir).resolve()
    out_doc  = Path(args.out_doc_dir).resolve()
    out_img  = Path(args.out_img_dir).resolve()
    workdir  = Path(args.workdir).resolve()

    if not hf_dir.exists():
        sys.exit(f"HF dir not found: {hf_dir}")
    load_script = hf_dir / "load.py"
    if not load_script.exists():
        sys.exit(f"HF load.py not found at: {load_script}")

    # 1) Run HF's load.py to materialize per-document JSON + per-image PNG.
    if workdir.exists():
        shutil.rmtree(workdir)
    print(f"[1/3] running HF load.py → {workdir}")
    import subprocess
    subprocess.run(
        [sys.executable, str(load_script), "-p", str(hf_dir), "-s", str(workdir)],
        check=True,
    )

    # 2) Adapt each JSON to LILaC schema.
    print(f"[2/3] adapting JSON → {out_doc}")
    out_doc.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    for src in sorted(workdir.glob("*.json")):
        with src.open() as f:
            hf_doc = json.load(f)
        lilac_doc = _adapt_document(hf_doc)
        # Use the restored stem as the LILaC filename so component IDs stay aligned.
        out_path = out_doc / src.name
        with out_path.open("w") as f:
            json.dump(lilac_doc, f, ensure_ascii=False, indent=2)
        n_docs += 1
    print(f"      wrote {n_docs} parsed_documents/dev/*.json")

    # 3) Move images.
    print(f"[3/3] moving images → {out_img}")
    out_img.mkdir(parents=True, exist_ok=True)
    src_img_dir = workdir / "images"
    n_imgs = 0
    if src_img_dir.exists():
        for src in src_img_dir.iterdir():
            shutil.move(str(src), out_img / src.name)
            n_imgs += 1
    print(f"      moved {n_imgs} images to image_components/dev/")
    print("Done.")


if __name__ == "__main__":
    main()
