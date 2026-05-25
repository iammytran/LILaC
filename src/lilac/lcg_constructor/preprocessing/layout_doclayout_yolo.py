#!/usr/bin/env python3
"""
DocLayout-YOLO layout analysis (preprocessing).

Runs DocLayout-YOLO over every image in ``--input_dir``, deduplicates boxes,
filters out boxes contained within other boxes (with a small carve-out for
non-figure boxes nested in figure boxes), and writes:

* ``<output_dir>/<image_basename>/<idx>_<class>_<sub_idx>.jpg`` — one crop per
  detected box.
* ``<output_dir>/<image_basename>/<image_basename>_annotated.jpg`` — the input
  image overlaid with the detected boxes.
* ``<output_dir>/<image_basename>/boxes.json`` — the cleaned-up box list
  (``{idx, class, confidence, xyxy}``).

This script is **decoupled from the dataset registry** by design — it can be
run on a folder of raw user images *before* the dataset is added to
``config/retriever/retriever_metadata.yaml``. Downstream code that builds the
``parsed_documents/dev/*.json`` files can then consume ``boxes.json``.

Model checkpoint default: ``models/DocLayout-YOLO-DocStructBench/
doclayout_yolo_docstructbench_imgsz1280_2501.pt`` (downloaded by
``models/download_layout_analyzers.sh``). Override via ``--checkpoint``.

Classes in the DocLayout-YOLO DocStructBench checkpoint (for reference):

    0  title             1  plain text        2  abandon
    3  figure            4  figure_caption    5  table
    6  table_caption     7  table_footnote    8  isolate_formula
    9  formula_caption
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from doclayout_yolo import YOLOv10

from src.utils.utils import REPO_ROOT


DEFAULT_CHECKPOINT = (
    f"{REPO_ROOT}/models/DocLayout-YOLO-DocStructBench/"
    f"doclayout_yolo_docstructbench_imgsz1024.pt"
)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="DocLayout-YOLO layout analysis.")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory of raw images.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write per-image layout results.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"Path to DocLayout-YOLO .pt checkpoint (default: {DEFAULT_CHECKPOINT}).")
    parser.add_argument("--image_size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.2)
    return parser.parse_args()


def _list_images(input_dir: str) -> List[str]:
    files = []
    for entry in sorted(os.listdir(input_dir)):
        if os.path.splitext(entry)[1].lower() in IMG_EXTS:
            files.append(entry)
    return files


def _clean_boxes(boxes, class_names) -> List[Dict]:
    """Convert raw DocLayout-YOLO boxes → cleaned, deduped, containment-filtered list."""
    boxes_list: List[Dict] = []
    for idx, box in enumerate(boxes):
        cls_idx = int(box.cls[0])
        xyxy_raw = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy_raw)
        boxes_list.append({
            "idx": idx,
            "class": class_names[cls_idx],
            "confidence": float(box.conf[0]),
            "xyxy": [x1, y1, x2, y2],
        })

    # Sort top-to-bottom, left-to-right.
    boxes_list.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))

    # Deduplicate exact-same xyxy keys — keep the box with the smaller class id.
    class_name_to_id = {v: k for k, v in class_names.items()}
    seen: Dict[tuple, Dict] = {}
    for b in boxes_list:
        key = tuple(b["xyxy"])
        if key not in seen or class_name_to_id[b["class"]] < class_name_to_id[seen[key]["class"]]:
            seen[key] = b
    unique = sorted(seen.values(), key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
    for i, b in enumerate(unique):
        b["idx"] = i

    # Containment filtering — drop boxes fully contained in another, with the
    # carve-out that any non-figure box may live inside a figure box.
    final: List[Dict] = []
    for i, bi in enumerate(unique):
        x1i, y1i, x2i, y2i = bi["xyxy"]
        contained = False
        for j, bj in enumerate(unique):
            if i == j:
                continue
            x1j, y1j, x2j, y2j = bj["xyxy"]
            if x1j <= x1i and y1j <= y1i and x2j >= x2i and y2j >= y2i:
                if bi["class"] != "figure" and bj["class"] != "figure":
                    contained = True
                    break
                if bi["class"] == "figure" and bj["class"] == "figure":
                    contained = True
                    break
        if not contained:
            final.append(bi)
    return final


def main():
    args = parse_arguments()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory not found: {args.input_dir}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"DocLayout-YOLO checkpoint not found: {args.checkpoint}\n"
            f"Run `./models/download_layout_analyzers.sh --only doclayout` first."
        )

    image_files = _list_images(args.input_dir)
    if not image_files:
        raise ValueError(f"No images found in {args.input_dir}")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[layout/doclayout-yolo] device={device}  checkpoint={args.checkpoint}")
    model = YOLOv10(args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)

    for img_file in tqdm(image_files):
        img_path = os.path.join(args.input_dir, img_file)
        det_res = model.predict(img_path, imgsz=args.image_size,
                                conf=args.confidence, device=device)
        if not det_res:
            continue
        res = det_res[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"    skip (cv2 cannot read): {img_path}")
            continue

        final_boxes = _clean_boxes(res.boxes, res.names)

        basename = os.path.splitext(img_file)[0]
        out_subdir = os.path.join(args.output_dir, basename)
        os.makedirs(out_subdir, exist_ok=True)

        # Save per-class crops with stable naming.
        by_class: Dict[str, List[Dict]] = defaultdict(list)
        for b in final_boxes:
            by_class[b["class"]].append(b)
        for cls, boxes_cls in by_class.items():
            for sub_idx, b in enumerate(boxes_cls):
                x1, y1, x2, y2 = b["xyxy"]
                crop = image[y1:y2, x1:x2]
                fname = f"{b['idx']}_{cls}_{sub_idx}.jpg"
                cv2.imwrite(os.path.join(out_subdir, fname), crop)

        # Save the annotated full image.
        annotated = res.plot(pil=True, line_width=5, font_size=20)
        annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_subdir, f"{basename}_annotated.jpg"), annotated_bgr)

        # Save boxes JSON.
        with open(os.path.join(out_subdir, "boxes.json"), "w") as f:
            json.dump(final_boxes, f, indent=2)


if __name__ == "__main__":
    main()
