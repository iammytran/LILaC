#!/usr/bin/env python3
"""
Adapter — DocLayout-YOLO layout output → LILaC parsed_documents schema.

Routes detected regions to the correct LILaC slot by dla_type, matching the
shipped MP-DocVQA convention:

  figure                                            -> subimage  (caption.text = Qwen-VL summary)
  plain text / title / abandon / figure_caption /   -> sentence  (text = Qwen-VL OCR verbatim)
  table_caption / table_footnote /
  isolate_formula / formula_caption
  table                                             -> table_segment (text = Qwen-VL OCR verbatim)

Per-page id conventions follow shipped MP-DocVQA:
  i_1                page-level image (top-level)
  i_1_i<N>           figure subimage
  i_1_p<N>           sentence (per-page global counter across text-like regions)
  i_1_t<N>           table_segment

Pipeline (resumable end-to-end):
  1. Walk per-page boxes.json, crop each region, build skeleton parsed_documents
     with empty text/caption fields. Save per-region crops under
     image_components_sub/<doc_id>/<crop_filename>.
  2. Group crops by destination prompt:
        - figure crops: summary prompt
        - text-class + table crops: OCR-verbatim prompt
     Run Qwen-VL caption_images for each group; per-crop .txt sidecars are
     written next to the crops.
  3. Merge sidecars back into parsed_documents:
        - figure → subimage[*].caption.text
        - text-class → sentence[*].text
        - table → table_segment[*].text
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import caption_images


# DocLayout-YOLO DocStructBench class id → name (from layout_doclayout_yolo.py):
#   0 title, 1 plain text, 2 abandon, 3 figure, 4 figure_caption,
#   5 table, 6 table_caption, 7 table_footnote,
#   8 isolate_formula, 9 formula_caption

FIGURE_CLASSES = {"figure"}
TABLE_CLASSES = {"table"}
# Everything else (text-like) becomes sentence. List exhaustively for safety —
# DocLayoutYOLO outputs may use space- or underscore-separated multi-word class names.
TEXT_LIKE_CLASSES = {
    "plain text", "plain_text",
    "title",
    "abandon",
    "figure_caption", "figure caption",
    "table_caption", "table caption",
    "table_footnote", "table footnote",
    "isolate_formula", "isolate formula",
    "formula_caption", "formula caption",
}

SUMMARY_PROMPT = (
    "Extract any text visible in this image region verbatim. "
    "If no text is present, give a brief one-sentence (<15 words) factual "
    "description of the visual content. Be terse — no preamble, no commentary."
)
TITLE_PROMPT = (
    "Generate a concise descriptive title (under 20 words) for this document "
    "page. Capture the main subject. Output only the title text — no quotes, "
    "no preamble, no commentary."
)
OCR_PROMPT = (
    "Extract all the text content visible in this image region verbatim. "
    "Output only the raw text, preserving line breaks where appropriate. "
    "Do not add any descriptions or commentary."
)


def _crop_box(page_img_path: str, xyxy: List[int], out_path: Path) -> bool:
    """Crop xyxy region from page image; return True if a non-empty crop was written."""
    if out_path.exists():
        return True
    img = cv2.imread(page_img_path)
    if img is None:
        print(f"[crop] cv2 cannot read {page_img_path}")
        return False
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img[y1:y2, x1:x2])
    return out_path.exists()


def _route_box(cls: str) -> str:
    """Return target slot for a YOLO class label: 'subimage' | 'sentence' | 'table_segment' | 'skip'."""
    if cls in FIGURE_CLASSES:
        return "subimage"
    if cls in TABLE_CLASSES:
        return "table_segment"
    if cls in TEXT_LIKE_CLASSES:
        return "sentence"
    # Unknown class — default to sentence (DocLayoutYOLO's classes are all
    # text-or-figure-or-table, so an unrecognized label is most likely a text variant).
    return "sentence"


def build_parsed_documents(
    layout_dir: Path,
    images_dir: Path,
    parsed_documents_out_dir: Path,
    crops_out_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Build skeleton parsed_documents + crops. Returns (summary_crops, ocr_crops)."""
    parsed_documents_out_dir.mkdir(parents=True, exist_ok=True)
    crops_out_dir.mkdir(parents=True, exist_ok=True)
    summary_crops: List[Path] = []
    ocr_crops: List[Path] = []

    page_dirs = sorted([p for p in layout_dir.iterdir() if p.is_dir()])
    print(f"[adapter/yolo] {len(page_dirs)} page subdirs in {layout_dir}")

    for page_dir in page_dirs:
        doc_id = page_dir.name
        boxes_json = page_dir / "boxes.json"
        if not boxes_json.exists():
            continue
        with open(boxes_json) as f:
            boxes_raw = json.load(f)

        page_img_candidates = [images_dir / f"{doc_id}.jpg",
                               images_dir / f"{doc_id}.jpeg",
                               images_dir / f"{doc_id}.png"]
        page_img_path = next((str(p) for p in page_img_candidates if p.exists()), None)
        if page_img_path is None:
            print(f"[adapter/yolo] page image not found for {doc_id}")
            continue
        page_filename = Path(page_img_path).name

        # Sort boxes top-to-bottom, left-to-right for stable reading order.
        boxes = sorted(boxes_raw, key=lambda b: (b["xyxy"][1], b["xyxy"][0]))

        i_counter = 0  # figure subimage
        p_counter = 0  # sentence (text-class)
        t_counter = 0  # table_segment

        subimage_entries: Dict[str, Dict] = {}
        sentence_entries: Dict[str, Dict] = {}
        table_segment_entries: Dict[str, Dict] = {}
        id_sequence: List[str] = ["i_1"]

        for box in boxes:
            cls = box.get("class", "")
            slot = _route_box(cls)

            if slot == "subimage":
                i_counter += 1
                cid = f"i_1_i{i_counter}"
                crop_fname = f"{doc_id}___i_{i_counter}.jpg"
            elif slot == "sentence":
                p_counter += 1
                cid = f"i_1_p{p_counter}"
                crop_fname = f"{doc_id}___p_{p_counter}.jpg"
            elif slot == "table_segment":
                t_counter += 1
                cid = f"i_1_t{t_counter}"
                crop_fname = f"{doc_id}___t_{t_counter}.jpg"
            else:
                continue

            crop_path = crops_out_dir / doc_id / crop_fname
            if not _crop_box(page_img_path, box["xyxy"], crop_path):
                if slot == "subimage":      i_counter -= 1
                elif slot == "sentence":    p_counter -= 1
                elif slot == "table_segment": t_counter -= 1
                continue

            common_meta = {
                "dla_idx": box.get("idx"),
                "dla_type": cls,
            }
            if slot == "subimage":
                subimage_entries[cid] = {
                    **common_meta,
                    "filename": crop_fname,
                    "caption": {"text": "", "edges": []},
                }
                summary_crops.append(crop_path)
            elif slot == "sentence":
                sentence_entries[cid] = {
                    **common_meta,
                    "text": "",
                    "edges": [],
                }
                ocr_crops.append(crop_path)
            else:  # table_segment
                table_segment_entries[cid] = {
                    **common_meta,
                    "text": "",
                    "edges": [],
                }
                ocr_crops.append(crop_path)

            id_sequence.append(cid)

        parsed_doc = {
            "title": doc_id,
            "hierarchy": {},
            "id_sequence": id_sequence,
            "header": {},
            "text": {},
            "table": {},
            "image": {
                "i_1": {
                    "filename": page_filename,
                    "caption": {"text": "", "edges": []},
                }
            },
            "sentence": sentence_entries,
            "proposition": {},
            "table_segment": table_segment_entries,
            "subimage": subimage_entries,
            "id_to_html": {},
        }
        with open(parsed_documents_out_dir / f"{doc_id}.json", "w") as f:
            json.dump(parsed_doc, f, indent=4)

    return summary_crops, ocr_crops


def _run_caption_pass(crops: List[Path], prompt: str, max_tokens: int,
                     num_gpus, label: str) -> None:
    if not crops:
        print(f"[adapter/yolo] no {label} crops to caption")
        return
    img_paths = [str(p) for p in crops]
    out_paths = [str(p.with_suffix(".txt")) for p in crops]
    caption_images(
        image_paths=img_paths,
        output_paths=out_paths,
        prompt=prompt,
        max_tokens=max_tokens,
        num_gpus=num_gpus,
    )


def merge_captions(parsed_documents_out_dir: Path, crops_out_dir: Path) -> None:
    """Read per-crop .txt sidecars and merge into the right slot of parsed_documents."""
    files = sorted(parsed_documents_out_dir.glob("*.json"))
    merged = {"subimage": 0, "sentence": 0, "table_segment": 0}
    missing = {"subimage": 0, "sentence": 0, "table_segment": 0}

    for pd_path in files:
        with open(pd_path) as f:
            pd = json.load(f)
        doc_id = pd_path.stem
        crops_dir = crops_out_dir / doc_id
        any_updated = False

        slot_letter = {"subimage": "i", "sentence": "p", "table_segment": "t"}
        for slot, letter in slot_letter.items():
            for cid, entry in pd.get(slot, {}).items():
                if slot == "subimage":
                    have = (entry.get("caption") or {}).get("text", "")
                else:
                    have = entry.get("text", "")
                if have:
                    continue
                # cid is `i_1_<letter><N>`, e.g. `i_1_p3`.
                tail = cid.rsplit("_", 1)[1]              # "p3"
                num = tail.lstrip("ipt")                   # "3"
                stem = f"{doc_id}___{letter}_{num}"
                sidecar = crops_dir / f"{stem}.txt"
                if not sidecar.exists():
                    missing[slot] += 1
                    continue
                text = sidecar.read_text(encoding="utf-8").strip()
                if slot == "subimage":
                    entry["caption"]["text"] = text
                else:
                    entry["text"] = text
                merged[slot] += 1
                any_updated = True

        if any_updated:
            with open(pd_path, "w") as f:
                json.dump(pd, f, indent=4)

    print(f"[adapter/yolo] merged captions: {merged}; missing sidecars: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--crops_out", required=True)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--num_gpus", type=int, default=0)
    ap.add_argument("--skip_caption", action="store_true",
                    help="Skip Qwen-VL caption passes (skeleton only)")
    args = ap.parse_args()

    layout_dir = Path(args.layout)
    images_dir = Path(args.images)
    pd_out = Path(args.output)
    crops_out = Path(args.crops_out)

    summary_crops, ocr_crops = build_parsed_documents(layout_dir, images_dir, pd_out, crops_out)
    print(f"[adapter/yolo] crops to caption — summary: {len(summary_crops)}, OCR: {len(ocr_crops)}")

    num_gpus = args.num_gpus or None

    if not args.skip_caption:
        # OCR pass first (text + table — typically the bulk).
        _run_caption_pass(ocr_crops, OCR_PROMPT, args.max_tokens, num_gpus, "OCR")
        # Then summary pass (figures only).
        _run_caption_pass(summary_crops, SUMMARY_PROMPT, args.max_tokens, num_gpus, "summary")
        merge_captions(pd_out, crops_out)


if __name__ == "__main__":
    main()
