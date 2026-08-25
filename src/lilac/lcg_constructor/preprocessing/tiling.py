from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TileBox:
    x1: int
    y1: int
    x2: int
    y2: int


def estimate_subcomponent_count(image_bgr: np.ndarray) -> int:
    """Estimate rough component density via connected components on foreground."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_inv = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    h, w = image_bgr.shape[:2]
    min_area = max(25, int(h * w * 0.00008))
    count = 0
    for i in range(1, n_labels):  # 0 = background
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return count


def build_crowded_layout_tiles(
    width: int,
    height: int,
    overlap_ratio: float = 0.12,
    max_tiles: int = 6,
) -> List[TileBox]:
    """Build a crowded-layout tile grid with overlap."""
    if width <= 0 or height <= 0:
        return []
    aspect_ratio = height / float(width)
    if aspect_ratio >= 2.2:
        rows, cols = 4, 1
    elif aspect_ratio >= 1.6:
        rows, cols = 3, 1
    else:
        rows, cols = 2, 2

    while rows * cols > max_tiles and rows > 1:
        rows -= 1
    while rows * cols > max_tiles and cols > 1:
        cols -= 1

    step_w = max(1, width // cols)
    step_h = max(1, height // rows)
    ox = int(step_w * overlap_ratio)
    oy = int(step_h * overlap_ratio)

    tiles: List[TileBox] = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, c * step_w - ox)
            y1 = max(0, r * step_h - oy)
            x2 = min(width, (c + 1) * step_w + ox)
            y2 = min(height, (r + 1) * step_h + oy)
            if x2 > x1 and y2 > y1:
                tiles.append(TileBox(x1, y1, x2, y2))
    return tiles


def normalize_box(xyxy: Sequence[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def union_boxes(boxes: List[Dict], iou_threshold: float = 0.75) -> List[Dict]:
    """Union tiled detections by class + high-IoU overlap (keep higher confidence)."""
    kept: List[Dict] = []
    for box in sorted(boxes, key=lambda b: float(b.get("confidence", 0.0)), reverse=True):
        merged = False
        for existing in kept:
            if existing.get("class") != box.get("class"):
                continue
            if iou_xyxy(existing["xyxy"], box["xyxy"]) >= iou_threshold:
                merged = True
                break
        if not merged:
            kept.append(box)
    kept.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
    for idx, b in enumerate(kept):
        b["idx"] = idx
    return kept


def filter_contained_boxes(boxes: List[Dict]) -> List[Dict]:
    """Remove fully-contained boxes with the same carve-out used in existing pipeline."""
    final: List[Dict] = []
    for i, bi in enumerate(boxes):
        x1i, y1i, x2i, y2i = bi["xyxy"]
        contained = False
        for j, bj in enumerate(boxes):
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
    for idx, b in enumerate(final):
        b["idx"] = idx
    return final


def decide_tiling_with_mllm(
    image_path: str,
    estimated_subcomponents: int,
    aspect_ratio: float,
    qwen_model,
    min_subcomponents_hint: int,
    min_aspect_ratio_hint: float,
) -> Dict:
    """Use a multimodal LLM to decide if tiled detection is needed."""
    prompt = (
        "You are deciding if document-page object detection should run with tiling.\n"
        f"Estimated subcomponents: {estimated_subcomponents}\n"
        f"Page height_to_width_ratio: {aspect_ratio:.4f}\n"
        f"Reference crowded threshold: subcomponents >= {min_subcomponents_hint}, "
        f"ratio >= {min_aspect_ratio_hint:.2f}\n"
        "Return ONLY strict JSON:\n"
        '{"tile": true|false, "reason": "<15 words>"}'
    )
    outputs = qwen_model.infer(
        [{"text": prompt, "images": [image_path]}],
        batch_size=1,
        max_tokens=96,
    )
    raw = outputs[0].strip() if outputs else ""
    if not raw:
        raise ValueError(f"Adaptive-tiling decision returned empty output for {image_path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _extract_first_json_object(raw)

    tile = bool(data.get("tile", False))
    reason = str(data.get("reason", "")).strip()
    return {
        "tile": tile,
        "reason": reason,
        "raw_response": raw,
        "estimated_subcomponents": estimated_subcomponents,
        "height_to_width_ratio": aspect_ratio,
    }


def _extract_first_json_object(text: str) -> Dict:
    stack: List[str] = []
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if not stack:
                continue
            stack.pop()
            if not stack and start >= 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    raise ValueError(f"Could not parse JSON object from LLM output: {text[:160]!r}")


def load_tiling_decisions(path: str | None) -> Dict[str, Dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tiling decisions file not found: {path}")
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Tiling decisions must be dict[str, dict], got {type(data)}")
    return data

def main():
    img_path = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/test/37033.jpeg"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    est = estimate_subcomponent_count(img)
    ratio = h / w
    tiles = build_crowded_layout_tiles(width=w, height=h, overlap_ratio=0.12, max_tiles=6)

    print("estimated_subcomponents:", est)
    print("height_to_width_ratio:", ratio)
    print("num_tiles:", len(tiles))
    print("tiles:", tiles)

if __name__ == "__main__":
    main()