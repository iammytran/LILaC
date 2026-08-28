"""Adaptive page tiling and DocLayout-YOLO result merging."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DECISION_PROMPT = (
    "Estimate how many visually distinct text, table, formula, or figure "
    "components are on this page. Decide whether splitting the page into "
    "overlapping tiles would improve small-region detection. Return ONLY JSON "
    'in this exact shape: {"tile": true|false, "estimated_subcomponents": 0}. '
    "Use tile=true for crowded pages or pages where small regions are likely "
    "to be missed. Consider the page height-to-width ratio."
)


def _parse_decision(raw: str) -> Tuple[bool, int]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"LLM tiling decision is not JSON: {raw!r}")
    decision = json.loads(raw[start:end + 1])
    if not isinstance(decision.get("tile"), bool):
        raise ValueError(f"LLM tiling decision has invalid tile value: {raw!r}")
    estimate = int(decision.get("estimated_subcomponents", 0))
    return decision["tile"], max(0, estimate)


def decide_tiling(image_path: Path, model) -> Tuple[bool, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read page image for tiling decision: {image_path}")
    height, width = image.shape[:2]
    raw = model.infer(
        [{"text": f"{DECISION_PROMPT}\nPage aspect ratio: {height / max(width, 1):.3f}.",
          "images": [str(image_path)]}],
        batch_size=1,
        max_tokens=96,
    )
    return _parse_decision(raw[0] if raw else "")


# def _tile_grid(height: int, width: int, estimated: int, min_subcomponents: int,
#                min_aspect_ratio: float, max_tiles: int) -> Tuple[int, int]:
#     print(f"min_subcomponents: {min_subcomponents}")
#     print(f"min_aspect_ratio: {min_aspect_ratio}")
#     print(f"max_tiles: {max_tiles}")
#     aspect = height / max(width, 1)
#     requested = max(2, math.ceil(estimated / max(min_subcomponents, 1)))
#     # Nhánh 1: Nếu ảnh quá dài (cao >> rộng) -> trả về (N, 1)
#     if aspect >= min_aspect_ratio:
#         return min(max_tiles, requested), 1

#     # Nhánh 2: Nếu ảnh quá rộng (rộng >> cao) -> trả về (1, N)
#     if aspect <= 1 / min_aspect_ratio:
#         return 1, min(max_tiles, requested)

#     # Nhánh 3: Chỉ khi không rơi vào 2 nhánh trên mới chạy tới đây
#     side = min(max_tiles, max(2, math.ceil(math.sqrt(requested))))
#     return side, side

def _tile_grid(
    height: int,
    width: int,
    estimated: int,
    min_subcomponents: int,
    min_aspect_ratio: float,
    max_tiles: int,
) -> Tuple[int, int]:
    aspect = height / max(width, 1)

    # 1. Tính toán trần động: Nếu ảnh siêu dài (aspect >= 4.0), nới trần lên tối đa 6 hoặc 8 tiles
    effective_max = max_tiles * 2 if (aspect >= 4.0 or aspect <= 1 / 4.0) else max_tiles
    
    # 2. Số lượng tile ước tính dựa trên mật độ thành phần
    requested = max(2, math.ceil(estimated / max(min_subcomponents, 1)))

    # Nhánh 1: Ảnh dài dọc (N hàng, 1 cột)
    if aspect >= min_aspect_ratio:
        return min(effective_max, requested), 1

    # Nhánh 2: Ảnh rộng ngang (1 hàng, N cột)
    if aspect <= 1 / min_aspect_ratio:
        return 1, min(effective_max, requested)

    # Nhánh 3: Ảnh vuông / cân đối (khống chế tổng số tiles <= max_tiles)
    if requested <= 3:
        return requested, 1
    return 2, 2

def prepare_tiled_inputs(
    images_dir: Path,
    staging_dir: Path,
    manifest_path: Path,
    min_subcomponents: int = 16,
    min_aspect_ratio: float = 1.3,
    max_tiles: int = 4,
    overlap_ratio: float = 0.12
) -> Dict[str, Dict]:
    """Stage original pages or overlapping tiles for the layout analyzer."""
    # from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL

    staging_dir.mkdir(parents=True, exist_ok=True)
    # model = Qwen2_5_VL()
    manifest: Dict[str, Dict] = {}
    for source in sorted(p for p in images_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTENSIONS):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Cannot read page image: {source}")
        height, width = image.shape[:2]
        # tìm file tile_decision
        image_name = source.stem
        tile_decision_json_path = Path("debug/tiling_decision") / f"{image_name}.json"
        tile_decision_data = ""
        with open(tile_decision_json_path, 'r') as file:
            tile_decision_data = file.read()
        should_tile, estimated = _parse_decision(tile_decision_data)
        if not should_tile:
            destination = staging_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            manifest[source.stem] = {"source": source.name, "tiles": [source.name]}
            continue

        rows, cols = _tile_grid(
            height, width, estimated, min_subcomponents, min_aspect_ratio, max_tiles
        )

        step_w = width / cols
        step_h = height / rows

        overlap_x = max(1, int(step_w * overlap_ratio))
        overlap_y = max(1, int(step_h * overlap_ratio))

        tile_names = []
        for row in range(rows):
            for col in range(cols):
                x1 = max(0, round(col * width / cols) - overlap_x)
                y1 = max(0, round(row * height / rows) - overlap_y)
                x2 = min(width, round((col + 1) * width / cols) + overlap_x)
                y2 = min(height, round((row + 1) * height / rows) + overlap_y)
                tile_name = f"{source.stem}__tile_{row}_{col}{source.suffix.lower()}"
                if not cv2.imwrite(str(staging_dir / tile_name), image[y1:y2, x1:x2]):
                    raise OSError(f"Cannot write tile: {staging_dir / tile_name}")
                tile_names.append(tile_name)
                manifest[tile_name] = {
                    "source": source.name,
                    "source_stem": source.stem,
                    "offset": [x1, y1],
                    "size": [x2 - x1, y2 - y1],
                }
        manifest[source.stem] = {"source": source.name, "tiles": tile_names}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def merge_tile_boxes(layout_dir: Path, manifest_path: Path) -> None:
    """Union tile detections into one global boxes.json per source page."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged: Dict[str, List[Dict]] = {}
    filtered_manifest = {k: v for k, v in manifest.items() if "tiles" in v}
    # print(filtered_manifest['10027'])
    for page_stem, page in filtered_manifest.items():
        if "source" not in page:
            continue
        boxes: List[Dict] = []
        # filtered_page = {k: v for k, v in manifest.items() if "tiles" in v}
        # print(f"page: {filtered_page}")
        for tile_name in page["tiles"]:
            tile_dir = layout_dir / Path(tile_name).stem
            boxes_path = tile_dir / "boxes.json"
            if not boxes_path.exists():
                continue
            tile_meta = manifest.get(tile_name, {})
            ox, oy = tile_meta.get("offset", [0, 0])
            for box in json.loads(boxes_path.read_text(encoding="utf-8")):
                x1, y1, x2, y2 = box["xyxy"]
                box = dict(box)
                box["xyxy"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                boxes.append(box)
        # Overlap from neighboring tiles is the same detection, not a new one.
        unique: List[Dict] = []
        for box in sorted(boxes, key=lambda b: b.get("confidence", 0), reverse=True):
            x1, y1, x2, y2 = box["xyxy"]
            duplicate = False
            for kept in unique:
                if box.get("class") != kept.get("class"):
                    continue
                a1, b1, a2, b2 = kept["xyxy"]
                ix1, iy1, ix2, iy2 = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = (x2 - x1) * (y2 - y1)
                kept_area = (a2 - a1) * (b2 - b1)
                if inter / max(min(area, kept_area), 1) >= 0.5:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(box)
        unique.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
        for idx, box in enumerate(unique):
            box["idx"] = idx
        output_dir = layout_dir / page_stem
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "boxes.json").write_text(json.dumps(unique, indent=2), encoding="utf-8")

if __name__ == "__main__":
    json_path = "debug/tiling_decision/10022.json"
    json_data = ""
    with open(json_path, 'r') as file:
        json_data = file.read()
    # json_str = json.dumps(json_data)
    a, b = _parse_decision(json_data)
    print(a)
    print(b) 