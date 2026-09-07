from pathlib import Path
import json
import shutil
import os
import subprocess
from typing import Dict, List, Tuple
import math
import cv2
import tempfile

DATASET_FOLDER = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/test"
TILE_DECISION_FOLDER = "/Users/mytnguyen/Documents/LILaC/debug/tiling_decision"
MAX_TILING_DEPTH = 2
OUTPUT_FOLDER = ""
MIN_SUBCOMPONENTS = 3
MIN_ASPECT_RATIO = 1.3
MAX_TILES = 4
OVERLAP_RATIO = 0.12

def check_content_list_tile_depth_1():
    mineru_depth_1_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs_tile_depth_1_ocr"

    return

def _box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(b1: list[int], b2: list[int]) -> int:
    x_inter = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_inter = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    return x_inter * y_inter


def _union_box(b1: list[int], b2: list[int]) -> list[int]:
    return [
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    ]

def _ios(b1: list[int], b2: list[int]) -> float:
    """Tỷ lệ diện tích giao nhau trên diện tích của box nhỏ hơn (Intersection over Smaller)."""
    inter = _intersection_area(b1, b2)
    smaller = min(_box_area(b1), _box_area(b2))
    return inter / smaller if smaller > 0 else 0.0

def clean_and_merge_content_list(
    content_list_path: Path,
    max_gap: int = 15,
    delete_duplicate_images: bool = True,
) -> list[dict]:
    content_list_path = Path(content_list_path)
    with content_list_path.open("r", encoding="utf-8") as f:
        content_list = json.load(f)

    if not isinstance(content_list, list):
        raise TypeError(
            f"Content list must be a JSON array: {content_list_path}"
        )

    # ----------------------------------------------------
    # BƯỚC 1 & 2: PHÂN LOẠI & LOẠI BỎ BOX RỖNG (text/header/footer)
    # ----------------------------------------------------
    raw_images: list[dict] = []
    text_items: list[dict] = []
    other_items: list[dict] = []

    text_like_types = {"text", "header", "footer"}

    for item in content_list:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("bbox"), list)
            or len(item["bbox"]) != 4
        ):
            continue

        item_type = item.get("type")

        # Loại bỏ box văn bản rỗng
        if item_type in text_like_types:
            if not str(item.get("text", "")).strip():
                continue
            text_items.append(dict(item))
        elif item_type == "image":
            raw_images.append(dict(item))
        else:
            other_items.append(dict(item))

    # ----------------------------------------------------
    # BƯỚC 1 (TIẾP): KHỬ TRÙNG LẶP IMAGE
    # ----------------------------------------------------
    # Sort theo diện tích giảm dần để giữ lại ảnh có bounding box to, đầy đủ nhất
    raw_images.sort(
        key=lambda it: _box_area(it["bbox"]),
        reverse=True,
    )

    cleaned_images: list[dict] = []
    images_to_delete_on_disk: list[str] = []

    for img in raw_images:
        is_duplicate = False
        for kept in cleaned_images:
            if (
                kept.get("page_idx") == img.get("page_idx")
                and _ios(kept["bbox"], img["bbox"]) > 0.6
            ):
                # Đã có ảnh lớn hơn bao quát vùng này -> đánh dấu trùng lặp
                is_duplicate = True
                if delete_duplicate_images and "img_path" in img:
                    images_to_delete_on_disk.append(img["img_path"])
                break

        if not is_duplicate:
            cleaned_images.append(img)

    # Xoá các file ảnh bị trùng khỏi ổ cứng nếu có cấu hình
    if delete_duplicate_images and images_to_delete_on_disk:
        base_dir = content_list_path.parent
        for rel_path in images_to_delete_on_disk:
            img_file = base_dir / rel_path
            if img_file.is_file():
                try:
                    img_file.unlink()
                except OSError:
                    pass

    # ----------------------------------------------------
    # BƯỚC 3: KHỬ TRÙNG & GỘP TEXT (Substring matching & Union bbox)
    # ----------------------------------------------------
    text_items.sort(
        key=lambda it: (it.get("page_idx", 0), it["bbox"][1], it["bbox"][0])
    )

    merged_texts: list[dict] = []
    for current in text_items:
        cx1, cy1, cx2, cy2 = current["bbox"]
        c_text = current["text"].strip()
        c_area = _box_area(current["bbox"])
        is_merged_or_dropped = False

        for prev in reversed(merged_texts):
            if prev.get("page_idx") != current.get("page_idx"):
                continue

            px1, py1, px2, py2 = prev["bbox"]
            p_text = prev["text"].strip()
            p_area = _box_area(prev["bbox"])

            ios = _ios(prev["bbox"], current["bbox"])
            inter = _intersection_area(prev["bbox"], current["bbox"])

            # 3A: TRÙNG LẶP / OVERLAP LỚN (IOS > 0.65)
            if ios > 0.65:
                # Substring matching: Giữ lại text dài hơn, mở rộng bbox theo union
                if c_text.lower() in p_text.lower():
                    prev["bbox"] = _union_box(prev["bbox"], current["bbox"])
                    is_merged_or_dropped = True
                    break
                elif p_text.lower() in c_text.lower():
                    prev["text"] = c_text
                    prev["bbox"] = _union_box(prev["bbox"], current["bbox"])
                    if "text_level" in current:
                        prev["text_level"] = current["text_level"]
                    is_merged_or_dropped = True
                    break

                # OCR Hallucination: Một box lọt thỏm sâu (> 80%) vào box còn lại
                if c_area > 0 and (inter / c_area) > 0.8:
                    is_merged_or_dropped = True
                    break
                elif p_area > 0 and (inter / p_area) > 0.8:
                    prev.update(current)
                    is_merged_or_dropped = True
                    break

            # 3B: GHÉP CÙNG DÒNG (Ngang hàng và sát cạnh nhau)
            v_overlap = min(py2, cy2) - max(py1, cy1)
            h_gap = max(cx1 - px2, px1 - cx2, 0)
            if v_overlap > 8 and h_gap <= max_gap:
                if (
                    prev.get("type") == current.get("type")
                    and prev.get("text_level") == current.get("text_level")
                ):
                    if px1 <= cx1:
                        prev["text"] = f"{p_text} {c_text}"
                    else:
                        prev["text"] = f"{c_text} {p_text}"
                    prev["bbox"] = _union_box(prev["bbox"], current["bbox"])
                    is_merged_or_dropped = True
                    break

        if not is_merged_or_dropped:
            merged_texts.append(current)

    # ----------------------------------------------------
    # BƯỚC 4: SẮP XẾP TOÀN BỘ THEO THỨ TỰ ĐỌC (READING ORDER DỌC)
    # ----------------------------------------------------
    final_results = cleaned_images + other_items + merged_texts
    final_results.sort(
        key=lambda it: (
            it.get("page_idx", 0),
            it["bbox"][1],  # Y1 từ trên xuống dưới
            it["bbox"][0],  # X1 từ trái qua phải
        )
    )

    # ----------------------------------------------------
    # GHI ĐÈ FILE AN TOÀN (ATOMIC WRITE)
    # ----------------------------------------------------
    content_list_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=content_list_path.parent,
        delete=False,
        encoding="utf-8",
        suffix=".tmp",
    ) as tmp_file:
        json.dump(final_results, tmp_file, indent=4, ensure_ascii=False)
        tmp_path = Path(tmp_file.name)

    tmp_path.replace(content_list_path)
    return final_results

def check_tiling_depth_1_results():
    manifest_json_file = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/tiled_input_depth_1/tiling_manifest_depth_1.json"
    with open(manifest_json_file, 'r', encoding="utf-8") as f:
        manifest = json.load(f)

    image_and_tiles = {}
    for image_source, image_tiles in manifest.items():
        if "tiles" in image_tiles:
            tiles = image_tiles["tiles"]
            image_and_tiles[image_source] = tiles

    # đi qua từng image (và các tiles của nó)
    # tìm tile_results trong mineru_tiles_output folder: /Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs_tile_depth_1_ocr 
    # lấy các content_list ra
    # lấy từng item trong các content list ra, tìm global coordinates dựa trên manifest.json
    # add từng item đó với global coordinates vào file content_list mới, mà file nay sẽ nằm trong folder cùng tên với image được tile ra
    # Bỏ folder trên vào folder /Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs
    mineru_tiles_output = Path("/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs_tile_depth_1_ocr")
    mineru_outputs = Path("/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs")

    if not mineru_tiles_output.is_dir():
        raise FileNotFoundError(f"MinerU tile output folder not found: {mineru_tiles_output}")
    mineru_outputs.mkdir(parents=True, exist_ok=True)

    for image_source, tiles in image_and_tiles.items():
        image_output_dir = mineru_outputs / image_source / "ocr"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        merged_content_list = []
        source_name = manifest[image_source].get("source", f"{image_source}.jpeg")
        source_image_path = Path(DATASET_FOLDER) / source_name
        if not source_image_path.is_file():
            raise FileNotFoundError(f"Original image not found: {source_image_path}")
        shutil.copy2(source_image_path, image_output_dir / source_name)

        for tile_name in tiles:
            tile_stem = Path(tile_name).stem
            tile_output_dir = mineru_tiles_output / tile_stem
            content_list_candidates = sorted(
                tile_output_dir.rglob(f"{tile_stem}_content_list.json")
            )
            if not content_list_candidates:
                raise FileNotFoundError(
                    f"Content list not found for tile {tile_name} under {tile_output_dir}"
                )

            content_list_path = content_list_candidates[0]
            shutil.copy2(
                content_list_path,
                image_output_dir / content_list_path.name,
            )
            with content_list_path.open("r", encoding="utf-8") as f:
                tile_content_list = json.load(f)
            if not isinstance(tile_content_list, list):
                raise TypeError(f"Content list must be a JSON array: {content_list_path}")

            tile_manifest = manifest.get(tile_name)
            if not isinstance(tile_manifest, dict):
                raise KeyError(f"Manifest entry not found for tile {tile_name}")
            offset = tile_manifest.get(
                "global_offset",
                tile_manifest.get("local_offset", tile_manifest.get("offset")),
            )
            if not isinstance(offset, list) or len(offset) != 2:
                raise ValueError(f"Invalid global offset for tile {tile_name}: {offset!r}")
            offset_x, offset_y = offset
            tile_width, tile_height = tile_manifest["size"]

            for item in tile_content_list:
                if not isinstance(item, dict):
                    raise TypeError(f"Content list item must be an object: {content_list_path}")
                merged_item = dict(item)
                bbox = merged_item.get("bbox")
                if bbox is not None:
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise ValueError(
                            f"Invalid bbox in {content_list_path}: {bbox!r}"
                        )
                    # MinerU content-list bboxes use a 0..1000 coordinate
                    # system for each tile, while the manifest stores pixel
                    # dimensions and offsets.
                    x_scale = tile_width / 1000
                    y_scale = tile_height / 1000
                    merged_item["bbox"] = [
                        round(bbox[0] * x_scale + offset_x),
                        round(bbox[1] * y_scale + offset_y),
                        round(bbox[2] * x_scale + offset_x),
                        round(bbox[3] * y_scale + offset_y),
                    ]

                img_path = merged_item.get("img_path")
                if isinstance(img_path, str):
                    source_image = content_list_path.parent / img_path
                    destination_image = image_output_dir / img_path
                    if not source_image.is_file():
                        raise FileNotFoundError(
                            f"Referenced image not found for {content_list_path}: {source_image}"
                        )
                    destination_image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_image, destination_image)

                merged_content_list.append(merged_item)

        output_content_list = image_output_dir / f"{image_source}_content_list.json"
        with output_content_list.open("w", encoding="utf-8") as f:
            json.dump(merged_content_list, f, indent=4, ensure_ascii=False)
        merged_content_list = clean_and_merge_content_list(output_content_list)
        print(
            f"[adaptive-tiling] merged {len(merged_content_list)} items "
            f"from {len(tiles)} tiles for {image_source}"
        )

    return

def _parse_decision(raw: str) -> Tuple[bool, int]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"LLM tiling decision is not JSON: {raw!r}")
    decision = json.loads(raw[start:end + 1])
    if not isinstance(decision.get("tile"), bool):
        raise ValueError(f"LLM tiling decision has invalid tile value: {raw!r}")
    estimate = int(decision.get("estimated_subcomponents", 0))
    return decision["tile"], max(0, estimate)

def prepare_tiled_inputs(
    images_dir: Path,
    staging_dir: Path,
    manifest_path: Path,):

    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict] = {}
    for source in images_dir.iterdir():
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Cannot read page image: {source}")
        height, width = image.shape[:2]
        image_name = source.stem
        # print(f"image_name: {image_name}")
        tile_decision_json_path = Path("debug/tiling_decision") / f"{image_name}.json"
        tile_decision_data = ""
        with open(tile_decision_json_path, 'r') as file:
            tile_decision_data = file.read()
        should_tile, estimated = _parse_decision(tile_decision_data)
        destination = staging_dir / source.name
        # if not destination.exists():
        #     shutil.copy2(source, destination)
        # manifest[source.stem] = {"source": source.name, "tiles": [source.name]}

        rows, cols = tile_grid(
            height, width, estimated
        )

        # if image_name == "10071":
        #     print(f"10071")
        #     print(f"height: {height}")
        #     print(f"width: {width}")

        step_w = width / cols
        step_h = height / rows

        overlap_x = max(1, int(step_w * OVERLAP_RATIO))
        overlap_y = max(1, int(step_h * OVERLAP_RATIO))

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
                # Since this is the first tiling level,
                # the parent is the original image.
                # Therefore local_offset == global_offset.
                local_offset = [x1, y1]
                global_offset = [x1, y1]

                manifest[tile_name] = {
                    "source": source.name,
                    "source_stem": source.stem,
                    # Offset relative to the parent image.
                    "local_offset": local_offset,

                    # Offset relative to the original infographic.
                    "global_offset": global_offset,
                    "size": [x2 - x1, y2 - y1],
                }
        manifest[source.stem] = {"source": source.name, "tiles": tile_names}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return

def tile_grid(
    height: int,
    width: int,
    estimated: int
) -> Tuple[int, int]:
    aspect = height / max(width, 1)

    # 1. Tính toán trần động: Nếu ảnh siêu dài (aspect >= 4.0), nới trần lên tối đa 6 hoặc 8 tiles
    effective_max = MAX_TILES * 2 if (aspect >= 4.0 or aspect <= 1 / 4.0) else MAX_TILES
    
    # 2. Số lượng tile ước tính dựa trên mật độ thành phần
    requested = max(2, math.ceil(estimated / max(MIN_SUBCOMPONENTS, 1)))

    # Nhánh 1: Ảnh dài dọc (N hàng, 1 cột)
    if aspect >= MIN_ASPECT_RATIO:
        return min(effective_max, requested), 1

    # Nhánh 2: Ảnh rộng ngang (1 hàng, N cột)
    if aspect <= 1 / MIN_ASPECT_RATIO:
        return 1, min(effective_max, requested)

    # Nhánh 3: Ảnh vuông / cân đối (khống chế tổng số tiles <= max_tiles)
    if requested <= 2:
        return 2, 1
    return 2, 2


def mainfn():
    should_tile_names = []
    should_not_tile_names = []
    should_tile_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/should_tile"
    should_not_tile_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/should_not_tile"
    os.makedirs(should_tile_folder, exist_ok=True)
    os.makedirs(should_not_tile_folder, exist_ok=True)
    for tile_decision in Path(TILE_DECISION_FOLDER).iterdir():
        tile_decision_json_path = f"{TILE_DECISION_FOLDER}/{tile_decision.name}"
        data = ""
        with open(tile_decision_json_path, 'r') as f:
            data = json.load(f)
        should_tile, estimated_subcomponents = data["tile"], data ["estimated_subcomponents"]
        image_path = f"{DATASET_FOLDER}/{tile_decision.stem}.jpeg"
        if should_tile:
            shutil.copy(image_path, should_tile_folder)
            should_tile_names.append(tile_decision.stem)
        # if not should_tile:
        #     import tempfile
        #     cmd_base = ["mineru", "-b", "pipeline", "-m", "auto", "-l", "el", "-p", "", "-o", ""]
        
        #     with tempfile.TemporaryDirectory(prefix="mineru_staged_") as staged_in:
        #         for fname in todo:
        #             src = os.path.abspath(os.path.join(args.input_dir, fname))
        #             dst = os.path.join(staged_in, fname)
        #             try:
        #                 os.symlink(src, dst)
        #             except FileExistsError:
        #                 pass
        
        #         cmd = cmd_base + ["-p", staged_in, "-o", args.output_dir]
        #         print(f"[layout/mineru] {' '.join(cmd[:6])} ... ({len(todo)} staged inputs)")
        #         # MinerU may emit per-doc errors for tricky pages (e.g. hybrid-auto-engine
        #         # tensor-dim crashes), but exits 0 for the batch. Don't `check=True` so
        #         # partial successes are kept; the caller can re-run for missed docs.
        #         subprocess.run(cmd, check=False)
        else:
            shutil.copy(image_path, should_not_tile_folder)
            should_not_tile_names.append(tile_decision.stem)

    # ## process những ảnh ko cần tile
    # source_mineru_outputs_folder = "/Users/mytnguyen/Documents/LILaC/artifacts/artifacts/InfoVQA/mineru_outputs_pipeline/test"
    # dest_mineru_outputs_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs"
    # # os.makedirs(mineru_outputs_folder, exist_ok=True)
    # # cmd = ["mineru", "-b", "pipeline", "-m", "auto", "-l", "el", "-p", should_tile_folder, "-o", mineru_outputs_folder]
    # # subprocess.run(cmd, check=False)
    # # giờ có sẵn mineru_outputs rồi nê copy paste qua
    # for name in should_not_tile_names:
    #     source_mineru_folder = f"{source_mineru_outputs_folder}/{name}"
    #     dest_mineru_folder = f"{dest_mineru_outputs_folder}/{name}"
    #     shutil.copytree(source_mineru_folder, dest_mineru_folder, dirs_exist_ok=True)

    # tile những hình mà cần tile
    tile_output_depth_1 = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/tiled_input_depth_1"
    manifest_depth_1 = f"{tile_output_depth_1}/tiling_manifest_depth_1.json"
    mineru_outputs_depth_1_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs_tile_depth_1_ocr"
    manifest = prepare_tiled_inputs(Path(should_tile_folder),
                                    Path(tile_output_depth_1),
                                    Path(manifest_depth_1))

    # file_path = Path("debug/tiles_not_processed.txt")

    # # Đọc file, tách theo dòng và loại bỏ dòng trống (nếu có)
    # items = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # # Đi qua từng item để xử lý
    # for item in items:
    #     tile_path = f"{tile_output_depth_1}/{item}.jpeg"
    #     cmd = ["mineru", "-b", "pipeline", "-m", "ocr", "-l", "el", "-p", tile_path, "-o", mineru_outputs_depth_1_folder]
    #     subprocess.run(cmd, check=False)

    # # get tile_decision
    # # if tile, tile it -> use mineru for each tile 
    #     # if parse tốt, merge kết quả của tile
    #     # if parse ko tốt, tile it tiếp -> use mineru
    #         # if parse tốt, merge kết quả của tile
    #         # if parse ko tốt, ...

    # # if not tile, use mineru pipeline
    return 

def check_image_size():
    # import cv2

    image = cv2.imread("/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/should_tile/10027.jpeg")
    if image is not None:
        height, width = image.shape[:2]
        print(f"Width: {width}, Height: {height}")
    else:
        print("Không thể đọc được ảnh!")

def tile_specific_image():
    image_name = "10388"
    image_path = f"/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/test/{image_name}.jpeg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read page image: {image_path}")
    tile_decision_json_path = Path("debug/tiling_decision") / f"{image_name}.json"
    tile_decision_data = ""
    with open(tile_decision_json_path, 'r') as file:
        tile_decision_data = file.read()
    # estimated = 30
    should_tile, estimated = _parse_decision(tile_decision_data)
    height, width = image.shape[:2]
    print(f"height: {height}")
    print(f"width: {width}")
    rows, cols = tile_grid(
        height, width, estimated
    )

    print(f"rows: {rows}")
    print(f"cols: {cols}")

    step_w = width / cols
    step_h = height / rows

    overlap_x = max(1, int(step_w * OVERLAP_RATIO))
    overlap_y = max(1, int(step_h * OVERLAP_RATIO))

    tile_names = []
    staging_dir = Path("/Users/mytnguyen/Documents/LILaC/debug/test_tiling")
    for row in range(rows):
        for col in range(cols):
            x1 = max(0, round(col * width / cols) - overlap_x)
            y1 = max(0, round(row * height / rows) - overlap_y)
            x2 = min(width, round((col + 1) * width / cols) + overlap_x)
            y2 = min(height, round((row + 1) * height / rows) + overlap_y)
            tile_name = f"{image_name}__tile_{row}_{col}.jpeg"
            print(tile_name)
            if not cv2.imwrite(str(staging_dir / tile_name), image[y1:y2, x1:x2]):
                raise OSError(f"Cannot write tile: {staging_dir / tile_name}")
            tile_names.append(tile_name)
            # Since this is the first tiling level,
            # the parent is the original image.
            # Therefore local_offset == global_offset.
            # local_offset = [x1, y1]
            # global_offset = [x1, y1]

            # manifest[tile_name] = {
            #     "source": source.name,
            #     "source_stem": source.stem,
            #     # Offset relative to the parent image.
            #     "local_offset": local_offset,

            #     # Offset relative to the original infographic.
            #     "global_offset": global_offset,
            #     "size": [x2 - x1, y2 - y1],
            # }
    return

from PIL import Image

def should_trigger_tiling(data, image_path, min_table_coverage=0.40):
    """
    Quy tắc:
    - Nếu detected_components_count thưa thớt (ví dụ <= 2):
        + NẾU trong đó có 1 component là 'table' to (chiếm >= 60% ảnh) -> KHÔNG TILE (False).
        + NGƯỢC LẠI (bị sót box, trượt layout, icon/text rời rạc bị miss) -> MỚI KÍCH HOẠT TILE (True).
    - Nếu components nhiều bình thường (> 2) -> KHÔNG CẦN TILE (False).
    """
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    total_img_area = img_w * img_h


    # 1. Lọc các component hợp lệ như bạn đang làm
    detected_components = [
        item for item in data 
        if item.get("type") not in ("header", "footer")
        and not (item.get("type") == "text" and not str(item.get("text", "")).strip())
    ]
    detected_components_count = len(detected_components)
    if image_path == "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/test/70529.jpeg":
        print(f"70529: {detected_components_count}")

    # 2. Nếu component dồi dào -> Xử lý bình thường, không cần tile
    if detected_components_count > MIN_SUBCOMPONENTS:
        return False, "Sufficient components detected."

    # 3. Xử lý trường hợp thưa thớt (count <= 2):
    # Kiểm tra xem có phải case "Cả ảnh là 1 table thật sự" hay không
    for item in detected_components:
        if item.get("type") == "table" or item.get("type") == "chart" or item.get("type") == "figure":
            # print(f"image có ít hơn min_components: {image_path}")
            
            bbox = item.get("bbox", [0, 0, 0, 0])  # [x1, y1, x2, y2]
            box_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            coverage_ratio = box_area / total_img_area

            # Nếu table chiếm trên 60% diện tích ảnh -> Bỏ qua tiling
            if coverage_ratio >= min_table_coverage:
                return False, f"Infographic is genuinely a full-page table ({coverage_ratio*100:.1f}% area). Do not tile."

    # Nếu thưa thớt mà KHÔNG PHẢI là 1 bảng to -> Chứng tỏ detector bị miss -> Cần tile
    return True, f"Sparse components ({detected_components_count} found) without a dominant table. Triggering tiling."

def main():
    should_tile = []
    should_not_tile = []
    image_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/test"
    should_tile_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/should_tile"
    should_not_tile_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/should_not_tile"
    Path(should_tile_folder).mkdir(parents=True, exist_ok=True)
    Path(should_not_tile_folder).mkdir(parents=True, exist_ok=True)
    mineru_outputs_folder = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/mineru_outputs_pipeline/test"
    content_list_files = Path(mineru_outputs_folder).rglob("*_content_list.json")
    data = ""
    for file in content_list_files:
        with open(file, 'r', encoding="utf-8") as file:
            data = json.load(file)
        detected_components = [
            item
            for item in data
            if item.get("type") not in ("header", "footer")
            and not (item.get("type") == "text" and not str(item.get("text", "")).strip())
        ]

        # Đếm số lượng từ biến đã lưu
        detected_components_count = len(detected_components)
        image_name = Path(file.name).stem.split("_")[0]
        image_path = f"{image_folder}/{image_name}.jpeg"
        should_tile_result, reason = should_trigger_tiling(data, image_path)
        if should_tile_result:
            should_tile.append(image_name)
            dest_dir = f"{should_tile_folder}/{image_name}.jpeg"
            shutil.copy(image_path, dest_dir)
        else:
            should_not_tile.append(image_name)
            dest_dir = f"{should_not_tile_folder}/{image_name}.jpeg"
            shutil.copy(image_path, dest_dir)

    # print(f"should_tile: {len(should_tile)}")
    # print(f"should_not_tile: {len(should_not_tile)}")
    print(f"should tile: {should_tile}")
    # lấy các file layout.pdf của các file should_tile
    layout_files = Path(mineru_outputs_folder).rglob("*_layout.pdf")
    dest_folder_layout_dir = "/Users/mytnguyen/Documents/LILaC/debug/should_tile_layout"
    Path(dest_folder_layout_dir).mkdir(exist_ok=True)
    for item in layout_files:
        image_name = item.stem.split("_")[0]
        if image_name in should_tile:
            shutil.copy2(item, dest_folder_layout_dir)
        # layout_path = f"{dest_folder_layout_dir}/{item}_layout.pdf"

    # check if the content_list.json less than minimum items
    # if no, then skip it
    # giờ có sẵn mineru_outputs rồi để copy paste qua
    # source_mineru_outputs_folder = "/Users/mytnguyen/Documents/LILaC/artifacts/artifacts/InfoVQA/mineru_outputs_pipeline/test"
    # dest_mineru_outputs_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs"
    # for name in should_not_tile:
    #     source_mineru_folder = f"{source_mineru_outputs_folder}/{name}"
    #     dest_mineru_folder = f"{dest_mineru_outputs_folder}/{name}"
    #     shutil.copytree(source_mineru_folder, dest_mineru_folder, dirs_exist_ok=True)
    
    # # if yes, then tile them
    # tile_output_depth_1 = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/tiled_input_depth_1"
    # manifest_depth_1 = f"{tile_output_depth_1}/tiling_manifest_depth_1.json"
    # mineru_outputs_depth_1_folder = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/mineru_outputs_tile_depth_1_ocr"
    # Path(mineru_outputs_depth_1_folder).mkdir(exist_ok=True)
    # manifest = prepare_tiled_inputs(Path(should_tile_folder),
    #                                 Path(tile_output_depth_1),
    #                                 Path(manifest_depth_1))
    
    # # Đi qua từng item để xử lý
    # for item in Path(tile_output_depth_1).iterdir():
    #     if item.stem != "tiling_manifest_depth_1":
    #         tile_path = f"{tile_output_depth_1}/{item.name}"
    #         cmd = ["mineru", "-b", "pipeline", "-m", "auto", "-p", tile_path, "-o", mineru_outputs_depth_1_folder]
    #         subprocess.run(cmd, check=False)
    
    return

if __name__ == "__main__":
    main()
    # check_tiling_depth_1_results()
    # check_image_size()
    # tile_specific_image()