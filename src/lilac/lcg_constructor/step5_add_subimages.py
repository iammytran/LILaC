#!/usr/bin/env python3
"""
Step 5 — populate `parsed_documents[...]["subimage"]` for every page.

This is a **dispatcher** that picks the subimage-extraction path based on
``retriever_metadata.yaml`` `type`:

    type: multimodalqa  →  Qwen-VL bbox-detect + crop (`qwen_bbox` path)
                            Used for inline-image documents (MMCoQA,
                            MultimodalQA) where figures are embedded inside
                            a text-document and need bbox detection per image.
    type: vqa           →  Layout-analyzer + adapter (`mineru` or `doclayout_yolo`)
                            Used for page-image documents (MP-DocVQA, SlideVQA,
                            InfoVQA) where the page IS the image and every
                            detected region becomes a subimage.

Override the type-based default with ``--analyzer {qwen_bbox|doclayout_yolo|mineru}``.

Both paths fill `parsed_documents[...]["subimage"][<sid>]["caption"]["text"]`
with a VLM/structured-text summary (single insertion — no duplicate writes
to `caption.summary` or external `image_summaries_sub/` files).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List
import shutil

from src.utils.utils import (
    read_json_or_jsonl, read_yaml, REPO_ROOT,
    input_subpath, artifact_subpath, ensure_artifact_parsed_documents,
)
from src.lilac.lcg_constructor.preprocessing.tiling import (
    decide_tiling_with_mllm,
    estimate_subcomponent_count,
)

from src.lilac.lcg_constructor.preprocessing.mineru_to_lilac import (
    _prepare_image_crops_before_build
)

LCG_CONFIG_PATH = f"{REPO_ROOT}/config/lcg_constructor/a.yaml"
RETRIEVER_METADATA_PATH = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"

# Conda env hosting each analyzer.
ANALYZER_ENV = {
    "doclayout_yolo": "lilac-doclayout-yolo",
    "mineru": "lilac-mineru",
}
ANALYZER_MODULE = {
    "doclayout_yolo": "src.lilac.lcg_constructor.preprocessing.layout_doclayout_yolo",
    "mineru":         "src.lilac.lcg_constructor.preprocessing.layout_mineru",
}
ADAPTER_MODULE = {
    "doclayout_yolo": "src.lilac.lcg_constructor.preprocessing.doclayout_yolo_to_lilac",
    "mineru":         "src.lilac.lcg_constructor.preprocessing.mineru_to_lilac",
}


def _conda_python(env_name: str) -> str:
    p = f"/opt/miniconda3/envs/{env_name}/bin/python"
    if not os.path.exists(p):
        raise FileNotFoundError(f"conda env python not found: {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments():
    lcg_config = read_yaml(LCG_CONFIG_PATH)
    valid_targets = (
        lcg_config["type_to_dataset"]["multimodalqa"]
        + lcg_config["type_to_dataset"]["vqa"]
    )
    parser = argparse.ArgumentParser(description="Step 5 — extract subimages.")
    parser.add_argument("--target_data", type=str, required=True,
                        help=f"One of {valid_targets} (or a registered "
                             f"dataset suffixed with an analyzer tag).")
    parser.add_argument(
        "--analyzer", choices=["auto", "qwen_bbox", "doclayout_yolo", "mineru"],
        default="auto",
        help="auto = decide by retriever_metadata.yaml type "
             "(multimodalqa→qwen_bbox, vqa→mineru). "
             "Override to use a different path.",
    )
    parser.add_argument(
        "--mode", choices=["all", "split", "add_to_parsed_documents"],
        default="all",
        help="Sub-mode for the qwen_bbox path (kept for backwards compat).",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Cap on parallel GPU workers in caption pass (0 = all visible)")
    parser.add_argument("--adaptive_tiling", action="store_true",
                        help="Enable adaptive tiling before detection.")
    parser.add_argument("--tiling_min_subcomponents", type=int, default=18,
                        help="Reference crowded-layout subcomponent threshold for tiling decision.")
    parser.add_argument("--tiling_min_aspect_ratio", type=float, default=1.6,
                        help="Reference height/width threshold for tiling decision.")
    parser.add_argument("--tiling_overlap", type=float, default=0.12,
                        help="Tile overlap ratio for tiled layout execution.")
    parser.add_argument("--tiling_max_tiles", type=int, default=6,
                        help="Max number of tiles for tiled layout execution.")
    return parser.parse_args(), lcg_config


def _resolve_type(target_data: str) -> str:
    meta = read_yaml(RETRIEVER_METADATA_PATH)
    dm = meta.get("dataset_metadata", {})
    if target_data in dm:
        return dm[target_data].get("type", "")
    # Analyzer-suffixed dataset name: fall back to base name lookup.
    base = target_data.rsplit("-", 1)[0]
    if base in dm:
        return dm[base].get("type", "")
    return ""


def main():
    args, lcg_config = parse_arguments()
    target = args.target_data
    type_ = _resolve_type(target)

    if args.analyzer == "auto":
        analyzer = "qwen_bbox" if type_ == "multimodalqa" else "mineru"
    else:
        analyzer = args.analyzer

    print(f"[step5] target={target} type={type_!r} analyzer={analyzer}")

    if analyzer == "qwen_bbox":
        run_qwen_bbox_path(args, lcg_config)
    elif analyzer in ("doclayout_yolo", "mineru"):
        run_layout_analyzer_path(args, analyzer)
    else:
        raise ValueError(f"Unknown analyzer: {analyzer!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Path 1: Qwen-VL bbox detection (multimodalqa-type datasets)
# ─────────────────────────────────────────────────────────────────────────────

def run_qwen_bbox_path(args, lcg_config):
    """Qwen-VL bbox detect-and-crop, then Qwen-VL summarize each crop into
    parsed_documents.subimage[<sid>].caption.text."""
    target = args.target_data
    parsed_documents_path = ensure_artifact_parsed_documents(lcg_config, target)
    output_documents_path = artifact_subpath(lcg_config, target, "temp_dirname", "dev")
    images_dir = input_subpath(lcg_config, target, "image_components_dirname", "dev")
    subimages_dir = artifact_subpath(lcg_config, target, "subimage_components_dirname", "dev")

    os.makedirs(output_documents_path, exist_ok=True)
    os.makedirs(subimages_dir, exist_ok=True)

    mode = args.mode

    if mode in ("all", "split"):
        _qwen_detect_and_crop(images_dir, subimages_dir, args)
        _qwen_caption_crops(subimages_dir, args)

    if mode in ("all", "add_to_parsed_documents"):
        _qwen_inject_into_parsed_documents(
            parsed_documents_path, output_documents_path, subimages_dir,
        )


DETECT_PROMPT = ("Detect all objects in the image and return ONLY a JSON list "
                 "of {class, bbox_2d:[x1,y1,x2,y2]}. Do NOT include markdown or extra text.")


def _detect_worker(gpu_id: int, slice_paths: List[str], subimages_dir: str,
                   show_progress: bool) -> None:
    """One-GPU worker: load Qwen-VL, detect+crop per image, write detections.json + crops."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL
    from PIL import Image
    from tqdm import tqdm

    model = Qwen2_5_VL()
    iterator = tqdm(slice_paths, desc=f"step5/detect gpu={gpu_id}", disable=not show_progress)
    for img_path in iterator:
        base = Path(img_path).stem
        out_dir = Path(subimages_dir) / base
        det_json = out_dir / "detections.json"
        if det_json.exists():
            continue
        try:
            raw_outputs = model.infer(
                [{"text": DETECT_PROMPT, "images": [img_path]}],
                batch_size=1, max_tokens=2048,
            )
            out_str = raw_outputs[0] if raw_outputs else ""
        except Exception as e:
            print(f"[step5/detect gpu={gpu_id}] infer error on {img_path}: {e}")
            continue
        detections = _extract_json_objects(out_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write of detections.json.
        tmp = det_json.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(detections, f, indent=2)
        tmp.replace(det_json)
        try:
            orig = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[step5/detect gpu={gpu_id}] cannot open {img_path}: {e}")
            continue
        for i, det in enumerate(detections, start=1):
            cls = det.get("class", "object")
            box = det.get("bbox_2d") or det.get("bbox")
            if not box or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = (int(v) for v in box)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = orig.crop((x1, y1, x2, y2))
                crop.save(out_dir / f"{i}_{cls}.png")
            except Exception as e:
                print(f"[step5/detect gpu={gpu_id}] crop error {img_path}: {e}")


def _qwen_detect_and_crop(images_dir: str, subimages_dir: str, args) -> None:
    """Multi-GPU sharded Qwen-VL bbox detection + crop. Resumable per-image."""
    import multiprocessing as mp
    from PIL import Image, UnidentifiedImageError

    image_filenames = sorted(os.listdir(images_dir))
    s = args.start_idx
    e = args.end_idx if args.end_idx > 0 else len(image_filenames)
    image_filenames = image_filenames[s:e]

    todo = []
    for f in image_filenames:
        stem = os.path.splitext(f)[0]
        det_json = os.path.join(subimages_dir, stem, "detections.json")
        if os.path.exists(det_json):
            continue
        p = os.path.join(images_dir, f)
        try:
            with Image.open(p) as img:
                img.load()
        except (UnidentifiedImageError, OSError) as ex:
            print(f"[step5/qwen_bbox] skip unreadable {p}: {ex}")
            continue
        todo.append(p)

    if not todo:
        print(f"[step5/qwen_bbox] all detections present (skip {len(image_filenames)})")
        return

    # Multi-GPU shard.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        gpus = [int(t) for t in cvd.split(",") if t.strip().isdigit()] or [0]
    else:
        gpus = [0]
    if args.num_gpus:
        gpus = gpus[:args.num_gpus]
    print(f"[step5/qwen_bbox] detecting on {len(todo)} pages across GPUs {gpus}")

    if len(gpus) == 1:
        _detect_worker(gpus[0], todo, subimages_dir, show_progress=True)
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    import math
    per_gpu = math.ceil(len(todo) / len(gpus))
    procs = []
    for rank, gpu in enumerate(gpus):
        a, b = rank * per_gpu, min((rank + 1) * per_gpu, len(todo))
        if a >= b:
            continue
        slice_paths = todo[a:b]
        p = mp.Process(
            target=_detect_worker,
            args=(gpu, slice_paths, subimages_dir, rank == 0),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"[step5/qwen_bbox] worker pid={p.pid} exit={p.exitcode}")


def _extract_json_objects(s: str) -> List[Dict]:
    """Scan a string for balanced {...} JSON objects and return them as dicts."""
    objs = []
    stack = []
    start_idx = None
    for i, ch in enumerate(s):
        if ch == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    try:
                        objs.append(json.loads(s[start_idx:i+1]))
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
    return objs


def _qwen_caption_crops(subimages_dir: str, args) -> None:
    """Caption every cropped PNG under subimages_dir/<page>/*.png via Qwen-VL."""
    from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import caption_images

    sub_root = Path(subimages_dir)
    crop_paths: List[Path] = []
    for page_dir in sorted(sub_root.iterdir()):
        if not page_dir.is_dir():
            continue
        for crop in sorted(page_dir.glob("*.png")):
            if crop.name in ("boxed.png", "BOXED.png"):
                continue
            crop_paths.append(crop)

    if not crop_paths:
        print("[step5/qwen_bbox] no crops to caption")
        return

    img_paths = [str(p) for p in crop_paths]
    out_paths = [str(p.with_suffix(".txt")) for p in crop_paths]
    caption_images(
        image_paths=img_paths,
        output_paths=out_paths,
        prompt=("Generate a summary of the given image region. "
                "Include all the texts within it."),
        max_tokens=512,
        num_gpus=(args.num_gpus or None),
    )


def _qwen_inject_into_parsed_documents(
    parsed_documents_path: str,
    output_documents_path: str,
    subimages_dir: str,
) -> None:
    """Walk parsed_documents, append subimage entries from each page's
    image_components_sub/<stem>/*.png and merge caption.txt sidecars."""
    from tqdm import tqdm
    filenames = sorted(os.listdir(parsed_documents_path))
    for filename in tqdm(filenames, desc="step5 inject"):
        file_path = os.path.join(parsed_documents_path, filename)
        parsed_document = read_json_or_jsonl(file_path)
        # Make sure subimage key exists.
        parsed_document.setdefault("subimage", {})
        for image_id in parsed_document.get("image", {}):
            image_obj = parsed_document["image"][image_id]
            if not image_obj.get("filename"):
                continue
            stem = os.path.splitext(image_obj["filename"])[0]
            sub_dir = Path(subimages_dir) / stem
            if not sub_dir.is_dir():
                continue
            for crop_path in sorted(sub_dir.glob("*.png")):
                if crop_path.name in ("boxed.png", "BOXED.png"):
                    continue
                # Index from filename prefix: "1_class.png" → "1"
                idx_part = crop_path.name.split("_", 1)[0]
                sid = f"{image_id}_s{idx_part}"
                if sid in parsed_document["subimage"]:
                    continue
                rel_path = f"{stem}/{crop_path.name}"
                entry = deepcopy(parsed_document["image"][image_id])
                entry["filename"] = rel_path
                # Merge caption text from sidecar.
                txt_sidecar = crop_path.with_suffix(".txt")
                caption_text = ""
                if txt_sidecar.exists():
                    caption_text = txt_sidecar.read_text(encoding="utf-8").strip()
                entry["caption"] = {"text": caption_text, "edges": []}
                parsed_document["subimage"][sid] = entry
        with open(os.path.join(output_documents_path, filename), "w") as f:
            json.dump(parsed_document, f, indent=4)

    # Atomic swap: parsed_documents <-> temp.
    parsed_parent = os.path.dirname(parsed_documents_path)
    output_parent = os.path.dirname(output_documents_path)
    swap_parent = parsed_parent + "__swap__"
    if os.path.exists(swap_parent):
        import shutil; shutil.rmtree(swap_parent)
    os.rename(parsed_parent, swap_parent)
    os.rename(output_parent, parsed_parent)
    os.rename(swap_parent, output_parent)


# ─────────────────────────────────────────────────────────────────────────────
# Path 2: Layout analyzer (vqa-type datasets)
# ─────────────────────────────────────────────────────────────────────────────

def run_layout_analyzer_path(args, analyzer: str):
    """Run DocLayoutYOLO or MinerU on page images → adapter → parsed_documents."""
    target = args.target_data
    images_dir = f"{REPO_ROOT}/datasets/{target}/image_components/test"
    layout_out = f"{REPO_ROOT}/artifacts/{target}/layout_{analyzer}/after_tiling"
    pd_out = f"{REPO_ROOT}/artifacts/{target}/parsed_documents/after_tiling"
    tiling_crops_out = Path(f"{REPO_ROOT}/artifacts/{target}/tiling_crops_out")
    crops_out = Path(f"{REPO_ROOT}/artifacts/{target}/crops_out")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Page images not found: {images_dir}")

    Path(layout_out).mkdir(parents=True, exist_ok=True)
    Path(pd_out).mkdir(parents=True, exist_ok=True)
    Path(tiling_crops_out).mkdir(parents=True, exist_ok=True)

    env_name = ANALYZER_ENV[analyzer]
    analyzer_module = ANALYZER_MODULE[analyzer]
    adapter_module = ADAPTER_MODULE[analyzer]
    tiling_decisions_path = None

    # if args.adaptive_tiling:
    #     tiling_decisions = _build_adaptive_tiling_decisions(
    #         images_dir=images_dir,
    #         layout_out=layout_out,
    #         min_subcomponents=args.tiling_min_subcomponents,
    #         min_aspect_ratio=args.tiling_min_aspect_ratio,
    #     )
    # lấy tiling_decisions từ các file json
    # tiling_decisions = {}
    # tiling_decisions_path = "debug/tiling_decision"
    # for path in Path(tiling_decisions_path).rglob("*.json"):
    #     image_name = path.stem
    #     json_content = {}
    #     with open(path, 'r') as file:
    #         json_content = json.load(file)
    #     tiling_decisions[image_name] = json_content
        

    # # # tiling_crops_out = "artifacts/InfoVQA/after_tiling_crops_out"
    # prepared_image_crops = _prepare_image_crops_before_build(
    #         layout_dir=layout_out,
    #         crops_out_dir=tiling_crops_out,
    #         adaptive_tiling=args.adaptive_tiling,
    #         tiling_decisions=tiling_decisions,
    #         tiling_overlap=args.tiling_overlap,
    #         tiling_max_tiles=args.tiling_max_tiles,
    #     )
    # print(f"[adapter/mineru] prepared image blocks before build: {len(prepared_image_crops)} docs")
    # print(f"prepared_image_crops: {prepared_image_crops}")
    # # with open(tiling_decisions_path, 'r', ''):
    # #     tiling_decisions.
    # # if tiling_decisions_path.title == "true":
    #     # tile images
    #     # save tiled_images to same folder 

    # # move all tiled_images to a folder
    # tiling_image_components_sub = Path(f"{REPO_ROOT}/artifacts/{target}/tiling_image_components_sub")
    # tiling_image_components_sub.mkdir(parents=True, exist_ok=True)

    # for img in tiling_crops_out.rglob("*.jpeg"):
    #     dest_dir = tiling_image_components_sub / img.name
    #     shutil.copy2(img, dest_dir)

    # # ── Stage A: run analyzer in its own conda env ──────────────────────────
    # print(f"[step5/{analyzer}] running layout analyzer (env={env_name})")
    # analyzer_cmd = [
    #     'python',
    #     "-m", analyzer_module,
    #     "--input_dir", str(tiling_image_components_sub),
    #     "--output_dir", layout_out,
    # ]
    # subprocess.run(analyzer_cmd, check=True, cwd=REPO_ROOT)

    # # # gom tất cả tiled_images vào 1 chỗ
    # # # gom tất cả components của tiled_images vào 1 file content_list.json để pass tạo parsed_document
    after_process_tiling_path = Path("artifacts/InfoVQA/layout_mineru/after_process_tiling")
    dummy = merge_tiling_images(after_process_tiling_path)
    merge_tiling_content_list(after_process_tiling_path)

    # # ── Stage B: run adapter in current env (has Qwen-VL for caption pass) ─
    # print(f"[step5/{analyzer}] running adapter → parsed_documents + caption pass")
    # subprocess.run(
    #     [
    #         sys.executable,
    #         "-m", adapter_module,
    #         "--layout",    layout_out,
    #         "--images",    images_dir,
    #         "--output",    pd_out,
    #         "--tiling_crops_out", tiling_crops_out,
    #         "--num_gpus",  str(args.num_gpus),
    #     ],
    #     check=True,
    #     cwd=REPO_ROOT,
    # )

def get_image_id(tile_image_path: Path) -> str:
    import re
    for parent in tile_image_path.parents:
        match = re.match(r"^(\d+)___tiling__tile_\d+$", parent.name)
        if match:
            return match.group(1)

    raise ValueError(
        f"Cannot find tiling directory in path: {tile_image_path}"
    )

def merge_tiling_images(after_process_tiling_path):
    after_tiling_path = Path("artifacts/InfoVQA/layout_mineru/after_tiling")
    # after_process_tiling_path = Path("artifacts/InfoVQA/layout_mineru/after_process_tiling")
    tiling_folders_grouped = {}
    # gom các folder tile của cùng hình
    all_tile_images = list(after_tiling_path.rglob("*.jpg"))
    all_content_list_files = list(after_tiling_path.rglob("*_content_list.json"))
    all_original_images = list(after_tiling_path.rglob("*_origin.pdf"))
    all_tile_layout = list(after_tiling_path.rglob("*_layout.pdf"))
    all_middle_files = list(after_tiling_path.rglob("*_middle.json"))
    for tile_image_path in all_tile_images:
        # print(f"hello")
        image_name = get_image_id(tile_image_path)
        after_processed_tiling_dir = after_process_tiling_path / image_name / "images"
        # print(f"after_processed_tiling_dir: {after_processed_tiling_dir}")
        after_processed_tiling_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            tile_image_path,
            after_processed_tiling_dir
        )

    for content_list_file in all_content_list_files:
        image_name = get_image_id(content_list_file)
        after_processed_tiling_dir = after_process_tiling_path / image_name
        after_processed_tiling_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            content_list_file,
            after_processed_tiling_dir
        )

    for original_image in all_original_images:
        image_name = get_image_id(original_image)
        after_processed_tiling_dir = after_process_tiling_path / image_name
        after_processed_tiling_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            original_image,
            after_processed_tiling_dir
        )

    for tile_layout in all_tile_layout:
        image_name = get_image_id(tile_layout)
        after_processed_tiling_dir = after_process_tiling_path / image_name
        after_processed_tiling_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            tile_layout,
            after_processed_tiling_dir
        )

    for middle_file in all_middle_files:
        image_name = get_image_id(middle_file)
        after_processed_tiling_dir = after_process_tiling_path / image_name
        after_processed_tiling_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            middle_file,
            after_processed_tiling_dir
        )

    return tiling_folders_grouped

def merge_tiling_content_list(after_process_tiling_path):
    for folder in Path(after_process_tiling_path).iterdir():
        folder_name = folder.name
        each_folder_name_dir = Path(after_process_tiling_path) / folder_name
        all_content_list_files = Path(each_folder_name_dir).rglob("*_content_list.json")
        output_file = Path(after_process_tiling_path) / folder_name / "unified_content_list.json"
        unify_tile_contents(all_content_list_files, output_file)

def unify_tile_contents(json_paths_list, output_json_path):
    import copy
    """json_paths_list: list các đường dẫn file content_list.json của các tile thuộc 1 ảnh."""
    all_items = []

    # 1. Đọc tất cả các file JSON của các tile
    for json_path in json_paths_list:
        p = Path(json_path)
        if not p.exists():
            continue

        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            item_copy = copy.deepcopy(item)
            # Xóa trường bbox nếu có
            item_copy.pop("bbox", None)
            all_items.append(item_copy)

    # 2. Lọc trùng lặp (Deduplication)
    unified_list = []
    seen_signatures = set()

    for item in all_items:
        # Tạo key định danh để kiểm tra trùng
        if item.get("type") == "text":
            sig = ("text", item.get("text", "").strip())
        elif item.get("type") == "image":
            sig = ("image", item.get("img_path", ""))
        else:
            sig = (item.get("type"), str(item))

        # Nếu chưa xuất hiện thì giữ lại
        if sig not in seen_signatures:
            seen_signatures.add(sig)
        unified_list.append(item)

    # 3. Gán dla_index từ 0 tăng dần
    # for idx, item in enumerate(unified_list):
    #     item["dla_index"] = idx

    # 4. Ghi ra file JSON kết quả
    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(unified_list, f, ensure_ascii=False, indent=4)

    return unified_list

def _build_adaptive_tiling_decisions(
    images_dir: str,
    layout_out: str,
    min_subcomponents: int,
    min_aspect_ratio: float,
) -> str:
    from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL
    import cv2

    decisions_dir = Path(layout_out) / "_adaptive_tiling"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = decisions_dir / "decisions.json"
    if decisions_path.exists():
        with open(decisions_path) as f:
            decisions = json.load(f)
        if not isinstance(decisions, dict):
            raise ValueError(f"Invalid tiling decisions cache format: {decisions_path}")
    else:
        decisions = {}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    todo = [p for p in image_paths if p.stem not in decisions]
    print(f"[step5/adaptive_tiling] decisions: {len(todo)} to infer, {len(decisions)} cached")
    if not todo:
        return str(decisions_path)

    qwen = Qwen2_5_VL()
    for image_path in todo:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read page image for tiling decision: {image_path}")
        height, width = image.shape[:2]
        if width == 0:
            raise ValueError(f"Invalid image width=0: {image_path}")
        aspect_ratio = float(height / width)
        estimated = estimate_subcomponent_count(image)
        decision = decide_tiling_with_mllm(
            image_path=str(image_path),
            estimated_subcomponents=estimated,
            aspect_ratio=aspect_ratio,
            qwen_model=qwen,
            min_subcomponents_hint=min_subcomponents,
            min_aspect_ratio_hint=min_aspect_ratio,
        )
        decisions[image_path.stem] = decision

    return decisions


if __name__ == "__main__":
    main()
