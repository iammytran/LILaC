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

from src.utils.utils import (
    read_json_or_jsonl, read_yaml, REPO_ROOT,
    input_subpath, artifact_subpath, ensure_artifact_parsed_documents,
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
    parser.add_argument("--images_dir", default="",
                        help="Override page-image directory (for example the InfoVQA test split).")
    parser.add_argument("--adaptive_tiling", action="store_true",
                        help="Ask a multimodal LLM which pages need overlapping tiles.")
    parser.add_argument("--tiling_min_subcomponents", type=int, default=18)
    parser.add_argument("--tiling_min_aspect_ratio", type=float, default=1.6)
    parser.add_argument("--tiling_max_tiles", type=int, default=4)
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
    images_dir = args.images_dir or f"{REPO_ROOT}/datasets/{target}/image_components/test_tiling"
    layout_out = f"{REPO_ROOT}/artifacts/{target}/layout_{analyzer}/test"
    pd_out = f"{REPO_ROOT}/datasets/{target}/parsed_documents/dev"
    crops_out = f"{REPO_ROOT}/datasets/{target}/image_components_sub"

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Page images not found: {images_dir}")

    Path(layout_out).mkdir(parents=True, exist_ok=True)
    Path(pd_out).mkdir(parents=True, exist_ok=True)
    Path(crops_out).mkdir(parents=True, exist_ok=True)

    env_name = ANALYZER_ENV[analyzer]
    analyzer_module = ANALYZER_MODULE[analyzer]
    adapter_module = ADAPTER_MODULE[analyzer]

    analyzer_images_dir = images_dir
    tiling_manifest = None
    if args.adaptive_tiling:
        if analyzer != "doclayout_yolo":
            raise ValueError("--adaptive_tiling is supported only with doclayout_yolo")
        from src.lilac.lcg_constructor.preprocessing.adaptive_tiling import (
            prepare_tiled_inputs, merge_tile_boxes,
        )
        tiled_input = Path(layout_out) / "_tiled_inputs"
        tiling_manifest = Path(layout_out) / "tiling_manifest.json"
        prepare_tiled_inputs(
            Path(images_dir), tiled_input, tiling_manifest,
            args.tiling_min_subcomponents, args.tiling_min_aspect_ratio,
            args.tiling_max_tiles,
        )
    #     analyzer_images_dir = str(tiled_input)

    # # ── Stage A: run analyzer in its own conda env ──────────────────────────
    # print(f"[step5/{analyzer}] running layout analyzer (env={env_name})")
    # subprocess.run(
    #     [
    #         _conda_python(env_name),
    #         "-m", analyzer_module,
    #         "--input_dir",  analyzer_images_dir,
    #         "--output_dir", layout_out,
    #     ],
    #     check=True,
    #     cwd=REPO_ROOT,
    # )
    # if tiling_manifest is not None:
    #     from src.lilac.lcg_constructor.preprocessing.adaptive_tiling import merge_tile_boxes
    #     merge_tile_boxes(Path(layout_out), tiling_manifest)

    # # ── Stage B: run adapter in current env (has Qwen-VL for caption pass) ─
    # print(f"[step5/{analyzer}] running adapter → parsed_documents + caption pass")
    # subprocess.run(
    #     [
    #         sys.executable,
    #         "-m", adapter_module,
    #         "--layout",    layout_out,
    #         "--images",    images_dir,
    #         "--output",    pd_out,
    #         "--crops_out", crops_out,
    #         "--num_gpus",  str(args.num_gpus),
    #     ],
    #     check=True,
    #     cwd=REPO_ROOT,
    # )


if __name__ == "__main__":
    main()
