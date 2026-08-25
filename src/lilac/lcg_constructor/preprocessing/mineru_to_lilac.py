#!/usr/bin/env python3
"""
Adapter — MinerU layout output → LILaC parsed_documents schema.

Routes MinerU content_list blocks to the correct LILaC slot, matching the
shipped MP-DocVQA convention:

  image (MinerU)                                     -> subimage  (caption.text = Qwen-VL summary)
  text / header / footer / page_number / equation    -> sentence  (text = MinerU native text)
  table                                              -> table_segment (text = MinerU native HTML/markdown body)

Per-page id conventions follow shipped MP-DocVQA:
  i_1                page-level image (top-level)
  i_1_i<N>           image subimage
  i_1_p<N>           sentence (per-page global counter)
  i_1_t<N>           table_segment

Pipeline:
  1. Walk every *_content_list.json under --layout. For each block, route to
     its target slot, populate text/caption directly from MinerU native
     output where available. For image-class blocks, save the crop and
     queue it for a Qwen-VL summary pass.
  2. Run caption_images on the image crops only (much smaller than the
     text/table blocks, which MinerU already gave us native text for).
  3. Merge per-crop .txt sidecars back into subimage[*].caption.text.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import re
import pandas as pd
from io import StringIO
import cv2

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import caption_images
from src.lilac.lcg_constructor.preprocessing.tiling import (
    build_crowded_layout_tiles,
    decide_tiling_with_mllm,
    estimate_subcomponent_count,
    load_tiling_decisions,
)


TEXT_LIKE_TYPES = {
    "text", "header", "footer", "page_number", "page_footnote",
    "equation", "code", "list", "aside_text",
}
TABLE_TYPES = {"table"}
IMAGE_TYPES = {"image", "chart", "seal"}

SUMMARY_PROMPT = (
    "Extract any text visible in this image region verbatim. "
    "If no text is present, give a brief one-sentence (<15 words) factual "
    "description of the visual content. Be terse — no preamble, no commentary."
)


def _find_content_lists(layout_dir: Path) -> Dict[str, Path]:
    """Locate every `*_content_list.json` under layout_dir, keyed by doc_id."""
    found: Dict[str, Path] = {}
    folders = layout_dir.rglob("*_content_list.json")
    for p in folders:
        try:
            rel = p.relative_to(layout_dir)
        except ValueError:
            continue
        doc_id = rel.parts[0] if rel.parts else p.stem.replace("_content_list", "")
        found.setdefault(doc_id, p)
    return found


def _block_text(block: Dict) -> Optional[str]:
    """Native text for a non-image block; None if nothing extractable."""
    btype = block.get("type", "")
    if btype in TEXT_LIKE_TYPES:
        return (block.get("text") or block.get("md") or "") or None
    if btype in TABLE_TYPES:
        body = block.get("table_body") or block.get("table") or block.get("md") or ""
        caption = block.get("table_caption") or ""
        if isinstance(caption, list):
            caption = " ".join(caption)
        text = (caption + "\n" + body) if body else caption
        return text.strip() or None
    return None


def _block_image_path(block: Dict, content_list_dir: Path) -> Optional[Path]:
    rel = block.get("img_path") or block.get("image_path")
    if not rel:
        return None
    p = (content_list_dir / rel).resolve()
    return p if p.exists() else None


def _write_tile_crops(
    src_img: Path,
    target_parent: Path,
    stem_prefix: str,
    overlap_ratio: float,
    max_tiles: int,
) -> List[Path]:
    image = cv2.imread(str(src_img))
    if image is None:
        raise ValueError(f"[adapter/mineru] cannot read image for tiling: {src_img}")
    h, w = image.shape[:2]
    tiles = build_crowded_layout_tiles(
        width=w,
        height=h,
        overlap_ratio=overlap_ratio,
        max_tiles=max_tiles,
    )
    print(f"tiles: {tiles}")
    if not tiles:
        return []
    target_parent.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    ext = src_img.suffix or ".jpg"
    for tile_idx, tile in enumerate(tiles, start=1):
        crop = image[tile.y1:tile.y2, tile.x1:tile.x2]
        if crop.size == 0:
            continue
        out_path = target_parent / f"{stem_prefix}_tile_{tile_idx}{ext}"
        if not out_path.exists():
            cv2.imwrite(str(out_path), crop)
        out_paths.append(out_path)
    return out_paths


def _prepare_image_crops_before_build(
    layout_dir: Path,
    crops_out_dir: Path,
    adaptive_tiling: bool,
    tiling_decisions: Dict[str, Dict],
    tiling_overlap: float,
    tiling_max_tiles: int,
) -> Dict[str, Dict[int, List[Path]]]:
    """
    Precompute image crops per (doc_id, block_index) before parsed-document building.

    Returns:
        {
          "<doc_id>": {
              <block_index>: [crop_path1, crop_path2, ...]
          }
        }
    """
    # content_lists = _find_content_lists(layout_dir)
    image_path_dir = Path("datasets/InfoVQA/image_components/test")
    image_names = [
        p.stem for p in sorted(image_path_dir.iterdir()) 
        if p.is_file()
    ]

    prepared: Dict[str, Dict[int, List[Path]]] = {}

    for doc_id in image_names:
        # should_tile = bool((tiling_decisions.get(doc_id) or {}).get("tile", False)) if adaptive_tiling else False
        should_tile = bool((tiling_decisions.get(doc_id) or {}).get("tile", False))
        crop_parent = crops_out_dir / doc_id
        crop_paths: List[Path] = []
        print(f"tiling_decisions.get(doc_id): {tiling_decisions.get(doc_id)}")
        print(f"should_tile: {should_tile}")
        if should_tile:
            print(f"should_tile = True")
            tile_stem = f"{doc_id}___tiling_"
            image_path = image_path_dir / f"{doc_id}.jpeg"
            crop_paths = _write_tile_crops(
                src_img=image_path,
                target_parent=crop_parent,
                stem_prefix=tile_stem,
                overlap_ratio=tiling_overlap,
                max_tiles=tiling_max_tiles,
            )
            if not crop_paths:
                print(f"[adapter/mineru] tiling produced no crops for {image_path_dir}, fallback to full crop")
        prepared[doc_id] = crop_paths
        # if not crop_paths:
        #     crop_fname = f"{doc_id}___b_{block_index}{src_img.suffix or '.jpg'}"
        #     target_crop = crop_parent / crop_fname
        #     target_crop.parent.mkdir(parents=True, exist_ok=True)
        #     if not target_crop.exists():
        #         shutil.copyfile(src_img, target_crop)
        #     crop_paths = [target_crop]

        # per_block[block_index] = crop_paths

    return prepared



def build_parsed_documents(
    layout_dir: Path,
    images_dir: Path,
    parsed_documents_out_dir: Path,
    crops_out_dir: Path,
    prepared_image_crops: Dict[str, Dict[int, List[Path]]],
) -> List[Path]:
    """Build skeleton parsed_documents + image crops. Returns crops needing Qwen-VL summary."""
    parsed_documents_out_dir.mkdir(parents=True, exist_ok=True)
    crops_out_dir.mkdir(parents=True, exist_ok=True)

    content_lists = _find_content_lists(layout_dir)
    # keys_to_keep = ['10183']

    # # # Create the subset (ignoring keys that might not exist)
    # content_lists = {k: content_lists[k] for k in keys_to_keep if k in content_lists}
    print(f"[adapter/mineru] {len(content_lists)} content_list.json under {layout_dir}")

    crops_to_summarize: List[Path] = []
    type_counts: Dict[str, int] = {}

    for doc_id, cl_path in content_lists.items():
        with open(cl_path) as f:
            blocks = json.load(f)
        if not isinstance(blocks, list):
            blocks = blocks.get("content_list", [])

        page_img_candidates = [images_dir / f"{doc_id}.jpg",
                               images_dir / f"{doc_id}.jpeg",
                               images_dir / f"{doc_id}.png"]
        page_img = next((p for p in page_img_candidates if p.exists()), None)
        if page_img is None:
            print(f"[adapter/mineru] page image not found for {doc_id}")
            continue
        i_counter = 0
        p_counter = 0
        t_counter = 0

        subimage_entries: Dict[str, Dict] = {}
        sentence_entries: Dict[str, Dict] = {}
        table_entries: Dict[str, Dict] = {}
        id_sequence: List[str] = ["i_1"]

        for block_index, block in enumerate(blocks):
            btype = block.get("type", "")
            type_counts[btype] = type_counts.get(btype, 0) + 1
            common_meta = {
                "dla_idx": block.get("dla_idx") or block.get("idx"),
                "dla_type": btype,
                "_bbox": block.get("bbox"),
            }

            if btype in IMAGE_TYPES:
                block_crops = (prepared_image_crops.get(doc_id) or {}).get(block_index, [])
                if not block_crops:
                    continue
                for block_crop in block_crops:
                    i_counter += 1
                    cid = f"i_1_i{i_counter}"
                    subimage_entries[cid] = {
                        **common_meta,
                        "filename": block_crop.name,
                        "caption": {"text": "", "edges": []},
                    }
                    crops_to_summarize.append(block_crop)
                    id_sequence.append(cid)
                continue

            native_text = _block_text(block)
            if native_text is None:
                continue  # unknown block with no text — skip

            if btype in TABLE_TYPES:
                t_counter += 1
                cid = f"i_1_t{t_counter}"
                rows = _parse_html_to_table(doc_id, native_text)
                crop_fname = f"{doc_id}___t_{t_counter}.jpg"
                table_data = []
                for row in rows:
                    row_data = []
                    for cell in row:
                        row_data.append({"text": cell})
                    table_data.append(row_data)

                table_entries[cid] = {
                    **common_meta,
                    "text": native_text,
                    "table_caption": block.get('table_caption', ''), 
                    "rows": len(rows),
                    "table": table_data,
                    "filename": crop_fname,
                    "edges": [],
                }
                # table_entries[cid] = {
                #     **common_meta,
                #     "text": _parse_html_to_table(doc_id, native_text),
                #     "edges": [],
                # }
            else:  # text-like
                p_counter += 1
                cid = f"i_1_p{p_counter}"
                sentence_entries[cid] = {
                    **common_meta,
                    "text": native_text,
                    "edges": [],
                }
            id_sequence.append(cid)

        parsed_doc = {
            "title": doc_id,
            "hierarchy": {},
            "id_sequence": id_sequence,
            "header": {},
            "text": {},
            "table": table_entries,
            "image": {
                "i_1": {
                    "filename": page_img.name,
                    "caption": {"text": "", "edges": []},
                }
            },
            "sentence": sentence_entries,
            "proposition": {},
            "table_segment": {},
            "subimage": subimage_entries,
            "id_to_html": {},
        }

        parsed_doc = transform_table_to_table_segment(parsed_doc)

        with open(parsed_documents_out_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(parsed_doc, f, indent=4)

    if type_counts:
        print(f"[adapter/mineru] block type counts: {type_counts}")
    return crops_to_summarize

def transform_table_to_table_segment(parsed_doc):
    table_entries = parsed_doc.get("table", {})
    table_segment_entries = {}

    for block_id, block_content in table_entries.items():
        ts_counter = 0
        table_content = block_content.get('table', [])
        if len(table_content) == 0:
            return parsed_doc
        header = table_content[0]
        body = table_content[1:]
        imagename = block_content.get('filename', '')
        parent_id = f"{block_id}"
        for row in body:
            ts_counter += 1
            table_segment_id = f"{parent_id}_ts{ts_counter}"
            table_segment_entries[table_segment_id] = {
                "dla_type": "table",
                "table": [
                    header,
                    row
                ],
                # "filename": imagename,
                "edges": [],
            }

    parsed_doc["table_segment"] = table_segment_entries

    return parsed_doc

def remove_img_tag_from_html_text(html_data):
    # 1. Bỏ thẻ img
    clean_html = re.sub(r"<img[^>]*>", "", html_data)

    # 2. Xóa các tiền tố thừa như ID / [SEP] nếu không cần thiết
    clean_html = re.sub(r"^\d+\s*\[SEP\]\s*", "", clean_html)
    return clean_html

def _fix_mineru_sticky_text(text):
    if not isinstance(text, str):
        return text
    
    # 1. Phát hiện chữ thường/số dính liền chữ hoa (ví dụ: eL trong peopleLarger)
    # Tiến hành chèn dấu chấm và khoảng trắng vào giữa chúng để tách câu rõ ràng
    cleaned = re.sub(r'([a-z0-9])([A-Z])', r'\1. \2', text)
    
    # 2. Xử lý thêm trường hợp dấu đóng ngoặc dính liền chữ (nếu có, ví dụ: 'pickforks)Lawnmowers')
    cleaned = re.sub(r'(\))([A-Z])', r'\1. \2', cleaned)

    cleaned = re.sub(r'(?<!<Image)(?<!http)(?<!https):([A-Z])', r': \1', cleaned)
    
    # 3. Thu gọn các khoảng trắng thừa thành 1 khoảng trắng duy nhất
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def _parse_html_to_table(doc_id, html_str):
    html_str = remove_img_tag_from_html_text(html_str)
    soup = BeautifulSoup(html_str, 'html.parser')
    table = soup.find('table')

    # Trường hợp không tìm thấy thẻ table để tránh crash
    if not table:
        return []
        
    # 1. DUYỆT QUA TỪNG Ô ĐỂ LÀM SẠCH TEXT TRƯỚC (Giữ nguyên logic custom của bạn)
    for cell in table.find_all(["td", "th"]):
        raw_content = _process_cell_preserving_tags(cell)
        # print(f"before fixe_mineru: {raw_content}")
        fixed_sticky_text = _fix_mineru_sticky_text(raw_content)
        
        # Thay thế nội dung bên trong ô bằng đoạn text đã xử lý sạch
        cell.string = fixed_sticky_text
    
    if not table.get_text(strip=True):
        rows = table.find_all('tr')
        if not rows: return []
        num_cols = len(rows[0].find_all(['td', 'th']))
        return [["" for _ in range(num_cols)] for _ in range(len(rows))]

    # 2. DÙNG PANDAS ĐỂ PARSE BẢNG ĐÃ ĐƯỢC LÀM SẠCH
    try:
        html_io = StringIO(str(table))
        dfs = pd.read_html(html_io, header=None) 
        df = dfs[0]
        
        # Bước 1: Ép kiểu toàn bộ bảng sang string (ô NaN sẽ tạm thời biến thành chuỗi "nan")
        df = df.astype(str)
        
        # Bước 2: Thay thế chuỗi "nan" thành chuỗi rỗng "" một cách đồng bộ
        df.replace("nan", "", inplace=True)
        return df.values.tolist()

    except Exception as e:
        print(f"Error for doc_id {doc_id}: {e}")
        return []

def _process_cell_preserving_tags(cell):
    parts = []
    # Duyệt qua từng phần tử con bên trong ô <td> theo đúng thứ tự đọc từ trái qua phải
    for content in cell.contents:
        if content.name == "img":
            print(f"found image: {content}")
            # # Nếu là thẻ img, lấy thuộc tính src và bọc nó lại thành dạng text đại diện
            # img_src = content.get("src", "")
            # parts.append(f" <Image: {img_src}> ")
        else:
            # print(f"not found image: {content}")
            # Nếu là text thuần, chỉ cần ép kiểu về chuỗi và clear khoảng trắng thừa
            text_inside = str(content).strip()
            if text_inside:
                parts.append(text_inside)
                
    # Gộp lại thành một chuỗi duy nhất của ô đó
    return "".join(parts).strip()
    
def _add_dla_idx():
    mineru_outputs_dir = Path("artifacts/InfoVQA/mineru_outputs_pipeline/dev")
    middle_file_paths = sorted(mineru_outputs_dir.rglob("*_middle.json"))
    content_list_file_paths = sorted(mineru_outputs_dir.rglob("*_content_list.json"))
    
    # Tạo list để lưu lại các doc_id thực sự được update trong lần chạy này
    updated_docs = []
    skipped_count = 0

    for middle_file_path in middle_file_paths:
        middle_file_path_name = middle_file_path.stem
        doc_id = middle_file_path_name.split("_")[0]

        content_list_name = f"{doc_id}_content_list.json"
        content_list_filepath = next((f for f in content_list_file_paths if f.name == content_list_name), None)
        
        if not content_list_filepath:
            print(f"❌ [Error] content_list file not found for doc_id: {doc_id}")
            continue

        with open(content_list_filepath, 'r', encoding="utf-8") as file:
            content_list_content = json.load(file)

        # Kiểm tra xem file đã được xử lý từ trước chưa
        if content_list_content and 'dla_idx' in content_list_content[0]:
            # print(f"⏭️ [Skipped] doc_id {doc_id} is already processed.")
            skipped_count += 1
            continue

        with open(middle_file_path, 'r', encoding="utf-8") as file:
            middle_file_content = json.load(file)

        # Get the blocks' dla_idx
        pdf_info_first = next(iter(middle_file_content.get("pdf_info", [])), {})

        preproc_blocks = pdf_info_first.get("preproc_blocks", [])
        discarded_blocks = pdf_info_first.get("discarded_blocks", [])
        all_blocks = preproc_blocks + discarded_blocks

        block_ids = []
        for block in all_blocks:
            block_id = block.get("index", -1)
            block_ids.append(block_id)
        
        print(f"block_ids: {block_ids}")
        # Attach the dla_idx
        for content_list_block, block_id in zip(content_list_content, block_ids):
            content_list_block['dla_idx'] = block_id

        # Write to JSON file
        with open(content_list_filepath, 'w', encoding="utf-8") as file:
            json.dump(content_list_content, file, indent=4, ensure_ascii=False)

        print(f"✅ [Success] Updated 'dla_idx' for doc_id: {doc_id}")
        print(f"   - Total blocks processed: {len(block_ids)}")
        print("-" * 60)
        
        # Thêm doc_id vào danh sách cập nhật thành công
        updated_docs.append(doc_id)

    print(f"[adapter/mineru] adding dla_idx for content_list.json: updated {len(updated_docs)} docs, and skipped {skipped_count} docs")
            
    return updated_docs  # Trả về list này để hàm khác có thể tái sử dụng nếu cần

def merge_subimage_captions(parsed_documents_out_dir: Path, crops_out_dir: Path) -> None:
    """Read per-crop .txt sidecars and append into subimage[*].caption.text."""
    files = sorted(parsed_documents_out_dir.glob("*.json"))
    merged, missing = 0, 0
    
    for pd_path in files:
        with open(pd_path, "r", encoding="utf-8") as f:
            pd = json.load(f)
            
        doc_id = pd_path.stem
        any_updated = False
        
        for sid, sv in pd.get("subimage", {}).items():
            crop_fname = sv.get("filename", "")
            if not crop_fname:
                missing += 1
                continue
                
            txt_path = (crops_out_dir / doc_id / crop_fname).with_suffix(".txt")
            if txt_path.exists():
                sidecar_text = txt_path.read_text(encoding="utf-8").strip()
                if not sidecar_text:
                    continue

                if not isinstance(sv.get("caption"), dict):
                    sv["caption"] = {}
                
                existing_text = sv["caption"].get("text", "").strip()
                
                # Nếu đã có text thì nối thêm vào (dùng "\n" hoặc " "), nếu chưa có thì gán mới
                if existing_text:
                    sv["caption"]["text"] = f"{existing_text}\n{sidecar_text}"
                else:
                    sv["caption"]["text"] = sidecar_text
                    
                merged += 1
                any_updated = True
            else:
                missing += 1
                
        if any_updated:
            with open(pd_path, "w", encoding="utf-8") as f:
                json.dump(pd, f, indent=4, ensure_ascii=False)
                
    print(f"[adapter/mineru] merged {merged} image captions; {missing} crops still uncaptioned")

def _create_subimage_folder(crops_out_dir, subimage_dir):
    subimage_dir.mkdir(parents=True, exist_ok=True)

    for img in crops_out_dir.rglob("*.jpg"):
        dest_dir = subimage_dir / img.name
        shutil.copy2(img, dest_dir)
    return 


def _build_adaptive_tiling_decisions_for_mineru(
    images_dir: Path,
    decisions_json: Optional[str],
    min_subcomponents: int,
    min_aspect_ratio: float,
) -> Dict[str, Dict]:
    if decisions_json:
        decisions = load_tiling_decisions(decisions_json)
    else:
        decisions = {}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    todo = [p for p in image_paths if p.stem not in decisions]
    if not todo:
        return decisions

    from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL

    qwen = Qwen2_5_VL()
    for image_path in todo:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"[adapter/mineru] could not read page image for tiling decision: {image_path}")
        height, width = image.shape[:2]
        if width == 0:
            raise ValueError(f"[adapter/mineru] invalid width=0 for {image_path}")
        decision = decide_tiling_with_mllm(
            image_path=str(image_path),
            estimated_subcomponents=estimate_subcomponent_count(image),
            aspect_ratio=float(height / width),
            qwen_model=qwen,
            min_subcomponents_hint=min_subcomponents,
            min_aspect_ratio_hint=min_aspect_ratio,
        )
        decisions[image_path.stem] = decision
    return decisions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--crops_out", required=True)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--num_gpus", type=int, default=0)
    ap.add_argument("--skip_caption", action="store_true",
                    help="Skip Qwen-VL caption pass on image crops")
    ap.add_argument("--adaptive_tiling", action="store_true",
                    help="Enable adaptive tiling for MinerU image blocks.")
    ap.add_argument("--tiling_decisions_json", default=None,
                    help="Optional path to precomputed tiling decisions JSON.")
    ap.add_argument("--tiling_min_subcomponents", type=int, default=18)
    ap.add_argument("--tiling_min_aspect_ratio", type=float, default=1.6)
    ap.add_argument("--tiling_overlap", type=float, default=0.12)
    ap.add_argument("--tiling_max_tiles", type=int, default=6)
    args = ap.parse_args()

    layout_dir = Path(args.layout)
    images_dir = Path(args.images)
    pd_out = Path(args.output)
    crops_out = Path(args.crops_out)
    subimage_dir = Path("artifacts/InfoVQA/image_components_sub/test")
    tiling_decisions: Dict[str, Dict] = {}

    _add_dla_idx()
    if args.adaptive_tiling:
        tiling_decisions = _build_adaptive_tiling_decisions_for_mineru(
            images_dir=images_dir,
            decisions_json=args.tiling_decisions_json,
            min_subcomponents=args.tiling_min_subcomponents,
            min_aspect_ratio=args.tiling_min_aspect_ratio,
        )
        print(f"[adapter/mineru] adaptive tiling enabled for {len(tiling_decisions)} pages")

    prepared_image_crops = _prepare_image_crops_before_build(
        layout_dir=layout_dir,
        crops_out_dir=crops_out,
        adaptive_tiling=args.adaptive_tiling,
        tiling_decisions=tiling_decisions,
        tiling_overlap=args.tiling_overlap,
        tiling_max_tiles=args.tiling_max_tiles,
    )
    print(f"[adapter/mineru] prepared image blocks before build: {len(prepared_image_crops)} docs")

    crops = build_parsed_documents(
        layout_dir,
        images_dir,
        pd_out,
        crops_out,
        prepared_image_crops=prepared_image_crops,
    )

    if not args.skip_caption and crops:
        # img_paths = [str(p) for p in crops]
        # out_paths = [str(p.with_suffix(".txt")) for p in crops]
        # caption_images(
        #     image_paths=img_paths,
        #     output_paths=out_paths,
        #     prompt=SUMMARY_PROMPT,
        #     max_tokens=args.max_tokens,
        #     num_gpus=(args.num_gpus or None),
        # )
        merge_subimage_captions(pd_out, crops_out)
    elif not crops:
        print("[adapter/mineru] no image blocks → no Qwen-VL pass needed")

    _create_subimage_folder(crops_out, subimage_dir)

if __name__ == "__main__":
    main()

    # layout_dir = Path("/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/mineru_outputs_pipeline/InfoVQA/mineru_outputs_pipeline/dev")
    # images_dir = Path("datasets/InfoVQA/image_components/dev")
    # pd_out = Path("datasets/InfoVQA/parsed_documents/dev")
    # crops_out = Path("artifacts/InfoVQA/crops_out/dev")
    # subimage_dir = Path("artifacts/InfoVQA/image_components_sub/dev")
    # subimage_summaries_dir = Path("artifacts/InfoVQA/image_summaries_sub/dev")

    # crops = build_parsed_documents(layout_dir, images_dir, pd_out, crops_out)