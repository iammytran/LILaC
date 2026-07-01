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
from itertools import islice
import os
import logging
import re

from src.lilac.lcg_constructor.preprocessing._caption_via_qwen_vl import caption_images

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     filename=os.path.join('debug', 'mineru_to_lilac.log'),
#                     filemode='a')


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
    for p in layout_dir.rglob("*_content_list.json"):
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
    if btype in IMAGE_TYPES:
        return (block.get("content") or block.get("md") or "") or None
    return None


def _block_image_path(block: Dict, content_list_dir: Path) -> Optional[Path]:
    rel = block.get("img_path") or block.get("image_path")
    if not rel:
        return None
    p = (content_list_dir / rel).resolve()
    return p if p.exists() else None


def build_parsed_documents(
    layout_dir: Path,
    images_dir: Path,
    parsed_documents_out_dir: Path,
    crops_out_dir: Path,
) -> List[Path]:
    """Build skeleton parsed_documents + image crops. Returns crops needing Qwen-VL summary."""
    parsed_documents_out_dir.mkdir(parents=True, exist_ok=True)
    crops_out_dir.mkdir(parents=True, exist_ok=True)
    content_lists = _find_content_lists(layout_dir)
    # content_lists = dict(islice(content_lists.items(), 30))
    print(f"[adapter/mineru] {len(content_lists)} content_list.json under {layout_dir}")

    crops_to_summarize: List[Path] = []
    type_counts: Dict[str, int] = {}

    for doc_id, cl_path in content_lists.items():
        # logging.info(f"doc_id: {doc_id}")
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
        text_entries: Dict[str, Dict] = {}
        table_entries: Dict[str, Dict] = {}
        id_sequence: List[str] = ["i_1"]

        for block in blocks:
            btype = block.get("type", "")
            type_counts[btype] = type_counts.get(btype, 0) + 1
            common_meta = {
                "dla_idx": block.get("dla_idx") or block.get("idx"),
                "dla_type": btype,
                "_bbox": block.get("bbox"),
            }

            native_text = _block_text(block)
            if native_text is None:
                continue  # unknown block with no text — skip

            if btype in IMAGE_TYPES:
                sub_type = block.get("sub_type", "")
                if btype == "chart":
                    native_text = f"A {sub_type} chart: {native_text}"
                if sub_type == "flowchart":
                    native_text = f"A {sub_type}: {native_text}"
                src_img = _block_image_path(block, cl_path.parent)
                if src_img is None:
                    continue
                i_counter += 1
                cid = f"i_1_i{i_counter}"
                crop_fname = f"{doc_id}___i_{i_counter}{src_img.suffix or '.jpg'}"
                target_crop = crops_out_dir / doc_id / crop_fname
                target_crop.parent.mkdir(parents=True, exist_ok=True)
                if not target_crop.exists():
                    shutil.copyfile(src_img, target_crop)
                subimage_entries[cid] = {
                    **common_meta,
                    "filename": crop_fname,
                    "caption": {"text": comprehensive_clean(native_text), "edges": []},
                }
                crops_to_summarize.append(target_crop)
                id_sequence.append(cid)
                continue

            if btype in TABLE_TYPES:
                # logging.info(f"block: {block} with btype as table")
                src_img = _block_image_path(block, cl_path.parent)
                if src_img is None:
                    # logging.info(f"src_img: {src_img} not exists")
                    continue  
                t_counter += 1
                cid = f"i_1_t{t_counter}"
                crop_fname = f"{doc_id}___t_{t_counter}{src_img.suffix or '.jpg'}"
                target_crop = crops_out_dir / doc_id / crop_fname
                target_crop.parent.mkdir(parents=True, exist_ok=True)
                if not target_crop.exists():
                    shutil.copyfile(src_img, target_crop)
                table_entries[cid] = {
                    **common_meta,
                    "text": comprehensive_clean(native_text),
                    "filename": crop_fname,
                    "edges": [],
                }
            else:  # text-like
                # logging.info(f"block: {block} with btype as text")
                p_counter += 1
                cid = f"i_1_p{p_counter}"
                text_entries[cid] = {
                    **common_meta,
                    "text": comprehensive_clean(native_text),
                    "edges": [],
                }
            id_sequence.append(cid)

        parsed_doc = {
            "title": doc_id,
            "hierarchy": {},
            "id_sequence": id_sequence,
            "header": {},
            "text": text_entries,
            "table": table_entries,
            "image": {
                "i_1": {
                    "filename": page_img.name,
                    "caption": {"text": "", "edges": []},
                }
            },
            "sentence": {},
            "proposition": {},
            "table_segment": {},
            "subimage": subimage_entries,
            "id_to_html": {},
        }

        # transform_text_to_sentence(parsed_doc)
        # transform_table_to_table_segment(parsed_doc)

        with open(parsed_documents_out_dir / f"{doc_id}.json", "w") as f:
            json.dump(parsed_doc, f, indent=4)

    if type_counts:
        print(f"[adapter/mineru] block type counts: {type_counts}")
    return crops_to_summarize

# TODO: transform texts to sentences
def transform_text_to_sentence(parsed_doc):
    text_entries = parsed_doc.get("text", {}) 
    sentence_entries = []
    # print(f"text_entries: {text_entries}")
    
    for text_block in text_entries.values():
        text = text_block.get("text", "")
        # Split by period
        sentences = text.split(".")
        for sent in sentences:
            sentence_entries.append(sent)
        # sentence_entries.append(sentence.strip() for sentence in sentences)

    print(sentence_entries)

    # Add sentence to sentence list

    return

def transform_table_to_table_segment(parsed_doc):
    return

# TODO: clean_text
def comprehensive_clean(text: str) -> str:
    if not text:
        return ""

    # 1. Hàm dịch chuẩn mọi mã Unicode (\u00e9, \u5e74, \u221a)
    def decode_unicode(match):
        val = match.group(0)
        try:
            # Dịch trực tiếp chuỗi \uXXXX thành ký tự thực tế
            return val.encode('utf-8').decode('unicode_escape')
        except (UnicodeDecodeError, ValueError):
            return "" # Nếu lỗi hoàn toàn (như \uabs) thì xóa bỏ

    # Quét chính xác cụm \u theo sau là đúng 4 ký tự hex (không phân biệt hoa thường)
    text = re.sub(r'\\u([0-9a-fA-F]{4})', decode_unicode, text)
    
    # Xóa bỏ các mã \u bị lỗi/bị cắt cụt (như \uabs, \u12) còn sót lại
    text = re.sub(r'\\u[a-zA-Z0-9]{1,3}', '', text)

    # 2. Xử lý các lỗi LaTeX đặc trưng của MinerU
    text = text.replace('\\%', '%').replace('$', '')
    text = text.replace('\\`', '`').replace('\\#', '#')

    # 3. Thay thế các ký tự xuống dòng, tab thành khoảng trắng
    text = re.sub(r'[\n\t\r]', ' ', text)

    # 4. Xóa các ký tự toán học/bullet lạ đứng một mình (như dấu √ ở câu trước)
    # Nhưng VẪN GIỮ LẠI các chữ có dấu hợp lệ (như chữ é trong Nestlé, hoặc tiếng Trung)
    # [^\w\s] trong Python 3 mặc định tự động giữ lại ký tự chữ của mọi ngôn ngữ (Unicode)
    text = re.sub(r'[^\w\s.,?!%\-\(\)\'\"]', '', text)

    # 5. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def merge_subimage_captions(parsed_documents_out_dir: Path, crops_out_dir: Path) -> None:
    """Read per-crop .txt sidecars and merge into subimage[*].caption.text."""
    files = sorted(parsed_documents_out_dir.glob("*.json"))
    files = files[:2]
    merged, missing = 0, 0
    for pd_path in files:
        with open(pd_path) as f:
            pd = json.load(f)
        doc_id = pd_path.stem
        print(f"doc_id: {doc_id}")
        any_updated = False
        for sid, sv in pd.get("subimage", {}).items():
            print(f"sid: {sid}")
            # 1. Lấy text hiện tại trong JSON ra trước (nếu không có thì mặc định là chuỗi rỗng "")
            current_text = (sv.get("caption") or {}).get("text", "")

            # 2. KIỂM TRA ĐIỀU KIỆN: Chỉ xử lý NẾU text hiện tại đang TRỐNG
            if not current_text:  
                print(f"text in parsed_doc is empty")
                crop_fname = sv.get("filename", "")
                if not crop_fname:
                    missing += 1
                    continue
                
                txt_path = (crops_out_dir / doc_id / crop_fname).with_suffix(".txt")
                
                if txt_path.exists():
                    content = txt_path.read_text(encoding="utf-8").strip()
                    
                    # (Tùy chọn) Chặn luôn nếu file .txt chứa chữ "None" hoặc bị trống rỗng
                    if content == "" or content == "None":
                        missing += 1
                        continue
                    
                    if "caption" not in sv or sv["caption"] is None:
                        sv["caption"] = {}
                        
                    # Tiến hành thêm vào vì chỗ này đang trống
                    sv["caption"]["text"] = content
                    merged += 1
                    any_updated = True
                else:
                    missing += 1
        if any_updated:
            with open(pd_path, "w") as f:
                json.dump(pd, f, indent=4)
    print(f"[adapter/mineru] merged {merged} image captions; {missing} crops still uncaptioned")


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
    args = ap.parse_args()

    layout_dir = Path(args.layout)
    images_dir = Path(args.images)
    pd_out = Path(args.output)
    crops_out = Path(args.crops_out)

    # merge_subimage_captions(pd_out, crops_out)

    crops = build_parsed_documents(layout_dir, images_dir, pd_out, crops_out)

    if not args.skip_caption and crops:
        img_paths = [str(p) for p in crops]
        out_paths = [str(p.with_suffix(".txt")) for p in crops]
        caption_images(
            image_paths=img_paths,
            output_paths=out_paths,
            prompt=SUMMARY_PROMPT,
            max_tokens=args.max_tokens,
            num_gpus=(args.num_gpus or None),
        )
        merge_subimage_captions(pd_out, crops_out)
    elif not crops:
        print("[adapter/mineru] no image blocks → no Qwen-VL pass needed")


if __name__ == "__main__":
    main()
    # content = ""
    # path = "datasets/InfoVQA/parsed_documents/dev/70436.json"
    # with open(path, 'r') as file:
    #     content = json.load(file)

    # transform_text_to_sentence(content)
