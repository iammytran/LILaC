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
  i_1_c<N>           table cell

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
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

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
        image_caption = ".".join(block.get("image_caption", []))
        image_footnote = ".".join(block.get("image_footnote", []))
        return block.get("content") or image_caption or image_footnote or block.get("md") or ""
    return None


def _block_image_path(block: Dict, content_list_dir: Path) -> Optional[Path]:
    rel = block.get("img_path") or block.get("image_path")
    if not rel:
        return None
    p = (content_list_dir / rel).resolve()
    return p if p.exists() else None

def _process_cell_preserving_tags(cell):
    parts = []
    # Duyệt qua từng phần tử con bên trong ô <td> theo đúng thứ tự đọc từ trái qua phải
    for content in cell.contents:
        if content.name == "img":
            # Nếu là thẻ img, lấy thuộc tính src và bọc nó lại thành dạng text đại diện
            img_src = content.get("src", "")
            parts.append(f" <Image: {img_src}> ")
        else:
            # Nếu là text thuần, chỉ cần ép kiểu về chuỗi và clear khoảng trắng thừa
            text_inside = str(content).strip()
            if text_inside:
                parts.append(text_inside)
                
    # Gộp lại thành một chuỗi duy nhất của ô đó
    return "".join(parts).strip()

def _extract_text_and_image(raw_text):
    # Pattern để tìm cấu trúc <Image: ...>
    img_pattern = r"<Image:\s*(.*?)>"
    
    # 1. Tìm image source
    match = re.search(img_pattern, raw_text)
    img_src = match.group(1).strip() if match else None
    
    # 2. Xóa cụm <Image: ...> ra khỏi chuỗi để lấy phần text sạch
    # Hàm re.sub sẽ thay thế cụm <Image: ...> bằng chuỗi rỗng '', đồng thời .strip() để dọn khoảng trắng thừa
    text_content = re.sub(img_pattern, "", raw_text).strip()
    
    # Thu gọn các khoảng trắng bị thừa ở giữa (nếu ảnh nằm giữa chữ)
    text_content = re.sub(r"\s+", " ", text_content)
    
    return text_content, img_src

def _parse_html_to_table(doc_id, html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    table = soup.find('table')

    # Trường hợp không tìm thấy thẻ table để tránh crash
    if not table:
        return []
        
    # 1. DUYỆT QUA TỪNG Ô ĐỂ LÀM SẠCH TEXT TRƯỚC (Giữ nguyên logic custom của bạn)
    for cell in table.find_all(["td", "th"]):
        raw_content = _process_cell_preserving_tags(cell)
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

def _add_dla_idx():
    mineru_outputs_dir = Path("artifacts/InfoVQA/mineru_outputs/dev")
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

def _crop_fname(crops_out_dir, doc_id, img_src, counter, key):
    crop_fname = f"{doc_id}___{key}_{counter}{Path(img_src).suffix or '.jpg'}"
    target_crop = crops_out_dir / doc_id / crop_fname
    target_crop.parent.mkdir(parents=True, exist_ok=True)
    if not target_crop.exists():
        shutil.copyfile(img_src, target_crop)
    return crop_fname

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

    # keys_to_keep = ['10090']

    # # Create the subset (ignoring keys that might not exist)
    # content_lists = {k: content_lists_full[k] for k in keys_to_keep if k in content_lists_full}
    print(f"[adapter/mineru] {len(content_lists)} content_list.json under {layout_dir}")

    crops_to_summarize: List[Path] = []
    type_counts: Dict[str, int] = {}

    for doc_id, cl_path in content_lists.items():
        with open(cl_path, "r", encoding="utf-8") as f:
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
                "dla_idx": v if (v := block.get("dla_idx")) is not None else block.get("idx"),
                "dla_type": btype,
                "_bbox": block.get("bbox"),
            }

            native_text = _block_text(block)
            if native_text is None:
                continue  # unknown block with no text — skip
            
            if btype in IMAGE_TYPES:
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
                    "caption": {"text": native_text, "edges": []},
                }
                crops_to_summarize.append(target_crop)
                id_sequence.append(cid)
                continue

            if btype in TABLE_TYPES:
                src_img = _block_image_path(block, cl_path.parent)
                if src_img is None:
                    continue  
                t_counter += 1
                cid = f"i_1_t{t_counter}"
                crop_fname = f"{doc_id}___t_{t_counter}{src_img.suffix or '.jpg'}"
                target_crop = crops_out_dir / doc_id / crop_fname
                target_crop.parent.mkdir(parents=True, exist_ok=True)
                if not target_crop.exists():
                    shutil.copyfile(src_img, target_crop)
                # rows = _parse_html_to_table(doc_id, native_text)

                # table_data = []
                # c_counter = 1
                # for row in rows:
                #     row_data = []
                #     for cell in row:
                #         if "<Image" in cell:
                #             text, img_data = _extract_text_and_image(cell)
                #             src_img = cl_path.parent / img_data
                #             row_data.append({
                #                 "text": text,
                #                 "image": {"filename": _crop_fname(crops_out_dir, doc_id, src_img, c_counter, 'c')}
                #             })
                #             c_counter += 1  # Clean, readable incrementation
                #         else:
                #             row_data.append({"text": cell})
                #     table_data.append(row_data)

                table_entries[cid] = {
                    **common_meta,
                    "text": native_text,
                    "table_caption": block.get('table_caption', ''), 
                    "filename": crop_fname,
                    "edges": [],
                }
            else:  # text-like
                # if "no text" in native_text.lower():
                #     break
                p_counter += 1
                cid = f"i_1_p{p_counter}"
                text_entries[cid] = {
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
            "table": {},
            "image": {
                "i_1": {
                    "filename": page_img.name,
                    "caption": {"text": "", "edges": []},
                }
            },
            "sentence": text_entries,
            "proposition": {},
            "table_segment": table_entries,
            "subimage": subimage_entries,
            "id_to_html": {},
        }
        
        # parsed_doc = transform_text_to_sentence(parsed_doc)
        # parsed_doc = transform_table_to_table_segment(parsed_doc)

        with open(parsed_documents_out_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(parsed_doc, f, indent=4, ensure_ascii=False)

    if type_counts:
        print(f"[adapter/mineru] block type counts: {type_counts}")
    return crops_to_summarize

def transform_text_to_sentence(parsed_doc):
    text_entries = parsed_doc.get("text", {}) 
    sentence_entries = {}
    
    for block_id, block_content in text_entries.items():
        st_counter = 0
        text = block_content.get("text", "")
        # Split by period
        sentences = text.split(".")
        for sentence in sentences:
            if sentence.strip() == "":
                continue
            st_counter += 1
            parent_id = f"{block_id}"
            sentence_id = f"{parent_id}_st{st_counter}"
            sentence_entries[sentence_id] = {
                "dla_type": "text",
                "text": sentence.strip(),
                "edges": [],
            }

    parsed_doc["sentence"] = sentence_entries
    return parsed_doc

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
                "filename": imagename,
                "edges": [],
            }

    parsed_doc["table_segment"] = table_segment_entries

    return parsed_doc

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

def merge_subimage_captions(parsed_documents_out_dir: Path, crops_out_dir: Path) -> None:
    """Read per-crop .txt sidecars and merge/append into subimage[*].caption.text."""
    files = sorted(parsed_documents_out_dir.glob("*.json"))
    merged, missing = 0, 0
    
    for pd_path in files:
        with open(pd_path) as f:
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
                new_text = txt_path.read_text(encoding="utf-8").strip()
                
                # Đảm bảo key "caption" tồn tại và là một dictionary
                if "caption" not in sv or not isinstance(sv["caption"], dict):
                    sv["caption"] = {}
                
                # Lấy text hiện tại ra (nếu không có hoặc là None thì mặc định là chuỗi rỗng "")
                current_text = sv["caption"].get("text") or ""
                
                if current_text:
                    # Nếu đã có text cũ: tiến hành APPEND (nối thêm vào, cách nhau bằng 1 khoảng trắng)
                    # Bạn có thể đổi " " thành "\n" nếu muốn xuống dòng khi append
                    sv["caption"]["text"] = f"{current_text}\n{new_text}"
                else:
                    # Nếu chưa có text (hoặc text rỗng): ĐIỀN MỚI hoàn toàn
                    sv["caption"]["text"] = new_text
                    
                merged += 1
                any_updated = True
            else:
                missing += 1
                
        if any_updated:
            with open(pd_path, "w") as f:
                json.dump(pd, f, indent=4)
                
    print(f"[adapter/mineru] processed {merged} image captions; {missing} crops still uncaptioned")

def _create_subimage_folder(crops_out_dir, subimage_dir):
    subimage_dir.mkdir(parents=True, exist_ok=True)

    for img in crops_out_dir.rglob("*.jpg"):
        dest_dir = subimage_dir / img.name
        shutil.copy2(img, dest_dir)
    return 

def _create_subimage_summaries_folder(crops_out_dir, subimage_summaries_dir):
    subimage_summaries_dir.mkdir(parents=True, exist_ok=True)

    for text in crops_out_dir.rglob("*.txt"):
        dest_dir = subimage_summaries_dir / text.name
        shutil.copy2(text, dest_dir)
    return 

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
    subimage_dir = Path("artifacts/InfoVQA/image_components_sub/dev")
    subimage_summaries_dir = Path("artifacts/InfoVQA/image_summaries_sub/dev")

    # merge_subimage_captions(pd_out, crops_out)
    _add_dla_idx()
    crops = build_parsed_documents(layout_dir, images_dir, pd_out, crops_out)

    # if not args.skip_caption and crops:
    #     img_paths = [str(p) for p in crops]
    #     out_paths = [str(p.with_suffix(".txt")) for p in crops]
    #     caption_images(
    #         image_paths=img_paths,
    #         output_paths=out_paths,
    #         prompt=SUMMARY_PROMPT,
    #         max_tokens=args.max_tokens,
    #         num_gpus=(args.num_gpus or None),
    #     )
    #     # merge_subimage_captions(pd_out, crops_out)
    # elif not crops:
    #     print("[adapter/mineru] no image blocks → no Qwen-VL pass needed")

    # _create_subimage_folder(crops_out, subimage_dir)
    # _create_subimage_summaries_folder(crops_out, subimage_summaries_dir)


if __name__ == "__main__":
    main()
    # _create_subimage_summaries_folder(crops_out, subimage_summaries_dir)
    # _create_subimage_folder()

    # layout_dir = Path("artifacts/InfoVQA/mineru_outputs/dev")
    # images_dir = Path("datasets/InfoVQA/image_components/dev")
    # pd_out = Path("datasets/InfoVQA/parsed_documents/dev")
    # crops_out = Path("artifacts/InfoVQA/crops_out/dev")
    # subimage_dir = Path("artifacts/InfoVQA/image_components_sub/dev")
    # subimage_summaries_dir = Path("artifacts/InfoVQA/image_summaries_sub/dev")


    # _create_subimage_folder(crops_out, subimage_dir)
    # _create_subimage_summaries_folder(crops_out, pd_out, subimage_summaries_dir)

    # crops = build_parsed_documents(layout_dir, images_dir, pd_out, crops_out)