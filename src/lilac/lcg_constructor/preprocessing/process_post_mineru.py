from __future__ import annotations
from PIL import Image
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import json


POST_MINERU_DIR = Path("artifacts/post_mineru")

def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Lấy (width, height) của ảnh gốc."""
    with Image.open(image_path) as img:
        return img.size

def _is_oversized_figure(bbox: list, img_w: int, img_h: int, threshold_ratio: float = 0.6) -> bool:
    """Kiểm tra xem bbox có chiếm trên `threshold_ratio` diện tích toàn trang hay không."""
    if not bbox or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox
    block_area = abs(x2 - x1) * abs(y2 - y1)
    total_area = img_w * img_h
    return (block_area / total_area) >= threshold_ratio

def _reprocess_oversized_figure(
    doc_id: str, 
    crop_path: Path, 
    page_img_path: Path,
    original_bbox: list,
    post_mineru_out: Path
) -> List[Dict]:
    """
    Xử lý lại figure lớn: Bóc tách text và bounding box con bên trong.
    Kết quả được lưu vào artifacts/post_mineru/<doc_id>/
    """
    doc_post_dir = post_mineru_out / doc_id
    doc_post_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Lưu metadata/crop cần xử lý lại vào post_mineru
    saved_crop = doc_post_dir / crop_path.name
    if not saved_crop.exists() and crop_path.exists():
        shutil.copyfile(crop_path, saved_crop)

    # 2. Bóc tách text/sub-blocks từ ảnh crop
    # (Tại đây bạn có thể gọi PaddleOCR, EasyOCR hoặc VLM endpoint)
    # Ví dụ minh họa cấu trúc sub-blocks trả về:
    new_sub_blocks = []
    
    # Giả lập: Nếu bạn dùng OCR (như PaddleOCR/VLM) để lấy text:
    # ocr_results = run_ocr_on_crop(saved_crop)
    # for item in ocr_results:
    #     new_sub_blocks.append({
    #         "type": "text",
    #         "text": item["text"],
    #         "bbox": item["adjusted_bbox"], # Tọa độ map ngược về trang gốc
    #     })

    # Lưu kết quả post-processed JSON để kiểm tra
    post_json_path = doc_post_dir / f"{crop_path.stem}_reprocessed.json"
    with open(post_json_path, "w", encoding="utf-8") as f:
        json.dump(new_sub_blocks, f, indent=4, ensure_ascii=False)

    return new_sub_blocks