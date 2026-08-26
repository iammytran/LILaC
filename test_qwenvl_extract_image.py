import json
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


import json
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from json_repair import repair_json


def parse_infographic_qwen2_5(
    image_path: str | Path,
    output_dir: str | Path = "artifacts/post_mineru",
    model_id: str = "models/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    images_crop_dir = output_dir / "images"
    images_crop_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    prompt_text = (
        "You are an expert document layout parser. Perform fine-grained visual component extraction. "
        "Extract every distinct text group, header, and visual/icon block. "
        "Return ONLY a valid JSON array of objects. "
        "Each object must have:\n"
        "- 'type': 'text' | 'header' | 'footer' | 'image' | 'table'\n"
        "- 'text': verbatim text string (empty string for image type)\n"
        "- 'bbox': [x1, y1, x2, y2] using absolute image pixel coordinates."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 2. Inference với Greedy Decoding (tắt sampling để cố định kết quả)
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,  # Cố định output
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 3. Parse JSON an toàn
    try:
        raw_blocks = repair_json(raw_output, return_objects=True)
        if not isinstance(raw_blocks, list):
            raw_blocks = [raw_blocks]
    except Exception as e:
        print(f"[Error] Không thể parse JSON: {e}")
        return []

    # 4. Mở ảnh gốc để tiến hành crop và lưu các sub-images
    orig_img = Image.open(image_path).convert("RGB")
    img_w, img_h = orig_img.size

    dla_blocks = []
    for idx, item in enumerate(raw_blocks, start=1):
        raw_bbox = item.get("bbox", item.get("bbox_2d", [0, 0, 0, 0]))
        if len(raw_bbox) != 4:
            continue

        # Giới hạn tọa độ trong khung kích thước ảnh
        x1 = max(0, min(int(raw_bbox[0]), img_w))
        y1 = max(0, min(int(raw_bbox[1]), img_h))
        x2 = max(0, min(int(raw_bbox[2]), img_w))
        y2 = max(0, min(int(raw_bbox[3]), img_h))

        # Đảm bảo box hợp lệ (x1 < x2, y1 < y2)
        if x2 <= x1 or y2 <= y1:
            continue

        block_type = item.get("type", "text")
        block_entry = {
            "type": block_type,
            "bbox": [x1, y1, x2, y2],
            "page_idx": 0,
            "dla_idx": idx,
        }

        # NẾU LÀ IMAGE: Cắt và lưu file ảnh thực tế xuống thư mục
        if block_type == "image":
            crop_filename = f"{image_path.stem}_crop_{idx}.jpg"
            crop_save_path = images_crop_dir / crop_filename

            # Thực hiện crop và lưu
            cropped_img = orig_img.crop((x1, y1, x2, y2))
            cropped_img.save(crop_save_path, quality=95)

            block_entry["img_path"] = f"images/{crop_filename}"
            block_entry["image_caption"] = []
            block_entry["image_footnote"] = []
        else:
            block_entry["text"] = item.get("text", "")

        dla_blocks.append(block_entry)

    # 5. Lưu file kết quả JSON
    output_json_path = output_dir / f"{image_path.stem}_reprocessed.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dla_blocks, f, ensure_ascii=False, indent=4)

    print(f"✅ Hoàn tất! Đã lưu {len(dla_blocks)} components và các ảnh con vào '{images_crop_dir}'.")
    return dla_blocks


if __name__ == "__main__":
    input_image = "43600.jpeg"
    result = parse_infographic_qwen2_5(input_image)
    
    output_path = Path("outputs/output_reprocessed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_path, "w", encoding="utf-8"):
    #     json.dumps(result, ensure_ascii=False, indent=4)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    
    print(f"Đã trích xuất {len(result)} components. Kết quả lưu tại: {output_path}")