import json
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_infographic_with_qwen(
    image_path: str | Path,
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict]:
    image_path = Path(image_path)
    
    # 1. Khởi tạo Model & Processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Lấy kích thước ảnh gốc để quy đổi tọa độ
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    # 3. Prompt chuẩn hóa yêu cầu trích xuất layout
    prompt_text = (
        "Extract all semantic visual blocks, text blocks, and standalone image/icon components from this infographic. "
        "Return ONLY a valid JSON array of objects. Do not include markdown code block formatting (e.g. no ```json). "
        "Each object must have exactly these keys:\n"
        "- 'type': string ('header', 'text', 'image', 'footer', or 'table')\n"
        "- 'text': verbatim text content string (empty string for pure image type)\n"
        "- 'bbox_2d': [ymin, xmin, ymax, xmax] normalized to 0-1000 scale."
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

    # 4. Tiền xử lý inputs
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 5. Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 6. Parse JSON & chuyển đổi tọa độ sang chuẩn MinerU/Lilac [x1, y1, x2, y2]
    cleaned_json_str = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE)
    try:
        raw_blocks = json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        print("[Error] Không thể parse trực tiếp JSON từ model, output thô:", raw_output)
        return []

    dla_blocks = []
    for idx, item in enumerate(raw_blocks, start=1):
        # Qwen2-VL thường trả về [ymin, xmin, ymax, xmax] tỉ lệ 1000
        ymin, xmin, ymax, xmax = item.get("bbox_2d", [0, 0, 0, 0])
        
        # Quy đổi về pixel ảnh gốc [x1, y1, x2, y2]
        x1 = int((xmin / 1000.0) * img_w)
        y1 = int((ymin / 1000.0) * img_h)
        x2 = int((xmax / 1000.0) * img_w)
        y2 = int((ymax / 1000.0) * img_h)

        block_entry = {
            "type": item.get("type", "text"),
            "bbox": [x1, y1, x2, y2],
            "page_idx": 0,
            "dla_idx": idx,
        }

        if item.get("type") == "image":
            block_entry["img_path"] = f"images/{image_path.stem}_{idx}.jpg"
            block_entry["image_caption"] = []
            block_entry["image_footnote"] = []
        else:
            block_entry["text"] = item.get("text", "")

        dla_blocks.append(block_entry)

    return dla_blocks


if __name__ == "__main__":
    input_image = "infographic_example.png"
    result = parse_infographic_with_qwen(input_image)
    
    output_path = Path("artifacts/post_mineru/output_reprocessed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    
    print(f"Đã trích xuất {len(result)} components. Kết quả lưu tại: {output_path}")