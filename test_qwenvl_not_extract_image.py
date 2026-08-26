import json
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# import json
# from pathlib import Path
# from PIL import Image
# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from json_repair import repair_json


def parse_infographic_qwen2_5(
    image_path: str | Path,
    model_id: str = "models/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict]:
    image_path = Path(image_path)
    
    # 1. Khởi tạo Model & Processor cho Qwen2.5-VL
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Prompt chỉ định rõ tọa độ pixel [x1, y1, x2, y2]
    prompt_text = (
        "Extract all semantic text and visual components from this infographic. "
        "Return ONLY a valid JSON array of objects. "
        "Each object must have these keys:\n"
        "- 'type': 'text' | 'header' | 'footer' | 'image' | 'table'\n"
        "- 'text': verbatim text content (empty string for image type)\n"
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

    # 3. Tiền xử lý inputs
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

    # 4. Tăng max_new_tokens lên 4096 để không bị ngắt giữa chừng
    with torch.no_grad():
        generated_ids = model.generate(**inputs, 
                                        max_new_tokens=4096, 
                                        do_sample=False,        # Tắt sampling (dùng Greedy search)
                                        temperature=None,       # Không dùng temperature
                                        top_p=None,)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 5. Dùng json_repair để vá JSON an toàn nếu chạm token limit
    try:
        raw_blocks = repair_json(raw_output, return_objects=True)
        if not isinstance(raw_blocks, list):
            raw_blocks = [raw_blocks]
    except Exception as e:
        print(f"[Error] Không thể parse output: {e}")
        return []

    # 6. Map dữ liệu về schema MinerU / Lilac
    dla_blocks = []
    for idx, item in enumerate(raw_blocks, start=1):
        bbox = item.get("bbox", item.get("bbox_2d", [0, 0, 0, 0]))
        
        # Nếu model trả [ymin, xmin, ymax, xmax] -> đổi sang [x1, y1, x2, y2]
        # x1, y1, x2, y2 = bbox[1], bbox[0], bbox[3], bbox[2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

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
    input_image = "43600.jpeg"
    result = parse_infographic_qwen2_5(input_image)
    
    output_path = Path("outputs/output_reprocessed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_path, "w", encoding="utf-8"):
    #     json.dumps(result, ensure_ascii=False, indent=4)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    
    print(f"Đã trích xuất {len(result)} components. Kết quả lưu tại: {output_path}")