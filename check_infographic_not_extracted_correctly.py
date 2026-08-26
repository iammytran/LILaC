import argparse
import json
from pathlib import Path


def export_under_extracted_doc_ids(
    mineru_outputs_dir: str | Path = "artifacts/InfoVQA/mineru_outputs_pipeline/test",
    min_components: int = 5,
    output_txt_path: str | Path = "debug/under_extracted_ids.txt",
):
    mineru_dir = Path(mineru_outputs_dir)
    if not mineru_dir.exists():
        print(f"❌ Thư mục '{mineru_dir}' không tồn tại!")
        return []

    # Quét tất cả file *_content_list.json
    content_list_files = sorted(mineru_dir.rglob("*_content_list.json"))
    under_extracted_ids = []

    for file_path in content_list_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Lấy danh sách blocks
            if isinstance(data, list):
                blocks = data
            elif isinstance(data, dict):
                blocks = data.get("content_list", [])
            else:
                blocks = []

            # Nếu số lượng component < min_components thì lấy ID
            if len(blocks) < min_components:
                # Tách lấy ID (vd: 23087 từ 23087_content_list.json)
                doc_id = file_path.name.replace("_content_list.json", "")
                under_extracted_ids.append(doc_id)

        except Exception as e:
            print(f"⚠️ Lỗi đọc file {file_path.name}: {e}")

    # Ghi danh sách doc_id vào file .txt (mỗi ID một dòng)
    out_path = Path(output_txt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(under_extracted_ids), encoding="utf-8")

    print(f"✅ Đã quét xong {len(content_list_files)} files.")
    print(f"📄 Tìm thấy {len(under_extracted_ids)} docs có < {min_components} components.")
    print(f"💾 Đã lưu danh sách ID vào: {out_path}")

    return under_extracted_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export under-extracted MinerU doc IDs")
    parser.add_argument(
        "--dir",
        default="artifacts/InfoVQA/mineru_outputs_pipeline/test",
        help="Đường dẫn folder chứa output mineru",
    )
    parser.add_argument(
        "--min_components",
        type=int,
        default=5,
        help="Ngưỡng component tối thiểu",
    )
    parser.add_argument(
        "--output_txt",
        default="debug/under_extracted_ids.txt",
        help="File txt xuất ra",
    )
    args = parser.parse_args()

    export_under_extracted_doc_ids(
        mineru_outputs_dir=args.dir,
        min_components=args.min_components,
        output_txt_path=args.output_txt,
    )