import os
import json
import copy

def process_parsed_documents(input_dir="artifacts/InfoVQA/parsed_documents_old/test", output_dir="artifacts/InfoVQA/parsed_documents/test"):
    # Tạo thư mục lưu kết quả nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Kiểm tra xem thư mục input có tồn tại không
    if not os.path.exists(input_dir):
        print(f"Thư mục '{input_dir}' không tồn tại. Vui lòng kiểm tra lại đường dẫn!")
        return

    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Sao chép sâu (deep copy) để giữ nguyên cấu trúc gốc
                new_data = copy.deepcopy(data)
                
                # Khởi tạo/Làm trống 'text' dictionary nếu cần, hoặc giữ cấu trúc cũ
                new_data["text"] = {}
                
                # Danh sách id_sequence mới để cập nhật lại các khóa tương ứng
                new_id_sequence = []
                
                # Duyệt qua id_sequence cũ để xử lý thay đổi ID của sentence -> text
                old_id_sequence = data.get("id_sequence", [])
                
                for item_id in old_id_sequence:
                    if item_id in data.get("sentence", {}):
                        # Đổi ID từ dạng kết thúc bằng _pX sang _sX (vd: i_1_p1 -> i_1_s1)
                        # Hoặc thay thế phần '_p' cuối cùng bằng '_s'
                        new_item_id = item_id.rsplit('_p', 1)
                        if len(new_item_id) == 2:
                            new_item_id = f"{new_item_id[0]}_s{new_item_id[1]}"
                        else:
                            new_item_id = item_id.replace('_p', '_s') # Fallback an toàn
                        
                        # Copy nội dung từ sentence sang text với ID mới
                        new_data["text"][new_item_id] = copy.deepcopy(data["sentence"][item_id])
                        new_id_sequence.append(new_item_id)
                    else:
                        # Nếu không thuộc sentence (ví dụ ảnh, bảng, v.v.), giữ nguyên ID cũ
                        new_id_sequence.append(item_id)
                
                # Cập nhật lại id_sequence mới trong dict đích
                new_data["id_sequence"] = new_id_sequence
                
                # Ghi ra file JSON mới ở thư mục output
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=4)
                    
                print(f"Đã xử lý xong: {filename} -> Lưu tại {output_dir}/")
                
            except Exception as e:
                print(f"Lỗi khi xử lý file {filename}: {e}")

if __name__ == "__main__":
    # Bạn có thể thay đổi tên thư mục input/output tùy ý ở đây
    process_parsed_documents()