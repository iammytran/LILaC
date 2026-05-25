import json
from pathlib import Path

mineru_directory_name = Path("mineru_outputs")
content_list_name_end = "content_list_v2.json"
middle_name_end = "middle.json"

current_file = Path(__file__).parent

def main():
    infographics = [f for f in mineru_directory_name.iterdir() if f.is_dir()]
    for infographic in infographics:
        infographic_name = infographic.name
        print(infographic_name)

        # read file middle.json
        middle_file = [f for f in infographic.iterdir() if f.is_file() and f.name.endswith(content_list_name_end)][0]
        with open(Path(middle_file), "r", encoding="utf-8") as file:
            content = json.load(file)
        try:
            blocks = content[0]
        except (KeyError, IndexError, TypeError):
            blocks = []
        # print(para_blocks)
        text_items = []
        for block in blocks:
            block_type = block['type']
            block_content = block['content']
            dynamic_key = f"{block_type}_content"
            block_type_content_value = block_content.get(dynamic_key, [])
            if block_type_content_value != []:
                if block_type_content_value[0]['type'] and block_type_content_value[0]['type'] == 'text':
                    item_content = block_type_content_value[0]['content']
                    if item_content.strip() !=  "[No text]":
                        text_items.append(item_content)
                
        print(text_items)


        # write text into file parsed json
        text_field = {
            f"p_{i}": {"text": str(s).strip(), "edges": []}
            for i, s in enumerate(text_items, start=1)
        }
        parsed_json_path = "/Users/mytnguyen/Desktop/my-lilac/LILaC/datasets/InfoVQA/parsed_documents/dev"
        parsed_json_path2 = f"{parsed_json_path}/{infographic_name}.json"
        # lilac_root = current_file.parents[2]

        # # 3. Đi vào thư mục chứa các file JSON bằng toán tử /
        # json_dir = lilac_root / "datasets" / "parsed_document"

        # # 4. Chỉ định chính xác file JSON bạn muốn đọc
        # target_file = json_dir / "10002.json"

        with open(parsed_json_path2, 'r', encoding="utf-8") as file:
            parsed_json_content = json.load(file)

        parsed_json_content['text'] = text_field
        # split text into sentences
        sentence_field = {
            f"s_{i}": {"text" : str(s).strip(), "edges": []}
            for i, s in enumerate(text_items, start=1)
        }
        parsed_json_content['sentence'] = sentence_field

        with open(parsed_json_path2, 'w', encoding="utf-8") as file:
            json.dump(parsed_json_content, file, indent=2)
    return 

if __name__ == "__main__":
    main()
