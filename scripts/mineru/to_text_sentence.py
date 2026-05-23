import json
from pathlib import Path

mineru_directory_name = Path("mineru_outputs")
content_list_name_end = "content_list_v2.json"
middle_name_end = "middle.json"


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
                    text_items.append(item_content)
                
        print(text_items)

            # inner_list = next(iter(block_content.values()))

            # # Bây giờ bạn có thể truy cập phần tử đầu tiên như bình thường
            # if inner_list:
            #     text_type = inner_list[0]["type"]
            #     text_content = inner_list[0]["content"]
            #     print(text_type, "->", text_content)

        # write text into file parsed json
        for id, i in enumerate(text_items):
            text_id = f'p_{}'

        # split text into sentences



    return 


if __name__ == "__main__":
    main()
