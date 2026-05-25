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

        # read file content_list.json
        middle_file = [f for f in infographic.iterdir() if f.is_file() and f.name.endswith(content_list_name_end)][0]
        with open(Path(middle_file), "r", encoding="utf-8") as file:
            content = json.load(file)

        # add image field
        image_item = {
            'im_1': {
                'filename': f"{infographic_name}.jpeg",
                'caption': "Placeholder",
                'source_url': f"datasets/InfoVQA/image_components/dev/{infographic_name}.jpeg"
            }
        }

        sub_image_items = {}

        # add subimage field
        sub_image_blocks = [block for block in content[0] if block['type']=='image']
        
        for i, sub_image_block in enumerate(sub_image_blocks, start=1):
            sub_image_path = sub_image_block['content']['image_source']['path']
            sub_image_name = sub_image_path.split('/')[-1]
            sub_image_item = {
                f'im_1_s{i}': {
                    'filename': f"{sub_image_name}",
                    'caption': {
                        'ocr': f"{sub_image_block['content']['content']}"
                    },
                    'source_url': f"mineru_outputs/{infographic_name}/{sub_image_path}" 
                }
            }

            sub_image_items.update(sub_image_item)
        
        # write into parsed document 
        parsed_json_path = "/Users/mytnguyen/Desktop/my-lilac/LILaC/datasets/InfoVQA/parsed_documents/dev"
        parsed_json_path2 = f"{parsed_json_path}/{infographic_name}.json"

        with open(parsed_json_path2, 'r', encoding="utf-8") as file:
            parsed_json_content = json.load(file)

        parsed_json_content['subimage'] = sub_image_items

        with open(parsed_json_path2, 'w', encoding="utf-8") as file:
            json.dump(parsed_json_content, file, indent=2)   
    return 


if __name__ == "__main__":
    main()