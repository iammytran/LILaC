from pathlib import Path
import json
from bs4 import BeautifulSoup

mineru_directory_name = Path("mineru_outputs")
content_list_name_end = "content_list_v2.json"
middle_name_end = "middle.json"

def main():
    infographics = [f for f in mineru_directory_name.iterdir() if f.is_dir()]
    # read file mineru_outputs
    for infographic in infographics:
        infographic_name = infographic.name

        print(infographic_name)

        content_v2_file = [f for f in infographic.iterdir() if f.is_file() and f.name.endswith(content_list_name_end)][0]

        with open(Path(content_v2_file), "r") as file:
            content = json.load(file)

        table_items = {}
        table_segment_items = {}
        
        table_blocks = [t for t in content[0] if t['type'] == 'table']

        # parse html to table
        for i, table_block in enumerate(table_blocks, start=1):
            html = table_block['content']['html']
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')
            column_headers = [header.text.strip() for header in table.find_all('tr')[0]]
            rows_body = []
            rows = table.find_all('tr')[1:]
            for row in rows:
                cells = [td.text.strip() for td in row.find_all('td')]
                rows_body.append(cells)
            
            table = {
                f"t_{i}": {
                    "table_caption": "A caption for the table.",
                    "columns": column_headers,
                    "rows": len(rows_body),
                    "table": rows_body,
                    "refs": []
                }
            }

            table_segment = {
                f't_{i}_s{row_i}': {
                     "table_caption": "A caption for the table.",
                    "columns": column_headers,
                    "rows": len(rows_body),
                    "table": [
                        column_headers,
                        row_body    
                    ],
                    "refs": []
                }
                for row_i, row_body in enumerate(rows_body)
            }

            table_items.update(table)
            table_segment_items.update(table_segment, start=1)

        # create table and table segment item
        print(table_items)
        
        # write back into parsed_document json
        parsed_json_path = "/Users/mytnguyen/Desktop/my-lilac/LILaC/datasets/InfoVQA/parsed_documents/dev"
        parsed_json_path2 = f"{parsed_json_path}/{infographic_name}.json"
        with open(parsed_json_path2, 'r', encoding="utf-8") as file:
            parsed_json_content = json.load(file)

        parsed_json_content['table'] = table_items
        parsed_json_content['table_segment'] = table_segment_items

        with open(parsed_json_path2, 'w', encoding="utf-8") as file:
            json.dump(parsed_json_content, file, indent=2)

    return 

if __name__ == "__main__":
    main()