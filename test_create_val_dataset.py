from datasets import load_dataset
import pandas as pd
import json
import logging
import os
# import logger

val_dir = "infographicsVQA_val_v1.0_withQT copy.json"
visrag_test_dir = "openbmb/VisRAG-Ret-Test-InfoVQA"

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('debug', 'val_dataset.log'),
                    filemode='a'
                    )

logger = logging.getLogger(__name__)

def download_visrag_ret_test():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="openbmb/VisRAG-Ret-Test-InfoVQA",
        repo_type="dataset",
        local_dir="./VisRAG-Ret-Test-InfoVQA",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8
    )
    print("Đã tải toàn bộ repo về thư mục ./VisRAG-Ret-Test-InfoVQA")

def get_query_from_queries():
    # # Chỉ đọc duy nhất 1 cột 'query'
    # df = pd.read_parquet("VisRAG-Ret-Test-InfoVQA/queries/train-00000-of-00001.parquet", columns=["query"])

    # # Chuyển thành list
    # all_queries = df["query"].tolist()
    # return all_queries
    file_path = "VisRAG-Ret-Test-InfoVQA/queries/train-00000-of-00001.parquet"

    # Đọc chỉ 2 cột cần lấy (tiết kiệm bộ nhớ)
    df = pd.read_parquet(file_path, columns=["query-id", "query"])
    records = df.to_dict(orient="records")
    return records

def map_query_with_qa_val(queries_from_hf):
    qa_val_json = "datasets/InfoVQA/QAs_val.json"
    data = []
    output_data = []
    output_file = "datasets/QAs_val_718.json"

    # get qid and question from qa_val.json
    with open(qa_val_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # # Lọc chỉ lấy qid và question
    # filtered_list_qa_val = [{"qid": item["qid"], "question": item["question"]} for item in data]

    # tracing
    for query_hf in queries_from_hf:
        query_id = query_hf['query-id']
        query = query_hf['query']
        for query_in_qa_val in data:
            query_id_in_qa_val = query_in_qa_val['qid']
            query_str_in_qa_val = query_in_qa_val['question']
            if query.casefold() == query_str_in_qa_val.casefold():
                query_in_qa_val['qid'] = query_id
                output_data.append(query_in_qa_val)
                logging.info(f"For {query_id}: {query}, found in qa_val at {query_id_in_qa_val}")
            # if qu
            #     logging.info(f"For {query_id}: {query}, not found any match")

    with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
    return

def main():
    return

if __name__ == "__main__":
    # download_visrag_ret_test()
    query_records = get_query_from_queries()
    map_query_with_qa_val(query_records)