from pathlib import Path
import json
import logging
import os

subquery_file = "artifacts/InfoVQA/query_decomposition/dev/subqueries.json"

revised_subquery_file_1 = "artifacts/InfoVQA/query_decomposition/dev/revised_subqueries_1.json"
revised_subquery_file = "artifacts/InfoVQA/query_decomposition/dev/revised_subqueries.json"
subquery_with_modality_file = "artifacts/InfoVQA/query_decomposition/dev/subqueries_with_modality.json"
revised_subquery_with_modality_file = 'artifacts/InfoVQA/query_decomposition/dev/revised_subqueries_with_modality.json'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('debug', 'subquery_with_modality.log'),
                    filemode='a'
                    )
subqueries_txt = "debug/compare_subqueries.txt"

def main():
    # load dict content
    content = ""
    with open(subquery_with_modality_file, 'r') as file:
        content = json.load(file)

    queries_not_have_modality = []
    subqueries_not_have_modality = []
    query_list = []

    # get subquery list
    # get query list
    for query in content.values():
        query_list.append(query)
        subquery_items = query['subqueries']
        subqueries = [sub_q['subquery'] for sub_q in subquery_items]
        # print(subqueries)
        for subquery in subquery_items:
            if 'modality' not in subquery:
                queries_not_have_modality.append(query)
                subqueries_not_have_modality.append(subquery)

    all_subqueries_without_modality_texts = [subquery['subquery'] for query in queries_not_have_modality for subquery in query['subqueries']]
    for index, subquery in enumerate(subqueries_not_have_modality):
        subquery_text = subquery['subquery']
        if all_subqueries_without_modality_texts.count(subquery_text) > 1:
            # find the query that it belong
            query_the_subquery_belongs = queries_not_have_modality[index]
            query_id_the_subquery_belongs = query_the_subquery_belongs['qid']
            
            # delete the subquery out of the subquery_list of the query
            revised_subquery_list = [subquery for subquery in query_the_subquery_belongs['subqueries'] if 'modality' in subquery]

            # put back the query to obj content
            content[f'{query_id_the_subquery_belongs}']['subqueries']= revised_subquery_list

    # print(f"len(subqueries_not_have_modality): {len(subqueries_not_have_modality)}")
    # print(f"subqueries_not_have_modality: {subqueries_not_have_modality[0]}")

    with open(revised_subquery_with_modality_file, 'w') as file:
        json.dump(content, file, indent=2)

def check_for_missing_modality(subqueries_with_modality_file_path):
    content = ""
    with open(subqueries_with_modality_file_path, 'r') as file:
        content = json.load(file)

    queries_not_have_modality = []
    subqueries_not_have_modality = []
    query_list = []

    # get subquery list
    # get query list
    for query in content.values():
        query_list.append(query)
        subquery_items = query['subqueries']
        subqueries = [sub_q['subquery'] for sub_q in subquery_items]
        # print(subqueries)
        for subquery in subquery_items:
            if 'modality' not in subquery:
                queries_not_have_modality.append(query)
                subqueries_not_have_modality.append(subquery)
    return not len(subqueries_not_have_modality) == 0

def compare_subqueries(subqueries_file, subqueries_with_modality_file):
    not_equal = 0
    # read file subquery
    subquery_content = {}
    with open(subqueries_file, "r") as file:
        subquery_content = json.load(file)

     # read file subquery_with_modality
    subquery_with_modality_content = {}
    with open(subqueries_with_modality_file, "r") as file:
        subquery_with_modality_content = json.load(file)

    # print(len(subquery_content.items()))
    # print(len(subquery_with_modality_content.items()))

    # print(len(subquery_content.items()))
    with open(subqueries_txt, 'w') as file:
        for subquery_content, subquery_with_modality_content in zip(subquery_content.values(), subquery_with_modality_content.values()):
            subquery_content_qid = subquery_content.get("qid", "")
            # subquery_with_modality_content_qid = subquery_with_modality_content.get("qid", "")
            subquery_content_subqueries = subquery_content.get("subqueries", [])
            subquery_with_modality_content_subqueries = subquery_with_modality_content.get("subqueries", [])
            if len(subquery_content_subqueries) != len(subquery_with_modality_content_subqueries):
                not_equal += 1
                file.write(str(subquery_content_qid))
                file.write("\n")

    print(not_equal)
    # print(len(subquery_with_modality_content.items()))
    return 

def check_for_subquery_duplicate():
    content = ""
    has_duplicate = {}
    dup = 0
    updated = 0

    with open(subquery_with_modality_file, 'r') as file:
        content = json.load(file)

    for qid, q_content in content.items():
        seen = set()
        subqueries = q_content.get("subqueries", [])
        # name_set = set()
        unique_subqueries = []
        for subquery in subqueries:
            subquery_name = subquery.get('subquery', '').lower()
            if subquery_name not in seen:
                seen.add(subquery_name)
                unique_subqueries.append(subquery)
            else:
                print(f"found duplicate for {qid}: {subquery}")
                updated += 1
        if unique_subqueries is not []:
            subqueries = unique_subqueries
        
        # content[qid]["subqueries"] = subqueries
                # updated += 1
        
        # if len(name_set) != len(subqueries):
        #     dup += 1
        #     has_duplicate[qid] = q_content

    # remove duplicate

    # with open(revised_subquery_file, 'w') as file:
    #     json.dump(content, file, indent = 4)
    

    
    # print(f"có {len(has_duplicate)} cases bị duplicate!")
    # print(has_duplicate)
    # assert len(name_set) === len(subqueries)



if __name__ == "__main__":
    # main()
    print(check_for_missing_modality(subquery_with_modality_file))
    # compare_subqueries(revised_subquery_file, revised_subquery_with_modality_file)
    # check_for_subquery_duplicate()