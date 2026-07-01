from pathlib import Path
import json
import logging
import os


subquery_with_modality_file = "artifacts/InfoVQA/query_decomposition/dev/subqueries_with_modality.json"
revised_subquery_with_modality_file = 'artifacts/InfoVQA/query_decomposition/dev/revised_subqueries_with_modality.json'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join('debug', 'subquery_with_modality.log'),
                    filemode='a'
                    )

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

if __name__ == "__main__":
    # main()
    print(check_for_missing_modality(revised_subquery_with_modality_file))