
import os
import re
import json

from src.lilac.basic_class.component import Component




class Text(Component):
    
    
    def __init__(self, 
        filename: str,
        document_title: str, 
        hierarchy_dict: dict, 
        component_id: str, 
        component_object: dict,
        images_dir: str = "",
        image_summaries_dir: str = ""
    ):

        super().__init__(filename, document_title, hierarchy_dict, component_id, component_object)
        self.text = self.component_obj["text"]
        
        return 


    

    def serialize(self, mode = None):
        serialized_metadata = f"{self.document_title} [SEP] {self.get_serialized_hierarchy_path()} [SEP] "

        serialization = serialized_metadata + self.text
        serialization = serialization.replace("\n", " ")
        serialization = serialization.replace("\t", " ")
        serialization = re.sub(r'(\s*\[SEP\]\s*)+', ' [SEP] ', serialization)
        
        embedding_obj = {
            "id": [self.filename, self.component_id],
            "target": {
                "text": serialization,
                "images": []
            }
        }

        return embedding_obj
    
    
    def serialize_into_chunks(self, mode = None, chunk_size = 512):
        
        raise(NotImplementedError("The method serialize_into_chunks() should not be used in Text class."))


    def serialize_into_prompt(self, next_image_idx):
                
        prompt = "/*\n"
        prompt += "[Passage]\n"
        prompt += "Title: " + self.document_title + "\n"
        prompt += "Section: " + ", ".join(self.hierarchy_path) + "\n\n"
        prompt += self.text
        prompt += "\n*/\n\n"
        
        return prompt, [], next_image_idx
    
    
    def get_text_object_for_split(self):
        
        obj = {
            "title": self.document_title,
            "section": self.hierarchy_path,
            "text": self.text
        }
        
        return obj
    
    
    def get_intra_edges_as_filenames_list(self):
        
        if "edges" not in self.component_obj:
            return []
        
        edges = []
        for edge_obj in self.component_obj["edges"]:
            edges.append(edge_obj["edge"])
        
        return edges

if __name__ == "__main__":
    filename = "datasets/InfoVQA/parsed_documents/dev/10002.json"
    with open(filename, 'r') as file:
        parsed_document = json.load(file)

    # print(parsed_document)
    image_dir = "datasets/InfoVQA/image_components/dev"
    summaries_dir = "artifacts/InfoVQA/image_summaries/dev"
    sub_image_dir = "artifacts/InfoVQA/image_components_sub/dev"
    subimage_summaries_dir = "artifacts/InfoVQA/image_components_sub/dev"

    # for mode in [["image", "summary"]]:
    #     doc_title = parsed_document["title"]
    #     hierarchy_dict  = parsed_document["hierarchy"]
    #     image_component_ids = parsed_document["image"]
    #     for component_id in image_component_ids:
    #         component = parsed_document["image"][component_id]
    #         image_component = Image(filename, doc_title, hierarchy_dict, component_id, component,
    #                                 images_dir = image_dir, image_summaries_dir = summaries_dir)
    #         print(image_component.serialize(mode))
        # image_serializations.append(image_component.serialize(mode))

    # subimage_serializations = []
    # for mode in [["image", "summary"]]:
    #     doc_title = parsed_document["title"]
    #     hierarchy_dict  = parsed_document["hierarchy"]
    #     subimage_component_ids = parsed_document["subimage"]
    #     for component_id in subimage_component_ids:
    #         component = parsed_document["subimage"][component_id]
    #         subimage_component = Image(filename, doc_title, hierarchy_dict, component_id, component,
    #                                     images_dir = sub_image_dir, image_summaries_dir = subimage_summaries_dir)
    #         subimage_serializations.append(subimage_component.serialize(mode))
    #     print(subimage_serializations[1])

    sentence_serializations = []
    doc_title = parsed_document["title"]
    hierarchy_dict  = parsed_document["hierarchy"]
    sentence_component_ids = parsed_document["sentence"]
    for component_id in sentence_component_ids:
        component = parsed_document["sentence"][component_id]
        sentence_component = Text(filename, doc_title, hierarchy_dict, component_id, component)
        sentence_serializations.append(sentence_component.serialize())
    print(sentence_serializations[0])