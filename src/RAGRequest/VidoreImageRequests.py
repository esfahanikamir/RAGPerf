from datasets import load_dataset, load_from_disk
from .TextsRAGRequest import WikipediaRequests

import os

class VidoreImageRequest(WikipediaRequests):
    def __init__(self, vidore_basepath, sub_dataset_name, run_name, collection_name, req_type, req_count):
        self.req_type = "query" # need to have this for later stages
        queries_folder_path = os.path.join(vidore_basepath, sub_dataset_name)
        queries_path = os.path.join(queries_folder_path, "queries")

        if os.path.exists(queries_path):
            print("Query dataset exists, loading from disk ...")
            ds = load_from_disk(queries_path)
            print(f"Loaded queries from {queries_path}")
        else:
            print("Downloading query dataset ...")
            os.makedirs(queries_folder_path, exist_ok=True)
            ds = load_dataset(f"vidore/{sub_dataset_name}", "queries")
            ds.save_to_disk(queries_path)
            print(f"Saved queries to {queries_path}")
        self.question_ds = ds

        super().__init__(run_name, collection_name, req_type, req_count )

    def get_questions(self, batch_size, start_idx): # stupid inputs for consistency
        questions = self.question_ds["test"][:self.req_count]["query"]
        gt_answers = self.question_ds["test"][:self.req_count]["answer"]
        return questions, gt_answers
        