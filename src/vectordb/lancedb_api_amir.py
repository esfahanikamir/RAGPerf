
import concurrent.futures
import os
import psutil
import sys
import time
import threading
from tqdm import tqdm
from .lancedb_api import lance_client

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove the dots completely
from src.Amir_Helpers.storage_read_bytes_test import evict_file_from_ram



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.reverse()

Retrieval_stats = {}
thread_timing_retrieval = {}  # keyed by thread_id
thread_timing_lock_retr = threading.Lock()


class lance_client_Amir(lance_client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "lancedb"
        self.thread_local_storage = threading.local()


    def query(self, collection_name, collection_abs_path, filter_expr, output_fields=None, limit=10, ignore_analyzes = False,):
        # Force hardware memory cache drop to ensure true cold-start numbers
        # print(f"DB file to be evicted from cache -> {collection_abs_path}")
        # evict_file_from_ram(collection_abs_path)
        # proc = psutil.Process(os.getpid())
        if not hasattr(self.thread_local_storage, "tbl_cache"):
            self.thread_local_storage.tbl_cache = {}
        if collection_name not in self.thread_local_storage.tbl_cache:
            self.thread_local_storage.tbl_cache[collection_name] = self.client.open_table(collection_name)
        tbl = self.thread_local_storage.tbl_cache[collection_name]   # CHANGED: reuse cached handl

        if output_fields is not None:
            if ignore_analyzes == False:
                t_db_start = time.monotonic_ns()
                plan = tbl.search().where(filter_expr).select(output_fields).limit(limit).analyze_plan()
                t_db_end = time.monotonic_ns()
                query_res = tbl.search().where(filter_expr).select(output_fields).limit(limit)
            else:
                plan = None
                t_db_start = time.monotonic_ns()
                query_res = tbl.search().where(filter_expr).select(output_fields).limit(limit)
                t_db_end = time.monotonic_ns() 
            query_df = query_res.to_pandas()
            t_pd_end = time.monotonic_ns()
        else:
            if ignore_analyzes == False:
                t_db_start = time.monotonic_ns()
                plan = tbl.search().where(filter_expr).limit(limit).analyze_plan()
                t_db_end = time.monotonic_ns()
                query_res = tbl.search().where(filter_expr).limit(limit)
            else:
                plan = None
                t_db_start = time.monotonic_ns() 
                query_res = tbl.search().where(filter_expr).limit(limit)
                t_db_end = time.monotonic_ns() 
            query_df = query_res.to_pandas()
            t_pd_end = time.monotonic_ns()
        

        # -------------------------------------------------------------
        # DICTIONARY REPORT PACKAGING
        # -------------------------------------------------------------
        # t_db = t_db_end - t_db_start if ignore_analyzes == True else 0
        db_fetch_stats = {
            "db_plan": plan,
            # "t_db": t_db,
            "t_pandas": t_pd_end - t_db_end,            # Isolated Python dataframe creation runtime
        }

        return (query_df, db_fetch_stats)

    
    def query_search_image(
        self,
        query_vector, # token embeddings of a single question(query)
        nprobe,
        topk,
        collection_name=None,
        search_batch_size=1,
        multithread=True,
        max_threads=4,
        consistency_level="Eventually",
        output_fields=["text", "vector"],
        ignore_analyzes = False,
    ):
        print(f"***Start query search in collection: {collection_name}")
        # print(f"check, max_thread = {max_threads} , multithread = {multithread}")

        # number of a single query tokens
        total_queries = len(query_vector)

        # Adjust search_batch_size if it exceeds total_queries
        if search_batch_size > total_queries:
            search_batch_size = total_queries

        results = [None] * total_queries

        num_batches = (total_queries + search_batch_size - 1) // search_batch_size

        def init_worker():
            tid = threading.get_ident()
            open_tbl_start = time.monotonic_ns()
            self.thread_local_storage.tbl = self.client.open_table(collection_name)
            open_tbl_end = time.monotonic_ns()
            with thread_timing_lock_retr:
                thread_timing_retrieval[tid] = {
                    "thread_start": open_tbl_start,
                    "open_tbl_end": open_tbl_end,
                    "search_profiles": [],  # placeholder, filled in by search_thread as tasks complete
                }
        def search_thread(start_idx, end_idx, batch_num):
            # print("inside multi-thread search_thread")
            tid = threading.get_ident()
            tbl = self.thread_local_storage.tbl
            patches_per_token = []  # unified flat structure, same as single-threaded branch

            b_vectors = query_vector[start_idx:end_idx]
            # batch_size = end_idx - start_idx
            actual_batch_size = end_idx - start_idx
            # b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(3).to_list()
            if ignore_analyzes == False:
                tbl_search_start = time.monotonic_ns()
                plan = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(nprobe).analyze_plan()
                tbl_search_end = time.monotonic_ns()
                b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(nprobe).to_list()
            # print(f"{'*' * 50} Retrieval for thread = {tid} {'*' * 50}")
            # print(plan)
            # print(f"{'*' * 50}")
            else:
            # t0 = time.monotonic_ns()
                tbl_search_start = time.monotonic_ns() 
                b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(nprobe).to_list()
                tbl_search_end = time.monotonic_ns() 
            with thread_timing_lock_retr:
                thread_timing_retrieval[tid]["search_profiles"].append((tbl_search_start, tbl_search_end))

            # t_setup = time.monotonic_ns()
            # arrow_result = lazy_plan.to_arrow()
            # t_exec = time.monotonic_ns()
            # b_results = arrow_result.to_pylist()
            # t_convert = time.monotonic_ns()
            if len(b_results) != actual_batch_size * topk:
                raise ValueError(
                    f"len(b_results) must be n*topk n = {actual_batch_size}, topk {topk}, but got {len(b_results)}"
                )
            b_results = [b_results[i * topk : (i + 1) * topk] for i in range(actual_batch_size)]

            results[start_idx:end_idx] = b_results
            if ignore_analyzes == False:
                tmp_dict = {
                    "thread_id": tid,
                    "token_batch_id": batch_num,
                    "plan": plan
                } 
                for j, query_results in enumerate(b_results):
                    token_id = start_idx + j  # global token index, not local j
                    for result in query_results:
                        patches_per_token.append({
                            'token_id': token_id,
                            'doc_id': result['doc_id'],
                            'patch_id': result['seq_id']
                        })
                tmp_dict["patches_per_token"] = patches_per_token
                Retrieval_stats[batch_num] = tmp_dict
            # print(f"check from lancedb_api -> {Retrieval_stats}")

        # start_time = time.time()
        # print(f"*** Start multithreaded search: total={self.retrieval_size}, batch_size={batch_size}, max_threads={max_threads}")
        if max_threads == 1 or not multithread:
            # Single-threaded search
            # print("check -> inside the single thread")
            tid = threading.get_ident()
            open_tbl_start = time.monotonic_ns()
            tbl = self.client.open_table(collection_name)
            open_tbl_end = time.monotonic_ns()
            thread_timing_retrieval[tid] = {
                                        "thread_start": open_tbl_start,
                                        "open_tbl_end": open_tbl_end,
                                        "search_profiles": [],  # placeholder, filled in by search_thread as tasks complete
            }
                                                    
            for i in tqdm(range(num_batches), desc="Searching batches"):
                start_idx = i * search_batch_size
                end_idx = min(start_idx + search_batch_size, total_queries)
                b_vectors = query_vector[start_idx:end_idx]
                actual_batch_size = end_idx - start_idx
                ###
                if ignore_analyzes == False:
                    tbl_search_start = time.monotonic_ns()
                    plan = (
                        tbl.search(b_vectors, vector_column_name='vector')
                        .limit(topk)
                        .nprobes(nprobe)
                        .analyze_plan()
                    )
                    tbl_search_end = time.monotonic_ns()
                    b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(nprobe).to_list()
                # print(f"{'*' * 50}")
                # print(plan)
                # print(f"{'*' * 50}")
                else:

                    tbl_search_start = time.monotonic_ns()
                    b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(nprobe).to_list()
                    tbl_search_end = time.monotonic_ns()
                thread_timing_retrieval[tid]["search_profiles"].append((tbl_search_start, tbl_search_end))

        
                
                # b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).to_list()
                # here it shows that the retriever searches for top_k mathces for each query token
                if len(b_results) != actual_batch_size * topk:
                    raise ValueError(
                        f"len(b_results) must be n*topk n = {actual_batch_size}, topk {topk}, but got {len(b_results)}"
                    )
                b_results = [b_results[i * topk : (i + 1) * topk] for i in range(actual_batch_size)]
                results[start_idx:end_idx] = b_results
                if ignore_analyzes == False:
                    tmp_dict = {
                        "thread_id": tid,
                        "token_batch_id": i, 
                        "plan": plan,
                    }
                    patches_per_token = []
                    # The union of all top-k matches across all tokens will be returned as the result
                    for j, query_results in enumerate(b_results):
                        token_id = start_idx + j
                        for result in query_results:
                            patches_per_token.append({
                                'token_id' : token_id,
                                'doc_id': result['doc_id'],
                                'patch_id': result['seq_id']
                            })
                    tmp_dict["patches_per_token"] = patches_per_token
                    Retrieval_stats[i] = tmp_dict
            ##
            # print("##Retrieval_stats under lancedb_api_amir.py##")
            # print(Retrieval_stats)

        else:
            # print("check -> inside the multi thread")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads, initializer=init_worker,) as executor:
                futures = []
                progress = tqdm(total=num_batches, desc="Searching batches")

                def callback(future):
                    progress.update(1)

                for i in range(num_batches):
                    start_idx = i * search_batch_size
                    end_idx = min(start_idx + search_batch_size, total_queries)
                    future = executor.submit(search_thread, start_idx, end_idx, i)
                    future.add_done_callback(callback)
                    futures.append(future)

                concurrent.futures.wait(futures)
                progress.close()

        # end_time = time.time()
        doc_ids = set()
        patches_per_token = []
        # here it shows that a union of all the top_k matches of all the tokens will be returned as the result
        # with open("query.out", "w") as fout:
        for i, query_results in enumerate(results):
            patches_per_token.append({i : []})
            # fout.write(f"query_results:\n{query_results}\n")
            for result in query_results:
                doc_ids.add(result["doc_id"])
                # fout.write(f"result:\n{result}")

        print(f"***Query search completed.")
        # The outputs are the doc_ids only -> nothing more implemented although 
        # there is too much in the function parameters
        return doc_ids

