
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


class lance_client_Amir(lance_client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "lancedb"
        self.thread_local_storage = threading.local()


    def query(self, collection_name, collection_abs_path, filter_expr, output_fields=None, limit=10):
        # Force hardware memory cache drop to ensure true cold-start numbers
        # print(f"DB file to be evicted from cache -> {collection_abs_path}")
        # evict_file_from_ram(collection_abs_path)
        # proc = psutil.Process(os.getpid())
        
        # -------------------------------------------------------------
        # STEP 1: OPEN TABLE METADATA
        # -------------------------------------------------------------
        # before_open = proc.io_counters()
        t0 = time.monotonic_ns()
        
        tbl = self.client.open_table(collection_name)
        
        t1 = time.monotonic_ns()
        # after_open = proc.io_counters()

        # -------------------------------------------------------------
        # STEP 2: LAZY SEARCH INITIALIZATION
        # -------------------------------------------------------------
        if output_fields is not None:
            data_plan = tbl.search().where(filter_expr).select(output_fields).limit(limit)
        else:
            data_plan = tbl.search().where(filter_expr).limit(limit)
            
        t2 = time.monotonic_ns()
        # after_setup = proc.io_counters()

        # -------------------------------------------------------------
        # STEP 3: PURE DATABASE EXECUTION (EAGER ARROW SCAN)
        # -------------------------------------------------------------
        # This forces LanceDB's Rust layer to complete all index-scanning,
        # vector calculation, and raw physical disk storage lookups.
        arrow_table = data_plan.to_arrow()
        
        t3 = time.monotonic_ns()
        # after_db_exec = proc.io_counters()

        # -------------------------------------------------------------
        # STEP 4: PANDAS DATA DATATYPE CONVERSION (CPU/MEM BOUND)
        # -------------------------------------------------------------
        # Pure Python memory space dictionary allocation and DataFrame copying.
        query_df = arrow_table.to_pandas()
        
        t4 = time.monotonic_ns()
        # after_pandas = proc.io_counters()

        # -------------------------------------------------------------
        # DICTIONARY REPORT PACKAGING
        # -------------------------------------------------------------
        db_fetch_stats = {
            "open_table_start": t0,
            "open_table_time_ns": t1 - t0,
            "lazy_search_start": t1,
            "lazy_search_time_ns": t2 - t1,       # Near zero due to lazy execution
            "db_fetch_pure_start" : t2,
            "db_fetch_pure_time_ns": t3 - t2,     # Isolated LanceDB database execution runtime
            "pandas_start": t3,
            "pandas_time_ns": t4 - t3,            # Isolated Python dataframe creation runtime
            "pandas_end" : t4
        }

        # Clean math calculation block function
        # def calculate_diffs(start_counter, end_counter, label_prefix, stats_dict):
        #     bytes_phys = end_counter.read_bytes - start_counter.read_bytes
        #     chars_start = getattr(start_counter, 'read_chars', start_counter.read_bytes)
        #     chars_end = getattr(end_counter, 'read_chars', end_counter.read_bytes)
            
        #     stats_dict[f"{label_prefix}_mb_phy"] = bytes_phys / (1024 ** 2)
        #     stats_dict[f"{label_prefix}_mb_log"] = (chars_end - chars_start) / (1024 ** 2)

        # # Run and append metrics explicitly to the reporting dictionary
        # calculate_diffs(before_open, after_open, "open_table", db_fetch_stats)
        # calculate_diffs(after_open, after_setup, "lazy_search", db_fetch_stats)
        # calculate_diffs(after_setup, after_db_exec, "db_fetch_pure", db_fetch_stats)
        # calculate_diffs(after_db_exec, after_pandas, "pandas", db_fetch_stats)

        ##
        # TEST
        if output_fields is not None:
            plan = tbl.search().where(filter_expr).select(output_fields).limit(limit).analyze_plan()
        else:
            plan = tbl.search().where(filter_expr).limit(limit).analyze_plan()

        print(f"{'#' * 50} RERANKING {'#' * 50}")
        print(plan)
        print(f"{'#' * 50} {'#' * 50}")
        ##

        return (query_df, db_fetch_stats)

    
    def query_search_image(
            self,
            query_vector,
            nprobe,
            topk,
            collection_name=None,
            search_batch_size=1,
            multithread=True,
            max_threads=4,
            consistency_level="Eventually",
            output_fields=["text", "vector"],
        ):
            print(f"***Start query search in collection: {collection_name}")
            # print(f"check, max_thread = {max_threads} , multithread = {multithread}")
    
            tbl = self.client.open_table(collection_name)
    
            total_queries = len(query_vector)
    
            # Adjust search_batch_size if it exceeds total_queries
            if search_batch_size > total_queries:
                search_batch_size = total_queries
    
            results = [None] * total_queries
    
            num_batches = (total_queries + search_batch_size - 1) // search_batch_size

            def init_worker():
                self.thread_local_storage.tbl = self.client.open_table(collection_name)
    
            def search_thread(start_idx, end_idx, batch_num):
                # print("inside multi-thread search_thread")
                tid = threading.get_ident()
                tbl = self.thread_local_storage.tbl
                patches_per_token = {}

                b_vectors = query_vector[start_idx:end_idx]
                batch_size = end_idx - start_idx
                # b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(3).to_list()
                t0 = time.monotonic_ns()
                b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).nprobes(nprobe).to_list()
                t1 = time.monotonic_ns()
                if len(b_results) != batch_size * topk:
                    raise ValueError(
                        f"len(b_results) must be n*topk n = {batch_size}, topk {topk}, but got {len(b_results)}"
                    )
                b_results = [b_results[i * topk : (i + 1) * topk] for i in range(batch_size)]

                ###
                # TEST
                plan = (
                    tbl.search(b_vectors, vector_column_name='vector')
                    .limit(topk)
                    .nprobes(nprobe)
                    .analyze_plan()
                )
                print(f"{'*' * 50} Retrieval for thread = {tid} {'*' * 50}")
                print(plan)
                print(f"{'*' * 50}")
                ###
    
                results[start_idx:end_idx] = b_results
                
                tmp_dict = {
                    "thread_id": tid,
                    "abs_start_time": t0,
                    "abs_end_time": t1,
                    "thread_time": (t1 - t0)
                }
                for j, query_results in enumerate(b_results):
                    patches_per_token[f'token_{j}'] = []
                    for result in query_results:
                        patches_per_token[f'token_{j}'].append({
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
                for i in tqdm(range(num_batches), desc="Searching batches"):
                    start_idx = i * search_batch_size
                    end_idx = min(start_idx + search_batch_size, total_queries)
                    b_vectors = query_vector[start_idx:end_idx]
                    t0 = time.monotonic_ns()
                    b_results = (
                        tbl.search(b_vectors, vector_column_name='vector')
                        .limit(topk)
                        .nprobes(nprobe)
                        .to_list()
                    )
                    t1 = time.monotonic_ns()
                    ###
                    # TEST
                    plan = (
                        tbl.search(b_vectors, vector_column_name='vector')
                        .limit(topk)
                        .nprobes(nprobe)
                        .analyze_plan()
                    )
                    print(f"{'*' * 50}")
                    print(plan)
                    print(f"{'*' * 50}")
                    ###
                    # b_results = tbl.search(b_vectors, vector_column_name='vector').limit(topk).to_list()
                    # here it shows that the retriever searches for top_k mathces for each query token
                    if len(b_results) != len(b_vectors) * topk:
                        raise ValueError(
                            f"len(b_results) must be n*topk n = {search_batch_size}, topk {topk}, but got {len(b_results)}"
                        )
                    b_results = [b_results[i * topk : (i + 1) * topk] for i in range(search_batch_size)]
                    results[start_idx:end_idx] = b_results
                    tid = threading.get_ident()
                    tmp_dict = {
                        "thread_id": tid,
                        "abs_start_time": t0,
                        "abs_end_time": t1,
                        "thread_time": (t1 - t0)
                    }
                    patches_per_token = {}
                    # The union of all top-k matches across all tokens will be returned as the result
                    for j, query_results in enumerate(b_results):
                        patches_per_token[f'token_{j}'] = []
                        for result in query_results:
                            patches_per_token[f'token_{j}'].append({
                                'doc_id': result['doc_id'],
                                'patch_id': result['seq_id']
                            })
                    tmp_dict["patches_per_token"] = patches_per_token
                    Retrieval_stats[i] = tmp_dict
                    
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

 
