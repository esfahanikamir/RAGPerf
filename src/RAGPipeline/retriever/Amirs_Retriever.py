import itertools
import resource
import torch
import threading
from .BaseRetriever import *
# from vectordb.lancedb_api import Retrieval_stats
from vectordb.lancedb_api_amir import Retrieval_stats, thread_timing_retrieval

# to return the cpu core of each thread
import ctypes
_libc = ctypes.CDLL("libc.so.6")
def current_cpu() -> int:
    return _libc.sched_getcpu()

# Global timing dictionaries
# data_fetch_time = {}  # doc_id -> seconds spent fetching from DB
# np_time = {}
# np_time_start = {}
# np_time_end = {}
# maxsim_time = {}     # doc_id -> seconds spent on dot product / similarity
# maxsim_time_start = {}
# maxsim_time_end = {}
# ProfilingStats = {} # per thread
# per_process_io_stat = {} # per whole process

rerank_thread_stats = {}          # keyed by native thread id
rerank_thread_stats_lock = threading.Lock()

# def get_process_io():
#     stats = {}

#     with open("/proc/self/io", "r") as f:
#         for line in f:
#             key, value = line.split(":")
#             stats[key.strip()] = int(value.strip())

#     return stats

def init_rerank_worker(client, collection_name):
    tid = threading.get_native_id()
    t0 = time.monotonic_ns()
    if not hasattr(client.thread_local_storage, "tbl_cache"):
        client.thread_local_storage.tbl_cache = {}
    client.thread_local_storage.tbl_cache[collection_name] = client.client.open_table(collection_name)
    t1 = time.monotonic_ns()
    with rerank_thread_stats_lock:
        rerank_thread_stats[tid] = {
            "thread_start_time": t0,
            "open_table_time_ns": t1 - t0,
            "doc_stats": [],   # (doc_id, db_plan, t_pandas, numpy_time_ns, maxsim_time_ns) per call
        }

class AmirsRetriever(BaseRetriever):
    def __init__(
        self, collection_name, collection_abs_path, nprobe, retrieval_evict_mem, rerank_evict_mem, top_k=5, retrieval_batch_size=1, client= None, dotp_device = "cpu", max_rerank_worker = 1, m_of_top_k = 0, max_retrieval_threads = 1, ignore_analyzes = True
    ):
        self.nprobe = nprobe
        self.dotp_device = dotp_device
        self.collection_abs_path = collection_abs_path
        self.max_rerank_worker = max_rerank_worker
        self.m_of_top_k = m_of_top_k
        self.max_retrieval_threads = max_retrieval_threads
        self.ignore_analyzes = ignore_analyzes
        self.retrieval_evict_mem = retrieval_evict_mem
        self.rerank_evict_mem = rerank_evict_mem
        super().__init__(
            collection_name = collection_name,
            top_k = top_k, 
            retrieval_batch_size = retrieval_batch_size,
            client = client
        )

    # def search_db(self, query_embeddings):
    #     # self.client.load_collection(self.collection_name)

    #     # results = []
    #     batch_size = self.retrieval_batch_size
    #     results = self.client.query_search(
    #         query_embeddings,
    #         self.top_k,
    #         collection_name=self.collection_name,
    #         search_batch_size=batch_size,
    #         multithread=True,
    #         max_threads=1,
    #         consistency_level="Eventually",
    #     )
    #     # self._release_collections()

    #     return results

    def search_db_image(self, query_embeddings): # retrieve
        # Perform a vector search on the collection to find the top-k most similar documents.
        # topk set to a reasonable large num
        # results = self.db_client.query_search(embeddings, topk=50, collection_name=self.collection_name, output_fields=["vector", "seq_id", "doc_id", "filepath"])
        # search_params = {"metric_type": "IP", "params": {}}

        global Retrieval_stats, thread_timing_retrieval
        batch_size = self.retrieval_batch_size
        Retrieval_stats.clear()
        thread_timing_retrieval.clear()
        results = self.client.query_search_image(
            query_embeddings,
            # int(50),
            topk = self.top_k,
            search_batch_size=batch_size,
            collection_name=self.collection_name,
            output_fields=["vector", "seq_id", "doc_id", "filepath"],
            max_threads = self.max_retrieval_threads,
            nprobe = self.nprobe,
            ignore_analyzes = self.ignore_analyzes,
            # search_params=search_params,
        )
        # print(f"inside search_db_image -> len(results) = # of pages after retrieval = {len(results)}")
        # print(f"check Amir_retriever -> {Retrieval_stats}")
        return (results, Retrieval_stats, thread_timing_retrieval)
    
    def pdfimage_rerank(self, query_embeddings, top_k_results, top_n): # rerank
        # print(f"len(top_k_results) = {len(top_k_results)}")
        t_rerank_start = time.monotonic_ns()
        
        self.m_of_top_k = min(self.m_of_top_k, len(top_k_results)) if self.m_of_top_k != 0 else len(top_k_results)

        scores = []
        # with open("out.out", "w") as f:
        #     pass

        def rerank_single_doc(doc_id, data, client, collection_name, dotp_device):
            tid = threading.get_native_id()
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            # here is the storage interaction part
            cpu_at_start = current_cpu()
            t_doc_start = time.monotonic_ns()
            (doc_colbert_vecs, detailed_fetch_stat) = client.query(
                collection_name=collection_name,
                collection_abs_path = self.collection_abs_path,
                filter_expr=f"doc_id in ({doc_id})",
                output_fields=["seq_id", "vector", "filepath"],
                limit=1024,
                ignore_analyzes = self.ignore_analyzes,
            )
            # print(f"inside rerank_single_doc: len(doc_colbert_vecs = {len(doc_colbert_vecs)})")
            # t1 = time.monotonic_ns()
            # data_fetch_time[doc_id] = t1 - t0
            # here is the part for the dot product computation -> currently on CPU
            if client.type == "lancedb":
                t2 = time.monotonic_ns()
                doc_vecs = np.vstack(doc_colbert_vecs["vector"].to_list())
                t3 = time.monotonic_ns()

                if dotp_device == "cpu":
                    t4 = time.monotonic_ns()
                    score = np.dot(data, doc_vecs.T).max(1).sum()
                    t5 = time.monotonic_ns()
                    cpu_at_end = current_cpu()
                # not for now
                # elif dotp_device.startswith("cuda"):
                #     t1 = time.monotonic_ns()
                #     data_tensor    = torch.tensor(data, device="cuda", dtype=torch.float32)
                #     doc_vecs_tensor = torch.tensor(doc_vecs, device="cuda", dtype=torch.float32)
                #     score = torch.matmul(data_tensor, doc_vecs_tensor.T).max(1).values.sum().item()
                else:
                    raise ValueError(f"Unsupported dotp_device: {dotp_device}")
            
                with rerank_thread_stats_lock:
                    rerank_thread_stats[tid]["doc_stats"].append(
                        {
                            "doc_id": doc_id,
                            "t_doc_start": t_doc_start,
                            "db_plan": detailed_fetch_stat["db_plan"],
                            "t_pandas": detailed_fetch_stat["t_pandas"],
                            "t_np": t3 - t2,
                            "t_maxsim": t5 - t4,
                            "t_doc_end": t5,
                            "t_doc_process": t5 - t_doc_start,
                            "cpu_core_start" : cpu_at_start,
                            "cpu_core_end" : cpu_at_end,
                        }
                    )        
                return (score, doc_id, doc_colbert_vecs["filepath"][0])
            
            # elif client.type == "milvus":
            #     doc_vecs = np.vstack([data["vector"] for data in doc_colbert_vecs])
            #     tpl =  (score, doc_id, doc_colbert_vecs[0]["filepath"])
            #     DotP_time[doc_id] = time.monotonic_ns() - t1
            #     return tpl
            elif client.type == "milvus":
                doc_vecs = np.vstack([d["vector"] for d in doc_colbert_vecs])
                
                if dotp_device == "cpu":
                    score = np.dot(data, doc_vecs.T).max(1).sum()
                elif dotp_device == "cuda":
                    data_tensor     = torch.tensor(data,     device="cuda", dtype=torch.float32)
                    doc_vecs_tensor = torch.tensor(doc_vecs, device="cuda", dtype=torch.float32)
                    score = torch.matmul(data_tensor, doc_vecs_tensor.T).max(1).values.sum().item()
                else:
                    raise ValueError(f"Unsupported dotp_device: {dotp_device}")
                
                tpl = (score, doc_id, doc_colbert_vecs[0]["filepath"])
                return tpl
            else:
                raise ValueError(f"Unsupported client type: {client.type}")

        # per process profiling pagefaults + io
        # io_before = get_process_io()
        # usage_before = resource.getrusage(resource.RUSAGE_SELF)

        # fix it
        # max_docs = 16
        # top_k_results = list(top_k_results)
        # top_k_results = top_k_results[:max_docs]

        # with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_rerank_worker,
                initializer=init_rerank_worker,
                initargs=(self.client, self.collection_name),
                ) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, query_embeddings, self.client, self.collection_name, self.dotp_device
                ): doc_id
                for doc_id in itertools.islice(top_k_results, self.m_of_top_k)
                # for doc_id in top_k_results[:self.m_of_top_k]
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id, filepath = future.result()
                scores.append((score, doc_id, filepath))
        t_join_threads = time.monotonic_ns()

        # io_after = get_process_io()
        # usage_after = resource.getrusage(resource.RUSAGE_SELF)

        # minor_faults = usage_after.ru_minflt - usage_before.ru_minflt
        # major_faults = usage_after.ru_majflt - usage_before.ru_majflt
        # read_bytes = io_after["read_bytes"] - io_before["read_bytes"]   
        # syscr = io_after["syscr"] - io_before["syscr"]

        # per_process_io_stat["minor_faults"] = minor_faults
        # per_process_io_stat["major_faults"]= major_faults
        # per_process_io_stat["read_bytes"]= read_bytes
        # per_process_io_stat["syscr"]= syscr
        
        scores.sort(key=lambda x: x[0], reverse=True)
        t_sort_done = time.monotonic_ns()

        def GetPDF(filepath):
            """
            Loads and returns the image at the given filepath.
            """
            if os.path.exists(filepath):
                image = Image.open(filepath)
                return image
            else:
                print(f"File does not exist: {filepath}")
                return None

        images_list = []
        if len(scores) >= top_n:
            # scores[: top_n]
            for hits in scores[: top_n]:
                images_list.append(GetPDF(hits[2]))
        else:
            for hits in scores:
                images_list.append(GetPDF(hits[2]))
        abstract_rerank_timing = {
            "t_rerank_start": t_rerank_start,
            "t_rerank_join": t_join_threads,
            "t_sort_done" : t_sort_done
        }
        return (images_list, abstract_rerank_timing)
