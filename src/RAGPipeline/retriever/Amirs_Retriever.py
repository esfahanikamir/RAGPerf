import resource
import torch
import threading
from .BaseRetriever import *


# Global timing dictionaries
Storage_time = {}  # doc_id -> seconds spent fetching from DB
DotP_time = {}     # doc_id -> seconds spent on dot product / similarity
ProfilingStats = {}

class AmirsRetriever(BaseRetriever):
    def __init__(
        self, collection_name, top_k=5, retrieval_batch_size=1, client: milvus_client = None, dotp_device = "cpu"
    ):
        self.dotp_device = dotp_device
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

    def search_db_image(self, query_embeddings):
        # Perform a vector search on the collection to find the top-k most similar documents.
        # topk set to a reasonable large num
        # results = self.db_client.query_search(embeddings, topk=50, collection_name=self.collection_name, output_fields=["vector", "seq_id", "doc_id", "filepath"])
        # search_params = {"metric_type": "IP", "params": {}}
        batch_size = self.retrieval_batch_size
        results = self.client.query_search_image(
            query_embeddings,
            # int(50),
            topk = self.top_k,
            search_batch_size=batch_size,
            collection_name=self.collection_name,
            output_fields=["vector", "seq_id", "doc_id", "filepath"],
            # search_params=search_params,
        )
        return results
    
    def pdfimage_rerank(self, query_embeddings, top_k_results, top_n):
        scores = []
        with open("out.out", "w") as f:
            pass

        def rerank_single_doc(doc_id, data, client, collection_name, dotp_device):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            # here is the storage interaction part
            t0 = time.monotonic_ns()
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter_expr=f"doc_id in ({doc_id})",
                output_fields=["seq_id", "vector", "filepath"],
                limit=1000,
            )
            # print(f"inside rerank_single_doc: len(doc_colbert_vecs = {len(doc_colbert_vecs)})")
            Storage_time[doc_id] = time.monotonic_ns() - t0
            # here is the part for the dot product computation -> currently on CPU
            t1 = time.monotonic_ns()
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            minor_faults = (
                usage_after.ru_minflt -
                usage_before.ru_minflt
            )

            major_faults = (
                usage_after.ru_majflt -
                usage_before.ru_majflt
)
            if client.type == "lancedb":
                doc_vecs = np.vstack(doc_colbert_vecs["vector"].to_list())

                if dotp_device == "cpu":
                    score = np.dot(data, doc_vecs.T).max(1).sum()
                elif dotp_device.startswith("cuda"):
                    data_tensor    = torch.tensor(data, device="cuda", dtype=torch.float32)
                    doc_vecs_tensor = torch.tensor(doc_vecs, device="cuda", dtype=torch.float32)
                    score = torch.matmul(data_tensor, doc_vecs_tensor.T).max(1).values.sum().item()
                else:
                    raise ValueError(f"Unsupported dotp_device: {dotp_device}")
                t2 = time.monotonic_ns()
                DotP_time[doc_id] = t2 - t1

                ProfilingStats[doc_id] = {
                    "thread_id": threading.get_ident(),
                    "storage_time_ns": Storage_time[doc_id],
                    "compute_time_ns": DotP_time[doc_id],
                    "num_patches": len(doc_colbert_vecs),
                    "query_tokens": data.shape[0],
                    "embedding_dim": 128,
                    "fetched_bytes":
                        len(doc_colbert_vecs) * 128 * 4,
                    "total_rerank_time": t2 - t0,
                    "minor_faults": minor_faults,
                    "major_faults": major_faults
                }
                # print(f"doc_id = {doc_id}, num_patches = {len(doc_colbert_vecs)}, storage_time = {Storage_time[doc_id]}, dotp_time = {DotP_time[doc_id]}")        
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
                
                DotP_time[doc_id] = time.monotonic_ns() - t1

                ProfilingStats[doc_id] = {
                    "thread_id": threading.get_ident(),

                    "storage_time_ns": Storage_time[doc_id],
                    "compute_time_ns": DotP_time[doc_id],

                    "num_patches": len(doc_colbert_vecs),

                    "query_tokens": data.shape[0],

                    "embedding_dim": 128,

                    "fetched_bytes":
                        len(doc_colbert_vecs) * 128 * 4,
                }
                tpl = (score, doc_id, doc_colbert_vecs[0]["filepath"])
                return tpl
            else:
                raise ValueError(f"Unsupported client type: {client.type}")

        # with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.top_k) as executor:

            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, query_embeddings, self.client, self.collection_name, self.dotp_device
                ): doc_id
                for doc_id in top_k_results
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id, filepath = future.result()
                scores.append((score, doc_id, filepath))

        scores.sort(key=lambda x: x[0], reverse=True)

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
        return images_list
