from .BaseRetriever import *


# Global timing dictionaries
Storage_time = {}  # doc_id -> seconds spent fetching from DB
DotP_time = {}     # doc_id -> seconds spent on dot product / similarity

class AmirsRetriever(BaseRetriever):
    def __init__(
        self, collection_name, top_k=5, retrieval_batch_size=1, client: milvus_client = None
    ):
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

        def rerank_single_doc(doc_id, data, client, collection_name):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            # here is the storage interaction part
            t0 = time.monotonic_ns()
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter_expr=f"doc_id in ({doc_id})",
                output_fields=["seq_id", "vector", "filepath"],
                limit=1000,
            )
            Storage_time[doc_id] = time.monotonic_ns() - t0
            # here is the part for the dot product computation -> currently on CPU
            t1 = time.monotonic_ns()
            if client.type == "lancedb":
                doc_vecs = np.vstack(doc_colbert_vecs["vector"].to_list())
                score = np.dot(data, doc_vecs.T).max(1).sum()
                DotP_time[doc_id] = time.monotonic_ns() - t1
                return (score, doc_id, doc_colbert_vecs["filepath"][0])
            
            # elif client.type == "milvus":
            #     doc_vecs = np.vstack([data["vector"] for data in doc_colbert_vecs])
            #     tpl =  (score, doc_id, doc_colbert_vecs[0]["filepath"])
            #     DotP_time[doc_id] = time.monotonic_ns() - t1
            #     return tpl
            elif client.type == "milvus":
                doc_vecs = np.vstack([d["vector"] for d in doc_colbert_vecs])
                score = np.dot(data, doc_vecs.T).max(1).sum()  # ← add this
                tpl = (score, doc_id, doc_colbert_vecs[0]["filepath"])
                DotP_time[doc_id] = time.monotonic_ns() - t1
                return tpl
            else:
                raise ValueError(f"Unsupported client type: {client.type}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, query_embeddings, self.client, self.collection_name
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
