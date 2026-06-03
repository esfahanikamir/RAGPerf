from abc import ABC, abstractmethod
import os
import time
import math
from RAGPipeline.responser.TextsResponser import VLLMResponser
from RAGPipeline.BaseRAGPipline import BaseRAGPipeline
from RAGPipeline.ImageRAGPipline import ImagesRAGPipeline
from encoder.sentenceTransformerEncoder import SentenceTransformerEncoder
from RAGPipeline.retriever.BaseRetriever import BaseRetriever

from RAGPipeline.retriever.Amirs_Retriever import AmirsRetriever  
from RAGPipeline.retriever.Amirs_Retriever import Storage_time, DotP_time  


from RAGPipeline.reranker.CrossEncoderReranker import CrossEncoderReranker
from evaluator.RagasEvaluator import RagasEvaluator
from datasets import Dataset
import utils.colored_print as cprint
from utils.logger import Logger, log_time_breakdown
from qwen_vl_utils import process_vision_info


# should make the pipeline fully modular with request queue passing

# class ModularRAGPipeline(ABC):
#     def __init__(self, **kwargs):
#         # self.run_name = kwargs.get("run_name", "default_run")


class ImagesRAGPipeline_rerank(ImagesRAGPipeline):
    def __init__(
        self,
        retriever: BaseRetriever,
        responser: VLLMResponser,
        embedder: SentenceTransformerEncoder,
        reranker : CrossEncoderReranker = None,
        evaluator: RagasEvaluator = None,
    ) -> None:
        self.reranker = reranker
        super().__init__(retriever= retriever,
                         responser= responser,
                         embedder= embedder)
        # self.retriever = retriever
        # self.responser = responser
        # self.embedder = embedder
        return

    # def generate_prompt(self, questions, contexts):
    #     chat_template = [
    #         {
    #             "role": "user",
    #             "content": [{"type": "image", "image": image} for image in contexts]
    #             + [{"type": "text", "text": questions}],
    #         }
    #     ]
    #     return chat_template

    def process(self, request, batch_size=2) -> None:
        global Storage_time, DotP_time
        if request.req_type == "query":
            cprint.iprintf(
                f"*** Processing {request.req_count} questions with batch size {batch_size}"
            )
            output_path = os.path.join(Logger().log_dirpath, "text_pipeline_stats.txt")
            with open(output_path, "w") as f:
                pass
            with open("prompts.out", "w") as f:
                pass
            with open("responses.out", "w") as f:
                pass
            with open("questions.out", "w") as f:
                pass


            # load models
            log_time_breakdown("start")
            cprint.iprintf(f"*** Loading models")
            self.embedder.load_encoder()
            self.responser.load_llm()
            if self.reranker is not None:
                # self.reranker.load_reranker()
                # imagepdf is not using reranker model -> only dot-product
                pass
            cprint.iprintf(f"*** Loading models done")

            # for now let us assume that the number of questions is devisible by the batch size
            nrounds = int(math.ceil(request.req_count / batch_size))
            cprint.iprintf(f"*** Will run {nrounds} rounds")
            # dead piece of code for the for loop
            # for round_idx in range(0, nrounds):
            #     start_sample_idx = round_idx * batch_size
            #     questions, gt_answer = request.get_questions(batch_size, start_idx=start_sample_idx)
            print(f"***Processing {request.req_count} questions")

            batch_embedding_time = [0] * nrounds

            retrieval_time = [[0] * batch_size for _ in range(nrounds)]
            batch_retrieval_time = [0] * nrounds

            rerank_time = [[0] * batch_size for _ in range(nrounds)]
            batch_rerank_time = [0] * nrounds

            rerank_storage_time = [[0] * batch_size for _ in range(nrounds)]
            rerank_dotp_time = [[0] * batch_size for _ in range(nrounds)]

            batch_rerank_storage_time = [0] * nrounds
            batch_rerank_dotp_time = [0] * nrounds

            prompt_time = [[0] * batch_size for _ in range(nrounds)]
            batch_prompt_time = [0] * nrounds

            generation_time = [[0] * batch_size for _ in range(nrounds)]
            batch_generation_time = [0] * nrounds

            batch_total_time = [0] * nrounds

            for i in range(0, request.req_count, batch_size):
                batch_num = i // batch_size
                questions, gt_answer = request.get_questions(batch_size, start_idx=i)

                print(f"there are {len(questions)} questoins for round {batch_num}")
                # print(f"questions for this round:\n {questions}")
                with open("questions.out", "a") as f:
                    f.write(f"batch #{batch_num}\n")
                    for qindex, question in enumerate(questions):
                        f.write(f"question #{qindex}\n{question}\n")

                # encode questions TODO: parameter
                # Embedding chunked texts
                # self.embedder.load_encoder()
                log_time_breakdown("embed")
                embedding_start_time = time.monotonic_ns()
                vectors = self.embedder.embedding_query(questions)
                print(f"len(vectors) = {len(vectors)}")
                embedding_end_time = time.monotonic_ns()
                # self.embedder.free_encoder()
                cprint.iprintf(f"*** Embedding done")

                batch_embedding_time[batch_num] = embedding_end_time - embedding_start_time

                for j, query in enumerate(vectors):
                    # print(f"query = {query}")
                    query = query.float().numpy()
                    # retrieval
                    log_time_breakdown("retrieve")
                    retrieval_start_time = time.monotonic_ns()
                    results = self.retriever.search_db_image(query)
                    retrieval_end_time = time.monotonic_ns()
                    cprint.iprintf(f"*** Retrieval done")

                    # Rerank Added by Amir
                    if self.reranker is not None:
                        cprint.iprintf(
                            f"*** Reranking top-{self.reranker.top_n} from {self.retriever.top_k} candidates"
                        )
                        log_time_breakdown("rerank")
                        rerank_start_time = time.monotonic_ns()
                        # print(f"results:\n {results}")

                        Storage_time.clear()
                        DotP_time.clear()

                        results = self.retriever.pdfimage_rerank(
                            query_embeddings = query,
                            top_k_results = results,
                            top_n = self.reranker.top_n
                        )
                        # results = self.reranker.rerank(questions[j], results)
                        rerank_end_time = time.monotonic_ns()
                        # self.reranker.free_reranker()

                        # get the separate storage time and the dotproduct time
                        rerank_storage_time_q = max(Storage_time.values()) if Storage_time else 0
                        rerank_dotp_time_q    = max(DotP_time.values())    if DotP_time    else 0

                        cprint.iprintf(f"*** Reranking done")

                    # augment
                    log_time_breakdown("prompt")
                    prompt_start_time = time.monotonic_ns()
                    prompts = self.generate_prompt(questions[j], results)
                    prompt_end_time = time.monotonic_ns()
                    cprint.iprintf(f"*** Prompt generation done")
                    with open("prompts.out", "a") as fout:
                        for idx, prompt in enumerate(prompts):
                            # there is always only one f... prompt
                            fout.write(f"=== Prompt {idx} for batch {batch_num}\nquestion #{j}:\n{questions[j]} ===\n")
                            fout.write(str(prompt) + "\n\n")

                    # generation
                    cprint.iprintf(f"*** Generating answers")
                    # self.responser.load_llm()
                    log_time_breakdown("generate")
                    generation_start_time = time.monotonic_ns()
                    responses = self.responser.query_llm(prompts)
                    generation_end_time = time.monotonic_ns()
                    # self.responser.free_llm()
                    cprint.iprintf(f"*** Generation done")

                    # with open("response.out", "w") as fout:
                    with open("responses.out", "a", encoding='utf-8') as fout:
                        for idx, response in enumerate(responses):
                            # there should be only one response for that one prompt I feel
                            fout.write(f"=== response {idx} for batch {batch_num}\nquestion #{j}:\n{questions[j]}===\n")
                            fout.write(response.strip() + "\n\n")
                    
                    retrieval_time[batch_num][j] = retrieval_end_time - retrieval_start_time
                    batch_retrieval_time[batch_num] += retrieval_time[batch_num][j]

                    # rerank_time[batch_num][j] = 0
                    if self.reranker is not None:
                        rerank_time[batch_num][j] = rerank_end_time - rerank_start_time
                        rerank_storage_time[batch_num][j] = rerank_storage_time_q
                        rerank_dotp_time[batch_num][j] = rerank_dotp_time_q
                    else:
                        rerank_time[batch_num][j] = 0
                        rerank_storage_time[batch_num][j] = 0
                        rerank_dotp_time[batch_num][j] = 0

                    batch_rerank_time[batch_num] += rerank_time[batch_num][j]
                    # batch_rerank_storage_time[batch_num] += rerank_storage_time[batch_num][j]
                    # batch_rerank_dotp_time[batch_num] += rerank_dotp_time[batch_num][j]

                    prompt_time[batch_num][j] = prompt_end_time - prompt_start_time
                    batch_prompt_time[batch_num] += prompt_time[batch_num][j]

                    generation_time[batch_num][j] = generation_end_time - generation_start_time
                    batch_generation_time[batch_num] += generation_time[batch_num][j]

                    if self.reranker is not None:
                        txt_to_print_q = (
                            f"Batch {batch_num} query {j}/{batch_size - 1} queries \n"

                            f"\t\tMax rerank_storage / rerank = {rerank_storage_time[batch_num][j] / rerank_time[batch_num][j] * 100:.2f}%\n"

                            f"\t\tMaxr rerank_dotp / rerank = {rerank_dotp_time[batch_num][j] / rerank_time[batch_num][j] * 100:.2f}%\n"

                            f"{'%' * 10}\n"
                        )
                        print(txt_to_print_q)
                        with open(output_path, "a") as f:
                            f.write(txt_to_print_q)


                batch_total_time[batch_num] = (
                    batch_embedding_time[batch_num] +
                    batch_retrieval_time[batch_num] +
                    batch_rerank_time[batch_num] +
                    batch_prompt_time[batch_num] +
                    batch_generation_time[batch_num]
                )

                txt_to_print = (
                    f"At the end of batch {batch_num} processing {batch_size} queries \n"

                    f"\tbatch embedding time: {batch_embedding_time[batch_num]} ns ({batch_embedding_time[batch_num] / 1e9} s, ({batch_embedding_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n"

                    f"\tSum of retrieval times: {batch_retrieval_time[batch_num]} ns ({batch_retrieval_time[batch_num] / 1e9} s, ({batch_retrieval_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n"

                    f"\tSum of reranking time: {batch_rerank_time[batch_num]} ns ({batch_rerank_time[batch_num] / 1e9} s, ({batch_rerank_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n"

                    f"\tSum of prompt times: {batch_prompt_time[batch_num]} ns ({batch_prompt_time[batch_num] / 1e9} s, ({batch_prompt_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n"

                    f"\tSum of generation times: {batch_generation_time[batch_num]} ns ({batch_generation_time[batch_num] / 1e9} s, ({batch_generation_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%)\n" 

                    f"\t{'#' * 10}\n\tReranking storage time vs dotp time\n"

                    # f"\t\tMax rerank_storage times: {batch_rerank_storage_time[batch_num]} ns ({batch_rerank_storage_time[batch_num] / 1e9} s, ({batch_rerank_storage_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%)\n"

                    # f"\t\tMax of rerank_dotp times: {batch_rerank_dotp_time[batch_num]} ns ({batch_rerank_dotp_time[batch_num] / 1e9} s, ({batch_rerank_dotp_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%)\n"
                    
                    # f"\t\trerank_storage / rerank = {batch_rerank_storage_time[batch_num] / batch_rerank_time[batch_num] * 100:.2f}%\n"

                    # f"\t\trerank_dotp / rerank = {batch_rerank_dotp_time[batch_num] / batch_rerank_time[batch_num] * 100:.2f}%\n"
                )

                with open(output_path, "a") as f:
                    f.write(txt_to_print)
                    f.write("\n")
                
                print(txt_to_print)
            
            t_embedding_time = sum(batch_embedding_time)
            t_retrieval_time = sum(batch_retrieval_time)
            t_rerank_time = sum(batch_rerank_time)
            # t_rerank_storage_time = sum(batch_rerank_storage_time)
            # t_rerank_dop_time = sum(batch_rerank_dotp_time)
            t_prompt_time = sum(batch_prompt_time)
            t_generation_time = sum(batch_generation_time)
            t_total_time = sum(batch_total_time)

            txt_to_print = ("***Complete timing profile***\n" +
                    # f"{i}\t" +
                    f"Embedding: {t_embedding_time} : {(100 * t_embedding_time / t_total_time):.2f}%\n" +
                    f"Retrieval: {t_retrieval_time} : {(100 * t_retrieval_time / t_total_time):.2f}%\n" +
                    f"Re-ranking: {t_rerank_time} : {(100 * t_rerank_time / t_total_time):.2f}%\n" +

                    # f"Re-ranking-storage: {t_rerank_time} : {(100 * t_rerank_storage_time / t_total_time):.2f}%\n" +
                    # f"Re-ranking-dotp: {t_rerank_time} : {(100 * t_rerank_dop_time / t_total_time):.2f}%\n" +

                    # f"Re-ranking-storage / rerank: {t_rerank_time} : {(100 * t_rerank_storage_time / t_rerank_time):.2f}%\n" +
                    # f"Re-ranking-dotp / rerank: {t_rerank_time} : {(100 * t_rerank_dop_time / t_rerank_time):.2f}%\n" +

                    f"Prompt: {t_prompt_time} : {(100 * t_prompt_time / t_total_time):.2f}%\n" +
                    f"Generation: {t_generation_time} : {(100 * t_generation_time / t_total_time):.2f}%\n" +
                    f"Total: {t_total_time}\n" 
                    f"{'*' * 50}\n"
            )
            
            with open(output_path, "a") as fout:
                fout.write(txt_to_print)

            print(txt_to_print)
            log_time_breakdown("free_models")

            # finished
            cprint.iprintf(f"*** Unloading models")
            self.embedder.free_encoder()
            self.responser.free_llm()
            if self.reranker is not None:
                # self.reranker.free_reranker()
                # no reranker model -> dot.product
                pass
            cprint.iprintf(f"*** Unloading models done")
            log_time_breakdown("done")
            
        return
