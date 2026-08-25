from abc import ABC, abstractmethod
import csv
import os
import time
import math


from RAGPipeline.responser.TextsResponser import VLLMResponser
from RAGPipeline.BaseRAGPipline import BaseRAGPipeline
from RAGPipeline.ImageRAGPipline import ImagesRAGPipeline
from encoder.sentenceTransformerEncoder import SentenceTransformerEncoder
from RAGPipeline.retriever.BaseRetriever import BaseRetriever

from RAGPipeline.retriever.Amirs_Retriever import AmirsRetriever  
from RAGPipeline.retriever.Amirs_Retriever import rerank_thread_stats
# from RAGPipeline.retriever.Amirs_Retriever import  per_process_io_stat   




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

    def process(self, request, batch_size=2, dont_answer= False) -> None:
        global rerank_thread_stats
        # global per_process_io_stat  
        # ToDo: evict cache to get the page faults      
        if request.req_type == "query":
            cprint.iprintf(
                f"*** Processing {request.req_count} questions with batch size {batch_size}"
            )
            output_path = os.path.join(Logger().log_dirpath, "text_pipeline_stats.txt")
            prompt_path = os.path.join(Logger().log_dirpath, "prompts.out")
            response_path = os.path.join(Logger().log_dirpath, "responses.out")
            questions_path = os.path.join(Logger().log_dirpath, "questions.out")
            retrieval_time_log_path = os.path.join(Logger().log_dirpath, "retrieval_time.csv")
            
            with open(output_path, "w") as f:
                pass
            with open(prompt_path, "w") as f:
                pass
            with open(response_path, "w") as f:
                pass
            with open(questions_path, "w") as f:
                pass
            with open(retrieval_time_log_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        # headers
                        writer.writerow([
                            "batch_num",
                            "query",
                            "total_time",
                        ])


            # load models
            log_time_breakdown("start")
            cprint.iprintf(f"*** Loading models")
            self.embedder.load_encoder()
            if dont_answer == False:
                self.responser.load_llm()
            if self.reranker is not None:
                # self.reranker.load_reranker()
                # imagepdf is not using reranker model -> only dot-product
                pass
            cprint.iprintf(f"*** Loading models done")

            # for now let us assume that the number of questions is devisible by the batch size
            nrounds = int(math.ceil(request.req_count / batch_size))
            cprint.iprintf(f"*** Will run {nrounds} rounds(bathces)")
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

            # rerank_data_fetch_time = [[0] * batch_size for _ in range(nrounds)]
            # rerank_dotp_time = [[0] * batch_size for _ in range(nrounds)]

            batch_rerank_data_fetch_time = [0] * nrounds
            batch_rerank_dotp_time = [0] * nrounds

            prompt_time = [[0] * batch_size for _ in range(nrounds)]
            batch_prompt_time = [0] * nrounds

            generation_time = [[0] * batch_size for _ in range(nrounds)]
            batch_generation_time = [0] * nrounds

            batch_total_time = [0] * nrounds

            # for each batch
            for i in range(0, request.req_count, batch_size):
                batch_details = ""
                batch_num = i // batch_size
                # gets {batch-size} questions
                questions, gt_answer = request.get_questions(batch_size, start_idx=i)

                print(f"there are {len(questions)} questoins for batch {batch_num}/{nrounds}")
                # print(f"questions for this round:\n {questions}")
                with open(questions_path, "a") as f:
                    f.write(f"batch #{batch_num}\n")
                    for qindex, question in enumerate(questions):
                        f.write(f"question #{qindex}\n{question}\n")

                # encode questions TODO: parameter
                # Embedding chunked texts
                # self.embedder.load_encoder()
                log_time_breakdown("embed")
                embedding_start_time = time.monotonic_ns()
                # embedds a batch of questions (with colpali)
                vectors = self.embedder.embedding_query(questions)
                embedding_end_time = time.monotonic_ns()
                print(f"len(vectors) = {len(vectors)}")
                for vi, vector in enumerate(vectors):
                    print(f"vector {vi}/total {len(vectors) - 1}\ntype {type(vector)}\nlen(this vector = {len(vector)})")
                # self.embedder.free_encoder()
                cprint.iprintf(f"*** Embedding done")

                batch_embedding_time[batch_num] = embedding_end_time - embedding_start_time

                # for each question(its embedding is called query now) in a batch of questions
                for j, query in enumerate(vectors):
                    # print(f"query = {query}")
                    query = query.float().numpy()
                    # retrieval
                    log_time_breakdown("retrieve")
                    retrieval_start_time = time.monotonic_ns()
                    results, Retrieval_stats, thread_timing_retrieval = self.retriever.search_db_image(query)
                    retrieval_end_time = time.monotonic_ns()
                    cprint.iprintf(f"*** Retrieval done")
                    retrieval_stats_filename = f"Batch_{batch_num}_Query_{j}_retrieval_stats.csv"
                    retrieval_stats_file = os.path.join(Logger().log_dirpath, retrieval_stats_filename)

                    retrieval_thread_profile_filename = f"Batch_{batch_num}_Query_{j}_retrieval_thread_profile.csv"
                    retrieval_thread_profile_file = os.path.join(Logger().log_dirpath, retrieval_thread_profile_filename)
                    save_thread_profiles(thread_timing_retrieval, retrieval_thread_profile_file)

                    fieldnames = [
                        "global_req#", "is_in_query_batch#", "token_id",
                        "in_token_batch#", "by_thread#", "plan",
                        "doc_id", "patch_id"
                    ]

                    with open(retrieval_stats_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for token_batch, stats in sorted(Retrieval_stats.items()):
                            first = True
                            for patch in stats["patches_per_token"]:
                                if first:
                                    writer.writerow({
                                        "global_req#": i,                    # now correctly the outer batch-start index
                                        "is_in_query_batch#": j,
                                        "token_id": patch["token_id"],        # fixed: was stats["token_id"]
                                        "in_token_batch#": stats["token_batch_id"],
                                        "by_thread#": stats["thread_id"],
                                        "plan": stats["plan"],
                                        "doc_id": patch["doc_id"],
                                        "patch_id": patch["patch_id"],
                                    })
                                    first = False
                                else:
                                    writer.writerow({
                                        "global_req#": "_",                    # now correctly the outer batch-start index
                                        "is_in_query_batch#": "_",
                                        "token_id": "_",        # fixed: was stats["token_id"]
                                        "in_token_batch#": "_",
                                        "by_thread#": "_",
                                        "plan": "_",
                                        "doc_id": patch["doc_id"],
                                        "patch_id": patch["patch_id"],
                                    })


                    # retrieval_start_time/end_time block and retrieval_time_log_path write removed entirely
                    with open(retrieval_time_log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # headers
                        writer.writerow([
                            f"{i}",
                            f"{j}",
                            f"{retrieval_end_time - retrieval_start_time}",
                        ])

                    # Rerank Added by Amir
                    if self.reranker is not None:
                        txt = f"\tBatch:{i}, Question:{j} of this batch, Reranking top-{self.reranker.top_n} from {len(results)} OR m_of_top_k = {self.retriever.m_of_top_k} candidate image pages (top_k = {self.retriever.top_k})\n"
                        batch_details += txt
                        cprint.iprintf(txt)

                        rerank_thread_stats.clear()
                        # per_process_io_stat.clear()
                        log_time_breakdown("rerank")
                        rerank_start_time = time.monotonic_ns()
                        # print(f"results:\n {results}")


                        m_of_top_k = len(results)
                        results, abstract_rerank_timing = self.retriever.pdfimage_rerank(
                            query_embeddings = query,
                            top_k_results = results,
                            top_n = self.reranker.top_n,
                            
                        )
                        # results = self.reranker.rerank(questions[j], results)
                        rerank_end_time = time.monotonic_ns()
                        # self.reranker.free_reranker()

                        # get the separate storage time and the dotproduct time per threads for each query
                        # max was not a good option, we save the data in csv files and later process them
                        # rerank_data_fetch_time_q = max(data_fetch_time.values()) if data_fetch_time else 0
                        # rerank_dotp_time_q    = max(DotP_time.values())    if DotP_time    else 0

                        query_stat_filename_thread = f"Batch_{batch_num}_Query_{j}_rerank_thread_stats.csv"
                        query_stats_file = os.path.join(Logger().log_dirpath, query_stat_filename_thread)
                        with open(query_stats_file, "w", newline="") as f:
                            writer = csv.writer(f)
                            # headers
                            writer.writerow([
                                "batch_num",
                                "query_id",
                                "thread_id",
                                "doc_id",
                                "t_doc_start",
                                ##
                                "db_plan",
                                "pandas_time_ms", 
                                "np_time_ms",
                                "maxsim_time_ms",
                                "t_doc_end",
                                "doc_process_time_ms",
                                "cpu_core_start",
                                "cpu_core_end",
                            ])
                            for tid, stats in (rerank_thread_stats.items()):
                                for doc_stat in stats["doc_stats"]:
                                    writer.writerow([
                                        batch_num,
                                        j,
                                        tid,
                                        doc_stat["doc_id"],
                                        ##
                                        doc_stat["t_doc_start"],
                                        doc_stat["db_plan"],
                                        doc_stat["t_pandas"] / 1e6,
                                        doc_stat["t_np"] / 1e6,
                                        doc_stat["t_maxsim"] / 1e6,
                                        doc_stat["t_doc_end"],
                                        doc_stat["t_doc_process"] / 1e6,
                                        doc_stat["cpu_core_start"],
                                        doc_stat["cpu_core_end"]
                                    ])

                        query_stat_filename_abstract = f"Batch_{batch_num}_Query_{j}_rerank_general_stats.csv"
                        query_stats_file = os.path.join(Logger().log_dirpath, query_stat_filename_abstract)
                        with open(query_stats_file, "w", newline="") as f:
                            writer = csv.writer(f)
                            # headers
                            writer.writerow([
                                "batch_num",
                                "query",
                                "t_rerank_start",
                                "t_rerank_join",
                                "multithread_time_ms",
                                "t_sort_done",
                                "sort_time_ms"
                            ])
                            
                            writer.writerow([
                                batch_num,
                                j,
                                abstract_rerank_timing["t_rerank_start"],
                                abstract_rerank_timing["t_rerank_join"],
                                (abstract_rerank_timing["t_rerank_join"] - abstract_rerank_timing["t_rerank_start"]) / 1e6,
                                abstract_rerank_timing["t_sort_done"],
                                (abstract_rerank_timing["t_sort_done"] - abstract_rerank_timing["t_rerank_join"]) / 1e6
                            ])

                            # f.write(f"\n{'$' * 20}\n"
                            #         f"minor_faults={per_process_io_stat['minor_faults']} "
                            #         f"major_faults={per_process_io_stat['major_faults']}\n"
                            #         f"read_bytes={per_process_io_stat['read_bytes']/1024:.1f}KB\n"
                            #         f"syscr={per_process_io_stat['syscr']}\n"
                            #         f"\n{'$' * 20}\n"
                            #         )
                        cprint.iprintf(f"*** Reranking done")
                        

                    # augment
                    log_time_breakdown("prompt")
                    prompt_start_time = time.monotonic_ns()
                    prompts = self.generate_prompt(questions[j], results)
                    prompt_end_time = time.monotonic_ns()
                    cprint.iprintf(f"*** Prompt generation done")
                    with open(prompt_path, "a") as fout:
                        for idx, prompt in enumerate(prompts):
                            # there is always only one f... prompt
                            fout.write(f"\n=== Prompt {idx} for batch {batch_num}\nquestion #{j}:\n{questions[j]} ===\n")
                            fout.write(str(prompt) + "\n\n")

                    # generation
                    cprint.iprintf(f"*** Generating answers")
                    # self.responser.load_llm()
                    log_time_breakdown("generate")
                    generation_start_time = time.monotonic_ns()
                    if dont_answer == False:
                        responses = self.responser.query_llm(prompts)
                    else:
                        responses = ["dont_answer = True"]
                    generation_end_time = time.monotonic_ns()
                    # self.responser.free_llm()
                    cprint.iprintf(f"*** Generation done")

                    # with open("response.out", "w") as fout:
                    with open(response_path, "a", encoding='utf-8') as fout:
                        for idx, response in enumerate(responses):
                            # there should be only one response for that one prompt I feel
                            fout.write(f"\n=== response {idx} for batch {batch_num}\nquestion #{j}:\n{questions[j]}===\n")
                            fout.write(response.strip() + "\n\n")
                    
                    retrieval_time[batch_num][j] = retrieval_end_time - retrieval_start_time
                    batch_retrieval_time[batch_num] += retrieval_time[batch_num][j]

                    # rerank_time[batch_num][j] = 0
                    if self.reranker is not None:
                        rerank_time[batch_num][j] = rerank_end_time - rerank_start_time
                    else:
                        rerank_time[batch_num][j] = 0

                    batch_rerank_time[batch_num] += rerank_time[batch_num][j]

                    prompt_time[batch_num][j] = prompt_end_time - prompt_start_time
                    batch_prompt_time[batch_num] += prompt_time[batch_num][j]

                    generation_time[batch_num][j] = generation_end_time - generation_start_time
                    batch_generation_time[batch_num] += generation_time[batch_num][j]
                    
                    # if self.reranker is not None:
                    #     txt_to_print_q = (
                    #         f"Batch {batch_num} query {j}/{batch_size - 1} queries \n"

                    #         # f"\t\tMax rerank_storage / rerank = {rerank_data_fetch_time[batch_num][j] / rerank_time[batch_num][j] * 100:.2f}%\n"

                    #         # f"\t\tMaxr rerank_dotp / rerank = {rerank_dotp_time[batch_num][j] / rerank_time[batch_num][j] * 100:.2f}%\n"

                    #         f"{'%' * 10}\n"
                    #     )
                    #     print(txt_to_print_q)
                        # with open(output_path, "a") as f:
                        #     f.write(txt_to_print_q)


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

                    f"*****\n\tRetrieval details:\n{batch_details}\n*****\n"

                    f"\tSum of reranking time: {batch_rerank_time[batch_num]} ns ({batch_rerank_time[batch_num] / 1e9} s, ({batch_rerank_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n"

                    f"\tSum of prompt times: {batch_prompt_time[batch_num]} ns ({batch_prompt_time[batch_num] / 1e9} s, ({batch_prompt_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n"

                    f"\tSum of generation times: {batch_generation_time[batch_num]} ns ({batch_generation_time[batch_num] / 1e9} s, ({batch_generation_time[batch_num] / batch_total_time[batch_num] * 100:.2f}%))\n" 
                )

                with open(output_path, "a") as f:
                    f.write(txt_to_print)
                    f.write("\n")
                
                print(txt_to_print)
            
            t_embedding_time = sum(batch_embedding_time)
            t_retrieval_time = sum(batch_retrieval_time)
            t_rerank_time = sum(batch_rerank_time)
            t_prompt_time = sum(batch_prompt_time)
            t_generation_time = sum(batch_generation_time)
            t_total_time = sum(batch_total_time)

            txt_to_print = ("***Complete timing profile for all batches***\n" +
                    # f"{i}\t" +
                    f"Embedding: {t_embedding_time} : {(100 * t_embedding_time / t_total_time):.2f}%\n" +
                    f"Retrieval: {t_retrieval_time} : {(100 * t_retrieval_time / t_total_time):.2f}%\n" +
                    f"Re-ranking: {t_rerank_time} : {(100 * t_rerank_time / t_total_time):.2f}%\n" +
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
            if dont_answer == False:
                self.responser.free_llm()
            if self.reranker is not None:
                # self.reranker.free_reranker()
                # no reranker model -> dot.product
                pass
            cprint.iprintf(f"*** Unloading models done")
            log_time_breakdown("done")
            
        return

    import csv

def save_thread_profiles(thread_timing_retrieval, filepath):
    fieldnames = [
        "thread_index",     # stable 0,1,2... assigned at save time, not the raw OS thread id
        "raw_thread_id",     # keep the original too, in case you need it
        "thread_start",
        "open_tbl_end",
        "open_table_time_ns",
        "task_index",        # which search call this is, in order, for this thread
        "search_start",
        "search_end",
        "search_duration_ns",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for thread_index, (raw_tid, stats) in enumerate(sorted(thread_timing_retrieval.items())):
            for task_index, (s_start, s_end) in enumerate(stats["search_profiles"]):
                writer.writerow({
                    "thread_index": thread_index,
                    "raw_thread_id": raw_tid,
                    "task_index": task_index,
                    "thread_start": stats["thread_start"],
                    "open_tbl_end": stats["open_tbl_end"],
                    "open_table_time_ns": stats["open_tbl_end"] - stats["thread_start"],
                    "search_start": s_start,
                    "search_end": s_end,
                    "search_duration_ns": s_end - s_start,
                })