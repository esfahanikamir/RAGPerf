"""
Embed every query once and write the QUERY store: the token embeddings of all questions in
exactly the same file format, the same number encodings and the same int8/int4 scale as the
CORPUS store written by src/Amir_CPU_baseline_helpers/gen_ssd_embedding_dbs/convert.py.

All bytes are produced by the shared helper modules in that directory (quantize.py for the
encodings, layout.py for the record layout, queries.py for the query store). This file only
runs the encoder and hands the result over. See FORMAT_GUIDE.md there for the byte-level format.

The query store is built FOR one corpus store: the clip value c of int8/int4 is read from
CORPUS_DB/metadata.json. It is never computed from the queries.

Where things go (function arguments win over the environment variables):

    RAGPERF_CORPUS_DB_DIR   corpus store directory (must contain metadata.json)      [required]
    RAGPERF_QUERY_DB_DIR    query store directory to create                          [default ./query_db]
    RAGPERF_QUERY_DB_FORCE  set to 1 to overwrite an existing query store            [default: refuse]

Output directory:

    query_metadata.json                sizes, clip value, corpus identity, checksums
    queries.jsonl                      one line per query: record, question, gt_answer, n_tokens
    qry_<P>.bin                        one fixed-size record per query, P in fp32 fp16 bf16 int8 int4 bin
    qry_<P>.offsets.u64 / .counts.u32  byte offset and token count of every record
"""

import os
import sys
import time

import numpy as np
import torch

import utils.colored_print as cprint
from utils.logger import Logger, log_time_breakdown

from .ImageRAGPipline_rerank import *

# The shared writers. `src` (the directory run_new.py is started from) is normally on sys.path.
try:
    from Amir_CPU_baseline_helpers.gen_ssd_embedding_dbs import layout as db_layout
    from Amir_CPU_baseline_helpers.gen_ssd_embedding_dbs import quantize as db_quantize
    from Amir_CPU_baseline_helpers.gen_ssd_embedding_dbs import queries as query_db
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Amir_CPU_baseline_helpers.gen_ssd_embedding_dbs import layout as db_layout
    from Amir_CPU_baseline_helpers.gen_ssd_embedding_dbs import quantize as db_quantize
    from Amir_CPU_baseline_helpers.gen_ssd_embedding_dbs import queries as query_db


ENV_CORPUS_DIR = "RAGPERF_CORPUS_DB_DIR"
ENV_QUERY_DIR = "RAGPERF_QUERY_DB_DIR"
ENV_FORCE = "RAGPERF_QUERY_DB_FORCE"
DEFAULT_QUERY_DIR = "./query_db"

_DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}


def _resolve(out_dir=None, corpus_dir=None, force=None):
    """Arguments -> environment variables -> defaults. The corpus directory has no default."""
    if out_dir is None:
        out_dir = os.environ.get(ENV_QUERY_DIR) or DEFAULT_QUERY_DIR
    if corpus_dir is None:
        corpus_dir = os.environ.get(ENV_CORPUS_DIR)
    if force is None:
        force = os.environ.get(ENV_FORCE, "").strip().lower() in ("1", "true", "yes")
    if not corpus_dir:
        raise ValueError(
            "No corpus store given. The queries are encoded with the clip value of the corpus, so the corpus "
            "store must exist first (python convert.py --input embeddings.pkl --outdir CORPUS_DB). Then set "
            f"{ENV_CORPUS_DIR}=CORPUS_DB (or pass corpus_dir=...).")
    return out_dir, corpus_dir, bool(force)


def _check_before_embedding(out_dir, corpus_dir, force):
    """Everything that can fail without the encoder, so a wrong path is reported before the slow part."""
    _, cmeta = db_layout.load_metadata(corpus_dir, "corpus")
    c = db_quantize.clip_from_meta(cmeta)
    existing = db_layout.existing_store_files(out_dir, "queries")
    if existing and not force:
        raise IOError(f"{os.path.abspath(out_dir)} already holds a query store ({existing[:3]} ...). "
                      f"Choose another {ENV_QUERY_DIR} or set {ENV_FORCE}=1 to overwrite it.")
    return c, int(cmeta["dim"]), int(cmeta["num_records"])


def _to_float32_exact(t):
    """Encoder output (tensor on any device) -> (float32 numpy array with EXACTLY the same values, dtype name).

    fp16 -> fp32 and bf16 -> fp32 only add zero bits, so nothing is rounded. This is checked: converting
    the float32 result back to the original dtype must give the identical bit pattern.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(np.asarray(t))
    t = t.detach().to("cpu").contiguous()
    if t.dtype not in _DTYPE_NAMES:
        raise TypeError(f"unsupported embedding dtype {t.dtype}; expected fp16, bf16 or fp32")
    f32 = t.to(torch.float32).contiguous()
    if t.dtype != torch.float32:
        back = f32.to(t.dtype).contiguous()
        if not torch.equal(back.view(torch.int16), t.view(torch.int16)):
            raise ValueError(f"{t.dtype} -> float32 conversion is not exact")
    return f32.numpy(), _DTYPE_NAMES[t.dtype]


def save_query_embeddings(questions, gt_answer, vectors, out_dir=None, max_tokens=None, extra_meta=None,
                          corpus_dir=None, force=None):
    """Write the query store for `vectors` (list of tensors [n_tokens_i, dim]). Returns its metadata dict.

    Every precision (fp32, fp16, bf16, int8, int4, bin) is written by the same code that writes the
    corpus, with the clip value c of the corpus store `corpus_dir`.
    """
    out_dir, corpus_dir, force = _resolve(out_dir, corpus_dir, force)
    if len(questions) != len(vectors):
        raise ValueError(f"{len(questions)} questions but {len(vectors)} embedding tensors")

    converted = [_to_float32_exact(v) for v in vectors]
    names = sorted({n for _, n in converted})
    if len(names) != 1:
        raise ValueError(f"embedding tensors have mixed dtypes: {names}")
    arrays = [a for a, _ in converted]

    answers = list(gt_answer) if gt_answer is not None else [None] * len(questions)
    if len(answers) != len(questions):
        raise ValueError(f"{len(questions)} questions but {len(answers)} ground-truth answers")

    source = {"encoder_output_dtype": names[0], "torch_version": torch.__version__}
    if extra_meta:
        source.update(extra_meta)
    return query_db.save_query_db(questions, answers, arrays, corpus_dir, out_dir,
                                  max_tokens=max_tokens, source=source, force=force)


def load_query_embeddings(out_dir=None, int8=False, precision=None):
    """Load a query store written by save_query_embeddings.

    Returns (questions, gt_answers, embeddings, meta). embeddings is a list of numpy arrays
    [n_tokens_i, dim] in `precision` (default fp32, the exact encoder values; int8=True is the
    same as precision="int8"). int4 comes back as int8 codes in [-7, 7], bin as 0/1 bits.
    """
    if out_dir is None:
        out_dir = os.environ.get(ENV_QUERY_DIR) or DEFAULT_QUERY_DIR
    if precision is None:
        precision = "int8" if int8 else "fp32"
    return query_db.load_query_db(out_dir, precision)


class ImagesRAGPipeline_rerank_save_q_embeddings(ImagesRAGPipeline_rerank):
    def __init__(self, retriever, responser, embedder, reranker=None, evaluator=None):
        super().__init__(retriever, responser, embedder, reranker, evaluator)

    def process(self, request, batch_size=2, dont_answer=True,
                out_dir=None, corpus_dir=None, max_tokens=None, force=None):
        """Embed all questions of `request` and write the query store.

        batch_size and dont_answer are accepted because run_new.py passes them; they are not used.
        Only the encoder is loaded: no answers are generated here, so the LLM is never needed.
        """
        out_dir, corpus_dir, force = _resolve(out_dir, corpus_dir, force)
        clip_c, dim, num_docs = _check_before_embedding(out_dir, corpus_dir, force)
        cprint.iprintf(f"*** Query store -> {os.path.abspath(out_dir)}")
        cprint.iprintf(f"*** Corpus store: {os.path.abspath(corpus_dir)} ({num_docs} docs, dim {dim}, "
                       f"clip c = {clip_c:.8g})")

        log_time_breakdown("load_models")
        cprint.iprintf(f"*** Loading models")
        self.embedder.load_encoder()
        cprint.iprintf(f"*** Loading models done")

        questions, gt_answer = request.get_questions(request.req_count, start_idx=0)

        log_time_breakdown("embed")
        embedding_start_time = time.monotonic_ns()
        vectors = self.embedder.embedding_query(questions)
        embedding_end_time = time.monotonic_ns()
        cprint.iprintf(f"*** Embedding done: {len(vectors)} queries in "
                       f"{(embedding_end_time - embedding_start_time) / 1e9:.2f} s")

        log_time_breakdown("save_q_embeddings")
        meta = save_query_embeddings(
            questions, gt_answer, vectors, out_dir=out_dir, max_tokens=max_tokens,
            corpus_dir=corpus_dir, force=force,
            extra_meta={
                "embedder": type(self.embedder).__name__,
                "model_name": getattr(self.embedder, "model_name", None),
                "embedding_time_s": (embedding_end_time - embedding_start_time) / 1e9,
            })

        cprint.iprintf(f"*** Saved {meta['num_queries']} queries ({meta['total_tokens']} tokens, "
                       f"{meta['min_tokens_observed']}-{meta['max_tokens_observed']} tokens/query, "
                       f"dim {meta['dim']}, encoder dtype {meta['source']['encoder_output_dtype']}) "
                       f"to {os.path.abspath(out_dir)}")
        cprint.iprintf(f"*** {meta['max_tokens']} vector slots per record; record bytes: "
                       + ", ".join(f"{p} {v['record_bytes']}" for p, v in meta["precisions"].items()))
        cprint.iprintf(f"*** Next: python verify.py --corpus {os.path.abspath(corpus_dir)} "
                       f"--queries {os.path.abspath(out_dir)}")
        return meta
