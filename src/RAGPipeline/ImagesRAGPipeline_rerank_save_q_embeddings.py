"""
Embed every query once and save the questions, ground-truth answers and token embeddings to disk,
so later runs (Python, or the C++ host of the FPGA pipeline) can load them without the encoder.

Output directory layout (default ./query_db):

    meta.json                 description of every file below (shapes, dtypes, sizes, scale)
    queries.jsonl             one line per query: idx, question, gt_answer, n_tokens, token_offset
    query_embeddings.pt       list of tensors [n_tokens_i, dim] in the encoder's own dtype (e.g. fp16)
    query_tokens_<dt>.bin     all query tokens back to back, row-major [total_tokens, dim], stored in the
                              encoder's own dtype without conversion: <dt> = fp16, bf16 or fp32
                              (bf16 is written as its raw 16-bit patterns; numpy has no bfloat16)
    query_index.bin           int64 pairs per query: (token_offset, n_tokens) into query_tokens_<dt>.bin
    query_db_int8.bin         (save_int8=True) same style as the document database: one fixed-size
                              record per query, query i at byte offset i * record_bytes,
                              record = max_tokens x dim int8, zero-padded past n_tokens
                              (record_bytes is a multiple of 4096, so O_DIRECT reads work)

int8 quantization: q = round(x * scale), clipped to [-127, 127], with one global
scale = 127 / max|x| over all query tokens (stored in meta.json). A positive per-query scale does
not change the ranking of documents for that query, but if you compare raw scores across queries
or against a CPU float reference, divide by scale (and by the document scale).
"""

import json
import math
import os
import time

import numpy as np
import torch

import utils.colored_print as cprint
from utils.logger import Logger, log_time_breakdown

from .ImageRAGPipline_rerank import *


RECORD_ALIGN = 4096        # bytes; matches O_DIRECT / XRT buffer alignment
TOKEN_ROUND = 32           # max_tokens is rounded up to a multiple of this


# torch dtype -> (short name used in file names / meta.json, numpy dtype used to store the raw values)
_DTYPES = {
    torch.float16:  ("fp16", np.float16),
    torch.bfloat16: ("bf16", np.uint16),   # raw bit patterns, no rounding
    torch.float32:  ("fp32", np.float32),
}


def _to_native(t):
    """Tensor/array (any device) -> (contiguous numpy array holding the original values exactly, dtype name)."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(np.asarray(t))
    t = t.detach().cpu().contiguous()
    if t.dtype not in _DTYPES:
        raise TypeError(f"unsupported embedding dtype {t.dtype}; expected fp16, bf16 or fp32")
    name, np_dt = _DTYPES[t.dtype]
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16), name
    return t.numpy().astype(np_dt, copy=False), name


def _native_to_f32(a, name):
    """Native stored array -> float32 (exact; used only for the quantization math)."""
    if name == "bf16":
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def save_query_embeddings(questions, gt_answer, vectors, out_dir="/scratch/amirk/RAG/vidore_v3_pharmaceuticals/queries",
                          max_tokens=None, save_int8=True, extra_meta=None):
    """Write questions, answers and embeddings to out_dir. Returns the meta dict."""
    if len(questions) != len(vectors):
        raise ValueError(f"{len(questions)} questions but {len(vectors)} embedding tensors")
    os.makedirs(out_dir, exist_ok=True)

    converted = [_to_native(v) for v in vectors]
    embs = [a for a, _ in converted]
    names = {n for _, n in converted}
    if len(names) > 1:
        raise ValueError(f"embedding tensors have mixed dtypes: {sorted(names)}")
    dt_name = names.pop() if names else "fp32"
    store_dtype = embs[0].dtype if embs else np.float32
    dims = {e.shape[1] for e in embs}
    if len(dims) != 1 or any(e.ndim != 2 for e in embs):
        raise ValueError(f"expected 2-D tensors with one shared dim, got {[e.shape for e in embs]}")
    dim = dims.pop()
    n_tokens = np.array([e.shape[0] for e in embs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_tokens)[:-1]]).astype(np.int64)
    total_tokens = int(n_tokens.sum())

    answers = list(gt_answer) if gt_answer is not None else [None] * len(questions)
    if len(answers) != len(questions):
        answers = (answers + [None] * len(questions))[:len(questions)]

    # 1) text + per-query metadata
    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            f.write(json.dumps({
                "idx": i,
                "question": q,
                "gt_answer": _jsonable(answers[i]),
                "n_tokens": int(n_tokens[i]),
                "token_offset": int(offsets[i]),
            }, ensure_ascii=False, default=str) + "\n")

    # 2) Python-friendly copy, in the encoder's own dtype
    torch.save([torch.from_numpy(e.view(np.int16)).view(torch.bfloat16) if dt_name == "bf16"
                else torch.from_numpy(e) for e in embs],
               os.path.join(out_dir, "query_embeddings.pt"))

    # 3) flat tokens in the native dtype + (offset, n_tokens) index, for C/C++ readers
    flat_name = f"query_tokens_{dt_name}.bin"
    flat = np.concatenate(embs, axis=0) if embs else np.zeros((0, dim), store_dtype)
    flat.tofile(os.path.join(out_dir, flat_name))
    np.stack([offsets, n_tokens], axis=1).astype(np.int64).tofile(os.path.join(out_dir, "query_index.bin"))

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_queries": len(questions),
        "dim": int(dim),
        "total_tokens": total_tokens,
        "min_tokens": int(n_tokens.min()) if len(n_tokens) else 0,
        "max_tokens_observed": int(n_tokens.max()) if len(n_tokens) else 0,
        "dtype": dt_name,
        "flat_tokens_file": flat_name,
        "files": {
            "queries.jsonl": "one JSON object per line: idx, question, gt_answer, n_tokens, token_offset",
            "query_embeddings.pt": f"torch list of {dt_name} tensors [n_tokens_i, dim]",
            flat_name: (f"{dt_name} little-endian, row-major [{total_tokens}, {dim}]"
                        + ("; raw bfloat16 bit patterns (float32 = uint32(bits) << 16)" if dt_name == "bf16" else "")),
            "query_index.bin": f"int64 little-endian [{len(questions)}, 2] = (token_offset, n_tokens)",
        },
    }

    # 4) fixed-size int8 records, same style as the document database
    if save_int8:
        observed = meta["max_tokens_observed"]
        if max_tokens is None:
            max_tokens = max(TOKEN_ROUND, int(math.ceil(observed / TOKEN_ROUND) * TOKEN_ROUND))
        if observed > max_tokens:
            raise ValueError(f"a query has {observed} tokens, more than max_tokens={max_tokens}")
        record_bytes = max_tokens * dim
        if record_bytes % RECORD_ALIGN:
            # grow max_tokens until a record is a whole number of 4 KiB blocks
            step = RECORD_ALIGN // math.gcd(RECORD_ALIGN, dim)
            max_tokens = int(math.ceil(max_tokens / step) * step)
            record_bytes = max_tokens * dim

        # quantization math in float32 (an exact copy of the fp16/bf16 values) to avoid extra rounding
        embs_f32 = [_native_to_f32(e, dt_name) for e in embs]
        absmax = max((float(np.abs(e).max()) for e in embs_f32 if e.size), default=1.0)
        scale = 127.0 / absmax if absmax > 0 else 1.0
        db = np.zeros((len(embs), max_tokens, dim), dtype=np.int8)
        for i, e in enumerate(embs_f32):
            db[i, :e.shape[0]] = np.clip(np.rint(e * np.float32(scale)), -127, 127).astype(np.int8)
        db.tofile(os.path.join(out_dir, "query_db_int8.bin"))

        meta.update({
            "max_tokens": int(max_tokens),
            "record_bytes": int(record_bytes),
            "int8_scale": scale,
            "int8_absmax": absmax,
        })
        meta["files"]["query_db_int8.bin"] = (
            f"int8 [{len(embs)}, {max_tokens}, {dim}]; query i at byte offset i * {record_bytes}; "
            f"rows >= n_tokens are zero; float = int8 / int8_scale")

    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def load_query_embeddings(out_dir="./query_db", int8=False):
    """Load what save_query_embeddings wrote.

    Returns (questions, gt_answers, embeddings, meta). embeddings is a list of tensors in the
    encoder's original dtype (meta["dtype"]: fp16, bf16 or fp32), or, with int8=True, a list of int8 numpy arrays [n_tokens_i, dim] cut from the padded records.
    """
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(out_dir, "queries.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    questions = [r["question"] for r in rows]
    answers = [r["gt_answer"] for r in rows]

    if int8:
        db = np.fromfile(os.path.join(out_dir, "query_db_int8.bin"), dtype=np.int8)
        db = db.reshape(meta["n_queries"], meta["max_tokens"], meta["dim"])
        embs = [db[i, :r["n_tokens"]] for i, r in enumerate(rows)]
    else:
        embs = torch.load(os.path.join(out_dir, "query_embeddings.pt"))
    return questions, answers, embs, meta


class ImagesRAGPipeline_rerank_save_q_embeddings(ImagesRAGPipeline_rerank):
    def __init__(self, retriever, responser, embedder, reranker=None, evaluator=None):
        super().__init__(retriever, responser, embedder, reranker, evaluator)

    def process(self, request, batch_size=2, dont_answer=True,
                out_dir="./query_db", max_tokens=None, save_int8=True):
        # Only the encoder is needed to embed queries; load the LLM only if explicitly asked.
        log_time_breakdown("load_models")
        cprint.iprintf(f"*** Loading models")
        self.embedder.load_encoder()
        if dont_answer == False:
            self.responser.load_llm()
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
            questions, gt_answer, vectors, out_dir=out_dir, max_tokens=max_tokens, save_int8=save_int8,
            extra_meta={
                "embedder": type(self.embedder).__name__,
                "embedding_time_s": (embedding_end_time - embedding_start_time) / 1e9,
            })

        cprint.iprintf(f"*** Saved {meta['n_queries']} queries ({meta['total_tokens']} tokens, "
                       f"{meta['min_tokens']}-{meta['max_tokens_observed']} tokens/query, dim {meta['dim']}) "
                       f"to {os.path.abspath(out_dir)}")
        if save_int8:
            cprint.iprintf(f"*** int8 records: {meta['max_tokens']} x {meta['dim']} = "
                           f"{meta['record_bytes']} bytes/query, scale {meta['int8_scale']:.4f}")
        return meta
