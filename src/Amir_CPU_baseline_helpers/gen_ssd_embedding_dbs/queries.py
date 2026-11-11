#!/usr/bin/env python3
"""Build the QUERY store: the token embeddings of every question, written in
the same format, with the same encodings and the same clip value c as the
corpus store. See FORMAT_GUIDE.md.

The query store is always built FOR a corpus store: c is read from
CORPUS_DB/metadata.json, never computed from the queries. There is no code
path that writes int8 / int4 queries with any other scale.

Normal use: the RAG pipeline (ImagesRAGPipeline_rerank_save_q_embeddings)
embeds the questions and calls save_query_db() below.

Command line: re-encode an existing query store for another (e.g. regenerated)
corpus store, without the GPU / encoder. The fp32 query file is lossless, so
this gives exactly what the pipeline would have written:

    python queries.py --src QUERY_DB --corpus CORPUS_DB --outdir QUERY_DB_NEW [--force]
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    from . import layout as L
    from . import quantize as Q
    from .reader import EmbeddingStore
except ImportError:
    import layout as L
    import quantize as Q
    from reader import EmbeddingStore

QUERIES_FILE = "queries.jsonl"
TOKEN_ROUND = 32  # max_tokens is rounded up to a multiple of this


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def save_query_db(questions, answers, vectors, corpus_dir, out_dir, max_tokens=None, source=None, force=False):
    """Write a complete query store and return its metadata dict.

    questions  : list of N strings.
    answers    : list of N ground-truth answers (or None).
    vectors    : list of N float32 arrays, vectors[i].shape == (n_tokens_i, dim).
                 They must hold the encoder output EXACTLY (fp16 / bf16 -> float32
                 is exact); the caller does that conversion.
    corpus_dir : corpus store written by convert.py. Its clip value c, dim and
                 pad_bytes are used.
    max_tokens : vector slots per record; default = longest query rounded up to
                 a multiple of 32.
    source     : free-form dict stored in the metadata (encoder name, dtype, ...).
    """
    _, cmeta = L.load_metadata(corpus_dir, "corpus")
    c = Q.clip_from_meta(cmeta)
    dim = int(cmeta["dim"])
    pad = int(cmeta["pad_bytes"])

    n = len(questions)
    if n == 0:
        raise ValueError("no queries")
    if len(vectors) != n:
        raise ValueError("%d questions but %d embedding arrays" % (n, len(vectors)))
    answers = list(answers) if answers is not None else [None] * n
    if len(answers) != n:
        raise ValueError("%d questions but %d answers" % (n, len(answers)))
    for i, v in enumerate(vectors):
        if not isinstance(v, np.ndarray) or v.dtype != np.float32:
            raise TypeError("vectors[%d] must be a float32 numpy array, got %s %s"
                            % (i, type(v).__name__, getattr(v, "dtype", "")))
        if v.ndim != 2 or v.shape[1] != dim or v.shape[0] < 1:
            raise ValueError("vectors[%d] has shape %s; expected (n_tokens >= 1, %d) to match the corpus dim"
                             % (i, v.shape, dim))
        if not np.all(np.isfinite(v)):
            raise ValueError("vectors[%d] contains NaN or Inf" % i)
    counts = np.array([v.shape[0] for v in vectors], dtype=np.int64)
    observed = int(counts.max())
    if max_tokens is None:
        max_tokens = L.round_up(observed, TOKEN_ROUND)
    max_tokens = int(max_tokens)
    if observed > max_tokens:
        raise ValueError("a query has %d tokens, more than max_tokens=%d" % (observed, max_tokens))

    os.makedirs(out_dir, exist_ok=True)
    existing = L.existing_store_files(out_dir, "queries")
    if existing:
        if not force:
            raise IOError("%s already has %s ... (pass force=True / --force to overwrite)" % (out_dir, existing[:3]))
        L.remove_store_files(out_dir, "queries")

    # text + per-query info
    qpath = os.path.join(out_dir, QUERIES_FILE)
    with open(qpath + ".tmp", "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            f.write(json.dumps({"record": i, "question": q, "gt_answer": _jsonable(answers[i]),
                                "n_tokens": int(counts[i])}, ensure_ascii=False, default=str) + "\n")
    os.replace(qpath + ".tmp", qpath)

    print("=== Writing query store %s (%d queries, %d..%d tokens, max_tokens %d, clip c %.8g from %s) ==="
          % (out_dir, n, counts.min(), observed, max_tokens, c, corpus_dir))
    pinfo = {}
    for p in Q.PRECISIONS:
        pinfo[p] = L.write_precision(out_dir, "qry", p, vectors, counts, max_tokens, dim, pad, c)

    meta = L.base_metadata("queries", n, max_tokens, dim, pad)
    meta["num_queries"] = int(n)             # same as num_records
    meta["max_tokens"] = int(max_tokens)     # same as slots_per_record
    meta["min_tokens_observed"] = int(counts.min())
    meta["max_tokens_observed"] = observed
    meta["total_tokens"] = int(counts.sum())
    meta["source"] = dict(source or {})
    meta["corpus"] = L.corpus_identity(cmeta)
    meta["corpus"]["path_when_written"] = os.path.abspath(corpus_dir)
    cq = cmeta["quantization"]
    meta["quantization"] = {
        "clip_c": cq["clip_c"],
        "clip_c_float32_hex": cq["clip_c_float32_hex"],
        "clip_source": "copied from the corpus store (never computed from the queries)",
        "int8_qmax": Q.INT8_QMAX,
        "int4_qmax": Q.INT4_QMAX,
        "zero_point": 0,
    }
    meta["packing"] = Q.PACKING
    meta["precisions"] = pinfo
    meta["sha256"] = {QUERIES_FILE: L.sha256_file(qpath)}
    for p in pinfo.values():
        meta["sha256"].update(p["sha256"])
    meta["queries_file"] = QUERIES_FILE
    meta["query_order"] = "record i = question i in the order given to save_query_db (dataset order)"
    L.write_metadata(out_dir, "queries", meta)  # written last: its presence marks a complete store

    _read_back_check(out_dir, vectors, c)
    return meta


def _read_back_check(out_dir, vectors, c):
    """Re-open the store from disk and compare every record with what was meant to be written."""
    st = EmbeddingStore(out_dir, "queries")
    for p in st.precisions:
        for i, v in enumerate(vectors):
            if not np.array_equal(st.raw(p, i), Q.encode(p, v, c)):
                raise IOError("read-back mismatch: %s record %d" % (p, i))
            if np.any(st.raw_record(p, i)[v.shape[0] * st.raw(p, i).shape[1]:]):
                raise IOError("read-back: non-zero padding in %s record %d" % (p, i))
    for i, v in enumerate(vectors):
        if not np.array_equal(st.get("fp32", i), v):
            raise IOError("read-back mismatch: fp32 values of record %d" % i)
    print("read-back check : OK (%d queries x %d precisions)" % (len(vectors), len(st.precisions)))


def load_query_db(qdir, precision="fp32", binary="bits"):
    """-> (questions, answers, list of per-query arrays in `precision`, metadata)."""
    st = EmbeddingStore(qdir, "queries")
    rows = st.rows()
    embs = [np.array(st.get(precision, i, binary=binary)) for i in range(st.num_records)]
    return [r["question"] for r in rows], [r["gt_answer"] for r in rows], embs, st.meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="existing query store (its lossless fp32 file is the input)")
    ap.add_argument("--corpus", required=True, help="corpus store whose clip value c is to be used")
    ap.add_argument("--outdir", required=True, help="query store to write (may equal --src with --force)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing query store in --outdir")
    args = ap.parse_args(argv)

    questions, answers, vectors, old = load_query_db(args.src, "fp32")  # fully in memory before anything is deleted
    source = dict(old.get("source") or {})
    source["re_encoded_from"] = {"path": os.path.abspath(args.src), "created_utc": old.get("created_utc"),
                                 "corpus_id": (old.get("corpus") or {}).get("corpus_id")}
    meta = save_query_db(questions, answers, vectors, args.corpus, args.outdir,
                         max_tokens=old["max_tokens"], source=source, force=args.force)
    print("wrote %s (%d queries); next: python verify.py --corpus %s --queries %s"
          % (args.outdir, meta["num_queries"], args.corpus, args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
