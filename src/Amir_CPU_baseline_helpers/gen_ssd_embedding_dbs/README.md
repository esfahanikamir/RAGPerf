# Flat multi-vector embedding stores (format version 2)

Page embeddings (the **corpus store**) and question embeddings (the **query store**) from a
late-interaction model, written as raw, headerless, 4 KiB-aligned binary files so that the CPU,
the GPU and the FPGA all read the same bytes. Six precisions: `fp32, fp16, bf16, int8, int4, bin`.

Both stores are written by the same code, with the same encodings, the same layout and the same
int8/int4 clip value `c` (computed from the corpus, copied to the queries).

**The byte-level specification, with examples, is [FORMAT_GUIDE.md](FORMAT_GUIDE.md).**

| file | role |
|------|------|
| `quantize.py` | number encodings and the clip value; the only place a float becomes bytes |
| `layout.py` | record/file layout and metadata; the only place records are written |
| `convert.py` | corpus pickle → corpus store |
| `queries.py` | `save_query_db()` used by the query pipeline; re-encode a query store for another corpus |
| `reader.py` | `EmbeddingStore(dir)` reads either store |
| `verify.py` | checks both stores and their consistency; writes reference top-k |
| `check_db.cpp` | independent C++ check, reference decode functions and MaxSim |
| `build_ivf.py` | corpus store → L1 index (IVF store): clusters all vectors, writes centroids, cluster table, members in every precision, member → page tables |
| `verify_ivf.py` | checks the IVF store against the corpus store; reports first-stage quality with a query store |
| `run_maxsim_all.sh` | scores every query against every page in every precision with `check_db` and saves the full rankings |

Dependencies: NumPy (tqdm optional). The query pipeline additionally needs the RAGPerf environment.

## Generate and verify

```bash
python convert.py --input /path/to/pages/embeddings.pkl --outdir /path/to/CORPUS_DB
```

Then run the query pipeline (`ImagesRAGPipeline_rerank_save_q_embeddings`, started through
`run_new.py` as before) with

```bash
export RAGPERF_CORPUS_DB_DIR=/path/to/CORPUS_DB
```

```bash
export RAGPERF_QUERY_DB_DIR=/path/to/QUERY_DB
```

and check the result:

```bash
python verify.py --corpus /path/to/CORPUS_DB --queries /path/to/QUERY_DB --max-queries 0 --save-topk topk.json | tee verify.log
```

```bash
g++ -O2 -std=c++17 -o check_db check_db.cpp
```

```bash
./check_db selftest
```

followed by the two `./check_db check …` command lines that `verify.py` prints. Every tool must
end with `ALL CHECKS PASSED` / `SELFTEST PASSED`.

Other commands:

```bash
python convert.py --input embeddings.pkl --outdir OUT --inspect-only
```

```bash
python queries.py --src QUERY_DB --corpus CORPUS_DB_NEW --outdir QUERY_DB_NEW
```

```bash
./check_db maxsim CORPUS_DB QUERY_DB 128 1031 <max_tokens> 4096 int8 10 5
```

Reference scores of all queries against all pages, one file per precision (see the top of the script):

```bash
./run_maxsim_all.sh CORPUS_DB QUERY_DB OUT_DIR
```

L1 index (first stage), see FORMAT_GUIDE.md section 10:

```bash
python build_ivf.py --corpus CORPUS_DB --outdir IVF_DB
```

```bash
python verify_ivf.py --corpus CORPUS_DB --ivf IVF_DB --queries QUERY_DB | tee verify_ivf.log
```
