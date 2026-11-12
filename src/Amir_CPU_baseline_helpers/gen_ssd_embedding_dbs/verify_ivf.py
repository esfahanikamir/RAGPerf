#!/usr/bin/env python3
"""Verify an L1 IVF store against the corpus store it was built from and, with a query store,
measure how good the first stage is.

    python verify_ivf.py --corpus CORPUS_DB --ivf IVF_DB [--queries QUERY_DB]
                         [--max-queries 32] [--k 10] [--nprobe 1,2,4,8,16,32,64]
                         [--precisions fp32,fp16,bf16,int8,int4,bin] [--save-hits hits.json]
                         [--skip-assign] [--skip-hash]

  1. structure  : metadata, cluster table, file sizes, zero tails, SHA-256, the nlist rule
  2. members    : the index holds every corpus vector exactly once, with the right page / position,
                  and the stored bytes equal the corpus store's bytes in every precision
  3. centroids  : every precision of the centroids is encode(fp32 centroids, c)
  4. assignment : every vector is in the cluster with the largest exact dot product (recomputed)
  5. quality    : (with --queries) per precision and per number of probed clusters: how many of the
                  true nearest vectors the first stage finds, how many pages become candidates, and
                  whether the pages that exact MaxSim ranks best are among them

Exit code is non-zero if any hard check fails.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from . import build_ivf as B
    from . import layout as L
    from . import quantize as Q
    from . import verify as V
    from .reader import EmbeddingStore, IVFStore
except ImportError:
    import build_ivf as B
    import layout as L
    import quantize as Q
    import verify as V
    from reader import EmbeddingStore, IVFStore

check = V.check


# --------------------------------------------------------------------------
# 1. structure
# --------------------------------------------------------------------------
def check_structure(ivf, cs, skip_hash):
    m, cm = ivf.meta, cs.meta
    n, k, dim, pad = ivf.num_vectors, ivf.nlist, ivf.dim, ivf.pad
    print("\n=== 1. Structure of the IVF store %s (vectors=%d, clusters=%d, dim=%d) ===" % (ivf.outdir, n, k, dim))
    counts_c = np.asarray(cs.counts("fp32"), dtype=np.int64)
    check(n == int(counts_c.sum()), "num_vectors == number of vectors in the corpus store (%d)" % counts_c.sum())
    check(m["corpus"]["corpus_id"] == cm["corpus_id"], "built from this corpus (corpus_id %s...)" % cm["corpus_id"][:16])
    check(m["quantization"]["clip_c_float32_hex"] == cm["quantization"]["clip_c_float32_hex"] and ivf.clip_c == cs.clip_c,
          "same clip value c as the corpus (%s)" % cm["quantization"]["clip_c_float32_hex"])
    check(dim == cs.dim and pad == cs.pad and m["num_docs"] == cs.num_records, "same dim, pad_bytes and page count as the corpus")
    check(list(m["precisions"].keys()) == list(cm["precisions"].keys()), "same precisions as the corpus (%s)" % ",".join(m["precisions"]))
    check(m["packing"] == cm["packing"], "identical packing conventions recorded")
    r = m["nlist_rule"]
    rule = B.choose_nlist(n, r["factor"], r["power_of_two"])
    check(r["rule_value"] == rule and (r["overridden"] or k == rule),
          "nlist rule: %s%g * sqrt(%d) -> %d%s" % ("smallest power of two >= " if r["power_of_two"] else "ceil of ",
                                                  r["factor"], n, rule, " (overridden: %d)" % k if r["overridden"] else ""))
    first, counts = ivf.first.astype(np.int64), ivf.counts.astype(np.int64)
    check(first[0] == 0 and first[-1] == n and np.all(np.diff(first) >= 0), "cluster_first starts at 0, never decreases, ends at num_vectors")
    check(np.array_equal(np.diff(first), counts) and counts.sum() == n, "cluster_counts == differences of cluster_first, sum == num_vectors")
    check(m["max_cluster_size"] == int(counts.max()), "max_cluster_size in metadata (%d)" % counts.max())
    check(B.size_stats(counts) == m["cluster_sizes"], "cluster size statistics in metadata match the table")
    for key, t in m["tables"].items():
        path = os.path.join(ivf.outdir, t["file"])
        check(os.path.getsize(path) == t["file_bytes"] == t["entries"] * np.dtype(t["dtype"]).itemsize, "table %s size" % t["file"])
        if not skip_hash:
            check(L.sha256_file(path) == t["sha256"] == m["sha256"].get(t["file"]), "sha256 %s" % t["file"])
    print("  %-5s %6s %16s %14s" % ("prec", "B/vec", "members bytes", "centroid bytes"))
    for p, pm in m["precisions"].items():
        try:
            ivf._open(p)
        except Exception as e:
            check(False, "%s: cannot open: %s" % (p, e))
            continue
        bpv = pm["bytes_per_vector"]
        print("  %-5s %6d %16d %14d" % (p, bpv, pm["members"]["file_bytes"], pm["centroids"]["file_bytes"]))
        check(bpv == Q.bytes_per_vector(p, dim), "%s bytes/vector = %d" % (p, bpv))
        for key, cnt in (("members", n), ("centroids", k)):
            f = pm[key]
            data = ivf.raw_file(p, key)
            check(f["vectors"] == cnt and f["payload_bytes"] == cnt * bpv and f["file_bytes"] == L.round_up(cnt * bpv, pad) == data.size,
                  "%s %s: %d vectors, %d bytes + zero tail to a multiple of %d" % (p, key, cnt, cnt * bpv, pad))
            check(not np.any(data[cnt * bpv:]), "%s %s: tail after the last vector is zero" % (p, key))
            if not skip_hash:
                check(L.sha256_file(os.path.join(ivf.outdir, f["file"])) == f["sha256"] == m["sha256"].get(f["file"]), "sha256 %s" % f["file"])


# --------------------------------------------------------------------------
# 2. members
# --------------------------------------------------------------------------
def check_members(ivf, cs):
    n, k = ivf.num_vectors, ivf.nlist
    print("\n=== 2. Members: the index holds every corpus vector exactly once ===")
    counts_c = np.asarray(cs.counts("fp32"), dtype=np.int64)
    doc_first = np.concatenate([[0], np.cumsum(counts_c)]).astype(np.int64)
    doc = ivf.member_doc.astype(np.int64)
    pos = ivf.member_pos.astype(np.int64)
    ok = bool(doc.max() < cs.num_records and np.all(pos < counts_c[np.minimum(doc, cs.num_records - 1)]))
    check(ok, "member_doc < number of pages, member_pos < vectors in that page")
    if not ok:
        return None
    g = doc_first[doc] + pos                         # corpus vector number of member m
    inv = np.full(n, -1, dtype=np.int64)
    inv[g] = np.arange(n)
    check(np.all(inv >= 0) and len(np.unique(g)) == n, "every corpus vector appears exactly once")
    cluster_of_member = np.repeat(np.arange(k), ivf.counts.astype(np.int64))
    check(np.array_equal(ivf.vector_cluster.astype(np.int64)[g], cluster_of_member),
          "vector_cluster (corpus order) agrees with the cluster each member is stored in")
    inside = np.diff(g) > 0                              # pair (m, m+1) is in ascending corpus order
    b = ivf.first[1:-1].astype(np.int64)
    inside[b[(b > 0) & (b < n)] - 1] = True              # ... except across a cluster boundary
    check(np.all(inside), "inside a cluster, members are sorted by (page, position)")
    for p in ivf.precisions:
        try:
            mem = ivf.raw_file(p)[:n * ivf.meta["precisions"][p]["bytes_per_vector"]].reshape(n, -1)
        except Exception as e:   # already reported in section 1
            check(False, "%-5s members cannot be read: %s" % (p, e))
            continue
        bad = 0
        for i in range(cs.num_records):
            if not np.array_equal(mem[inv[doc_first[i]:doc_first[i + 1]]], cs.raw(p, i)):
                bad += 1
        check(bad == 0, "%-5s member bytes == corpus store bytes for all %d pages (%d vectors)" % (p, cs.num_records, n))
    return g


# --------------------------------------------------------------------------
# 3. centroids
# --------------------------------------------------------------------------
def check_centroids(ivf):
    print("\n=== 3. Centroids ===")
    try:
        c32 = np.array(ivf.centroids("fp32"))
    except Exception as e:
        check(False, "fp32 centroids cannot be read: %s" % e)
        return None
    check(np.all(np.isfinite(c32)), "fp32 centroids: no NaN / Inf")
    norms = np.linalg.norm(c32.astype(np.float64), axis=1)
    check(abs(norms.min() - 1) < 1e-5 and abs(norms.max() - 1) < 1e-5, "fp32 centroids are unit length (norms %.7f .. %.7f)" % (norms.min(), norms.max()))
    print("  distinct fp32 centroids: %d of %d" % (len(np.unique(c32, axis=0)), ivf.nlist))
    for p in ivf.precisions:
        if p != "fp32":
            try:
                same = np.array_equal(ivf.centroids_raw(p), Q.encode(p, c32, ivf.clip_c))
            except Exception:
                same = False
            check(same, "%-5s centroid bytes == encode(fp32 centroids, c)" % p)
    return c32


# --------------------------------------------------------------------------
# 4. assignment
# --------------------------------------------------------------------------
def check_assignment(ivf, cs, c32):
    print("\n=== 4. Assignment: every vector is in the cluster with the largest exact dot product ===")
    t0 = time.time()
    x, _ = B.load_vectors(cs)
    assign, info = B.assign_exact(x, c32)
    diff = int(np.count_nonzero(assign.astype(np.int64) != ivf.vector_cluster.astype(np.int64)))
    check(diff == 0, "recomputed assignment of all %d vectors == stored assignment (%d differ; %d were close calls) in %.1fs"
          % (x.shape[0], diff, info["rechecked_in_float64"], time.time() - t0))


# --------------------------------------------------------------------------
# 5. quality of the first stage
# --------------------------------------------------------------------------
def search_token(qv, cen_f, first, get_members, nprobe, k):
    """Reference L1 search for one query vector.

    1. score all centroids, take the nprobe best (ties: lower cluster number first);
    2. score every member of those clusters;
    3. return the k best members (ties: lower member number first).
    Returns (member numbers, scores, number of members scanned)."""
    with np.errstate(all="ignore"):
        s = cen_f @ qv
    probe = np.sort(np.argsort(-s, kind="stable")[:nprobe])
    idx = np.concatenate([np.arange(first[j], first[j + 1]) for j in probe])      # ascending member numbers
    if len(idx) == 0:
        return idx, np.zeros(0, np.float32), 0
    with np.errstate(all="ignore"):
        sc = get_members(idx) @ qv
    top = np.argsort(-sc, kind="stable")[:k]
    return idx[top], sc[top], len(idx)


def true_topk_scores(ivf, q32, k):
    """For every query vector: the k-th best exact fp32 score over ALL members (brute force)."""
    n, dim = ivf.num_vectors, ivf.dim
    mem = ivf.raw_file("fp32")[:n * 4 * dim].view("<f4").reshape(n, dim)
    best = np.full((k, q32.shape[0]), -np.inf, dtype=np.float32)
    qt = np.ascontiguousarray(q32.T)
    step = max(1, (256 << 20) // (4 * max(1, q32.shape[0])))
    for a in range(0, n, step):
        with np.errstate(all="ignore"):
            s = np.asarray(mem[a:a + step]) @ qt
        kk = min(k, s.shape[0])
        part = -np.partition(-s, kk - 1, axis=0)[:kk]
        best = -np.partition(-np.concatenate([best, part], axis=0), k - 1, axis=0)[:k]
    return best.min(axis=0)


def quality(ivf, cs, qs, precisions, nq, k, nprobes, save_hits):
    n, dim, c = ivf.num_vectors, ivf.dim, ivf.clip_c
    nq = qs.num_records if nq <= 0 else min(nq, qs.num_records)
    qcounts = np.asarray(qs.counts("fp32"), dtype=np.int64)[:nq]
    seg = np.concatenate([[0], np.cumsum(qcounts)]).astype(np.int64)
    T = int(seg[-1])
    nprobes = [p for p in nprobes if p <= ivf.nlist]
    first = ivf.first.astype(np.int64)
    doc = ivf.member_doc.astype(np.int64)
    print("\n=== 5. Quality of the first stage: first %d queries (%d query vectors), top-%d members per query vector ===" % (nq, T, k))
    q32 = np.concatenate([np.asarray(qs.get("fp32", i), dtype=np.float32) for i in range(nq)], axis=0)
    t0 = time.time()
    kth = true_topk_scores(ivf, q32, k)
    exact = V.maxsim_all(cs, "fp32", q32, seg[:-1])
    kp = min(10, cs.num_records)
    best_pages = np.argsort(-exact, axis=1, kind="stable")[:, :kp]
    print("  brute force reference (true top-%d members per query vector, exact fp32 MaxSim page ranking) in %.1fs" % (k, time.time() - t0))
    print("  vector recall = share of the returned members whose TRUE fp32 score is among the true top-%d" % k)
    print("  page columns  = exact-MaxSim best page / best %d pages that are among the candidate pages" % kp)
    mem32 = ivf.raw_file("fp32")[:n * 4 * dim].view("<f4").reshape(n, dim)
    dump = {"k": k, "num_queries": nq, "note": "hits[precision][nprobe][query][query vector] = list of "
            "[member, page, position, score]; int8 / int4 / bin scores are exact integers", "hits": {}}
    for p in precisions:
        cen_f = V._as_float(p, ivf.centroids(p))
        qf = np.concatenate([V._as_float(p, qs.get(p, i)) for i in range(nq)], axis=0)
        memp = ivf.raw_file(p)[:n * ivf.meta["precisions"][p]["bytes_per_vector"]].reshape(n, -1)
        get = lambda idx, p=p, memp=memp: V._as_float(p, Q.decode(p, memp[idx], dim))
        print("  %s" % p)
        print("    %7s %14s %15s %12s %12s %14s" % ("nprobe", "scanned/vector", "vector recall", "candidates", "best page", "best %d pages" % kp))
        for npb in nprobes:
            scanned, rec, cands, top1, topk, hits = 0, 0.0, 0, 0, 0.0, []
            for qi in range(nq):
                pages, qhits = set(), []
                for t in range(int(seg[qi]), int(seg[qi + 1])):
                    idx, sc, ns = search_token(qf[t], cen_f, first, get, npb, k)
                    scanned += ns
                    if len(idx):
                        with np.errstate(all="ignore"):
                            true = np.asarray(mem32[np.sort(idx)]) @ q32[t]
                        rec += np.count_nonzero(true >= kth[t] - 1e-6) / float(k)
                    pages.update(doc[idx].tolist())
                    if save_hits and npb == nprobes[-1]:
                        qhits.append([[int(m), int(doc[m]), int(ivf.member_pos[m]),
                                       int(round(float(s))) if p in ("int8", "int4", "bin") else float(s)] for m, s in zip(idx, sc)])
                cands += len(pages)
                top1 += int(best_pages[qi, 0]) in pages
                topk += len(pages & set(best_pages[qi].tolist())) / float(kp)
                hits.append(qhits)
            print("    %7d %14.0f %15.4f %12.1f %12.4f %14.4f" % (npb, scanned / float(T) + ivf.nlist, rec / T, cands / float(nq),
                                                                   top1 / float(nq), topk / float(nq)))
            if save_hits and npb == nprobes[-1]:
                dump["hits"][p] = {"nprobe": npb, "queries": hits}
    print("  (scanned/vector includes the %d centroids; brute force would scan %d members per query vector)" % (ivf.nlist, n))
    if save_hits:
        with open(save_hits, "w") as f:
            json.dump(dump, f)
        print("  wrote the hits for nprobe=%d to %s" % (nprobes[-1], save_hits))


def facts(ivf, corpus_dir):
    m = ivf.meta
    s = m["cluster_sizes"]
    print("  IVF store %s" % os.path.abspath(ivf.outdir))
    print("    vectors %d, clusters %d (%.1f vectors per cluster), dim %d, clip c %.9g (%s)"
          % (ivf.num_vectors, ivf.nlist, ivf.num_vectors / float(ivf.nlist), ivf.dim, ivf.clip_c, m["quantization"]["clip_c_float32_hex"]))
    print("    cluster sizes: min %d, median %d, p99 %d, max %d (%.2f%% of all vectors), %d empty"
          % (s["min"], s["median"], s["p99"], s["max"], 100 * s["share_of_largest"], s["empty_clusters"]))
    print("    members bytes: " + ", ".join("%s %d" % (p, v["members"]["payload_bytes"]) for p, v in m["precisions"].items()))
    print("    C++ check: ./check_db ivfcheck %s %s %d %d %d %s"
          % (ivf.outdir, corpus_dir, ivf.dim, m["corpus_slots_per_record"], ivf.pad, m["quantization"]["clip_c_float32_hex"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus store the index was built from")
    ap.add_argument("--ivf", required=True, help="IVF store written by build_ivf.py")
    ap.add_argument("--queries", help="query store; enables the quality section")
    ap.add_argument("--max-queries", type=int, default=32, help="queries used for the quality section; 0 = all (default 32)")
    ap.add_argument("--k", type=int, default=10, help="members returned per query vector (default 10)")
    ap.add_argument("--nprobe", default="1,2,4,8,16,32,64", help="numbers of probed clusters to evaluate")
    ap.add_argument("--precisions", default="", help="precisions for the quality section (default: all)")
    ap.add_argument("--save-hits", help="write the hits of the largest nprobe to this JSON file")
    ap.add_argument("--skip-assign", action="store_true", help="skip recomputing the assignment (section 4)")
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)   # show progress even when piped into tee
    except AttributeError:
        pass

    cs = EmbeddingStore(args.corpus, "corpus")
    ivf = IVFStore(args.ivf)
    check_structure(ivf, cs, args.skip_hash)
    check_members(ivf, cs)
    c32 = check_centroids(ivf)
    if V.FAILS:
        print("\nthe store is damaged; skipping the assignment and quality sections")
    elif not args.skip_assign:
        check_assignment(ivf, cs, c32)
    if args.queries and not V.FAILS:
        qs = EmbeddingStore(args.queries, "queries")
        check(qs.meta["corpus"]["corpus_id"] == cs.meta["corpus_id"], "the query store was built for this corpus too")
        precs = [Q.canonical_precision(p) for p in args.precisions.split(",") if p.strip()] or ivf.precisions
        quality(ivf, cs, qs, precs, args.max_queries, args.k, sorted(set(int(v) for v in args.nprobe.split(","))), args.save_hits)

    print("\n=== Facts ===")
    facts(ivf, args.corpus)
    print("\n%s" % ("ALL CHECKS PASSED" if not V.FAILS else "%d CHECK(S) FAILED:\n  " % len(V.FAILS) + "\n  ".join(V.FAILS)))
    return 1 if V.FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
