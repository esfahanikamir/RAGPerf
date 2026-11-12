#!/usr/bin/env python3
"""Build the L1 (first-stage) IVF index from a corpus store. See FORMAT_GUIDE.md section 10.

    python build_ivf.py --corpus CORPUS_DB --outdir IVF_DB
                        [--cluster-factor 4] [--no-pow2] [--nlist N] [--min-points 30]
                        [--iters 25] [--seed 0] [--train-sample 0] [--init-centroids FILE.npy] [--force]

What it does
  1. reads every fp32 vector of the corpus store (N vectors);
  2. chooses the number of clusters from N:  nlist = smallest power of two >= factor * sqrt(N)
     (factor 4 = lower end of the FAISS range 4*sqrt(N) .. 16*sqrt(N); --nlist overrides);
  3. clusters them with spherical k-means (unit-length centroids, "largest dot product wins");
  4. assigns every vector to its cluster with an exactly defined rule (see assign_exact);
  5. writes the index: centroids, the cluster table, the member vectors grouped by cluster in
     every precision (same encodings and same clip value c as the corpus store), and for every
     member the page it came from.

The index contains the SAME vectors as the corpus store, only in a different order.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    from . import layout as L
    from . import quantize as Q
    from .reader import EmbeddingStore
except ImportError:
    import layout as L
    import quantize as Q
    from reader import EmbeddingStore

BLOCK_BYTES = 256 << 20     # size of one score block (rows x nlist float32) in the BLAS passes
MARGIN_F32 = 1e-4           # float32 scores closer than this are re-decided in float64
MARGIN_F64 = 1e-11          # float64 scores closer than this are re-decided exactly (math.fsum)


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


# --------------------------------------------------------------------------
# Number of clusters
# --------------------------------------------------------------------------
def choose_nlist(n, factor=4.0, pow2=True):
    """nlist = smallest power of two >= factor * sqrt(n)   (without pow2: ceil(factor * sqrt(n)))."""
    target = int(math.ceil(factor * math.sqrt(n)))
    target = max(target, 1)
    return (1 << (target - 1).bit_length()) if pow2 else target


# --------------------------------------------------------------------------
# Scoring / assignment
# --------------------------------------------------------------------------
def _rows_per_block(nlist):
    return max(1, BLOCK_BYTES // (nlist * 4))


def assign_blas(x, cent, want_second=False, desc=None):
    """Nearest centroid of every row by float32 matmul. Returns (index, best score[, second best score])."""
    n, k = x.shape[0], cent.shape[0]
    ct = np.ascontiguousarray(cent.T)
    idx = np.empty(n, dtype=np.int32)
    best = np.empty(n, dtype=np.float32)
    second = np.empty(n, dtype=np.float32) if want_second else None
    step = _rows_per_block(k)
    rows = np.arange(step)
    for a in range(0, n, step):
        b = min(n, a + step)
        with np.errstate(all="ignore"):  # macOS Accelerate emits spurious FP warnings in sgemm
            s = x[a:b] @ ct
        i = s.argmax(axis=1)
        idx[a:b] = i
        best[a:b] = s[rows[:b - a], i]
        if want_second:
            s[rows[:b - a], i] = -np.inf
            second[a:b] = s.max(axis=1) if k > 1 else -np.inf
    return (idx, best, second) if want_second else (idx, best)


def centroid_sums(x, assign, nlist):
    """Per-cluster sum of the member vectors (float64) and member counts."""
    n, dim = x.shape
    counts = np.bincount(assign, minlength=nlist)
    order = np.argsort(assign, kind="stable")
    ids = assign[order]
    sums = np.zeros((nlist, dim), dtype=np.float64)
    step = 1 << 18
    for a in range(0, n, step):
        b = min(n, a + step)
        seg = ids[a:b]
        starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        sums[seg[starts]] += np.add.reduceat(x[order[a:b]].astype(np.float64), starts, axis=0)
    return sums, counts


def kmeans_spherical(x, nlist, iters, seed):
    """Spherical k-means: centroids are unit length, a vector belongs to the centroid with the largest
    dot product. Returns (centroids float32 (nlist, dim), list of per-iteration log dicts).

    The training itself does not have to be reproducible bit for bit (BLAS results differ between
    machines); what is stored and checked afterwards is the final assignment made by assign_exact()."""
    n, dim = x.shape
    rng = np.random.default_rng(seed)
    cent = np.array(x[np.sort(rng.choice(n, size=nlist, replace=False))], dtype=np.float64)
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-30)
    cent = cent.astype(np.float32)
    prev = None
    log = []
    for it in range(iters):
        t0 = time.time()
        assign, best = assign_blas(x, cent)
        changed = n if prev is None else int(np.count_nonzero(assign != prev))
        prev = assign
        sums, counts = centroid_sums(x, assign, nlist)
        norms = np.linalg.norm(sums, axis=1)
        dead = (counts == 0) | (norms < 1e-12)
        new = np.zeros_like(sums)
        new[~dead] = sums[~dead] / norms[~dead, None]
        ndead = int(dead.sum())
        if ndead:  # restart empty clusters at the vectors that fit their current centroid worst
            worst = np.argpartition(best, ndead - 1)[:ndead]
            w = x[np.sort(worst)].astype(np.float64)
            new[dead] = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        cent = new.astype(np.float32)
        log.append({"iteration": it + 1, "mean_best_score": float(best.mean(dtype=np.float64)), "changed": changed,
                    "empty_clusters_restarted": ndead, "seconds": round(time.time() - t0, 2)})
        print("  k-means %3d/%d  mean score %.6f  changed %9d  empty restarted %5d  %.1fs"
              % (it + 1, iters, log[-1]["mean_best_score"], changed, ndead, time.time() - t0))
        if changed == 0 and ndead == 0:
            break
    return cent, log


def exact_dot(a, b):
    """The dot product of two float32 vectors, correctly rounded to a double. Each product of two
    float32 values is exact in a double, and math.fsum adds them without rounding error."""
    return math.fsum(float(u) * float(v) for u, v in zip(a, b))


def assign_exact(x, cent):
    """Final assignment, defined so that it does not depend on the machine:

        cluster(x) = the centroid with the largest EXACT dot product sum_d x[d] * c[d];
                     exact ties go to the lowest cluster number.

    Almost every vector is decided by a float32 matmul because its best and second-best scores are
    far apart. The close ones are recomputed in float64, and the still-closer ones exactly.
    Returns (assign int32, info dict)."""
    n, k = x.shape[0], cent.shape[0]
    idx, best, second = assign_blas(x, cent, want_second=True)
    amb = np.flatnonzero(best - second < MARGIN_F32)
    c64 = cent.astype(np.float64)
    ct = np.ascontiguousarray(cent.T)
    n_exact = 0
    for r in amb:
        with np.errstate(all="ignore"):
            s = x[r] @ ct
            cand = np.flatnonzero(s >= s.max() - MARGIN_F32)   # ascending cluster numbers
            d = c64[cand] @ x[r].astype(np.float64)
        close = cand[d >= d.max() - MARGIN_F64]
        if len(close) == 1:
            idx[r] = close[0]
        else:
            n_exact += 1
            e = [exact_dot(x[r], cent[j]) for j in close]
            idx[r] = close[int(np.argmax(e))]                   # argmax returns the first (lowest) on ties
    return idx, {"decided_by_float32": int(n - len(amb)), "rechecked_in_float64": int(len(amb)),
                 "rechecked_exactly": int(n_exact), "margin_float32": MARGIN_F32, "margin_float64": MARGIN_F64}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def load_vectors(cs):
    """All fp32 vectors of the corpus store in corpus order: (N, dim) float32, and doc_first (n_docs+1)."""
    counts = np.asarray(cs.counts("fp32"), dtype=np.int64)
    doc_first = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    x = np.empty((int(doc_first[-1]), cs.dim), dtype=np.float32)
    for i in range(cs.num_records):
        x[doc_first[i]:doc_first[i + 1]] = cs.get("fp32", i)
    return x, doc_first


def size_stats(counts):
    c = np.sort(np.asarray(counts, dtype=np.int64))
    q = lambda p: int(c[min(len(c) - 1, int(p * len(c)))])
    return {"min": int(c[0]), "median": q(0.5), "mean": float(c.mean()), "p90": q(0.9), "p99": q(0.99), "max": int(c[-1]),
            "empty_clusters": int(np.count_nonzero(c == 0)), "largest_5": c[::-1][:5].tolist(),
            "share_of_largest": float(c[-1]) / float(c.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus store written by convert.py")
    ap.add_argument("--outdir", required=True, help="IVF store directory to create")
    ap.add_argument("--cluster-factor", type=float, default=4.0,
                    help="nlist = smallest power of two >= factor * sqrt(N) (default 4; FAISS range is 4 .. 16)")
    ap.add_argument("--no-pow2", action="store_true", help="use ceil(factor * sqrt(N)) without rounding up to a power of two")
    ap.add_argument("--nlist", type=int, default=0, help="force this number of clusters (overrides the rule)")
    ap.add_argument("--min-points", type=float, default=30.0,
                    help="stop if there are fewer than this many vectors per cluster on average (FAISS minimum 30; 0 disables)")
    ap.add_argument("--iters", type=int, default=25, help="k-means iterations (default 25, as in FAISS)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the initial centroids / training sample")
    ap.add_argument("--train-sample", type=int, default=0, help="train k-means on this many random vectors (0 = all)")
    ap.add_argument("--init-centroids", help=".npy file (nlist, dim) with centroids trained elsewhere; skips k-means")
    ap.add_argument("--force", action="store_true", help="overwrite an existing IVF store")
    args = ap.parse_args(argv)

    t0 = time.time()
    cs = EmbeddingStore(args.corpus, "corpus")
    cm = cs.meta
    dim, pad, c = cs.dim, cs.pad, cs.clip_c
    x, doc_first = load_vectors(cs)
    n = x.shape[0]
    print("corpus          : %s (%d pages, %d vectors, dim %d, clip c %.8g)" % (args.corpus, cs.num_records, n, dim, c))
    if int(doc_first[-1]) >= 2 ** 32 or int(np.diff(doc_first).max()) > 2 ** 16:
        die("too many vectors for the uint32 / uint16 tables")

    rule_nlist = choose_nlist(n, args.cluster_factor, not args.no_pow2)
    nlist = args.nlist if args.nlist > 0 else rule_nlist
    print("clusters        : %d  (rule: %s%.3g * sqrt(%d) = %.1f -> %d%s)"
          % (nlist, "ceil of " if args.no_pow2 else "smallest power of two >= ", args.cluster_factor, n,
             args.cluster_factor * math.sqrt(n), rule_nlist, "; OVERRIDDEN by --nlist" if args.nlist > 0 else ""))
    if nlist > n:
        die("more clusters (%d) than vectors (%d)" % (nlist, n))
    if args.min_points > 0 and n < args.min_points * nlist:
        die("only %.1f vectors per cluster on average; at least %g are needed for a meaningful k-means "
            "(FAISS minimum). Use a smaller --cluster-factor / --nlist, or --min-points 0 to build anyway."
            % (n / float(nlist), args.min_points))

    os.makedirs(args.outdir, exist_ok=True)
    existing = L.existing_store_files(args.outdir, "ivf")
    if existing:
        if not args.force:
            die("output directory already has %s ... (use --force to overwrite)" % existing[:3])
        L.remove_store_files(args.outdir, "ivf")

    # ---- centroids -------------------------------------------------------------------------------
    if args.init_centroids:
        cent = np.load(args.init_centroids).astype(np.float64)
        if cent.shape != (nlist, dim):
            die("--init-centroids has shape %s, expected (%d, %d)" % (cent.shape, nlist, dim))
        cent = (cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-30)).astype(np.float32)
        klog, training = [], {"method": "centroids loaded from file", "file": os.path.abspath(args.init_centroids)}
    else:
        if args.train_sample and args.train_sample < n:
            if args.train_sample < nlist:
                die("--train-sample is smaller than the number of clusters")
            sel = np.sort(np.random.default_rng(args.seed + 1).choice(n, size=args.train_sample, replace=False))
            xt = x[sel]
        else:
            xt = x
        print("k-means         : %d clusters, %d training vectors, up to %d iterations, seed %d"
              % (nlist, xt.shape[0], args.iters, args.seed))
        cent, klog = kmeans_spherical(xt, nlist, args.iters, args.seed)
        training = {"method": "spherical k-means (numpy)", "iterations_run": len(klog), "max_iterations": args.iters,
                    "seed": args.seed, "training_vectors": int(xt.shape[0]),
                    "init": "random distinct training vectors; empty clusters restarted at the worst-fitting vectors",
                    "final_mean_best_score": klog[-1]["mean_best_score"], "log": klog}
        del xt
    # The fp32 centroids are the master and keep full float32 precision. (Rounding them to bf16 first
    # would make near-identical centroids exactly identical; all their members would then fall into
    # the lowest-numbered one, leaving empty clusters and one very large cluster.)
    if not np.all(np.isfinite(cent)):
        die("non-finite centroids")

    # ---- final assignment and ordering -------------------------------------------------------------
    t1 = time.time()
    assign, ainfo = assign_exact(x, cent)
    print("assignment      : %d by float32, %d rechecked in float64, %d rechecked exactly  (%.1fs)"
          % (ainfo["decided_by_float32"], ainfo["rechecked_in_float64"], ainfo["rechecked_exactly"], time.time() - t1))
    counts = np.bincount(assign, minlength=nlist).astype(np.int64)
    first = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    perm = np.argsort(assign, kind="stable")                    # member m is corpus vector perm[m]
    member_doc = np.searchsorted(doc_first, perm, side="right") - 1
    member_pos = perm - doc_first[member_doc]
    stats = size_stats(counts)
    print("cluster sizes   : min %d, median %d, mean %.1f, p99 %d, max %d (%.2f%% of all vectors), %d empty"
          % (stats["min"], stats["median"], stats["mean"], stats["p99"], stats["max"],
             100 * stats["share_of_largest"], stats["empty_clusters"]))

    # ---- write -------------------------------------------------------------------------------------
    print("=== Writing %s to %s ===" % (",".join(Q.PRECISIONS), args.outdir))
    tables = {
        "cluster_first": L.write_table(args.outdir, "ivf_cluster_first.u64", first, "<u8"),
        "cluster_counts": L.write_table(args.outdir, "ivf_cluster_counts.u32", counts, "<u4"),
        "member_doc": L.write_table(args.outdir, "ivf_member_doc.u32", member_doc, "<u4"),
        "member_pos": L.write_table(args.outdir, "ivf_member_pos.u16", member_pos, "<u2"),
        "vector_cluster": L.write_table(args.outdir, "ivf_vector_cluster.u32", assign, "<u4"),
    }
    step = 1 << 16
    pinfo = {}
    for p in Q.PRECISIONS:
        t2 = time.time()
        cen = L.write_dense(args.outdir, "ivf_centroids_%s.bin" % p, p, [cent], nlist, dim, pad, c)
        mem = L.write_dense(args.outdir, "ivf_members_%s.bin" % p, p,
                            (x[perm[a:a + step]] for a in range(0, n, step)), n, dim, pad, c)
        pinfo[p] = {"bytes_per_vector": mem["bytes_per_vector"], "element_dtype": Q.ELEMENT_DTYPE[p],
                    "numpy_dtype": Q.NUMPY_DTYPE[p], "bits_per_value": Q.BITS_PER_VALUE[p],
                    "members": mem, "centroids": cen}
        print("  %-5s bytes/vector=%4d  members %14d B  centroids %10d B  %.1fs"
              % (p, mem["bytes_per_vector"], mem["file_bytes"], cen["file_bytes"], time.time() - t2))

    meta = {
        "format": L.FORMAT, "format_version": L.FORMAT_VERSION, "kind": "ivf", "file_prefix": "ivf",
        "created_utc": L.utc_now(), "numpy_version": np.__version__,
        "num_vectors": int(n), "nlist": int(nlist), "dim": int(dim), "pad_bytes": int(pad),
        "num_docs": int(cs.num_records), "corpus_slots_per_record": int(cs.slots), "max_cluster_size": int(counts.max()),
        "nlist_rule": {"text": "nlist = %s(factor * sqrt(num_vectors))" % ("ceil" if args.no_pow2 else "smallest power of two >= "),
                       "factor": args.cluster_factor, "power_of_two": not args.no_pow2, "rule_value": int(rule_nlist),
                       "overridden": bool(args.nlist > 0), "vectors_per_cluster": n / float(nlist)},
        "similarity": "dot product with the stored fp32 centroid (centroids are unit length up to float32 rounding)",
        "assignment": "cluster(x) = centroid with the largest exact dot product; exact ties -> lowest cluster number",
        "assignment_info": ainfo,
        "member_order": "inside a cluster, members are sorted by (page record, vector number in the page)",
        "indexing": "member m: vector bytes at m * bytes_per_vector in ivf_members_<P>.bin; page = member_doc[m]; "
                    "vector in the page = member_pos[m]. cluster j owns members cluster_first[j] .. cluster_first[j+1]-1. "
                    "cluster_first counts MEMBERS, not bytes. No padding between vectors or clusters; only the end of "
                    "each file is zero-padded to pad_bytes.",
        "training": training,
        "cluster_sizes": stats,
        "corpus": dict(L.corpus_identity(cm), path_when_written=os.path.abspath(args.corpus)),
        "quantization": {"clip_c": cm["quantization"]["clip_c"], "clip_c_float32_hex": cm["quantization"]["clip_c_float32_hex"],
                         "clip_source": "copied from the corpus store", "int8_qmax": Q.INT8_QMAX, "int4_qmax": Q.INT4_QMAX,
                         "zero_point": 0},
        "packing": Q.PACKING,
        "tables": tables,
        "precisions": pinfo,
    }
    meta["sha256"] = dict((t["file"], t["sha256"]) for t in tables.values())
    for v in pinfo.values():
        meta["sha256"][v["members"]["file"]] = v["members"]["sha256"]
        meta["sha256"][v["centroids"]["file"]] = v["centroids"]["sha256"]
    L.write_metadata(args.outdir, "ivf", meta)  # written last: its presence marks a complete store
    print("wrote ivf_metadata.json; total %.1fs" % (time.time() - t0))
    print("next: python verify_ivf.py --corpus %s --ivf %s [--queries QUERY_DB]" % (args.corpus, args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
