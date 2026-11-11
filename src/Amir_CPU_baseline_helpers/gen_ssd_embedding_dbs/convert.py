#!/usr/bin/env python3
"""Build the CORPUS store: convert a pickle of per-vector fp32 page embeddings
into flat, headerless, 4 KiB-aligned binary files at every precision
(fp32, fp16, bf16, int8, int4, bin). See FORMAT_GUIDE.md.

    python convert.py --input embeddings.pkl --outdir CORPUS_DB \
        [--pad 4096] [--percentile 99.99] [--precisions fp32,fp16,bf16,int8,int4,bin] [--force]

The clip value c used by int8 / int4 is computed here, from the corpus, and
stored in CORPUS_DB/metadata.json. The query store is later encoded with that
same c (queries.py), never with a value of its own.
"""

import argparse
import collections
import os
import pickle
import sys
import time

import numpy as np

try:
    from . import layout as L
    from . import quantize as Q
except ImportError:
    import layout as L
    import quantize as Q

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kw):
        return it

def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


# --------------------------------------------------------------------------
# Inspection + grouping
# --------------------------------------------------------------------------
def inspect_and_index(obj, dim):
    """One pass over the pickle: validate entries, collect doc_id/seq_id/filepath.
    Returns (keys, doc_ids, seq_ids, filepaths) and prints a summary."""
    print("=== Input summary ===")
    print("type            : %s" % type(obj).__name__)
    if isinstance(obj, dict):
        keys = list(obj.keys())
    elif isinstance(obj, (list, tuple)):
        keys = list(range(len(obj)))
    else:
        die("expected a dict or list of entries, got %s" % type(obj).__name__)
    m = len(keys)
    print("entries         : %d" % m)
    if m == 0:
        die("input is empty")
    key_types = collections.Counter(type(k).__name__ for k in keys)
    print("key types       : %s" % dict(key_types))
    if all(isinstance(k, (int, np.integer)) for k in keys):
        ka = np.asarray(keys, dtype=np.int64)
        print("key range       : %d .. %d (contiguous 0..M-1: %s)"
              % (ka.min(), ka.max(), bool(ka.min() == 0 and ka.max() == m - 1 and len(np.unique(ka)) == m)))

    first = obj[keys[0]]
    if not isinstance(first, dict):
        die("entries must be dicts, got %s" % type(first).__name__)
    print("entry fields    : %s" % sorted(first.keys()))
    for f in ("vector", "doc_id", "seq_id"):
        if f not in first:
            die("entry has no %r field" % f)

    doc_ids = np.empty(m, dtype=np.int64)
    seq_ids = np.empty(m, dtype=np.int64)
    filepaths = [None] * m
    shapes = collections.Counter()
    dtypes = collections.Counter()
    fields = collections.Counter()
    for i, k in enumerate(tqdm(keys, desc="scan", mininterval=2)):
        e = obj[k]
        v = e["vector"]
        shapes[getattr(v, "shape", None)] += 1
        dtypes[str(getattr(v, "dtype", type(v).__name__))] += 1
        doc_ids[i] = e["doc_id"]
        seq_ids[i] = e["seq_id"]
        filepaths[i] = e.get("filepath")
        if i < 1000 or i % 997 == 0:
            fields[tuple(sorted(e.keys()))] += 1
    print("vector shapes   : %s" % dict(shapes))
    print("vector dtypes   : %s" % dict(dtypes))
    if len(fields) > 1:
        print("WARNING: entries have differing field sets: %s" % dict(fields))
    print("doc_id range    : %d .. %d" % (doc_ids.min(), doc_ids.max()))
    print("seq_id range    : %d .. %d" % (seq_ids.min(), seq_ids.max()))
    if set(shapes) != {(dim,)}:
        die("every vector must have shape (%d,), found %s" % (dim, dict(shapes)))
    return keys, doc_ids, seq_ids, filepaths


def build_array(obj, keys, doc_ids, seq_ids, filepaths, vpd, dim, release=True):
    """Group by doc_id, order by seq_id, fill one contiguous (N, vpd, dim) float32 array."""
    order = np.lexsort((seq_ids, doc_ids))  # sorted by doc_id, then seq_id
    udocs, counts = np.unique(doc_ids, return_counts=True)
    n = len(udocs)
    hist = collections.Counter(counts.tolist())
    print("documents       : %d" % n)
    print("vectors/doc hist: %s" % dict(sorted(hist.items())))
    bad = udocs[counts != vpd]
    if len(bad):
        die("%d documents do not have exactly %d vectors, e.g. doc_id %s -> %s"
            % (len(bad), vpd, bad[:5].tolist(), counts[counts != vpd][:5].tolist()))

    s_sorted = seq_ids[order].reshape(n, vpd)
    if np.any(np.diff(s_sorted, axis=1) == 0):
        die("duplicate seq_id within a document")
    per_doc_zero_based = bool(np.all(s_sorted == np.arange(vpd)[None, :]))
    print("seq_id layout   : %s" % ("per-document 0..%d" % (vpd - 1) if per_doc_zero_based
                                    else "not per-document 0..%d (only used for ordering)" % (vpd - 1)))

    # Destination row of every entry in the flat (N*vpd, dim) array.
    dest = np.empty(len(keys), dtype=np.int64)
    dest[order] = np.arange(len(keys), dtype=np.int64)

    # One filepath per document.
    doc_fp = [None] * n
    rec_of_entry = dest // vpd
    for i in range(len(keys)):
        r = rec_of_entry[i]
        fp = filepaths[i]
        if doc_fp[r] is None:
            doc_fp[r] = fp
        elif doc_fp[r] != fp:
            die("doc_id %d has more than one filepath: %r vs %r" % (udocs[r], doc_fp[r], fp))

    nbytes = n * vpd * dim * 4
    print("allocating      : (%d, %d, %d) float32 = %.2f GiB" % (n, vpd, dim, nbytes / 2.0 ** 30))
    x = np.empty((n * vpd, dim), dtype=np.float32)
    is_dict = isinstance(obj, dict)
    for i, k in enumerate(tqdm(keys, desc="fill", mininterval=2)):
        x[dest[i]] = obj[k]["vector"]
        if release:  # drop the per-entry arrays as we go to keep peak memory low
            if is_dict:
                del obj[k]
            else:
                obj[k] = None
    if not np.all(np.isfinite(x)):
        die("input contains NaN or Inf")
    return x.reshape(n, vpd, dim), udocs, doc_fp, per_doc_zero_based


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pickle with per-vector entries")
    ap.add_argument("--outdir", required=True, help="corpus store directory to create")
    ap.add_argument("--pad", type=int, default=4096, help="record alignment in bytes (default 4096)")
    ap.add_argument("--percentile", type=float, default=99.99, help="clip percentile of |x| (default 99.99)")
    ap.add_argument("--precisions", default=",".join(Q.PRECISIONS),
                    help="comma list from %s (default: all; fp32 is always written)" % ",".join(Q.PRECISIONS))
    ap.add_argument("--vectors-per-doc", type=int, default=1031)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--calib-max-exact", type=int, default=0,
                    help="0 (default): the percentile uses every value, so c depends only on the data. "
                         "N > 0: if there are more than N values, use a random sample instead (saves memory, "
                         "but c may then differ between NumPy versions)")
    ap.add_argument("--calib-sample", type=int, default=32 << 20,
                    help="sample size when --calib-max-exact is exceeded (default 32Mi)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the calibration sample")
    ap.add_argument("--inspect-only", action="store_true", help="print the input summary and exit")
    ap.add_argument("--force", action="store_true", help="overwrite an existing corpus store")
    args = ap.parse_args(argv)

    precs = ["fp32"]  # the lossless master; everything else can be re-derived and checked from it
    for p in args.precisions.split(","):
        p = Q.canonical_precision(p.strip())
        if p not in precs:
            precs.append(p)
    precs = [p for p in Q.PRECISIONS if p in precs]  # canonical order
    if args.pad <= 0:
        die("--pad must be positive")
    if not 0 < args.percentile <= 100:
        die("--percentile must be in (0, 100]")

    t0 = time.time()
    print("loading %s (%.2f GiB)" % (args.input, os.path.getsize(args.input) / 2.0 ** 30))
    with open(args.input, "rb") as h:
        obj = pickle.load(h)
    print("loaded in %.1fs" % (time.time() - t0))

    keys, doc_ids, seq_ids, filepaths = inspect_and_index(obj, args.dim)
    if args.inspect_only:
        udocs, counts = np.unique(doc_ids, return_counts=True)
        print("documents       : %d" % len(udocs))
        print("vectors/doc hist: %s" % dict(sorted(collections.Counter(counts.tolist()).items())))
        return 0

    x, udocs, doc_fp, seq_zero_based = build_array(obj, keys, doc_ids, seq_ids, filepaths,
                                                   args.vectors_per_doc, args.dim)
    del obj, keys, doc_ids, seq_ids, filepaths
    n, vpd, dim = x.shape

    xmin, xmax = float(x.min()), float(x.max())
    low16_zero = float(np.mean((x.reshape(-1)[:: max(1, x.size // (16 << 20))].view("<u4") & 0xFFFF) == 0))
    print("value range     : %.6g .. %.6g" % (xmin, xmax))
    print("bf16-like       : %.4f of values have zero low 16 mantissa bits" % low16_zero)

    c, calib = Q.calibrate_clip(x, args.percentile, args.calib_max_exact, args.calib_sample, args.seed)
    print("clip c          : %.8g (%s, %s, %d values)" % (c, calib["clip_c_float32_hex"], calib["method"],
                                                          calib["values_used"]))
    if Q.clip_from_hex(calib["clip_c_float32_hex"]) != c:
        die("internal error: clip value does not survive the hex round trip")

    os.makedirs(args.outdir, exist_ok=True)
    existing = L.existing_store_files(args.outdir, "corpus")
    if existing:
        if not args.force:
            die("output directory already has %s ... (use --force to overwrite)" % existing[:3])
        L.remove_store_files(args.outdir, "corpus")

    counts = np.full(n, vpd, dtype=np.int64)
    print("=== Writing %s to %s (pad %d) ===" % (",".join(precs), args.outdir, args.pad))
    pinfo = collections.OrderedDict()
    for p in precs:
        pinfo[p] = L.write_precision(args.outdir, "emb", p, x, counts, vpd, dim, args.pad, c,
                                     progress=lambda it, desc: tqdm(it, desc=desc, mininterval=2))

    meta = L.base_metadata("corpus", n, vpd, dim, args.pad)
    meta["num_docs"] = int(n)             # same as num_records
    meta["vectors_per_doc"] = int(vpd)    # same as slots_per_record
    meta["source"] = {"path": os.path.abspath(args.input), "bytes": os.path.getsize(args.input),
                      "entries": int(n * vpd), "seq_id_per_document_0_based": seq_zero_based,
                      "fraction_bf16_like_sampled": low16_zero}
    meta["value_range_fp32"] = [xmin, xmax]
    meta["quantization"] = calib
    meta["quantization"]["clip_source"] = "computed from this corpus"
    meta["quantization"]["int8_qmax"] = Q.INT8_QMAX
    meta["quantization"]["int4_qmax"] = Q.INT4_QMAX
    meta["quantization"]["zero_point"] = 0
    meta["packing"] = Q.PACKING
    meta["precisions"] = pinfo
    meta["sha256"] = {}
    for p in pinfo.values():
        meta["sha256"].update(p["sha256"])
    meta["corpus_id"] = pinfo["fp32"]["sha256"][pinfo["fp32"]["data_file"]]
    meta["doc_order"] = "records sorted by ascending doc_id"
    meta["docs"] = [{"record": i, "doc_id": int(d), "filepath": doc_fp[i]} for i, d in enumerate(udocs)]
    L.write_metadata(args.outdir, "corpus", meta)  # written last: its presence marks a complete store
    print("wrote metadata.json; total %.1fs" % (time.time() - t0))
    print("next: python verify.py --corpus %s" % args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
