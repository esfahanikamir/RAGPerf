#!/usr/bin/env python3
"""Convert a pickle of per-vector fp32 embeddings into flat, headerless,
4 KiB-aligned binary files at several precisions. See README.md.

    python convert.py --input embeddings.pkl --outdir out/ \
        [--pad 4096] [--percentile 99.99] [--precisions fp32,fp16,int8,int4,bin]
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

import quantize as Q

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kw):
        return it

FORMAT_VERSION = 1


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def round_up(n, a):
    """
    Total amount of bytes after blocking and padding
    """
    return (n + a - 1) // a * a


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
# Writing
# --------------------------------------------------------------------------
def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_precision(outdir, prec, x, counts, pad, c, chunk_docs=64):
    n, vpd, dim = x.shape
    bpv = Q.bytes_per_vector(prec, dim)
    payload = counts.astype(np.uint64) * np.uint64(bpv)
    padded = (payload + np.uint64(pad - 1)) // np.uint64(pad) * np.uint64(pad)
    offsets = np.zeros(n + 1, dtype="<u8")
    offsets[1:] = np.cumsum(padded)

    data_name = Q.file_name(prec)
    path = os.path.join(outdir, data_name)
    h = hashlib.sha256()
    t0 = time.time()
    with open(path, "wb") as f:
        for a in tqdm(range(0, n, chunk_docs), desc="write " + prec, mininterval=2):
            b = min(n, a + chunk_docs)
            enc = Q.encode(prec, x[a:b], c)  # (b-a, vpd, bpv) uint8
            rec = int(padded[a])
            if not np.all(padded[a:b] == padded[a]):  # variable-size records: write one by one
                for i in range(a, b):
                    buf = np.zeros(int(padded[i]), dtype=np.uint8)
                    buf[:int(payload[i])] = enc[i - a].reshape(-1)
                    f.write(buf)
                    h.update(buf)
                continue
            buf = np.zeros((b - a, rec), dtype=np.uint8)
            buf[:, :int(payload[a])] = enc.reshape(b - a, -1)
            f.write(buf)
            h.update(buf)
    dt = time.time() - t0
    size = os.path.getsize(path)
    if size != int(offsets[-1]):
        die("%s: size %d != offsets[-1] %d" % (data_name, size, offsets[-1]))

    off_name = "emb_%s.offsets.u64" % prec
    cnt_name = "emb_%s.counts.u32" % prec
    offsets.tofile(os.path.join(outdir, off_name))
    counts.astype("<u4").tofile(os.path.join(outdir, cnt_name))

    rec_sizes = sorted(set(int(v) for v in padded))
    info = {
        "data_file": data_name,
        "offsets_file": off_name,
        "counts_file": cnt_name,
        "element_dtype": {"fp32": "float32 LE (IEEE binary32)", "fp16": "float16 LE (IEEE binary16)",
                          "int8": "int8 (two's complement)", "int4": "int4 two's complement, 2 per byte",
                          "bin": "1 bit, 8 per byte"}[prec],
        "numpy_dtype": {"fp32": "<f4", "fp16": "<f2", "int8": "i1", "int4": "u1 (packed)", "bin": "u1 (packed)"}[prec],
        "bits_per_value": {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "bin": 1}[prec],
        "bytes_per_vector": bpv,
        "payload_bytes_per_record": int(payload[0]) if len(set(payload.tolist())) == 1 else None,
        "record_bytes": rec_sizes[0] if len(rec_sizes) == 1 else None,
        "file_bytes": size,
        "sha256": {data_name: h.hexdigest(),
                   off_name: sha256_file(os.path.join(outdir, off_name)),
                   cnt_name: sha256_file(os.path.join(outdir, cnt_name))},
    }
    print("  %-5s bpv=%4d record=%7s file=%14d B  %.1f MB/s"
          % (prec, bpv, info["record_bytes"], size, size / max(dt, 1e-9) / 1e6))
    return info


PACKING = {
    "byte_order": "little-endian for every multi-byte value (fp32, fp16, offsets u64, counts u32)",
    "record_layout": "record = counts[i] vectors, row-major (vector-major, dimension fastest), "
                     "vectors in ascending seq_id order, followed by zero bytes up to the next multiple of pad_bytes",
    "fp32": "x stored as IEEE-754 binary32",
    "fp16": "x.astype('<f2'): IEEE-754 binary16, round-to-nearest-even, overflow -> inf",
    "int8": "q = clip(rint((float32(x) / c) * 127), -127, 127); one int8 per dimension; dequant x ~= q * c / 127",
    "int4": "q = clip(rint((float32(x) / c) * 7), -7, 7); byte k = (q[2k] & 0xF) | ((q[2k+1] & 0xF) << 4): "
            "even dimension in LOW nibble, odd dimension in HIGH nibble, nibbles two's complement; dequant x ~= q * c / 7",
    "bin": "bit = (x > 0); np.packbits(bits, bitorder='little'): bit j (LSB=0) of byte b is dimension 8*b + j; "
           "unused tail bits are 0",
    "rounding": "np.rint = round half to even, applied to the float32 value (x / c) * qmax",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pickle with per-vector entries")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pad", type=int, default=4096, help="record alignment in bytes (default 4096)")
    ap.add_argument("--percentile", type=float, default=99.99, help="clip percentile of |x| (default 99.99)")
    ap.add_argument("--precisions", default=",".join(Q.PRECISIONS),
                    help="comma list from fp32,fp16,int8,int4,bin (default: all)")
    ap.add_argument("--vectors-per-doc", type=int, default=1031)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--calib-max-exact", type=int, default=64 << 20,
                    help="use all values for the percentile if there are at most this many (default 64Mi)")
    ap.add_argument("--calib-sample", type=int, default=32 << 20, help="sample size otherwise (default 32Mi)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the calibration sample")
    ap.add_argument("--inspect-only", action="store_true", help="print the input summary and exit")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args(argv)

    precs = []
    for p in args.precisions.split(","):
        p = Q.canonical_precision(p.strip())
        if p not in precs:
            precs.append(p)
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

    os.makedirs(args.outdir, exist_ok=True)
    existing = [f for f in os.listdir(args.outdir) if f.startswith("emb_") or f == "metadata.json"]
    if existing and not args.force:
        die("output directory already has %s ... (use --force to overwrite)" % existing[:3])

    counts = np.full(n, vpd, dtype="<u4")
    print("=== Writing %s to %s (pad %d) ===" % (",".join(precs), args.outdir, args.pad))
    pinfo = collections.OrderedDict()
    for p in precs:
        pinfo[p] = write_precision(args.outdir, p, x, counts, args.pad, c)

    meta = collections.OrderedDict()
    meta["format"] = "flat-multivector-embeddings"
    meta["format_version"] = FORMAT_VERSION
    meta["created_utc"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    meta["source"] = {"path": os.path.abspath(args.input), "bytes": os.path.getsize(args.input),
                      "entries": int(n * vpd), "seq_id_per_document_0_based": seq_zero_based}
    meta["numpy_version"] = np.__version__
    meta["num_docs"] = int(n)
    meta["vectors_per_doc"] = int(vpd)
    meta["dim"] = int(dim)
    meta["pad_bytes"] = int(args.pad)
    meta["value_range_fp32"] = [xmin, xmax]
    meta["quantization"] = calib
    meta["quantization"]["int8_qmax"] = Q.INT8_QMAX
    meta["quantization"]["int4_qmax"] = Q.INT4_QMAX
    meta["quantization"]["zero_point"] = 0
    meta["packing"] = PACKING
    meta["precisions"] = pinfo
    meta["sha256"] = {}
    for p in pinfo.values():
        meta["sha256"].update(p["sha256"])
    meta["doc_order"] = "records sorted by ascending doc_id"
    meta["docs"] = [{"record": i, "doc_id": int(d), "filepath": doc_fp[i]} for i, d in enumerate(udocs)]
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote metadata.json; total %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
