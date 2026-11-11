"""File layout of the flat multi-vector embedding format (version 2).

THIS IS THE ONLY PLACE where records are laid out in files. The corpus writer
(convert.py) and the query writer (queries.py) both call `write_precision()`
below, so a corpus directory and a query directory have exactly the same
structure and can be read by the same code.

A "store" is one directory holding, for every precision P:

    <prefix>_P.bin           N fixed-size records, no header
    <prefix>_P.offsets.u64   N+1 x uint64 LE, offsets[i] = i * record_bytes
    <prefix>_P.counts.u32    N   x uint32 LE, counts[i] = real vectors in record i

plus one metadata JSON file.

    kind      prefix  metadata file          a record is     slots_per_record
    corpus    emb     metadata.json          one page        vectors_per_doc (1031)
    queries   qry     query_metadata.json    one question    max_tokens

Record i = counts[i] vectors (row-major), then zero bytes up to
record_bytes = round_up(slots_per_record * bytes_per_vector, pad_bytes).
"""

import hashlib
import json
import os
import time

import numpy as np

try:  # imported as part of the package (from the RAG pipeline) ...
    from . import quantize as Q
except ImportError:  # ... or run as a plain script from this directory
    import quantize as Q

FORMAT = "flat-multivector-embeddings"
FORMAT_VERSION = 2

KINDS = {
    "corpus": {"prefix": "emb", "metadata": "metadata.json"},
    "queries": {"prefix": "qry", "metadata": "query_metadata.json"},
}


def round_up(n, a):
    """Smallest multiple of a that is >= n."""
    return (n + a - 1) // a * a


def record_bytes(slots, bytes_per_vector, pad):
    return round_up(slots * bytes_per_vector, pad)


def file_names(prefix, precision):
    """(data file, offsets file, counts file) of one precision."""
    p = Q.canonical_precision(precision)
    return ("%s_%s.bin" % (prefix, p), "%s_%s.offsets.u64" % (prefix, p), "%s_%s.counts.u32" % (prefix, p))


def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _replace_file(tmp, path):
    os.replace(tmp, path)


def _write_table(path, arr):
    tmp = path + ".tmp"
    arr.tofile(tmp)
    _replace_file(tmp, path)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def write_precision(outdir, prefix, precision, records, counts, slots, dim, pad, c, progress=None):
    """Write one precision of a store and return its metadata block.

    records : sequence with records[i] = float32 array (counts[i], dim).
              A (N, V, dim) ndarray or a list of 2-D arrays both work.
    counts  : real vectors per record, every one in 1..slots.
    slots   : vector slots per record; fixes the record size.
    c       : clip value (needed for int8 / int4, ignored otherwise).

    Every value goes through quantize.encode(); nothing else produces bytes.
    """
    p = Q.canonical_precision(precision)
    n = len(counts)
    counts = np.asarray(counts, dtype=np.int64)
    if n == 0:
        raise ValueError("no records to write")
    if len(records) != n:
        raise ValueError("%d records but %d counts" % (len(records), n))
    if counts.min() < 1 or counts.max() > slots:
        raise ValueError("counts must be in 1..%d, found %d..%d" % (slots, counts.min(), counts.max()))
    if counts.max() >= 2 ** 32:
        raise ValueError("count does not fit uint32")
    bpv = Q.bytes_per_vector(p, dim)
    rec = record_bytes(slots, bpv, pad)
    data_name, off_name, cnt_name = file_names(prefix, p)
    path = os.path.join(outdir, data_name)
    tmp = path + ".tmp"

    h = hashlib.sha256()
    total = 0        # values written
    exact = 0        # values whose stored form decodes back to exactly the fp32 value
    clipped = 0      # values with |x| > c (int8 / int4 only)
    max_abs_err = 0.0
    t0 = time.time()
    it = range(n) if progress is None else progress(range(n), "write %s_%s" % (prefix, p))
    with open(tmp, "wb") as f:
        for i in it:
            x = np.ascontiguousarray(records[i], dtype=np.float32)
            k = int(counts[i])
            if x.shape != (k, dim):
                raise ValueError("record %d has shape %s, expected (%d, %d)" % (i, x.shape, k, dim))
            if not np.all(np.isfinite(x)):
                raise ValueError("record %d contains NaN or Inf" % i)
            enc = Q.encode(p, x, c)  # (k, bpv) uint8, exactly the stored bytes
            if enc.shape != (k, bpv):
                raise AssertionError("encode returned %s, expected (%d, %d)" % (enc.shape, k, bpv))
            buf = np.zeros(rec, dtype=np.uint8)
            buf[:k * bpv] = enc.reshape(-1)
            f.write(buf)
            h.update(buf)

            total += x.size
            if p in ("fp16", "bf16"):
                back = np.asarray(Q.decode(p, enc, dim), dtype=np.float32)
                if not np.all(np.isfinite(back)):
                    raise ValueError("record %d overflows %s" % (i, p))
                err = np.abs(back - x)
                exact += int(np.count_nonzero(err == 0))
                max_abs_err = max(max_abs_err, float(err.max()))
            elif Q.needs_clip(p):
                clipped += int(np.count_nonzero(np.abs(x) > np.float32(c)))
    dt = time.time() - t0
    size = os.path.getsize(tmp)
    if size != n * rec:
        os.remove(tmp)
        raise IOError("%s: wrote %d bytes, expected %d" % (data_name, size, n * rec))
    _replace_file(tmp, path)

    offsets = (np.arange(n + 1, dtype=np.uint64) * np.uint64(rec)).astype("<u8")
    _write_table(os.path.join(outdir, off_name), offsets)
    _write_table(os.path.join(outdir, cnt_name), counts.astype("<u4"))

    same = bool(np.all(counts == counts[0]))
    info = {
        "data_file": data_name,
        "offsets_file": off_name,
        "counts_file": cnt_name,
        "element_dtype": Q.ELEMENT_DTYPE[p],
        "numpy_dtype": Q.NUMPY_DTYPE[p],
        "bits_per_value": Q.BITS_PER_VALUE[p],
        "bytes_per_vector": bpv,
        "slots_per_record": int(slots),
        "payload_bytes_per_record": int(counts[0]) * bpv if same else None,
        "record_bytes": int(rec),
        "file_bytes": int(size),
        "sha256": {data_name: h.hexdigest(),
                   off_name: sha256_file(os.path.join(outdir, off_name)),
                   cnt_name: sha256_file(os.path.join(outdir, cnt_name))},
    }
    if p in ("fp16", "bf16"):
        info["fraction_exact_vs_fp32"] = exact / float(total)
        info["max_abs_error_vs_fp32"] = max_abs_err
    elif Q.needs_clip(p):
        info["fraction_clipped"] = clipped / float(total)
    print("  %-5s bytes/vector=%4d record=%8d file=%14d B  %.1f MB/s"
          % (p, bpv, rec, size, size / max(dt, 1e-9) / 1e6))
    return info


def base_metadata(kind, num_records, slots, dim, pad):
    """Keys shared by the corpus and the query metadata file."""
    if kind not in KINDS:
        raise ValueError("kind must be one of %s" % sorted(KINDS))
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "file_prefix": KINDS[kind]["prefix"],
        "created_utc": utc_now(),
        "numpy_version": np.__version__,
        "num_records": int(num_records),
        "slots_per_record": int(slots),
        "dim": int(dim),
        "pad_bytes": int(pad),
    }


def write_metadata(outdir, kind, meta):
    path = os.path.join(outdir, KINDS[kind]["metadata"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    _replace_file(tmp, path)
    return path


def remove_store_files(outdir, kind):
    """Delete the metadata file first (so a half-written directory is never
    mistaken for a finished one), then every data / table file of this kind."""
    k = KINDS[kind]
    mpath = os.path.join(outdir, k["metadata"])
    if os.path.exists(mpath):
        os.remove(mpath)
    for name in os.listdir(outdir):
        if name.startswith(k["prefix"] + "_") and name.split(".")[-1] in ("bin", "u64", "u32", "tmp"):
            os.remove(os.path.join(outdir, name))


def existing_store_files(outdir, kind):
    k = KINDS[kind]
    if not os.path.isdir(outdir):
        return []
    return sorted(n for n in os.listdir(outdir) if n.startswith(k["prefix"] + "_") or n == k["metadata"])


# --------------------------------------------------------------------------
# Reading metadata
# --------------------------------------------------------------------------
def load_metadata(outdir, kind=None):
    """Return (kind, metadata dict). kind=None auto-detects; it must be given
    if the directory holds both a corpus and a query store."""
    found = [k for k in KINDS if os.path.exists(os.path.join(outdir, KINDS[k]["metadata"]))]
    if kind is None:
        if not found:
            raise IOError("%s holds neither %s nor %s" % (outdir, KINDS["corpus"]["metadata"],
                                                         KINDS["queries"]["metadata"]))
        if len(found) > 1:
            raise ValueError("%s holds both a corpus and a query store; pass kind='corpus' or 'queries'" % outdir)
        kind = found[0]
    elif kind not in found:
        raise IOError("%s has no %s" % (outdir, KINDS[kind]["metadata"]))
    with open(os.path.join(outdir, KINDS[kind]["metadata"]), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format") != FORMAT:
        raise ValueError("%s: not a %s store" % (outdir, FORMAT))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError("%s: format_version %r, this code reads version %d. Regenerate the files with "
                         "convert.py / the query pipeline." % (outdir, meta.get("format_version"), FORMAT_VERSION))
    if meta.get("kind") != kind:
        raise ValueError("%s: metadata says kind %r, expected %r" % (outdir, meta.get("kind"), kind))
    return kind, meta


def corpus_identity(corpus_meta):
    """What a query store records about the corpus it was encoded for."""
    fp32 = corpus_meta["precisions"]["fp32"]
    return {
        "corpus_id": fp32["sha256"][fp32["data_file"]],  # SHA-256 of emb_fp32.bin
        "num_docs": corpus_meta["num_records"],
        "dim": corpus_meta["dim"],
        "created_utc": corpus_meta.get("created_utc"),
        "clip_c_float32_hex": corpus_meta["quantization"]["clip_c_float32_hex"],
    }
