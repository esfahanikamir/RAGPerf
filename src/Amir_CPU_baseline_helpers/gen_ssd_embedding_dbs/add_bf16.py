#!/usr/bin/env python3
"""Add a bf16 precision to an already-converted output directory.

Reads emb_fp32.bin (+ its offsets/counts) written by convert.py, rounds every
value to bfloat16 and writes emb_bf16.bin, emb_bf16.offsets.u64,
emb_bf16.counts.u32 in the same flat, headerless, pad-aligned layout. Then
adds a "bf16" entry to metadata.json (precisions, packing, sha256). No other
file is touched, and the original pickle is not needed.

    python add_bf16.py --outdir OUT [--force] [--chunk-docs 64]

bf16 encoding: the upper 16 bits of the IEEE-754 binary32 value after
round-to-nearest-even on the lower 16 bits, stored as uint16 little-endian:

    u   = float32_bits(x)
    u16 = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    x  ~= float32_from_bits(u16 << 16)

Read back in NumPy:
    h = raw.view('<u2'); x = (h.astype('<u4') << 16).view('<f4')

Note: quantize.py / reader.py only know fp32,fp16,int8,int4,bin; use
`read_bf16()` below (or the two lines above) to read the bf16 file.
"""

import argparse
import collections
import hashlib
import json
import os
import sys
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kw):
        return it

PREC = "bf16"
DATA_NAME = "emb_bf16.bin"
OFF_NAME = "emb_bf16.offsets.u64"
CNT_NAME = "emb_bf16.counts.u32"


def die(msg):
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(1)


def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fp32_to_bf16_bits(x):
    """float32 array -> uint16 bf16 bit patterns (round to nearest, ties to even).
    Finite inputs only (convert.py already rejected NaN/Inf); overflow -> +-inf."""
    u = np.ascontiguousarray(x, dtype="<f4").view("<u4").astype(np.uint64)
    u = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    return u.astype("<u2")


def bf16_bits_to_fp32(h):
    """uint16 bf16 bit patterns -> float32 (exact)."""
    return (np.asarray(h, dtype="<u2").astype("<u4") << np.uint32(16)).view("<f4")


def read_bf16(outdir, idx):
    """Record `idx` of emb_bf16.bin as a (count, dim) float32 array."""
    with open(os.path.join(outdir, "metadata.json")) as f:
        meta = json.load(f)
    pm = meta["precisions"][PREC]
    dim = int(meta["dim"])
    off = np.fromfile(os.path.join(outdir, pm["offsets_file"]), dtype="<u8")
    cnt = np.fromfile(os.path.join(outdir, pm["counts_file"]), dtype="<u4")
    data = np.memmap(os.path.join(outdir, pm["data_file"]), dtype=np.uint8, mode="r")
    start, n = int(off[idx]), int(cnt[idx])
    raw = np.array(data[start:start + n * dim * 2])
    return bf16_bits_to_fp32(raw.view("<u2").reshape(n, dim))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True, help="directory written by convert.py (must contain fp32)")
    ap.add_argument("--chunk-docs", type=int, default=64, help="records converted per batch (default 64)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing bf16 output")
    args = ap.parse_args(argv)

    t0 = time.time()
    meta_path = os.path.join(args.outdir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f, object_pairs_hook=collections.OrderedDict)

    if "fp32" not in meta.get("precisions", {}):
        die("metadata.json has no fp32 precision; bf16 is derived from emb_fp32.bin")
    if PREC in meta["precisions"] and not args.force:
        die("metadata.json already has a bf16 entry (use --force to overwrite)")
    for name in (DATA_NAME, OFF_NAME, CNT_NAME):
        if os.path.exists(os.path.join(args.outdir, name)) and not args.force:
            die("%s already exists (use --force to overwrite)" % name)

    n = int(meta["num_docs"])
    dim = int(meta["dim"])
    pad = int(meta["pad_bytes"])
    src = meta["precisions"]["fp32"]

    src_off = np.fromfile(os.path.join(args.outdir, src["offsets_file"]), dtype="<u8")
    counts = np.fromfile(os.path.join(args.outdir, src["counts_file"]), dtype="<u4")
    if len(src_off) != n + 1 or len(counts) != n:
        die("fp32 offsets/counts length (%d, %d) does not match num_docs %d" % (len(src_off), len(counts), n))
    src_path = os.path.join(args.outdir, src["data_file"])
    if os.path.getsize(src_path) != int(src_off[-1]):
        die("%s size %d != offsets[-1] %d" % (src["data_file"], os.path.getsize(src_path), src_off[-1]))
    src_bpv = 4 * dim
    data = np.memmap(src_path, dtype=np.uint8, mode="r")

    bpv = 2 * dim
    payload = counts.astype(np.uint64) * np.uint64(bpv)
    padded = (payload + np.uint64(pad - 1)) // np.uint64(pad) * np.uint64(pad)
    offsets = np.zeros(n + 1, dtype="<u8")
    offsets[1:] = np.cumsum(padded)

    print("source          : %s (%d records, dim %d, pad %d)" % (src_path, n, dim, pad))
    print("output          : %s (%d bytes expected)" % (os.path.join(args.outdir, DATA_NAME), int(offsets[-1])))

    out_path = os.path.join(args.outdir, DATA_NAME)
    tmp_path = out_path + ".tmp"
    h = hashlib.sha256()
    total = 0
    exact = 0
    max_abs_err = 0.0
    max_rel_err = 0.0
    t1 = time.time()
    with open(tmp_path, "wb") as f:
        for a in tqdm(range(0, n, args.chunk_docs), desc="write bf16", mininterval=2):
            b = min(n, a + args.chunk_docs)
            for i in range(a, b):
                s = int(src_off[i])
                c = int(counts[i])
                x = np.array(data[s:s + c * src_bpv]).view("<f4").reshape(c, dim)
                if not np.all(np.isfinite(x)):
                    die("record %d of emb_fp32.bin contains NaN or Inf" % i)
                hb = fp32_to_bf16_bits(x)
                back = bf16_bits_to_fp32(hb)
                if not np.all(np.isfinite(back)):
                    die("record %d overflows bf16" % i)
                err = np.abs(back - x)
                exact += int(np.count_nonzero(err == 0))
                total += x.size
                max_abs_err = max(max_abs_err, float(err.max()))
                nz = x != 0
                if np.any(nz):
                    max_rel_err = max(max_rel_err, float((err[nz] / np.abs(x[nz])).max()))
                buf = np.zeros(int(padded[i]), dtype=np.uint8)
                buf[:int(payload[i])] = hb.view(np.uint8).reshape(-1)
                f.write(buf)
                h.update(buf)
    dt = time.time() - t1
    size = os.path.getsize(tmp_path)
    if size != int(offsets[-1]):
        os.remove(tmp_path)
        die("%s: size %d != offsets[-1] %d" % (DATA_NAME, size, offsets[-1]))
    os.replace(tmp_path, out_path)
    offsets.tofile(os.path.join(args.outdir, OFF_NAME))
    counts.astype("<u4").tofile(os.path.join(args.outdir, CNT_NAME))

    rec_sizes = sorted(set(int(v) for v in padded))
    info = collections.OrderedDict()
    info["data_file"] = DATA_NAME
    info["offsets_file"] = OFF_NAME
    info["counts_file"] = CNT_NAME
    info["element_dtype"] = "bfloat16 LE (upper 16 bits of IEEE binary32)"
    info["numpy_dtype"] = "<u2 (bf16 bit pattern; x = (u.astype('<u4') << 16).view('<f4'))"
    info["bits_per_value"] = 16
    info["bytes_per_vector"] = bpv
    info["payload_bytes_per_record"] = int(payload[0]) if len(set(payload.tolist())) == 1 else None
    info["record_bytes"] = rec_sizes[0] if len(rec_sizes) == 1 else None
    info["file_bytes"] = size
    info["sha256"] = {DATA_NAME: h.hexdigest(),
                      OFF_NAME: sha256_file(os.path.join(args.outdir, OFF_NAME)),
                      CNT_NAME: sha256_file(os.path.join(args.outdir, CNT_NAME))}
    info["derived_from"] = src["data_file"]
    info["fraction_exact_vs_fp32"] = exact / float(total)
    info["max_abs_error_vs_fp32"] = max_abs_err
    info["max_rel_error_vs_fp32"] = max_rel_err

    print("  %-5s bpv=%4d record=%7s file=%14d B  %.1f MB/s"
          % (PREC, bpv, info["record_bytes"], size, size / max(dt, 1e-9) / 1e6))
    print("exact vs fp32   : %.6f of values (max abs err %.3g, max rel err %.3g)"
          % (info["fraction_exact_vs_fp32"], max_abs_err, max_rel_err))

    # Insert bf16 right after fp16 so the precision order reads naturally.
    new_prec = collections.OrderedDict()
    for k, v in meta["precisions"].items():
        if k == PREC:
            continue
        new_prec[k] = v
        if k == "fp16":
            new_prec[PREC] = info
    if PREC not in new_prec:
        new_prec[PREC] = info
    meta["precisions"] = new_prec
    meta.setdefault("packing", collections.OrderedDict())
    meta["packing"]["bf16"] = ("u = bits(float32(x)); u16 = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 "
                               "(round to nearest even); stored as uint16 LE; x ~= float_from_bits(u16 << 16)")
    meta.setdefault("sha256", collections.OrderedDict())
    meta["sha256"].update(info["sha256"])

    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp_meta, meta_path)
    print("updated metadata.json; total %.1fs" % (time.time() - t0))

    # Spot-check: first and last record read back through the offset table.
    for i in sorted({0, n - 1}):
        s = int(src_off[i])
        c = int(counts[i])
        x = np.array(data[s:s + c * src_bpv]).view("<f4").reshape(c, dim)
        if not np.array_equal(read_bf16(args.outdir, i), bf16_bits_to_fp32(fp32_to_bf16_bits(x))):
            die("read-back mismatch at record %d" % i)
    print("read-back check : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
