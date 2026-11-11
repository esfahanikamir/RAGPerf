#!/usr/bin/env python3
"""Verify a corpus store and, optionally, the query store built for it.

    python verify.py --corpus CORPUS_DB [--queries QUERY_DB]
                     [--docs 0] [--spot 2000] [--max-queries 64] [--k 1,5,10]
                     [--save-topk topk.json] [--skip-hash] [--skip-retrieval]

For each store:
  1. structure : metadata, file sizes, offset/count tables, alignment, zero padding, SHA-256
  2. content   : EVERY record of every precision is byte-identical to encode(fp32 record, c)
                 (--docs K checks K random records instead), value ranges, exactness of fp16/bf16
  3. spot      : random single values read with plain seek/read at the documented byte address
                 and decoded with scalar arithmetic (independent of the NumPy reader)
Once:
  4. unittests : bit layout of int4 / binary / bf16 on hand-written vectors
  5. cross     : the query store was built for THIS corpus (same c, dim, pad, corpus id, precisions)
  6. retrieval : MaxSim top-k of every precision against fp32 (real queries if given)

Exit code is non-zero if any hard check fails.
"""

import argparse
import json
import os
import struct
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

FAILS = []


def check(cond, msg):
    cond = bool(cond)
    print("  [%s] %s" % ("ok" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)
    return cond


# --------------------------------------------------------------------------
# 1. structure
# --------------------------------------------------------------------------
def check_structure(st, skip_hash):
    m = st.meta
    n, slots, dim, pad = st.num_records, st.slots, st.dim, st.pad
    tag = st.kind
    print("\n=== 1. Structure of the %s store %s (records=%d, slots/record=%d, dim=%d, pad=%d) ==="
          % (tag, st.outdir, n, slots, dim, pad))
    check(m["file_prefix"] == L.KINDS[tag]["prefix"], "%s: file prefix is %r" % (tag, m["file_prefix"]))
    check("fp32" in m["precisions"], "%s: fp32 (the lossless master) is present" % tag)
    missing = [p for p in Q.PRECISIONS if p not in m["precisions"]]
    check(not missing, "%s: all precisions present (%s)%s"
          % (tag, ",".join(Q.PRECISIONS), "; missing: %s" % missing if missing else ""))
    print("  %-5s %6s %10s %16s" % ("prec", "B/vec", "record", "file bytes"))
    ref_counts = None
    for p, pm in m["precisions"].items():
        try:
            data, offsets, counts, bpv = st._open(p)
        except Exception as e:  # the reader refuses inconsistent files
            check(False, "%s %s: cannot open: %s" % (tag, p, e))
            continue
        rec = int(pm["record_bytes"])
        size = os.path.getsize(os.path.join(st.outdir, pm["data_file"]))
        print("  %-5s %6d %10d %16d" % (p, bpv, rec, size))
        names = L.file_names(m["file_prefix"], p)
        check((pm["data_file"], pm["offsets_file"], pm["counts_file"]) == names, "%s %s file names %s" % (tag, p, names[0]))
        check(bpv == Q.bytes_per_vector(p, dim), "%s %s bytes/vector = %d" % (tag, p, bpv))
        check(rec == L.round_up(slots * bpv, pad) and rec % pad == 0,
              "%s %s record_bytes = round_up(%d * %d, %d) = %d" % (tag, p, slots, bpv, pad, rec))
        check(size == n * rec == pm["file_bytes"], "%s %s file size == records * record_bytes == %d" % (tag, p, n * rec))
        check(len(offsets) == n + 1 and np.array_equal(offsets, np.arange(n + 1, dtype=np.uint64) * np.uint64(rec)),
              "%s %s offsets[i] == i * record_bytes, offsets[N] == file size" % (tag, p))
        check(len(counts) == n and counts.min() >= 1 and counts.max() <= slots,
              "%s %s counts in 1..%d (found %d..%d)" % (tag, p, slots, counts.min(), counts.max()))
        if ref_counts is None:
            ref_counts = np.array(counts)
        else:
            check(np.array_equal(counts, ref_counts), "%s %s counts identical to the other precisions" % (tag, p))
        arr = np.asarray(data).reshape(n, rec)
        if np.all(counts == counts[0]):
            nz = int(np.count_nonzero(arr[:, int(counts[0]) * bpv:]))
        else:
            nz = sum(int(np.count_nonzero(arr[i, int(counts[i]) * bpv:])) for i in range(n))
        check(nz == 0, "%s %s padding bytes all zero" % (tag, p))
        for fname, want in pm["sha256"].items():
            check(m["sha256"].get(fname) == want, "%s metadata lists the same sha256 twice for %s" % (tag, fname))
            if not skip_hash:
                check(L.sha256_file(os.path.join(st.outdir, fname)) == want, "%s sha256 %s" % (tag, fname))

    if ref_counts is None:
        return
    if tag == "corpus":
        docs = m["docs"]
        check(m["num_docs"] == n and m["vectors_per_doc"] == slots, "corpus: num_docs / vectors_per_doc aliases agree")
        check(np.all(ref_counts == slots), "corpus: every record holds exactly %d vectors" % slots)
        check(len(docs) == n and all(d["record"] == i for i, d in enumerate(docs)), "corpus: docs[] lists records 0..N-1")
        ids = [d["doc_id"] for d in docs]
        check(all(a < b for a, b in zip(ids, ids[1:])), "corpus: doc_id strictly ascending with the record number")
        check(all(d.get("filepath") for d in docs), "corpus: every record has a filepath")
        fp32 = m["precisions"].get("fp32")
        if fp32:
            check(m.get("corpus_id") == fp32["sha256"][fp32["data_file"]], "corpus: corpus_id == sha256 of emb_fp32.bin")
        print("  record == doc_id for every record: %s" % all(d["record"] == d["doc_id"] for d in docs))
    else:
        rows = st.rows()
        check(m["num_queries"] == n and m["max_tokens"] == slots, "queries: num_queries / max_tokens aliases agree")
        check(len(rows) == n and all(r["record"] == i for i, r in enumerate(rows)), "queries: queries.jsonl lists records 0..N-1")
        check(len(rows) == n and all(r["n_tokens"] == int(ref_counts[i]) for i, r in enumerate(rows)),
              "queries: n_tokens in queries.jsonl == counts table")
        check(all(isinstance(r["question"], str) and r["question"] for r in rows), "queries: every record has a question text")
        check(m["min_tokens_observed"] == int(ref_counts.min()) and m["max_tokens_observed"] == int(ref_counts.max())
              and m["total_tokens"] == int(ref_counts.sum()), "queries: token statistics in metadata match the counts table "
              "(%d..%d, total %d)" % (ref_counts.min(), ref_counts.max(), ref_counts.sum()))
        if not skip_hash:
            check(L.sha256_file(os.path.join(st.outdir, m["queries_file"])) == m["sha256"].get(m["queries_file"]),
                  "queries sha256 %s" % m["queries_file"])


# --------------------------------------------------------------------------
# 2. content
# --------------------------------------------------------------------------
def check_content(st, ndocs, seed):
    m = st.meta
    n, dim, c, tag = st.num_records, st.dim, st.clip_c, st.kind
    precs = [p for p in m["precisions"] if p != "fp32"]
    if "fp32" not in m["precisions"]:
        print("\n=== 2. Content of the %s store: fp32 missing, skipped ===" % tag)
        return
    full = ndocs <= 0 or ndocs >= n
    idx = np.arange(n) if full else np.sort(np.random.default_rng(seed).choice(n, size=ndocs, replace=False))
    print("\n=== 2. Content of the %s store vs its fp32 master (c = %.8g = %s; %s) ==="
          % (tag, c, m["quantization"]["clip_c_float32_hex"],
             "all %d records" % n if full else "%d random records" % len(idx)))
    bad = dict((p, 0) for p in precs)
    exact = dict((p, 0) for p in precs)
    cmin = dict((p, 127) for p in precs)
    cmax = dict((p, -128) for p in precs)
    err_unclipped = dict((p, 0.0) for p in precs)
    total = clipped = nonfinite = 0
    xmin, xmax = np.inf, -np.inf
    t0 = time.time()
    for i in idx:
        i = int(i)
        x = np.array(st.get("fp32", i))
        nonfinite += int(np.count_nonzero(~np.isfinite(x)))
        total += x.size
        over = np.abs(x) > np.float32(c)
        clipped += int(np.count_nonzero(over))
        xmin, xmax = min(xmin, float(x.min())), max(xmax, float(x.max()))
        for p in precs:
            raw = np.asarray(st.raw(p, i))
            if not np.array_equal(raw, Q.encode(p, x, c)):
                bad[p] += 1
            codes = Q.decode(p, raw, dim)
            if p in ("fp16", "bf16"):
                exact[p] += int(np.count_nonzero(np.asarray(codes, dtype=np.float32) == x))
            elif p in ("int8", "int4"):
                cmin[p], cmax[p] = min(cmin[p], int(codes.min())), max(cmax[p], int(codes.max()))
                err = np.abs(Q.dequantize(p, codes, c) - x)
                if not np.all(over):
                    err_unclipped[p] = max(err_unclipped[p], float(err[~over].max()))
            elif p == "bin":
                exact[p] += int(np.count_nonzero(codes.astype(bool) == (x > 0)))
    print("  %d values in %.1fs; fp32 range %.6g .. %.6g; %.3e of values have |x| > c (clipped in int8/int4)"
          % (total, time.time() - t0, xmin, xmax, clipped / float(total)))
    check(nonfinite == 0, "%s fp32: no NaN / Inf" % tag)
    for p in precs:
        check(bad[p] == 0, "%s %s: %d / %d records byte-identical to encode(fp32 record, c)"
              % (tag, p, len(idx) - bad[p], len(idx)))
        pm = m["precisions"][p]
        if p in ("fp16", "bf16"):
            frac = exact[p] / float(total)
            print("        %s: %.8f of values equal their fp32 value exactly (%d not exact)" % (p, frac, total - exact[p]))
            if full:
                check(abs(frac - pm["fraction_exact_vs_fp32"]) < 1e-12, "%s %s: exact fraction matches metadata" % (tag, p))
        elif p in ("int8", "int4"):
            qmax = Q.INT8_QMAX if p == "int8" else Q.INT4_QMAX
            check(-qmax <= cmin[p] and cmax[p] <= qmax, "%s %s: codes in [-%d, %d] (found %d..%d)" % (tag, p, qmax, qmax, cmin[p], cmax[p]))
            bound = c / qmax / 2
            check(err_unclipped[p] <= bound * (1 + 1e-5), "%s %s: unclipped error %.3e <= c/(2*%d) = %.3e"
                  % (tag, p, err_unclipped[p], qmax, bound))
            if full:
                check(abs(clipped / float(total) - pm["fraction_clipped"]) < 1e-12, "%s %s: clipped fraction matches metadata" % (tag, p))
        elif p == "bin":
            check(exact[p] == total, "%s bin: every bit == (x > 0)" % tag)


# --------------------------------------------------------------------------
# 3. independent spot check: plain file I/O + scalar arithmetic at the documented address
# --------------------------------------------------------------------------
def _read_at(f, off, nbytes):
    f.seek(off)
    b = f.read(nbytes)
    if len(b) != nbytes:
        raise IOError("short read at %d" % off)
    return b


def _scalar_code(xf, c, qmax):
    """q = clip(round_half_even((x / c) * qmax)) with float32 scalars and Python's round()."""
    t = np.float32(np.float32(xf) / np.float32(c)) * np.float32(qmax)
    return int(max(-qmax, min(qmax, round(float(t)))))


def spot_check(st, nspot, seed):
    m = st.meta
    n, dim, c, tag = st.num_records, st.dim, st.clip_c, st.kind
    precs = list(m["precisions"].keys())
    if nspot <= 0 or "fp32" not in precs:
        return
    print("\n=== 3. Spot check of the %s store: %d random values per precision, seek/read + scalar decode ===" % (tag, nspot))
    files = {}
    for p in precs:
        pm = m["precisions"][p]
        files[p] = tuple(open(os.path.join(st.outdir, pm[k]), "rb") for k in ("data_file", "offsets_file", "counts_file"))
    rng = np.random.default_rng(seed + 7)
    wrong = dict((p, 0) for p in precs)
    try:
        for _ in range(nspot):
            i = int(rng.integers(n))
            addr = {}
            cnt = None
            for p in precs:
                data, off, cntf = files[p]
                addr[p] = struct.unpack("<Q", _read_at(off, 8 * i, 8))[0]       # offsets[i]: uint64 LE at byte 8*i
                k = struct.unpack("<I", _read_at(cntf, 4 * i, 4))[0]            # counts[i]: uint32 LE at byte 4*i
                cnt = k if cnt is None else cnt
                if k != cnt:
                    wrong[p] += 1
            v = int(rng.integers(cnt))
            d = int(rng.integers(dim))
            b4 = _read_at(files["fp32"][0], addr["fp32"] + v * 4 * dim + 4 * d, 4)
            xf = struct.unpack("<f", b4)[0]
            u = struct.unpack("<I", b4)[0]
            for p in precs:
                data = files[p][0]
                if p == "fp32":
                    continue
                if p == "fp16":
                    ok = _read_at(data, addr[p] + v * 2 * dim + 2 * d, 2) == struct.pack("<e", xf)
                elif p == "bf16":
                    h = struct.unpack("<H", _read_at(data, addr[p] + v * 2 * dim + 2 * d, 2))[0]
                    ok = h == ((u + 0x7FFF + ((u >> 16) & 1)) >> 16)
                elif p == "int8":
                    q = struct.unpack("<b", _read_at(data, addr[p] + v * dim + d, 1))[0]
                    ok = q == _scalar_code(xf, c, 127)
                elif p == "int4":
                    b = _read_at(data, addr[p] + v * (dim // 2) + d // 2, 1)[0]
                    nib = (b >> 4) if d % 2 else (b & 0x0F)       # even dim: low nibble, odd dim: high nibble
                    q = nib - 16 if nib >= 8 else nib             # sign-extend 4 bits
                    ok = q == _scalar_code(xf, c, 7)
                elif p == "bin":
                    b = _read_at(data, addr[p] + v * ((dim + 7) // 8) + d // 8, 1)[0]
                    ok = ((b >> (d % 8)) & 1) == (1 if xf > 0 else 0)
                if not ok:
                    wrong[p] += 1
    finally:
        for fs in files.values():
            for f in fs:
                f.close()
    for p in precs:
        if p != "fp32":
            check(wrong[p] == 0, "%s %s: %d / %d spot values correct at offsets[i] + v*bytes_per_vector + ..."
                  % (tag, p, nspot - wrong[p], nspot))


# --------------------------------------------------------------------------
# 4. unit tests of the bit layout
# --------------------------------------------------------------------------
def unit_tests():
    print("\n=== 4. Bit-layout unit tests ===")
    # int4 with c = 7 so that code = rint(x) (values chosen away from .5 ties)
    x = np.array([1, -1, 7, -7, 0, 3, -9, 2.3, 3.7, -2.3, 0.2, 0.8, 5, -3, 6, -6], dtype=np.float32)
    codes = Q.quantize_int4(x, 7.0)
    want_codes = [1, -1, 7, -7, 0, 3, -7, 2, 4, -2, 0, 1, 5, -3, 6, -6]
    check(codes.tolist() == want_codes, "int4 codes %s" % codes.tolist())
    #  dims (0,1)=(1,-1) -> lo 0x1 hi 0xF -> 0xF1 ; (7,-7) -> 0x97 ; (0,3) -> 0x30 ; (-7,2) -> 0x29
    #  (4,-2) -> 0xE4 ; (0,1) -> 0x10 ; (5,-3) -> 0xD5 ; (6,-6) -> 0xA6
    want = [0xF1, 0x97, 0x30, 0x29, 0xE4, 0x10, 0xD5, 0xA6]
    got = Q.encode("int4", x, 7.0).tolist()
    check(got == want, "int4 bytes %s == %s" % (["0x%02X" % b for b in got], ["0x%02X" % b for b in want]))
    check(Q.decode("int4", np.array(want, np.uint8), 16).tolist() == want_codes, "int4 unpack (sign-extend) round-trips")
    allc = np.arange(-8, 8, dtype=np.int8)
    check(np.array_equal(Q.unpack_int4(Q.pack_int4(np.stack([allc, allc[::-1]], -1).reshape(-1))),
                         np.stack([allc, allc[::-1]], -1).reshape(-1)), "int4 pack/unpack all 16 nibble values")

    # int8 with c = 127 -> code = rint(x) for these values; 2.5 and 3.5 are exact ties (round half to even)
    x8 = np.array([0, 1, -1, 126.6, -200, 300, 63.2, -63.7, 2.5, 3.5, -2.5, -3.5], dtype=np.float32)
    q8 = Q.quantize_int8(x8, 127.0)
    check(q8.tolist() == [0, 1, -1, 127, -127, 127, 63, -64, 2, 4, -2, -4], "int8 codes %s (ties to even)" % q8.tolist())
    check(Q.encode("int8", x8[:8], 127.0).tolist() == [0x00, 0x01, 0xFF, 0x7F, 0x81, 0x7F, 0x3F, 0xC0], "int8 bytes (two's complement)")

    # binary: dims 0..15
    xb = np.array([+1, -1, +2, +3, 0, -1, -2, +.5, -1, -1, -1, -1, +1, +1, +1, +1], dtype=np.float32)
    # bits: d0=1 d1=0 d2=1 d3=1 d4=0 (x==0 -> 0) d5=0 d6=0 d7=1 -> 0b10001101 = 0x8D ; d8..11=0, d12..15=1 -> 0xF0
    got = Q.encode("bin", xb).tolist()
    check(got == [0x8D, 0xF0], "bin bytes %s == ['0x8D', '0xF0'] (bit j of byte b = dim 8b+j)" % ["0x%02X" % b for b in got])
    check(Q.decode("bin", np.array([0x8D, 0xF0], np.uint8), 16).tolist() == (xb > 0).astype(int).tolist(), "bin unpack")
    check(Q.encode("bin", np.array([0.0, -0.0], np.float32)).tolist() == [0x00], "bin: +0.0 and -0.0 give bit 0")
    x10 = np.ones(10, np.float32)
    check(Q.encode("bin", x10).tolist() == [0xFF, 0x03], "bin dim=10 -> 2 bytes, unused tail bits zero")

    # bf16: upper 16 bits of the float32 pattern, low half rounded to nearest even
    bits = np.array([0x3E290000, 0xBBE80000, 0x3E298000, 0x3E2A8000, 0x3E298001, 0x3E297FFF, 0x00000000, 0x80000000], dtype="<u4")
    want16 = [0x3E29, 0xBBE8, 0x3E2A, 0x3E2A, 0x3E2A, 0x3E29, 0x0000, 0x8000]
    got16 = Q.quantize_bf16(bits.view("<f4")).tolist()
    check(got16 == want16, "bf16 bits %s (ties to even)" % ["0x%04X" % b for b in got16])
    check(Q.encode("bf16", bits[:2].view("<f4")).tolist() == [0x29, 0x3E, 0xE8, 0xBB], "bf16 bytes little-endian: 0x3E29 -> 29 3E")
    check(Q.dequantize_bf16(np.array(want16, "<u2")).view("<u4").tolist() == [w << 16 for w in want16], "bf16 decode = bits << 16")
    # fp16 of the same value has different bits
    check(Q.encode("fp16", np.array([0.1650390625], np.float32)).tolist() == [0x48, 0x31], "fp16 bytes of 0.1650390625: 48 31 (0x3148)")

    # clip value: hex <-> float32 is exact
    for h in ("0x3eae0000", "0x3e99999a", "0x3f800000"):
        check(Q.clip_to_hex(Q.clip_from_hex(h)) == h, "clip value %s survives hex -> float32 -> hex" % h)


# --------------------------------------------------------------------------
# 5. cross checks corpus <-> queries
# --------------------------------------------------------------------------
def cross_check(cs, qs):
    cm, qm = cs.meta, qs.meta
    print("\n=== 5. The query store was built for this corpus store ===")
    check(qs.dim == cs.dim, "same dim (%d)" % cs.dim)
    check(qs.pad == cs.pad, "same pad_bytes (%d)" % cs.pad)
    check(list(qm["precisions"].keys()) == list(cm["precisions"].keys()),
          "same precisions in the same order (%s)" % ",".join(cm["precisions"].keys()))
    ch, qh = cm["quantization"]["clip_c_float32_hex"], qm["quantization"]["clip_c_float32_hex"]
    check(ch == qh and cs.clip_c == qs.clip_c, "same clip value c: corpus %s, queries %s (= %.8g)" % (ch, qh, cs.clip_c))
    for k in ("int8_qmax", "int4_qmax", "zero_point"):
        check(cm["quantization"][k] == qm["quantization"][k], "same %s (%s)" % (k, cm["quantization"][k]))
    check(qm["corpus"]["corpus_id"] == cm["corpus_id"], "query store records this corpus_id (%s...)" % cm["corpus_id"][:16])
    check(qm["corpus"]["num_docs"] == cs.num_records, "query store records the corpus size (%d docs)" % cs.num_records)
    check(qm["packing"] == cm["packing"], "identical packing conventions recorded in both")
    for p in cm["precisions"]:
        if p in qm["precisions"]:
            check(cm["precisions"][p]["bytes_per_vector"] == qm["precisions"][p]["bytes_per_vector"],
                  "%s: same bytes/vector in both stores (%d)" % (p, cm["precisions"][p]["bytes_per_vector"]))


# --------------------------------------------------------------------------
# 6. retrieval sanity check
# --------------------------------------------------------------------------
def _as_float(p, codes):
    """Decoded codes -> float32 matrix whose dot products are the precision's scores."""
    if p == "bin":
        return Q.dequantize_binary(codes)  # +-1, dot = dim - 2*hamming
    return np.asarray(codes, dtype=np.float32)


def maxsim_all(st, p, qf, seg):
    """qf: (T, dim) float32, all query vectors back to back in the precision's code space;
    seg[j] = first row of query j. Returns (num_queries, N) MaxSim scores as float64.
    Integer precisions are exact: |dot| < 2^24, so float32 holds every partial sum exactly."""
    n = st.num_records
    out = np.empty((len(seg), n), dtype=np.float64)
    counts = np.asarray(st.counts(p), dtype=np.int64)
    budget = 256 << 20  # bytes for one score block
    a = 0
    while a < n:
        b = a + 1
        rows = int(counts[a])
        while b < n and (rows + int(counts[b])) * qf.shape[0] * 4 <= budget:
            rows += int(counts[b])
            b += 1
        d = np.concatenate([_as_float(p, st.get(p, i, binary="bits")) for i in range(a, b)], axis=0)
        with np.errstate(all="ignore"):  # macOS Accelerate emits spurious FP warnings in sgemm
            s = qf @ d.T  # (T, rows)
        if not np.all(np.isfinite(s)):
            raise FloatingPointError("non-finite MaxSim scores for %s" % p)
        start = 0
        for i in range(a, b):
            k = int(counts[i])
            best = s[:, start:start + k].max(axis=1).astype(np.float64)   # max over the REAL vectors only
            out[:, i] = np.add.reduceat(best, seg)                          # sum over each query's tokens
            start += k
        a = b
    return out


def _int_maxsim(q, d):
    """Exact integer MaxSim of code matrices (reference for the float matmul)."""
    return int((q.astype(np.int64) @ d.astype(np.int64).T).max(axis=1).sum())


def hamming_maxsim(qbits_packed, dbits_packed, dim):
    """MaxSim on packed bits with explicit popcount: sum_q max_d (dim - 2*popcount(q xor d))."""
    pc = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    ham = pc[qbits_packed[:, None, :] ^ dbits_packed[None, :, :]].sum(-1)
    return int((dim - 2 * ham).max(axis=1).sum())


def retrieval(cs, qs, nqueries, nq_vec, ks, seed, save_topk):
    m = cs.meta
    n, dim, c = cs.num_records, cs.dim, cs.clip_c
    precs = list(m["precisions"].keys())
    if "fp32" not in precs:
        print("\n=== 6. Retrieval: fp32 missing, skipped ===")
        return
    ks = [k for k in ks if k <= n]
    kmax = max(ks) if ks else 0

    if qs is not None:
        nq = qs.num_records if nqueries <= 0 else min(nqueries, qs.num_records)
        qcounts = np.asarray(qs.counts("fp32"), dtype=np.int64)[:nq]
        seg = np.concatenate([[0], np.cumsum(qcounts)[:-1]]).astype(np.int64)
        get_q = lambda p: np.concatenate([_as_float(p, qs.get(p, i, binary="bits")) for i in range(nq)], axis=0)
        src = None
        print("\n=== 6. Retrieval: first %d REAL queries of the query store (%d tokens) against all %d docs ==="
              % (nq, int(qcounts.sum()), n))
        print("  every precision scores its own query file against its own corpus file")
    else:
        # no query store: use vectors of random documents as queries, encoded with c like real queries would be
        rng = np.random.default_rng(seed + 1)
        src = np.sort(rng.choice(n, size=min(max(nqueries, 1), n), replace=False))
        nq = len(src)
        vpd = cs.slots
        nq_vec = vpd if nq_vec <= 0 else min(nq_vec, vpd)
        rows = [np.sort(rng.choice(vpd, size=nq_vec, replace=False)) for _ in src]
        qx = np.concatenate([np.asarray(cs.get("fp32", int(d)))[r] for d, r in zip(src, rows)], axis=0)
        seg = np.arange(nq, dtype=np.int64) * nq_vec
        get_q = lambda p: _as_float(p, Q.decode(p, Q.encode(p, qx, c), dim))
        print("\n=== 6. Retrieval: %d synthetic queries x %d vectors taken from docs %s%s, against all %d docs ==="
              % (nq, nq_vec, src[:8].tolist(), " ..." if nq > 8 else "", n))

    t = time.time()
    ref = maxsim_all(cs, "fp32", get_q("fp32"), seg)
    ref_rank = np.argsort(-ref, axis=1, kind="stable")
    line = "  fp32 MaxSim in %.1fs" % (time.time() - t)
    if src is not None:
        line += "; source doc ranked #1 for %d/%d queries" % (int(np.sum(ref.argmax(1) == src)), nq)
    print(line)
    dump = {"note": "top-k records per query; integer precisions (int8, int4, bin) are exact, float ones may differ "
                    "in the last digits between machines", "k": kmax, "num_queries": nq, "num_docs": n,
            "corpus_id": m.get("corpus_id"), "precisions": {}}
    dump["precisions"]["fp32"] = {"records": ref_rank[:, :kmax].tolist(),
                                  "scores": np.take_along_axis(ref, ref_rank[:, :kmax], 1).tolist()}

    print("  %-5s" % "prec" + "".join("  overlap@%-3d" % k for k in ks) + "  top1==fp32   time")
    for p in precs:
        if p == "fp32":
            continue
        t = time.time()
        qf = get_q(p)
        sc = maxsim_all(cs, p, qf, seg)
        dt = time.time() - t
        rank = np.argsort(-sc, axis=1, kind="stable")
        line = "  %-5s" % p
        for k in ks:
            ov = np.mean([len(set(ref_rank[i, :k]) & set(rank[i, :k])) / float(k) for i in range(nq)])
            line += "  %11.3f " % ov
        print(line + "  %9.3f  %5.1fs" % (np.mean(rank[:, 0] == ref_rank[:, 0]), dt))
        top = rank[:, :kmax]
        scores = np.take_along_axis(sc, top, 1)
        if p in ("int8", "int4", "bin"):
            scores = np.rint(scores).astype(np.int64)
        dump["precisions"][p] = {"records": top.tolist(), "scores": scores.tolist()}

        # cross-check the float matmul against exact integer arithmetic for query 0 and its best document
        j = int(rank[0, 0])
        q0 = slice(int(seg[0]), int(seg[1]) if len(seg) > 1 else qf.shape[0])
        if p in ("int8", "int4"):
            qi = np.rint(qf[q0]).astype(np.int64)
            check(_int_maxsim(qi, cs.get(p, j)) == int(round(sc[0, j])),
                  "%s: integer MaxSim of query 0 vs record %d == %d (exact)" % (p, j, int(round(sc[0, j]))))
        elif p == "bin":
            qp = Q.pack_binary(qf[q0] > 0)
            dp = np.asarray(cs.get("bin", j, binary="packed"))
            check(hamming_maxsim(qp, dp, dim) == int(round(sc[0, j])),
                  "bin: sum max (dim - 2*popcount(q xor d)) of query 0 vs record %d == %d (exact)" % (j, int(round(sc[0, j]))))
    if save_topk:
        with open(save_topk, "w") as f:
            json.dump(dump, f)
        print("  wrote top-%d records and scores per query and precision to %s" % (kmax, save_topk))


# --------------------------------------------------------------------------
def facts(st):
    m = st.meta
    print("  %s store %s" % (st.kind, os.path.abspath(st.outdir)))
    print("    records %d, slots/record %d, dim %d, pad %d, clip c %.9g (%s)"
          % (st.num_records, st.slots, st.dim, st.pad, st.clip_c, m["quantization"]["clip_c_float32_hex"]))
    print("    record bytes: " + ", ".join("%s %d" % (p, v["record_bytes"]) for p, v in m["precisions"].items()))
    if st.kind == "queries":
        print("    tokens per query %d..%d, total %d; encoder dtype %s"
              % (m["min_tokens_observed"], m["max_tokens_observed"], m["total_tokens"],
                 (m.get("source") or {}).get("encoder_output_dtype")))
    print("    C++ check: ./check_db check %s %s %d %d %d %s"
          % (st.outdir, m["file_prefix"], st.dim, st.slots, st.pad, m["quantization"]["clip_c_float32_hex"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", "--outdir", dest="corpus", required=True, help="corpus store (written by convert.py)")
    ap.add_argument("--queries", help="query store (written by the query pipeline / queries.py)")
    ap.add_argument("--docs", type=int, default=0, help="records for the content check; 0 = all (default)")
    ap.add_argument("--spot", type=int, default=2000, help="random values per precision for the spot check")
    ap.add_argument("--max-queries", type=int, default=64,
                    help="queries used for the retrieval check; 0 = all (default 64)")
    ap.add_argument("--query-vectors", type=int, default=32,
                    help="vectors per synthetic query when no query store is given (0 = all)")
    ap.add_argument("--k", default="1,5,10", help="top-k values for overlap")
    ap.add_argument("--save-topk", help="write the top-k records and scores of every query and precision to this JSON file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args(argv)

    cs = EmbeddingStore(args.corpus, "corpus")
    qs = EmbeddingStore(args.queries, "queries") if args.queries else None
    for st in (cs, qs):
        if st is not None:
            check_structure(st, args.skip_hash)
            check_content(st, args.docs, args.seed)
            spot_check(st, args.spot, args.seed)
    unit_tests()
    if qs is not None:
        cross_check(cs, qs)
    if not args.skip_retrieval:
        retrieval(cs, qs, args.max_queries, args.query_vectors, [int(k) for k in args.k.split(",")], args.seed,
                  args.save_topk)

    print("\n=== Facts ===")
    facts(cs)
    if qs is not None:
        facts(qs)
    print("\n%s" % ("ALL CHECKS PASSED" if not FAILS else "%d CHECK(S) FAILED:\n  " % len(FAILS) + "\n  ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
