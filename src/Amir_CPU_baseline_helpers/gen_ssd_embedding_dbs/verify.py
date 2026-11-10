#!/usr/bin/env python3
"""Verify an output directory written by convert.py.

    python verify.py --outdir out/ [--docs 32] [--queries 16] [--query-vectors 32] [--skip-hash]

1. structure : file sizes, offset/count tables, zero padding, SHA-256
2. roundtrip : decode random documents at every precision, error vs fp32,
               re-encode fp32 and compare bytes, fp16 bit-exactness over all docs
3. unittests : int4 / binary bit layout on hand-written vectors
4. retrieval : MaxSim top-k overlap of each precision vs fp32
Exit code is non-zero if any hard check fails.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

import quantize as Q
from reader import EmbeddingStore

FAILS = []


def check(cond, msg):
    print("  [%s] %s" % ("ok" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)
    return cond


def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 1. structure
# --------------------------------------------------------------------------
def check_structure(st, skip_hash):
    m = st.meta
    n, vpd, dim, pad = m["num_docs"], m["vectors_per_doc"], m["dim"], m["pad_bytes"]
    print("\n=== 1. Structure (N=%d, vectors/doc=%d, dim=%d, pad=%d) ===" % (n, vpd, dim, pad))
    print("  %-5s %6s %10s %10s %16s" % ("prec", "B/vec", "payload", "record", "file bytes"))
    for p, pm in m["precisions"].items():
        data, offsets, counts, bpv = st._open(p)
        path = os.path.join(st.outdir, pm["data_file"])
        size = os.path.getsize(path)
        payload = counts.astype(np.uint64) * np.uint64(bpv)
        padded = (payload + np.uint64(pad - 1)) // np.uint64(pad) * np.uint64(pad)
        print("  %-5s %6d %10d %10d %16d" % (p, bpv, int(payload[0]), int(padded[0]), size))
        check(bpv == Q.bytes_per_vector(p, dim), "%s bytes/vector = %d" % (p, bpv))
        check(len(offsets) == n + 1 and len(counts) == n, "%s table lengths N+1=%d / N=%d" % (p, len(offsets), len(counts)))
        check(np.all(counts == vpd), "%s all counts == %d" % (p, vpd))
        check(offsets[0] == 0 and np.all(offsets % pad == 0), "%s offsets start at 0 and are %d-aligned" % (p, pad))
        check(np.array_equal(np.diff(offsets), padded), "%s offsets[i+1]-offsets[i] == round_up(count*bpv, pad)" % p)
        check(size == int(offsets[-1]) == pm["file_bytes"], "%s file size == offsets[-1] == %d" % (p, size))
        # zero padding
        if len(set(padded.tolist())) == 1 and len(set(payload.tolist())) == 1:
            rec, pl = int(padded[0]), int(payload[0])
            tail = np.asarray(data).reshape(n, rec)[:, pl:]
            nz = int(np.count_nonzero(tail))
        else:
            nz = sum(int(np.count_nonzero(data[int(offsets[i]) + int(payload[i]):int(offsets[i + 1])])) for i in range(n))
        check(nz == 0, "%s padding bytes all zero" % p)
        if not skip_hash:
            for fname, want in pm["sha256"].items():
                check(sha256_file(os.path.join(st.outdir, fname)) == want, "sha256 %s" % fname)


# --------------------------------------------------------------------------
# 2. roundtrip
# --------------------------------------------------------------------------
def roundtrip(st, ndocs, seed):
    m = st.meta
    n, dim, c = m["num_docs"], m["dim"], st.clip_c
    precs = list(m["precisions"].keys())
    print("\n=== 2. Round-trip vs fp32 (c = %.6g) ===" % c)
    if "fp32" not in precs:
        print("  fp32 not present, skipping")
        return
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(ndocs, n), replace=False))
    print("  %d random documents: %s%s" % (len(idx), idx[:10].tolist(), " ..." if len(idx) > 10 else ""))
    print("  %-5s %12s %12s %14s %12s %s" % ("prec", "max|err|", "mean|err|", "max|err| unclp", "bound", "notes"))
    ref = np.stack([np.array(st.get("fp32", int(i))) for i in idx])  # (k, vpd, dim)
    clipped = np.abs(ref) > c
    for p in precs:
        if p == "fp32":
            continue
        raw = np.stack([np.array(st.raw(p, int(i))) for i in idx])
        check(np.array_equal(raw, Q.encode(p, ref, c)), "%s stored bytes == encode(fp32 record)" % p)
        codes = Q.decode(p, raw, dim)
        if p == "bin":
            agree = np.mean(codes.astype(bool) == (ref > 0))
            print("  %-5s %12s %12s %14s %12s sign agreement %.6f" % (p, "-", "-", "-", "-", agree))
            check(agree == 1.0, "bin bits == (x > 0)")
            continue
        deq = Q.dequantize(p, codes, c)
        err = np.abs(deq - ref)
        if p == "fp16":
            exact = np.mean(deq == ref)
            print("  %-5s %12.3e %12.3e %14s %12s fraction exact %.6f" % (p, err.max(), err.mean(), "-", "-", exact))
        else:
            qmax = Q.INT8_QMAX if p == "int8" else Q.INT4_QMAX
            bound = c / qmax / 2
            eu = err[~clipped].max() if np.any(~clipped) else 0.0
            print("  %-5s %12.3e %12.3e %14.3e %12.3e clipped %.2e of values"
                  % (p, err.max(), err.mean(), eu, bound, clipped.mean()))
            check(eu <= bound * (1 + 1e-5), "%s unclipped error <= c/(2*qmax)" % p)

    if "fp16" in precs:
        # bit-exactness over the whole dataset
        n_exact, n_tot, n_diff_docs, worst = 0, 0, 0, 0.0
        for i in range(n):
            a = np.asarray(st.get("fp32", i))
            b = np.asarray(st.get("fp16", i)).astype(np.float32)
            e = int(np.count_nonzero(a == b))
            n_exact += e
            n_tot += a.size
            n_diff_docs += e != a.size
            if e != a.size:
                worst = max(worst, float(np.abs(a[a != b]).max()))
        a = np.asarray(st.get("fp32", int(idx[0])))
        bf16 = np.mean((a.view("<u4") & 0xFFFF) == 0)
        print("  fp16 over ALL %d docs: %d / %d values exact (%.8f), %d docs not bit-exact"
              % (n, n_exact, n_tot, n_exact / float(n_tot), n_diff_docs))
        if worst:
            print("  largest |x| that is not exact in fp16: %.3e (fp16 smallest normal = 6.10e-05)" % worst)
        print("  fp16 lossless: %s   (doc %d: %.4f of fp32 values have zero low 16 bits, i.e. bf16-representable)"
              % ("YES" if n_exact == n_tot else "NO", idx[0], bf16))


# --------------------------------------------------------------------------
# 3. unit tests of the bit layout
# --------------------------------------------------------------------------
def unit_tests():
    print("\n=== 3. Bit-layout unit tests ===")
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

    # int8 with c = 127 -> code = rint(x) for these values
    x8 = np.array([0, 1, -1, 126.6, -200, 300, 63.2, -63.7], dtype=np.float32)
    q8 = Q.quantize_int8(x8, 127.0)
    check(q8.tolist() == [0, 1, -1, 127, -127, 127, 63, -64], "int8 codes %s" % q8.tolist())
    check(Q.encode("int8", x8, 127.0).tolist() == [0x00, 0x01, 0xFF, 0x7F, 0x81, 0x7F, 0x3F, 0xC0], "int8 bytes (two's complement)")

    # binary: dims 0..15
    xb = np.array([+1, -1, +2, +3, 0, -1, -2, +.5, -1, -1, -1, -1, +1, +1, +1, +1], dtype=np.float32)
    # bits: d0=1 d1=0 d2=1 d3=1 d4=0 (x==0 -> 0) d5=0 d6=0 d7=1 -> 0b10001101 = 0x8D ; d8..11=0, d12..15=1 -> 0xF0
    got = Q.encode("bin", xb).tolist()
    check(got == [0x8D, 0xF0], "bin bytes %s == ['0x8D', '0xF0'] (bit j of byte b = dim 8b+j)" % ["0x%02X" % b for b in got])
    check(Q.decode("bin", np.array([0x8D, 0xF0], np.uint8), 16).tolist() == (xb > 0).astype(int).tolist(), "bin unpack")
    x10 = np.ones(10, np.float32)
    check(Q.encode("bin", x10).tolist() == [0xFF, 0x03], "bin dim=10 -> 2 bytes, unused tail bits zero")


# --------------------------------------------------------------------------
# 4. retrieval sanity check
# --------------------------------------------------------------------------
def _as_float(p, codes):
    if p == "bin":
        return Q.dequantize_binary(codes)  # +-1, dot = dim - 2*hamming
    return np.asarray(codes, dtype=np.float32)


def _query_codes(p, qx, c):
    if p == "fp32":
        return qx
    if p == "fp16":
        return Q.quantize_fp16(qx)
    if p == "int8":
        return Q.quantize_int8(qx, c)
    if p == "int4":
        return Q.quantize_int4(qx, c)
    if p == "bin":
        return Q.quantize_binary(qx)


def maxsim_all(st, p, qf, nq_per, chunk=32):
    """qf: (nQ*nq, dim) float32 query matrix in the precision's code space.
    Returns (nQ, N) MaxSim scores."""
    n, vpd = st.num_docs, st.meta["vectors_per_doc"]
    nQ = qf.shape[0] // nq_per
    out = np.empty((nQ, n), dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = np.concatenate([_as_float(p, st.get(p, i, binary="bits")) for i in range(a, b)], axis=0)
        with np.errstate(all="ignore"):  # macOS Accelerate emits spurious FP warnings in sgemm
            s = qf @ d.T  # (nQ*nq, (b-a)*vpd)
        if not np.all(np.isfinite(s)):
            raise FloatingPointError("non-finite MaxSim scores for %s" % p)
        out[:, a:b] = s.reshape(nQ, nq_per, b - a, vpd).max(axis=3).sum(axis=1)
    return out


def hamming_maxsim(qbits_packed, dbits_packed, dim):
    """Reference MaxSim on packed bits with explicit popcount:
    score = sum_q max_d (dim - 2*popcount(q xor d))."""
    pc = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    ham = pc[qbits_packed[:, None, :] ^ dbits_packed[None, :, :]].sum(-1)
    return float((dim - 2 * ham).max(axis=1).sum())


def retrieval(st, nqueries, nq_vec, ks, seed):
    m = st.meta
    n, vpd, dim, c = m["num_docs"], m["vectors_per_doc"], m["dim"], st.clip_c
    precs = [p for p in m["precisions"] if p != "fp32"]
    if "fp32" not in m["precisions"]:
        print("\n=== 4. Retrieval: fp32 missing, skipped ===")
        return
    rng = np.random.default_rng(seed + 1)
    qdocs = np.sort(rng.choice(n, size=min(nqueries, n), replace=False))
    nq_vec = vpd if nq_vec <= 0 else min(nq_vec, vpd)
    rows = [np.sort(rng.choice(vpd, size=nq_vec, replace=False)) for _ in qdocs]
    qx = np.concatenate([np.asarray(st.get("fp32", int(d)))[r] for d, r in zip(qdocs, rows)], axis=0)
    ks = [k for k in ks if k <= n]
    print("\n=== 4. Retrieval sanity check: %d queries x %d vectors (from docs %s%s), %d docs ==="
          % (len(qdocs), nq_vec, qdocs[:8].tolist(), " ..." if len(qdocs) > 8 else "", n))
    print("  queries are quantized with the same c; bin uses Hamming (score = sum max (dim - 2*ham))")

    t = time.time()
    ref = maxsim_all(st, "fp32", qx, nq_vec)
    print("  fp32 MaxSim in %.1fs; source doc ranked #1 for %d/%d queries"
          % (time.time() - t, int(np.sum(ref.argmax(1) == qdocs)), len(qdocs)))
    ref_rank = np.argsort(-ref, axis=1, kind="stable")

    hdr = "  %-5s" % "prec" + "".join("  overlap@%-3d" % k for k in ks) + "  top1==fp32  self@1   time"
    print(hdr)
    for p in precs:
        t = time.time()
        qf = _as_float(p, _query_codes(p, qx, c))
        sc = maxsim_all(st, p, qf, nq_vec)
        dt = time.time() - t
        rank = np.argsort(-sc, axis=1, kind="stable")
        line = "  %-5s" % p
        for k in ks:
            ov = np.mean([len(set(ref_rank[i, :k]) & set(rank[i, :k])) / float(k) for i in range(len(qdocs))])
            line += "  %11.3f " % ov
        line += "  %9.3f  %6.3f  %5.1fs" % (np.mean(rank[:, 0] == ref_rank[:, 0]), np.mean(rank[:, 0] == qdocs), dt)
        print(line)
        if p == "bin":
            # cross-check the +-1 matmul against explicit popcount for one query/doc pair
            qp = Q.pack_binary(qx[:nq_vec] > 0)
            dp = np.asarray(st.get("bin", int(qdocs[0]), binary="packed"))
            check(abs(hamming_maxsim(qp, dp, dim) - sc[0, qdocs[0]]) < 1e-6,
                  "bin: +-1 dot product == dim - 2*popcount(xor) MaxSim")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--docs", type=int, default=32, help="random documents for the round-trip check")
    ap.add_argument("--queries", type=int, default=16, help="number of query documents")
    ap.add_argument("--query-vectors", type=int, default=32, help="vectors per query (0 = all)")
    ap.add_argument("--k", default="1,5,10", help="top-k values for overlap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args(argv)

    st = EmbeddingStore(args.outdir)
    check_structure(st, args.skip_hash)
    roundtrip(st, args.docs, args.seed)
    unit_tests()
    if not args.skip_retrieval:
        retrieval(st, args.queries, args.query_vectors, [int(k) for k in args.k.split(",")], args.seed)
    print("\n%s" % ("ALL CHECKS PASSED" if not FAILS else "%d CHECK(S) FAILED:\n  " % len(FAILS) + "\n  ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
