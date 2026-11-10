"""Quantizers / dequantizers for the flat multi-vector embedding format.

Everything here is plain NumPy so that queries can later be quantized with
exactly the same arithmetic as the stored documents.

Conventions (see README.md for the byte-level specification):

* fp32 : IEEE-754 binary32, little-endian.
* fp16 : IEEE-754 binary16, little-endian, x.astype('<f2') (round-to-nearest-even).
* int8 : q = clip(rint((x / c) * 127), -127, 127), int8, zero-point 0.
* int4 : q = clip(rint((x / c) * 7), -7, 7), two's-complement nibbles,
         two per byte, even dimension -> LOW nibble, odd dimension -> HIGH nibble.
* bin  : bit = (x > 0), np.packbits(..., bitorder='little'):
         bit j of byte b  <->  dimension 8*b + j.

`c` is a single global, symmetric clip value (float32), shared by int8 and int4.
All int arithmetic is done in float32: x is cast to float32, c is float32,
(x / c) is computed first, then multiplied by 127 (or 7), then np.rint
(round half to even), then clipped. A C/HLS reimplementation should use
float32 division, float32 multiplication and rintf()/nearbyintf() under the
default rounding mode to reproduce the bytes exactly.
"""

import numpy as np

INT8_QMAX = 127
INT4_QMAX = 7

# Canonical precision names, in output order.
PRECISIONS = ("fp32", "fp16", "int8", "int4", "bin")
_ALIASES = {"binary": "bin", "f32": "fp32", "f16": "fp16", "float32": "fp32", "float16": "fp16"}


def canonical_precision(name):
    n = _ALIASES.get(name.lower(), name.lower())
    if n not in PRECISIONS:
        raise ValueError("unknown precision %r (choose from %s)" % (name, ", ".join(PRECISIONS)))
    return n


def bytes_per_vector(precision, dim):
    p = canonical_precision(precision)
    if p == "fp32":
        return 4 * dim
    if p == "fp16":
        return 2 * dim
    if p == "int8":
        return dim
    if p == "int4":
        if dim % 2:
            raise ValueError("int4 needs an even dimension, got %d" % dim)
        return dim // 2
    if p == "bin":
        return (dim + 7) // 8
    raise AssertionError(p)


def file_name(precision):
    return "emb_%s.bin" % canonical_precision(precision)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibrate_clip(x, percentile=99.99, max_exact=64 * 1024 * 1024, sample_size=32 * 1024 * 1024, seed=0):
    """Global symmetric clip value c = percentile of |x| over all values.

    If x has more than `max_exact` elements, the percentile is taken over a
    uniform random sample (with replacement) of `sample_size` elements drawn
    with np.random.default_rng(seed). Returns (c as Python float holding a
    float32 value, info dict for metadata).
    """
    flat = np.asarray(x).reshape(-1)
    n = flat.size
    if n == 0:
        raise ValueError("cannot calibrate on an empty array")
    if n <= max_exact:
        vals = np.abs(flat.astype(np.float32, copy=False))
        method = "exact"
        used = n
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=sample_size)
        idx.sort()  # friendlier memory access, does not change the result
        vals = np.abs(flat[idx].astype(np.float32, copy=False))
        method = "random_sample_with_replacement"
        used = sample_size
    c = np.float32(np.percentile(vals, percentile))
    if not np.isfinite(c) or c <= 0:
        raise ValueError("bad clip value c=%r (all-zero data?)" % c)
    info = {
        "clip_c": float(c),
        "clip_c_float32_hex": "0x%08x" % int(np.array(c, dtype="<f4").view("<u4")),
        "percentile": float(percentile),
        "percentile_of": "|x| over all vector components",
        "percentile_interpolation": "numpy default (linear)",
        "method": method,
        "values_used": int(used),
        "values_total": int(n),
        "seed": int(seed) if method != "exact" else None,
        "fraction_clipped_in_calibration_set": float(np.mean(vals > c)),
    }
    return float(c), info


# --------------------------------------------------------------------------
# Elementwise quantizers (return integer codes, not packed)
# --------------------------------------------------------------------------
def _scaled(x, c, qmax):
    x = np.asarray(x, dtype=np.float32)
    return (x / np.float32(c)) * np.float32(qmax)


def quantize_fp16(x):
    return np.asarray(x, dtype=np.float32).astype("<f2")


def dequantize_fp16(h):
    return np.asarray(h).astype(np.float32)


def quantize_int8(x, c):
    """-> int8 codes in [-127, 127]."""
    s = _scaled(x, c, INT8_QMAX)
    return np.clip(np.rint(s), -INT8_QMAX, INT8_QMAX).astype(np.int8)


def dequantize_int8(q, c):
    return np.asarray(q, dtype=np.float32) * np.float32(np.float32(c) / np.float32(INT8_QMAX))


def quantize_int4(x, c):
    """-> int8 array holding int4 codes in [-7, 7] (unpacked)."""
    s = _scaled(x, c, INT4_QMAX)
    return np.clip(np.rint(s), -INT4_QMAX, INT4_QMAX).astype(np.int8)


def dequantize_int4(q, c):
    return np.asarray(q, dtype=np.float32) * np.float32(np.float32(c) / np.float32(INT4_QMAX))


def quantize_binary(x):
    """-> uint8 0/1 bits (unpacked), bit = x > 0."""
    return (np.asarray(x) > 0).astype(np.uint8)


def dequantize_binary(bits):
    """0/1 bits -> float32 {-1, +1}."""
    return np.asarray(bits, dtype=np.float32) * np.float32(2) - np.float32(1)


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------
def pack_int4(q):
    """int4 codes (int8 array, last axis = dim, even) -> uint8 (..., dim/2).

    byte k = (q[2k] & 0xF) | ((q[2k+1] & 0xF) << 4)
    """
    q = np.asarray(q, dtype=np.int8)
    if q.shape[-1] % 2:
        raise ValueError("int4 packing needs an even last dimension")
    if q.size and (q.min() < -8 or q.max() > 7):
        raise ValueError("int4 codes out of range")
    u = q.view(np.uint8) & np.uint8(0x0F)
    return (u[..., 0::2] | (u[..., 1::2] << np.uint8(4))).astype(np.uint8)


def unpack_int4(b):
    """uint8 (..., dim/2) -> int8 (..., dim) sign-extended int4 values."""
    b = np.asarray(b, dtype=np.uint8)
    lo = (b & np.uint8(0x0F)).astype(np.int8)
    hi = (b >> np.uint8(4)).astype(np.int8)
    out = np.empty(b.shape[:-1] + (2 * b.shape[-1],), dtype=np.int8)
    out[..., 0::2] = (lo ^ np.int8(8)) - np.int8(8)
    out[..., 1::2] = (hi ^ np.int8(8)) - np.int8(8)
    return out


def pack_binary(bits):
    """0/1 (or bool) array, last axis = dim -> uint8 (..., ceil(dim/8)).
    bit j (LSB = 0) of byte b holds dimension 8*b + j; tail bits are 0."""
    return np.packbits(np.asarray(bits, dtype=bool), axis=-1, bitorder="little")


def unpack_binary(b, dim):
    """uint8 (..., ceil(dim/8)) -> uint8 0/1 (..., dim)."""
    return np.unpackbits(np.asarray(b, dtype=np.uint8), axis=-1, count=dim, bitorder="little")


# --------------------------------------------------------------------------
# Whole-vector encode / decode (raw on-disk bytes)
# --------------------------------------------------------------------------
def encode(precision, x, c=None):
    """float vectors (..., dim) -> uint8 (..., bytes_per_vector) exactly as on disk."""
    p = canonical_precision(precision)
    x = np.asarray(x, dtype=np.float32)
    if p == "fp32":
        out = np.ascontiguousarray(x.astype("<f4", copy=False))
    elif p == "fp16":
        out = np.ascontiguousarray(quantize_fp16(x))
    elif p == "int8":
        out = np.ascontiguousarray(quantize_int8(x, c))
    elif p == "int4":
        out = pack_int4(quantize_int4(x, c))
    elif p == "bin":
        out = pack_binary(x > 0)
    return out.view(np.uint8).reshape(x.shape[:-1] + (bytes_per_vector(p, x.shape[-1]),))


def decode(precision, raw, dim, binary="bits"):
    """on-disk bytes (..., bytes_per_vector) -> codes in a usable dtype.

    fp32 -> float32, fp16 -> float16, int8 -> int8, int4 -> int8 in [-8, 7],
    bin  -> binary='bits'  : uint8 0/1 (..., dim)
            binary='packed': uint8 (..., ceil(dim/8)) unchanged
            binary='pm1'   : float32 -1/+1 (..., dim)
    """
    p = canonical_precision(precision)
    raw = np.asarray(raw, dtype=np.uint8)
    lead = raw.shape[:-1]
    if p == "fp32":
        return np.ascontiguousarray(raw).view("<f4").reshape(lead + (dim,))
    if p == "fp16":
        return np.ascontiguousarray(raw).view("<f2").reshape(lead + (dim,))
    if p == "int8":
        return np.ascontiguousarray(raw).view(np.int8).reshape(lead + (dim,))
    if p == "int4":
        return unpack_int4(raw)
    if p == "bin":
        if binary == "packed":
            return raw
        bits = unpack_binary(raw, dim)
        if binary == "bits":
            return bits
        if binary == "pm1":
            return dequantize_binary(bits)
        raise ValueError("binary must be 'bits', 'packed' or 'pm1'")


def dequantize(precision, codes, c=None):
    """Decoded codes (output of `decode`, binary='bits') -> float32 approximation of x.
    For 'bin' this returns -1/+1 (sign only; magnitude is not stored)."""
    p = canonical_precision(precision)
    if p == "fp32":
        return np.asarray(codes, dtype=np.float32)
    if p == "fp16":
        return dequantize_fp16(codes)
    if p == "int8":
        return dequantize_int8(codes, c)
    if p == "int4":
        return dequantize_int4(codes, c)
    if p == "bin":
        return dequantize_binary(codes)
