# Flat multi-vector embedding format (version 1)

Page embeddings from a late-interaction model are stored as raw, headerless binary files
so that the CPU, the GPU and the FPGA can all read **the same bytes**. Each document
(one page image) has `V = 1031` vectors of `D = 128` dimensions.

```
convert.py   pickle -> emb_<prec>.bin + offset/count tables + metadata.json
quantize.py  quantizers / dequantizers / bit packing (import these for queries)
reader.py    np.memmap reader: EmbeddingStore(outdir).get(prec, record)
verify.py    structure, round-trip, bit-layout unit tests, MaxSim retrieval check
```

Dependencies: NumPy (tqdm optional). Python ≥ 3.6.

```bash
python convert.py --input embeddings.pkl --outdir OUT [--pad 4096] [--percentile 99.99] \
                  [--precisions fp32,fp16,int8,int4,bin] [--inspect-only] [--force]
python verify.py  --outdir OUT [--docs 32] [--queries 16] [--query-vectors 32] [--skip-hash]
```

---

## 1. Files in the output directory

For every precision `P` ∈ {`fp32`, `fp16`, `int8`, `int4`, `bin`} there are three files:

| file                  | content |
|-----------------------|---------|
| `emb_P.bin`           | N records, back to back, no header |
| `emb_P.offsets.u64`   | `N+1` × uint64 little-endian. `offsets[i]` = byte offset of record *i* in `emb_P.bin`; `offsets[N]` = file size |
| `emb_P.counts.u32`    | `N` × uint32 little-endian. `counts[i]` = number of vectors in record *i* (always 1031 today) |

and one shared `metadata.json`.

All multi-byte values in every file are **little-endian**. No file has a header, a magic number
or a trailer.

## 2. Records

* Record *i* holds document number *i*. Documents are ordered by **ascending `doc_id`**;
  `metadata.json → docs[i] = {record, doc_id, filepath}` gives the mapping.
* A record is `counts[i]` vectors stored one after another (vector-major, dimension fastest),
  in ascending `seq_id` order, i.e. a C array `T vec[counts[i]][bytes_per_vector]`.
* After the last vector the record is **zero-padded** up to the next multiple of
  `pad_bytes` (default 4096, recorded in `metadata.json`). Padding bytes are always `0x00`.
* `offsets[i+1] - offsets[i] = round_up(counts[i] * bytes_per_vector, pad_bytes)`.
  `offsets[0] = 0`; every offset is a multiple of `pad_bytes`, so every record start is
  4 KiB-aligned for O_DIRECT / NVMe P2P transfers.

With `V = 1031`, `D = 128`, `pad = 4096`:

| precision | file          | bytes/vector | payload bytes | record bytes (padded) | 4 KiB blocks |
|-----------|---------------|-------------:|--------------:|----------------------:|-------------:|
| fp32      | `emb_fp32.bin`| 512          | 527,872       | 528,384               | 129 |
| fp16      | `emb_fp16.bin`| 256          | 263,936       | 266,240               | 65  |
| int8      | `emb_int8.bin`| 128          | 131,968       | 135,168               | 33  |
| int4      | `emb_int4.bin`| 64           | 65,984        | 69,632                | 17  |
| binary    | `emb_bin.bin` | 16           | 16,496        | 20,480                | 5   |

File size = N × record bytes. Since the count is fixed, a reader may compute
`offset = i * record_bytes` directly, but the offset table is the normative source and keeps
the format valid if counts become variable.

Byte address of dimension `d` of vector `v` in record `i` (for fp32/fp16/int8):
`offsets[i] + v * bytes_per_vector + d * sizeof(T)`.

## 3. Element encodings

Let `x` be the original float32 value and `c` the global clip value
(`metadata.json → quantization.clip_c`, also given as raw float32 bits in `clip_c_float32_hex`).

### fp32
IEEE-754 binary32, little-endian. Identical to the source vectors.

### fp16
IEEE-754 binary16, little-endian; `x.astype('<f2')` (round to nearest, ties to even).
The source values look like bf16, so fp16 is exact for nearly all values; values smaller than
the fp16 normal range (|x| < 6.1e-5) can lose bits. `verify.py` reports the exact fraction.

### int8 — one signed byte per dimension
```
q = clip(rint((x / c) * 127), -127, 127)        // stored as int8, two's complement
x ≈ q * c / 127
```
`-128` never appears. Zero-point is 0 (symmetric).

### int4 — two dimensions per byte
```
q = clip(rint((x / c) * 7), -7, 7)              // 4-bit two's complement, -8 never appears
byte[k] = (q[2k] & 0xF) | ((q[2k+1] & 0xF) << 4)  // k = 0 .. D/2-1
```
**Even dimension → low nibble (bits 3..0), odd dimension → high nibble (bits 7..4).**
Decode in C:
```c
int8_t lo = (int8_t)((b & 0x0F) ^ 0x08) - 0x08;   // dim 2k
int8_t hi = (int8_t)((b >> 4)   ^ 0x08) - 0x08;   // dim 2k+1
```
`x ≈ q * c / 7`. Example (from `verify.py`, c = 7): dimensions `(1, -1)` → byte `0xF1`;
`(7, -7)` → `0x97`; `(4, -2)` → `0xE4`.

### binary — eight dimensions per byte
```
bit(d) = (x[d] > 0) ? 1 : 0                     // x == 0 and -0 give 0
byte[b] bit j (j = 0 is the LSB) = bit(8*b + j)  // np.packbits(..., bitorder='little')
```
So bit 0 of byte 0 is dimension 0, bit 7 of byte 0 is dimension 7, bit 0 of byte 1 is
dimension 8. If D is not a multiple of 8 the unused high bits of the last byte are 0.
Reading the 16 bytes of a vector as two little-endian `uint64` gives dimensions 0–63 in the
first word (bit *d* = dimension *d*) and 64–127 in the second.
Example: signs `+ - + + 0 - - +  - - - -  + + + +` → bytes `0x8D 0xF0`.

### Rounding details (for bit-exact reimplementation)
The int8 and int4 codes are computed in **float32**: `t = (float)x / (float)c` (float32
division), then `t * 127.0f` (or `* 7.0f`), then `rintf` (round half to even, default FP
environment), then clip. Using `x * (127/c)` instead can change a few codes at rounding ties.

## 4. Quantization parameters

* One global symmetric clip value `c` is shared by int8 and int4. `c` is the
  `percentile`-th percentile (default 99.99, NumPy linear interpolation) of `|x|` over all
  vector components. If there are more than 64 Mi values, it is taken over a uniform random
  sample of 32 Mi values (with replacement) from `np.random.default_rng(seed)`, seed 0; the
  method, sample size and seed are recorded in `metadata.json → quantization`.
* Zero-point is 0, so integer dot products are proportional to the real ones:
  `dot(x, y) ≈ (c/127)² · dot(qx, qy)` for int8 and `(c/7)²` for int4.
* **Queries must be quantized with the same `c`** (`quantize.quantize_int8(q, c)` etc.).

## 5. Scoring conventions used in `verify.py`

MaxSim(query, doc) = Σ over query vectors of max over doc vectors of dot(q, d).
* fp32 / fp16: float dot products.
* int8 / int4: integer dot products of the codes (max |dot| = 128·127² < 2²¹, fits int32).
* binary: `dot = D − 2·popcount(q XOR d)`, i.e. `max dot` = `D − 2·min Hamming distance`.

## 6. `metadata.json`

| key | meaning |
|-----|---------|
| `format`, `format_version` | `"flat-multivector-embeddings"`, `1` |
| `num_docs`, `vectors_per_doc`, `dim`, `pad_bytes` | N, V, D, padding |
| `quantization` | `clip_c`, `clip_c_float32_hex`, `percentile`, `method`, `values_used`, `seed`, `int8_qmax` = 127, `int4_qmax` = 7, `zero_point` = 0 |
| `packing` | the conventions above as text |
| `precisions.<P>` | `data_file`, `offsets_file`, `counts_file`, `bytes_per_vector`, `payload_bytes_per_record`, `record_bytes`, `file_bytes`, `sha256` |
| `sha256` | SHA-256 of every output file except `metadata.json` itself |
| `docs` | list of `{record, doc_id, filepath}` in record order |
| `source` | input path and size, whether `seq_id` was per-document 0..V−1 |

## 7. Minimal C reader

```c
// read record i of emb_int4.bin into codes[V][128]
uint64_t off[2];
pread(fd_off, off, 16, 8 * (uint64_t)i);          // offsets[i], offsets[i+1]
uint32_t n;  pread(fd_cnt, &n, 4, 4 * (uint64_t)i);
uint8_t *buf = aligned_alloc(4096, off[1] - off[0]);
pread(fd_bin, buf, off[1] - off[0], off[0]);       // 4 KiB aligned: O_DIRECT safe
for (uint32_t v = 0; v < n; v++)
  for (int k = 0; k < 64; k++) {
    uint8_t b = buf[v * 64 + k];
    codes[v][2*k]   = (int8_t)((b & 0xF) ^ 8) - 8;
    codes[v][2*k+1] = (int8_t)((b >> 4)  ^ 8) - 8;
  }
```
