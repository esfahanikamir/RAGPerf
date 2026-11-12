# Corpus and query embedding stores: byte-level guide (format version 2)

This guide explains every file of the **corpus store** (page embeddings) and the **query store**
(question embeddings), in plain language, with worked examples. It is written for the C++ host
code and the hardware (HLS/RTL) that will read these bytes.

All hex examples are real bytes produced by the code in this directory on a small test dataset.

---

## 0. The consistency guarantees, and how they are enforced

| guarantee | how the code enforces it |
|-----------|--------------------------|
| Documents and queries use the **same number encodings** | Every stored byte comes from one function, `quantize.encode()`. Both writers call it. |
| Documents and queries use the **same int8/int4 scale** | The clip value `c` is computed once, from the corpus, and stored in the corpus `metadata.json`. The query writer reads it from there. There is no code that computes a scale from queries. |
| Documents and queries use the **same file layout** | Every record is written by one function, `layout.write_precision()`. Both writers call it. |
| A query store **belongs to one corpus store** | The query metadata records the corpus `corpus_id` (SHA-256 of `emb_fp32.bin`) and `c`. `verify.py` rejects a mismatch. |
| The files are **what the code meant to write** | SHA-256 of every file is in the metadata; the query writer re-reads its output; `verify.py` re-derives every record from the fp32 file. |
| The format is **understood the same way in C++** | `check_db.cpp` shares no code with Python, re-derives every byte from the fp32 file and recomputes MaxSim. It must agree with `verify.py`. |
| Results are **reproducible** | Same input → byte-identical output on NumPy 1.21, 1.26 and 2.0 (tested). |

Rules that the format cannot enforce for you. Keep them in mind when writing readers:

1. **Never score past `counts[i]`.** Records are padded with zero bytes. Padding is not data (§3.3).
2. **`fp16` and `bf16` files have the same size but different bits** (§4.2, §4.3).
3. **A record number is not a page number.** Use `metadata.json → docs[i].filepath` (§2.3).
4. **Quantize in float32 in the exact operation order** if you ever re-implement it (§4.4).
5. **`gt_answer` is answer text, not the correct page.** Retrieval labels (qrels) are not in these files (§6.3).

---

## 1. The big picture

The corpus is a set of page images. The model (ColPali) turns one page into **1031 vectors**,
each with **128 numbers** ("dimensions"). A question is turned into a *variable* number of
vectors (one per token), each also with 128 numbers.

```
one page      -> 1031 vectors x 128 dims        (fixed)
one question  -> n_tokens vectors x 128 dims    (n_tokens differs per question)
```

Scoring a question against a page ("MaxSim"):

```
score = sum over the question's vectors q of ( max over the page's vectors d of dot(q, d) )
```

Both stores hold the same data six times, each time with fewer bits per number:

| name | bits per number | bytes per vector | what a number is |
|------|----------------:|-----------------:|------------------|
| fp32 | 32 | 512 | normal C `float`. **The lossless master**: every other file is derived from it. |
| fp16 | 16 | 256 | IEEE half-precision float |
| bf16 | 16 | 256 | "brain float": the top 16 bits of a `float` |
| int8 | 8  | 128 | signed integer −127..127 |
| int4 | 4  | 64  | signed integer −7..7, two per byte |
| bin  | 1  | 16  | sign bit only, eight per byte |

Rules that hold for every file:

* **No headers.** Byte 0 of a `.bin` file is already data.
* **Little-endian everywhere** (the native order of x86 and ARM).
* **Row-major.** Vectors one after another; inside a vector, dimension 0 first.
  In C terms: `T data[num_vectors][dim]`.

---

## 2. What is in a store

A store is one directory. The corpus store and the query store have **exactly the same
structure**; only the file prefix and the metadata file name differ.

| | corpus store | query store |
|---|---|---|
| a record is | one page | one question |
| file prefix | `emb` | `qry` |
| metadata file | `metadata.json` | `query_metadata.json` |
| vector slots per record (`slots_per_record`) | 1031 (`vectors_per_doc`) | `max_tokens` |
| real vectors in record `i` (`counts[i]`) | always 1031 | `n_tokens` of question `i` |
| extra file | — | `queries.jsonl` (the question texts) |

### 2.1 Files, for each precision `P` in `fp32, fp16, bf16, int8, int4, bin`

| file | type | length | meaning |
|------|------|--------|---------|
| `<prefix>_P.bin` | raw bytes | N records | the embeddings |
| `<prefix>_P.offsets.u64` | uint64 LE | N+1 numbers | `offsets[i]` = byte position of record `i`. Always `i × record_bytes`. The last entry is the file size. |
| `<prefix>_P.counts.u32` | uint32 LE | N numbers | `counts[i]` = number of **real** vectors in record `i` |

The counts are the same in all six precisions.

Example (query store with 9 questions , int8):

```
<assupmtion: max query token num = 64> + 4KB pages
--------------------------------------------------
qry_int8.offsets.u64 : 0, 8192, 16384, 24576, ... , 73728        (record_bytes = 8192)
qry_int8.counts.u32  : 0B 00 00 00 | 0E 00 00 00 | 11 00 00 00 ...
                       = 11          = 14          = 17             tokens
```

Example (corpus store, fp32):

```
<Notice: 4KB pages>
-------------------
emb_fp32.offsets.u64 : 00 00 00 00 00 00 00 00 | 00 10 08 00 00 00 00 00 | ...
                       = 0                       = 0x081000 = 528384
emb_fp32.counts.u32  : 07 04 00 00 | 07 04 00 00 ...        = 0x0407 = 1031
```

### 2.2 Which question is record `i`?

Record `i` is question `i` in dataset order (the first `question_num` questions of the test
split). `queries.jsonl` has one line per record:

```json
{"record": 0, "question": "…", "gt_answer": "…", "n_tokens": 11}
```

Inside a record, the vectors are in the order the model produced them (token order).

### 2.3 Which page is record `i`?

* Records are sorted by ascending `doc_id`.
* `doc_id` is only the position of the page in the list that was handed to the encoder. That
  list came from `os.listdir()` (see `src/run_new.py`), which has **no guaranteed order**.
* The reliable link is `metadata.json`:

```json
"docs": [ {"record": 0, "doc_id": 0, "filepath": "/.../pages/0_0.png"}, ... ]
```

Inside a record, vectors are in ascending `seq_id` order (0..1030), the order the model produced.

---

## 3. Layout of a record

### 3.1 Sizes

```
record_bytes = round_up( slots_per_record × bytes_per_vector , 4096 )
record i     = counts[i] real vectors, then zero bytes up to record_bytes
record i starts at byte  i × record_bytes   (= offsets[i])
file size    = N × record_bytes
```

Every record has the same size, starts at a multiple of 4096 and is a multiple of 4096 long.
That is what allows `O_DIRECT` reads and SSD→FPGA transfers of whole records (your buffer must
also be 4096-aligned, e.g. `aligned_alloc(4096, record_bytes)`).

Corpus store (1031 slots):

| precision | bytes/vector | payload bytes | padding bytes | record bytes | 4 KiB blocks |
|-----------|-------------:|--------------:|--------------:|-------------:|-------------:|
| fp32 | 512 | 527,872 | 512   | 528,384 | 129 |
| fp16 | 256 | 263,936 | 2,304 | 266,240 | 65 |
| bf16 | 256 | 263,936 | 2,304 | 266,240 | 65 |
| int8 | 128 | 131,968 | 3,200 | 135,168 | 33 |
| int4 | 64  | 65,984  | 3,648 | 69,632  | 17 |
| bin  | 16  | 16,496  | 3,984 | 20,480  | 5 |

Query store: `max_tokens` = the longest question, rounded up to a multiple of 32. With
`max_tokens = 64`:

| precision | record bytes |
|-----------|-------------:|
| fp32 | 32,768 |
| fp16, bf16 | 16,384 |
| int8 | 8,192 |
| int4, bin | 4,096 |

The real values for your data are in the metadata files (`precisions.<P>.record_bytes`) and
are printed by `verify.py` under "Facts".

### 3.2 Finding one number

```
record_start = i × record_bytes
vector_start = record_start + v × bytes_per_vector          (v < counts[i])
```

then, inside the vector:

| precision | where dimension `d` is |
|-----------|------------------------|
| fp32 | 4 bytes at `vector_start + 4*d` |
| fp16, bf16 | 2 bytes at `vector_start + 2*d` |
| int8 | 1 byte at `vector_start + d` |
| int4 | byte `vector_start + d/2`; low 4 bits if `d` is even, high 4 bits if `d` is odd |
| bin  | byte `vector_start + d/8`; bit number `d % 8` (bit 0 = least significant) |

Worked example (corpus, fp32, record 2, vector 5, dimension 3):
`2×528384 + 5×512 + 3×4 = 1059340`.

Worked example (queries, int8, record 1, token 2, dimension 0): `1×8192 + 2×128 + 0 = 8448`.

`verify.py` section 3 tests exactly these formulas on random positions with plain
`seek`/`read`.

### 3.3 The padding rule

Padding is always `0x00`, and zero bytes look like valid data in every precision. If you read
a whole record and treat all of it as vectors, you get phantom vectors:

| precision | padding looks like | phantom vectors in a corpus record |
|-----------|--------------------|-----------------------------------:|
| fp32 | vectors of 0.0 | 1 |
| fp16 / bf16 | vectors of 0.0 | 9 |
| int8 | vectors of 0 | 25 |
| int4 | vectors of 0 | 57 |
| bin  | vectors where every dimension is "negative" | 249 |

Why it matters:

* **Document side.** A zero vector gives `dot = 0`. If the true best match of a query token is
  negative, the phantom 0 wins and the score becomes too high: a rare, silent error. For
  binary, an all-zero vector is a real sign pattern and can beat real vectors.
* **Query side.** A zero query vector adds `max(0, …) = 0` in fp32/fp16/bf16/int8/int4, so it
  happens to be harmless there. In binary it is not: it adds `max over d of (128 − 2·popcount(d))`.

**One rule for both stores and all precisions: loop over exactly `counts[i]` vectors.**
For the corpus that is always 1031; for a question it is its `n_tokens`.

---

## 4. How one number is encoded

Running example: one real corpus vector (record 2, vector 5), dimensions 0–7, with
`c = 0.33984375`:

```
d:      0          1          2         3          4         5         6          7
x:  -0.007080   0.006409  -0.097656  -0.021118  -0.136719  0.128906  -0.134766   0.012268

fp32 : 00 00 E8 BB | 00 00 D2 3B | 00 00 C8 BD | 00 00 AD BC | 00 00 0C BE | 00 00 04 3E | 00 00 0A BE | 00 00 49 3C
bf16 : E8 BB | D2 3B | C8 BD | AD BC | 0C BE | 04 3E | 0A BE | 49 3C
fp16 : 40 9F | 90 1E | 40 AE | 68 A5 | 60 B0 | 20 30 | 50 B0 | 48 22
int8 : FD 02 DC F8 CD 30 CE 05
int4 : 00 0E 3D 0D
bin  : A2
```

The same for a query vector (record 1, token 2), dimensions 0–3, encoded with the same `c`:

```
x:   0.0000963  -0.036133  -0.065918   0.047852

fp32 : 00 00 CA 38 | 00 00 14 BD | 00 00 87 BD | 00 00 44 3D
bf16 : CA 38 | 14 BD | 87 BD | 44 3D
fp16 : 50 06 | A0 A8 | 38 AC | 20 2A
int8 : 00 F2 E7 12            (0, -14, -25, 18)
int4 : F0 1F                  (0, -1 | -1, 1)
bin  : 29  (first 8 dimensions)
```

### 4.1 fp32

A normal 32-bit IEEE float, little-endian. For the corpus this is the original data; for the
queries it is the encoder output widened to float32, which is exact for fp16 and bf16.

Bytes `00 00 E8 BB` → `0xBBE80000` → −0.007080078125.

```cpp
float x; std::memcpy(&x, p, 4);
```

The two low bytes are `00 00` because the model computed in bf16. `convert.py` prints the
fraction of such values ("bf16-like").

### 4.2 bf16 (bfloat16)

**The top 16 bits of the fp32 number**: 1 sign bit, 8 exponent bits, 7 mantissa bits, stored as
a uint16, little-endian.

fp32 `0xBBE80000` → bf16 `0xBBE8` → bytes `E8 BB`.

```cpp
uint16_t h; std::memcpy(&h, p, 2);
uint32_t u = (uint32_t)h << 16;
float x;   std::memcpy(&x, &u, 4);       // exact
```

In hardware: `fp32 = {bf16, 16'b0}`.

When writing, the lower 16 bits are rounded to nearest, ties to even:
`u16 = (u + 0x7FFF + ((u >> 16) & 1)) >> 16`. For bf16 model output nothing is lost. The exact
fraction is in the metadata: `precisions.bf16.fraction_exact_vs_fp32` (expected `1.0`).

### 4.3 fp16 (IEEE half precision)

A different 16-bit format: 1 sign bit, **5** exponent bits, **10** mantissa bits. Same file size
as bf16, different bits:

```
value 0.1650390625:   fp32 0x3E290000    bf16 0x3E29    fp16 0x3148
```

fp16 has a smaller exponent range. Numbers below about 6.1e-5 in magnitude lose bits:

```
value 6.467e-06:      bf16 0x36D9 (exact)    fp16 0x006C -> reads back as 6.437e-06
```

So **fp16 is very slightly lossy for this data, bf16 is not**. The exact fraction is in
`precisions.fp16.fraction_exact_vs_fp32` (about 0.99996 on the test data).

```cpp
// x86 with F16C:  _mm_cvtph_ps / _mm256_cvtph_ps
// portable:       f16_to_f32() in check_db.cpp (tested against the compiler's _Float16 for all values)
```

Nothing in the bytes tells you whether a file is fp16 or bf16. Only the file name does.

### 4.4 int8

One signed byte per dimension (two's complement), range −127..127. −128 never appears.

```
q = clip( round_half_even( (x / c) * 127 ), -127, 127 )
x ≈ q * c / 127
```

Example: `(−0.007080 / 0.33984375) × 127 = −2.6458` → `−3` → byte `0xFD`.

**The clip value `c`**

* One number for the whole corpus **and** all queries. Anything larger in magnitude is clamped.
* It is the 99.99th percentile of `|x|` over **all** corpus values, computed with an explicit
  formula (`quantize.percentile_linear`) so it does not depend on the NumPy version.
* Corpus: `metadata.json → quantization.clip_c` and `clip_c_float32_hex` (the raw float32 bits,
  e.g. `0x3eae0000`). Queries: the same two keys in `query_metadata.json`, copied from the corpus.
* Use the hex form in C++/HLS: `float c = bits_to_float(0x3eae0000)`.

To reproduce the bytes exactly:

```cpp
float t = x / c;                       // float32 division, NOT x * (1/c) and NOT x * (127/c)
t = t * 127.0f;                        // float32 multiply
float r = std::nearbyintf(t);          // round half to even (default rounding mode)
int   q = (int)std::fmin(std::fmax(r, -127.0f), 127.0f);
```

Round-half-to-even means 2.5 → 2, 3.5 → 4. `std::round()` rounds halves away from zero and is
wrong here. Do not compile with `-ffast-math`.

### 4.5 int4

Range −7..7 (−8 never appears), two dimensions per byte:

```
q = clip( round_half_even( (x / c) * 7 ), -7, 7 )
byte k = (q[2k] & 0xF)  |  ((q[2k+1] & 0xF) << 4)

  bit:   7 6 5 4 | 3 2 1 0
         odd dim | even dim
```

Example: dimensions 2, 3 → `−2, 0` → low nibble `0xE`, high nibble `0x0` → byte `0x0E`.
Dimensions 4, 5 → `−3, 3` → byte `0x3D`.

Reading needs sign extension:

```cpp
uint8_t b = p[k];
int8_t even = (int8_t)(((b & 0x0F) ^ 0x08) - 0x08);   // dimension 2k
int8_t odd  = (int8_t)(((b >> 4)   ^ 0x08) - 0x08);   // dimension 2k+1
```

HLS: `ap_int<4> even = b.range(3,0); ap_int<4> odd = b.range(7,4);`

### 4.6 bin (binary)

One bit per dimension: `1` if `x > 0`, otherwise `0` (exactly zero and −0.0 give `0`).
**Dimension 0 is the least significant bit of byte 0:**

```
byte b, bit j (j = 0 is the LSB)  holds dimension 8*b + j
```

Example: signs `− + − − − + − +` → bits (d0..d7) `0 1 0 0 0 1 0 1` → `2 + 32 + 128` → `0xA2`.

A vector is 16 bytes = two little-endian `uint64_t`: word 0 holds dimensions 0–63 (bit `d` =
dimension `d`), word 1 holds dimensions 64–127.

### 4.7 One rule for hardware bit positions

Load a vector's bytes into one wide integer with **byte 0 as the least significant byte**.
Then, in every precision, dimension `d` is the bit-field

```
[ d*w + w-1 : d*w ]        w = 32 (fp32), 16 (fp16, bf16), 8 (int8), 4 (int4), 1 (bin)
```

---

## 5. Scoring

| precision | dot product | range of one dot product |
|-----------|-------------|--------------------------|
| fp32, fp16, bf16 | float | about −1 … 1 (vectors are unit length) |
| int8 | integer dot of the codes | \|dot\| ≤ 128 × 127² = 2,064,512 (< 2²¹) |
| int4 | integer dot of the codes | \|dot\| ≤ 128 × 7² = 6,272 (< 2¹³) |
| bin  | `128 − 2 × popcount(q XOR d)` | −128 … +128, always even |

```cpp
// binary
uint64_t q[2], d[2];                       // memcpy 16 bytes each
int dot = 128 - 2 * (__builtin_popcountll(q[0]^d[0]) + __builtin_popcountll(q[1]^d[1]));
```

A MaxSim sum over 64 query tokens of int8 dots is at most 132,128,768 and fits `int32`.

Because documents and queries share `c`, integer scores are the real scores times a constant:

```
real ≈ (c/127)²        × int8_score
real ≈ (c/7)²          × int4_score
real ≈ (c/127)·(c/7)   × score of int8 queries against int4 documents (or the reverse)
```

`check_db.cpp` (`maxsim`) is a complete reference implementation of this section for all six
precisions, reading both stores from disk.

---

## 6. Metadata

### 6.1 Keys common to `metadata.json` and `query_metadata.json`

| key | meaning |
|-----|---------|
| `format`, `format_version`, `kind` | `"flat-multivector-embeddings"`, `2`, `"corpus"` or `"queries"` |
| `file_prefix` | `"emb"` or `"qry"` |
| `num_records`, `slots_per_record`, `dim`, `pad_bytes` | N, slots, 128, 4096 |
| `quantization.clip_c`, `clip_c_float32_hex`, `int8_qmax`, `int4_qmax`, `zero_point` | `c` (decimal and exact bits), 127, 7, 0 |
| `precisions.<P>` | file names, `bytes_per_vector`, `record_bytes`, `file_bytes`, `sha256`, and `fraction_exact_vs_fp32` (fp16, bf16) or `fraction_clipped` (int8, int4) |
| `sha256` | checksum of every file |
| `packing` | the encoding rules as text |

After copying files to another machine, `sha256sum <file>` must equal the value here.

### 6.2 Corpus only

| key | meaning |
|-----|---------|
| `num_docs`, `vectors_per_doc` | same as `num_records`, `slots_per_record` |
| `corpus_id` | SHA-256 of `emb_fp32.bin`: the identity of this corpus |
| `quantization.percentile`, `method`, `values_used` | how `c` was computed |
| `value_range_fp32` | smallest and largest value |
| `docs` | `{record, doc_id, filepath}` for every record |

### 6.3 Queries only

| key | meaning |
|-----|---------|
| `num_queries`, `max_tokens` | same as `num_records`, `slots_per_record` |
| `min_tokens_observed`, `max_tokens_observed`, `total_tokens` | token statistics |
| `corpus.corpus_id`, `corpus.num_docs`, `corpus.clip_c_float32_hex` | the corpus this store was encoded for |
| `source.encoder_output_dtype` | what the model produced: `bf16`, `fp16` or `fp32` |
| `queries_file` | `queries.jsonl` |

`gt_answer` in `queries.jsonl` is the dataset's reference **answer text**. It does not say
which page is correct. Measuring recall or nDCG needs the dataset's relevance labels (qrels)
and a mapping from pages to record numbers; neither is part of these stores.

---

## 7. Facts that come from the model, not from the files

Not recorded in any file and not verified by the tools here. From general knowledge of ColPali.

* Every vector is L2-normalized, so each of its 128 numbers is small. That is why `c` is well
  below 1.
* The 1031 page vectors are 1024 image-patch vectors plus 7 vectors for the fixed text prompt.
* A question's tokens include the prefix and the padding-style "augmentation" tokens the
  processor appends. All are stored and all take part in MaxSim.
* Pages and questions were encoded one at a time (batch size 1), so there are no padding
  vectors inside the data itself.
* The corpus and the queries must come from the **same model with the same adapter**. The
  encoder prints an adapter fingerprint at load time; it should be identical in both runs. The
  files cannot tell you this.

---

## 8. Generating and verifying the stores

Run steps 1 and 3 in the Python environment that wrote the pickle (a pickle written with
NumPy 2 cannot be loaded with NumPy 1).

**1. Corpus store** (CPU, NumPy only)

```bash
python convert.py --input /path/to/pages/embeddings.pkl --outdir /path/to/CORPUS_DB
```

**2. Query store** (GPU, runs the encoder). Run the query pipeline as before, with two
environment variables set. The pipeline stops before loading the model if the corpus store is
missing or the output directory already holds a query store.

```bash
export RAGPERF_CORPUS_DB_DIR=/path/to/CORPUS_DB
```

```bash
export RAGPERF_QUERY_DB_DIR=/path/to/QUERY_DB
```

(`RAGPERF_QUERY_DB_FORCE=1` allows overwriting an existing query store.)

**3. Verify**

```bash
python verify.py --corpus /path/to/CORPUS_DB --queries /path/to/QUERY_DB --max-queries 0 --save-topk topk.json | tee verify.log
```

The last line must be `ALL CHECKS PASSED`. The "Facts" block lists N, `c`, the record sizes
and the two C++ command lines for step 4.

**4. Independent C++ check**

```bash
g++ -O2 -std=c++17 -o check_db check_db.cpp
```

```bash
./check_db selftest
```

then the two `./check_db check …` lines printed by `verify.py`. Each must end with
`ALL CHECKS PASSED`.

**If the corpus store is regenerated**, `c` may change. Re-encode the queries without the GPU
(the fp32 query file is lossless, so the result is what the pipeline would write):

```bash
python queries.py --src /path/to/QUERY_DB --corpus /path/to/CORPUS_DB_NEW --outdir /path/to/QUERY_DB_NEW
```

---

## 9. The code

| file | role |
|------|------|
| `quantize.py` | the number encodings and the clip value. The only place a float becomes bytes. |
| `layout.py` | the record/file layout and metadata. The only place records are written. |
| `convert.py` | corpus pickle → corpus store |
| `queries.py` | `save_query_db()` (called by the pipeline) and re-encoding of a query store for another corpus |
| `reader.py` | `EmbeddingStore(dir)`: reads a corpus or a query store; `IVFStore(dir)`: reads the L1 index; both refuse truncated or inconsistent files |
| `verify.py` | all checks, top-k dump |
| `build_ivf.py` | corpus store → L1 index (IVF store), section 10 |
| `verify_ivf.py` | checks the IVF store against the corpus store; first-stage quality report |
| `check_db.cpp` | independent C++ check, reference decode functions, reference MaxSim, IVF check and reference L1 search |
| `run_maxsim_all.sh` | exact MaxSim of every query against every page in every precision, saved to files |
| `src/RAGPipeline/ImagesRAGPipeline_rerank_save_q_embeddings.py` | runs the encoder and calls `queries.save_query_db()` |

```python
from reader import EmbeddingStore
docs = EmbeddingStore("CORPUS_DB")
qrys = EmbeddingStore("QUERY_DB")
d = docs.get("int8", 17)        # (1031, 128) int8
q = qrys.get("int8", 3)         # (n_tokens, 128) int8, padding already removed
qrys.question(3); docs.filepath(17); docs.clip_c
```

---

## 10. The L1 index (IVF store)

The corpus and query stores serve the exact second stage (L2): given page numbers, read those
pages and compute MaxSim. The IVF store serves the **first stage (L1)**: for a query vector, find
similar page vectors quickly without looking at all of them, and report which pages they belong to.

The IVF store holds **exactly the same vectors as the corpus store, in a different order**:
grouped by cluster instead of by page. It is meant to be loaded completely into memory.

All examples in this section are from a test index with 309,300 vectors in 4,096 clusters.

### 10.1 How it is built (`build_ivf.py`)

1. **Read** every fp32 vector of the corpus store. Call their number N.
2. **Choose the number of clusters from N.** Nothing is hard-coded:

   ```
   nlist = smallest power of two that is at least 4 × √N
   ```

   | N (vectors) | 4 × √N | nlist |
   |---|---|---|
   | 309,300 (test index) | 2,225 | 4,096 |
   | 2,384,703 (2,313 pages) | 6,177 | 8,192 |
   | 10,000,000 | 12,649 | 16,384 |

   The factor 4 is the lower end of the FAISS range (4·√N to 16·√N). `--cluster-factor` changes
   it, `--nlist` forces a value, `--no-pow2` turns the rounding off. The build stops if there
   would be fewer than 30 vectors per cluster on average (the FAISS minimum).
3. **Cluster** with spherical k-means: every centroid has length 1, and a vector belongs to the
   centroid with the **largest dot product**.
4. **Assign** every vector to its cluster with an exactly defined rule:

   ```
   cluster(x) = the centroid with the largest exact dot product  Σ x[d] · c[d]
                (c = the stored fp32 centroid); exact ties go to the lowest cluster number
   ```

5. **Write** the files below, in all six precisions.

K-means training is not bit-reproducible between machines (it depends on the math library). That
does not matter: the centroids are stored, and step 4 is recomputed from them by `verify_ivf.py`
and checked by `check_db ivfcheck`. On the same machine, building twice gives identical files.

### 10.2 Files

| file | type | length | meaning |
|------|------|--------|---------|
| `ivf_centroids_<P>.bin` | vectors | nlist | the centroids, back to back |
| `ivf_members_<P>.bin` | vectors | N | all vectors, grouped by cluster, back to back |
| `ivf_cluster_first.u64` | uint64 LE | nlist + 1 | `first[j]` = **member number** of the first member of cluster `j`; `first[nlist]` = N |
| `ivf_cluster_counts.u32` | uint32 LE | nlist | `counts[j]` = number of members of cluster `j` |
| `ivf_member_doc.u32` | uint32 LE | N | page record of each member |
| `ivf_member_pos.u16` | uint16 LE | N | vector number of each member inside its page (0..1030) |
| `ivf_vector_cluster.u32` | uint32 LE | N | the reverse direction: cluster of corpus vector `page × 1031 + position` |
| `ivf_metadata.json` | text | — | parameters, statistics, checksums |

`<P>` is `fp32, fp16, bf16, int8, int4, bin`, with the same encodings (section 4) and the same
clip value `c` as the corpus store. The cluster tables and the page tables are shared by all
precisions: member 211 is the same vector in every precision file.

### 10.3 One indexing rule: the member number

Every vector in the index has a **member number** `m` from 0 to N−1. Everything is addressed by it:

```
vector bytes of member m   at byte  m × bytes_per_vector   in ivf_members_<P>.bin
page of member m           at byte  4 × m                  in ivf_member_doc.u32
position inside that page  at byte  2 × m                  in ivf_member_pos.u16
cluster j owns members     first[j] .. first[j+1] − 1      (that is counts[j] members)
centroid j                 at byte  j × bytes_per_vector   in ivf_centroids_<P>.bin
```

Example:

```
ivf_cluster_first.u64 : 00 00 00 00 00 00 00 00 | 0E 00 00 00 00 00 00 00 | 80 00 00 00 00 00 00 00 ...
                        = 0                       = 14                      = 128
first  = 0, 14, 128, 209, 304, 308, ...
counts = 14, 114, 81, 95, 4, ...
```

Cluster 3 owns members 209 to 303 (95 members). Its third member is member 211:

```
int8 vector : byte 211 × 128 = 27008 in ivf_members_int8.bin     -> 2C D2 27 34 EC 0E F1 04 ...
fp32 vector : byte 211 × 512 = 108032 in ivf_members_fp32.bin
page        : byte 4 × 211 = 844 in ivf_member_doc.u32           -> page record 0
position    : byte 2 × 211 = 422 in ivf_member_pos.u16           -> vector 192 of that page
```

The corpus store has the same bytes `2C D2 27 34 EC 0E F1 04` at page 0, vector 192 of
`emb_int8.bin`, and `ivf_vector_cluster.u32` says corpus vector `0 × 1031 + 192` is in cluster 3.

Inside a cluster, members are sorted by (page, position).

### 10.4 How this differs from the corpus and query stores

| | corpus / query store | IVF store |
|---|---|---|
| unit of the offset table | **bytes** (`offsets`) | **member numbers** (`cluster_first`) |
| padding between records | zero bytes up to 4096 per record | **none**: vectors are back to back |
| padding at the end of a file | none needed | zero bytes up to a multiple of 4096 |
| size of a record / cluster | fixed | **varies**; `max_cluster_size` is in the metadata |
| empty record | never | a cluster **can** be empty (`counts[j] = 0`) |

So there are no phantom vectors inside the data, but the padding rule still applies at the end
of each file: a file holds exactly N (or nlist) vectors, and whatever follows is not data.

### 10.5 Centroids

* `ivf_centroids_fp32.bin` is the master. Centroids have length 1 and full float32 precision.
* The other precisions are made by the same encoder with the same `c`.
* Unlike the page vectors, the centroids are **not** bf16 numbers, so `ivf_centroids_bf16.bin` is a
  rounded copy of the fp32 file, not an exact one. (For the members, bf16 is still exact.)
* Cluster membership was decided with the fp32 centroids. A search that uses lower-precision
  centroids and members will sometimes look in slightly different clusters. That is expected, and
  `verify_ivf.py` measures its effect.

### 10.6 The reference search

For one query vector, with two query-time settings `nprobe` and `k` (neither is stored in the index):

1. score **all** centroids against the query vector, keep the `nprobe` best
   (ties: lower cluster number first);
2. score **every member** of those clusters;
3. keep the `k` best members (ties: lower member number first);
4. the pages of those members (`ivf_member_doc`) are candidates for the second stage.

Scores are computed as in section 5, in the precision's own arithmetic. `check_db ivfsearch` is
the reference implementation and prints one line per query vector:

```
q <query> t <query vector> <member>:<page>:<position>:<score> ...

q 0 t 0 213131:234:114:134217 160763:181:395:110878 160777:234:437:108276
```

For int8, int4 and bin the scores are exact integers, so another implementation must reproduce
these lines exactly. For the float precisions, near-ties can be ordered differently.

How hits from all query vectors are combined into the candidate list for L2, and how many
candidates are passed on, is a query-time decision and not part of this format.

### 10.7 Metadata (`ivf_metadata.json`)

| key | meaning |
|-----|---------|
| `num_vectors`, `nlist`, `dim`, `pad_bytes`, `num_docs` | N, number of clusters, 128, 4096, number of pages |
| `nlist_rule` | factor, power-of-two rounding, the value the rule gave, whether it was overridden |
| `max_cluster_size`, `cluster_sizes` | largest cluster, and min / median / p90 / p99 / max / empty count |
| `similarity`, `assignment`, `assignment_info` | the rules of 10.1, and how many vectors were close calls |
| `training` | k-means settings and the per-iteration log |
| `corpus.corpus_id`, `quantization.clip_c_float32_hex` | the corpus this index belongs to, and `c` |
| `tables.<name>` | file, type, entries and SHA-256 of the five tables |
| `precisions.<P>.members`, `.centroids` | file, vectors, `payload_bytes`, `file_bytes`, SHA-256 |

### 10.8 Building and verifying

```bash
python build_ivf.py --corpus /path/to/CORPUS_DB --outdir /path/to/IVF_DB
```

```bash
python verify_ivf.py --corpus /path/to/CORPUS_DB --ivf /path/to/IVF_DB --queries /path/to/QUERY_DB | tee verify_ivf.log
```

`verify_ivf.py` checks the structure, that every corpus vector is in the index exactly once
with the right page and position, that the bytes equal the corpus store's bytes in every
precision, and that every vector is in the right cluster. With `--queries` it also prints, per
precision and per `nprobe`, how many of the true nearest vectors the first stage finds, how many
pages become candidates, and whether the pages that exact MaxSim ranks best are among them.

Then the independent C++ check (the exact line is printed under "Facts"):

```
./check_db ivfcheck  <ivf_dir> <corpus_dir> <dim> <doc_slots> <pad_bytes> <clip hex> [assign_sample]
./check_db ivfsearch <ivf_dir> <query_dir> <dim> <query_slots> <pad_bytes> <precision> <nprobe> <k> [num_queries]
```

`ivfcheck` recomputes the cluster of 20,000 vectors by default; pass `0` as the last argument to
recompute all of them (slow: every vector against every centroid in scalar code).

```python
from reader import IVFStore
ivf = IVFStore("IVF_DB")
C = ivf.centroids("int8")            # (nlist, 128)
M = ivf.members("int8", 3)           # members of cluster 3: (counts[3], 128)
lo, hi = ivf.cluster_range(3)        # member numbers lo .. hi-1
ivf.member_doc[lo:hi], ivf.member_pos[lo:hi]
```
