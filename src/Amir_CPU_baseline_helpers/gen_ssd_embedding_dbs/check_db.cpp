// check_db.cpp - independent C++ check of the flat multi-vector embedding stores (format version 2).
//
// This file shares no code with the Python writers. It re-derives every stored byte from the fp32
// file using only the rules in FORMAT_GUIDE.md, and it scores queries against documents straight from
// the files. If it agrees with verify.py, the format is understood the same way in both languages.
// The small functions at the top are meant to be copied into the C++ host code.
//
// Build (do NOT add -ffast-math; it breaks bit-exact rounding):
//     g++ -O2 -std=c++17 -o check_db check_db.cpp
//
// Use (verify.py prints the exact "check" command lines for your stores):
//     ./check_db check  <dir> <emb|qry> <dim> <slots_per_record> <pad_bytes> <clip_c_float32_hex>
//     ./check_db maxsim <corpus_dir> <query_dir> <dim> <doc_slots> <query_slots> <pad_bytes>
//                       <fp32|fp16|bf16|int8|int4|bin> <k> [num_queries]
//     ./check_db selftest [full]
//
// Exit code 0 = everything matched.

#ifdef __FAST_MATH__
#error "compile without -ffast-math"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// ------------------------------------------------------------------------------------------------
// Number encodings (FORMAT_GUIDE.md section 4)
// ------------------------------------------------------------------------------------------------
static inline uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// bf16 = upper 16 bits of the float32 pattern, lower 16 bits rounded to nearest (ties to even).
static inline uint16_t f32_to_bf16(float f) {
    uint64_t u = f32_bits(f);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline float bf16_to_f32(uint16_t h) { return bits_f32((uint32_t)h << 16); }  // exact

// IEEE half precision <-> float, in plain integer arithmetic (no F16C / _Float16 needed).
static uint16_t f32_to_f16(float f) {  // round to nearest, ties to even; overflow -> inf
    uint32_t x = f32_bits(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp = (x >> 23) & 0xFFu;
    uint32_t man = x & 0x7FFFFFu;
    if (exp == 0xFF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));  // inf / nan
    int e = (int)exp - 127 + 15;                                                // half exponent field
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);                             // too large -> inf
    if (e <= 0) {                                                               // half subnormal or zero
        uint32_t shift = 126u - exp;                                            // >= 14
        if (shift > 24) return (uint16_t)sign;                                  // below half the smallest subnormal
        uint32_t m = man | 0x800000u;
        uint32_t q = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(sign | q);
    }
    uint32_t q = ((uint32_t)e << 10) | (man >> 13);
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) q++;                     // may carry into the exponent
    return (uint16_t)(sign | q);
}
static float f16_to_f32(uint16_t h) {  // exact
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) return bits_f32(sign);
        int e = -1;
        do { e++; man <<= 1; } while (!(man & 0x400u));                         // normalize the subnormal
        return bits_f32(sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13));
    }
    if (exp == 31) return bits_f32(sign | 0x7F800000u | (man << 13));
    return bits_f32(sign | ((exp + 112u) << 23) | (man << 13));
}

// int8 / int4 code: q = clip(round_half_even((x / c) * qmax), -qmax, qmax), all in float32.
static inline int quantize(float x, float c, float qmax) {
    float t = x / c;                 // float32 division (not x * (1/c))
    t = t * qmax;                    // float32 multiplication
    float r = std::nearbyintf(t);    // round half to even under the default rounding mode
    if (r > qmax) r = qmax;
    if (r < -qmax) r = -qmax;
    return (int)r;
}
// int4 byte k holds dimension 2k in the LOW nibble and dimension 2k+1 in the HIGH nibble.
static inline uint8_t pack_int4(int even, int odd) { return (uint8_t)((even & 0xF) | ((odd & 0xF) << 4)); }
static inline int int4_even(uint8_t b) { return (int)(int8_t)(((b & 0x0F) ^ 0x08) - 0x08); }
static inline int int4_odd(uint8_t b) { return (int)(int8_t)(((b >> 4) ^ 0x08) - 0x08); }

static int bytes_per_vector(const std::string& p, int dim) {
    if (p == "fp32") return 4 * dim;
    if (p == "fp16" || p == "bf16") return 2 * dim;
    if (p == "int8") return dim;
    if (p == "int4") return dim / 2;
    if (p == "bin") return (dim + 7) / 8;
    std::fprintf(stderr, "unknown precision %s\n", p.c_str());
    std::exit(2);
}

// Encode one float32 vector exactly as it is stored.
static void encode(const std::string& p, const float* x, int dim, float c, uint8_t* out) {
    if (p == "fp32") {
        std::memcpy(out, x, 4 * (size_t)dim);
    } else if (p == "fp16") {
        for (int d = 0; d < dim; d++) { uint16_t h = f32_to_f16(x[d]); std::memcpy(out + 2 * d, &h, 2); }
    } else if (p == "bf16") {
        for (int d = 0; d < dim; d++) { uint16_t h = f32_to_bf16(x[d]); std::memcpy(out + 2 * d, &h, 2); }
    } else if (p == "int8") {
        for (int d = 0; d < dim; d++) out[d] = (uint8_t)(int8_t)quantize(x[d], c, 127.0f);
    } else if (p == "int4") {
        for (int k = 0; k < dim / 2; k++)
            out[k] = pack_int4(quantize(x[2 * k], c, 7.0f), quantize(x[2 * k + 1], c, 7.0f));
    } else {  // bin: bit j (LSB = 0) of byte b is dimension 8*b + j; bit = (x > 0)
        std::memset(out, 0, (size_t)(dim + 7) / 8);
        for (int d = 0; d < dim; d++)
            if (x[d] > 0.0f) out[d / 8] |= (uint8_t)(1u << (d % 8));
    }
}

// ------------------------------------------------------------------------------------------------
// File layout (FORMAT_GUIDE.md sections 2 and 3)
// ------------------------------------------------------------------------------------------------
static void die(const std::string& msg) { std::fprintf(stderr, "ERROR: %s\n", msg.c_str()); std::exit(2); }

static uint64_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) die("cannot stat " + path);
    return (uint64_t)st.st_size;
}

static void pread_full(int fd, void* buf, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) die("short read");
        p += r; n -= (size_t)r; off += (uint64_t)r;
    }
}

struct Store {              // one precision of one store
    std::string p;
    int fd = -1, dim = 0, bpv = 0;
    uint64_t n = 0, rec = 0;
    std::vector<uint64_t> off;  // N+1 x uint64 little-endian: byte offset of record i
    std::vector<uint32_t> cnt;  // N   x uint32 little-endian: real vectors in record i

    // Opens <dir>/<prefix>_<p>.bin and its two tables. Returns the number of structural errors found.
    int open_(const std::string& dir, const std::string& prefix, const std::string& prec, int dim_, long slots, long pad) {
        p = prec; dim = dim_; bpv = bytes_per_vector(p, dim);
        rec = ((uint64_t)slots * (uint64_t)bpv + (uint64_t)pad - 1) / (uint64_t)pad * (uint64_t)pad;
        const std::string base = dir + "/" + prefix + "_" + p;
        uint64_t csz = file_size(base + ".counts.u32"), osz = file_size(base + ".offsets.u64");
        if (csz % 4) die(base + ".counts.u32: size is not a multiple of 4");
        n = csz / 4;
        cnt.resize(n);
        off.resize(osz / 8);
        int f = open((base + ".counts.u32").c_str(), O_RDONLY);
        if (f < 0) die("cannot open counts of " + base);
        pread_full(f, cnt.data(), csz, 0); close(f);
        f = open((base + ".offsets.u64").c_str(), O_RDONLY);
        if (f < 0) die("cannot open offsets of " + base);
        pread_full(f, off.data(), off.size() * 8, 0); close(f);
        fd = open((base + ".bin").c_str(), O_RDONLY);
        if (fd < 0) die("cannot open " + base + ".bin");

        int bad = 0;
        if (osz != 8 * (n + 1)) { std::printf("  FAIL %s: offsets table has %llu bytes, expected %llu\n", p.c_str(), (unsigned long long)osz, (unsigned long long)(8 * (n + 1))); return 1; }
        for (uint64_t i = 0; i <= n; i++) if (off[i] != i * rec) bad++;
        if (bad) std::printf("  FAIL %s: %d offsets differ from i * record_bytes (%llu)\n", p.c_str(), bad, (unsigned long long)rec);
        if (file_size(base + ".bin") != n * rec) { std::printf("  FAIL %s: data file size != N * record_bytes\n", p.c_str()); bad++; }
        for (uint64_t i = 0; i < n; i++) if (cnt[i] < 1 || cnt[i] > (uint64_t)slots) { bad++; break; }
        return bad;
    }
    void read_record(uint64_t i, std::vector<uint8_t>& buf) const { buf.resize(rec); pread_full(fd, buf.data(), rec, off[i]); }
};

static const char* PRECS[] = {"fp32", "fp16", "bf16", "int8", "int4", "bin"};

// ------------------------------------------------------------------------------------------------
// check: every record of every precision == encode(fp32 record), padding == 0
// ------------------------------------------------------------------------------------------------
static int cmd_check(int argc, char** argv) {
    if (argc != 8) die("usage: check <dir> <emb|qry> <dim> <slots_per_record> <pad_bytes> <clip_c_float32_hex>");
    const std::string dir = argv[2], prefix = argv[3];
    const int dim = std::atoi(argv[4]);
    const long slots = std::atol(argv[5]), pad = std::atol(argv[6]);
    const float c = bits_f32((uint32_t)std::strtoul(argv[7], nullptr, 16));
    std::printf("check %s (%s): dim %d, slots %ld, pad %ld, clip c = %.9g\n", dir.c_str(), prefix.c_str(), dim, slots, pad, (double)c);
    if (!(c > 0.0f) || dim <= 0 || dim % 8) die("bad dim or clip value");

    int fails = 0;
    Store st[6];
    for (int k = 0; k < 6; k++) fails += st[k].open_(dir, prefix, PRECS[k], dim, slots, pad);
    if (fails) { std::printf("structure errors, stopping\n"); return 1; }
    const uint64_t n = st[0].n;
    for (int k = 1; k < 6; k++)
        if (st[k].n != n || st[k].cnt != st[0].cnt) { std::printf("  FAIL %s: counts differ from fp32\n", PRECS[k]); fails++; }
    if (fails) return 1;

    uint64_t bad_rec[6] = {0}, bad_pad[6] = {0}, values = 0, exact16 = 0, exactbf = 0, bad_decode = 0;
    std::vector<uint8_t> raw32, raw, want;
    std::vector<float> x;
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t cnt = st[0].cnt[i];
        st[0].read_record(i, raw32);
        x.resize((size_t)cnt * dim);
        std::memcpy(x.data(), raw32.data(), x.size() * 4);
        for (size_t j = (size_t)cnt * st[0].bpv; j < raw32.size(); j++) if (raw32[j]) { bad_pad[0]++; break; }
        values += x.size();
        for (int k = 1; k < 6; k++) {
            const Store& s = st[k];
            s.read_record(i, raw);
            want.assign((size_t)cnt * s.bpv, 0);
            for (uint32_t v = 0; v < cnt; v++) encode(s.p, &x[(size_t)v * dim], dim, c, &want[(size_t)v * s.bpv]);
            if (std::memcmp(raw.data(), want.data(), want.size()) != 0) bad_rec[k]++;
            for (size_t j = want.size(); j < raw.size(); j++) if (raw[j]) { bad_pad[k]++; break; }

            // Decode the STORED bytes with the reader-side snippets and compare with the fp32 value.
            for (uint32_t v = 0; v < cnt; v++) {
                const uint8_t* b = &raw[(size_t)v * s.bpv];
                const float* xv = &x[(size_t)v * dim];
                if (s.p == "fp16") {
                    for (int d = 0; d < dim; d++) { uint16_t h; std::memcpy(&h, b + 2 * d, 2); if (f32_bits(f16_to_f32(h)) == f32_bits(xv[d]) || (xv[d] == 0.0f && f16_to_f32(h) == 0.0f)) exact16++; }
                } else if (s.p == "bf16") {
                    for (int d = 0; d < dim; d++) { uint16_t h; std::memcpy(&h, b + 2 * d, 2); if (f32_bits(bf16_to_f32(h)) == f32_bits(xv[d])) exactbf++; }
                } else if (s.p == "int8") {
                    for (int d = 0; d < dim; d++) if ((int)(int8_t)b[d] != quantize(xv[d], c, 127.0f)) bad_decode++;
                } else if (s.p == "int4") {
                    for (int d = 0; d < dim; d += 2)
                        if (int4_even(b[d / 2]) != quantize(xv[d], c, 7.0f) || int4_odd(b[d / 2]) != quantize(xv[d + 1], c, 7.0f)) bad_decode++;
                } else {  // bin: read as little-endian 64-bit words; bit d%64 of word d/64 is dimension d
                    for (int w = 0; w < dim / 64; w++) {
                        uint64_t word; std::memcpy(&word, b + 8 * w, 8);
                        for (int j = 0; j < 64; j++) if (((word >> j) & 1u) != (xv[64 * w + j] > 0.0f ? 1u : 0u)) bad_decode++;
                    }
                }
            }
        }
    }
    std::printf("  %llu records, %llu values\n", (unsigned long long)n, (unsigned long long)values);
    for (int k = 0; k < 6; k++) {
        if (k) {
            std::printf("  [%s] %s: %llu / %llu records byte-identical to encode(fp32 record)\n", bad_rec[k] ? "FAIL" : "ok", PRECS[k],
                        (unsigned long long)(n - bad_rec[k]), (unsigned long long)n);
            if (bad_rec[k]) fails++;
        }
        std::printf("  [%s] %s: padding all zero\n", bad_pad[k] ? "FAIL" : "ok", PRECS[k]);
        if (bad_pad[k]) fails++;
    }
    std::printf("  [%s] int8 / int4 / bin: stored bytes decode to the expected codes\n", bad_decode ? "FAIL" : "ok");
    if (bad_decode) fails++;
    std::printf("  fp16: %.8f of values equal their fp32 value exactly\n", (double)exact16 / (double)values);
    std::printf("  bf16: %.8f of values equal their fp32 value exactly\n", (double)exactbf / (double)values);
    std::printf("%s\n", fails ? "CHECK FAILED" : "ALL CHECKS PASSED");
    return fails ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------
// maxsim: score = sum over query vectors of max over the document's REAL vectors of dot(q, d)
// ------------------------------------------------------------------------------------------------
struct Decoded {                       // one record, ready for dot products
    uint32_t cnt = 0;
    std::vector<float> f;              // fp32 / fp16 / bf16
    std::vector<int8_t> i8;            // int8 / int4 (one code per dimension)
    std::vector<uint64_t> w;           // bin (dim/64 words per vector)
};

static void decode_record(const Store& s, uint64_t i, std::vector<uint8_t>& raw, Decoded& out) {
    s.read_record(i, raw);
    const int dim = s.dim;
    out.cnt = s.cnt[i];                // only the first cnt vectors are real; the rest is padding
    const size_t nv = out.cnt;
    if (s.p == "fp32") { out.f.resize(nv * dim); std::memcpy(out.f.data(), raw.data(), nv * dim * 4); }
    else if (s.p == "fp16" || s.p == "bf16") {
        out.f.resize(nv * dim);
        for (size_t j = 0; j < nv * dim; j++) { uint16_t h; std::memcpy(&h, &raw[2 * j], 2); out.f[j] = s.p == "fp16" ? f16_to_f32(h) : bf16_to_f32(h); }
    } else if (s.p == "int8") { out.i8.resize(nv * dim); std::memcpy(out.i8.data(), raw.data(), nv * dim); }
    else if (s.p == "int4") {
        out.i8.resize(nv * dim);
        for (size_t j = 0; j < nv * dim / 2; j++) { out.i8[2 * j] = (int8_t)int4_even(raw[j]); out.i8[2 * j + 1] = (int8_t)int4_odd(raw[j]); }
    } else { out.w.resize(nv * (dim / 64)); std::memcpy(out.w.data(), raw.data(), nv * (dim / 8)); }
}

static double maxsim(const std::string& p, int dim, const Decoded& q, const Decoded& d) {
    double total = 0.0;
    for (uint32_t a = 0; a < q.cnt; a++) {
        double best = -1e300;
        for (uint32_t b = 0; b < d.cnt; b++) {
            double dot;
            if (p == "int8" || p == "int4") {
                int32_t acc = 0;       // |dot| <= 128 * 127 * 127 < 2^21
                const int8_t* x = &q.i8[(size_t)a * dim]; const int8_t* y = &d.i8[(size_t)b * dim];
                for (int k = 0; k < dim; k++) acc += (int32_t)x[k] * (int32_t)y[k];
                dot = acc;
            } else if (p == "bin") {
                int ham = 0;           // dot of +-1 vectors = dim - 2 * popcount(q xor d)
                for (int k = 0; k < dim / 64; k++) ham += __builtin_popcountll(q.w[(size_t)a * (dim / 64) + k] ^ d.w[(size_t)b * (dim / 64) + k]);
                dot = dim - 2 * ham;
            } else {
                double acc = 0.0;
                const float* x = &q.f[(size_t)a * dim]; const float* y = &d.f[(size_t)b * dim];
                for (int k = 0; k < dim; k++) acc += (double)x[k] * (double)y[k];
                dot = acc;
            }
            if (dot > best) best = dot;
        }
        total += best;
    }
    return total;
}

static int cmd_maxsim(int argc, char** argv) {
    if (argc != 10 && argc != 11) die("usage: maxsim <corpus_dir> <query_dir> <dim> <doc_slots> <query_slots> <pad_bytes> <precision> <k> [num_queries]");
    const int dim = std::atoi(argv[4]);
    const std::string p = argv[8];
    const size_t k = (size_t)std::atol(argv[9]);
    if (dim <= 0 || dim % 64) die("dim must be a multiple of 64");
    Store docs, qrys;
    if (docs.open_(argv[2], "emb", p, dim, std::atol(argv[5]), std::atol(argv[7])) || qrys.open_(argv[3], "qry", p, dim, std::atol(argv[6]), std::atol(argv[7]))) return 1;
    uint64_t nq = qrys.n;
    if (argc == 11 && (uint64_t)std::atol(argv[10]) < nq) nq = (uint64_t)std::atol(argv[10]);

    std::vector<uint8_t> raw;
    std::vector<Decoded> q(nq);
    for (uint64_t i = 0; i < nq; i++) decode_record(qrys, i, raw, q[i]);
    std::vector<std::vector<std::pair<double, uint64_t>>> scores(nq);
    Decoded d;
    for (uint64_t j = 0; j < docs.n; j++) {
        decode_record(docs, j, raw, d);
        for (uint64_t i = 0; i < nq; i++) scores[i].push_back({maxsim(p, dim, q[i], d), j});
    }
    const bool integer = (p == "int8" || p == "int4" || p == "bin");
    for (uint64_t i = 0; i < nq; i++) {
        std::sort(scores[i].begin(), scores[i].end(), [](const std::pair<double, uint64_t>& a, const std::pair<double, uint64_t>& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;   // best score first, ties: lower record first
        });
        std::printf("q %llu", (unsigned long long)i);
        for (size_t r = 0; r < k && r < scores[i].size(); r++) {
            if (integer) std::printf(" %llu:%lld", (unsigned long long)scores[i][r].second, (long long)scores[i][r].first);
            else std::printf(" %llu:%.9g", (unsigned long long)scores[i][r].second, scores[i][r].first);
        }
        std::printf("\n");
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------
// selftest of the conversion functions
// ------------------------------------------------------------------------------------------------
static int cmd_selftest(bool full) {
    int fails = 0;
    // every half value survives half -> float -> half
    for (uint32_t h = 0; h < 65536; h++) {
        if (((h >> 10) & 0x1F) == 31 && (h & 0x3FF)) continue;  // NaN payloads
        if (f32_to_f16(f16_to_f32((uint16_t)h)) != h) fails++;
    }
    std::printf("  [%s] fp16 -> fp32 -> fp16 is the identity for all 65536 half values\n", fails ? "FAIL" : "ok");
    struct { uint32_t f; uint16_t h; } known[] = {
        {0x3E290000u, 0x3148}, {0x3F800000u, 0x3C00}, {0xBF800000u, 0xBC00}, {0x36D8FF1Bu, 0x006C}, {0x33000000u, 0x0000},
        {0x33000001u, 0x0001}, {0x33800000u, 0x0001}, {0x477FE000u, 0x7BFF}, {0x477FF000u, 0x7C00}, {0x00000000u, 0x0000}, {0x80000000u, 0x8000}};
    int bad = 0;
    for (auto& t : known) if (f32_to_f16(bits_f32(t.f)) != t.h) { bad++; std::printf("    0x%08X -> 0x%04X, expected 0x%04X\n", t.f, f32_to_f16(bits_f32(t.f)), t.h); }
    std::printf("  [%s] float -> half on hand-checked values (ties, subnormals, overflow)\n", bad ? "FAIL" : "ok"); fails += bad;
    bad = (f32_to_bf16(bits_f32(0x3E298000u)) != 0x3E2A) + (f32_to_bf16(bits_f32(0x3E2A8000u)) != 0x3E2A) + (f32_to_bf16(bits_f32(0x3E298001u)) != 0x3E2A) +
          (f32_to_bf16(bits_f32(0x3E297FFFu)) != 0x3E29) + (f32_to_bf16(bits_f32(0xBBE80000u)) != 0xBBE8) + (f32_bits(bf16_to_f32(0x3E29)) != 0x3E290000u);
    std::printf("  [%s] bf16 rounding (ties to even) and decode\n", bad ? "FAIL" : "ok"); fails += bad;
    bad = (quantize(2.5f, 127.0f, 127.0f) != 2) + (quantize(3.5f, 127.0f, 127.0f) != 4) + (quantize(-2.5f, 127.0f, 127.0f) != -2) + (quantize(300.0f, 127.0f, 127.0f) != 127) +
          (quantize(-200.0f, 127.0f, 127.0f) != -127) + (quantize(-63.7f, 127.0f, 127.0f) != -64) + (pack_int4(1, -1) != 0xF1) + (pack_int4(7, -7) != 0x97) + (pack_int4(4, -2) != 0xE4) +
          (int4_even(0xE4) != 4) + (int4_odd(0xE4) != -2) + (int4_odd(0x97) != -7);
    std::printf("  [%s] int8 / int4 rounding (ties to even), clipping, nibble packing\n", bad ? "FAIL" : "ok"); fails += bad;
    float xb[16] = {1, -1, 2, 3, 0, -1, -2, .5f, -1, -1, -1, -1, 1, 1, 1, 1}; uint8_t ob[2];
    encode("bin", xb, 16, 1.0f, ob);
    bad = (ob[0] != 0x8D) + (ob[1] != 0xF0);
    std::printf("  [%s] binary bit order: bit j of byte b is dimension 8b+j\n", bad ? "FAIL" : "ok"); fails += bad;
#ifdef __FLT16_MANT_DIG__
    if (full) {  // compare with the compiler's own half type over ALL 2^32 float patterns
        uint64_t diff = 0;
        for (uint64_t u = 0; u < (1ull << 32); u++) {
            float f = bits_f32((uint32_t)u);
            if (f != f) continue;  // NaN
            _Float16 n16 = (_Float16)f; uint16_t nb; std::memcpy(&nb, &n16, 2);
            if (f32_to_f16(f) != nb) diff++;
        }
        std::printf("  [%s] float -> half equals the compiler's _Float16 for all 2^32 floats (%llu differ)\n", diff ? "FAIL" : "ok", (unsigned long long)diff);
        fails += diff != 0;
        diff = 0;
        for (uint32_t h = 0; h < 65536; h++) {
            uint16_t hh = (uint16_t)h; _Float16 n16; std::memcpy(&n16, &hh, 2);
            float f = (float)n16;
            if (f == f && f32_bits(f) != f32_bits(f16_to_f32(hh))) diff++;
        }
        std::printf("  [%s] half -> float equals the compiler's _Float16 for all halfs\n", diff ? "FAIL" : "ok");
        fails += diff != 0;
    }
#else
    if (full) std::printf("  (this compiler has no _Float16; exhaustive comparison skipped)\n");
#endif
    std::printf("%s\n", fails ? "SELFTEST FAILED" : "SELFTEST PASSED");
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    const uint16_t one = 1;
    if (*(const uint8_t*)&one != 1) die("this program assumes a little-endian host");
    if (argc >= 2 && std::string(argv[1]) == "check") return cmd_check(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "maxsim") return cmd_maxsim(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "selftest") return cmd_selftest(argc >= 3 && std::string(argv[2]) == "full");
    std::fprintf(stderr, "usage: check_db check|maxsim|selftest ...  (see the top of check_db.cpp)\n");
    return 2;
}
