"""Reader for the flat multi-vector embedding format, version 2 (see FORMAT_GUIDE.md).

The same class reads a corpus store and a query store:

    from reader import EmbeddingStore
    docs = EmbeddingStore("/path/to/CORPUS_DB")            # kind auto-detected
    qrys = EmbeddingStore("/path/to/QUERY_DB")
    a = docs.get("int4", 17)                  # (count, dim) int8 values in [-7, 7]
    b = docs.get("bin", 17, binary="packed")  # (count, 16) uint8
    x = docs.get_dequantized("int8", 17)      # (count, dim) float32
    q = qrys.get("int8", 3)                   # (n_tokens of query 3, dim) int8
    qrys.question(3), docs.filepath(17)
"""

import json
import os

import numpy as np

try:
    from . import layout as L
    from . import quantize as Q
except ImportError:
    import layout as L
    import quantize as Q


class EmbeddingStore(object):
    def __init__(self, outdir, kind=None):
        """kind: 'corpus', 'queries' or None (auto-detect from the metadata file present)."""
        self.outdir = outdir
        self.kind, self.meta = L.load_metadata(outdir, kind)
        self.dim = int(self.meta["dim"])
        self.num_records = int(self.meta["num_records"])
        self.num_docs = self.num_records  # old name, kept for existing callers
        self.slots = int(self.meta["slots_per_record"])
        self.pad = int(self.meta["pad_bytes"])
        self.clip_c = Q.clip_from_meta(self.meta)
        self._cache = {}
        self._rows = None

    @property
    def precisions(self):
        return list(self.meta["precisions"].keys())

    def _open(self, precision):
        p = Q.canonical_precision(precision)
        if p not in self._cache:
            pm = self.meta["precisions"].get(p)
            if pm is None:
                raise KeyError("precision %r not present in %s" % (p, self.outdir))
            d = self.outdir
            bpv = int(pm["bytes_per_vector"])
            rec = int(pm["record_bytes"])
            data = np.memmap(os.path.join(d, pm["data_file"]), dtype=np.uint8, mode="r")
            offsets = np.fromfile(os.path.join(d, pm["offsets_file"]), dtype="<u8")
            counts = np.fromfile(os.path.join(d, pm["counts_file"]), dtype="<u4")
            n = self.num_records
            # Refuse to hand out data from a truncated or mismatched file.
            if bpv != Q.bytes_per_vector(p, self.dim) or rec != L.record_bytes(self.slots, bpv, self.pad):
                raise ValueError("%s: metadata sizes are inconsistent for %s" % (d, p))
            if len(offsets) != n + 1 or len(counts) != n:
                raise ValueError("%s: %s tables have %d / %d entries, expected %d / %d"
                                 % (d, p, len(offsets), len(counts), n + 1, n))
            if data.size != n * rec:
                raise ValueError("%s: %s is %d bytes, expected %d" % (d, pm["data_file"], data.size, n * rec))
            if not np.array_equal(offsets, np.arange(n + 1, dtype=np.uint64) * np.uint64(rec)):
                raise ValueError("%s: %s offsets are not i * record_bytes" % (d, p))
            if counts.size and (counts.min() < 1 or counts.max() > self.slots):
                raise ValueError("%s: %s counts outside 1..%d" % (d, p, self.slots))
            self._cache[p] = (data, offsets, counts, bpv)
        return self._cache[p]

    def offsets(self, precision):
        return self._open(precision)[1]

    def counts(self, precision="fp32"):
        return self._open(precision)[2]

    def record_bytes(self, precision):
        return int(self.meta["precisions"][Q.canonical_precision(precision)]["record_bytes"])

    def raw_record(self, precision, idx):
        """The whole record `idx` including its zero padding, as a 1-D uint8 memmap view."""
        data, offsets, counts, bpv = self._open(precision)
        if not 0 <= idx < len(counts):
            raise IndexError(idx)
        return data[int(offsets[idx]):int(offsets[idx + 1])]

    def raw(self, precision, idx):
        """Payload bytes of record `idx` (padding removed) as a (count, bytes_per_vector) uint8 memmap view."""
        data, offsets, counts, bpv = self._open(precision)
        if not 0 <= idx < len(counts):
            raise IndexError(idx)
        start = int(offsets[idx])
        n = int(counts[idx])
        return data[start:start + n * bpv].reshape(n, bpv)

    def get(self, precision, idx, binary="bits"):
        """Decoded codes of record `idx`: fp32->float32, fp16->float16, bf16->float32, int8->int8,
        int4->int8, bin->uint8 0/1 ('bits'), packed uint8 ('packed') or float32 +-1 ('pm1')."""
        return Q.decode(precision, self.raw(precision, idx), self.dim, binary=binary)

    def get_dequantized(self, precision, idx):
        """float32 approximation of the original vectors (bin -> +-1)."""
        return Q.dequantize(precision, self.get(precision, idx, binary="bits"), self.clip_c)

    # ---- corpus only -----------------------------------------------------
    def doc_id(self, idx):
        return self.meta["docs"][idx]["doc_id"]

    def filepath(self, idx):
        return self.meta["docs"][idx]["filepath"]

    # ---- queries only ----------------------------------------------------
    def rows(self):
        """queries.jsonl as a list of dicts: record, question, gt_answer, n_tokens."""
        if self.kind != "queries":
            raise ValueError("rows() is only available for a query store")
        if self._rows is None:
            with open(os.path.join(self.outdir, self.meta["queries_file"]), encoding="utf-8") as f:
                self._rows = [json.loads(line) for line in f if line.strip()]
        return self._rows

    def question(self, idx):
        return self.rows()[idx]["question"]

    def gt_answer(self, idx):
        return self.rows()[idx]["gt_answer"]


def read_doc(outdir, precision, idx, binary="bits"):
    return EmbeddingStore(outdir).get(precision, idx, binary=binary)


class IVFStore(object):
    """Reader for the L1 (first-stage) IVF store written by build_ivf.py.

    Everything is indexed by MEMBER NUMBER m (0 .. num_vectors-1). Members are grouped by
    cluster: cluster j owns members first[j] .. first[j+1]-1, i.e. counts[j] of them.

        ivf = IVFStore("/path/to/IVF_DB")
        C = ivf.centroids("fp32")            # (nlist, dim)
        M = ivf.members("int8", 17)          # (counts[17], dim) codes of cluster 17
        ivf.member_doc[m], ivf.member_pos[m] # page record and vector number inside that page
    """

    def __init__(self, outdir):
        self.outdir = outdir
        self.kind, self.meta = L.load_metadata(outdir, "ivf")
        m = self.meta
        self.dim = int(m["dim"])
        self.pad = int(m["pad_bytes"])
        self.nlist = int(m["nlist"])
        self.num_vectors = int(m["num_vectors"])
        self.clip_c = Q.clip_from_meta(m)
        t = m["tables"]
        rd = lambda key: np.fromfile(os.path.join(outdir, t[key]["file"]), dtype=t[key]["dtype"])
        self.first = rd("cluster_first")          # nlist+1 x uint64, member number of each cluster's first member
        self.counts = rd("cluster_counts")        # nlist   x uint32
        self.member_doc = rd("member_doc")        # N x uint32, page record of member m
        self.member_pos = rd("member_pos")        # N x uint16, vector number of member m inside its page
        self.vector_cluster = rd("vector_cluster")  # N x uint32, cluster of corpus vector g (corpus order)
        n, k = self.num_vectors, self.nlist
        if (len(self.first), len(self.counts), len(self.member_doc), len(self.member_pos), len(self.vector_cluster)) \
                != (k + 1, k, n, n, n):
            raise ValueError("%s: table lengths do not match nlist=%d / num_vectors=%d" % (outdir, k, n))
        if self.first[0] != 0 or self.first[-1] != n or not np.array_equal(np.diff(self.first), self.counts):
            raise ValueError("%s: cluster_first / cluster_counts are inconsistent" % outdir)
        self._cache = {}

    @property
    def precisions(self):
        return list(self.meta["precisions"].keys())

    def _open(self, precision):
        p = Q.canonical_precision(precision)
        if p not in self._cache:
            pm = self.meta["precisions"].get(p)
            if pm is None:
                raise KeyError("precision %r not present in %s" % (p, self.outdir))
            bpv = int(pm["bytes_per_vector"])
            if bpv != Q.bytes_per_vector(p, self.dim):
                raise ValueError("%s: bytes_per_vector of %s is inconsistent" % (self.outdir, p))
            out = []
            for key, n in (("members", self.num_vectors), ("centroids", self.nlist)):
                data = np.memmap(os.path.join(self.outdir, pm[key]["file"]), dtype=np.uint8, mode="r")
                if data.size != L.round_up(n * bpv, self.pad):
                    raise ValueError("%s: %s is %d bytes, expected %d"
                                     % (self.outdir, pm[key]["file"], data.size, L.round_up(n * bpv, self.pad)))
                out.append(data)
            self._cache[p] = (out[0], out[1], bpv)
        return self._cache[p]

    def raw_file(self, precision, which="members"):
        """The whole members (or centroids) file including its zero tail, as a uint8 memmap."""
        mem, cen, _ = self._open(precision)
        return mem if which == "members" else cen

    def centroids_raw(self, precision):
        _, cen, bpv = self._open(precision)
        return cen[:self.nlist * bpv].reshape(self.nlist, bpv)

    def centroids(self, precision, binary="bits"):
        """All centroids decoded: (nlist, dim)."""
        return Q.decode(precision, self.centroids_raw(precision), self.dim, binary=binary)

    def members_raw(self, precision, cluster=None, lo=None, hi=None):
        """Stored bytes of the members of one cluster, or of member numbers lo..hi-1: (k, bytes_per_vector)."""
        mem, _, bpv = self._open(precision)
        if cluster is not None:
            lo, hi = int(self.first[cluster]), int(self.first[cluster + 1])
        return mem[lo * bpv:hi * bpv].reshape(hi - lo, bpv)

    def members(self, precision, cluster=None, lo=None, hi=None, binary="bits"):
        """Members decoded like EmbeddingStore.get(): (k, dim)."""
        return Q.decode(precision, self.members_raw(precision, cluster, lo, hi), self.dim, binary=binary)

    def cluster_range(self, cluster):
        """(first member number, one past the last member number) of a cluster."""
        return int(self.first[cluster]), int(self.first[cluster + 1])
