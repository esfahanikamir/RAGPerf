"""Reader for the flat multi-vector embedding format (see README.md).

    from reader import EmbeddingStore
    st = EmbeddingStore("/path/to/outdir")
    a = st.get("int4", 17)                 # (count, dim) int8 values in [-7, 7]
    b = st.get("bin", 17, binary="packed") # (count, 16) uint8
    x = st.get_dequantized("int8", 17)     # (count, dim) float32
"""

import json
import os

import numpy as np

import quantize as Q


class EmbeddingStore(object):
    def __init__(self, outdir):
        self.outdir = outdir
        with open(os.path.join(outdir, "metadata.json")) as f:
            self.meta = json.load(f)
        self.dim = int(self.meta["dim"])
        self.num_docs = int(self.meta["num_docs"])
        self.clip_c = float(self.meta["quantization"]["clip_c"])
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
            d = self.outdir
            data = np.memmap(os.path.join(d, pm["data_file"]), dtype=np.uint8, mode="r")
            offsets = np.fromfile(os.path.join(d, pm["offsets_file"]), dtype="<u8")
            counts = np.fromfile(os.path.join(d, pm["counts_file"]), dtype="<u4")
            self._cache[p] = (data, offsets, counts, int(pm["bytes_per_vector"]))
        return self._cache[p]

    def offsets(self, precision):
        return self._open(precision)[1]

    def counts(self, precision):
        return self._open(precision)[2]

    def raw(self, precision, idx):
        """Raw payload bytes of record `idx` as a (count, bytes_per_vector) uint8 memmap view."""
        data, offsets, counts, bpv = self._open(precision)
        if not 0 <= idx < len(counts):
            raise IndexError(idx)
        start = int(offsets[idx])
        n = int(counts[idx])
        return data[start:start + n * bpv].reshape(n, bpv)

    def get(self, precision, idx, binary="bits"):
        """Decoded codes of record `idx`: fp32->float32, fp16->float16, int8->int8,
        int4->int8, bin->uint8 0/1 ('bits'), packed uint8 ('packed') or float32 +-1 ('pm1')."""
        return Q.decode(precision, self.raw(precision, idx), self.dim, binary=binary)

    def get_dequantized(self, precision, idx):
        """float32 approximation of the original vectors (bin -> +-1)."""
        return Q.dequantize(precision, self.get(precision, idx, binary="bits"), self.clip_c)

    def doc_id(self, idx):
        return self.meta["docs"][idx]["doc_id"]

    def filepath(self, idx):
        return self.meta["docs"][idx]["filepath"]


def read_doc(outdir, precision, idx, binary="bits"):
    return EmbeddingStore(outdir).get(precision, idx, binary=binary)
