#!/usr/bin/env bash
# Score EVERY query against EVERY page with the reference C++ MaxSim (check_db maxsim), once per
# precision, and save the complete rankings. The result is the correctness reference for other
# implementations (CPU baseline, FPGA) and shows how well each precision reproduces the fp32 ranking.
#
#   ./run_maxsim_all.sh CORPUS_DB QUERY_DB OUT_DIR [precision ...]
#
#   precision ...   default: fp32 fp16 bf16 int8 int4 bin
#   JOBS=n          precisions scored at the same time (default: all of them, one process each)
#   TOPK_JSON=file  also compare with the top-k file written by "verify.py --save-topk file"
#
# Output in OUT_DIR:
#   scores_<precision>.txt   one line per query, ALL pages, best first:
#                                q <query record> <page record>:<score> <page record>:<score> ...
#                            ties are ordered by the lower page record. int8 / int4 / bin scores are
#                            exact integers; fp32 / fp16 / bf16 scores are floats (compare those with
#                            a small tolerance: float sums depend on the order of the additions).
#   scores_<precision>.log   messages and run time of that precision
#   summary.txt              parameters, checks, and agreement of every precision with fp32
#
# This is reference code: scalar and single-threaded per precision. Do not use its run time as a
# performance baseline.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
CORPUS=${1%/}
QUERIES=${2%/}
OUT=${3%/}
shift 3
if [ "$#" -gt 0 ]; then PRECS=("$@"); else PRECS=(fp32 fp16 bf16 int8 int4 bin); fi
JOBS=${JOBS:-${#PRECS[@]}}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN="$HERE/check_db"

# ---- build the reference program if it is missing or older than its source ------------------------
if [ ! -x "$BIN" ] || [ "$HERE/check_db.cpp" -nt "$BIN" ]; then
    echo "building $BIN"
    g++ -O2 -std=c++17 -o "$BIN" "$HERE/check_db.cpp"
fi
"$BIN" selftest > /dev/null || { echo "ERROR: check_db selftest failed"; exit 1; }

# ---- take every parameter from the metadata, and refuse stores that do not belong together --------
PARAMS=$(python3 -c '
import json, sys
c = json.load(open(sys.argv[1] + "/metadata.json"))
q = json.load(open(sys.argv[2] + "/query_metadata.json"))
assert c["format_version"] == 2 and q["format_version"] == 2, "not format version 2"
assert c["kind"] == "corpus" and q["kind"] == "queries", "wrong kind of store"
assert q["corpus"]["corpus_id"] == c["corpus_id"], "the query store was built for another corpus store"
assert q["quantization"]["clip_c_float32_hex"] == c["quantization"]["clip_c_float32_hex"], "clip value differs"
assert q["dim"] == c["dim"] and q["pad_bytes"] == c["pad_bytes"], "dim or pad_bytes differs"
print(c["dim"], c["slots_per_record"], q["slots_per_record"], c["pad_bytes"], c["num_records"], q["num_records"],
      c["quantization"]["clip_c_float32_hex"], c["corpus_id"])
' "$CORPUS" "$QUERIES")
read -r DIM DSLOTS QSLOTS PAD NDOCS NQ CLIP CORPUS_ID <<< "$PARAMS"

mkdir -p "$OUT"
{
    echo "MaxSim reference scores, written by run_maxsim_all.sh"
    echo "date            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "corpus store    : $CORPUS  ($NDOCS pages, $DSLOTS vectors each)"
    echo "query store     : $QUERIES  ($NQ queries, $QSLOTS slots each)"
    echo "dim / pad       : $DIM / $PAD"
    echo "clip c          : $CLIP"
    echo "corpus_id       : $CORPUS_ID"
    echo "precisions      : ${PRECS[*]}"
    echo "file format     : scores_<precision>.txt, one line per query, all pages, best first:"
    echo "                  q <query record> <page record>:<score> ..."
    echo
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"

# ---- score: one process per precision, at most JOBS at a time -------------------------------------
run_one() {
    local p=$1 start=$SECONDS
    if "$BIN" maxsim "$CORPUS" "$QUERIES" "$DIM" "$DSLOTS" "$QSLOTS" "$PAD" "$p" "$NDOCS" \
            > "$OUT/scores_$p.txt.tmp" 2> "$OUT/scores_$p.log"; then
        mv "$OUT/scores_$p.txt.tmp" "$OUT/scores_$p.txt"
        echo "$p: done in $((SECONDS - start)) s" | tee -a "$OUT/scores_$p.log"
    else
        rm -f "$OUT/scores_$p.txt.tmp"
        echo "$p: FAILED, see $OUT/scores_$p.log"
        return 1
    fi
}

echo "scoring $NQ queries x $NDOCS pages for: ${PRECS[*]}  ($JOBS at a time)"
FAILED=0
i=0
while [ "$i" -lt "${#PRECS[@]}" ]; do
    pids=()
    for p in "${PRECS[@]:$i:$JOBS}"; do
        rm -f "$OUT/scores_$p.txt"
        run_one "$p" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || FAILED=1; done
    i=$((i + JOBS))
done
if [ "$FAILED" -ne 0 ]; then echo "ERROR: at least one precision failed"; exit 1; fi

# ---- check the files and summarise --------------------------------------------------------------
python3 - "$OUT" "$NQ" "$NDOCS" "${TOPK_JSON:-}" "${PRECS[@]}" <<'EOF' | tee -a "$OUT/summary.txt"
import json, os, sys

out, nq, ndocs, topk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
precs = sys.argv[5:]
INT = ("int8", "int4", "bin")
bad = 0


def check(cond, msg):
    global bad
    print("  [%s] %s" % ("ok" if cond else "FAIL", msg))
    bad += not cond


rank, score = {}, {}
for p in precs:
    R, S, ok = [], [], True
    with open(os.path.join(out, "scores_%s.txt" % p)) as f:
        for i, line in enumerate(f):
            t = line.split()
            pairs = [x.split(":") for x in t[2:]]
            recs = [int(a) for a, _ in pairs]
            vals = [float(b) for _, b in pairs]
            ok &= t[0] == "q" and int(t[1]) == i and sorted(recs) == list(range(ndocs))   # every page exactly once
            if p in INT:   # exact scores: strictly best first, ties by the lower page record
                ok &= all((a > b) or (a == b and ra < rb) for a, b, ra, rb in zip(vals, vals[1:], recs, recs[1:]))
            else:          # floats are printed with 9 digits, so only require "not increasing"
                ok &= all(a >= b for a, b in zip(vals, vals[1:]))
            R.append(recs)
            S.append(vals)
    check(ok and len(R) == nq, "%-5s %d queries, each ranks all %d pages once, best first" % (p, len(R), ndocs))
    rank[p], score[p] = R, S

if "fp32" in rank:
    ks = [k for k in (1, 5, 10, 100) if k <= ndocs]
    print("\nAgreement of each precision with the fp32 ranking (mean over %d queries):" % nq)
    print("  overlap@k = share of the fp32 top-k pages that are also in this precision's top-k")
    print("  %-5s" % "prec" + "".join("  overlap@%-4d" % k for k in ks) + "  same top-1 page")
    ref = rank["fp32"]
    for p in precs:
        if p == "fp32":
            continue
        line = "  %-5s" % p
        for k in ks:
            line += "  %12.4f" % (sum(len(set(ref[i][:k]) & set(rank[p][i][:k])) for i in range(nq)) / float(nq * k))
        line += "  %15.4f" % (sum(ref[i][0] == rank[p][i][0] for i in range(nq)) / float(nq))
        print(line)
    print("  (this compares precisions with each other; it does not say whether the top pages are the\n"
          "   correct answers, because the stores hold no relevance labels)")

if topk:
    d = json.load(open(topk))
    k, n = d["k"], min(d["num_queries"], nq)
    print("\nComparison with %s (verify.py, NumPy): top-%d of the first %d queries" % (topk, k, n))
    for p in precs:
        if p not in d["precisions"]:
            continue
        r = d["precisions"][p]
        same = all(rank[p][i][:k] == r["records"][i] for i in range(n))
        if p in INT:
            same &= all([int(s) for s in score[p][i][:k]] == r["scores"][i] for i in range(n))
            check(same, "%-5s same pages and exactly the same integer scores" % p)
        else:
            rel = max(abs(a - b) / max(abs(b), 1e-30) for i in range(n) for a, b in zip(score[p][i][:k], r["scores"][i]))
            check(same and rel < 1e-5, "%-5s same pages, largest relative score difference %.1e" % (p, rel))

print("\n%s" % ("ALL CHECKS PASSED" if not bad else "%d CHECK(S) FAILED" % bad))
sys.exit(1 if bad else 0)
EOF
echo "results in $OUT"
