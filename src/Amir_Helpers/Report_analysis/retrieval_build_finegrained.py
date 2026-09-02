"""
Reads Batch_0_Query_0_retrieval_stats.csv (one row per token_id/doc_id/patch_id
triple, with `plan` populated only on the FIRST row of each token_id group)
and produces one row per token in finegrained.csv, with exactly these fields:

    thread_num, token_id,
    ANNIVF_total_time, ANNIVF_bytes_read, ANNIVF_requests,
    ANNIVF_compute_time, ANNIVF_data_time,
    ANNSub_total_time, ANNSub_parts_loaded, ANNSub_bytes_read,
    ANNSub_requests, ANNSub_index_comparisons,
    Sort_total_time,
    LanceRead_total_time, LanceRead_bytes_read, LanceRead_requests,
    LanceRead_compute, LanceRead_rest

All "*_total_time" fields are EXCLUSIVE time (this node's own elapsed minus
its child's cumulative elapsed), not the raw cumulative elapsed reported
in the plan text.
"""

import re
import sys
import pandas as pd


# ---- generic plan parser (unchanged from earlier validated version) ----

_DURATION_RE = re.compile(r"^([\d.]+)\s*(ns|µs|ms|s)$")
_SIZE_RE = re.compile(r"^([\d.]+)\s*(K|M|G)?\s*B?$")
_SOME_RE = re.compile(r"^Some\((.*)\)$")
_LINE_RE = re.compile(r"^(\s*)(\w+):\s*(.*)$")
_LINE_NO_COLON_RE = re.compile(r"^(\s*)(\w+)\s+(.*)$")
_METRICS_RE = re.compile(r"metrics=\[(.*)\]\s*$")


def _duration_to_ms(raw):
    m = _DURATION_RE.match(raw.strip())
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    factor = {"ns": 1e-6, "µs": 1e-3, "ms": 1.0, "s": 1000.0}[unit]
    return value * factor


def _size_to_number(raw):
    m = _SIZE_RE.match(raw.strip())
    if not m:
        return None
    value, mult = float(m.group(1)), m.group(2)
    factor = {None: 1, "K": 1e3, "M": 1e6, "G": 1e9}[mult]
    return value * factor


def _normalize_value(raw):
    raw = raw.strip()
    some_match = _SOME_RE.match(raw)
    if some_match:
        return _normalize_value(some_match.group(1))
    dur = _duration_to_ms(raw)
    if dur is not None:
        return dur
    size = _size_to_number(raw)
    if size is not None:
        return size
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _split_kv_pairs(text):
    pairs = {}
    depth = 0
    current = ""
    chunks = []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            chunks.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        chunks.append(current)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        pairs[key.strip()] = _normalize_value(value)
    return pairs


def parse_lance_plan(plan_text):
    nodes = []
    for line in plan_text.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line) or _LINE_NO_COLON_RE.match(line)
        if not m:
            continue
        indent, node_name, rest = m.groups()
        depth = len(indent) // 2
        metrics_match = _METRICS_RE.search(rest)
        metrics = _split_kv_pairs(metrics_match.group(1)) if metrics_match else {}
        attrs_text = rest[: metrics_match.start()] if metrics_match else rest
        attrs_text = attrs_text.rstrip(", ")
        attrs = _split_kv_pairs(attrs_text)
        elapsed_ms = attrs.pop("elapsed", None)
        nodes.append({"depth": depth, "node": node_name, "elapsed_ms": elapsed_ms,
                      "attrs": attrs, "metrics": metrics})
    return nodes


def _node_by_name(nodes, name):
    for n in nodes:
        if n["node"] == name:
            return n
    return None


def extract_finegrained_fields(plan_text):
    """
    Returns exactly the field set specified, computed directly from the
    parsed plan. Exclusive times are cumulative-elapsed minus child's
    cumulative-elapsed, matching the plan's actual nesting.
    """
    nodes = parse_lance_plan(plan_text)

    annivf = _node_by_name(nodes, "ANNIvfPartition")
    annsub = _node_by_name(nodes, "ANNSubIndex")
    sortex = _node_by_name(nodes, "SortExec")
    lread  = _node_by_name(nodes, "LanceRead")

    annivf_elapsed = annivf["elapsed_ms"]
    annsub_elapsed = annsub["elapsed_ms"]
    sort_elapsed   = sortex["elapsed_ms"]
    lread_elapsed  = lread["elapsed_ms"]

    find_partitions_elapsed = annivf["attrs"].get("find_partitions_elapsed") \
        or annivf["metrics"].get("find_partitions_elapsed")

    result = {
        "ANNIVF_total_time":  annivf_elapsed,
        "ANNIVF_bytes_read":  annivf["metrics"].get("bytes_read"),
        "ANNIVF_requests":    annivf["metrics"].get("requests"),
        "ANNIVF_compute_time": find_partitions_elapsed,
        "ANNIVF_data_time":   annivf_elapsed - find_partitions_elapsed
                              if (annivf_elapsed is not None and find_partitions_elapsed is not None) else None,

        "ANNSub_total_time":  annsub_elapsed - annivf_elapsed,
        "ANNSub_parts_loaded": annsub["metrics"].get("parts_loaded"),
        "ANNSub_bytes_read":  annsub["metrics"].get("bytes_read"),
        "ANNSub_requests":    annsub["metrics"].get("requests"),
        "ANNSub_index_comparisons": annsub["metrics"].get("index_comparisons"),

        "Sort_total_time":    sort_elapsed - annsub_elapsed,

        "LanceRead_total_time": lread_elapsed - sort_elapsed,
        "LanceRead_bytes_read": lread["metrics"].get("bytes_read"),
        "LanceRead_requests":   lread["metrics"].get("requests"),
        "LanceRead_compute":    lread["metrics"].get("elapsed_compute"),
    }
    result["LanceRead_rest"] = (
        result["LanceRead_total_time"] - result["LanceRead_compute"]
        if (result["LanceRead_total_time"] is not None and result["LanceRead_compute"] is not None) else None
    )
    return result


def build_finegrained_csv(retrieval_stats_csv, profile_csv, output_csv):
    df = pd.read_csv(retrieval_stats_csv)
    df = df[df['token_id'] != '_'].dropna(subset=['token_id']).copy()
    df['token_id'] = df['token_id'].astype(int)
    df['thread_id'] = df['thread_id'].astype(float).astype(int) if 'thread_id' in df.columns else df['by_thread#'].astype(float).astype(int)

    rows = []
    for token_id, group in df.groupby('token_id'):
        plan_rows = group[group['plan'].notna()]
        if len(plan_rows) == 0:
            print(f"[warn] token_id={token_id} has no plan row, skipping")
            continue
        plan_row = plan_rows.iloc[0]

        fields = extract_finegrained_fields(plan_row['plan'])
        fields['thread_num'] = plan_row['thread_id'] if 'thread_id' in plan_row else plan_row['by_thread#']
        fields['token_id'] = token_id
        rows.append(fields)

    result_df = pd.DataFrame(rows)

    # ---- join with the profile CSV for real timestamps ----
    result_df = result_df.sort_values(['thread_num', 'token_id'])
    result_df['rank'] = result_df.groupby('thread_num').cumcount()

    prof = pd.read_csv(profile_csv)
    prof = prof.sort_values(['raw_thread_id', 'task_index'])
    prof['rank'] = prof.groupby('raw_thread_id').cumcount()

    result_df = result_df.merge(
        prof[['raw_thread_id', 'rank', 'thread_start', 'open_tbl_end', 'search_start']],
        left_on=['thread_num', 'rank'], right_on=['raw_thread_id', 'rank'], how='left'
    )
    result_df = result_df.rename(columns={'thread_start': 'Thread_start_time'})
    result_df = result_df.drop(columns=['raw_thread_id', 'rank'])

    if result_df['Thread_start_time'].isna().any():
        n_missing = result_df['Thread_start_time'].isna().sum()
        print(f"[warn] {n_missing} token(s) could not be matched to a profile row "
              f"(thread_num/rank pairing failed) -- timestamps left blank for those.")

    id_cols = ['token_id', 'thread_num', 'Thread_start_time', 'open_tbl_end', 'search_start']
    other_cols = [c for c in result_df.columns if c not in id_cols]
    result_df = result_df[id_cols + other_cols]
    result_df = result_df.sort_values(['thread_num', 'token_id']).reset_index(drop=True)

    result_df.to_csv(output_csv, index=False)
    print(f"Saved {len(result_df)} tokens to {output_csv}")
    return result_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_finegrained.py <folder>")
        sys.exit(1)

    folder = sys.argv[1]
    retrieval_stats_csv = f"{folder}/Batch_0_Query_0_retrieval_stats.csv"
    profile_csv = f"{folder}/Batch_0_Query_0_retrieval_thread_profile.csv"
    output_csv = f"{folder}/Batch_0_Query_0_retrieval_finegrained.csv"

    build_finegrained_csv(retrieval_stats_csv, profile_csv, output_csv)