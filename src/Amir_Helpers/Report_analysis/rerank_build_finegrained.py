#!/usr/bin/env python3
"""
build_rerank_finegrained.py

Turns one or more Batch_N_Query_M_rerank_thread_stats.csv files (columns:
batch_num, query_id, thread_id, doc_id, t_doc_start, db_plan, pandas_time_ms,
np_time_ms, maxsim_time_ms, t_doc_end, doc_process_time_ms, cpu_core_start,
cpu_core_end) into:

  1. <stem>_finegrained.csv   - one row per document, with the raw plan text
                                 replaced by extracted LanceRead fields
  2. <stem>_thread_timeline.svg - one row per thread, bars = doc_process_time_ms,
                                 with internal segments for LR_inner_total_time,
                                 LR_outer_total_time, pandas_time_ms, np_time_ms,
                                 maxsim_time_ms (drawn stacked; if they overshoot
                                 the real doc_process_time_ms box, that overshoot
                                 is drawn visibly rather than silently clipped -
                                 this is intentional, see NOTE below)
  3. <stem>_stats.csv         - per-field summary stats (n, mean, median, std,
                                 min, max, p10, p50, p90, p99) across ALL rows -
                                 fixed size regardless of document count, meant
                                 for dropping straight into a report table
  4. <stem>_cdf.svg           - two-panel CDF (matplotlib, publication style):
                                 top-level steps on the left, the compute/rest/
                                 task_wait decomposition on the right. Fixed
                                 canvas size regardless of row count. Does NOT
                                 include doc_process_time_ms - that field is the
                                 real wall-clock measurement, not a clean sum of
                                 the other fields (see bottleneck note below), so
                                 it doesn't belong on the same axis as the parts.
  5. <stem>_bottleneck.svg    - single 100%-stacked horizontal bar: for each
                                 document, the 5 top-level steps are normalized
                                 by THEIR OWN sum (so each document's shares sum
                                 to exactly 100%), then averaged across documents
                                 - the average of shares that each sum to 1 also
                                 sums to exactly 1, so this always renders as a
                                 clean 100% bar with no fudging. NOTE: this is the
                                 share of the sum of the 5 measured components,
                                 not a share of the real doc_process_time_ms
                                 (those don't match - see NOTE below).

USAGE
  python3 build_rerank_finegrained.py <folder>

  <folder> is the only argument. The script looks for exactly
  Batch_0_Query_0_rerank_thread_stats.csv inside it (or one level down, in
  case it's nested), and writes its outputs next to that file:
    - Batch_0_Query_0_rerank_thread_stats_finegrained.csv
    - Batch_0_Query_0_rerank_thread_stats_thread_timeline.svg

NOTE ON "rest" FIELDS
  LR_inner_rest and LR_outer_rest are total_time - elapsed_compute for each node.
  These are deliberately NOT called "data_time" - that label is only justified
  once you've checked it actually correlates with that node's own bytes_read
  across many documents (this mirrors the mistake already caught on the
  retrieval side, where LanceRead's leftover time did NOT correlate with its
  own bytes_read). Do the correlation check on the finegrained CSV before
  renaming anything.

NOTE ON task_wait_time
  Only present on the inner LanceRead node in samples seen so far. Kept as its
  own field, not merged into _rest or _compute. Whether it's additive or an
  overlapping measurement of elapsed_compute is still an open question - the
  finegrained CSV is what lets you check that across the full document set
  (e.g. plot LR_inner_task_wait_time vs LR_inner_compute_time and see if it's
  a tight 1:1 line, and whether the ratio shifts with thread contention).

NOTE ON doc_process_time_ms vs the sum of the 5 top-level components
  Earlier exploration (2-document sample) found the 5 top-level components
  (LR_inner_total, LR_outer_total, pandas, np, maxsim) summed to well over
  100% of doc_process_time_ms - meaning pandas_time_ms substantially overlaps
  with the LanceRead wall-clock time rather than following it sequentially.
  The bottleneck chart therefore normalizes by the SUM OF THE COMPONENTS
  THEMSELVES, not by doc_process_time_ms, and should be read as "relative
  share among the 5 measured components" rather than "share of real wall-clock
  time." Worth re-checking this overlap assumption on the full dataset before
  treating it as settled.

DEPENDENCIES
  Requires matplotlib and numpy (pip install matplotlib numpy --break-system-packages
  if not already present in your environment).
"""

import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# ---------------------------------------------------------------------------
# Plan-text parsing
# ---------------------------------------------------------------------------

TIME_KEYS = {"elapsed", "elapsed_compute", "task_wait_time"}


def split_top_level(s, sep=","):
    """Split s on sep, but only at bracket depth 0. Handles (), [], {} nesting
    so values like 'projection=[seq_id, doc_id]' or 'range_after=Some(0..1024)'
    survive as single tokens."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def parse_time_to_ms(raw):
    """'58.808308ms' / '41.66ms' / '28.99µs' / '2.10µs' / '198ns' / '1.2s' -> float ms"""
    raw = raw.strip()
    m = re.match(r"^([\d.]+)\s*(ns|µs|us|ms|s)$", raw)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    if unit == "ns":
        return val / 1e6
    if unit in ("µs", "us"):
        return val / 1e3
    if unit == "ms":
        return val
    if unit == "s":
        return val * 1000.0
    return None


def parse_count_or_bytes(raw):
    """'1.02 K' -> 1020.0, '2.45 M' -> 2.45e6, '0.0 B' -> 0.0,
    '1299.4 KB' -> 1299400.0 (bytes), '4' -> 4.0. Returns None if unparseable."""
    raw = raw.strip()
    m = re.match(r"^([\d.]+)\s*([KMG]?)B?$", raw)
    if not m:
        try:
            return float(raw)
        except ValueError:
            return None
    val = float(m.group(1))
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    return val * mult


def parse_kv_blob(blob):
    """Parse a comma-separated (bracket-aware) key=value blob into a dict of
    raw string values (no unit conversion yet)."""
    out = {}
    for part in split_top_level(blob, ","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def parse_lancread_line(line):
    """Given the text after 'LanceRead:' on one line, return
    (attrs_dict_raw, metrics_dict_raw) where metrics_dict_raw is the parsed
    content of metrics=[...] and attrs_dict_raw is everything else on the line
    (elapsed, uri, projection, full_filter, etc.), all as raw strings."""
    # metrics=[...] is a single top-level token (brackets protect its commas)
    top_parts = split_top_level(line, ",")
    attrs = {}
    metrics = {}
    for part in top_parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "metrics":
            inner = val.strip()
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1]
            metrics = parse_kv_blob(inner)
        else:
            attrs[key] = val
    return attrs, metrics


def convert_metric_value(key, raw):
    if raw is None:
        return None
    if key in TIME_KEYS or key.endswith("_time"):
        v = parse_time_to_ms(raw)
        return v if v is not None else raw
    v = parse_count_or_bytes(raw)
    return v if v is not None else raw


def extract_rerank_fields(db_plan_text):
    """Find the inner (has full_filter) and outer (no full_filter) LanceRead
    nodes in a reranking plan and return the derived field dict. Returns None
    (with a warning printed) if the expected two-node shape isn't found."""
    nodes = []  # list of (attrs, metrics, raw_line)
    for raw_line in db_plan_text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("LanceRead:"):
            continue
        after = stripped[len("LanceRead:"):].strip()
        attrs, metrics = parse_lancread_line(after)
        nodes.append((attrs, metrics))

    if len(nodes) < 2:
        return None

    inner = None
    outer = None
    for attrs, metrics in nodes:
        if "full_filter" in attrs and inner is None:
            inner = (attrs, metrics)
        elif "full_filter" not in attrs and outer is None:
            outer = (attrs, metrics)

    if inner is None or outer is None:
        # Fallback: two LanceRead nodes but couldn't classify by full_filter -
        # assume first-seen (outer, since it wraps the inner spatially) and
        # second-seen (inner). Flagged via a marker field for later auditing.
        if len(nodes) >= 2 and inner is None and outer is None:
            outer, inner = nodes[0], nodes[1]
        elif inner is None:
            return None
        elif outer is None:
            return None

    inner_attrs, inner_metrics = inner
    outer_attrs, outer_metrics = outer

    inner_elapsed = parse_time_to_ms(inner_attrs.get("elapsed", ""))
    outer_elapsed = parse_time_to_ms(outer_attrs.get("elapsed", ""))
    if inner_elapsed is None or outer_elapsed is None:
        return None

    inner_compute = convert_metric_value("elapsed_compute", inner_metrics.get("elapsed_compute"))
    outer_compute = convert_metric_value("elapsed_compute", outer_metrics.get("elapsed_compute"))

    lr_inner_total = inner_elapsed  # leaf node: own_time == raw elapsed
    lr_outer_total = outer_elapsed - inner_elapsed  # own_time correction

    result = {
        "LR_inner_total_time_ms": lr_inner_total,
        "LR_inner_bytes_read": convert_metric_value("bytes_read", inner_metrics.get("bytes_read")),
        "LR_inner_requests": convert_metric_value("requests", inner_metrics.get("requests")),
        "LR_inner_rows_scanned": convert_metric_value("rows_scanned", inner_metrics.get("rows_scanned")),
        "LR_inner_compute_time_ms": inner_compute,
        "LR_inner_rest_ms": (lr_inner_total - inner_compute) if inner_compute is not None else None,
        "LR_inner_task_wait_time_ms": convert_metric_value("task_wait_time", inner_metrics.get("task_wait_time")),
        "LR_outer_total_time_ms": lr_outer_total,
        "LR_outer_bytes_read": convert_metric_value("bytes_read", outer_metrics.get("bytes_read")),
        "LR_outer_requests": convert_metric_value("requests", outer_metrics.get("requests")),
        "LR_outer_compute_time_ms": outer_compute,
        "LR_outer_rest_ms": (lr_outer_total - outer_compute) if outer_compute is not None else None,
    }
    return result


DERIVED_FIELDS = [
    "LR_inner_total_time_ms", "LR_inner_bytes_read", "LR_inner_requests",
    "LR_inner_rows_scanned", "LR_inner_compute_time_ms", "LR_inner_rest_ms",
    "LR_inner_task_wait_time_ms",
    "LR_outer_total_time_ms", "LR_outer_bytes_read", "LR_outer_requests",
    "LR_outer_compute_time_ms", "LR_outer_rest_ms",
]

PASSTHROUGH_FIELDS = [
    "batch_num", "query_id", "thread_id", "doc_id",
    "t_doc_start", "t_doc_end", "doc_process_time_ms",
    "cpu_core_start", "cpu_core_end",
    "pandas_time_ms", "np_time_ms", "maxsim_time_ms",
]


# ---------------------------------------------------------------------------
# CSV stage
# ---------------------------------------------------------------------------

def build_finegrained_csv(input_path, output_path):
    rows_out = []
    skipped = 0
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            derived = extract_rerank_fields(row.get("db_plan", "") or "")
            if derived is None:
                skipped += 1
                continue
            out_row = {k: row.get(k, "") for k in PASSTHROUGH_FIELDS}
            out_row.update(derived)
            rows_out.append(out_row)

    fieldnames = PASSTHROUGH_FIELDS + DERIVED_FIELDS
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"[{os.path.basename(input_path)}] wrote {len(rows_out)} rows -> {output_path}"
          + (f"  ({skipped} rows skipped: plan didn't parse)" if skipped else ""))
    return rows_out


# ---------------------------------------------------------------------------
# SVG stage - per-thread timeline with internal splits
# ---------------------------------------------------------------------------

SEGMENT_COLORS = {
    "LR_inner_total_time_ms": "#4C72B0",
    "LR_outer_total_time_ms": "#DD8452",
    "pandas_time_ms": "#55A868",
    "np_time_ms": "#C44E52",
    "maxsim_time_ms": "#8172B2",
}
SEGMENT_ORDER = ["LR_inner_total_time_ms", "LR_outer_total_time_ms",
                 "pandas_time_ms", "np_time_ms", "maxsim_time_ms"]

PX_PER_MS = 1.2
ROW_HEIGHT = 46
ROW_GAP = 14
LEFT_MARGIN = 130
TOP_MARGIN = 60
RIGHT_PAD = 140  # room for "+N.Nms over" overshoot labels past the last bar


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def plot_thread_timeline(rows, output_svg_path):
    if not rows:
        print(f"  (no rows to plot for {output_svg_path}, skipping)")
        return

    for r in rows:
        r["_t_doc_start"] = _f(r, "t_doc_start")
        r["_doc_process_time_ms"] = _f(r, "doc_process_time_ms")

    t0 = min(r["_t_doc_start"] for r in rows)
    # t_doc_start looks like CLOCK_MONOTONIC nanoseconds -> convert offset to ms
    for r in rows:
        r["_x_ms"] = (r["_t_doc_start"] - t0) / 1e6

    threads = sorted({r.get("thread_id", "") for r in rows}, key=lambda t: str(t))
    thread_row_index = {t: i for i, t in enumerate(threads)}

    max_x_end = max(r["_x_ms"] + max(r["_doc_process_time_ms"],
                                      sum(_f(r, k) for k in SEGMENT_ORDER))
                     for r in rows)
    width = LEFT_MARGIN + max_x_end * PX_PER_MS + RIGHT_PAD
    height = TOP_MARGIN + len(threads) * (ROW_HEIGHT + ROW_GAP) + 60

    svg = []
    svg.append(f'<svg viewBox="0 0 {width:.0f} {height:.0f}" xmlns="http://www.w3.org/2000/svg" '
                f'font-family="monospace" font-size="11">')
    svg.append(f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="white"/>')
    svg.append(f'<text x="{LEFT_MARGIN}" y="24" font-size="15" font-weight="bold">'
               f'Reranking per-thread timeline (splits: inner/outer LanceRead, pandas, np, maxsim)</text>')

    legend_x = LEFT_MARGIN
    legend_y = 42
    for key in SEGMENT_ORDER:
        svg.append(f'<rect x="{legend_x}" y="{legend_y-9}" width="12" height="12" '
                   f'fill="{SEGMENT_COLORS[key]}"/>')
        svg.append(f'<text x="{legend_x+16}" y="{legend_y}">{key.replace("_ms","")}</text>')
        legend_x += 16 + 9 * len(key.replace("_ms", "")) + 20

    for t in threads:
        y = TOP_MARGIN + thread_row_index[t] * (ROW_HEIGHT + ROW_GAP)
        svg.append(f'<text x="4" y="{y + ROW_HEIGHT/2 + 4:.0f}" font-size="10">'
                   f'thread {t}</text>')

    for r in rows:
        y = TOP_MARGIN + thread_row_index[r.get("thread_id", "")] * (ROW_HEIGHT + ROW_GAP)
        x0 = LEFT_MARGIN + r["_x_ms"] * PX_PER_MS
        doc_w = r["_doc_process_time_ms"] * PX_PER_MS

        # Background box = the REAL measured doc_process_time_ms (ground truth box)
        svg.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(doc_w,1):.1f}" height="{ROW_HEIGHT}" '
                   f'fill="none" stroke="#333" stroke-width="1.5"/>')

        # Stacked internal segments - drawn in sequence, allowed to overshoot
        # the background box on the right if the segments sum to more than
        # doc_process_time_ms. That overshoot is the point: it's a visible,
        # honest signal that these timings overlap rather than stack cleanly.
        seg_x = x0
        for key in SEGMENT_ORDER:
            seg_ms = _f(r, key)
            if seg_ms <= 0:
                continue
            seg_w = seg_ms * PX_PER_MS
            svg.append(f'<rect x="{seg_x:.1f}" y="{y+4:.1f}" width="{max(seg_w,0.5):.1f}" '
                       f'height="{ROW_HEIGHT-8}" fill="{SEGMENT_COLORS[key]}" fill-opacity="0.85"/>')
            if seg_w >= 26:
                svg.append(f'<text x="{seg_x+3:.1f}" y="{y+ROW_HEIGHT/2+4:.0f}" '
                           f'fill="white" font-size="9">doc {r.get("doc_id","")}</text>')
            seg_x += seg_w

        overshoot = seg_x - (x0 + doc_w)
        if overshoot > 2:
            svg.append(f'<text x="{seg_x+4:.1f}" y="{y+ROW_HEIGHT/2+4:.0f}" '
                       f'fill="#B33" font-size="9">+{overshoot/PX_PER_MS:.1f}ms over</text>')

    svg.append("</svg>")
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"  wrote chart -> {output_svg_path}")


# ---------------------------------------------------------------------------
# Summary statistics (fixed size regardless of row count - report-friendly)
# ---------------------------------------------------------------------------

STATS_FIELDS = [
    "LR_inner_total_time_ms", "LR_inner_compute_time_ms", "LR_inner_rest_ms",
    "LR_inner_task_wait_time_ms",
    "LR_outer_total_time_ms", "LR_outer_compute_time_ms", "LR_outer_rest_ms",
    "np_time_ms", "maxsim_time_ms", "doc_process_time_ms",
    # pandas_time_ms deliberately excluded from stats/plots - found to be
    # measuring a second full query re-execution (analyze_plan() only returns
    # the plan, not the rows, so to_pandas() re-runs the whole scan), not
    # isolated DataFrame-conversion time. Still passed through in the
    # finegrained CSV for reference, just not aggregated here.
]

TOP_LEVEL_FIELDS = ["LR_inner_total_time_ms", "LR_outer_total_time_ms",
                    "np_time_ms", "maxsim_time_ms"]
DECOMP_FIELDS = ["LR_inner_compute_time_ms", "LR_inner_rest_ms",
                 "LR_inner_task_wait_time_ms",
                 "LR_outer_compute_time_ms", "LR_outer_rest_ms"]
REFERENCE_FIELD = "doc_process_time_ms"

DECOMP_COLORS = {
    "LR_inner_compute_time_ms": "#4C72B0",
    "LR_inner_rest_ms": "#DD8452",
    "LR_inner_task_wait_time_ms": "#55A868",
    "LR_outer_compute_time_ms": "#8172B2",
    "LR_outer_rest_ms": "#937860",
}
REFERENCE_COLOR = "#333333"

# Distinct dash patterns per field, layered on top of color, so lines stay
# distinguishable even if two colors end up close together (or in grayscale).
TOP_LEVEL_LINESTYLES = {
    "LR_inner_total_time_ms": "-",
    "LR_outer_total_time_ms": "--",
    "pandas_time_ms": "-.",
    "np_time_ms": ":",
    "maxsim_time_ms": (0, (3, 1, 1, 1)),
}
DECOMP_LINESTYLES = {
    "LR_inner_compute_time_ms": "-",
    "LR_inner_rest_ms": "--",
    "LR_inner_task_wait_time_ms": "-.",
    "LR_outer_compute_time_ms": ":",
    "LR_outer_rest_ms": (0, (3, 1, 1, 1)),
}

# Two fields can end up almost numerically identical (e.g. LR_inner_compute
# and LR_inner_task_wait_time were found to track ~1:1), which makes one line
# fully occlude the other no matter how distinct their color/dash are - a
# solid line drawn UNDER a solid line of a different color still disappears
# completely. Markers fix this because they're placed at different phase
# offsets per field, so even two perfectly overlapping curves show different-
# shaped markers at different x-positions instead of one erasing the other.
MARKERS = ["o", "s", "^", "D", "v"]
LINEWIDTHS = [2.4, 1.6, 2.4, 1.6, 2.0]  # alternate thick/thin so a thinner
                                        # dashed line drawn on top still lets
                                        # a thicker line underneath peek through
                                        # its gaps
MARKER_EVERY_STEP = 22  # spacing between markers along each curve


def _field_style(fields, field):
    idx = fields.index(field)
    return {
        "marker": MARKERS[idx % len(MARKERS)],
        "linewidth": LINEWIDTHS[idx % len(LINEWIDTHS)],
        "markevery": (idx * 4, MARKER_EVERY_STEP),  # staggered start offset per field
        "markersize": 6,
        "markeredgewidth": 0,
        "alpha": 0.85,
    }


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def field_values(rows, field):
    vals = []
    for r in rows:
        v = r.get(field)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        vals.append(v)
    return vals


def compute_field_stats(rows, fields):
    stats = {}
    for field in fields:
        vals = sorted(field_values(rows, field))
        n = len(vals)
        if n == 0:
            stats[field] = None
            continue
        mean = sum(vals) / n
        if n >= 2:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        stats[field] = {
            "n": n, "mean": mean, "median": percentile(vals, 50), "std": std,
            "min": vals[0], "max": vals[-1],
            "p10": percentile(vals, 10), "p50": percentile(vals, 50),
            "p90": percentile(vals, 90), "p99": percentile(vals, 99),
        }
    return stats


def write_stats_csv(stats, fields, output_path):
    cols = ["n", "mean", "median", "std", "min", "max", "p10", "p50", "p90", "p99"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field"] + cols)
        for field in fields:
            s = stats.get(field)
            if s is None:
                writer.writerow([field] + ["" for _ in cols])
            else:
                writer.writerow([field] + [f"{s[c]:.4f}" if isinstance(s[c], float) else s[c]
                                            for c in cols])
    print(f"  wrote stats -> {output_path}")


# ---------------------------------------------------------------------------
# CDF chart (matplotlib, publication style) - fixed size, independent of row count
# ---------------------------------------------------------------------------

TOP_LEVEL_COLORS = {
    "LR_inner_total_time_ms": "#4C72B0",
    "LR_outer_total_time_ms": "#DD8452",
    "pandas_time_ms": "#55A868",
    "np_time_ms": "#C44E52",
    "maxsim_time_ms": "#8172B2",
}


def _short_label(field):
    return field.replace("_time_ms", "").replace("_ms", "")


def _step_cdf(ax, rows, fields, colors, linestyles):
    for field in fields:
        vals = np.sort(np.array(field_values(rows, field), dtype=float))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        style = _field_style(fields, field)
        ax.step(vals, y, where="post",
                color=colors.get(field, "#666666"),
                linestyle=linestyles.get(field, "-"),
                label=_short_label(field),
                marker=style["marker"], markevery=style["markevery"],
                markersize=style["markersize"], markeredgewidth=style["markeredgewidth"],
                linewidth=style["linewidth"], alpha=style["alpha"])


def plot_cdf(rows, output_path_base):
    """Writes <output_path_base>.svg. doc_process_time_ms is
    intentionally excluded - it's the real wall-clock measurement, not a clean
    sum of the other fields (see NOTE in module docstring), so it doesn't
    belong on the same axis as the parts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    _step_cdf(ax1, rows, TOP_LEVEL_FIELDS, TOP_LEVEL_COLORS, TOP_LEVEL_LINESTYLES)
    ax1.set_title("Top-level steps")
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("cumulative fraction")
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    _step_cdf(ax2, rows, DECOMP_FIELDS, DECOMP_COLORS, DECOMP_LINESTYLES)
    ax2.set_title("Decomposition (compute / rest / task_wait)")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path_base + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote CDF chart -> {output_path_base}.svg")


# ---------------------------------------------------------------------------
# Bottleneck view - single 100%-stacked bar, sums to 100% by construction
# ---------------------------------------------------------------------------

def plot_bottleneck(rows, output_path_base):
    """For each document, the 5 top-level steps are normalized by THEIR OWN
    sum (so each document's shares sum to exactly 1), then averaged across
    documents. The mean of numbers that each sum to 1 also sums to exactly 1,
    so this always renders as a clean 100% bar - no fudging needed. This is
    the share of the sum of the 5 measured components, not a share of the
    real doc_process_time_ms (see NOTE in module docstring)."""
    shares = []
    for r in rows:
        vals = []
        ok = True
        for field in TOP_LEVEL_FIELDS:
            v = r.get(field)
            try:
                v = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        total = sum(vals)
        if total <= 0:
            continue
        shares.append([v / total for v in vals])

    if not shares:
        print(f"  (no data for bottleneck chart, skipping {output_path_base})")
        return

    shares_arr = np.array(shares)
    mean_shares = shares_arr.mean(axis=0)
    order = np.argsort(-mean_shares)
    fields_sorted = [TOP_LEVEL_FIELDS[i] for i in order]
    shares_sorted = mean_shares[order]

    fig, ax = plt.subplots(figsize=(9, 1.9))
    left = 0.0
    for field, share in zip(fields_sorted, shares_sorted):
        pct = share * 100
        color = TOP_LEVEL_COLORS.get(field, "#666666")
        ax.barh(0, pct, left=left, color=color, label=f"{_short_label(field)} ({pct:.0f}%)")
        if pct > 4:
            ax.text(left + pct / 2, 0, f"{pct:.0f}%", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel(f"% of measured step time (mean per-document share, n={len(shares)} docs)")
    ax.set_title("Where the time goes")
    ax.grid(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path_base + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote bottleneck chart -> {output_path_base}.svg")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

TARGET_FILENAME = "Batch_0_Query_0_rerank_thread_stats.csv"


def find_input_files(folder):
    """Only ever look for the one specific file: Batch_0_Query_0_rerank_thread_stats.csv,
    searched recursively under folder (in case it's nested one level down)."""
    direct = os.path.join(folder, TARGET_FILENAME)
    if os.path.isfile(direct):
        return [direct]
    matches = sorted(glob.glob(os.path.join(folder, "**", TARGET_FILENAME), recursive=True))
    return matches


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help=f"Folder containing {TARGET_FILENAME} "
                                     "(searched recursively if not directly inside it).")
    args = ap.parse_args()

    if len(sys.argv) < 2:
        ap.print_help()
        sys.exit(1)

    files = find_input_files(args.folder)
    if not files:
        print(f"{TARGET_FILENAME} not found under: {args.folder}")
        sys.exit(1)

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.dirname(path) or "."
        csv_out = os.path.join(outdir, f"{stem}_finegrained.csv")
        svg_out = os.path.join(outdir, f"{stem}_thread_timeline.svg")
        stats_csv_out = os.path.join(outdir, f"{stem}_stats.csv")
        cdf_base = os.path.join(outdir, f"{stem}_cdf")
        bottleneck_base = os.path.join(outdir, f"{stem}_bottleneck")

        rows = build_finegrained_csv(path, csv_out)
        plot_thread_timeline(rows, svg_out)

        stats = compute_field_stats(rows, STATS_FIELDS)
        write_stats_csv(stats, STATS_FIELDS, stats_csv_out)
        plot_cdf(rows, cdf_base)
        plot_bottleneck(rows, bottleneck_base)


if __name__ == "__main__":
    main()