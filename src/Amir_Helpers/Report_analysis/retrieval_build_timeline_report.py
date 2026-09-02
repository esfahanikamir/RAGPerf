"""
End-to-end: reads Batch_0_Query_0_retrieval_finegrained.csv and produces
both the intermediate CSVs and the final SVG plots, all from one script.

--- CSV stage ---
1. timeline_segments.csv -- one row per (thread, token, stage) segment,
   with REAL absolute start/end times. Includes a "setup" segment
   (Thread_start_time -> open_tbl_end) per thread, plus the four stage
   segments per token, placed via cumulative summation of each token's
   own exclusive stage times, starting at that token's real search_start.

2. amount_loaded_by_time.csv -- one row per token, sorted by real
   search_start (chronological), with each split's bytes_read as its
   own column.

--- SVG stage ---
3. retrieval_timeline_by_thread.svg -- per-thread Gantt-style timeline,
   real wall-clock x-axis, stage segments colored and labeled by token.

4. amount_loaded_<Split>.svg (one per split: ANNIvfPartition, ANNSubIndex,
   LanceRead) -- bar chart of bytes_read per token, chronological order.

Usage:
    python3 build_timeline_report.py <folder>
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


STAGE_TIME_COLS = ["ANNIVF_total_time", "ANNSub_total_time", "Sort_total_time", "LanceRead_total_time"]
STAGE_NAMES = ["ANNIvfPartition", "ANNSubIndex", "SortExec", "LanceRead"]

STAGE_COLORS = {
    'setup':            '#e8a33d',
    'ANNIvfPartition':  '#2166ac',
    'ANNSubIndex':      '#b2182b',
    'SortExec':         '#999999',
    'LanceRead':        '#e6ac00',
}


# ============================================================
# CSV stage
# ============================================================

def build_timeline_segments(fine):
    """
    Returns a long-format dataframe: one row per (thread, token-or-setup,
    stage) segment, with real absolute start_ns/end_ns and derived
    start_ms/end_ms/duration_ms.
    """
    rows = []

    for thread_num, group in fine.groupby("thread_num"):
        group = group.sort_values("search_start")

        thread_start = group["Thread_start_time"].iloc[0]
        open_tbl_end = group["open_tbl_end"].iloc[0]
        rows.append({
            "thread_num": thread_num, "token_id": None, "stage": "setup",
            "start_ns": thread_start, "end_ns": open_tbl_end,
        })

        for _, r in group.iterrows():
            t = r["search_start"]
            for time_col, stage_name in zip(STAGE_TIME_COLS, STAGE_NAMES):
                dur_ns = r[time_col] * 1e6
                rows.append({
                    "thread_num": thread_num, "token_id": r["token_id"], "stage": stage_name,
                    "start_ns": t, "end_ns": t + dur_ns,
                })
                t += dur_ns

    seg = pd.DataFrame(rows)
    t0 = seg["start_ns"].min()
    seg["start_ms"] = (seg["start_ns"] - t0) / 1e6
    seg["end_ms"] = (seg["end_ns"] - t0) / 1e6
    seg["duration_ms"] = seg["end_ms"] - seg["start_ms"]
    return seg


def build_amount_loaded_by_time(fine):
    """
    One row per token, sorted by real chronological order (search_start),
    with each split's bytes_read as its own column.
    """
    m = fine.sort_values("search_start").reset_index(drop=True)
    return m[[
        "token_id", "thread_num", "search_start",
        "ANNIVF_bytes_read", "ANNSub_bytes_read", "LanceRead_bytes_read",
    ]].copy()


def generate_csvs(finegrained_csv, output_dir):
    """Runs the CSV stage end to end, saves both files, returns fine + both dataframes."""
    fine = pd.read_csv(finegrained_csv)

    seg = build_timeline_segments(fine)
    seg_path = f"{output_dir}/timeline_segments.csv"
    seg.to_csv(seg_path, index=False)
    print(f"Saved {len(seg)} segments to {seg_path}")

    loaded = build_amount_loaded_by_time(fine)
    loaded_path = f"{output_dir}/amount_loaded_by_time.csv"
    loaded.to_csv(loaded_path, index=False)
    print(f"Saved {len(loaded)} tokens to {loaded_path}")

    return fine, seg, loaded


# ============================================================
# SVG stage
# ============================================================

def plot_timeline_by_thread(seg, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))

    thread_start = seg[seg['stage'] == 'setup'].set_index('thread_num')['start_ms']
    thread_ids_sorted = thread_start.sort_values().index.tolist()
    thread_to_row = {tid: i for i, tid in enumerate(thread_ids_sorted)}

    for _, r in seg.iterrows():
        row = thread_to_row[r['thread_num']]
        ax.barh(row, r['duration_ms'], left=r['start_ms'],
                color=STAGE_COLORS[r['stage']], height=0.7)

    token_rows = seg.dropna(subset=['token_id'])
    for (thread_num, token_id), grp in token_rows.groupby(['thread_num', 'token_id']):
        row = thread_to_row[thread_num]
        start = grp['start_ms'].min()
        end = grp['end_ms'].max()
        width = end - start
        mid = (start + end) / 2
        label = f"Token {int(token_id)}"
        if width < 60:
            ax.text(end + 5, row, label, ha='left', va='center',
                    fontsize=7, color='black', fontweight='bold')
        else:
            ax.text(mid, row, label, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    ax.set_yticks(list(thread_to_row.values()))
    ax.set_yticklabels([f"thread {i}" for i in range(len(thread_ids_sorted))])
    ax.invert_yaxis()
    ax.set_xlabel('real wall-clock time (ms, relative to earliest event)')
    ax.set_title('Real per-thread timeline: stage breakdown at true chronological position')

    legend_elems = [mpatches.Patch(color=c, label=l) for l, c in STAGE_COLORS.items()]
    ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, format='svg')
    plt.close(fig)


def plot_amount_loaded(loaded, output_dir):
    stage_specs = [
        ('ANNIVF_bytes_read',    'ANNIvfPartition', '#2166ac'),
        ('ANNSub_bytes_read',    'ANNSubIndex',     '#b2182b'),
        ('LanceRead_bytes_read', 'LanceRead',       '#e6ac00'),
    ]

    saved_paths = []
    for col, label, color in stage_specs:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(loaded)), loaded[col], color=color, edgecolor='k')
        ax.set_xticks(range(len(loaded)))
        ax.set_xticklabels([f"T{int(t)}" for t in loaded['token_id']], rotation=90, fontsize=7)
        ax.set_xlabel('tokens, sorted by real time (earliest -> latest)')
        ax.set_ylabel(f'{label} bytes_read')
        ax.set_title(f'{label}: amount loaded per token, over real chronological order')
        plt.tight_layout()

        out_path = f"{output_dir}/amount_loaded_{label}.svg"
        plt.savefig(out_path, format='svg')
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def plot_time_efficiency(fine, output_dir):
    """
    Per split: two stacked panels, tokens sorted by real chronological
    order (search_start) --
      top    = time_spent_on_split / bytes_read  (us per KB)
      bottom = time_spent_on_split / requests    (ms per request)
    Expectation: these should be roughly flat/balanced across tokens if
    the stage's cost is a stable function of its own data/request count.
    A rising trend, scattered spikes, or a wave pattern all indicate a
    real imbalance worth explaining (contention, specific unlucky reads,
    round-based scheduling effects, etc.) -- see the write-up for what
    each of the three splits actually shows.
    """
    m = fine.sort_values('search_start').reset_index(drop=True)

    specs = [
        ('ANNIVF_total_time', 'ANNIVF_bytes_read', 'ANNIVF_requests', 'ANNIvfPartition', '#2166ac'),
        ('ANNSub_total_time', 'ANNSub_bytes_read', 'ANNSub_requests', 'ANNSubIndex', '#b2182b'),
        ('LanceRead_total_time', 'LanceRead_bytes_read', 'LanceRead_requests', 'LanceRead', '#e6ac00'),
    ]

    saved_paths = []
    for time_col, bytes_col, req_col, label, color in specs:
        # avoid division by zero: rows with nothing transferred stay blank (NaN), not zero
        time_per_byte = np.where(m[bytes_col] > 0, (m[time_col] * 1000) / (m[bytes_col] / 1024), np.nan)
        time_per_req = np.where(m[req_col] > 0, m[time_col] / m[req_col], np.nan)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        ax1.bar(range(len(m)), time_per_byte, color=color, edgecolor='k')
        ax1.set_ylabel('time / byte (\u00b5s/KB)')
        ax1.set_title(f'{label}: time-efficiency per token, sorted by real chronological order')

        ax2.bar(range(len(m)), time_per_req, color=color, edgecolor='k', alpha=0.7)
        ax2.set_ylabel('time / request (ms)')
        ax2.set_xlabel('tokens, sorted by real time (earliest -> latest)')
        ax2.set_xticks(range(len(m)))
        ax2.set_xticklabels([f"T{int(t)}" for t in m['token_id']], rotation=90, fontsize=7)

        plt.tight_layout()
        out_path = f"{output_dir}/time_efficiency_{label}.svg"
        plt.savefig(out_path, format='svg')
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def generate_svgs(fine, seg, loaded, output_dir):
    """Runs the SVG stage end to end, saves all files, prints their paths."""
    timeline_path = f"{output_dir}/retrieval_timeline_by_thread.svg"
    plot_timeline_by_thread(seg, timeline_path)
    print(f"Saved: {timeline_path}")

    for p in plot_amount_loaded(loaded, output_dir):
        print(f"Saved: {p}")

    for p in plot_time_efficiency(fine, output_dir):
        print(f"Saved: {p}")


# ============================================================
# Entry point
# ============================================================

def main(folder):
    finegrained_csv = f"{folder}/Batch_0_Query_0_retrieval_finegrained.csv"
    fine, seg, loaded = generate_csvs(finegrained_csv, folder)
    generate_svgs(fine, seg, loaded, folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_timeline_report.py <folder>")
        sys.exit(1)
    main(sys.argv[1])