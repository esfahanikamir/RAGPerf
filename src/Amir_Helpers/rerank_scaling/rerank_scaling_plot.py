import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_ROOT   = "/local/amirk/RAGPerf/src/output"
FOLDER_PREFIX = "physics_second_"
FOLDER_SUFFIX = "_gpuembed"   # set to "" if folders don't have this suffix

CSV_OUT_DIR  = "rerank_scaling_csvs"
PLOT_OUT_DIR = "rerank_scaling_plots"

THREAD_COL    = "Threads"
THREAD_ID_COL = "thread_id"

# Timing metrics for flatness plots (per-document, should be flat across thread counts)
PER_DOC_METRICS = {
    "data_fetch_time_ms":    "DataFetch (stage)",
    "open_table_time_ms":    "open_table (substage)",
    "lazy_search_time_ms":   "lazy_search (substage)",
    "db_fetch_pure_time_ms": "db_fetch_pure (substage)",
    "pandas_time_ms":        "pandas (substage)",
    "numpy_time_ms":         "Numpy (stage)",
    "dotp_time_ms":          "DotProduct (stage)",
    "total_rerank_time_ms":  "Total Rerank per-doc",
}

# Columns to keep (drops MB memory-tracking columns)
KEEP_COLS = [
    "batch_num", "query_id", "thread_id", "doc_id",
    "num_patches", "query_tokens",
    "data_fetch_time_ms", "open_table_time_ms", "lazy_search_time_ms",
    "db_fetch_pure_time_ms", "pandas_time_ms",
    "numpy_time_ms", "dotp_time_ms", "fetched_kb", "total_rerank_time_ms",
    "abs_start", "abs_end",
]

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#e5e5e5",
    "grid.linestyle":   ":",
    "grid.linewidth":   0.8,
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "axes.labelcolor":  "#333333",
    "text.color":       "#333333",
}


# ── Step 1: discover experiment folders ────────────────────────────────────────

def discover_thread_folders(root, prefix, suffix):
    """Return dict {n_threads: folder_path} for all matching experiment folders."""
    folders = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)T{re.escape(suffix)}$")
    for name in os.listdir(root):
        m = pattern.match(name)
        if m:
            n = int(m.group(1))
            folders[n] = os.path.join(root, name)
    return dict(sorted(folders.items()))


# ── Step 2: load all CSVs ──────────────────────────────────────────────────────

def load_all_data(thread_folders):
    """
    Read every Batch_*_Query_*_stats.csv from every thread folder.
    Returns one concatenated DataFrame with an added 'Threads' column.
    """
    frames = []
    for n_threads, folder in thread_folders.items():
        csvs = sorted(f for f in os.listdir(folder) if f.endswith("0_stats.csv"))
        if not csvs:
            print(f"WARNING: no CSVs found in {folder}")
            continue
        for csv_name in csvs:
            path = os.path.join(folder, csv_name)
            df = pd.read_csv(path)
            # drop malformed/empty rows
            df = df.dropna(subset=["doc_id"])
            # keep only relevant columns that exist
            cols = [c for c in KEEP_COLS if c in df.columns]
            df = df[cols].copy()
            df[THREAD_COL] = n_threads
            frames.append(df)
        print(f"  Loaded {len(csvs)} CSVs  ←  {n_threads}T folder")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows : {len(combined)}")
    print(f"Thread counts : {sorted(combined[THREAD_COL].unique())}")
    bq = sorted(combined.groupby(["batch_num", "query_id"]).groups.keys())
    print(f"(batch, query) pairs : {bq}\n")
    return combined


# ── Step 3: save aggregated CSVs ───────────────────────────────────────────────

def save_aggregated_csvs(df, out_dir):
    """
    For each (batch_num, query_id) write one CSV containing all per-document
    rows across all thread counts, sorted by Threads then doc_id.
    Returns dict {(batch, query): sub-DataFrame}.
    """
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for (batch, query), grp in df.groupby(["batch_num", "query_id"]):
        grp = grp.sort_values([THREAD_COL, "doc_id"]).reset_index(drop=True)
        fname = f"batch{int(batch)}_query{int(query)}_aggregated.csv"
        grp.to_csv(os.path.join(out_dir, fname), index=False)
        groups[(batch, query)] = grp
        print(f"  Saved: {fname}  "
              f"({len(grp)} rows, "
              f"threads={sorted(grp[THREAD_COL].unique())})")
    return groups


# ── Step 4: wall-clock reconstruction (exact, using abs timestamps) ────────────

def reconstruct_wall_clock(grp):
    """
    For each thread count:
        wall_clock_ms = (max abs_end  -  min abs_start) / 1e6
    This is exact — no assumptions about thread arrangement needed.
    Requires abs_start and abs_end columns (raw nanoseconds from time.monotonic_ns).
    """
    has_abs = ("abs_start" in grp.columns and "abs_end" in grp.columns)
    records = {}
    for n_threads, tgrp in grp.groupby(THREAD_COL):
        tgrp = tgrp.dropna(subset=(["abs_start", "abs_end"] if has_abs
                                    else ["total_rerank_time_ms"]))
        if has_abs:
            wall_clock_ms = (tgrp["abs_end"].max() - tgrp["abs_start"].min()) / 1e6
        else:
            # fallback if abs columns are missing (old CSVs without timestamps)
            print(f"  WARNING: abs_start/abs_end missing for {n_threads}T — "
                  f"falling back to max(sum per thread), which may be inaccurate.")
            thread_totals = tgrp.groupby(THREAD_ID_COL)["total_rerank_time_ms"].sum()
            wall_clock_ms = thread_totals.max()
        records[n_threads] = wall_clock_ms
    return pd.Series(records).sort_index()


# ── Step 5: strong scaling plot ────────────────────────────────────────────────

def plot_strong_scaling(wall_clock_series, batch, query, out_dir):
    thread_counts = wall_clock_series.index.values.astype(int)
    wall_clocks   = wall_clock_series.values
    baseline_wc   = wall_clocks[0]
    speedup       = baseline_wc / wall_clocks
    ideal         = thread_counts / thread_counts[0]

    # detect over-provisioning point (n_threads >= n_docs has no more gain)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(thread_counts, ideal, "k--", linewidth=1.5, label="Ideal (linear)")
        ax.plot(thread_counts, speedup, marker="o", linewidth=2,
                color="#2a78d6", markersize=7, label="Measured speedup")

        for n, s in zip(thread_counts, speedup):
            ax.annotate(f"{s:.2f}×", xy=(n, s), xytext=(4, 6),
                        textcoords="offset points", fontsize=9, color="#2a78d6")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(thread_counts)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Number of threads (N)")
        ax.set_ylabel("Speedup  =  wall_clock(1T) / wall_clock(NT)")
        ax.set_title(f"Strong scaling — exact wall-clock rerank time\n"
                     f"Batch {batch}, Query {query}")
        ax.legend()
        fig.tight_layout()

        fname = os.path.join(out_dir, "strong_scaling_wall_clock.svg")
        fig.savefig(fname, format="svg")
        plt.close(fig)
        print(f"  Saved: {os.path.relpath(fname)}")

    # print summary table
    print(f"  {'Threads':>8}  {'wall_clock_ms':>14}  {'speedup':>8}")
    for n, wc, s in zip(thread_counts, wall_clocks, speedup):
        print(f"  {n:>8}  {wc:>14.2f}  {s:>8.3f}×")


# ── Step 6: flatness plots (per-document time should not change with N) ────────

def plot_flatness(grp, col, label, batch, query, out_dir):
    """
    Box plot of per-document time distributions for each thread count.
    Flat boxes = clean parallelism. Growing boxes = contention/overhead.
    """
    thread_counts  = sorted(grp[THREAD_COL].unique())
    data_by_thread = [grp.loc[grp[THREAD_COL] == n, col].dropna().values
                      for n in thread_counts]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.boxplot(
            data_by_thread,
            labels=[str(n) for n in thread_counts],
            patch_artist=True,
            medianprops=dict(color="#e34948", linewidth=2),
            boxprops=dict(facecolor="#e6f1fb", color="#185fa5", linewidth=1),
            whiskerprops=dict(color="#185fa5", linewidth=1),
            capprops=dict(color="#185fa5", linewidth=1.5),
            flierprops=dict(marker=".", color="#aaaaaa", markersize=4, alpha=0.5),
        )

        means = [np.mean(d) if len(d) > 0 else np.nan for d in data_by_thread]
        ax.plot(range(1, len(thread_counts) + 1), means,
                marker="D", color="#eda100", markersize=5,
                linewidth=1.2, linestyle="--", label="mean", zorder=5)

        ax.set_xlabel("Number of threads (N)")
        ax.set_ylabel("Time per document (ms)")
        ax.set_title(f"Per-document time vs thread count — {label}\n"
                     f"Batch {batch}, Query {query}  (expect flat)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        fname = os.path.join(out_dir, f"flatness_{col.replace('_time_ms','')}.svg")
        fig.savefig(fname, format="svg")
        plt.close(fig)
        print(f"  Saved: {os.path.relpath(fname)}")


# ── Step 7: Gantt chart (thread execution timeline) ────────────────────────────

def plot_gantt(grp, n_threads, batch, query, out_dir):
    """
    One horizontal bar per document, positioned by real abs_start / abs_end.
    Each unique thread_id gets its own y-position and color.
    Shows: are threads actually overlapping? Is there GIL serialization?
    Is there load imbalance (one thread finishing early)?
    """
    if "abs_start" not in grp.columns or "abs_end" not in grp.columns:
        print(f"  Skipping Gantt for {n_threads}T — no abs_start/abs_end columns")
        return

    tgrp = grp[grp[THREAD_COL] == n_threads].dropna(
        subset=["abs_start", "abs_end"]).copy()
    if tgrp.empty:
        return

    # normalize to ms relative to the very first doc start in this call
    t0_global = tgrp["abs_start"].min()
    tgrp["start_ms"] = (tgrp["abs_start"] - t0_global) / 1e6
    tgrp["end_ms"]   = (tgrp["abs_end"]   - t0_global) / 1e6
    tgrp["dur_ms"]   = tgrp["end_ms"] - tgrp["start_ms"]

    # assign compact thread index for y-axis (0, 1, 2, ...)
    thread_ids  = sorted(tgrp["thread_id"].unique())
    tid_to_idx  = {tid: i for i, tid in enumerate(thread_ids)}
    tgrp["thread_idx"] = tgrp["thread_id"].map(tid_to_idx)
    n_unique = len(thread_ids)

    # color map — wrap around tab20 for large thread counts
    cmap = plt.cm.get_cmap("tab20", 20)

    with plt.rc_context(PLOT_STYLE):
        bar_h      = max(0.5, min(0.9, 40 / max(n_unique, 1)))
        fig_height = max(4, n_unique * (bar_h + 0.15) + 1.5)
        fig, ax    = plt.subplots(figsize=(12, fig_height))

        for _, row in tgrp.iterrows():
            ax.barh(
                y=row["thread_idx"],
                width=row["dur_ms"],
                left=row["start_ms"],
                height=bar_h,
                color=cmap(int(row["thread_idx"]) % 20),
                alpha=0.85,
                edgecolor="none",
            )

        # true wall-clock line
        wall_clock_ms = tgrp["end_ms"].max()
        ax.axvline(wall_clock_ms, color="#e34948", linestyle="--",
                   linewidth=1.5, label=f"wall-clock = {wall_clock_ms:.1f} ms")

        ax.set_xlabel("Time (ms, relative to first doc start)")
        ax.set_ylabel("Thread index")
        ax.set_yticks(range(n_unique))
        ax.set_yticklabels([str(i) for i in range(n_unique)], fontsize=8)
        ax.set_title(
            f"Thread execution timeline — {n_threads}T  |  "
            f"Batch {batch}, Query {query}\n"
            f"{n_unique} unique threads, {len(tgrp)} docs, "
            f"wall-clock = {wall_clock_ms:.1f} ms"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()

        fname = os.path.join(out_dir, f"gantt_{n_threads}T.svg")
        fig.savefig(fname, format="svg")
        plt.close(fig)
        print(f"  Saved: {os.path.relpath(fname)}")


# ── Step 8: summary wall-clock table across all (batch, query) ────────────────

def print_and_save_summary(all_wall_clocks, out_dir):
    rows = []
    for (batch, query), wc_series in sorted(all_wall_clocks.items()):
        baseline = wc_series.iloc[0]
        for n_threads, ms in wc_series.items():
            rows.append({
                "batch":          int(batch),
                "query":          int(query),
                "threads":        int(n_threads),
                "wall_clock_ms":  round(ms, 3),
                "speedup":        round(baseline / ms, 4),
            })
    summary_df = pd.DataFrame(rows)

    print("\n── Wall-clock summary ──────────────────────────────────────────")
    print(summary_df.to_string(index=False))

    fname = os.path.join(out_dir, "wall_clock_summary.csv")
    summary_df.to_csv(fname, index=False)
    print(f"\nSummary saved: {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. discover folders
    print("── Discovering experiment folders ──────────────────────────────")
    thread_folders = discover_thread_folders(OUTPUT_ROOT, FOLDER_PREFIX, FOLDER_SUFFIX)
    print(f"Found {len(thread_folders)} configurations: "
          f"{list(thread_folders.keys())}\n")

    # 2. load data
    print("── Loading CSVs ─────────────────────────────────────────────────")
    df = load_all_data(thread_folders)

    # 3. save aggregated CSVs
    print("── Saving aggregated CSVs ───────────────────────────────────────")
    groups = save_aggregated_csvs(df, CSV_OUT_DIR)

    # 4. plot per (batch, query)
    all_wall_clocks = {}
    thread_counts_global = sorted(df[THREAD_COL].unique())

    for (batch, query), grp in groups.items():
        label    = f"batch{int(batch)}_query{int(query)}"
        plot_out = os.path.join(PLOT_OUT_DIR, label)
        os.makedirs(plot_out, exist_ok=True)

        print(f"\n── {label} ─────────────────────────────────────────────────")

        # 4a. strong scaling (exact wall-clock)
        wc_series = reconstruct_wall_clock(grp)
        all_wall_clocks[(batch, query)] = wc_series
        plot_strong_scaling(wc_series, batch, query, plot_out)

        # 4b. flatness plots (one per metric)
        for col, metric_label in PER_DOC_METRICS.items():
            if col in grp.columns:
                plot_flatness(grp, col, metric_label, batch, query, plot_out)

        # 4c. Gantt charts (one per thread count)
        for n_threads in thread_counts_global:
            plot_gantt(grp, n_threads, batch, query, plot_out)

    # 5. summary
    print("\n── Summary ──────────────────────────────────────────────────────")
    print_and_save_summary(all_wall_clocks, CSV_OUT_DIR)

    # 6. report counts
    n_groups      = len(groups)
    n_threads_cfg = len(thread_counts_global)
    n_per_group   = (
        1                          # strong scaling
        + len(PER_DOC_METRICS)     # flatness plots
        + n_threads_cfg            # gantt per thread count
    )
    print(f"\n── Done ─────────────────────────────────────────────────────────")
    print(f"Aggregated CSVs : {CSV_OUT_DIR}/  ({n_groups} files + 1 summary)")
    print(f"Plot folders    : {PLOT_OUT_DIR}/  ({n_groups} subfolders)")
    print(f"Plots per folder: 1 scaling + {len(PER_DOC_METRICS)} flatness "
          f"+ {n_threads_cfg} Gantt = {n_per_group}")
    print(f"Total SVGs      : {n_groups * n_per_group}")