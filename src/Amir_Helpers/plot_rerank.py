"""
RAGPerf Reranking Profiler – plot_rerank.py
============================================
Usage:
    python plot_rerank.py <output_dir> [--save-dir <dir>]

Examples:
    python plot_rerank.py output/b001_20_08_thread04
    python plot_rerank.py output/b001_20_08_thread04 --save-dir figs/b001_20_08_thread04

    # Compare all experiments at once (pass multiple dirs):
    python plot_rerank.py output/b0001_10_04_thread01 output/b0001_20_04_thread01 \
                          output/b0001_20_08_thread01 --save-dir figs/comparison

Output SVGs (per-experiment mode):
    01_cdf_total_rerank.svg          – CDF of total_rerank_time_ms
    02_cdf_stage_breakdown.svg       – CDF of each reranking sub-stage
    03_boxplot_stages.svg            – Box-plot of sub-stage latencies
    04_stacked_bar_stages.svg        – Mean stage contribution per query
    05_scatter_patches_total.svg     – num_patches vs total_rerank_time_ms
    06_thread_comparison_cdf.svg     – Per-thread CDF (multi-thread runs only)

Output SVGs (comparison mode, ≥2 dirs):
    cmp_cdf_total.svg                – CDFs of total_rerank_time_ms across configs
    cmp_boxplot_stages.svg           – Stage box-plots side by side
    cmp_mean_stack.svg               – Mean stage stacks per config
"""

import argparse
import sys
import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STAGE_COLS = [
    "open_table_time_ms",
    "lazy_search_time_ms",
    "db_fetch_pure_time_ms",
    "pandas_time_ms",
    "numpy_time_ms",
    "dotp_time_ms",
]
STAGE_LABELS = {
    "open_table_time_ms":   "Open table",
    "lazy_search_time_ms":  "Lazy search",
    "db_fetch_pure_time_ms":"DB fetch (pure)",
    "pandas_time_ms":       "Pandas conv.",
    "numpy_time_ms":        "NumPy prep",
    "dotp_time_ms":         "Dot product",
}
STAGE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759",
    "#76b7b2", "#59a14f", "#edc948",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_dir(dirpath: str) -> pd.DataFrame:
    """Load all Batch_*_Query_*_stats.csv from a directory into one DataFrame.

    Adds a '_source_file' column (basename without extension) so callers can
    determine which rows came from the same CSV — i.e. which threads were
    truly concurrent (within one file) vs. sequential (across files).
    """
    csvs = sorted(glob.glob(os.path.join(dirpath, "Batch_*_Query_*_stats.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {dirpath}")
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f, sep=None, engine="python")
            df.columns = df.columns.str.strip()
            df["_source_file"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(df)
        except Exception as e:
            print(f"  [warn] skipping {f}: {e}", file=sys.stderr)
    if not frames:
        raise ValueError(f"All CSVs failed to parse in: {dirpath}")
    out = pd.concat(frames, ignore_index=True)
    for col in STAGE_COLS + ["total_rerank_time_ms", "num_patches", "query_tokens"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def max_concurrent_threads(df: pd.DataFrame) -> int:
    """Return the maximum number of distinct thread_ids seen in a single CSV file.

    This is the true thread concurrency: OS thread IDs differ across sequential
    single-thread runs (each query may be handled by a different worker thread),
    so a global unique-count would be misleading.
    """
    if "thread_id" not in df.columns or "_source_file" not in df.columns:
        return 1
    return int(df.groupby("_source_file")["thread_id"].nunique().max())


def parse_config(dirpath: str) -> dict:
    """Extract config from folder name like b001_20_08_thread04."""
    name = Path(dirpath).name
    m = re.match(r"b(\d+)_(\d+)_(\d+)_thread(\d+)", name)
    if m:
        ratio  = float("0." + m.group(1))
        top_k  = int(m.group(2))
        top_n  = int(m.group(3))
        nthrd  = int(m.group(4))
        return dict(ratio=ratio, top_k=top_k, top_n=top_n, threads=nthrd, name=name)
    return dict(name=name)


def short_label(cfg: dict) -> str:
    if "top_k" in cfg:
        return f"k={cfg['top_k']},n={cfg['top_n']},t={cfg['threads']},db={cfg['ratio']}"
    return cfg["name"]


def cdf(values):
    """Return (sorted_values, cumulative_probs) for a CDF."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    p = np.arange(1, len(v) + 1) / len(v)
    return v, p


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── single-experiment plots ────────────────────────────────────────────────────

def plot_cdf_total(df: pd.DataFrame, cfg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    v, p = cdf(df["total_rerank_time_ms"])
    ax.plot(v, p, lw=2, color="#4e79a7")
    ax.axvline(np.median(v), color="red", ls="--", lw=1, label=f"median={np.median(v):.1f} ms")
    ax.axvline(np.percentile(v, 95), color="orange", ls="--", lw=1,
               label=f"p95={np.percentile(v,95):.1f} ms")
    ax.set_xlabel("total_rerank_time_ms")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF – Total rerank time\n{short_label(cfg)}")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    save(fig, os.path.join(out_dir, "01_cdf_total_rerank.svg"))


def plot_cdf_stages(df: pd.DataFrame, cfg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color in zip(STAGE_COLS, STAGE_COLORS):
        if col not in df.columns:
            continue
        v, p = cdf(df[col])
        ax.plot(v, p, lw=1.8, color=color, label=STAGE_LABELS[col])
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF – Per-stage latency\n{short_label(cfg)}")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    save(fig, os.path.join(out_dir, "02_cdf_stage_breakdown.svg"))


def plot_boxplot_stages(df: pd.DataFrame, cfg: dict, out_dir: str):
    available = [c for c in STAGE_COLS if c in df.columns]
    data = [df[c].dropna().values for c in available]
    labels = [STAGE_LABELS[c] for c in available]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], STAGE_COLORS[:len(available)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"Box-plot – Stage latencies\n{short_label(cfg)}")
    save(fig, os.path.join(out_dir, "03_boxplot_stages.svg"))


def plot_stacked_bar(df: pd.DataFrame, cfg: dict, out_dir: str):
    """Mean stage contribution per (batch, query) pair, stacked bar."""
    available = [c for c in STAGE_COLS if c in df.columns]
    # group by batch_num + query_id → mean per row group (across threads/docs)
    grp = df.groupby(["batch_num", "query_id"])[available].mean().reset_index()
    x = np.arange(len(grp))
    fig, ax = plt.subplots(figsize=(max(8, len(grp) * 0.55), 5))
    bottom = np.zeros(len(grp))
    for col, color in zip(available, STAGE_COLORS):
        vals = grp[col].fillna(0).values
        ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85,
               label=STAGE_LABELS[col], width=0.75)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"B{r.batch_num}\nQ{r.query_id}" for _, r in grp.iterrows()],
        fontsize=8
    )
    ax.set_ylabel("mean time (ms)")
    ax.set_title(f"Stage breakdown per query (mean over docs/threads)\n{short_label(cfg)}")
    ax.legend(loc="upper right", ncol=2)
    save(fig, os.path.join(out_dir, "04_stacked_bar_stages.svg"))


def plot_scatter_patches(df: pd.DataFrame, cfg: dict, out_dir: str):
    if "num_patches" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(df["num_patches"], df["total_rerank_time_ms"],
                    c=df["dotp_time_ms"] if "dotp_time_ms" in df.columns else "steelblue",
                    cmap="viridis", alpha=0.5, s=15)
    if "dotp_time_ms" in df.columns:
        plt.colorbar(sc, ax=ax, label="dotp_time_ms")
    ax.set_xlabel("num_patches per document")
    ax.set_ylabel("total_rerank_time_ms")
    ax.set_title(f"Patches vs total rerank time\n{short_label(cfg)}")
    save(fig, os.path.join(out_dir, "05_scatter_patches_total.svg"))


def plot_thread_cdfs(df: pd.DataFrame, cfg: dict, out_dir: str):
    """Per-thread CDF, but only for experiments where multiple threads truly
    ran concurrently within the same query CSV.

    For thread01 runs the OS may reuse different thread IDs across sequential
    queries, making the global unique-thread-count misleadingly large.  We
    check the maximum concurrent thread count per file instead.
    """
    if "thread_id" not in df.columns or "_source_file" not in df.columns:
        return
    true_concurrency = max_concurrent_threads(df)
    if true_concurrency < 2:
        print(f"   [skip] 06_thread_comparison_cdf – no concurrent threads "
              f"(max per-file unique thread_ids = {true_concurrency})")
        return

    # Collect per-thread samples only from files that had concurrent threads
    # Use a consistent integer rank (0, 1, 2, …) rather than raw OS thread IDs.
    concurrent_files = (
        df.groupby("_source_file")["thread_id"]
        .nunique()
        .pipe(lambda s: s[s >= 2])
        .index
    )
    sub_df = df[df["_source_file"].isin(concurrent_files)].copy()

    # Build a stable per-file rank for each thread_id so legends read "thread 0/1/2"
    rank_map = {}
    for fname, grp in sub_df.groupby("_source_file"):
        for rank, tid in enumerate(sorted(grp["thread_id"].unique())):
            rank_map.setdefault(tid, rank)
    sub_df["_thread_rank"] = sub_df["thread_id"].map(rank_map)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors
    for rank in sorted(sub_df["_thread_rank"].unique()):
        vals = sub_df[sub_df["_thread_rank"] == rank]["total_rerank_time_ms"].dropna()
        if len(vals) == 0:
            continue
        v, p = cdf(vals)
        ax.plot(v, p, lw=1.5, color=colors[rank % 10], label=f"thread {rank}")
    ax.set_xlabel("total_rerank_time_ms")
    ax.set_ylabel("CDF")
    ax.set_title(
        f"Per-thread CDF  (concurrent threads = {true_concurrency})\n{short_label(cfg)}"
    )
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    save(fig, os.path.join(out_dir, "06_thread_comparison_cdf.svg"))


def run_single(dirpath: str, out_dir: str):
    print(f"\n▶  Processing: {dirpath}")
    os.makedirs(out_dir, exist_ok=True)
    df = load_dir(dirpath)
    cfg = parse_config(dirpath)
    true_conc = max_concurrent_threads(df)
    print(f"   rows={len(df)}, max_concurrent_threads_per_query={true_conc}")
    plot_cdf_total(df, cfg, out_dir)
    plot_cdf_stages(df, cfg, out_dir)
    plot_boxplot_stages(df, cfg, out_dir)
    plot_stacked_bar(df, cfg, out_dir)
    plot_scatter_patches(df, cfg, out_dir)
    plot_thread_cdfs(df, cfg, out_dir)


# ── multi-experiment comparison plots ─────────────────────────────────────────

def plot_cmp_cdf_total(datasets: list, out_dir: str):
    """CDF of total_rerank_time_ms for each config on one axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (df, cfg) in enumerate(datasets):
        v, p = cdf(df["total_rerank_time_ms"])
        ax.plot(v, p, lw=2, color=colors[i % 10], label=short_label(cfg))
    ax.set_xlabel("total_rerank_time_ms")
    ax.set_ylabel("CDF")
    ax.set_title("CDF comparison – Total rerank time")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    save(fig, os.path.join(out_dir, "cmp_cdf_total.svg"))


def plot_cmp_boxplot_stages(datasets: list, out_dir: str):
    """Side-by-side box-plots for each stage, grouped by config."""
    available = [c for c in STAGE_COLS
                 if all(c in df.columns for df, _ in datasets)]
    n_stages = len(available)
    n_cfgs = len(datasets)
    fig, axes = plt.subplots(1, n_stages, figsize=(3.5 * n_stages, 5), sharey=False)
    if n_stages == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for ax, col in zip(axes, available):
        data = [df[col].dropna().values for df, _ in datasets]
        labels = [short_label(cfg) for _, cfg in datasets]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, color in zip(bp["boxes"], colors[:n_cfgs]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(STAGE_LABELS[col], fontsize=10)
        ax.set_xticks(range(1, n_cfgs + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ms")
    fig.suptitle("Stage latency comparison", y=1.01)
    save(fig, os.path.join(out_dir, "cmp_boxplot_stages.svg"))


def plot_cmp_mean_stack(datasets: list, out_dir: str):
    """Stacked bar of mean stage times per config."""
    available = [c for c in STAGE_COLS
                 if all(c in df.columns for df, _ in datasets)]
    labels = [short_label(cfg) for _, cfg in datasets]
    means = {col: [df[col].mean() for df, _ in datasets] for col in available}
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.5), 5))
    bottom = np.zeros(len(datasets))
    for col, color in zip(available, STAGE_COLORS):
        vals = np.array(means[col])
        ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85,
               label=STAGE_LABELS[col], width=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean time (ms)")
    ax.set_title("Mean stage breakdown by configuration")
    ax.legend(loc="upper right", ncol=2)
    save(fig, os.path.join(out_dir, "cmp_mean_stack.svg"))


def run_comparison(dirs: list, out_dir: str):
    print(f"\n▶  Comparison mode: {len(dirs)} experiments → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    datasets = []
    for d in dirs:
        df = load_dir(d)
        cfg = parse_config(d)
        datasets.append((df, cfg))
        print(f"   {cfg['name']}: {len(df)} rows")
    plot_cmp_cdf_total(datasets, out_dir)
    plot_cmp_boxplot_stages(datasets, out_dir)
    plot_cmp_mean_stack(datasets, out_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGPerf reranking profiler plots")
    parser.add_argument("dirs", nargs="+", help="One or more experiment output directories")
    parser.add_argument("--save-dir", default=None,
                        help="Where to save SVGs (default: inside each input dir)")
    args = parser.parse_args()

    if len(args.dirs) == 1:
        d = args.dirs[0]
        out = args.save_dir if args.save_dir else d
        run_single(d, out)
    else:
        if args.save_dir:
            # comparison mode: all experiments go to one dir
            run_comparison(args.dirs, args.save_dir)
        else:
            # per-experiment mode + comparison in a sibling folder
            for d in args.dirs:
                run_single(d, d)
            parent = str(Path(args.dirs[0]).parent)
            cmp_dir = os.path.join(parent, "comparison")
            run_comparison(args.dirs, cmp_dir)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
