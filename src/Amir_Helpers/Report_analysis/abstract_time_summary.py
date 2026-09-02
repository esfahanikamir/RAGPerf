"""
Extracts abstract-level per-stage timing for one run folder, combining:
  - time_break_down.txt  -> embed, prompt, generate
  - Batch_0_Query_0_rerank_general_stats.csv -> rerank
  - Batch_0_Query_0_retrieval_thread_profile.csv -> retrieval

Assumes, per the current setup: batch_size = 1, single query per run.
"""

import os
import re
import pandas as pd
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_time_breakdown(path):
    """
    Parses "label, timestamp_ns" lines into a dict: {label: int(timestamp_ns)}.
    Ignores blank lines.
    """
    stamps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, ts = line.split(",", 1)
            stamps[label.strip()] = int(ts.strip())
    return stamps


def extract_rerank_time_ms(csv_path):
    """
    Reads the rerank_general_stats.csv single-row file.
    Total rerank time = multithread_time_ms + sort_time_ms.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["multithread_time_ms"])
    if len(df) == 0:
        raise ValueError(f"No valid rows in {csv_path}")
    row = df.iloc[0]
    return row["multithread_time_ms"] + row["sort_time_ms"]


def extract_retrieval_time_ms(csv_path):
    """
    Reads the retrieval_thread_profile.csv.
    Total retrieval time = last search_end across all threads/tasks
                            minus the earliest thread_start.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["thread_start", "search_end"])
    if len(df) == 0:
        raise ValueError(f"No valid rows in {csv_path}")
    span_ns = df["search_end"].max() - df["thread_start"].min()
    return span_ns / 1e6


def extract_run_summary(run_folder, batch_num=0, query_id=0):
    """
    Given one run folder, returns a dict with all 5 stage timings in ms,
    plus the total end-to-end time from the time_break_down.txt file.
    """
    tbd_path = os.path.join(run_folder, "time_break_down.txt")
    rerank_path = os.path.join(
        run_folder, f"Batch_{batch_num}_Query_{query_id}_rerank_general_stats.csv"
    )
    retrieval_path = os.path.join(
        run_folder, f"Batch_{batch_num}_Query_{query_id}_retrieval_thread_profile.csv"
    )

    for p in (tbd_path, rerank_path, retrieval_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing expected file: {p}")

    stamps = parse_time_breakdown(tbd_path)

    embed_ms = (stamps["retrieve"] - stamps["embed"]) / 1e6
    prompt_ms = (stamps["prompt_logs"] - stamps["prompt"]) / 1e6
    generate_ms = (stamps["gen_logs"] - stamps["generate"]) / 1e6

    rerank_ms = extract_rerank_time_ms(rerank_path)
    retrieval_ms = extract_retrieval_time_ms(retrieval_path)

    total_ms = (stamps["done"] - stamps["load_models"]) / 1e6

    return {
        "run_folder": os.path.basename(run_folder.rstrip("/")),
        "embed_ms": embed_ms,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
        "prompt_ms": prompt_ms,
        "generate_ms": generate_ms,
        "total_ms": total_ms,
    }


def plot_summary(summary, run_folder, filename="timing_breakdown.svg"):
    """
    Draws a horizontal bar chart of the 5 tracked stages (embed, retrieval,
    rerank, prompt, generate), labeled with the stage name, its percentage,
    and its raw ms value. Saves an SVG into `run_folder` and returns the
    full output path.

    Percentages are each stage's share of the SUM OF THE 5 STAGES ONLY
    (not total_ms) — i.e. item / (embed_ms + retrieval_ms + rerank_ms +
    prompt_ms + generate_ms). total_ms / any load/teardown overhead is
    ignored entirely, so the 5 bars always add up to 100%.
    """
    stage_keys = ["embed_ms", "retrieval_ms", "rerank_ms", "prompt_ms", "generate_ms"]

    stages = [k.replace("_ms", "") for k in stage_keys]
    values_ms = [summary[k] for k in stage_keys]

    stage_sum_ms = sum(values_ms)
    percentages = [v / stage_sum_ms * 100 for v in values_ms]

    fig_height = 0.6 * len(stages) + 1.5
    fig, ax = plt.subplots(figsize=(8, fig_height))

    colors = plt.cm.viridis(
        [i / max(len(stages) - 1, 1) for i in range(len(stages))]
    )
    bars = ax.barh(stages, percentages, color=colors)
    ax.invert_yaxis()  # keep embed/retrieval/rerank/prompt/generate top-down
    ax.set_xlabel("% of tracked stage time (embed+retrieval+rerank+prompt+generate)")
    ax.set_xlim(0, max(percentages) * 1.2)
    ax.set_title(f"Stage timing breakdown — {summary['run_folder']}")

    max_pct = max(percentages)
    for bar, pct, ms in zip(bars, percentages, values_ms):
        ax.text(
            bar.get_width() + max_pct * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}% ({ms:.1f} ms)",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = os.path.join(run_folder, filename)
    fig.savefig(output_path, format="svg")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # quick single-folder test
    if len(sys.argv) < 2:
            print("Usage: python3 abstract_time_summary.py <folder>")
            sys.exit(1)

    folder = sys.argv[1]
    summary = extract_run_summary(folder)
    for k, v in summary.items():
        print(f"{k:15s}: {v}")

    svg_path = plot_summary(summary, folder)
    print(f"\nSaved plot to: {svg_path}")