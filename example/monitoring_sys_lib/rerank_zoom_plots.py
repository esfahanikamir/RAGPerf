"""
rerank_zoom_plots.py — zoomed rerank figures: per-task SUB-PHASE boundaries
(open_table, lazy_search, db_fetch_pure, pandas, numpy, dotp) drawn as
vertical lines over the per-core CPU activity, one shared CLOCK_MONOTONIC
time axis.

Reads Batch_*_Query_*_stats.csv from the run output folder. For every
(batch, query, thread) the rerank span is split into pages of
`tasks_per_fig` consecutive tasks; each page becomes one SVG:

  TOP    the tasks as phase-segmented bars (doc id + core annotated)
  BOTTOM one CPU strip (stacked fields) per core the tasks touched
  LINES  vertical dashed lines at every sub-phase start, colored by phase,
         in the same style as the pipeline event lines; task starts are
         solid grey and cross all panels.

Wire-up (main plotting script, CPUMeter branch, after extract_core_deltas):

    from rerank_zoom_plots import generate_rerank_zooms
    generate_rerank_zooms(output_folder, t, dt, d, t0_ns, cores,
                          iprintf=cprint.iprintf)

Resolution note: the CPU panels resolve nothing finer than the meter period
(and never finer than one 10 ms jiffy); sub-phases shorter than that (lazy,
pandas, numpy, dotp) get boundary lines but no visible CPU structure of
their own — the lines still localize them exactly on the shared clock.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpu_core_plots import (FIELD_COLORS, FIELDS, STACK_ORDER, IMG_EXT)

# (label, start_col, duration_ms_col, color) — temporal order inside a task
PHASES = [
    ("open_table", "open_table_start",    "open_table_time_ms",    "#57B4E9"),
    ("lazy",       "lazy_search_start",   "lazy_search_time_ms",   "#CC79A7"),
    ("db_fetch",   "db_fetch_pure_start", "db_fetch_pure_time_ms", "#0072B2"),
    ("pandas",     "pandas_start",        "pandas_time_ms",        "#F0E442"),
    ("numpy",      "numpy_time_start",    "numpy_time_ms",         "#019E73"),
    ("dotp",       "dotp_time_start",     "dotp_time_ms",          "#E69F00"),
]

REQUIRED = (["thread_id", "doc_id", "abs_start", "abs_end"]
            + [c for _, c, _, _ in PHASES] + [c for _, _, c, _ in PHASES])


def _load_csvs(folder):
    frames = []
    pat = re.compile(r"^Batch_(\d+)_Query_(\d+)_stats\.csv$")
    for fp in sorted(glob.glob(os.path.join(folder,
                                            "Batch_*_Query_*_stats.csv"))):
        m = pat.match(os.path.basename(fp))
        if m is None:        # e.g. Batch_0_Query_0_retrieval_stats.csv
            continue
        df = pd.read_csv(fp).dropna(subset=["thread_id"])
        df["batch_num"], df["query_id"] = int(m.group(1)), int(m.group(2))
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"stats CSVs missing columns: {missing}")
    df["thread_id"] = df["thread_id"].astype(int)
    return df


def _core_strip(ax, t, dt, d_core, x0, x1):
    order = [FIELDS.index(f) for f in STACK_ORDER]
    m = (t >= x0) & (t <= x1)
    if m.sum() < 2:                       # widen to nearest samples
        i0 = max(np.searchsorted(t, x0) - 1, 0)
        i1 = min(np.searchsorted(t, x1) + 1, len(t) - 1)
        m = np.zeros_like(m); m[i0:i1 + 1] = True
    pct = np.clip(d_core[m][:, order].astype(float)
                  / (dt[m][:, None] * 100.0) * 100.0, 0, 100).T
    ax.stackplot(t[m], pct, colors=[FIELD_COLORS[f] for f in STACK_ORDER],
                 linewidth=0)
    ax.set_ylim(0, 100)
    ax.set_yticks([])


def _document_figure(task, t, dt, d, t0_ns, cores,
                     save_path,
                     line_phases=("db_fetch", "dotp")):

    x0 = (task["abs_start"] - t0_ns) / 1e9 - 0.01
    x1 = (task["abs_end"]   - t0_ns) / 1e9 + 0.01

    used = sorted({
        int(task["cpu_core_start"]),
        int(task["cpu_core_end"])
    })

    fig, axes = plt.subplots(
        1 + len(used),
        1,
        figsize=(12, 2.2 + len(used) * 1.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4] + [1] * len(used)}
    )

    axes = np.atleast_1d(axes)

    ax_task = axes[0]
    core_axes = dict(zip(used, axes[1:]))

    lined = [
        p for p in PHASES
        if line_phases == "all" or p[0] in line_phases
    ]

    #
    # ----- task bar -----
    #

    for label, s_col, d_col, color in PHASES:

        ax_task.barh(
            0,
            task[d_col] / 1e3,
            left=(task[s_col] - t0_ns) / 1e9,
            height=0.7,
            color=color,
            linewidth=0
        )

    task_start = (task["abs_start"] - t0_ns) / 1e9

    migrated = (
        int(task["cpu_core_start"])
        !=
        int(task["cpu_core_end"])
    )

    if migrated:

        ax_task.barh(
            0,
            (task["abs_end"] - task["abs_start"]) / 1e9,
            left=task_start,
            height=0.7,
            fill=False,
            edgecolor="red",
            linewidth=1
        )

    ax_task.text(
        task_start,
        0.45,
        f"d{int(task['doc_id'])}@c{int(task['cpu_core_start'])}",
        fontsize=8,
        rotation=90,
        va="bottom"
    )

    #
    # phase boundaries
    #

    for ax in axes:

        ax.axvline(
            task_start,
            color="dimgray",
            linewidth=0.8
        )

        for label, s_col, _, color in lined:

            ax.axvline(
                (task[s_col] - t0_ns) / 1e9,
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.85
            )

    ax_task.set_ylim(-0.6, 1.2)
    ax_task.set_yticks([])
    ax_task.set_ylabel("rerank",
                       rotation=0,
                       ha="right",
                       va="center")

    #
    # ----- CPU -----
    #

    for core in used:

        _core_strip(
            core_axes[core],
            t,
            dt,
            d[:, cores.index(core), :],
            x0,
            x1
        )

        core_axes[core].set_ylabel(
            f"cpu{core}",
            rotation=0,
            ha="right",
            va="center"
        )

    axes[-1].set_xlim(x0, x1)
    axes[-1].set_xlabel("Time (s)")

    #
    # legend
    #

    #
# ----- legends -----
#

    # Row 1: phase-duration bars
    bar_h = [
        plt.Rectangle((0, 0), 1, 1, fc=c)
        for *_, c in PHASES
    ]

    # Row 2: phase boundary lines
    ln_h = [
        plt.Line2D([0], [0],
                color=c,
                linestyle="--",
                linewidth=1.2)
        for *_, c in lined
    ]

    ln_h.append(
        plt.Line2D([0], [0],
                color="dimgray",
                linewidth=1.2)
    )

    # Row 3: CPU activity fields
    fl_h = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=FIELD_COLORS[f],
            ec="lightgray" if f == "idle" else "none"
        )
        for f in STACK_ORDER
    ]

    leg1 = fig.legend(
        bar_h,
        [f"{p[0]} (bar)" for p in PHASES],
        loc="upper center",
        ncol=len(PHASES),
        fontsize=6,
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
    )

    leg2 = fig.legend(
        ln_h,
        [f"{p[0]} start" for p in lined] + ["task start"],
        loc="upper center",
        ncol=len(lined) + 1,
        fontsize=6,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
    )

    leg3 = fig.legend(
        fl_h,
        STACK_ORDER,
        loc="upper center",
        ncol=len(STACK_ORDER),
        fontsize=6,
        bbox_to_anchor=(0.5, 1.00),
        frameon=False,
    )

    fig.add_artist(leg1)
    fig.add_artist(leg2)
    fig.add_artist(leg3)

    tid = int(task["thread_id"])
    b = int(task["batch_num"])
    q = int(task["query_id"])
    doc = int(task["doc_id"])

    fig.suptitle(
        f"B{b}Q{q} tid {tid} — document {doc}",
        fontsize=9,
        y=1.15,
    )
    fig.suptitle(
        f"B{b}Q{q} tid {tid} — document {doc}",
        fontsize=9,
        y=1.15,
    )

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def generate_rerank_zooms(msg, output_folder, data_file_name, x_pos,
                          tasks_per_fig=8, iprintf=print, max_figures=None,
                          line_phases=("db_fetch", "dotp")):
    """Same calling convention as cpu_core_plots.generate_cpu_figures:
    takes the raw CPUMeter msg and extracts NATIVE-resolution deltas
    itself (the zoom must not use rebinned data). x_pos is accepted for
    signature symmetry; sub-phase lines come from the CSVs, so it is
    currently unused."""
    from cpu_core_plots import PARTITION_CORES, extract_core_deltas
    cores = list(PARTITION_CORES)
    t, dt, d, t0_ns = extract_core_deltas(msg, cores, iprintf)
    return _generate_rerank_zooms_from_arrays(
        output_folder, t, dt, d, t0_ns, cores,
        tasks_per_fig=tasks_per_fig, iprintf=iprintf,
        max_figures=max_figures, line_phases=line_phases)


def _generate_rerank_zooms_from_arrays(output_folder, t, dt, d, t0_ns,
                                       cores, tasks_per_fig=8, iprintf=print,
                                       max_figures=None,
                                       line_phases=("db_fetch", "dotp")):
    """t, dt, d, t0_ns from cpu_core_plots.extract_core_deltas (NATIVE
    resolution, not rebinned); cores = the same core list passed there."""
    df = _load_csvs(output_folder)
    if df is None:
        iprintf("[rerank-zoom] no Batch_*_Query_*_stats.csv — skipped")
        return 0

    sample_s = float(np.median(dt))
    med_task_s = df["total_rerank_time_ms"].median() / 1e3 \
        if "total_rerank_time_ms" in df else np.nan
    if med_task_s and med_task_s < 2 * sample_s:
        iprintf(f"[rerank-zoom] WARNING: median task "
                f"{med_task_s*1e3:.0f} ms < 2× meter period "
                f"{sample_s*1e3:.0f} ms — CPU panels show task-level, not "
                f"sub-phase-level, activity")

    out_dir = os.path.join(output_folder, "rerank_zoom")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for (b, q, tid), g in df.groupby(
        ["batch_num", "query_id", "thread_id"]):

        g = g.sort_values("abs_start").reset_index(drop=True)

        for idx, (_, task) in enumerate(g.iterrows()):

            if max_figures is not None and n >= max_figures:

                iprintf(
                    f"[rerank-zoom] stopped at max_figures={max_figures}"
                )
                return n

            save_name = (
                f"B{b}_Q{q}_tid{tid}"
                f"_doc{int(task['doc_id'])}"
                f"_{idx:03d}.{IMG_EXT}"
            )

            _document_figure(
                task,
                t,
                dt,
                d,
                t0_ns,
                cores,
                os.path.join(out_dir, save_name),
                line_phases=line_phases
            )

            n += 1
    iprintf(f"[rerank-zoom] saved {n} zoom page(s) in {out_dir}/")
    return n