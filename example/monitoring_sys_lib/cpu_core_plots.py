"""
cpu_core_plots.py — per-core + aggregate CPU plots for MSys recordings,
full per-field detail, SVG output.

Outputs per recording:
  <name>_per_core.svg          - combined verification grid (all partition cores)
  <name>_cores/cpu<N>.svg      - one standalone figure per core
  <name>_aggregate.svg         - fields summed over the AGGREGATION SET,
                                 normalized by that set's capacity

Aggregation set selection:
  1. run_meta.json {"allocated_cores": [...]} in the output folder  (preferred:
     write it from the benchmark so plots always match the run config)
  2. otherwise: auto-detected active cores (mean busy > ACTIVE_THRESHOLD %)
  3. fallback if nothing is active: the whole partition

Layout facts (verified against cpu_meter.cc):
  core_stats[0] = aggregate 'cpu' line (unused); core_stats[i>=1] = cpu(i-1)
  guest/guest_nice are subsets of user/nice -> never stacked (double count)
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

PARTITION_CORES = list(range(16, 32))   # cores shown in verification views
ACTIVE_THRESHOLD = 5.0                  # % mean busy to count a core "active"
IMG_EXT = "svg"

FIELDS = ["user", "nice", "system", "idle", "iowait",
          "irq", "softirq", "steal"]
STACK_ORDER = ["user", "nice", "system", "iowait",
               "irq", "softirq", "steal", "idle"]
N_BUSY = STACK_ORDER.index("idle")      # bands before idle count as busy

FIELD_COLORS = {
    "user":    "#57B4E9",
    "nice":    "#019E73",
    "system":  "#E69F00",
    "iowait":  "#B11000",
    "irq":     "#5B2680",
    "softirq": "#56B449",
    "steal":   "#329E73",
    "idle":    "#FFFFFF",
}


def load_meta_cores(output_folder: str):
    meta_path = os.path.join(output_folder, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "allocated_cores" in meta:
            return sorted(meta["allocated_cores"])
    return None


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------

def extract_core_deltas(msg, cores, iprintf=print):
    n = len(msg.metrics)
    assert n >= 2, "need at least two samples to form deltas"
    ncores_recorded = len(msg.metrics[0].core_stats) - 1
    for c in cores:
        assert c < ncores_recorded, (
            f"cpu{c} not present in recording ({ncores_recorded} cores) — "
            f"topology changed between run and analysis?")

    ts = np.array([m.timestamp for m in msg.metrics], dtype=np.int64)
    raw = np.empty((n, len(cores), 8), dtype=np.int64)
    guest_seen = 0
    for i, m in enumerate(msg.metrics):
        for j, c in enumerate(cores):
            cs = m.core_stats[c + 1]
            raw[i, j] = (cs.user, cs.nice, cs.system, cs.idle,
                         cs.iowait, cs.irq, cs.softirq, cs.steal)
            guest_seen |= cs.guest | cs.guest_nice
    if guest_seen:
        iprintf("[cpu-plots] NOTE: nonzero guest time (contained in "
                "user/nice; not stacked separately).")

    d = np.diff(raw, axis=0)
    dt = np.diff(ts) / 1e9
    t_sec = (ts[1:] - ts[0]) / 1e9
    return t_sec, dt, d, ts[0]


def field_percent(d, dt, core_idx=None):
    """% of capacity per field in STACK_ORDER.
    core_idx None      -> per-core     (8, n-1, C)
    core_idx = [j...]  -> summed over那 subset, capacity = len(subset)
                          -> (8, n-1)
    """
    order = [FIELDS.index(f) for f in STACK_ORDER]
    x = d[..., order].astype(np.float64)
    if core_idx is None:
        cap = dt[:, None] * 100.0
        pct = (x / cap[..., None]) * 100.0
        return np.clip(pct, 0.0, 100.0).transpose(2, 0, 1)
    x = x[:, core_idx, :].sum(axis=1)
    cap = len(core_idx) * dt * 100.0
    pct = (x / cap[:, None]) * 100.0
    return np.clip(pct, 0.0, 100.0).T


# ----------------------------------------------------------------------------
# event lines with staggered, readable labels
# ----------------------------------------------------------------------------

def parse_events(x_pos, t0_ns):
    events = []
    for s in x_pos:
        try:
            idx = s.index(",")
            tag = s[:idx].strip()
            t = (int(s[idx + 1:].strip()) - t0_ns) / 1e9
        except Exception:
            continue
        if t >= 0:
            events.append((tag, t))
    return events


def draw_event_lines(ax, events, label=True, min_spacing_frac=0.02):
    """Vertical lines; labels staggered ABOVE the axes so they never cover
    data. Labels within min_spacing (fraction of x-range) cycle 3 height
    levels."""
    if not events:
        return
    x0, x1 = ax.get_xlim()
    span = max(x1 - x0, 1e-9)
    levels = [1.02, 1.10, 1.18]
    last_x, lvl = -np.inf, 0
    for tag, t in events:
        ax.axvline(x=t, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
        if not label:
            continue
        lvl = (lvl + 1) % len(levels) if (t - last_x) / span < \
            min_spacing_frac else 0
        last_x = t
        ax.annotate(tag, xy=(t, 1.0), xytext=(t, levels[lvl]),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    fontsize=6, rotation=90, ha="center", va="bottom",
                    annotation_clip=False)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def _stack(ax, t, bands):
    ax.stackplot(t, bands, colors=[FIELD_COLORS[f] for f in STACK_ORDER],
                 linewidth=0)


def _field_handles():
    return [plt.Rectangle((0, 0), 1, 1, fc=FIELD_COLORS[f],
                          ec="lightgray" if f == "idle" else "none")
            for f in STACK_ORDER]


def _field_legend(fig, fontsize=7, y=1.0):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=FIELD_COLORS[f],
                             ec="lightgray" if f == "idle" else "none")
               for f in STACK_ORDER]
    fig.legend(handles, STACK_ORDER, loc="upper center",
               ncol=len(STACK_ORDER), bbox_to_anchor=(0.5, y),
               fontsize=fontsize)


def plot_per_core_grid(t, pct_core, cores, events, save_path, active):
    C = len(cores)
    fig, axes = plt.subplots(C, 1, figsize=(12, 0.8 * C),
                             sharex=True, sharey=True)
    axes = [axes] if C == 1 else list(axes)
    for j, (ax, core) in enumerate(zip(axes, cores)):
        _stack(ax, t, pct_core[:, :, j])
        ax.set_ylim(0, 100)
        ax.set_yticks([])
        ax.set_ylabel(f"cpu{core}", rotation=0, ha="right", va="center",
                      fontsize=8)
        busy = pct_core[:N_BUSY, :, j].sum(axis=0)
        mark = " *" if core in active else ""
        ax.text(1.002, 0.5, f"{busy.mean():4.1f}%{mark}",
                transform=ax.transAxes, fontsize=7, va="center")
        ax.set_xlim(0, t[-1])
        draw_event_lines(ax, events, label=(j == 0))
    axes[-1].set_xlabel("Time (s)")
    _field_legend(fig, y=1.03)
    fig.suptitle("Per-core activity — partition (right: mean busy %, "
                 "* = active)", fontsize=10, y=1.06)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_single_core(t, bands, core, events, save_path):
    fig = plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    _stack(ax, t, bands)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.grid(color="lightgray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPU (%)")
    draw_event_lines(ax, events)
    busy = np.clip(bands[:N_BUSY].sum(axis=0), 0, 100)
    ax.set_title(f"cpu{core} — mean busy {busy.mean():.1f}%, "
                 f"peak {busy.max():.1f}%", fontsize=10, pad=50)
    ax.legend(handles=_field_handles(), labels=STACK_ORDER,
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=len(STACK_ORDER), fontsize=7, frameon=False)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(t, pct_agg, agg_cores, agg_source, events, save_path):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot(111)
    _stack(ax, t, pct_agg)
    busy = np.clip(pct_agg[:N_BUSY].sum(axis=0), 0, 100)
    ax.plot(t, busy, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.grid(color="lightgray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"% of capacity ({len(agg_cores)} cores)")
    draw_event_lines(ax, events)
    core_str = ",".join(str(c) for c in agg_cores)
    ax.set_title(f"Aggregate over cores [{core_str}] ({agg_source}) — "
                 f"mean busy {busy.mean():.1f}%, peak {busy.max():.1f}%",
                 fontsize=10, pad=50)
    ax.legend(handles=_field_handles(), labels=STACK_ORDER,
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=len(STACK_ORDER), fontsize=7, frameon=False)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------

def generate_cpu_figures(msg, output_folder, data_file_name, x_pos,
                         iprintf=print):
    cores = list(PARTITION_CORES)
    t, dt, d, t0_ns = extract_core_deltas(msg, cores, iprintf)
    events = parse_events(x_pos, t0_ns)

    pct_core = field_percent(d, dt)                          # (8, n-1, C)
    mean_busy = pct_core[:N_BUSY].sum(axis=0).mean(axis=0)   # (C,)
    active = [c for c, b in zip(cores, mean_busy) if b > ACTIVE_THRESHOLD]

    # aggregation set: run_meta.json > detected-active > whole partition
    meta_cores = load_meta_cores(output_folder)
    if meta_cores:
        agg_cores, agg_source = meta_cores, "run_meta.json"
    elif active:
        agg_cores, agg_source = active, "auto-detected active"
    else:
        agg_cores, agg_source = cores, "partition (nothing active)"
    agg_idx = [cores.index(c) for c in agg_cores]
    pct_agg = field_percent(d, dt, core_idx=agg_idx)         # (8, n-1)

    iprintf(f"[cpu-plots] partition={cores}")
    iprintf(f"[cpu-plots] detected-active={active}")
    iprintf(f"[cpu-plots] aggregating over {agg_cores} ({agg_source})")

    inactive_idx = [j for j, c in enumerate(cores) if c not in active]
    if inactive_idx and mean_busy[inactive_idx].max() > 2.0:
        iprintf(f"[cpu-plots] WARNING: inactive partition core averages "
                f"{mean_busy[inactive_idx].max():.1f}% busy — contamination "
                f"or leaked threads?")

    base = os.path.join(output_folder, data_file_name)
    plot_per_core_grid(t, pct_core, cores, events,
                       f"{base}_per_core.{IMG_EXT}", active)

    core_dir = f"{base}_cores"
    os.makedirs(core_dir, exist_ok=True)
    for j, core in enumerate(cores):
        plot_single_core(t, pct_core[:, :, j], core, events,
                         os.path.join(core_dir, f"cpu{core}.{IMG_EXT}"))

    plot_aggregate(t, pct_agg, agg_cores, agg_source, events,
                   f"{base}_aggregate.{IMG_EXT}")
    iprintf(f"[cpu-plots] saved: _per_core.{IMG_EXT}, "
            f"_cores/cpu*.{IMG_EXT} ({len(cores)} files), "
            f"_aggregate.{IMG_EXT}")