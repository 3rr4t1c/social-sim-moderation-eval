"""
plotting.py
-----------
Publication-quality plots for the dynamic moderation evaluation.
Primary metric: **lq_fraction** (fraction of low-quality actions per bin).
Figures have no titles (captions go in the paper).

Three output files per network:

distributions.pdf
    Horizontal violin: PRE vs. POST distribution of daily lq_fraction.
    Legend at bottom (Pre / Post per condition column).

timeseries.pdf
    Single-panel smoothed time-series with CI band.
    Legend at top.

cumulative.pdf
    Single-panel running LQ fraction over time.
    Legend at top.

Colour conventions — Wong (2011) colorblind-safe palette.
PRE  = condition colour, alpha 0.30
POST = condition colour, alpha 0.85
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_WONG = [
    "#E69F00", "#56B4E9", "#009E73", "#0072B2",
    "#D55E00", "#CC79A7", "#F0E442",
]
_MARKERS    = ["o", "s", "^", "D", "v", "P", "*"]
_LW         = 1.8
_MS         = 5
_ALPHA_BAND = 0.20
_ALPHA_PRE  = 0.30
_ALPHA_POST = 0.85

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "legend.fontsize": 9,
    "figure.dpi":      150,
})


def _color(i: int) -> str:
    return _WONG[i % len(_WONG)]


def _marker(i: int) -> str:
    return _MARKERS[i % len(_MARKERS)]


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    return (pd.Series(arr)
            .rolling(window=w, center=True, min_periods=1)
            .mean()
            .to_numpy())


def _save(fig: plt.Figure, path: Path | None, show: bool) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _sorted_conditions(all_stats: dict, include_baseline: bool = False) -> list[tuple[int, str, object]]:
    """Return (colour_idx, name, ConditionStats) — sorted, optionally with baseline."""
    baselines = sorted(
        [(k, v) for k, v in all_stats.items() if v.condition.is_baseline],
        key=lambda x: x[0],
    ) if include_baseline else []
    moderated = sorted(
        [(k, v) for k, v in all_stats.items() if not v.condition.is_baseline],
        key=lambda x: x[0],
    )
    combined = baselines + moderated
    return [(i, n, cs) for i, (n, cs) in enumerate(combined)]


# ---------------------------------------------------------------------------
# Shared legend builder — top of figure, horizontal
# ---------------------------------------------------------------------------

def _top_legend(
    fig: plt.Figure,
    cond_handles: list,
    t_mods: set[int],
) -> None:
    """
    Place a legend above the plot area, spread horizontally.
    Condition curves first, then — slightly separated — the t_mod handle.
    """
    handles = list(cond_handles)
    if t_mods:
        # blank spacer
        handles.append(Patch(color="none", label=""))
        for t in sorted(t_mods):
            handles.append(
                Line2D([0], [0], color="grey", ls="--", lw=1.0,
                       label=f"$t_{{\\mathrm{{mod}}}}={t}$")
            )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(handles),
        framealpha=0.9,
        handlelength=1.8,
        columnspacing=1.5,
    )


# ---------------------------------------------------------------------------
# distributions.pdf — single-panel horizontal violin
# ---------------------------------------------------------------------------

def plot_distributions(
    all_stats: dict,
    network_name: str,
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Horizontal violin: PRE vs POST distribution of daily lq_fraction.

    Each method gets well-spaced groups with pre/post violins close together.
    Includes baseline (no moderation) when present.
    """
    # Include baseline in violin plots
    all_conds = []
    baseline_conds = sorted(
        [(k, v) for k, v in all_stats.items() if v.condition.is_baseline],
        key=lambda x: x[0],
    )
    moderated_conds = sorted(
        [(k, v) for k, v in all_stats.items() if not v.condition.is_baseline],
        key=lambda x: x[0],
    )
    for i, (n, cs) in enumerate(baseline_conds + moderated_conds):
        all_conds.append((i, n, cs))

    n = len(all_conds)

    within_gap = 0.18       # pre/post closer together
    group_gap  = 1.4        # more space between methods
    violin_width = 0.30
    fig, ax = plt.subplots(figsize=(8, max(3.0, n * 1.1 + 1.5)))

    ytick_pos: list[float] = []
    ytick_lbl: list[str]   = []

    for i, (ci, name, cs) in enumerate(all_conds):
        color  = _color(ci)
        centre = -i * group_gap

        for data, pos, alpha in [
            (cs.pre_lq_frac,  centre + within_gap, _ALPHA_PRE),
            (cs.post_lq_frac, centre - within_gap, _ALPHA_POST),
        ]:
            if data.size == 0:
                continue
            vp = ax.violinplot([data], positions=[pos], vert=False,
                               widths=violin_width,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(alpha)
                body.set_edgecolor("none")
            vp["cmedians"].set_color("white")
            vp["cmedians"].set_linewidth(2.0)

        ytick_pos.append(centre)
        ytick_lbl.append(cs.condition.label())

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lbl)
    ax.set_xlabel("LQ action fraction")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    # Simple legend: just Pre / Post (colors per method already on y-axis)
    handles = [
        Patch(facecolor="grey", alpha=_ALPHA_PRE,  label="Pre-moderation"),
        Patch(facecolor="grey", alpha=_ALPHA_POST, label="Post-moderation"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        framealpha=0.9,
        handlelength=1.8,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, output_path, show)
    return fig


# ---------------------------------------------------------------------------
# timeseries.pdf — single panel
# ---------------------------------------------------------------------------

def plot_timeseries(
    all_stats: dict,
    network_name: str,
    smooth_window: int = 7,
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Single-panel LQ fraction time-series with CI band."""
    sorted_conds = _sorted_conditions(all_stats, include_baseline=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    t_mods: set[int] = set()
    cond_handles: list = []

    for ci, name, cs in sorted_conds:
        ts = cs.timeseries_lq_frac
        if ts.empty:
            continue
        color = _color(ci)
        mark  = _marker(ci)
        t     = ts["time_mid"].to_numpy()
        mean  = _smooth(ts["mean"].to_numpy(), smooth_window)
        lo    = _smooth(ts["lo"].to_numpy(),   smooth_window)
        hi    = _smooth(ts["hi"].to_numpy(),   smooth_window)
        me    = max(1, len(t) // 18)
        ax.plot(t, mean, color=color, lw=_LW, marker=mark, ms=_MS,
                markevery=me, zorder=3)
        ax.fill_between(t, lo, hi, color=color, alpha=_ALPHA_BAND, zorder=2)
        if cs.t_mod is not None:
            if cs.t_mod not in t_mods:
                ax.axvline(cs.t_mod, color="grey", ls="--", lw=1.0, zorder=1)
            t_mods.add(cs.t_mod)
        cond_handles.append(
            Line2D([0], [0], color=color, lw=_LW, marker=mark, ms=_MS,
                   label=cs.condition.label())
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("LQ action fraction")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(lw=0.4, alpha=0.5)

    _top_legend(fig, cond_handles, t_mods)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, output_path, show)
    return fig


# ---------------------------------------------------------------------------
# cumulative.pdf — single panel running LQ fraction
# ---------------------------------------------------------------------------

def plot_cumulative(
    all_stats: dict,
    network_name: str,
    smooth_window: int = 1,
    output_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Single-panel running (cumulative) LQ fraction with CI band."""
    sorted_conds = _sorted_conditions(all_stats)
    fig, ax = plt.subplots(figsize=(12, 5))
    t_mods: set[int] = set()
    cond_handles: list = []

    for ci, name, cs in sorted_conds:
        ts = cs.cumulative_lq_frac_ts
        if ts.empty:
            continue
        color = _color(ci)
        mark  = _marker(ci)
        t     = ts["time_mid"].to_numpy()
        mean  = _smooth(ts["mean"].to_numpy(), smooth_window)
        lo    = _smooth(ts["lo"].to_numpy(),   smooth_window)
        hi    = _smooth(ts["hi"].to_numpy(),   smooth_window)
        me    = max(1, len(t) // 18)
        ax.plot(t, mean, color=color, lw=_LW, marker=mark, ms=_MS,
                markevery=me, zorder=3)
        ax.fill_between(t, lo, hi, color=color, alpha=_ALPHA_BAND, zorder=2)
        if cs.t_mod is not None:
            if cs.t_mod not in t_mods:
                ax.axvline(cs.t_mod, color="grey", ls="--", lw=1.0, zorder=1)
            t_mods.add(cs.t_mod)
        cond_handles.append(
            Line2D([0], [0], color=color, lw=_LW, marker=mark, ms=_MS,
                   label=cs.condition.label())
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Running LQ action fraction")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(lw=0.4, alpha=0.5)

    _top_legend(fig, cond_handles, t_mods)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, output_path, show)
    return fig
