"""
comparison.py
-------------
Static vs dynamic moderation comparison on the same synthetic network.

Produces per-method plots showing three smoothed LQ-fraction time-series:
  · No moderation (baseline)
  · Static moderation (retroactive removal on the no_moderation runs)
  · Dynamic moderation (in-simulation ban)

and a summary LaTeX table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .data_loader import load_synt_activities
from .metrics import compute_bins, split_pre_post
from .aggregation import (
    describe_array,
    _agg_timeseries,
)
from .static_on_synthetic import apply_static_moderation
from .names import method_display_name

matplotlib.use("Agg")


def _cohens_d_lower_better(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Cohen's d for a "lower is better" metric (lq_fraction).

    Computed as (mean_pre − mean_post) / pooled_std, so that a positive
    value indicates improvement (post lq_fraction is lower than pre).
    """
    pre_c = pre[~np.isnan(pre)]
    post_c = post[~np.isnan(post)]
    if pre_c.size < 2 or post_c.size < 2:
        return float("nan")
    pooled = np.sqrt((np.var(pre_c, ddof=1) + np.var(post_c, ddof=1)) / 2.0)
    if pooled == 0.0:
        return float("nan")
    return float((np.mean(pre_c) - np.mean(post_c)) / pooled)

# ---------------------------------------------------------------------------
# Style  (edit FONT / COLORS to adjust all plots uniformly)
# ---------------------------------------------------------------------------

# Font sizes — matched to static_evaluation/src/evaluation/plotting.py.
# Uses matplotlib default font family (DejaVu Sans) for consistency.
FONT = {
    "label":    13,   # axis labels (xlabel / ylabel)
    "title":    14,   # subplot in-panel titles (A, B, …)
    "legend":   13,   # legend entries
    "tick":     12,   # tick labels on both axes
    "suptitle": 14,   # figure-level suptitle (if used)
}

_C_NOMOD   = "#888888"   # grey
_C_STATIC  = "#E69F00"   # orange
_C_DYNAMIC = "#0072B2"   # blue
_LW = 1.8
_ALPHA_BAND = 0.18
_DPI = 150


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    return (pd.Series(arr)
            .rolling(window=w, center=True, min_periods=1)
            .mean()
            .to_numpy())


# ---------------------------------------------------------------------------
# Aggregate multiple runs into timeseries + summary stats
# ---------------------------------------------------------------------------

def _aggregate_runs(
    bins_list: list[pd.DataFrame],
    t_mod: float,
    quality_threshold: float,
) -> dict:
    """
    Aggregate bins from multiple runs into:
      - timeseries: per-bin median + 5th/95th percentile band across runs
        (inter-run dispersion, matching the static pipeline; NOT a CI of
        the mean)
      - pre/post lq_fraction stats (pooled per-bin)
      - Cohen's d on pooled per-bin lq_fraction (pre vs post)
    """
    ts = _agg_timeseries(bins_list, "lq_fraction")

    pre_lq_parts = []
    post_lq_parts = []
    for bins in bins_list:
        pre, post = split_pre_post(bins, t_mod)
        pre_lq_parts.append(pre["lq_fraction"].dropna().to_numpy())
        post_lq_parts.append(post["lq_fraction"].dropna().to_numpy())

    pre_lq = np.concatenate(pre_lq_parts) if pre_lq_parts else np.array([])
    post_lq = np.concatenate(post_lq_parts) if post_lq_parts else np.array([])

    return {
        "timeseries": ts,
        "pre_stats": describe_array(pre_lq),
        "post_stats": describe_array(post_lq),
        "cohens_d": _cohens_d_lower_better(pre_lq, post_lq),
    }


# ---------------------------------------------------------------------------
# Core: compute all three conditions for one ranking method
# ---------------------------------------------------------------------------

def compute_method_comparison(
    nomod_run_paths: list[Path],
    dynamic_run_paths: list[Path],
    ranker_name: str,
    t_mod: float,
    top_k: int,
    quality_threshold: float = 0.39,
    credibility_threshold: float = 39.0,
    bin_width: float = 1.0,
) -> dict[str, dict]:
    """
    Compute three conditions for one ranking method.

    Returns dict with keys "no_moderation", "static", "dynamic",
    each containing timeseries + summary stats.
    """
    # --- No moderation (baseline) ---
    nomod_bins = []
    for p in nomod_run_paths:
        df = load_synt_activities(p)
        nomod_bins.append(compute_bins(df, bin_width, quality_threshold))

    nomod_agg = _aggregate_runs(nomod_bins, t_mod, quality_threshold)

    # --- Static moderation (retroactive on no_moderation runs) ---
    static_bins = []
    for p in nomod_run_paths:
        df = load_synt_activities(p)
        filtered = apply_static_moderation(
            df, ranker_name, t_mod, top_k,
            credibility_threshold=credibility_threshold,
        )
        static_bins.append(compute_bins(filtered, bin_width, quality_threshold))

    static_agg = _aggregate_runs(static_bins, t_mod, quality_threshold)

    # --- Dynamic moderation (in-simulation) ---
    dynamic_bins = []
    for p in dynamic_run_paths:
        df = load_synt_activities(p)
        dynamic_bins.append(compute_bins(df, bin_width, quality_threshold))

    dynamic_agg = _aggregate_runs(dynamic_bins, t_mod, quality_threshold)

    return {
        "no_moderation": nomod_agg,
        "static": static_agg,
        "dynamic": dynamic_agg,
    }


# ---------------------------------------------------------------------------
# Plot: three-line comparison for one method
# ---------------------------------------------------------------------------

def plot_method_comparison(
    comparison: dict[str, dict],
    ranker_name: str,
    t_mod: float,
    smooth_window: int = 7,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot no-mod / static / dynamic LQ fraction time-series for one method.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=_DPI)

    configs = [
        ("no_moderation", "No moderation",        _C_NOMOD,   "--"),
        ("static",        "Static moderation",     _C_STATIC,  "-"),
        ("dynamic",       "Dynamic moderation",    _C_DYNAMIC, "-"),
    ]

    handles = []
    for key, label, color, ls in configs:
        ts = comparison[key]["timeseries"]
        if ts.empty:
            continue
        t = ts["time_mid"].to_numpy()
        center = _smooth(ts["center"].to_numpy(), smooth_window)
        lo = _smooth(ts["lo"].to_numpy(), smooth_window)
        hi = _smooth(ts["hi"].to_numpy(), smooth_window)

        ax.plot(t, center, color=color, lw=_LW, ls=ls, zorder=3)
        ax.fill_between(t, lo, hi, color=color, alpha=_ALPHA_BAND, zorder=2)
        handles.append(Line2D([0], [0], color=color, lw=_LW, ls=ls,
                               label=label))

    ax.axvline(t_mod, color="grey", ls=":", lw=1.0, zorder=1)
    handles.append(Line2D([0], [0], color="grey", ls=":", lw=1.0,
                          label=f"$t_{{\\mathrm{{mod}}}}={int(t_mod)}$"))

    ax.set_xlabel("Time (days)", fontsize=FONT["label"])
    ax.set_ylabel("LQ action fraction", fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(lw=0.4, alpha=0.5)

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(handles),
        fontsize=FONT["legend"],
        framealpha=0.9,
        handlelength=2.2,
        columnspacing=2.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 2x2 grid summary plot
# ---------------------------------------------------------------------------

# Fixed method order and labels for the grid plot
_GRID_METHODS: list[tuple[str, str]] = [
    ("tash_index",         "A"),
    ("random_forest",      "B"),
    ("repost_count",       "C"),
    ("cosine_eigenvector", "D"),
]


def plot_grid_comparison(
    all_comparisons: dict[str, dict],
    t_mod: float,
    smooth_window: int = 7,
    output_path: Path | None = None,
) -> plt.Figure | None:
    """
    2x2 grid of per-method comparison plots.

    Rows = [TASH-Index, Random Forest, Repost Count, Cosine Eigenvector]
    (in the fixed order A, B, C, D).  Single shared legend at top,
    shared Y-axis for direct comparison.

    Missing methods are skipped: if not all four are present, returns None.
    """
    missing = [m for m, _ in _GRID_METHODS if m not in all_comparisons]
    if missing:
        print(f"    Grid plot skipped: missing methods {missing}")
        return None

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), dpi=_DPI,
        sharex=True, sharey=True,
    )
    axes_flat = axes.flatten()

    configs = [
        ("no_moderation", "No moderation",     _C_NOMOD,   "--"),
        ("static",        "Static moderation",  _C_STATIC,  "-"),
        ("dynamic",       "Dynamic moderation", _C_DYNAMIC, "-"),
    ]

    for ax, (method, letter) in zip(axes_flat, _GRID_METHODS):
        comp = all_comparisons[method]
        for key, _label, color, ls in configs:
            ts = comp[key]["timeseries"]
            if ts.empty:
                continue
            t = ts["time_mid"].to_numpy()
            center = _smooth(ts["center"].to_numpy(), smooth_window)
            lo = _smooth(ts["lo"].to_numpy(), smooth_window)
            hi = _smooth(ts["hi"].to_numpy(), smooth_window)
            ax.plot(t, center, color=color, lw=_LW, ls=ls, zorder=3)
            ax.fill_between(t, lo, hi, color=color, alpha=_ALPHA_BAND, zorder=2)

        ax.axvline(t_mod, color="grey", ls=":", lw=1.0, zorder=1)
        ax.tick_params(labelsize=FONT["tick"])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(lw=0.4, alpha=0.5)

        # Panel label (A/B/C/D) + method name
        display = method_display_name(method)
        ax.text(
            0.02, 0.96,
            f"({letter}) {display}",
            transform=ax.transAxes,
            fontsize=FONT["title"], fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, pad=3.0),
        )

    # Shared axis labels (only on outer panels)
    for ax in axes[-1, :]:
        ax.set_xlabel("Days after moderation", fontsize=FONT["label"])
    for ax in axes[:, 0]:
        ax.set_ylabel("LQ action fraction", fontsize=FONT["label"])

    # Shared legend at the top
    handles = [
        Line2D([0], [0], color=_C_NOMOD,   lw=_LW, ls="--",
               label="No moderation"),
        Line2D([0], [0], color=_C_STATIC,  lw=_LW, ls="-",
               label="Static moderation"),
        Line2D([0], [0], color=_C_DYNAMIC, lw=_LW, ls="-",
               label="Dynamic moderation"),
        Line2D([0], [0], color="grey", ls=":", lw=1.0,
               label=f"$t_{{\\mathrm{{mod}}}}={int(t_mod)}$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        fontsize=FONT["legend"],
        framealpha=0.9,
        handlelength=2.2,
        columnspacing=2.0,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# LaTeX comparison table
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _hcell(s: str) -> str:
    return "\\textbf{" + s + "}"


def _delta_cell(pre_v: float, post_v: float, d: int = 3, bold: bool = False) -> str:
    """Format a Δ cell. Arrow indicates direction; bold only when requested."""
    if pre_v != pre_v or post_v != post_v:
        return "---"
    delta = post_v - pre_v
    abs_d = abs(delta)
    fmt = f"{abs_d:.{d}f}"
    if delta < 0:
        inner = f"$\\downarrow${fmt}"
    elif delta > 0:
        inner = f"$\\uparrow${fmt}"
    else:
        inner = f"$=$ {fmt}"
    return f"\\textbf{{{inner}}}" if bold else inner


def _d_cell(val: float, bold: bool = False) -> str:
    """Format a Cohen's d cell."""
    if val != val:
        return "---"
    s = f"{val:+.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def _gap_fmt(gap: float, bold: bool = False) -> str:
    """Format a gap value (sign-preserving)."""
    if gap != gap:
        return "---"
    s = f"{gap:+.3f}"
    return f"\\textbf{{{s}}}" if bold else s


# Absolute gap in Δ mean above which a discrepancy is highlighted
_DELTA_GAP_THRESHOLD = 0.01
# Absolute gap in Cohen's d above which a discrepancy is highlighted
_D_GAP_THRESHOLD = 0.30


def build_comparison_table(
    all_comparisons: dict[str, dict[str, dict]],
    network_name: str,
    t_mod: float,
    top_k: int,
    quality_threshold: float = 0.39,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Build a LaTeX table comparing static vs dynamic for all methods.

    Columns: Static Δ mean, Dynamic Δ mean, Gap | Static d, Dynamic d, Gap.
    Bold highlights large gaps (discrepancies) between the two regimes.
    """
    methods = sorted(all_comparisons.keys())
    if not methods:
        return ""

    if caption is None:
        caption = (
            f"Static vs.\\ dynamic moderation on {_escape(network_name)} "
            f"(moderation at day~{int(t_mod)}, top-{top_k} users banned). "
            f"Each ranking method is evaluated under two regimes: "
            f"\\emph{{static}} (retroactive removal of the top-$k$ users' "
            f"actions from the post-moderation period, applied to "
            f"unmoderated simulation runs) and \\emph{{dynamic}} "
            f"(users banned during the simulation, affecting subsequent "
            f"network activity). "
            f"$\\Delta$\\,mean: change in mean daily low-quality (LQ) "
            f"fraction between pre- and post-moderation "
            f"($\\downarrow$~=~less misinformation). "
            f"Cohen's~$d$: standardised effect size on daily LQ fraction "
            f"(larger positive~=~stronger reduction). "
            f"Gap~=~Static$-$Dynamic: a positive $\\Delta$\\,mean gap "
            f"means the static evaluation overestimates the improvement "
            f"(it predicts a larger reduction than actually occurs); "
            f"a positive Cohen's~$d$ gap means the static effect size "
            f"exceeds the dynamic one. "
            f"\\textbf{{Bold}} marks notable discrepancies "
            f"($|\\text{{gap}}| \\geq {_DELTA_GAP_THRESHOLD}$ for "
            f"$\\Delta$\\,mean, "
            f"$|\\text{{gap}}| \\geq {_D_GAP_THRESHOLD}$ for "
            f"Cohen's~$d$)."
        )
    if label is None:
        label = f"tab:static_vs_dynamic_{network_name.replace(' ', '_')}"

    # Method | S Δm | D Δm | Gap_Δm | S d | D d | Gap_d
    col_spec = "@{}l cc c cc c@{}"
    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")
    h_delta_mean = _hcell("$\\Delta$\\,mean")
    h_cohens_d   = _hcell("Cohen's $d$")
    lines.append(
        "    & "
        f"\\multicolumn{{3}}{{c}}{{{h_delta_mean}}}"
        " & "
        f"\\multicolumn{{3}}{{c}}{{{h_cohens_d}}}"
        " \\\\"
    )
    lines.append("    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    lines.append(
        "    " + " & ".join([
            _hcell("Method"),
            "S", "D", _hcell("Gap"),
            "S", "D", _hcell("Gap"),
        ]) + " \\\\"
    )
    lines.append("    \\midrule")

    for method in methods:
        comp = all_comparisons[method]
        lbl = _escape(method_display_name(method))

        # Static
        s = comp["static"]
        s_pre  = s["pre_stats"].get("mean", float("nan"))
        s_post = s["post_stats"].get("mean", float("nan"))
        s_delta = s_post - s_pre if (s_pre == s_pre and s_post == s_post) else float("nan")
        s_d = s["cohens_d"]

        # Dynamic
        d_ = comp["dynamic"]
        d_pre  = d_["pre_stats"].get("mean", float("nan"))
        d_post = d_["post_stats"].get("mean", float("nan"))
        d_delta = d_post - d_pre if (d_pre == d_pre and d_post == d_post) else float("nan")
        d_d = d_["cohens_d"]

        # Gaps (static − dynamic)
        delta_gap = (s_delta - d_delta) if (s_delta == s_delta and d_delta == d_delta) else float("nan")
        d_gap = (s_d - d_d) if (s_d == s_d and d_d == d_d) else float("nan")

        bold_delta_gap = abs(delta_gap) >= _DELTA_GAP_THRESHOLD if delta_gap == delta_gap else False
        bold_d_gap     = abs(d_gap) >= _D_GAP_THRESHOLD if d_gap == d_gap else False

        cells = [
            lbl,
            _delta_cell(s_pre, s_post),
            _delta_cell(d_pre, d_post),
            _gap_fmt(delta_gap, bold=bold_delta_gap),
            _d_cell(s_d),
            _d_cell(d_d),
            _gap_fmt(d_gap, bold=bold_d_gap),
        ]
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
