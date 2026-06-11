"""
Visualization tools for dismantling analysis.

Provides functions to plot dismantling curves comparing different
ranking methods on real and synthetic data, including horizon-based
effectiveness decay analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# Consistent styling
COLORS = plt.cm.tab10.colors
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

# Font sizes — increase these for publication-ready figures.
# Rule of thumb: if the figure is scaled to X% of its original size in the paper,
# multiply all values here by (100 / X). E.g. 14" figure → 7" column = 50% scale → ×2.
FONT = {
    "label": 13,    # axis labels (xlabel, ylabel)
    "title": 14,    # subplot titles
    "legend": 13,   # legend entries (top shared legend)
    "tick": 12,     # tick labels
    "suptitle": 14, # figure suptitle
}


# Explicit display-name overrides for ranking methods.
# Any method not in this dict falls back to title-case with underscores removed.
DISPLAY_NAMES = {
    "tash_index": "TASH-index",
    "cosine_eigenvector": "Coordination Centrality",
}


def _display_name(name: str) -> str:
    """Convert internal snake_case names to human-readable labels.

    Uses DISPLAY_NAMES overrides when available, otherwise falls back to
    title-case with underscores replaced by spaces.
    """
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name.replace("_", " ").title()


def generate_log_scale_positions(n: int) -> List[int]:
    """
    Generate logarithmically-spaced marker positions.

    Used to place markers at visually balanced intervals on log-scale plots.
    """
    if n < 1:
        return []

    positions = []
    power = 1

    while power <= n:
        start = power
        end = min(n + 1, power * 10)
        positions.extend(range(start, end, power))
        power *= 10

    return positions


def aggregate_synthetic_traces(
    traces: List[np.ndarray],
    ci_lower_percentile: float = 5.0,
    ci_upper_percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate multiple dismantling traces with median and percentile-based CI.
    """
    if not traces:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_len = max(len(t) for t in traces)

    padded = []
    for trace in traces:
        if len(trace) < max_len:
            padding = np.full(max_len - len(trace), trace[-1])
            trace = np.concatenate([trace, padding])
        padded.append(trace)

    stacked = np.vstack(padded)

    median_trace = np.median(stacked, axis=0)
    ci_lower = np.percentile(stacked, ci_lower_percentile, axis=0)
    ci_upper = np.percentile(stacked, ci_upper_percentile, axis=0)
    x_positions = np.arange(max_len)

    return median_trace, ci_lower, ci_upper, x_positions


def compute_effectiveness_at_top_k(
    trace: np.ndarray,
    top_k: int = 5,
) -> float:
    """
    Compute effectiveness as % of misinformation removed after removing top K nodes.

    Args:
        trace: Array of remaining misinformation fractions after each node removal.
               trace[0] = 1.0 (full network), trace[i] = fraction after removing i nodes.
               Only nodes present in the test network contribute trace entries.
        top_k: Number of top-ranked nodes to remove (clamped to available nodes).

    Returns:
        Effectiveness as percentage (0-100) of misinformation removed.
        Returns 0.0 if the network is empty (trace has no removal entries).
    """
    # trace always starts with the initial state (FULL = 1.0).
    # If only that entry exists the test network had no ranked nodes → 0%.
    if len(trace) <= 1:
        return 0.0

    # Clamp: can't remove more nodes than exist in the test network.
    k_index = min(top_k, len(trace) - 1)
    remaining = trace[k_index]
    effectiveness = (1.0 - remaining) * 100.0

    return effectiveness


def _plot_dismantling_on_ax(
    ax: plt.Axes,
    traces_dict: Dict[str, np.ndarray],
    ci_lower: Optional[Dict[str, np.ndarray]] = None,
    ci_upper: Optional[Dict[str, np.ndarray]] = None,
    title: str = "",
    xlabel: str = "Nodes removed (log scale)",
    ylabel: str = "Remaining LQ reshares fraction",
    xlim: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
) -> List[plt.Line2D]:
    """
    Plot dismantling curves on a given axes. Returns line handles for shared legend.
    """
    max_len = max(len(t) for t in traces_dict.values()) if traces_dict else 0
    handles = []

    for i, (name, trace) in enumerate(traces_dict.items()):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        x = np.arange(len(trace))
        trace_log_positions = [
            p for p in generate_log_scale_positions(max_len - 1) if p < len(trace)
        ]

        (line,) = ax.plot(
            x,
            trace,
            label=_display_name(name),
            color=color,
            linestyle="-",
            linewidth=1.5,
            marker=marker,
            markevery=trace_log_positions if trace_log_positions else None,
            markersize=6,
        )
        handles.append(line)

        # Add CI if provided
        if ci_lower and ci_upper and name in ci_lower and name in ci_upper:
            ax.fill_between(x, ci_lower[name], ci_upper[name], color=color, alpha=0.2)

    ax.set_xscale("symlog")
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(xlabel, fontsize=FONT["label"])
    ax.set_ylabel(ylabel, fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])
    if title:
        ax.set_title(title, fontsize=FONT["title"], fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.axhline(y=0.5, color="silver", linestyle="--", alpha=0.7)

    if show_legend:
        ax.legend(fontsize=FONT["legend"], loc="best")

    return handles


def plot_dismantling_comparison(
    traces_dict: Dict[str, np.ndarray],
    title: str = "Misinformation network dismantling",
    xlabel: str = "Nodes removed (log scale)",
    ylabel: str = "Remaining LQ reshares fraction",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
    output_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Axes:
    """
    Plot dismantling curves for multiple ranking methods (single plot).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    _plot_dismantling_on_ax(
        ax, traces_dict, title=title, xlabel=xlabel, ylabel=ylabel, show_legend=True
    )

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return ax


def plot_dismantling_with_confidence(
    median_traces: Dict[str, np.ndarray],
    ci_lower_traces: Dict[str, np.ndarray],
    ci_upper_traces: Dict[str, np.ndarray],
    title: str = "Synthetic data dismantling",
    xlabel: str = "Nodes removed (log scale)",
    ylabel: str = "Remaining LQ reshares fraction",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
    xlim: Optional[Tuple[float, float]] = None,
    output_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Axes:
    """
    Plot dismantling curves with confidence intervals (single plot).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    _plot_dismantling_on_ax(
        ax,
        median_traces,
        ci_lower=ci_lower_traces,
        ci_upper=ci_upper_traces,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        show_legend=True,
    )

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return ax


def plot_comparison_dismantling(
    real_traces: Dict[str, np.ndarray],
    synthetic_median: Dict[str, np.ndarray],
    synthetic_ci_lower: Dict[str, np.ndarray],
    synthetic_ci_upper: Dict[str, np.ndarray],
    real_title: str = "Real Data",
    synthetic_title: str = "Synthetic Data",
    suptitle: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create side-by-side comparison of real vs synthetic dismantling.

    Uses a shared horizontal legend positioned between the suptitle and plots.
    """
    # Calculate shared x-axis limits
    all_lengths = list(len(t) for t in real_traces.values())
    all_lengths.extend(len(t) for t in synthetic_median.values())
    max_len = max(all_lengths) if all_lengths else 1
    shared_xlim = (0, max_len - 1)

    # Create figure with space for legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot real data
    handles = _plot_dismantling_on_ax(
        ax1, real_traces, title=real_title, xlim=shared_xlim
    )

    # Plot synthetic data (no y-label: shares the y-axis with the left subplot)
    _plot_dismantling_on_ax(
        ax2,
        synthetic_median,
        ci_lower=synthetic_ci_lower,
        ci_upper=synthetic_ci_upper,
        title=synthetic_title,
        ylabel="",
        xlim=shared_xlim,
    )

    # Shared horizontal legend at the top
    labels = [_display_name(k) for k in real_traces.keys()]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(labels), 6),
        fontsize=FONT["legend"],
        frameon=True,
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT["suptitle"], y=1.12)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def _plot_effectiveness_on_ax(
    ax: plt.Axes,
    effectiveness_by_method: Dict[str, Dict[int, float]],
    horizons: List[int],
    ci_lower: Optional[Dict[str, Dict[int, float]]] = None,
    ci_upper: Optional[Dict[str, Dict[int, float]]] = None,
    title: str = "",
    xlabel: str = "Days after moderation",
    ylabel: Optional[str] = None,
    top_k: int = 5,
    show_legend: bool = False,
) -> List[plt.Line2D]:
    """
    Plot effectiveness decay on a given axes. Returns line handles for shared legend.
    """
    if ylabel is None:
        ylabel = f"LQ reshares removed (%) (top {top_k} nodes)"
    handles = []

    for i, (method_name, horizon_values) in enumerate(effectiveness_by_method.items()):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        x = horizons
        y = [horizon_values.get(h, np.nan) for h in horizons]

        (line,) = ax.plot(
            x,
            y,
            label=_display_name(method_name),
            color=color,
            marker=marker,
            markersize=7,
            linewidth=1.5,
        )
        handles.append(line)

        if ci_lower and ci_upper and method_name in ci_lower and method_name in ci_upper:
            y_lower = [ci_lower[method_name].get(h, np.nan) for h in horizons]
            y_upper = [ci_upper[method_name].get(h, np.nan) for h in horizons]
            ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)

    ax.set_xlabel(xlabel, fontsize=FONT["label"])
    ax.set_ylabel(ylabel, fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])
    if title:
        ax.set_title(title, fontsize=FONT["title"], fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, 105)
    ax.set_xlim(min(horizons) - 1, max(horizons) + max(horizons) * 0.05)
    ax.set_xticks(horizons)

    if show_legend:
        ax.legend(fontsize=FONT["legend"], loc="best")

    return handles


def plot_effectiveness_decay(
    effectiveness_by_method: Dict[str, Dict[int, float]],
    horizons: List[int],
    ci_lower: Optional[Dict[str, Dict[int, float]]] = None,
    ci_upper: Optional[Dict[str, Dict[int, float]]] = None,
    title: str = "Moderation effectiveness over time",
    xlabel: str = "Days after moderation",
    ylabel: Optional[str] = None,
    top_k: int = 5,
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot effectiveness decay for a single dataset (single plot).
    """
    fig, ax = plt.subplots(figsize=figsize)

    _plot_effectiveness_on_ax(
        ax,
        effectiveness_by_method,
        horizons,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        top_k=top_k,
        show_legend=True,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_comparison_effectiveness(
    real_effectiveness: Dict[str, Dict[int, float]],
    synthetic_effectiveness: Dict[str, Dict[int, float]],
    horizons: List[int],
    synthetic_ci_lower: Optional[Dict[str, Dict[int, float]]] = None,
    synthetic_ci_upper: Optional[Dict[str, Dict[int, float]]] = None,
    real_title: str = "Real Data",
    synthetic_title: str = "Synthetic Data",
    suptitle: Optional[str] = None,
    top_k: int = 5,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create side-by-side comparison of real vs synthetic effectiveness decay.

    Uses a shared horizontal legend positioned between the suptitle and plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot real data
    handles = _plot_effectiveness_on_ax(
        ax1, real_effectiveness, horizons, title=real_title, top_k=top_k
    )

    # Plot synthetic data (no y-label: shares the y-axis with the left subplot)
    _plot_effectiveness_on_ax(
        ax2,
        synthetic_effectiveness,
        horizons,
        ci_lower=synthetic_ci_lower,
        ci_upper=synthetic_ci_upper,
        title=synthetic_title,
        ylabel="",
        top_k=top_k,
    )

    # Shared horizontal legend at the top
    labels = [_display_name(k) for k in real_effectiveness.keys()]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(labels), 6),
        fontsize=FONT["legend"],
        frameon=True,
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT["suptitle"], y=1.12)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig
