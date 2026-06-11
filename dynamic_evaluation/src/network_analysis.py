"""
network_analysis.py
-------------------
Build and analyse low-quality (LQ) reshare networks for pre- and
post-moderation periods, computing network-level and node-level metrics
and exporting .gml files for Gephi visualisation.

A **LQ reshare network** is a weighted directed graph where:
  · Nodes are users.
  · An edge u → v with weight w means user u reshared LQ content
    originally posted by v a total of w times within the period.

An action is "LQ" when its quality ≤ the threshold (default 0.39).

Metrics computed
~~~~~~~~~~~~~~~~
Network-level:
  · n_nodes, n_edges
  · density
  · total_weight (sum of edge weights)
  · largest_wcc_nodes (weakly connected component)
  · largest_wcc_fraction
  · avg_weighted_in_degree, avg_weighted_out_degree
  · reciprocity (fraction of edges with a reverse edge)

Node-level (attached as GML attributes):
  · weighted_in_degree, weighted_out_degree
  · in_degree, out_degree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Build LQ reshare graph
# ---------------------------------------------------------------------------

def build_lq_reshare_graph(
    df: pd.DataFrame,
    quality_threshold: float = 0.39,
    exclude_self_loops: bool = True,
) -> nx.DiGraph:
    """
    Build a directed, weighted LQ-reshare network from an activities DataFrame.

    Only *reshare* actions (action_type == "reshare") with quality ≤ threshold
    are included.  Edges go from ``original_uid`` → ``user_id`` (the original
    author towards the resharer), following the convention that misinformation
    "flows" from source to consumer.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: user_id, quality, action_type,
        original_uid (original post author).
    quality_threshold : float
        Actions with quality ≤ this value are considered LQ.
    exclude_self_loops : bool
        Drop edges where source == target.

    Returns
    -------
    nx.DiGraph with integer/string node IDs and ``weight`` edge attribute.
    """
    lq = df[
        (df["action_type"] == "reshare") &
        (df["quality"] <= quality_threshold)
    ].copy()

    if lq.empty:
        return nx.DiGraph()

    # Drop rows with missing original_uid (shouldn't happen for reshares)
    lq = lq.dropna(subset=["original_uid"])

    if exclude_self_loops:
        lq = lq[lq["user_id"] != lq["original_uid"]]

    # Aggregate edge weights
    edges = (
        lq.groupby(["original_uid", "user_id"])
        .size()
        .reset_index(name="weight")
    )

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["original_uid"], row["user_id"], weight=int(row["weight"]))

    return G


# ---------------------------------------------------------------------------
# Network metrics
# ---------------------------------------------------------------------------

@dataclass
class NetworkMetrics:
    """Container for network-level metrics."""
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    total_weight: int = 0
    largest_wcc_nodes: int = 0
    largest_wcc_fraction: float = 0.0
    avg_weighted_in_degree: float = 0.0
    avg_weighted_out_degree: float = 0.0
    reciprocity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "total_weight": self.total_weight,
            "largest_wcc_nodes": self.largest_wcc_nodes,
            "largest_wcc_fraction": self.largest_wcc_fraction,
            "avg_weighted_in_degree": self.avg_weighted_in_degree,
            "avg_weighted_out_degree": self.avg_weighted_out_degree,
            "reciprocity": self.reciprocity,
        }


def compute_network_metrics(G: nx.DiGraph) -> NetworkMetrics:
    """Compute network-level metrics for a directed weighted graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return NetworkMetrics()

    total_w = sum(d["weight"] for _, _, d in G.edges(data=True))
    density = nx.density(G)

    # Largest weakly connected component
    if n > 0:
        wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        lcc_size = len(wccs[0]) if wccs else 0
    else:
        lcc_size = 0

    # Weighted degrees
    w_in  = [d for _, d in G.in_degree(weight="weight")]
    w_out = [d for _, d in G.out_degree(weight="weight")]
    avg_w_in  = float(np.mean(w_in))  if w_in  else 0.0
    avg_w_out = float(np.mean(w_out)) if w_out else 0.0

    # Reciprocity
    recip = nx.reciprocity(G) if m > 0 else 0.0

    return NetworkMetrics(
        n_nodes=n,
        n_edges=m,
        density=density,
        total_weight=total_w,
        largest_wcc_nodes=lcc_size,
        largest_wcc_fraction=lcc_size / n if n > 0 else 0.0,
        avg_weighted_in_degree=avg_w_in,
        avg_weighted_out_degree=avg_w_out,
        reciprocity=recip,
    )


def _attach_node_attrs(G: nx.DiGraph) -> None:
    """Add degree attributes to nodes (for GML export)."""
    for node in G.nodes():
        G.nodes[node]["weighted_in_degree"]  = G.in_degree(node, weight="weight")
        G.nodes[node]["weighted_out_degree"] = G.out_degree(node, weight="weight")
        G.nodes[node]["in_degree"]  = G.in_degree(node)
        G.nodes[node]["out_degree"] = G.out_degree(node)


# ---------------------------------------------------------------------------
# GML export
# ---------------------------------------------------------------------------

def export_gml(G: nx.DiGraph, path: Path) -> None:
    """
    Export the graph to GML format readable by Gephi.

    Attaches node-level degree attributes before writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _attach_node_attrs(G)
    # GML requires string or numeric node labels
    H = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    nx.write_gml(H, str(path))


# ---------------------------------------------------------------------------
# Pre/post analysis for a single run
# ---------------------------------------------------------------------------

@dataclass
class PrePostNetworks:
    """Pre- and post-moderation network analysis for a single run."""
    pre_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    post_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    pre_metrics: NetworkMetrics = field(default_factory=NetworkMetrics)
    post_metrics: NetworkMetrics = field(default_factory=NetworkMetrics)


def analyse_pre_post(
    df: pd.DataFrame,
    t_mod: float,
    quality_threshold: float = 0.39,
    window: float | None = None,
) -> PrePostNetworks:
    """
    Split activities at t_mod, build LQ reshare networks for each period,
    and compute network metrics.

    Parameters
    ----------
    window : float, optional
        Symmetric time window in days.  If given, only actions within
        ``[t_mod - window, t_mod)`` (pre) and ``[t_mod, t_mod + window)``
        (post) are included.  This ensures both periods have the same
        duration for a fair comparison.  If *None*, the post-period
        duration is used as the window for both periods.
    """
    t_max = df["clock_time"].max()
    if window is None:
        # Default: match pre-window to actual post duration
        window = t_max - t_mod

    pre_start = max(0.0, t_mod - window)
    post_end  = t_mod + window

    pre_df  = df[(df["clock_time"] >= pre_start) & (df["clock_time"] < t_mod)]
    post_df = df[(df["clock_time"] >= t_mod) & (df["clock_time"] < post_end)]

    pre_g  = build_lq_reshare_graph(pre_df,  quality_threshold)
    post_g = build_lq_reshare_graph(post_df, quality_threshold)

    return PrePostNetworks(
        pre_graph=pre_g,
        post_graph=post_g,
        pre_metrics=compute_network_metrics(pre_g),
        post_metrics=compute_network_metrics(post_g),
    )


# ---------------------------------------------------------------------------
# Aggregate metrics across multiple runs
# ---------------------------------------------------------------------------

def aggregate_network_metrics(
    metrics_list: list[NetworkMetrics],
) -> dict[str, dict[str, float]]:
    """
    Aggregate a list of NetworkMetrics into mean ± std for each field.

    Returns dict[field_name] → {"mean": ..., "std": ...}
    """
    if not metrics_list:
        return {}

    keys = list(metrics_list[0].to_dict().keys())
    result: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([m.to_dict()[k] for m in metrics_list], dtype=float)
        result[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# LaTeX table: pre vs post network metrics
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _hcell(s: str) -> str:
    return "\\textbf{" + s + "}"


def _fmt_val(v: float, d: int = 2) -> str:
    if v != v:
        return "---"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{d}f}"


def _delta_pct_raw(pre: float, post: float) -> float:
    """Compute percentage change (raw value)."""
    if pre == 0 or pre != pre or post != post:
        return float("nan")
    return (post - pre) / pre * 100


def _delta_pct_fmt(pct: float, bold: bool = False) -> str:
    """Format a percentage change with arrow; bold only when requested."""
    if pct != pct:
        return "---"
    if pct < 0:
        inner = f"$\\downarrow${abs(pct):.1f}\\%"
    elif pct > 0:
        inner = f"$\\uparrow${pct:.1f}\\%"
    else:
        inner = "$=$ 0.0\\%"
    return f"\\textbf{{{inner}}}" if bold else inner


_COMPACT_METRICS = [
    ("n_nodes",       "Nodes"),
    ("n_edges",       "Edges"),
    ("total_weight",  "Weight"),
    ("largest_wcc_fraction", "LCC frac."),
]

# Threshold (pp) above which a static-vs-dynamic gap is considered notable
_GAP_THRESHOLD_PP = 5.0


def _gap_cell(gap: float, bold: bool = False) -> str:
    """Format a gap value (percentage points)."""
    if gap != gap:
        return "---"
    s = f"{gap:+.1f}\\,pp"
    return f"\\textbf{{{s}}}" if bold else s


def build_network_metrics_table(
    method_results: dict[str, dict[str, dict]],
    network_name: str,
    t_mod: float,
    nomod_result: dict[str, dict] | None = None,
    window_days: float | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Build a LaTeX table: rows = methods, per metric show Static Δ%,
    Dynamic Δ%, and Gap (Static − Dynamic, in pp).

    Bold highlights large discrepancies (|gap| ≥ threshold) to draw
    attention to where static and dynamic evaluations diverge.

    Parameters
    ----------
    method_results : dict
        {method_name: {"static": {"pre": agg, "post": agg},
                       "dynamic": {"pre": agg, "post": agg}}}
    nomod_result : dict, optional
        {"pre": agg, "post": agg} for the no-moderation baseline.
    window_days : float, optional
        Duration of each symmetric window (for the caption).
    """
    methods = sorted(method_results.keys())
    if not methods:
        return ""

    win_str = ""
    if window_days is not None:
        win_str = (
            f"Pre- and post-moderation windows have equal duration "
            f"({int(window_days)}~days each). "
        )

    if caption is None:
        caption = (
            f"Percentage change in the LQ reshare network between "
            f"symmetric pre- and post-moderation periods on "
            f"{_escape(network_name)} (moderation at day~{int(t_mod)}). "
            f"{win_str}"
            f"A LQ reshare network is a weighted directed graph where "
            f"an edge from user~$u$ to user~$v$ with weight~$w$ means "
            f"that $v$ reshared low-quality content originally authored "
            f"by~$u$ a total of~$w$ times within the period. "
            f"Nodes: users involved in LQ resharing. "
            f"Edges: distinct source--resharer pairs. "
            f"Weight: total LQ reshares (sum of edge weights). "
            f"LCC~frac.: share of nodes in the largest weakly connected "
            f"component; a drop signals fragmentation. "
            f"Gap~=~Static$-$Dynamic (in percentage points): "
            f"a positive gap means static overestimates the reduction, "
            f"a negative gap means static underestimates it. "
            f"\\textbf{{Bold}} marks gaps $\\geq${_GAP_THRESHOLD_PP:.0f}\\,pp "
            f"in absolute value."
        )
    if label is None:
        label = f"tab:netmetrics_{network_name.replace(' ', '_')}"

    n_met = len(_COMPACT_METRICS)
    # Method | per metric: Static Δ%, Dynamic Δ%, Gap
    col_spec = "@{}l" + " rr r" * n_met + "@{}"

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\setlength{\\tabcolsep}{3pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Top-level headers: one group per metric
    header1 = "    "
    cmidrules = "    "
    col = 2
    for _, display in _COMPACT_METRICS:
        header1 += f" & \\multicolumn{{3}}{{c}}{{{_hcell(display)}}}"
        cmidrules += f" \\cmidrule(lr){{{col}-{col + 2}}}"
        col += 3
    header1 += " \\\\"
    lines.append(header1)
    lines.append(cmidrules)

    # Sub-headers: S / D / Gap repeated
    sub = f"    {_hcell('Method')}"
    for _ in _COMPACT_METRICS:
        sub += f" & S & D & Gap"
    sub += " \\\\"
    lines.append(sub)
    lines.append("    \\midrule")

    def _row_raw_pcts(result: dict) -> list[float]:
        pcts = []
        pre_agg  = result.get("pre", {})
        post_agg = result.get("post", {})
        for key, _ in _COMPACT_METRICS:
            pre_v  = pre_agg.get(key, {}).get("mean", float("nan"))
            post_v = post_agg.get(key, {}).get("mean", float("nan"))
            pcts.append(_delta_pct_raw(pre_v, post_v))
        return pcts

    # Baseline row
    if nomod_result is not None:
        nomod_pcts = _row_raw_pcts(nomod_result)
        row = "    No moderation"
        for p in nomod_pcts:
            fv = _delta_pct_fmt(p)
            row += f" & {fv} & {fv} & ---"
        row += " \\\\"
        lines.append(row)
        lines.append("    \\midrule")

    # Method rows
    from .names import method_display_name
    for method in methods:
        mr = method_results[method]
        lbl = _escape(method_display_name(method))

        s_pcts = _row_raw_pcts(mr.get("static", {}))
        d_pcts = _row_raw_pcts(mr.get("dynamic", {}))

        row = f"    {lbl}"
        for i in range(n_met):
            sp, dp = s_pcts[i], d_pcts[i]
            gap = (sp - dp) if (sp == sp and dp == dp) else float("nan")
            bold_gap = abs(gap) >= _GAP_THRESHOLD_PP if gap == gap else False
            row += f" & {_delta_pct_fmt(sp)}"
            row += f" & {_delta_pct_fmt(dp)}"
            row += f" & {_gap_cell(gap, bold=bold_gap)}"
        row += " \\\\"
        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)
