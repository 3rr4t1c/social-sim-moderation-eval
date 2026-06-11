"""
latex_table.py
--------------
Generate a LaTeX summary table for the dynamic evaluation results.

Single metric: **lq_fraction** (fraction of low-quality actions per bin,
lower is better).

Table structure
---------------
One row per ranking method (plus optional baseline row).

  Method | Δ mean | Δ median | Cohen's d | Activity Δ

- Δ mean / Δ median: post − pre, with ↓ (improvement) or ↑ (worsening).
  Bold when the change is an improvement.
- Cohen's d: standardised effect size (positive = improvement).
- Activity Δ: percent change in total actions post-moderation (a "cost" indicator).

The baseline row (no moderation) is included when present, providing
context for interpreting the moderated conditions' deltas.

Requires packages: booktabs.
"""

from __future__ import annotations

from pathlib import Path

from names import method_display_name


# ---------------------------------------------------------------------------
# Row-label helper
# ---------------------------------------------------------------------------

def _row_label(condition) -> str:
    """Full display name of the ranking method."""
    if condition.is_baseline:
        return "No moderation"
    return method_display_name(condition.method or "")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, d: int = 3) -> str:
    return "---" if v != v else f"{v:.{d}f}"


def _delta_cell(pre_v: float, post_v: float, d: int = 3) -> str:
    """
    Format a Δ cell for a "lower is better" metric.

    Shows the absolute difference with an arrow:
      ↓ 0.124  (improvement, bold)
      ↑ 0.031  (worsening, not bold)
    """
    if pre_v != pre_v or post_v != post_v:
        return "---"
    delta = post_v - pre_v
    abs_d = abs(delta)
    if delta < 0:
        # improvement: post is lower
        return f"\\textbf{{$\\downarrow${_fmt(abs_d, d)}}}"
    elif delta > 0:
        return f"$\\uparrow${_fmt(abs_d, d)}"
    else:
        return f"$=$ {_fmt(abs_d, d)}"


def _escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _hcell(s: str) -> str:
    return "\\textbf{" + s + "}"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_summary_table(
    all_stats: dict,
    network_name: str,
    quality_threshold: float = 0.39,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Build a LaTeX table from ``{condition_name: ConditionStats}``.

    Includes the baseline row (no moderation) when present.
    """
    # Separate baseline from moderated conditions
    baseline_names = sorted(
        n for n, cs in all_stats.items() if cs.condition.is_baseline
    )
    sorted_names = sorted(
        n for n, cs in all_stats.items() if not cs.condition.is_baseline
    )

    # ---- Auto-build caption with metric definitions ----
    if caption is None:
        thr_pct = int(round(quality_threshold * 100))
        mods       = [all_stats[n].condition for n in sorted_names]
        t_mods_set = sorted({c.t_mod  for c in mods if c.t_mod  is not None})
        top_k_set  = sorted({c.top_k  for c in mods if c.top_k  is not None})
        type_set   = sorted({c.mod_type for c in mods if c.mod_type is not None})
        def _join(vals) -> str:
            return ", ".join(str(v) for v in vals) if vals else "---"

        caption = (
            f"Dynamic moderation evaluation on {_escape(network_name)}. "
            f"Moderation at day {_join(t_mods_set)}, "
            f"top-{_join(top_k_set)} users "
            f"({_join(type_set)} strategy). "
            f"LQ threshold: quality $\\leq$ {thr_pct}/100. "
            "$\\Delta$\\,mean and $\\Delta$\\,median show the post$-$pre "
            "change in the daily LQ fraction "
            "($\\downarrow$~=~improvement, \\textbf{bold}). "
            "Cohen's $d$ is the standardised effect size on daily LQ fraction "
            "(larger positive~=~stronger improvement). "
            "Activity $\\Delta$\\% is the percent change in total actions "
            "post-moderation (a measure of moderation cost)."
        )
    if label is None:
        label = f"tab:dyn_{network_name.replace(' ', '_')}"

    # 5 columns: Method | Δ mean | Δ median | Cohen's d | Activity Δ
    col_spec = "@{}l cc c c@{}"

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header
    lines.append(
        "    "
        + " & ".join([
            _hcell("Method"),
            _hcell("$\\Delta$\\,mean"),
            _hcell("$\\Delta$\\,median"),
            _hcell("Cohen's $d$"),
            _hcell("Activity $\\Delta$\\%"),
        ])
        + " \\\\"
    )
    lines.append("    \\midrule")

    def _build_row(name: str) -> str:
        cs  = all_stats[name]
        pre = cs.pre_stats_lq_frac
        post = cs.post_stats_lq_frac

        method_lbl = _escape(_row_label(cs.condition))

        delta_mean   = _delta_cell(pre.get("mean", float("nan")),
                                   post.get("mean", float("nan")))
        delta_median = _delta_cell(pre.get("median", float("nan")),
                                   post.get("median", float("nan")))

        # Cohen's d (positive = improvement)
        d_val = cs.cohens_d_lq_frac
        if d_val == d_val:
            d_str = f"{d_val:+.2f}"
            if d_val > 0:
                d_str = f"\\textbf{{{d_str}}}"
        else:
            d_str = "---"

        # Activity drop
        ad = cs.activity_drop_pct
        if ad == ad:
            ad_str = f"{ad:+.1f}\\%"
        else:
            ad_str = "---"

        cells = [method_lbl, delta_mean, delta_median, d_str, ad_str]
        return "    " + " & ".join(cells) + " \\\\"

    # Baseline rows first (if any)
    for name in baseline_names:
        lines.append(_build_row(name))

    if baseline_names and sorted_names:
        lines.append("    \\midrule")

    # Moderated condition rows
    for name in sorted_names:
        lines.append(_build_row(name))

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_comparison_table(
    static_stats: dict,
    dynamic_stats: dict,
    static_name: str,
    dynamic_name: str,
    quality_threshold: float = 0.39,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Build a LaTeX comparison table: static (real) vs dynamic (synthetic).

    Rows = ranking methods (matched by internal method name).
    Columns = Δ mean and Cohen's d for each evaluation mode.
    """

    # Collect method → stats for each evaluation type
    def _by_method(all_stats: dict) -> dict[str, object]:
        result = {}
        for name, cs in all_stats.items():
            if cs.condition.is_baseline:
                continue
            method = cs.condition.method
            if method:
                result[method] = cs
        return result

    static_by_method  = _by_method(static_stats)
    dynamic_by_method = _by_method(dynamic_stats)
    methods = sorted(set(static_by_method) & set(dynamic_by_method))

    if not methods:
        return ""

    # Infer shared params from dynamic stats
    mods = [dynamic_by_method[m].condition for m in methods]
    t_mods_set = sorted({c.t_mod for c in mods if c.t_mod is not None})
    top_k_set  = sorted({c.top_k for c in mods if c.top_k is not None})
    type_set   = sorted({c.mod_type for c in mods if c.mod_type is not None})
    def _join(vals) -> str:
        return ", ".join(str(v) for v in vals) if vals else "---"

    if caption is None:
        thr_pct = int(round(quality_threshold * 100))
        caption = (
            f"Static vs.\\ dynamic moderation evaluation. "
            f"Static: {_escape(static_name)} (retroactive removal); "
            f"Dynamic: {_escape(dynamic_name)} (in-simulation moderation). "
            f"Moderation at day {_join(t_mods_set)}, "
            f"top-{_join(top_k_set)} users "
            f"({_join(type_set)} strategy). "
            f"LQ threshold: quality $\\leq$ {thr_pct}/100. "
            "$\\Delta$\\,mean: post$-$pre change in daily LQ fraction "
            "($\\downarrow$~=~improvement, \\textbf{bold}). "
            "Cohen's $d$: effect size on individual action quality "
            "(positive~=~improvement)."
        )
    if label is None:
        label = "tab:static_vs_dynamic"

    col_spec = "@{}l cc cc@{}"

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\setlength{\\tabcolsep}{5pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Two-level header
    lines.append(
        "    & "
        f"\\multicolumn{{2}}{{c}}{{{_hcell('Static')}}}"
        " & "
        f"\\multicolumn{{2}}{{c}}{{{_hcell('Dynamic')}}}"
        " \\\\"
    )
    lines.append("    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    lines.append(
        "    "
        + " & ".join([
            _hcell("Method"),
            _hcell("$\\Delta$\\,mean"),
            _hcell("Cohen's $d$"),
            _hcell("$\\Delta$\\,mean"),
            _hcell("Cohen's $d$"),
        ])
        + " \\\\"
    )
    lines.append("    \\midrule")

    for method in methods:
        s_cs = static_by_method[method]
        d_cs = dynamic_by_method[method]

        method_lbl = _escape(method_display_name(method))

        # Static
        s_pre  = s_cs.pre_stats_lq_frac
        s_post = s_cs.post_stats_lq_frac
        s_delta = _delta_cell(
            s_pre.get("mean", float("nan")),
            s_post.get("mean", float("nan")),
        )
        s_d = s_cs.cohens_d_lq_frac
        s_d_str = f"{s_d:+.2f}" if s_d == s_d else "---"
        if s_d == s_d and s_d > 0:
            s_d_str = f"\\textbf{{{s_d_str}}}"

        # Dynamic
        d_pre  = d_cs.pre_stats_lq_frac
        d_post = d_cs.post_stats_lq_frac
        d_delta = _delta_cell(
            d_pre.get("mean", float("nan")),
            d_post.get("mean", float("nan")),
        )
        d_d = d_cs.cohens_d_lq_frac
        d_d_str = f"{d_d:+.2f}" if d_d == d_d else "---"
        if d_d == d_d and d_d > 0:
            d_d_str = f"\\textbf{{{d_d_str}}}"

        cells = [method_lbl, s_delta, s_d_str, d_delta, d_d_str]
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def save_latex(table_str: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table_str, encoding="utf-8")
