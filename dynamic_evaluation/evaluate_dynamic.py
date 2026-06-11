#!/usr/bin/env python3
"""
evaluate_dynamic.py
-------------------
CLI entry point for the static-vs-dynamic moderation evaluation.

For each synthetic network folder matching the convention
``<net>_day<N>_top<K>_<modType>/``, the script:

  1. Compares three conditions per ranking method:
       · No moderation (baseline)
       · Static moderation  (retroactive removal on no-moderation runs)
       · Dynamic moderation (in-simulation ban)
     and produces per-method time-series plots + a summary LaTeX table.

  2. Builds LQ reshare networks (pre / post moderation) and computes
     network-level metrics, exporting .gml files for Gephi.

Outputs (per comparison folder)
-------------------------------
  comparison_<method>.pdf   — three-line smoothed LQ fraction plot
  comparison_table.tex      — Δ mean + Cohen's d, static vs dynamic
  network_metrics_table.tex — network structure Δ%, static vs dynamic
  network_analysis/         — .gml files per condition

Usage
-----
  python evaluate_dynamic.py
  python evaluate_dynamic.py --threshold 0.35 --smooth-window 14
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_synt_activities
from src.latex_table import save_latex
from src.comparison import (
    compute_method_comparison,
    plot_method_comparison,
    plot_grid_comparison,
    build_comparison_table,
)
from src.network_analysis import (
    analyse_pre_post,
    aggregate_network_metrics,
    build_network_metrics_table,
    export_gml,
)
from src.static_on_synthetic import apply_static_moderation

_DYN_ROOT  = Path(__file__).parent
_SYNT_ROOT = _DYN_ROOT / "input" / "synt_data"
_OUTPUT    = _DYN_ROOT / "output"

# Matches folders like  50kNet_day186_top5_ban
_FOLDER_RE = re.compile(
    r"^(?P<net>.+?)_day(?P<t_mod>\d+)_top(?P<top_k>\d+)_(?P<mod_type>\w+)$"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Static vs dynamic moderation evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--synt-root",     type=Path, default=_SYNT_ROOT)
    p.add_argument("--output",        type=Path, default=_OUTPUT)
    p.add_argument("--threshold",     type=float, default=0.39,
                   help="LQ quality threshold (0–1 scale).")
    p.add_argument("--bin-width",     type=float, default=1.0,  dest="bin_width",
                   help="Time bin width in days.")
    p.add_argument("--smooth-window", type=int,   default=7,    dest="smooth_window",
                   help="Rolling-mean window for time-series plot.")
    p.add_argument("--confidence",    type=float, default=0.95)
    p.add_argument("--show",          action="store_true", default=False)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args      = build_parser().parse_args()
    out       = args.output.expanduser().resolve()
    synt_root = args.synt_root.expanduser().resolve()

    print(f"Output        : {out}")
    print(f"Threshold     : {args.threshold} (0–1 scale)")
    print(f"Bin width     : {args.bin_width} day(s)")
    print(f"Smooth window : {args.smooth_window} bin(s)")
    print(f"Confidence    : {args.confidence * 100:.0f}%")

    # Discover comparison folders
    comparison_dirs = (
        sorted(p for p in synt_root.iterdir()
               if p.is_dir() and not p.name.startswith("."))
        if synt_root.is_dir() else []
    )

    found_any = False

    for comp_dir in comparison_dirs:
        m = _FOLDER_RE.match(comp_dir.name)
        if not m:
            continue

        nomod_dir = comp_dir / "no_moderation"
        if not nomod_dir.is_dir():
            print(f"\n  Skipping {comp_dir.name}: no 'no_moderation' subfolder.")
            continue

        found_any = True
        net_name  = m.group("net")
        svd_t_mod = int(m.group("t_mod"))
        svd_top_k = int(m.group("top_k"))

        print(f"\n{'=' * 60}")
        print(f"  {comp_dir.name}")
        print(f"  network={net_name}  t_mod={svd_t_mod}  top_k={svd_top_k}")
        print(f"{'=' * 60}")

        # --- No-moderation run paths ---
        nomod_paths = sorted(
            r / "activities.csv"
            for r in nomod_dir.iterdir()
            if r.is_dir() and (r / "activities.csv").exists()
        )
        print(f"  No-moderation runs: {len(nomod_paths)}")

        # --- Discover method folders ---
        method_dirs = sorted(
            d for d in comp_dir.iterdir()
            if d.is_dir() and d.name != "no_moderation"
            and not d.name.startswith(".")
        )
        if not method_dirs:
            print("  No method folders found — skipping.")
            continue

        comp_out = out / comp_dir.name
        comp_out.mkdir(parents=True, exist_ok=True)

        # ==================================================================
        # 1. LQ-fraction comparison (static vs dynamic)
        # ==================================================================
        print(f"\n  --- LQ fraction comparison ---")
        all_comparisons: dict[str, dict] = {}

        for mdir in method_dirs:
            method_name = mdir.name
            dyn_paths = sorted(
                r / "activities.csv"
                for r in mdir.iterdir()
                if r.is_dir() and (r / "activities.csv").exists()
            )
            if not dyn_paths:
                continue

            print(f"    {method_name}: {len(dyn_paths)} dynamic run(s)")

            comparison = compute_method_comparison(
                nomod_run_paths=nomod_paths,
                dynamic_run_paths=dyn_paths,
                ranker_name=method_name,
                t_mod=svd_t_mod,
                top_k=svd_top_k,
                quality_threshold=args.threshold,
                credibility_threshold=args.threshold * 100,
                bin_width=args.bin_width,
            )
            all_comparisons[method_name] = comparison

            plot_method_comparison(
                comparison,
                method_name,
                t_mod=svd_t_mod,
                smooth_window=args.smooth_window,
                output_path=comp_out / f"comparison_{method_name}.pdf",
            )
            print(f"      Saved: comparison_{method_name}.pdf")

        if all_comparisons:
            plot_grid_comparison(
                all_comparisons,
                t_mod=svd_t_mod,
                smooth_window=args.smooth_window,
                output_path=comp_out / "comparison_grid.pdf",
            )
            print(f"    Saved: comparison_grid.pdf")

            table = build_comparison_table(
                all_comparisons,
                network_name=net_name,
                t_mod=svd_t_mod,
                top_k=svd_top_k,
                quality_threshold=args.threshold,
            )
            save_latex(table, comp_out / "comparison_table.tex")
            print(f"    Saved: comparison_table.tex")

        # ==================================================================
        # 2. Network metrics analysis
        # ==================================================================
        print(f"\n  --- Network metrics analysis ---")

        net_out = comp_out / "network_analysis"
        net_out.mkdir(parents=True, exist_ok=True)

        # Determine window duration from first no-mod run
        _first_df = load_synt_activities(nomod_paths[0])
        _window_days = _first_df["clock_time"].max() - svd_t_mod
        print(f"    Symmetric window: {_window_days:.0f} days each "
              f"(pre and post)")

        # No moderation baseline
        nomod_pre_metrics  = []
        nomod_post_metrics = []
        for i, p in enumerate(nomod_paths):
            df  = load_synt_activities(p)
            ppn = analyse_pre_post(df, svd_t_mod, args.threshold)
            nomod_pre_metrics.append(ppn.pre_metrics)
            nomod_post_metrics.append(ppn.post_metrics)
            if i == 0:
                export_gml(ppn.pre_graph,
                           net_out / "no_moderation" / "pre_lq_reshare.gml")
                export_gml(ppn.post_graph,
                           net_out / "no_moderation" / "post_lq_reshare.gml")

        nomod_result = {
            "pre":  aggregate_network_metrics(nomod_pre_metrics),
            "post": aggregate_network_metrics(nomod_post_metrics),
        }
        print(f"    No moderation: {len(nomod_paths)} run(s)")

        # Per-method: static + dynamic
        method_net_results: dict[str, dict] = {}

        for mdir in method_dirs:
            method_name = mdir.name

            # Static (retroactive removal on no-moderation runs)
            static_pre  = []
            static_post = []
            for i, p in enumerate(nomod_paths):
                df = load_synt_activities(p)
                filtered = apply_static_moderation(
                    df, method_name, svd_t_mod, svd_top_k,
                    credibility_threshold=args.threshold * 100,
                )
                ppn = analyse_pre_post(filtered, svd_t_mod, args.threshold)
                static_pre.append(ppn.pre_metrics)
                static_post.append(ppn.post_metrics)
                if i == 0:
                    export_gml(
                        ppn.pre_graph,
                        net_out / f"static_{method_name}" / "pre_lq_reshare.gml")
                    export_gml(
                        ppn.post_graph,
                        net_out / f"static_{method_name}" / "post_lq_reshare.gml")

            print(f"    Static  {method_name}: {len(nomod_paths)} run(s)")

            # Dynamic (in-simulation)
            dyn_paths = sorted(
                r / "activities.csv"
                for r in mdir.iterdir()
                if r.is_dir() and (r / "activities.csv").exists()
            )
            dyn_pre  = []
            dyn_post = []
            for i, p in enumerate(dyn_paths):
                df  = load_synt_activities(p)
                ppn = analyse_pre_post(df, svd_t_mod, args.threshold)
                dyn_pre.append(ppn.pre_metrics)
                dyn_post.append(ppn.post_metrics)
                if i == 0:
                    export_gml(
                        ppn.pre_graph,
                        net_out / f"dynamic_{method_name}" / "pre_lq_reshare.gml")
                    export_gml(
                        ppn.post_graph,
                        net_out / f"dynamic_{method_name}" / "post_lq_reshare.gml")

            print(f"    Dynamic {method_name}: {len(dyn_paths)} run(s)")

            method_net_results[method_name] = {
                "static": {
                    "pre":  aggregate_network_metrics(static_pre),
                    "post": aggregate_network_metrics(static_post),
                },
                "dynamic": {
                    "pre":  aggregate_network_metrics(dyn_pre),
                    "post": aggregate_network_metrics(dyn_post),
                },
            }

        if method_net_results:
            table = build_network_metrics_table(
                method_net_results,
                network_name=net_name,
                t_mod=svd_t_mod,
                nomod_result=nomod_result,
                window_days=_window_days,
            )
            save_latex(table, comp_out / "network_metrics_table.tex")
            print(f"    Saved: network_metrics_table.tex")
            print(f"    GML files: {net_out}/")

    if not found_any:
        print("\nNo comparison folders found in", synt_root)
        print("Expected folders matching: <net>_day<N>_top<K>_<modType>/")

    print("\nDone.")


if __name__ == "__main__":
    main()
