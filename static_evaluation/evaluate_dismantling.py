#!/usr/bin/env python3
"""
Misinformation Reshare Network Dismantling Evaluation

CLI entry point for comparing dismantling strategies on real vs synthetic data.

Usage:
    python evaluate_dismantling.py                              # Run with defaults
    python evaluate_dismantling.py --horizons 1 7 30 60 120    # Custom horizons
    python evaluate_dismantling.py --rankers tash_index repost_count
    python evaluate_dismantling.py --observation-days 180      # Limit observation period
    python evaluate_dismantling.py --list-rankers              # Show available rankers

Examples:
    # Run horizon-based evaluation (default)
    python evaluate_dismantling.py

    # Run with specific horizons
    python evaluate_dismantling.py --horizons 1 7 30 60 120

    # Run with specific rankers
    python evaluate_dismantling.py --rankers tash_index early_reposter random_forest

    # Limit observation period to study effect of less data
    python evaluate_dismantling.py --observation-days 180

    # Specify custom data and output directories
    python evaluate_dismantling.py --data-dir ./my_data --output-dir ./my_output
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_horizon_evaluation, DEFAULT_HORIZONS
from src.ranking import RANKER_REGISTRY


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate misinformation network dismantling strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Path to data directory (default: ./data)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Path to output directory (default: ./output)",
    )

    parser.add_argument(
        "--rankers",
        nargs="+",
        default=["tash_index", "time_aware_influential", "repost_count", "cosine_eigenvector", "mean_post_credibility"],
        help="Ranker methods to evaluate (default: tash_index time_aware_influential repost_count cosine_eigenvector mean_post_credibility)",
    )

    parser.add_argument(
        "--cred-threshold",
        type=float,
        default=39.0,
        help="Credibility threshold for low-credibility content (default: 39.0)",
    )

    parser.add_argument(
        "--extra-index",
        type=int,
        default=0,
        help="Index of credibility score in the 'extra' list column (default: 0)",
    )

    parser.add_argument(
        "--default-credibility",
        type=float,
        default=100.0,
        help="Credibility assigned to missing 'extra' values. High value (100) means "
             "missing data is treated as NOT misinformation (default: 100.0)",
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help=f"Evaluation horizons in days (default: {DEFAULT_HORIZONS}). "
             "Train period = total_days - max(horizons).",
    )

    parser.add_argument(
        "--observation-days",
        type=int,
        default=None,
        help="Limit observation (train) period to this many days from start. "
             "Useful for studying effect of observing less data. "
             "Default: use all available days minus max horizon.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top nodes for effectiveness metric (default: 5). "
             "Effectiveness = %% misinformation removed after removing top K nodes.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--list-rankers",
        action="store_true",
        help="List all available rankers and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-rankers
    if args.list_rankers:
        print("\nAvailable ranking methods:")
        print("-" * 60)

        # Group by category
        categories = {}
        for name, info in RANKER_REGISTRY.items():
            cat = info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, info["description"]))

        for category, rankers in sorted(categories.items()):
            print(f"\n{category.upper()}:")
            for name, desc in sorted(rankers):
                print(f"  {name:25s} - {desc}")

        print()
        return 0

    # Validate rankers
    available = set(RANKER_REGISTRY.keys())
    invalid = [r for r in args.rankers if r not in available]

    if invalid:
        print(f"Error: Unknown rankers: {invalid}")
        print(f"Use --list-rankers to see available options")
        return 1

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    real_dir = args.data_dir / "real"
    synthetic_dir = args.data_dir / "synthetic"

    if not real_dir.exists():
        print(f"Error: Real data directory not found: {real_dir}")
        return 1

    if not synthetic_dir.exists():
        print(f"Error: Synthetic data directory not found: {synthetic_dir}")
        return 1

    # Run evaluation
    verbose = not args.quiet
    horizons = args.horizons if args.horizons else DEFAULT_HORIZONS

    if verbose:
        print("\n" + "=" * 60)
        print("MISINFORMATION NETWORK DISMANTLING EVALUATION")
        print("=" * 60)
        print(f"\nData directory:        {args.data_dir}")
        print(f"Output directory:      {args.output_dir}")
        print(f"Rankers:               {', '.join(args.rankers)}")
        print(f"Credibility threshold: {args.cred_threshold} (content with score <= this is misinformation)")
        print(f"Extra list index:      {args.extra_index}")
        print(f"Missing value default: {args.default_credibility} (for empty 'extra' field)")
        print(f"Horizons (days):       {horizons}")
        if args.observation_days:
            print(f"Observation period:    {args.observation_days} days (limited)")
        else:
            print(f"Observation period:    total_days - {max(horizons)} (auto)")
        print(f"Effectiveness metric:  % removed with top {args.top_k} nodes")
        print()

    try:
        success = run_horizon_evaluation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            ranker_names=args.rankers,
            horizons=horizons,
            observation_days=args.observation_days,
            credibility_threshold=args.cred_threshold,
            extra_index=args.extra_index,
            default_credibility=args.default_credibility,
            top_k=args.top_k,
            verbose=verbose,
        )
        if not success:
            print("\n❌ Evaluation failed: test period too short for requested horizons.")
            print("   Try reducing --horizons or using a dataset with longer time span.")
            return 1
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
