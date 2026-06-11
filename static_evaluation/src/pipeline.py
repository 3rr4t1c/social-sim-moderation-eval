"""
Pipeline for horizon-based dismantling evaluation.

Evaluates how moderation effectiveness decays over different time horizons.

Key concepts:
- Training period: observation_days OR (total_days - max_horizon)
- Test period: max_horizon days after training ends
- Each horizon evaluates a cumulative network from moderation day to that horizon
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .data_loader import (
    load_reshare_data,
    discover_simulation_folders,
    discover_real_datasets,
    load_all_runs,
)
from .ranking import get_ranker
from .ranking.utils import build_reshare_network
from .evaluation.dismantling import (
    compute_optimal_ranking,
    dismantle_network,
    trace_to_array,
)
from .evaluation.plotting import (
    aggregate_synthetic_traces,
    compute_effectiveness_at_top_k,
    plot_dismantling_comparison,
    plot_dismantling_with_confidence,
    plot_effectiveness_decay,
    plot_comparison_dismantling,
    plot_comparison_effectiveness,
)

# Default evaluation horizons (days after moderation)
DEFAULT_HORIZONS = [1, 7, 30, 60, 120]


def _format_sim_title(sim_name: str, n_runs: int) -> str:
    """
    Format a simulation folder name into a human-readable plot title.

    Words starting with a digit (e.g. "10k", "75k") keep their original case;
    all other words are title-cased.

    Examples:
        "75k_nodes_no_multi_reshare" -> "75k Nodes No Multi Reshare (n = 10 runs)"
        "10k_engage"                 -> "10k Engage (n = 3 runs)"
    """
    label = sim_name.replace("_", " ").title()
    return f"{label} (n={n_runs} runs)"


def compute_ground_truth_ranking(
    reshare_df: pd.DataFrame,
    credibility_threshold: float = 39.0,
) -> List[Tuple]:
    """
    Compute ground truth ranking based on optimal node removal order.
    """
    network = build_reshare_network(
        reshare_df,
        credibility_threshold=credibility_threshold,
    )

    if network.empty:
        return []

    optimal_df = compute_optimal_ranking(network)

    ranking = [
        (row["node"], row["outgoing_weight"] + row["incoming_weight"])
        for _, row in optimal_df.iterrows()
    ]

    return ranking


def evaluate_at_horizon(
    df: pd.DataFrame,
    ranker_names: List[str],
    train_end_time: float,
    horizon_days: int,
    credibility_threshold: float = 39.0,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Evaluate rankers at a specific horizon (cumulative network from moderation to horizon).

    Args:
        df: DataFrame with 'time_delta' in days
        ranker_names: Ranker names to evaluate
        train_end_time: End of training period (in days from start)
        horizon_days: Days after moderation to include in test network
        credibility_threshold: Threshold for low-credibility content

    Returns:
        Tuple of (traces_dict, test_network_edge_count)
    """
    time_col = "time_delta"

    # Test window: from train_end to train_end + horizon (cumulative)
    test_end = train_end_time + horizon_days

    train_df = df[df[time_col] <= train_end_time].copy()
    test_df = df[(df[time_col] > train_end_time) & (df[time_col] <= test_end)].copy()

    if len(test_df) == 0:
        return {}, 0

    # Build cumulative test network (only misinformation reshares)
    test_network = build_reshare_network(
        test_df,
        credibility_threshold=credibility_threshold,
    )

    if test_network.empty:
        # Diagnostic: all test reshares have credibility > threshold.
        # Common cause: synthetic data with missing 'extra' column uses default_credibility=100.
        n_test = len(test_df)
        n_below = (test_df["credibility_score"] <= credibility_threshold).sum()
        print(
            f"    [WARN] horizon={horizon_days}d: test network empty "
            f"({n_below}/{n_test} reshares with credibility <= {credibility_threshold}). "
            f"Check 'extra' column in data (default_credibility={test_df['credibility_score'].mean():.1f})."
        )
        return {}, 0

    # Check training data has enough misinformation for meaningful rankings
    n_train_misinfo = (train_df["credibility_score"] <= credibility_threshold).sum()
    if n_train_misinfo == 0:
        print(
            f"    [WARN] horizon={horizon_days}d: train data has 0 misinformation reshares "
            f"(credibility <= {credibility_threshold}). Rankers will produce uninformative rankings."
        )

    traces = {}

    # Ground truth on test data
    gt_ranking = compute_ground_truth_ranking(test_df, credibility_threshold)
    gt_trace = dismantle_network(test_network, gt_ranking)
    traces["Ground Truth"] = trace_to_array(gt_trace)

    # Get all nodes in test network
    test_nodes = set(test_network["source"].unique()) | set(
        test_network["target"].unique()
    )

    # Evaluate each ranker (trained on train_df, evaluated on test_network)
    for ranker_name in ranker_names:
        try:
            ranker_func = get_ranker(ranker_name)
            ranking = ranker_func(train_df, credibility_threshold=credibility_threshold)

            # Add nodes missing from ranking (new users in test period).
            # Sort for determinism: same ranking order regardless of set hash state.
            ranked_nodes = {node for node, _ in ranking}
            missing_nodes = sorted(test_nodes - ranked_nodes)
            if missing_nodes:
                min_score = min((score for _, score in ranking), default=0.0)
                for node in missing_nodes:
                    ranking.append((node, min_score - 1))

            trace = dismantle_network(test_network, ranking)
            traces[ranker_name] = trace_to_array(trace)
        except Exception as e:
            print(f"    Warning: {ranker_name} failed at horizon {horizon_days}d: {e}")

    return traces, len(test_network)


def evaluate_horizons(
    df: pd.DataFrame,
    ranker_names: List[str],
    horizons: List[int],
    observation_days: Optional[int] = None,
    credibility_threshold: float = 39.0,
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[
    Dict[int, Dict[str, np.ndarray]],
    Dict[str, Dict[int, float]],
    int,
    int,
]:
    """
    Evaluate rankers at multiple temporal horizons.

    Training period is determined by:
    - If observation_days is set: use exactly observation_days
    - Otherwise: total_days - max(horizons)

    Args:
        df: DataFrame with 'time_delta' in days
        ranker_names: Ranker names to evaluate
        horizons: List of horizon values in days
        observation_days: Optional fixed observation period (days)
        credibility_threshold: Threshold for low-credibility content
        top_k: Number of top nodes for effectiveness calculation
        verbose: Whether to show progress

    Returns:
        Tuple of:
        - traces_by_horizon: {horizon: {method: trace_array}}
        - effectiveness_by_method: {method: {horizon: effectiveness}}
        - train_days: Actual training period used
        - available_test_days: Days available for testing
    """
    time_col = "time_delta"
    min_time = df[time_col].min()
    max_time = df[time_col].max()
    total_days = int(max_time - min_time)

    max_horizon = max(horizons)

    # Determine training period
    if observation_days is not None:
        train_days = observation_days
    else:
        train_days = total_days - max_horizon

    train_end_time = min_time + train_days
    available_test_days = int(max_time - train_end_time)

    if verbose:
        print(f"  Total data span: {total_days} days")
        print(f"  Training (observation) period: {train_days} days")
        print(f"  Test period available: {available_test_days} days")
        print(f"  Max horizon requested: {max_horizon} days")

    # Validate
    if available_test_days < max_horizon:
        print(f"\n⚠️  WARNING: Test period too short!")
        print(f"   Available: {available_test_days} days")
        print(f"   Required for max horizon: {max_horizon} days")
        return {}, {}, train_days, available_test_days

    traces_by_horizon = {}
    effectiveness_by_method = {name: {} for name in ["Ground Truth"] + ranker_names}

    for horizon in horizons:
        if horizon > available_test_days:
            if verbose:
                print(f"  Skipping horizon {horizon}d (exceeds available)")
            continue

        if verbose:
            print(f"  Evaluating horizon: {horizon} days...")

        traces, network_size = evaluate_at_horizon(
            df, ranker_names, train_end_time, horizon, credibility_threshold
        )

        if not traces:
            if verbose:
                print(f"    No data for horizon {horizon}d")
            continue

        traces_by_horizon[horizon] = traces

        # Compute effectiveness
        for method_name, trace in traces.items():
            eff = compute_effectiveness_at_top_k(trace, top_k)
            effectiveness_by_method[method_name][horizon] = eff

        if verbose:
            print(f"    Network: {network_size} edges")

    return traces_by_horizon, effectiveness_by_method, train_days, available_test_days


def evaluate_simulation_horizons(
    simulation_dir: Path,
    ranker_names: List[str],
    horizons: List[int],
    observation_days: Optional[int] = None,
    credibility_threshold: float = 39.0,
    extra_index: int = 0,
    default_credibility: float = 100.0,
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[
    Dict[int, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    Dict[str, Dict[int, float]],
    int,
]:
    """
    Evaluate simulation at multiple horizons, aggregating across runs.

    Returns:
        Tuple of:
        - traces_by_horizon: {horizon: (median, ci_lower, ci_upper)}
        - effectiveness_median: {method: {horizon: median}}
        - effectiveness_ci_lower: {method: {horizon: ci_lower}}
        - effectiveness_ci_upper: {method: {horizon: ci_upper}}
        - available_days: Days available for testing
    """
    runs = load_all_runs(
        simulation_dir,
        verbose,
        extra_index=extra_index,
        default_credibility=default_credibility,
        time_unit="days",
    )

    if not runs:
        raise ValueError(f"No runs found in {simulation_dir}")

    n_runs = len(runs)

    all_traces_by_horizon: Dict[int, Dict[str, List[np.ndarray]]] = {}
    all_effectiveness: Dict[str, Dict[int, List[float]]] = {}
    min_available_days = float("inf")

    for run_name, df in tqdm(runs, disable=not verbose, desc="  Runs"):
        traces_by_horizon, effectiveness, _, available_days = evaluate_horizons(
            df,
            ranker_names,
            horizons,
            observation_days,
            credibility_threshold,
            top_k,
            verbose=False,
        )

        min_available_days = min(min_available_days, available_days)

        for horizon, traces in traces_by_horizon.items():
            if horizon not in all_traces_by_horizon:
                all_traces_by_horizon[horizon] = {}
            for method, trace in traces.items():
                if method not in all_traces_by_horizon[horizon]:
                    all_traces_by_horizon[horizon][method] = []
                all_traces_by_horizon[horizon][method].append(trace)

        for method, horizon_values in effectiveness.items():
            if method not in all_effectiveness:
                all_effectiveness[method] = {}
            for horizon, eff in horizon_values.items():
                if horizon not in all_effectiveness[method]:
                    all_effectiveness[method][horizon] = []
                all_effectiveness[method][horizon].append(eff)

    # Aggregate traces
    aggregated_traces: Dict[int, Tuple[Dict, Dict, Dict]] = {}
    for horizon, method_traces in all_traces_by_horizon.items():
        median_traces = {}
        ci_lower_traces = {}
        ci_upper_traces = {}
        for method, traces_list in method_traces.items():
            median, ci_lower, ci_upper, _ = aggregate_synthetic_traces(traces_list)
            median_traces[method] = median
            ci_lower_traces[method] = ci_lower
            ci_upper_traces[method] = ci_upper
        aggregated_traces[horizon] = (median_traces, ci_lower_traces, ci_upper_traces)

    # Aggregate effectiveness
    effectiveness_median: Dict[str, Dict[int, float]] = {}
    effectiveness_ci_lower: Dict[str, Dict[int, float]] = {}
    effectiveness_ci_upper: Dict[str, Dict[int, float]] = {}

    for method, horizon_values in all_effectiveness.items():
        effectiveness_median[method] = {}
        effectiveness_ci_lower[method] = {}
        effectiveness_ci_upper[method] = {}
        for horizon, values in horizon_values.items():
            effectiveness_median[method][horizon] = np.median(values)
            effectiveness_ci_lower[method][horizon] = np.percentile(values, 5)
            effectiveness_ci_upper[method][horizon] = np.percentile(values, 95)

    return (
        aggregated_traces,
        effectiveness_median,
        effectiveness_ci_lower,
        effectiveness_ci_upper,
        int(min_available_days),
        n_runs,
    )


def run_horizon_evaluation(
    data_dir: Path,
    output_dir: Path,
    ranker_names: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    observation_days: Optional[int] = None,
    credibility_threshold: float = 39.0,
    extra_index: int = 0,
    default_credibility: float = 100.0,
    top_k: int = 5,
    verbose: bool = True,
) -> bool:
    """
    Run horizon-based evaluation pipeline.

    Args:
        data_dir: Path to data directory with 'real' and 'synthetic' subdirs
        output_dir: Path to output directory
        ranker_names: Ranker names (default: tash_index, repost_count, cosine_eigenvector)
        horizons: Horizon values in days (default: [1, 7, 30, 60, 120])
        observation_days: Optional fixed observation period
        credibility_threshold: Threshold for low-credibility content
        extra_index: Index of credibility score in 'extra' list
        default_credibility: Default credibility for missing values
        top_k: Number of top nodes for effectiveness metric
        verbose: Whether to show progress

    Returns:
        True if successful, False if test period too short
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ranker_names is None:
        ranker_names = ["tash_index", "time_aware_influential", "repost_count", "cosine_eigenvector", "mean_post_credibility"]

    if horizons is None:
        horizons = DEFAULT_HORIZONS

    horizons = sorted(horizons)
    max_horizon = max(horizons)

    real_dir = data_dir / "real"
    synthetic_dir = data_dir / "synthetic"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    real_output = output_dir / "real"
    synthetic_output = output_dir / "synthetic"
    real_output.mkdir(exist_ok=True)
    synthetic_output.mkdir(exist_ok=True)

    if verbose:
        print("=" * 60)
        print("HORIZON-BASED EVALUATION")
        print("=" * 60)
        print(f"Horizons: {horizons} days")
        print(f"Effectiveness metric: % removed with top {top_k} nodes")
        print(f"Rankers: {', '.join(ranker_names)}")
        if observation_days:
            print(f"Fixed observation period: {observation_days} days")

    # ==================== REAL DATA ====================
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATING REAL DATA")
        print("=" * 60)

    real_datasets = discover_real_datasets(real_dir)
    all_real_results = {}  # Store for comparison plots

    for dataset_path in real_datasets:
        dataset_name = dataset_path.stem

        if verbose:
            print(f"\nDataset: {dataset_name}")

        df = load_reshare_data(
            dataset_path,
            extra_index=extra_index,
            default_credibility=default_credibility,
            time_unit="days",
        )

        traces_by_horizon, effectiveness, train_days, available_days = evaluate_horizons(
            df,
            ranker_names,
            horizons,
            observation_days,
            credibility_threshold,
            top_k,
            verbose,
        )

        if available_days < max_horizon:
            print(f"\n❌ Test period too short for {dataset_name}")
            return False

        # Store for comparison
        all_real_results[dataset_name] = {
            "traces_by_horizon": traces_by_horizon,
            "effectiveness": effectiveness,
        }

        # Save individual plots
        dataset_output = real_output / dataset_name
        dataset_output.mkdir(exist_ok=True)

        for horizon, traces in traces_by_horizon.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_dismantling_comparison(traces, title=f"{horizon}d after moderation", ax=ax)
            fig.savefig(dataset_output / f"dismantling_{horizon}d.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)

        valid_horizons = [h for h in horizons if h in effectiveness.get("Ground Truth", {})]
        if valid_horizons:
            plot_effectiveness_decay(
                effectiveness,
                valid_horizons,
                title=f"{dataset_name} - Effectiveness Decay",
                top_k=top_k,
                output_path=dataset_output / "effectiveness_decay.pdf",
            )
            plt.close("all")

    # ==================== SYNTHETIC DATA ====================
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATING SYNTHETIC DATA")
        print("=" * 60)

    simulations = discover_simulation_folders(synthetic_dir)
    all_synthetic_results = {}

    for sim_dir in simulations:
        sim_name = sim_dir.name.replace("_simulation", "")

        if verbose:
            print(f"\nSimulation: {sim_name}")

        (
            aggregated_traces,
            effectiveness_median,
            effectiveness_ci_lower,
            effectiveness_ci_upper,
            available_days,
            n_runs,
        ) = evaluate_simulation_horizons(
            sim_dir,
            ranker_names,
            horizons,
            observation_days,
            credibility_threshold,
            extra_index,
            default_credibility,
            top_k,
            verbose,
        )

        if available_days < max_horizon:
            print(f"\n❌ Test period too short for {sim_name}")
            return False

        # Store for comparison
        all_synthetic_results[sim_name] = {
            "traces_by_horizon": aggregated_traces,
            "effectiveness_median": effectiveness_median,
            "effectiveness_ci_lower": effectiveness_ci_lower,
            "effectiveness_ci_upper": effectiveness_ci_upper,
            "n_runs": n_runs,
        }

        # Save individual plots
        sim_output = synthetic_output / sim_name
        sim_output.mkdir(exist_ok=True)

        for horizon, (median, ci_lower, ci_upper) in aggregated_traces.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_dismantling_with_confidence(
                median, ci_lower, ci_upper, title=f"{horizon}d after moderation", ax=ax
            )
            fig.savefig(sim_output / f"dismantling_{horizon}d.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)

        valid_horizons = [h for h in horizons if h in effectiveness_median.get("Ground Truth", {})]
        if valid_horizons:
            plot_effectiveness_decay(
                effectiveness_median,
                valid_horizons,
                ci_lower=effectiveness_ci_lower,
                ci_upper=effectiveness_ci_upper,
                title=f"{sim_name} - Effectiveness Decay",
                top_k=top_k,
                output_path=sim_output / "effectiveness_decay.pdf",
            )
            plt.close("all")

    # ==================== COMPARISON PLOTS ====================
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 60)

    # For each real dataset x synthetic simulation pair
    for real_name, real_data in all_real_results.items():
        for sim_name, syn_data in all_synthetic_results.items():
            if verbose:
                print(f"  {real_name} vs {sim_name}")

            # Use max horizon for comparison dismantling plot
            max_h = max(h for h in horizons if h in real_data["traces_by_horizon"])
            real_traces = real_data["traces_by_horizon"].get(max_h, {})
            syn_traces = syn_data["traces_by_horizon"].get(max_h, ({}, {}, {}))

            if real_traces and syn_traces[0]:
                plot_comparison_dismantling(
                    real_traces,
                    syn_traces[0],  # median
                    syn_traces[1],  # ci_lower
                    syn_traces[2],  # ci_upper
                    real_title="Real Data",
                    synthetic_title=_format_sim_title(sim_name, syn_data["n_runs"]),
                    output_path=output_dir / f"comparison_dismantling_{real_name}_{sim_name}.pdf",
                )
                plt.close("all")

            # Effectiveness comparison
            valid_horizons = sorted(
                set(real_data["effectiveness"].get("Ground Truth", {}).keys())
                & set(syn_data["effectiveness_median"].get("Ground Truth", {}).keys())
            )

            if valid_horizons:
                plot_comparison_effectiveness(
                    real_data["effectiveness"],
                    syn_data["effectiveness_median"],
                    valid_horizons,
                    synthetic_ci_lower=syn_data["effectiveness_ci_lower"],
                    synthetic_ci_upper=syn_data["effectiveness_ci_upper"],
                    real_title="Real Data",
                    synthetic_title=_format_sim_title(sim_name, syn_data["n_runs"]),
                    top_k=top_k,
                    output_path=output_dir / f"comparison_effectiveness_{real_name}_{sim_name}.pdf",
                )
                plt.close("all")

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Output saved to: {output_dir}")

    return True
