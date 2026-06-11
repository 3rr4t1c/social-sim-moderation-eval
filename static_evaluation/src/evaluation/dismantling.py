"""
Network dismantling algorithms and metrics.

Dismantling is the process of iteratively removing nodes from a network
to measure how effectively a ranking identifies key misinformation spreaders.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import ndcg_score


def compute_optimal_ranking(edgelist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the optimal node removal order for dismantling.

    The optimal ranking is based on total node strength (sum of incident
    edge weights), prioritizing nodes with high outgoing weight.

    Args:
        edgelist_df: DataFrame with columns [source, target, weight]

    Returns:
        DataFrame with columns [node, outgoing_weight, incoming_weight]
        sorted by outgoing_weight then incoming_weight (descending)
    """
    # Sum of outgoing edge weights
    outgoing = edgelist_df.groupby("source")["weight"].sum().reset_index()
    outgoing.columns = ["node", "outgoing_weight"]

    # Sum of incoming edge weights
    incoming = edgelist_df.groupby("target")["weight"].sum().reset_index()
    incoming.columns = ["node", "incoming_weight"]

    # Merge to get total weights per node
    ranking = pd.merge(
        outgoing,
        incoming,
        on="node",
        how="outer",
    )
    ranking = ranking.fillna(0)

    # Sort by outgoing (primary), incoming (secondary), node ID (tiebreaker for determinism).
    ranking = ranking.sort_values(
        by=["outgoing_weight", "incoming_weight", "node"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    return ranking


def dismantle_network(
    network_df: pd.DataFrame,
    ranking: List[Tuple],
    max_removals: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Perform network dismantling by iteratively removing nodes.

    Removes nodes in the order specified by the ranking and tracks
    the remaining misinformation fraction after each removal.

    Args:
        network_df: DataFrame with columns [source, target, weight]
        ranking: List of (node_id, score) tuples in removal order
        max_removals: Maximum number of nodes to remove (None = remove all)

    Returns:
        List of (node_id, remaining_fraction) tuples.
        First element is always ("FULL", 1.0) representing the initial state.
    """
    total_weight = network_df["weight"].sum()

    if total_weight == 0:
        return [("FULL", 1.0)]

    # Build efficient data structures for O(1) lookups
    # node -> list of (edge_index, weight, is_source)
    node_to_edges: Dict[str, List[Tuple[int, float]]] = {}
    edge_weights = network_df["weight"].values.copy()
    sources = network_df["source"].values
    targets = network_df["target"].values

    for idx in range(len(network_df)):
        src, tgt, w = sources[idx], targets[idx], edge_weights[idx]
        if src not in node_to_edges:
            node_to_edges[src] = []
        if tgt not in node_to_edges:
            node_to_edges[tgt] = []
        node_to_edges[src].append((idx, w))
        node_to_edges[tgt].append((idx, w))

    removed_edges = set()
    current_weight = total_weight
    trace = [("FULL", 1.0)]

    # Track nodes in the network for consistent trace length
    all_network_nodes = set(node_to_edges.keys())

    for i, (node_id, _) in enumerate(ranking, start=1):
        if max_removals is not None and i > max_removals:
            break

        # Only process nodes that are in the network
        if node_id not in all_network_nodes:
            continue

        # Remove all edges involving this node
        weight_removed = 0.0
        if node_id in node_to_edges:
            for edge_idx, w in node_to_edges[node_id]:
                if edge_idx not in removed_edges:
                    removed_edges.add(edge_idx)
                    weight_removed += w
            del node_to_edges[node_id]

        # Always add to trace for nodes in network (even if no weight removed)
        # This ensures all rankings have same trace length
        if weight_removed > 0:
            current_weight -= weight_removed
        remaining = current_weight / total_weight
        trace.append((node_id, remaining))

    return trace


def compute_dismantling_trace(
    reshare_df: pd.DataFrame,
    ranking: List[Tuple],
    credibility_threshold: float = 39.0,
    author_col: str = "author_id",
    target_col: str = "target_author_id",
    credibility_col: str = "credibility_score",
) -> List[Tuple[str, float]]:
    """
    Compute dismantling trace from raw reshare data and a ranking.

    Convenience function that builds the network and performs dismantling.

    Args:
        reshare_df: DataFrame with reshare data
        ranking: List of (user_id, score) tuples in removal order
        credibility_threshold: Only include reshares with credibility <= this
        author_col: Column name for resharer ID
        target_col: Column name for original author ID
        credibility_col: Column name for credibility score

    Returns:
        Dismantling trace as list of (node_id, remaining_fraction) tuples
    """
    from ..ranking.utils import build_reshare_network

    network = build_reshare_network(
        reshare_df,
        author_col=author_col,
        target_col=target_col,
        credibility_col=credibility_col,
        credibility_threshold=credibility_threshold,
    )

    return dismantle_network(network, ranking)


def compute_ndcg_score(
    true_ranking: List[Tuple],
    test_ranking: List[Tuple],
    k: Optional[int] = None,
) -> float:
    """
    Compute NDCG score between two rankings.

    Normalized Discounted Cumulative Gain measures how well the test ranking
    approximates the true ranking, with emphasis on top positions.

    Args:
        true_ranking: Ground truth ranking as list of (id, score) tuples
        test_ranking: Test ranking as list of (id, score) tuples
        k: If specified, compute NDCG@k

    Returns:
        NDCG score between 0 and 1
    """
    true_dict = dict(true_ranking)
    test_dict = dict(test_ranking)

    # Align rankings on test keys
    all_keys = set(test_dict.keys())

    true_scores = [true_dict.get(k, 0) for k in all_keys]
    test_scores = [test_dict.get(k, 0) for k in all_keys]

    if len(true_scores) == 0:
        return 0.0

    return ndcg_score([true_scores], [test_scores], k=k, ignore_ties=False)


def trace_to_array(trace: List[Tuple[str, float]]) -> np.ndarray:
    """
    Convert a dismantling trace to a numpy array of remaining fractions.

    Args:
        trace: List of (node_id, remaining_fraction) tuples

    Returns:
        Array of remaining fractions (including initial 1.0)
    """
    return np.array([frac for _, frac in trace])
