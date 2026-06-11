"""
Network dismantling algorithms and metrics.

Dismantling is the process of iteratively removing nodes from a network
to measure how effectively a ranking identifies key misinformation spreaders.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


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


def trace_to_array(trace: List[Tuple[str, float]]) -> np.ndarray:
    """
    Convert a dismantling trace to a numpy array of remaining fractions.

    Args:
        trace: List of (node_id, remaining_fraction) tuples

    Returns:
        Array of remaining fractions (including initial 1.0)
    """
    return np.array([frac for _, frac in trace])
