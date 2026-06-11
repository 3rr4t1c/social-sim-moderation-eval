"""
Coordinated accounts ranking methods.

These methods identify users who exhibit coordinated behavior patterns,
suggesting they may be part of an organized misinformation campaign.

Key metrics:
- Cosine Eigenvector: Eigenvector centrality in a co-retweet similarity network
- Cosine Max: Maximum similarity with any other user
"""

import pandas as pd
import networkx as nx
from typing import List, Tuple

from .utils import add_unranked_users, build_co_retweet_graph


def cosine_eigenvector_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    min_retweets_per_user: int = 10,  # Higher default for performance
    user_col: str = "author_id",
    tweet_col: str = "action_id",
    target_tweet_col: str = "target_action_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by eigenvector centrality in the co-retweet network.

    Builds a network where users are connected if they reshare similar content.
    Edge weights are cosine similarities of TF-IDF vectors.
    Users with high eigenvector centrality are connected to other well-connected
    users, indicating potential coordination.

    Args:
        train_data: DataFrame with reshare data
        credibility_threshold: Only consider reshares with credibility <= this
        min_retweets_per_user: Minimum reshares required to include a user
        user_col: Column name for resharer ID
        tweet_col: Column name for reshare action ID
        target_tweet_col: Column name for original tweet ID
        credibility_col: Column name for credibility score

    Returns:
        List of (user_id, eigenvector_centrality) tuples sorted descending
    """
    # Filter to low-credibility reshares
    filtered = train_data[train_data[credibility_col] <= credibility_threshold]

    # Prepare data for co-retweet graph construction
    retweets_df = filtered[[tweet_col, user_col, target_tweet_col]].copy()
    retweets_df.columns = ["author_id", "author_id_orig", "target_action_id"]

    # Build co-retweet graph
    co_rt_graph = build_co_retweet_graph(
        retweets_df,
        user_col="author_id_orig",
        tweet_col="target_action_id",
        min_retweets_per_user=min_retweets_per_user,
    )

    if len(co_rt_graph) == 0:
        ranking = []
        add_unranked_users(train_data, ranking)
        return ranking

    # Remove self-loops (similarity = 1 with self)
    co_rt_graph.remove_edges_from(nx.selfloop_edges(co_rt_graph))

    # Compute eigenvector centrality
    try:
        centrality = nx.eigenvector_centrality(co_rt_graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        # Fall back to degree centrality if eigenvector fails
        centrality = nx.degree_centrality(co_rt_graph)

    ranking = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    add_unranked_users(train_data, ranking)

    return ranking


def cosine_max_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    min_retweets_per_user: int = 10,  # Higher default for performance
    user_col: str = "author_id",
    tweet_col: str = "action_id",
    target_tweet_col: str = "target_action_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by their maximum cosine similarity with any other user.

    Users with high max similarity are likely coordinating with at least
    one other account, even if they're not central in the network.

    Args:
        train_data: DataFrame with reshare data
        credibility_threshold: Only consider reshares with credibility <= this
        min_retweets_per_user: Minimum reshares required to include a user
        user_col: Column name for resharer ID
        tweet_col: Column name for reshare action ID
        target_tweet_col: Column name for original tweet ID
        credibility_col: Column name for credibility score

    Returns:
        List of (user_id, max_similarity) tuples sorted descending
    """
    # Filter to low-credibility reshares
    filtered = train_data[train_data[credibility_col] <= credibility_threshold]

    # Prepare data for co-retweet graph construction
    retweets_df = filtered[[tweet_col, user_col, target_tweet_col]].copy()
    retweets_df.columns = ["author_id", "author_id_orig", "target_action_id"]

    # Build co-retweet graph
    co_rt_graph = build_co_retweet_graph(
        retweets_df,
        user_col="author_id_orig",
        tweet_col="target_action_id",
        min_retweets_per_user=min_retweets_per_user,
    )

    if len(co_rt_graph) == 0:
        ranking = []
        add_unranked_users(train_data, ranking)
        return ranking

    # Remove self-loops
    co_rt_graph.remove_edges_from(nx.selfloop_edges(co_rt_graph))

    # For each node, find maximum edge weight
    ranking = []
    for node in co_rt_graph.nodes():
        edges = co_rt_graph.edges(node, data=True)
        max_weight = max(
            (data.get("weight", 0) for _, _, data in edges),
            default=0,
        )
        ranking.append((node, max_weight))

    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    add_unranked_users(train_data, ranking)

    return ranking
