"""
Utility functions shared across ranking methods.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional


def exponential_moving_average(values: np.ndarray, alpha: float) -> float:
    """
    Compute the Exponential Moving Average (EMA) of a sequence.

    The EMA gives more weight to recent values, with the decay controlled by alpha.
    Values are assumed to be in chronological order (most recent last).

    Args:
        values: Array of numeric values
        alpha: Smoothing/decay factor (0 < alpha <= 1). The most recent value
            receives weight (1 - alpha) and older values decay by a factor of
            alpha per step, so a *lower* alpha puts *more* weight on recent
            values. The weights sum to 1.

    Returns:
        The EMA value as a single float
    """
    values = np.asarray(values)
    n = len(values)

    # Compute weights: older values get exponentially less weight
    weights = (1 - alpha) * (alpha ** np.arange(n - 1, -1, -1))
    weights[0] = alpha ** (n - 1)

    return np.dot(values, weights)


def add_unranked_users(
    reshares_df: pd.DataFrame,
    ranking: List[Tuple],
    author_col: str = "author_id",
    target_col: str = "target_author_id",
    min_score: float = 0.0,
) -> None:
    """
    Add users missing from the ranking with a minimum score.

    This ensures all users from the dataset appear in the ranking, even if they
    weren't captured by the ranking algorithm. Missing users are added at the
    end with the minimum score (creating ties).

    Args:
        reshares_df: DataFrame with reshare data
        ranking: List of (user_id, score) tuples to modify in-place
        author_col: Column name for reshare authors
        target_col: Column name for original post authors
        min_score: Score to assign to unranked users
    """
    all_users = pd.concat([
        reshares_df[author_col],
        reshares_df[target_col]
    ]).unique()

    ranked_users = {user_id for user_id, _ in ranking}

    for user_id in all_users:
        if user_id not in ranked_users:
            ranking.append((user_id, min_score))


def split_data(
    df: pd.DataFrame,
    tail_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into head and tail portions by rows.

    Used for train/test splitting in temporal data where we want to
    train on earlier data and evaluate on later data.

    Args:
        df: DataFrame to split
        tail_ratio: Fraction of data to put in tail (0 to 1)

    Returns:
        Tuple of (head_df, tail_df)
    """
    split_idx = len(df) - int(len(df) * tail_ratio)

    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_co_retweet_graph(
    retweets: pd.DataFrame,
    user_col: str = "author_id",
    tweet_col: str = "target_action_id",
    min_retweets_per_user: int = 20,
    min_retweets_per_tweet: int = 1,
) -> nx.Graph:
    """
    Build a co-retweet network where users are linked by cosine similarity.

    Two users are connected if they retweeted similar content. Edge weights
    represent the TF-IDF weighted cosine similarity of their retweet patterns.

    Based on: https://github.com/ValeriaPante/coordinatedActivity

    Args:
        retweets: DataFrame with retweet data
        user_col: Column name for user IDs
        tweet_col: Column name for retweeted tweet IDs
        min_retweets_per_user: Minimum retweets required to include a user
        min_retweets_per_tweet: Minimum retweets required to include a tweet

    Returns:
        NetworkX Graph with users as nodes and cosine similarity as edge weights
    """
    # Filter users with sufficient activity
    user_counts = retweets.groupby(user_col)[tweet_col].count()
    active_users = user_counts[user_counts >= min_retweets_per_user].index

    filtered = retweets[retweets[user_col].isin(active_users)].drop_duplicates(
        subset=[user_col, tweet_col]
    )

    # Filter tweets with sufficient retweets
    tweet_counts = filtered.groupby(tweet_col)[user_col].count()
    popular_tweets = tweet_counts[tweet_counts > min_retweets_per_tweet].index

    filtered = filtered[filtered[tweet_col].isin(popular_tweets)]

    if len(filtered) == 0:
        return nx.Graph()

    # Create mappings for sparse matrix construction
    user_map = {uid: idx for idx, uid in enumerate(filtered[user_col].astype(str).unique())}
    tweet_map = {tid: idx for idx, tid in enumerate(filtered[tweet_col].unique())}

    # Build sparse user-tweet matrix
    rows = filtered[user_col].astype(str).map(user_map).values
    cols = filtered[tweet_col].map(tweet_map).values

    sparse_matrix = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(user_map), len(tweet_map)),
    )

    # Apply TF-IDF weighting and compute cosine similarity
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    # Convert to graph
    df_adj = pd.DataFrame(
        similarities.toarray(),
        index=user_map.keys(),
        columns=user_map.keys()
    )

    G = nx.from_pandas_adjacency(df_adj)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def build_reshare_network(
    reshares_df: pd.DataFrame,
    author_col: str = "author_id",
    target_col: str = "target_author_id",
    credibility_col: str = "credibility_score",
    credibility_threshold: Optional[float] = None,
    exclude_self_loops: bool = True,
    return_dataframe: bool = True,
) -> pd.DataFrame:
    """
    Build an edge list from reshare data.

    Creates a weighted directed graph where edges represent resharing relationships.
    Edge weight is the count of reshares between two users.

    Args:
        reshares_df: DataFrame with reshare data
        author_col: Column for reshare author
        target_col: Column for original post author
        credibility_col: Column for credibility scores
        credibility_threshold: If set, only include reshares with credibility <= threshold
        exclude_self_loops: Whether to exclude self-reshares
        return_dataframe: If True, return DataFrame; else return dict

    Returns:
        DataFrame with columns [source, target, weight] or dict of edge weights
    """
    df = reshares_df

    if credibility_threshold is not None:
        df = df[df[credibility_col] <= credibility_threshold]

    edges: Dict[Tuple, int] = {}

    for _, row in df[[target_col, author_col]].iterrows():
        source, target = row[target_col], row[author_col]

        if exclude_self_loops and source == target:
            continue

        edge = (source, target)
        edges[edge] = edges.get(edge, 0) + 1

    if return_dataframe:
        return pd.DataFrame(
            [(src, tgt, w) for (src, tgt), w in edges.items()],
            columns=["source", "target", "weight"],
        )

    return edges


def h_index_bisect(sorted_list: List, key=lambda x: x) -> int:
    """
    Compute h-index using binary search.

    Given a descending sorted list, finds the h-index: the largest h such that
    there are at least h items with value >= h.

    Args:
        sorted_list: List sorted in descending order by the key function
        key: Function to extract the comparable value from each item

    Returns:
        The h-index value
    """
    lo, hi = 0, len(sorted_list)

    while lo < hi:
        mid = (lo + hi) // 2
        value = key(sorted_list[mid])

        if value > mid:
            lo = mid + 1
        elif value < mid:
            hi = mid
        else:
            return mid

    return lo
