"""
Amplifier ranking methods.

These methods identify users who actively reshare misinformation content,
amplifying its spread through the network.

Key metrics:
- Early Reposter: Users who reshare content early (before it goes viral)
- Repost Count: Total number of low-credibility reshares
- Node Strength: Weighted degree in the reshare network
- TAR-index: Time-Aware Repost index
"""

import math
import statistics
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm

from .utils import (
    exponential_moving_average,
    add_unranked_users,
    build_reshare_network,
)


class TARIndex:
    """
    Time-Aware Repost index calculator.

    Tracks how many reshares each user makes over time slots,
    computing an aggregate at each slot and smoothing with EMA.

    A high TAR-index indicates a user who consistently reshares
    low-credibility content (amplifier).
    """

    def __init__(
        self,
        time_slot_size: Optional[float] = 60.0,
        alpha_smoothing: Optional[float] = 0.5,
        credibility_threshold: float = 39.0,
    ):
        self.time_slot_size = time_slot_size
        self.alpha_smoothing = alpha_smoothing if alpha_smoothing else 1.0
        self.credibility_threshold = credibility_threshold
        self.current_time_slot = 0

        self.author_repost_counts: dict = {}
        self.author_indices: dict = {}

    def _compute_repost_index_for_slot(self, is_training: bool) -> None:
        """Compute repost index for each author."""
        for author_id, target_counts in self.author_repost_counts.items():
            total = sum(t["count"] for t in target_counts.values())

            if author_id not in self.author_indices:
                self.author_indices[author_id] = []
            self.author_indices[author_id].append(total)

            if is_training:
                self.author_repost_counts[author_id] = {}

    def update(self, record: tuple) -> None:
        """Process a single reshare record."""
        time_delta = record[0]
        resharer_id = record[1]
        original_author_id = record[3]
        credibility = record[4]

        if credibility > self.credibility_threshold:
            return
        if original_author_id == resharer_id:
            return

        if self.time_slot_size:
            new_slot = time_delta // self.time_slot_size
            if new_slot > self.current_time_slot:
                self._compute_repost_index_for_slot(is_training=True)
                self.current_time_slot = new_slot

        # Track resharer's activity per original author (target).
        # Key: original_author_id — tracks who the resharer amplifies.
        if resharer_id not in self.author_repost_counts:
            self.author_repost_counts[resharer_id] = {}

        targets = self.author_repost_counts[resharer_id]
        if original_author_id not in targets:
            targets[original_author_id] = {"count": 0}
        targets[original_author_id]["count"] += 1

    def fit(self, data, verbose: bool = False) -> "TARIndex":
        iterator = tqdm(data) if verbose else data
        for record in iterator:
            self.update(record)
        return self

    def get_ranking(self) -> List[Tuple]:
        self._compute_repost_index_for_slot(is_training=False)

        ranking = []
        for author_id, indices in self.author_indices.items():
            score = exponential_moving_average(indices, self.alpha_smoothing)
            ranking.append((author_id, score))
            self.author_indices[author_id].pop()

        return sorted(ranking, key=lambda x: x[1], reverse=True)


class EarlyReposterModel:
    """
    Early Reposter ranking model.

    Identifies users who tend to reshare content early, before it becomes viral.
    Users who consistently reshare content when it's still new are likely to be
    deliberate amplifiers rather than passive consumers.

    Score is based on mean rank position (earlier = higher score) scaled by activity.
    """

    def __init__(self):
        # post_id -> remaining repost count (counting down)
        self.post_remaining_count: dict = {}
        # user_id -> [(post_id, rank_position), ...]
        self.user_repost_ranks: dict = {}
        self.user_scores: dict = {}
        self.max_posts = 1

    def _preload_counters(self, reshare_data) -> None:
        """First pass: count total reshares per post."""
        for record in reshare_data:
            post_id = record[2]  # target_action_id
            self.post_remaining_count[post_id] = (
                self.post_remaining_count.get(post_id, 0) + 1
            )

    def _record_rank_positions(self, reshare_data) -> None:
        """Second pass: record each user's rank positions."""
        for record in reshare_data:
            resharer_id = record[1]  # author_id
            post_id = record[2]  # target_action_id

            rank_position = self.post_remaining_count[post_id]

            if resharer_id not in self.user_repost_ranks:
                self.user_repost_ranks[resharer_id] = []
            self.user_repost_ranks[resharer_id].append((post_id, rank_position))

            self.post_remaining_count[post_id] -= 1

    def _compute_scaling_factor(self, num_posts: int, min_factor: float = 0.01) -> float:
        """
        Compute activity scaling factor.

        Uses square root scaling to balance high-activity users while
        still giving credit to lower-activity users.
        """
        num_posts = max(1, num_posts)
        self.max_posts = max(1, self.max_posts)

        scaling = math.sqrt(num_posts / self.max_posts)
        return max(scaling, min_factor)

    def _compute_scores(self) -> None:
        """Compute final scores for all users."""
        self.max_posts = max(len(ranks) for ranks in self.user_repost_ranks.values())

        for user_id, ranks in self.user_repost_ranks.items():
            positions = [pos for _, pos in ranks]
            factor = self._compute_scaling_factor(len(positions))
            self.user_scores[user_id] = statistics.mean(positions) * factor

    def fit(self, reshare_data: List[tuple]) -> "EarlyReposterModel":
        """Fit the model on reshare data (needs to be a list for two passes)."""
        self._preload_counters(reshare_data)
        self._record_rank_positions(reshare_data)
        self._compute_scores()
        return self

    def get_ranking(self) -> List[Tuple]:
        ranking = list(self.user_scores.items())
        return sorted(ranking, key=lambda x: x[1], reverse=True)


# ============================================================================
# Ranker functions with standardized interface
# ============================================================================


def repost_count_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    author_col: str = "author_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by total count of low-credibility reshares.

    Simple but effective: users who reshare a lot of misinformation rank higher.
    """
    filtered = train_data[train_data[credibility_col] <= credibility_threshold]

    counts = (
        filtered.groupby(author_col)[[credibility_col]]
        .count()
        .rename(columns={credibility_col: "score"})
        .sort_values(by="score", ascending=False)
    )

    ranking = list(counts["score"].to_dict().items())
    add_unranked_users(train_data, ranking)

    return ranking


def early_reposter_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by how early they reshare low-credibility content.

    Users who consistently reshare content before it goes viral are
    likely deliberate amplifiers.
    """
    model = EarlyReposterModel()

    filtered = train_data[train_data[credibility_col] <= credibility_threshold]
    model.fit(list(filtered.itertuples(index=False)))

    ranking = model.get_ranking()
    add_unranked_users(train_data, ranking)

    return ranking


def tar_index_ranker(
    train_data: pd.DataFrame,
    time_slot_size: float = 4.0,
    alpha_smoothing: float = 0.1,
    credibility_threshold: float = 39.0,
    **kwargs,
) -> List[Tuple]:
    """
    Time-Aware Repost (TAR) index ranker.

    Tracks resharing activity over time with EMA smoothing.

    Args:
        train_data: DataFrame with reshare data
        time_slot_size: Size of each time slot in days (default: 4.0)
        alpha_smoothing: EMA smoothing factor (default: 0.1)
        credibility_threshold: Only count low-credibility reshares

    Note:
        time_delta must be in days for default parameters to work correctly.
    """
    model = TARIndex(
        time_slot_size=time_slot_size,
        alpha_smoothing=alpha_smoothing,
        credibility_threshold=credibility_threshold,
    )
    model.fit(train_data.itertuples(index=False))
    ranking = model.get_ranking()
    add_unranked_users(train_data, ranking)
    return ranking


def node_degree_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by degree in the reshare network.

    Degree = number of unique users a node is connected to (either as
    resharer or original author).
    """
    edgelist = build_reshare_network(
        train_data,
        credibility_threshold=credibility_threshold,
    )

    if edgelist.empty:
        ranking = []
        add_unranked_users(train_data, ranking)
        return ranking

    # Count edges per node (both as source and target)
    source_counts = edgelist.groupby("source")[["weight"]].count()
    target_counts = edgelist.groupby("target")[["weight"]].count()

    combined = pd.concat([source_counts, target_counts])
    totals = combined.reset_index().groupby("index")[["weight"]].sum()
    totals = totals.sort_values(by="weight", ascending=False)

    ranking = list(totals["weight"].to_dict().items())
    add_unranked_users(train_data, ranking)

    return ranking


def node_strength_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    outgoing_only: bool = False,
    author_col: str = "author_id",
    target_col: str = "target_author_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by strength (weighted degree) in the reshare network.

    Strength = sum of edge weights connected to a node.
    """
    edgelist = build_reshare_network(
        train_data,
        author_col=author_col,
        target_col=target_col,
        credibility_col=credibility_col,
        credibility_threshold=credibility_threshold,
    )

    if edgelist.empty:
        ranking = []
        add_unranked_users(train_data, ranking)
        return ranking

    # Outgoing strength (as resharer)
    strengths = edgelist.groupby("source")[["weight"]].sum()

    if not outgoing_only:
        # Add incoming strength (as original author)
        incoming = edgelist.groupby("target")[["weight"]].sum()
        combined = pd.concat([strengths, incoming])
        strengths = combined.reset_index().groupby("index")[["weight"]].sum()

    strengths = strengths.sort_values(by="weight", ascending=False)

    ranking = list(strengths["weight"].to_dict().items())
    add_unranked_users(train_data, ranking)

    return ranking


def self_repost_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    author_col: str = "author_id",
    target_col: str = "target_author_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by count of self-reshares.

    Self-resharing low-credibility content is a suspicious pattern.
    """
    filtered = train_data[train_data[credibility_col] <= credibility_threshold]
    self_reshares = filtered[filtered[author_col] == filtered[target_col]]

    counts = (
        self_reshares.groupby(author_col)[target_col]
        .count()
        .sort_values(ascending=False)
        .to_dict()
    )

    ranking = list(counts.items())
    add_unranked_users(train_data, ranking)

    return ranking


def mean_repost_credibility_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    author_col: str = "author_id",
    target_col: str = "target_author_id",
    post_col: str = "target_action_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by mean credibility of content they reshare.

    Lower mean credibility = higher rank (more misinformation amplification).
    """
    filtered = train_data[train_data[credibility_col] <= credibility_threshold].copy()

    # Mean credibility of reshared content
    reshare_means = (
        filtered[[author_col, post_col, credibility_col]]
        .drop_duplicates()
        .groupby(author_col)[credibility_col]
        .mean()
        .to_dict()
    )

    # Mean credibility of authored content (for users not in reshare_means)
    author_means = (
        filtered[[target_col, post_col, credibility_col]]
        .drop_duplicates()
        .groupby(target_col)[credibility_col]
        .mean()
        .to_dict()
    )

    for user_id, score in author_means.items():
        if user_id not in reshare_means:
            reshare_means[user_id] = score

    # Sort ascending (lower credibility = higher rank)
    return sorted(reshare_means.items(), key=lambda x: x[1])
