"""
Superspreader ranking methods.

These methods identify users who create viral misinformation content that gets widely reshared.

Key metrics:
- TASH-index: Time-Aware Social H-index
- TAI-index: Time-Aware Influential index
- Social H-index: Static version without time awareness
- Influential index: Total reshare count received
"""

import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm

from .utils import exponential_moving_average, add_unranked_users, h_index_bisect


class TASHIndex:
    """
    Time-Aware Social H-index calculator.

    Tracks how many reshares each user's posts receive over time slots,
    computing an h-index at each slot and smoothing with EMA.

    A high TASH-index indicates a user who consistently creates content
    that gets widely reshared (superspreader).
    """

    def __init__(
        self,
        time_slot_size: Optional[float] = 15.0,
        alpha_smoothing: Optional[float] = 0.4,
        credibility_threshold: float = 39.0,
    ):
        """
        Args:
            time_slot_size: Duration of each time slot (in data time units).
                           None disables time slotting.
            alpha_smoothing: EMA smoothing factor (0-1). None uses simple average.
            credibility_threshold: Only count reshares with credibility <= this value
        """
        self.time_slot_size = time_slot_size
        self.alpha_smoothing = alpha_smoothing if alpha_smoothing else 1.0
        self.credibility_threshold = credibility_threshold
        self.current_time_slot = 0

        # author_id -> {post_id -> {"count": int}}
        self.author_post_counts: dict = {}
        # author_id -> [h_index_per_slot]
        self.author_h_indices: dict = {}

    def _compute_h_indices_for_slot(self, is_training: bool) -> None:
        """Compute h-index for each author and optionally reset counts."""
        for author_id, post_counts in self.author_post_counts.items():
            counts = list(post_counts.values())
            counts.sort(key=lambda x: x["count"], reverse=True)

            h_idx = h_index_bisect(counts, key=lambda x: x["count"])

            if author_id not in self.author_h_indices:
                self.author_h_indices[author_id] = []
            self.author_h_indices[author_id].append(h_idx)

            if is_training:
                self.author_post_counts[author_id] = {}

    def update(self, record: tuple) -> None:
        """
        Process a single reshare record.

        Expected tuple format (index=False): (time_delta, author_id, target_action_id, target_author_id, credibility_score, ...)
        """
        time_delta = record[0]
        resharer_id = record[1]
        post_id = record[2]
        original_author_id = record[3]
        credibility = record[4]

        # Skip high-credibility content and self-reshares
        if credibility > self.credibility_threshold:
            return
        if original_author_id == resharer_id:
            return

        # Check for time slot change
        if self.time_slot_size:
            new_slot = time_delta // self.time_slot_size
            if new_slot > self.current_time_slot:
                self._compute_h_indices_for_slot(is_training=True)
                self.current_time_slot = new_slot

        # Initialize tracking for original author if needed
        if original_author_id not in self.author_post_counts:
            self.author_post_counts[original_author_id] = {}

        # Increment reshare count for this post
        posts = self.author_post_counts[original_author_id]
        if post_id not in posts:
            posts[post_id] = {"count": 0}
        posts[post_id]["count"] += 1

        # Also track the resharer (they might become tracked later)
        if resharer_id not in self.author_post_counts:
            self.author_post_counts[resharer_id] = {}

    def fit(self, data, verbose: bool = False) -> "TASHIndex":
        """Fit the model on reshare data."""
        iterator = tqdm(data) if verbose else data
        for record in iterator:
            self.update(record)
        return self

    def get_ranking(self) -> List[Tuple]:
        """
        Compute final ranking after processing all data.

        Returns:
            List of (author_id, tash_score) tuples sorted by score descending
        """
        # Final flush for current time slot
        self._compute_h_indices_for_slot(is_training=False)

        ranking = []
        for author_id, h_indices in self.author_h_indices.items():
            score = exponential_moving_average(h_indices, self.alpha_smoothing)
            ranking.append((author_id, score))
            # Remove last index to allow re-use
            self.author_h_indices[author_id].pop()

        return sorted(ranking, key=lambda x: x[1], reverse=True)


class TAIIndex:
    """
    Time-Aware Influential index calculator.

    Similar to TASH but uses total reshare count instead of h-index.
    A simpler metric that captures overall influence without the h-index structure.
    """

    def __init__(
        self,
        time_slot_size: Optional[float] = 30.0,
        alpha_smoothing: Optional[float] = 0.5,
        credibility_threshold: float = 39.0,
    ):
        self.time_slot_size = time_slot_size
        self.alpha_smoothing = alpha_smoothing if alpha_smoothing else 1.0
        self.credibility_threshold = credibility_threshold
        self.current_time_slot = 0

        self.author_post_counts: dict = {}
        self.author_influence_indices: dict = {}

    def _compute_influence_for_slot(self, is_training: bool) -> None:
        """Compute influence index (total reshares) for each author."""
        for author_id, post_counts in self.author_post_counts.items():
            total_reshares = sum(p["count"] for p in post_counts.values())

            if author_id not in self.author_influence_indices:
                self.author_influence_indices[author_id] = []
            self.author_influence_indices[author_id].append(total_reshares)

            if is_training:
                self.author_post_counts[author_id] = {}

    def update(self, record: tuple) -> None:
        """Process a single reshare record."""
        time_delta = record[0]
        resharer_id = record[1]
        post_id = record[2]
        original_author_id = record[3]
        credibility = record[4]

        if credibility > self.credibility_threshold:
            return
        if original_author_id == resharer_id:
            return

        if self.time_slot_size:
            new_slot = time_delta // self.time_slot_size
            if new_slot > self.current_time_slot:
                self._compute_influence_for_slot(is_training=True)
                self.current_time_slot = new_slot

        if original_author_id not in self.author_post_counts:
            self.author_post_counts[original_author_id] = {}

        posts = self.author_post_counts[original_author_id]
        if post_id not in posts:
            posts[post_id] = {"count": 0}
        posts[post_id]["count"] += 1

        if resharer_id not in self.author_post_counts:
            self.author_post_counts[resharer_id] = {}

    def fit(self, data, verbose: bool = False) -> "TAIIndex":
        iterator = tqdm(data) if verbose else data
        for record in iterator:
            self.update(record)
        return self

    def get_ranking(self) -> List[Tuple]:
        self._compute_influence_for_slot(is_training=False)

        ranking = []
        for author_id, indices in self.author_influence_indices.items():
            score = exponential_moving_average(indices, self.alpha_smoothing)
            ranking.append((author_id, score))
            self.author_influence_indices[author_id].pop()

        return sorted(ranking, key=lambda x: x[1], reverse=True)


# ============================================================================
# Ranker functions with standardized interface
# ============================================================================


def social_h_index_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    **kwargs,
) -> List[Tuple]:
    """
    Static Social H-index ranker (no time awareness).

    Computes a single h-index over all data for each user based on
    how many of their posts received at least h reshares.
    """
    model = TASHIndex(
        time_slot_size=None,
        alpha_smoothing=None,
        credibility_threshold=credibility_threshold,
    )
    model.fit(train_data.itertuples(index=False))
    ranking = model.get_ranking()
    add_unranked_users(train_data, ranking)
    return ranking


def influential_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    **kwargs,
) -> List[Tuple]:
    """
    Static Influential index ranker (no time awareness).

    Ranks users by total reshares received on their posts.
    """
    model = TAIIndex(
        time_slot_size=None,
        alpha_smoothing=None,
        credibility_threshold=credibility_threshold,
    )
    model.fit(train_data.itertuples(index=False))
    ranking = model.get_ranking()
    add_unranked_users(train_data, ranking)
    return ranking


def time_aware_influential_ranker(
    train_data: pd.DataFrame,
    time_slot_size: float = 18.0,
    alpha_smoothing: float = 0.6,
    credibility_threshold: float = 39.0,
    **kwargs,
) -> List[Tuple]:
    """
    Time-Aware Influential index ranker.

    Like influential_ranker but with time slot aggregation and EMA smoothing.

    Args:
        train_data: DataFrame with reshare data
        time_slot_size: Size of each time slot in days (default: 18.0)
        alpha_smoothing: EMA smoothing factor (default: 0.6)
        credibility_threshold: Only count low-credibility reshares

    Note:
        time_delta must be in days for default parameters to work correctly.
    """
    model = TAIIndex(
        time_slot_size=time_slot_size,
        alpha_smoothing=alpha_smoothing,
        credibility_threshold=credibility_threshold,
    )
    model.fit(train_data.itertuples(index=False))
    ranking = model.get_ranking()
    add_unranked_users(train_data, ranking)
    return ranking


def tash_index_ranker(
    train_data: pd.DataFrame,
    time_slot_size: float = 14.0,
    alpha_smoothing: float = 0.5,
    credibility_threshold: float = 39.0,
    **kwargs,
) -> List[Tuple]:
    """
    Time-Aware Social H-index (TASH) ranker.

    The primary superspreader detection method. Computes h-index per time slot
    and smooths with exponential moving average.

    Args:
        train_data: DataFrame with reshare data
        time_slot_size: Size of each time slot in days (default: 14.0, optimized)
        alpha_smoothing: EMA smoothing factor (default: 0.5, optimized)
        credibility_threshold: Only count low-credibility reshares

    Note:
        Default parameters (14 days, alpha=0.5) were obtained via optimization.
        time_delta must be in days for these defaults to work correctly.
    """
    model = TASHIndex(
        time_slot_size=time_slot_size,
        alpha_smoothing=alpha_smoothing,
        credibility_threshold=credibility_threshold,
    )
    model.fit(train_data.itertuples(index=False))
    ranking = model.get_ranking()
    add_unranked_users(train_data, ranking)
    return ranking


def mean_post_credibility_ranker(
    train_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    author_col: str = "target_author_id",
    resharer_col: str = "author_id",
    post_col: str = "target_action_id",
    credibility_col: str = "credibility_score",
    **kwargs,
) -> List[Tuple]:
    """
    Rank users by mean credibility of their posts.

    Lower credibility = higher rank (more misinformation).
    """
    filtered = train_data[train_data[credibility_col] <= credibility_threshold].copy()

    # Mean credibility of posts authored
    post_means = (
        filtered[[author_col, post_col, credibility_col]]
        .drop_duplicates()
        .groupby(author_col)[credibility_col]
        .mean()
        .to_dict()
    )

    # Mean credibility of posts reshared (for users not in post_means)
    reshare_means = (
        filtered[[resharer_col, post_col, credibility_col]]
        .drop_duplicates()
        .groupby(resharer_col)[credibility_col]
        .mean()
        .to_dict()
    )

    # Merge, prioritizing post authorship
    for user_id, score in reshare_means.items():
        if user_id not in post_means:
            post_means[user_id] = score

    # Sort ascending (lower credibility = higher rank)
    return sorted(post_means.items(), key=lambda x: x[1])
