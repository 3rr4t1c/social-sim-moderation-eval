"""
static_on_synthetic.py
----------------------
Apply retroactive (static) moderation to synthetic no_moderation runs.

This simulates what a researcher would do with real data: observe the
pre-period, rank users, then remove top-K users' actions from the
post-period.  Crucially, the ``source_uid`` information (the intermediary
who surfaced the content) is **not** used for removal, because this
information is unavailable in real datasets.

The resulting filtered DataFrame can be compared against:
  · the original no_moderation run (baseline)
  · the dynamic-moderation run (same ranking method, in-simulation ban)

to quantify the gap between static and dynamic evaluation.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reference date: same epoch as vaccinitaly for timestamp conversion
# ---------------------------------------------------------------------------
_EPOCH = pd.Timestamp("2020-12-28T00:00:00+00:00")


def _clock_to_timestamp(clock_time: pd.Series) -> pd.Series:
    """Convert clock_time (float days) to UTC datetime."""
    return _EPOCH + pd.to_timedelta(clock_time, unit="D")


# ---------------------------------------------------------------------------
# Convert synthetic activities to the format expected by static rankers
# ---------------------------------------------------------------------------

def _to_ranker_input(df: pd.DataFrame, quality_threshold: float) -> pd.DataFrame:
    """
    Convert synthetic reshares to the column layout expected by
    static-evaluation rankers.

    Positional columns (accessed via ``itertuples``):
        time_delta | author_id | target_action_id | target_author_id
        | credibility_score

    Extra named columns:
        action_id  — needed by cosine_eigenvector / cosine_max
        timestamp  — needed by random_forest
    """
    reshares = df[df["content_id"].str.startswith("R")].copy()
    if reshares.empty:
        return reshares

    # Map synthetic columns → ranker columns
    reshares["time_delta"] = reshares["clock_time"]
    reshares["author_id"] = reshares["user_id"]
    reshares["target_action_id"] = reshares["original_id"]
    reshares["target_author_id"] = reshares["original_uid"]
    # quality is 0-1, credibility_score expected 0-100
    reshares["credibility_score"] = reshares["quality"] * 100.0
    reshares["action_id"] = reshares["content_id"]
    reshares["timestamp"] = _clock_to_timestamp(reshares["clock_time"])

    return (
        reshares[[
            "time_delta",
            "author_id",
            "target_action_id",
            "target_author_id",
            "credibility_score",
            "action_id",
            "timestamp",
        ]]
        .dropna(subset=["target_action_id", "target_author_id"])
        .reset_index(drop=True)
    )


def _call_ranker(ranker_fn, ranker_input: pd.DataFrame,
                 credibility_threshold: float) -> list[tuple]:
    """Call a ranker, passing credibility_threshold only if it accepts it."""
    sig = inspect.signature(ranker_fn)
    kwargs = (
        {"credibility_threshold": credibility_threshold}
        if "credibility_threshold" in sig.parameters
        else {}
    )
    return ranker_fn(ranker_input, **kwargs)


# ---------------------------------------------------------------------------
# Main: apply static moderation to a single no_moderation run
# ---------------------------------------------------------------------------

def apply_static_moderation(
    activities_df: pd.DataFrame,
    ranker_name: str,
    t_mod: float,
    top_k: int,
    credibility_threshold: float = 39.0,
    static_eval_root: Path | None = None,
) -> pd.DataFrame:
    """
    Apply retroactive (static) moderation to a no_moderation activities DF.

    Steps:
      1. Split at ``t_mod`` into pre-period and post-period.
      2. Convert pre-period reshares to ranker input format.
      3. Run the specified ranker to get user ranking.
      4. Identify top-K users.
      5. Remove from post-period:
         a. Actions where ``user_id`` is in the top-K set.
         b. Reshares where ``original_uid`` (original author) is in the top-K set.
         NOTE: ``source_uid`` (intermediary) is NOT used — this information
         is unavailable in real data.
      6. Return pre-period (unchanged) + filtered post-period.

    Parameters
    ----------
    activities_df:
        Full activities DataFrame from a no_moderation run.
    ranker_name:
        Name of the ranking method (must be in RANKER_REGISTRY).
    t_mod:
        Moderation trigger time (in clock_time units = days).
    top_k:
        Number of users to ban.
    credibility_threshold:
        Threshold on 0–100 scale passed to rankers.
    static_eval_root:
        Path to ``static_evaluation/``.  Inferred if None.

    Returns
    -------
    Filtered activities DataFrame (pre unchanged + post filtered).
    """
    # Import ranker registry
    if static_eval_root is None:
        static_eval_root = (
            Path(__file__).parent.parent.parent / "static_evaluation"
        )
    static_src = static_eval_root / "src"
    if str(static_src) not in sys.path:
        sys.path.insert(0, str(static_src))
    from ranking import RANKER_REGISTRY  # type: ignore

    if ranker_name not in RANKER_REGISTRY:
        raise ValueError(f"Unknown ranker: {ranker_name}")

    # Split
    pre_df = activities_df[activities_df["clock_time"] < t_mod].copy()
    post_df = activities_df[activities_df["clock_time"] >= t_mod].copy()

    # Prepare ranker input from pre-period reshares
    ranker_input = _to_ranker_input(pre_df, credibility_threshold)
    if ranker_input.empty:
        # No reshares in pre-period → can't rank → return unmodified
        return activities_df

    # Rank
    ranking = _call_ranker(
        RANKER_REGISTRY[ranker_name]["func"],
        ranker_input,
        credibility_threshold,
    )
    top_users = {str(uid) for uid, _ in ranking[:top_k]}

    # Remove from post-period:
    #   - any action by a top-K user
    #   - any reshare OF a top-K user's original content
    # NOTE: source_uid (intermediary) is intentionally NOT used
    is_reshare = post_df["content_id"].str.startswith("R")
    author_hit = post_df["user_id"].isin(top_users)
    original_hit = is_reshare & post_df["original_uid"].isin(top_users)
    post_filtered = post_df[~(author_hit | original_hit)]

    n_removed = len(post_df) - len(post_filtered)
    n_post = len(post_df)
    pct = n_removed / max(n_post, 1) * 100
    print(f"    [{ranker_name}] removed {n_removed:,}/{n_post:,} "
          f"post actions ({pct:.1f}%)")

    return pd.concat([pre_df, post_filtered], ignore_index=True)
