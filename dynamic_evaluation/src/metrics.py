"""
metrics.py
----------
Temporal quality metrics computed from a single simulation run.

Two base metrics are computed per time bin:

avg_quality
    Mean quality of ALL actions (posts + reshares) in the bin, on the 0–1
    scale.  NaN for empty bins.

lq_fraction
    Fraction of actions whose quality is ≤ ``quality_threshold``.
    NaN for empty bins.

Additionally, the cumulative LQ action count is computed as a running total
across bins (useful for cumulative-LQ plots).

The split between pre- and post-moderation periods is handled by
:func:`split_pre_post`, which partitions a bins DataFrame at a given t_mod.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bins(
    df: pd.DataFrame,
    bin_width: float = 1.0,
    quality_threshold: float = 0.39,
) -> pd.DataFrame:
    """
    Compute per-bin temporal quality metrics for a single simulation run.

    Parameters
    ----------
    df:
        Normalised activities DataFrame as returned by
        :func:`data_loader.load_synt_activities`.
        Must contain columns ``clock_time`` and ``quality`` (0–1 scale).
    bin_width:
        Width of each time bin in days (default: 1 day).
    quality_threshold:
        Quality value (0–1 scale) at or below which an action is considered
        low-quality (default: 0.39).

    Returns
    -------
    DataFrame with one row per bin and columns:

    ``time_start``, ``time_end``, ``time_mid``
        Bin boundaries and midpoint (days).
    ``n_actions``
        Total actions in the bin.
    ``n_lq_actions``
        Low-quality actions (quality ≤ threshold).
    ``avg_quality``
        Mean quality of all actions in the bin (0–1). NaN if bin is empty.
    ``lq_fraction``
        ``n_lq_actions / n_actions``. NaN if bin is empty.
    ``cumulative_lq``
        Running total of ``n_lq_actions`` up to and including this bin.
    ``cumulative_avg_q``
        Running mean quality up to and including this bin
        (cumulative sum of quality / cumulative action count).
    ``cumulative_lq_frac``
        Running LQ fraction up to and including this bin
        (cumulative_lq / cumulative action count).
        These two running-proportion columns normalise for activity volume and
        make the accumulated effect of moderation easier to compare visually.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "time_start", "time_end", "time_mid",
            "n_actions", "n_lq_actions",
            "avg_quality", "lq_fraction",
            "cumulative_lq", "cumulative_avg_q", "cumulative_lq_frac",
        ])

    t_min = df["clock_time"].min()
    t_max = df["clock_time"].max()

    # Bin edges aligned to 0
    edges = np.arange(
        np.floor(t_min / bin_width) * bin_width,
        t_max + bin_width,
        bin_width,
    )
    n_bins = len(edges) - 1

    # Assign each action to a bin index
    bin_idx = np.searchsorted(edges, df["clock_time"].to_numpy(), side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    tmp = df[["quality"]].copy()
    tmp["bin"]   = bin_idx
    tmp["is_lq"] = (df["quality"].to_numpy() <= quality_threshold).astype("int32")

    grouped = tmp.groupby("bin", sort=True).agg(
        n_actions   =("quality", "count"),
        n_lq_actions=("is_lq",   "sum"),
        sum_quality =("quality", "sum"),
    )

    result = pd.DataFrame({
        "time_start": edges[:n_bins],
        "time_end":   edges[1:],
        "time_mid":   (edges[:n_bins] + edges[1:]) / 2,
    })
    result = result.join(grouped, how="left")

    result["n_actions"]    = result["n_actions"].fillna(0).astype("int64")
    result["n_lq_actions"] = result["n_lq_actions"].fillna(0).astype("int64")
    result["sum_quality"]  = result["sum_quality"].fillna(0.0)

    # avg_quality and lq_fraction: NaN for empty bins
    mask_nonempty = result["n_actions"] > 0
    result["avg_quality"] = np.where(
        mask_nonempty,
        result["sum_quality"] / result["n_actions"],
        np.nan,
    )
    result["lq_fraction"] = np.where(
        mask_nonempty,
        result["n_lq_actions"] / result["n_actions"],
        np.nan,
    )

    result["cumulative_lq"] = result["n_lq_actions"].cumsum()

    cum_n_actions   = result["n_actions"].cumsum()
    cum_sum_quality = result["sum_quality"].cumsum()
    safe_cum_n = cum_n_actions.replace(0, np.nan)
    result["cumulative_avg_q"]  = cum_sum_quality / safe_cum_n
    result["cumulative_lq_frac"] = result["cumulative_lq"] / safe_cum_n

    result = result.drop(columns=["sum_quality"])
    return result.reset_index(drop=True)


def split_pre_post(
    bins_df: pd.DataFrame,
    t_mod: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a bins DataFrame into pre- and post-moderation periods.

    Parameters
    ----------
    bins_df:
        Output of :func:`compute_bins`.
    t_mod:
        Moderation trigger day.  Bins with ``time_start < t_mod`` go to
        pre; bins with ``time_start >= t_mod`` go to post.

    Returns
    -------
    ``(pre_df, post_df)``  — both are views, not copies.
    """
    pre  = bins_df[bins_df["time_start"] <  t_mod].copy()
    post = bins_df[bins_df["time_start"] >= t_mod].copy()
    return pre, post
