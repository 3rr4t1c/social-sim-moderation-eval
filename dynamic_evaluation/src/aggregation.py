"""
aggregation.py
--------------
Aggregate per-run temporal metrics across multiple simulation runs.

Primary metric: **lq_fraction** (fraction of low-quality actions per bin).
avg_quality is still computed internally but is not shown in the default
outputs (it is nearly redundant when quality has a bimodal distribution).

Additional table metrics:
  · Cohen's d — standardised effect size (pre vs post lq_fraction)
  · Activity drop % — percent change in total action count post-moderation

The main entry point is :func:`compute_all_conditions`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Descriptive statistics helpers
# ---------------------------------------------------------------------------

def describe_array(arr: np.ndarray) -> dict[str, float]:
    """
    Compute mean, std, median, min, max of a 1-D array, ignoring NaNs.
    """
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "median": nan, "min": nan, "max": nan}
    return {
        "mean":   float(np.mean(clean)),
        "std":    float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        "median": float(np.median(clean)),
        "min":    float(np.min(clean)),
        "max":    float(np.max(clean)),
    }


def cohens_d(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Cohen's d for a "lower is better" metric.

    Computed as (mean_pre − mean_post) / pooled_std, so that a positive
    value indicates improvement (post is lower than pre).
    """
    pre_c  = pre[~np.isnan(pre)]
    post_c = post[~np.isnan(post)]
    if pre_c.size < 2 or post_c.size < 2:
        return float("nan")
    pooled = np.sqrt((np.var(pre_c, ddof=1) + np.var(post_c, ddof=1)) / 2.0)
    if pooled == 0.0:
        return float("nan")
    return float((np.mean(pre_c) - np.mean(post_c)) / pooled)


# ---------------------------------------------------------------------------
# Per-bin time-series aggregation (mean ± CI across runs)
# ---------------------------------------------------------------------------

def _agg_timeseries(
    runs_bins: list[pd.DataFrame],
    column: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate one column across runs onto a common time grid.
    Returns DataFrame: time_mid, mean, lo, hi, n_valid.
    """
    all_mids = np.unique(
        np.concatenate([df["time_mid"].to_numpy() for df in runs_bins])
    )

    stacked = np.full((len(runs_bins), len(all_mids)), np.nan)
    for i, df in enumerate(runs_bins):
        reindexed = df.set_index("time_mid")[column].reindex(all_mids)
        stacked[i] = reindexed.to_numpy()

    alpha = 1.0 - confidence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std  = np.nanstd(stacked,  axis=0, ddof=1)

    n_valid = (~np.isnan(stacked)).sum(axis=0)
    t_crit  = np.where(
        n_valid > 1,
        stats.t.ppf(1.0 - alpha / 2.0, df=np.maximum(n_valid - 1, 1)),
        np.nan,
    )
    ci = t_crit * std / np.sqrt(np.maximum(n_valid, 1))

    return pd.DataFrame({
        "time_mid": all_mids,
        "mean":     mean,
        "lo":       np.maximum(mean - ci, 0.0),
        "hi":       mean + ci,
        "n_valid":  n_valid,
    })


# ---------------------------------------------------------------------------
# Per-condition result container
# ---------------------------------------------------------------------------

@dataclass
class ConditionStats:
    """
    All aggregated statistics for a single condition.

    Fields used by the default outputs (table + plots) are lq_fraction-based.
    avg_quality fields are computed and stored for supplementary analysis but
    not shown by default.
    """
    condition: object               # Condition dataclass
    network: str

    # Pooled per-bin arrays --------------------------------------------------
    pre_lq_frac:  np.ndarray = field(default_factory=lambda: np.array([]))
    post_lq_frac: np.ndarray = field(default_factory=lambda: np.array([]))
    pre_avg_q:  np.ndarray = field(default_factory=lambda: np.array([]))
    post_avg_q: np.ndarray = field(default_factory=lambda: np.array([]))

    # Descriptive stats dicts ------------------------------------------------
    pre_stats_lq_frac:  dict = field(default_factory=dict)
    post_stats_lq_frac: dict = field(default_factory=dict)
    pre_stats_avg_q:   dict = field(default_factory=dict)
    post_stats_avg_q:  dict = field(default_factory=dict)

    # Time-series DataFrames (time_mid, mean, lo, hi, n_valid) ---------------
    timeseries_lq_frac:    pd.DataFrame = field(default_factory=pd.DataFrame)
    cumulative_lq_frac_ts: pd.DataFrame = field(default_factory=pd.DataFrame)
    timeseries_avg_q:      pd.DataFrame = field(default_factory=pd.DataFrame)

    # Raw action quality arrays (for Cohen's d) --------------------------------
    pre_raw_quality:  np.ndarray = field(default_factory=lambda: np.array([]))
    post_raw_quality: np.ndarray = field(default_factory=lambda: np.array([]))

    # Table-only metrics -----------------------------------------------------
    cohens_d_lq_frac: float = float("nan")
    pre_total_actions:  int = 0
    post_total_actions: int = 0
    activity_drop_pct: float = float("nan")

    # Metadata ---------------------------------------------------------------
    t_mod: int | None = None
    n_runs: int = 0


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate_condition(
    condition_data,
    bin_width: float = 1.0,
    quality_threshold: float = 0.39,
    confidence: float = 0.95,
) -> ConditionStats:
    """Aggregate all runs of one condition into a :class:`ConditionStats`."""
    from .metrics import compute_bins, split_pre_post

    cond  = condition_data.condition
    t_mod = cond.t_mod

    all_bins: list[pd.DataFrame] = []
    pre_avg_q_parts:    list[np.ndarray] = []
    post_avg_q_parts:   list[np.ndarray] = []
    pre_lq_frac_parts:  list[np.ndarray] = []
    post_lq_frac_parts: list[np.ndarray] = []
    pre_action_counts:  list[int] = []
    post_action_counts: list[int] = []
    pre_raw_quality_parts:  list[np.ndarray] = []
    post_raw_quality_parts: list[np.ndarray] = []

    for df in condition_data.iter_dataframes():
        bins = compute_bins(df, bin_width=bin_width,
                            quality_threshold=quality_threshold)
        all_bins.append(bins)

        if t_mod is not None:
            pre, post = split_pre_post(bins, t_mod)
            # Collect raw action quality values for Cohen's d
            pre_actions  = df[df["clock_time"] <  t_mod]
            post_actions = df[df["clock_time"] >= t_mod]
        else:
            pre, post = bins, bins.iloc[0:0]
            pre_actions  = df
            post_actions = df.iloc[0:0]

        pre_avg_q_parts.append(pre["avg_quality"].dropna().to_numpy())
        post_avg_q_parts.append(post["avg_quality"].dropna().to_numpy())
        pre_lq_frac_parts.append(pre["lq_fraction"].dropna().to_numpy())
        post_lq_frac_parts.append(post["lq_fraction"].dropna().to_numpy())
        pre_action_counts.append(int(pre["n_actions"].sum()))
        post_action_counts.append(int(post["n_actions"].sum()))
        pre_raw_quality_parts.append(
            pre_actions["quality"].dropna().to_numpy().astype("float64"))
        post_raw_quality_parts.append(
            post_actions["quality"].dropna().to_numpy().astype("float64"))

    n_runs = len(all_bins)

    pre_avg_q    = np.concatenate(pre_avg_q_parts)    if pre_avg_q_parts    else np.array([])
    post_avg_q   = np.concatenate(post_avg_q_parts)   if post_avg_q_parts   else np.array([])
    pre_lq_frac  = np.concatenate(pre_lq_frac_parts)  if pre_lq_frac_parts  else np.array([])
    post_lq_frac = np.concatenate(post_lq_frac_parts) if post_lq_frac_parts else np.array([])
    pre_raw_quality  = np.concatenate(pre_raw_quality_parts)  if pre_raw_quality_parts  else np.array([])
    post_raw_quality = np.concatenate(post_raw_quality_parts) if post_raw_quality_parts else np.array([])

    pre_total  = sum(pre_action_counts)
    post_total = sum(post_action_counts)
    act_drop   = ((post_total - pre_total) / pre_total * 100.0
                  if pre_total > 0 else float("nan"))

    # Time-series
    ts_lq_frac     = _agg_timeseries(all_bins, "lq_fraction",        confidence=confidence)
    ts_cum_lq_frac = _agg_timeseries(all_bins, "cumulative_lq_frac", confidence=confidence)
    ts_avg_q       = _agg_timeseries(all_bins, "avg_quality",        confidence=confidence)

    return ConditionStats(
        condition=cond,
        network=condition_data.network,
        pre_lq_frac=pre_lq_frac,
        post_lq_frac=post_lq_frac,
        pre_avg_q=pre_avg_q,
        post_avg_q=post_avg_q,
        pre_stats_lq_frac=describe_array(pre_lq_frac),
        post_stats_lq_frac=describe_array(post_lq_frac),
        pre_stats_avg_q=describe_array(pre_avg_q),
        post_stats_avg_q=describe_array(post_avg_q),
        timeseries_lq_frac=ts_lq_frac,
        cumulative_lq_frac_ts=ts_cum_lq_frac,
        timeseries_avg_q=ts_avg_q,
        pre_raw_quality=pre_raw_quality,
        post_raw_quality=post_raw_quality,
        cohens_d_lq_frac=cohens_d(pre_lq_frac, post_lq_frac),
        pre_total_actions=pre_total,
        post_total_actions=post_total,
        activity_drop_pct=act_drop,
        t_mod=t_mod,
        n_runs=n_runs,
    )


def compute_all_conditions(
    condition_data_list: list,
    bin_width: float = 1.0,
    quality_threshold: float = 0.39,
    confidence: float = 0.95,
) -> dict[str, "ConditionStats"]:
    """Aggregate all conditions in a network."""
    return {
        cd.condition.name: aggregate_condition(
            cd,
            bin_width=bin_width,
            quality_threshold=quality_threshold,
            confidence=confidence,
        )
        for cd in condition_data_list
        if cd.runs
    }
