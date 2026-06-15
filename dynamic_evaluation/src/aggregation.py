"""
aggregation.py
--------------
Helpers to aggregate per-run temporal metrics across multiple simulation runs.

This module provides two utilities used by :mod:`comparison`:

``describe_array``
    Mean / std / median / min / max of a 1-D array (NaN-aware).

``_agg_timeseries``
    Aggregate one per-bin column across runs onto a common time grid,
    returning the per-bin median plus a 5th/95th percentile band
    (inter-run dispersion, matching the static pipeline).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Per-bin time-series aggregation (median + percentile band across runs)
# ---------------------------------------------------------------------------

def _agg_timeseries(
    runs_bins: list[pd.DataFrame],
    column: str,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> pd.DataFrame:
    """
    Aggregate one column across runs onto a common time grid.

    Returns DataFrame: time_mid, center, lo, hi, n_valid, where
    ``center`` is the per-bin median across runs and ``lo``/``hi`` are the
    ``lo_pct``/``hi_pct`` percentiles across runs.

    The band describes the **dispersion across runs** (inter-run
    variability), matching the static pipeline's convention
    (``aggregate_synthetic_traces``: median + 5th/95th percentile). It is
    NOT a confidence interval of the mean: with each run being a distinct
    network realisation, inter-run spread is the quantity of interest.
    """
    all_mids = np.unique(
        np.concatenate([df["time_mid"].to_numpy() for df in runs_bins])
    )

    stacked = np.full((len(runs_bins), len(all_mids)), np.nan)
    for i, df in enumerate(runs_bins):
        reindexed = df.set_index("time_mid")[column].reindex(all_mids)
        stacked[i] = reindexed.to_numpy()

    n_valid = (~np.isnan(stacked)).sum(axis=0)

    # Bins where every run is NaN yield NaN (and a warning we silence).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(stacked, axis=0)
        lo = np.nanpercentile(stacked, lo_pct, axis=0)
        hi = np.nanpercentile(stacked, hi_pct, axis=0)

    return pd.DataFrame({
        "time_mid": all_mids,
        "center":   center,
        "lo":       lo,
        "hi":       hi,
        "n_valid":  n_valid,
    })
