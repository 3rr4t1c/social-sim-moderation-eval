"""
data_loader.py
--------------
Loading and normalisation of dynamic evaluation data.

Synthetic data folder convention
---------------------------------
  <synt_root>/
    <network_name>/
      no_moderation/            (optional baseline — treated as baseline condition)
        <run_id>/
          activities.csv
      <method>_day<t_mod>_top<k>_<mod_type>/
        <run_id>/
          activities.csv

Each ``activities.csv`` has columns:
  content_id, user_id, quality, appeal, source_id, source_uid,
  original_id, original_uid, clock_time
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic CSV loading
# ---------------------------------------------------------------------------

_SYNT_DTYPE = {
    "content_id":   "string",
    "user_id":      "string",
    "quality":      "float32",
    "appeal":       "float32",
    "source_id":    "string",
    "source_uid":   "string",
    "original_id":  "string",
    "original_uid": "string",
    "clock_time":   "float64",
}


def load_synt_activities(path: Path) -> pd.DataFrame:
    """
    Load a synthetic ``activities.csv`` and return a normalised DataFrame.

    Added columns
    -------------
    ``action_type``
        ``"post"`` (prefix ``P``) or ``"reshare"`` (prefix ``R``).
    """
    df = pd.read_csv(path, dtype=_SYNT_DTYPE)

    prefix = df["content_id"].str[0]
    df["action_type"] = prefix.map({"P": "post", "R": "reshare"}).astype("category")

    # Keep quality on 0-1 scale; threshold comparisons use the same scale.
    return df
