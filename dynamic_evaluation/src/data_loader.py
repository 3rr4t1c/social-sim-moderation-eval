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

Real data folder convention
----------------------------
  <real_root>/
    <dataset>_dataset.csv               (raw, unfiltered)
    <dataset>_<method>_day<t>_top<k>_ban.csv   (generated filtered version)

Each real CSV has columns:
  (index), action_id, timestamp, author_id, action_type,
  target_action_id, target_author_id, extra

where ``extra`` encodes quality as a Python list literal, e.g. ``[64.5]``.
Time is converted to fractional days from the first action in the file.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from names import method_display_name, method_abbrev


# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """Metadata parsed from a condition folder name."""
    name: str                       # raw folder name
    is_baseline: bool               # True for 'no_moderation'
    method: str | None = None       # ranking method, e.g. "tash_index"
    t_mod: int | None = None        # simulation day when moderation fires
    top_k: int | None = None        # number of users moderated
    mod_type: str | None = None     # "ban", "shadowban", etc.

    def label(self) -> str:
        """
        Human-readable label used everywhere (plots, tables).
        Identifies the ranking method only; k / t_mod / mod_type belong in captions.
        """
        if self.is_baseline:
            return "No moderation"
        return method_display_name(self.method)

    def short_label(self) -> str:
        """Abbreviated label (e.g. for filenames / internal keys)."""
        if self.is_baseline:
            return "no-mod"
        return method_abbrev(self.method)


_CONDITION_RE = re.compile(
    r"^(?P<method>.+?)_day(?P<t_mod>\d+)_top(?P<top_k>\d+)_(?P<mod_type>.+)$"
)


def parse_condition(folder_name: str) -> Condition:
    """
    Parse a condition folder name into a :class:`Condition`.

    Accepted formats
    ----------------
    ``no_moderation``
        Baseline — no moderation applied.
    ``<method>_day<t_mod>_top<k>_<mod_type>``
        Moderated condition, e.g. ``tash_index_day150_top10_ban``.
    """
    if folder_name == "no_moderation":
        return Condition(name=folder_name, is_baseline=True)

    m = _CONDITION_RE.match(folder_name)
    if not m:
        raise ValueError(
            f"Cannot parse condition folder name: '{folder_name}'. "
            "Expected 'no_moderation' or '<method>_day<N>_top<K>_<mod_type>'."
        )
    return Condition(
        name=folder_name,
        is_baseline=False,
        method=m.group("method"),
        t_mod=int(m.group("t_mod")),
        top_k=int(m.group("top_k")),
        mod_type=m.group("mod_type"),
    )


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


# ---------------------------------------------------------------------------
# Real data CSV loading
# ---------------------------------------------------------------------------

def _parse_quality(extra_str: str) -> float | None:
    """Extract quality from the ``extra`` column (encoded as ``[value]``)."""
    try:
        lst = ast.literal_eval(str(extra_str))
        if isinstance(lst, list) and lst:
            return float(lst[0]) / 100.0   # convert 0-100 → 0-1
    except Exception:
        pass
    return None


def load_real_data(path: Path, normalize_time: bool = True) -> pd.DataFrame:
    """
    Load a real-dataset CSV and return a normalised DataFrame compatible
    with the synthetic pipeline.

    The returned DataFrame has the same key columns as :func:`load_synt_activities`:
    - ``user_id``       (str)
    - ``action_type``   ("post" | "reshare", category)
    - ``quality``       (float32, 0–1 scale)
    - ``clock_time``    (float64, fractional days from first action if normalize_time)

    Parameters
    ----------
    path:
        Path to a real CSV file.
    normalize_time:
        If True (default), convert timestamps to fractional days elapsed
        since the earliest action in the file. This makes ``t_mod`` (given
        in days offset) directly comparable.
    """
    df = pd.read_csv(path, index_col=0)

    # Rename columns to match synthetic schema
    df = df.rename(columns={
        "author_id": "user_id",
    })
    df["user_id"] = df["user_id"].astype("string")

    # Ensure action_type is category
    df["action_type"] = df["action_type"].astype("category")

    # Parse quality from `extra` column (e.g. "[64.5]")
    df["quality"] = df["extra"].apply(_parse_quality).astype("float32")

    # Parse timestamps and normalise to fractional days
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if normalize_time:
        t0 = df["timestamp"].min()
        df["clock_time"] = (df["timestamp"] - t0).dt.total_seconds() / 86400.0
    else:
        df["clock_time"] = (
            df["timestamp"].astype("int64") / 1e9 / 86400.0
        )

    df["clock_time"] = df["clock_time"].astype("float64")

    return df


# ---------------------------------------------------------------------------
# Network / condition / run traversal
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One simulation run within a condition."""
    index: int           # 1-based run index (sorted by run-id folder name)
    source_folder: Path  # run-id folder
    condition: Condition
    network: str


@dataclass
class ConditionData:
    """All runs for a single condition."""
    condition: Condition
    network: str
    runs: list[Run] = field(default_factory=list)

    def iter_dataframes(self) -> Iterator[pd.DataFrame]:
        """Yield a loaded DataFrame for each run."""
        for run in self.runs:
            yield load_synt_activities(run.source_folder / "activities.csv")


def discover_network(
    network_path: Path,
    include_baseline: bool = False,
) -> list[ConditionData]:
    """
    Discover all conditions and their runs under a network folder.

    Parameters
    ----------
    network_path:
        Path to a network folder, e.g. ``…/input/synt_data/network_50k``.
    include_baseline:
        If False (default), skip ``no_moderation`` folders.  The pre-period
        of each moderated condition serves as the baseline.

    Returns
    -------
    List of :class:`ConditionData`, one per recognised condition subfolder.
    """
    if not network_path.is_dir():
        raise FileNotFoundError(f"Network folder not found: {network_path}")

    network_name = network_path.name
    results: list[ConditionData] = []

    for cond_folder in sorted(network_path.iterdir()):
        if not cond_folder.is_dir() or cond_folder.name.startswith("."):
            continue

        try:
            condition = parse_condition(cond_folder.name)
        except ValueError:
            continue   # skip unrecognised folders silently

        if condition.is_baseline and not include_baseline:
            continue   # pre-period is the baseline; skip no_moderation

        # Collect run subfolders that contain activities.csv
        run_folders = sorted(
            f for f in cond_folder.iterdir()
            if f.is_dir() and (f / "activities.csv").exists()
        )

        if not run_folders:
            continue

        cd = ConditionData(condition=condition, network=network_name)
        for idx, folder in enumerate(run_folders, start=1):
            cd.runs.append(Run(
                index=idx,
                source_folder=folder,
                condition=condition,
                network=network_name,
            ))

        results.append(cd)

    return results


def discover_all_networks(synt_root: Path) -> dict[str, list[ConditionData]]:
    """
    Discover all networks under *synt_root* (``input/synt_data/`` by default).

    Returns
    -------
    ``{network_name: [ConditionData, …]}``
    """
    if not synt_root.is_dir():
        raise FileNotFoundError(f"Synthetic data root not found: {synt_root}")

    return {
        folder.name: discover_network(folder)
        for folder in sorted(synt_root.iterdir())
        if folder.is_dir() and not folder.name.startswith(".")
    }
