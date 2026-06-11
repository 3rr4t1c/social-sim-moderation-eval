"""
real_data.py
------------
Process a real dataset to produce moderation-filtered variants that mirror
the structure of the synthetic simulator output.

Pipeline (per ranker)
---------------------
1. Load the full real dataset; normalise time to fractional days from t=0.
2. Split at ``t_mod``: pre-period (< t_mod) and post-period (≥ t_mod).
3. Rank users with the chosen static-evaluation ranker on pre-period reshares.
4. Identify the top-k users.
5. Remove from the post-period:
   a. Any action whose ``author_id`` is in the top-k set.
   b. Any reshare whose ``target_author_id`` is in the top-k set.
6. Save pre-period (unchanged) + filtered post-period as
   ``<dataset>_<ranker>_day<t_mod>_top<k>_ban.csv``.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _parse_cred(value) -> float:
    """Extract credibility score from the ``extra`` column (0–100 scale)."""
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list) and parsed:
            return float(parsed[0])
    except Exception:
        pass
    return 100.0   # default: high credibility (not low-quality)


def _load_full(path: Path) -> pd.DataFrame:
    """
    Load the full real CSV with normalised time and credibility columns added.

    Extra columns added:
      ``time_delta_days``   — fractional days from the first action
      ``credibility_score`` — quality on 0–100 scale (from ``extra``)
    """
    df = pd.read_csv(
        path,
        index_col=0,
        dtype={
            "action_id":        str,
            "author_id":        str,
            "target_action_id": str,
            "target_author_id": str,
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    t0 = df["timestamp"].min()
    df["time_delta_days"] = (df["timestamp"] - t0).dt.total_seconds() / 86400.0
    df["credibility_score"] = df["extra"].apply(_parse_cred)
    return df


def _to_ranker_input(reshares_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename columns for static-evaluation rankers.

    Column order must match ``static_evaluation``'s ``load_reshare_data``
    output, because many rankers access columns **positionally** via
    ``itertuples``:

        time_delta | author_id | target_action_id | target_author_id
        | credibility_score

    Extra columns appended **after** the positional ones (accessed by name):
        action_id  — required by cosine_eigenvector / cosine_max
        timestamp  — required by random_forest's extract_temporal_features
    """
    return (
        reshares_df[[
            "time_delta_days",
            "author_id",
            "target_action_id",
            "target_author_id",
            "credibility_score",
            "action_id",
            "timestamp",
        ]]
        .rename(columns={"time_delta_days": "time_delta"})
        .dropna(subset=["target_action_id", "target_author_id"])
        .reset_index(drop=True)
    )


def _call_ranker(ranker_fn, ranker_input: pd.DataFrame,
                 credibility_threshold: float) -> list[tuple]:
    """
    Call a ranker function, passing ``credibility_threshold`` only when the
    function signature accepts it.
    """
    sig = inspect.signature(ranker_fn)
    kwargs = (
        {"credibility_threshold": credibility_threshold}
        if "credibility_threshold" in sig.parameters
        else {}
    )
    return ranker_fn(ranker_input, **kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_real_conditions(
    dataset_path: Path,
    output_dir: Path,
    rankers: Sequence[str],
    t_mod: int,
    top_k: int,
    credibility_threshold: float = 39.0,
    static_eval_root: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Generate moderation-filtered CSV files for every specified ranker.

    Parameters
    ----------
    dataset_path:
        Raw real CSV (e.g. ``input/real_data/vaccinitaly_dataset.csv``).
    output_dir:
        Destination for generated CSVs.
    rankers:
        Ranker names from the static-evaluation registry.
    t_mod:
        Moderation day (fractional days from dataset start).
    top_k:
        Number of users to ban.
    credibility_threshold:
        Threshold on 0–100 scale passed to rankers that accept it.
    static_eval_root:
        Path to ``static_evaluation/``. Inferred if None.
    overwrite:
        Re-generate files that already exist.

    Returns
    -------
    Paths of all successfully generated (or already present) output files.
    """
    # Import the static-evaluation ranking registry
    if static_eval_root is None:
        static_eval_root = Path(__file__).parent.parent.parent / "static_evaluation"
    static_src = static_eval_root / "src"
    if str(static_src) not in sys.path:
        sys.path.insert(0, str(static_src))
    try:
        from ranking import RANKER_REGISTRY  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Cannot import static_evaluation rankers from {static_src}"
        ) from exc

    # Load and split the dataset
    dataset_stem = dataset_path.stem.replace("_dataset", "")
    print(f"  Loading: {dataset_path.name}")
    full_df = _load_full(dataset_path)
    print(f"    {len(full_df):,} actions  |  "
          f"{full_df['time_delta_days'].min():.1f}–"
          f"{full_df['time_delta_days'].max():.1f} days")

    pre_df  = full_df[full_df["time_delta_days"] <  t_mod].copy()
    post_df = full_df[full_df["time_delta_days"] >= t_mod].copy()
    print(f"    Pre  (< day {t_mod}): {len(pre_df):,}  |  "
          f"Post (≥ day {t_mod}): {len(post_df):,}")

    pre_reshares = pre_df[pre_df["action_type"] == "reshare"]
    if pre_reshares.empty:
        print("  WARNING: no reshares in pre-period — ranking will be empty.")
    ranker_input = _to_ranker_input(pre_reshares)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for ranker_name in rankers:
        if ranker_name not in RANKER_REGISTRY:
            print(f"  [{ranker_name}] not in registry — skipping.")
            continue

        out_file = (
            output_dir
            / f"{dataset_stem}_{ranker_name}_day{t_mod}_top{top_k}_ban.csv"
        )

        if out_file.exists() and not overwrite:
            print(f"  [{ranker_name}] {out_file.name} already exists — skipping.")
            generated.append(out_file)
            continue

        print(f"  [{ranker_name}] ranking…", end=" ", flush=True)
        try:
            ranking = _call_ranker(
                RANKER_REGISTRY[ranker_name]["func"],
                ranker_input,
                credibility_threshold,
            )
        except Exception as exc:
            print(f"FAILED ({exc}) — skipping.")
            continue

        top_users: set[str] = {str(uid) for uid, _ in ranking[:top_k]}
        print(f"top-{top_k} identified.")

        author_removed = post_df["author_id"].isin(top_users)
        target_removed = (
            (post_df["action_type"] == "reshare")
            & post_df["target_author_id"].isin(top_users)
        )
        post_filtered = post_df[~(author_removed | target_removed)]

        n_rm = len(post_df) - len(post_filtered)
        print(f"    Removed {n_rm:,} post-period actions "
              f"({n_rm / max(len(post_df), 1) * 100:.1f}%)")

        combined = pd.concat([pre_df, post_filtered], ignore_index=True)
        combined = combined.drop(
            columns=["time_delta_days", "credibility_score"], errors="ignore"
        )
        combined.to_csv(out_file)
        print(f"    → {out_file.name}")
        generated.append(out_file)

    return generated
