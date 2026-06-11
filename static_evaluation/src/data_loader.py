"""
Data loading utilities for reshare network analysis.

Handles loading and preprocessing of CSV data files, including timestamp
parsing/normalization and credibility score extraction from the 'extra' field.
"""

import ast
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def parse_credibility_from_extra(
    extra_value,
    index: int = 0,
    default_value: float = 100.0,
) -> float:
    """
    Extract credibility score from the 'extra' field.

    The 'extra' field typically contains a list like "[64.5]" as a string.

    Args:
        extra_value: Value from the 'extra' column
        index: Index of the value to extract from the list (default: 0)
        default_value: Value to return if parsing fails (default: 100.0 = high credibility)

    Returns:
        Credibility score as float
    """
    if pd.isna(extra_value):
        return default_value

    if isinstance(extra_value, (int, float)):
        return float(extra_value)

    if isinstance(extra_value, list):
        if len(extra_value) > index:
            return float(extra_value[index])
        return default_value

    if isinstance(extra_value, str):
        try:
            parsed = ast.literal_eval(extra_value)
            if isinstance(parsed, list) and len(parsed) > index:
                return float(parsed[index])
            elif isinstance(parsed, (int, float)):
                return float(parsed)
            return default_value
        except (ValueError, SyntaxError):
            return default_value

    return default_value


def load_reshare_data(
    filepath: Path,
    filter_reshares_only: bool = True,
    normalize_time: bool = True,
    extra_index: int = 0,
    default_credibility: float = 100.0,
    time_unit: str = "days",
) -> pd.DataFrame:
    """
    Load and preprocess reshare data from a CSV file.

    Standardizes column names, extracts credibility scores, and optionally
    filters to reshare actions only.

    Args:
        filepath: Path to CSV file
        filter_reshares_only: If True, keep only 'reshare' action types
        normalize_time: If True, convert timestamps to time delta from start
        extra_index: Index of the credibility score in the 'extra' list (default: 0)
        default_credibility: Default credibility for missing values (default: 100.0)
        time_unit: Unit for time_delta - "days" (default) or "seconds"

    Returns:
        Preprocessed DataFrame with standard column names
    """
    # ID columns that should always be read as strings
    id_columns = {
        "action_id": str,
        "author_id": str,
        "target_action_id": str,
        "target_author_id": str,
    }

    # Load CSV with proper dtypes for ID columns
    df = pd.read_csv(filepath, dtype=id_columns)

    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Verify required columns exist
    required = ["action_id", "timestamp", "author_id", "action_type", "extra"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract credibility score from 'extra' field
    df["credibility_score"] = df["extra"].apply(
        lambda x: parse_credibility_from_extra(x, index=extra_index, default_value=default_credibility)
    )

    # Filter to reshares only
    if filter_reshares_only:
        df = df[df["action_type"] == "reshare"].copy()

    if len(df) == 0:
        raise ValueError(f"No reshare actions found in {filepath}")

    # Parse timestamp to datetime (handle mixed formats and timezones)
    # Use ISO8601 format to handle both with and without microseconds
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    # Compute time_delta from start in specified units
    if normalize_time:
        start_time = df["timestamp"].min()
        delta_seconds = (df["timestamp"] - start_time).dt.total_seconds()

        if time_unit == "days":
            df["time_delta"] = delta_seconds / 86400.0  # Convert to days
        else:
            df["time_delta"] = delta_seconds  # Keep as seconds
    else:
        df["time_delta"] = 0.0

    # Ensure target columns have no NaN for reshares
    df = df.dropna(subset=["target_action_id", "target_author_id"])

    # Reset index
    df = df.reset_index(drop=True)

    # Reorder columns for ranker compatibility
    # Rankers using itertuples expect: (index, time_delta, author_id, target_action_id, target_author_id, credibility_score, ...)
    ranker_columns = [
        "time_delta",
        "author_id",
        "target_action_id",
        "target_author_id",
        "credibility_score",
    ]
    other_columns = [c for c in df.columns if c not in ranker_columns]
    df = df[ranker_columns + other_columns]

    return df


def discover_simulation_folders(synthetic_dir: Path) -> List[Path]:
    """
    Find all simulation folders in the synthetic data directory.

    Simulation folders are expected to end with '_simulation'.

    Args:
        synthetic_dir: Path to synthetic data directory

    Returns:
        List of paths to simulation folders
    """
    if not synthetic_dir.exists():
        return []

    simulations = [
        p for p in synthetic_dir.iterdir()
        if p.is_dir() and p.name.endswith("_simulation")
    ]

    return sorted(simulations)


def discover_run_files(simulation_dir: Path) -> List[Path]:
    """
    Find all run files in a simulation folder.

    Run files are expected to match pattern 'run_*.csv'.

    Args:
        simulation_dir: Path to simulation folder

    Returns:
        List of paths to run CSV files
    """
    if not simulation_dir.exists():
        return []

    runs = list(simulation_dir.glob("run_*.csv"))

    # Sort by run number
    def get_run_number(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return sorted(runs, key=get_run_number)


def discover_real_datasets(real_dir: Path) -> List[Path]:
    """
    Find all real dataset files.

    Args:
        real_dir: Path to real data directory

    Returns:
        List of paths to real dataset CSV files
    """
    if not real_dir.exists():
        return []

    return sorted(real_dir.glob("*.csv"))


def load_all_runs(
    simulation_dir: Path,
    verbose: bool = True,
    extra_index: int = 0,
    default_credibility: float = 100.0,
    time_unit: str = "days",
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all run files from a simulation folder.

    Args:
        simulation_dir: Path to simulation folder
        verbose: Whether to print progress
        extra_index: Index of the credibility score in the 'extra' list
        default_credibility: Default credibility for missing values
        time_unit: Unit for time_delta - "days" (default) or "seconds"

    Returns:
        List of (run_name, dataframe) tuples
    """
    run_files = discover_run_files(simulation_dir)

    if not run_files:
        raise ValueError(f"No run files found in {simulation_dir}")

    runs = []
    for run_path in run_files:
        if verbose:
            print(f"  Loading {run_path.name}...")

        df = load_reshare_data(
            run_path,
            extra_index=extra_index,
            default_credibility=default_credibility,
            time_unit=time_unit,
        )
        runs.append((run_path.stem, df))

    return runs
