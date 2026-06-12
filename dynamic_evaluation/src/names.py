"""
names.py
--------
Centralised display-name mapping for ranking methods.

Edit ``METHOD_DISPLAY_NAMES`` to change how method names appear in plots and
LaTeX tables throughout the dynamic evaluation pipeline.

Keys   : internal method names as they appear in condition folder names and
         ranker registry (e.g. ``"random_forest"``, ``"tash_index"``).
Values : human-readable labels used in figures and tables.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Edit this dict to change display names everywhere (plots + tables).
# ---------------------------------------------------------------------------

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "tash_index": "TASH-Index",
    "social_h_index": "Social H-Index",
    "influential": "Influential",
    "time_aware_influential": "TAI",
    "mean_post_credibility": "Mean Post Cred.",
    "repost_count": "Repost Count",
    "early_reposter": "Early Reposter",
    "tar_index": "TAR-Index",
    "node_degree": "Node Degree",
    "node_strength": "Node Strength",
    "self_repost": "Self-Repost",
    "mean_repost_credibility": "Mean Repost Cred.",
    "cosine_eigenvector": "Coordination Centrality",
    # Max cosine similarity with any other user; called "Coordination Edge
    # Weight" in the paper. Used as a Random Forest feature rather than as a
    # standalone method in the final comparison grid.
    "cosine_max": "Coordination Edge Weight",
    "random_forest": "Random Forest",
}

# Short abbreviations for LaTeX table rows (one cell per condition)
METHOD_ABBREV: dict[str, str] = {
    "tash_index": "TASH",
    "social_h_index": "SHI",
    "influential": "INF",
    "time_aware_influential": "TAI",
    "mean_post_credibility": "MPC",
    "repost_count": "RC",
    "early_reposter": "ER",
    "tar_index": "TAR",
    "node_degree": "ND",
    "node_strength": "NS",
    "self_repost": "SR",
    "mean_repost_credibility": "MRC",
    "cosine_eigenvector": "CE",
    "cosine_max": "CEW",
    "random_forest": "RF",
}


def method_display_name(method: str) -> str:
    """Return the display name for a method, falling back to a title-cased version."""
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())


def method_abbrev(method: str) -> str:
    """Return the short abbreviation for a method (used in LaTeX table rows)."""
    return METHOD_ABBREV.get(method, method.replace("_", "-").upper())
