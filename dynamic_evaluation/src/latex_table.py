"""
latex_table.py
--------------
Small helper to write a LaTeX table string to disk.

The dynamic-evaluation tables themselves are built in :mod:`comparison`
(``build_comparison_table``) and :mod:`network_analysis`
(``build_network_metrics_table``); this module only persists them.
"""

from __future__ import annotations

from pathlib import Path


def save_latex(table_str: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table_str, encoding="utf-8")
