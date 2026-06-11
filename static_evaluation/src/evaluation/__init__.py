"""
Evaluation tools for misinformation network dismantling.

This module provides:
- Network dismantling algorithms
- Metrics for comparing rankings (NDCG)
- Visualization tools for dismantling curves
"""

from .dismantling import (
    compute_optimal_ranking,
    dismantle_network,
    compute_dismantling_trace,
    compute_ndcg_score,
)

from .plotting import (
    plot_dismantling_comparison,
    plot_real_vs_synthetic,
    aggregate_synthetic_traces,
    compute_effectiveness_at_top_k,
    plot_effectiveness_decay,
    plot_comparison_dismantling,
    plot_comparison_effectiveness,
)
