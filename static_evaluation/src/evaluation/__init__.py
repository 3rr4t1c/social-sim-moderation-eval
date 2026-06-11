"""
Evaluation tools for misinformation network dismantling.

This module provides:
- Network dismantling algorithms
- Visualization tools for dismantling curves
"""

from .dismantling import (
    compute_optimal_ranking,
    dismantle_network,
    trace_to_array,
)

from .plotting import (
    plot_dismantling_comparison,
    aggregate_synthetic_traces,
    compute_effectiveness_at_top_k,
    plot_effectiveness_decay,
    plot_comparison_dismantling,
    plot_comparison_effectiveness,
)
