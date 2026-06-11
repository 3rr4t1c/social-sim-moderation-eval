"""
User ranking methods for misinformation network analysis.

Available ranking strategies:
- Superspreader-based: TASH-index, Social H-index, Influential index
- Amplifier-based: Early Reposter, Repost Count, Node Strength
- Coordinated-based: Cosine Eigenvector, Cosine Max
- ML-based: Random Forest hybrid ranking
"""

from .superspreaders import (
    TASHIndex,
    TAIIndex,
    social_h_index_ranker,
    influential_ranker,
    tash_index_ranker,
    time_aware_influential_ranker,
    mean_post_credibility_ranker,
)

from .amplifiers import (
    TARIndex,
    EarlyReposterModel,
    repost_count_ranker,
    early_reposter_ranker,
    tar_index_ranker,
    node_degree_ranker,
    node_strength_ranker,
    self_repost_ranker,
    mean_repost_credibility_ranker,
)

from .coordinated import (
    cosine_eigenvector_ranker,
    cosine_max_ranker,
)

from .ml_ranking import (
    random_forest_ranker,
)

from .utils import (
    add_unranked_users,
    build_co_retweet_graph,
    build_reshare_network,
)

# Registry of available rankers with their callable and description
RANKER_REGISTRY = {
    # Superspreader methods
    "tash_index": {
        "func": tash_index_ranker,
        "description": "Time-Aware Social H-index",
        "category": "superspreader",
    },
    "social_h_index": {
        "func": social_h_index_ranker,
        "description": "Social H-index (no time awareness)",
        "category": "superspreader",
    },
    "influential": {
        "func": influential_ranker,
        "description": "Influential index",
        "category": "superspreader",
    },
    "time_aware_influential": {
        "func": time_aware_influential_ranker,
        "description": "Time-Aware Influential index",
        "category": "superspreader",
    },
    "mean_post_credibility": {
        "func": mean_post_credibility_ranker,
        "description": "Mean post credibility score",
        "category": "superspreader",
    },
    # Amplifier methods
    "repost_count": {
        "func": repost_count_ranker,
        "description": "Total repost count",
        "category": "amplifier",
    },
    "early_reposter": {
        "func": early_reposter_ranker,
        "description": "Early reposter ranking",
        "category": "amplifier",
    },
    "tar_index": {
        "func": tar_index_ranker,
        "description": "Time-Aware Repost index",
        "category": "amplifier",
    },
    "node_degree": {
        "func": node_degree_ranker,
        "description": "Node degree in reshare network",
        "category": "amplifier",
    },
    "node_strength": {
        "func": node_strength_ranker,
        "description": "Node strength (weighted degree)",
        "category": "amplifier",
    },
    "self_repost": {
        "func": self_repost_ranker,
        "description": "Self-repost count",
        "category": "amplifier",
    },
    "mean_repost_credibility": {
        "func": mean_repost_credibility_ranker,
        "description": "Mean repost credibility score",
        "category": "amplifier",
    },
    # Coordinated methods
    "cosine_eigenvector": {
        "func": cosine_eigenvector_ranker,
        "description": "Eigenvector centrality on cosine similarity graph",
        "category": "coordinated",
    },
    "cosine_max": {
        "func": cosine_max_ranker,
        "description": "Max cosine similarity",
        "category": "coordinated",
    },
    # ML methods
    "random_forest": {
        "func": random_forest_ranker,
        "description": "Random Forest hybrid ranking",
        "category": "ml",
    },
}


def get_ranker(name: str):
    """Get a ranker function by name."""
    if name not in RANKER_REGISTRY:
        available = ", ".join(RANKER_REGISTRY.keys())
        raise ValueError(f"Unknown ranker '{name}'. Available: {available}")
    return RANKER_REGISTRY[name]["func"]


def list_rankers():
    """List all available rankers with descriptions."""
    return {name: info["description"] for name, info in RANKER_REGISTRY.items()}
