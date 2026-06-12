"""
Machine Learning hybrid ranking methods.

Combines multiple ranking features using supervised learning to predict
user scores based on historical data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from .utils import split_data, add_unranked_users
from . import amplifiers as amp
from . import coordinated as coo
from . import superspreaders as sup


def extract_archetype_features(
    reshare_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Extract features from multiple ranking methods.

    Uses a selection of rankers as feature extractors to capture different
    aspects of user behavior. These are the six features described in the
    paper (code name -> paper name):

    - repost_count            -> Repost Count
    - early_reposter          -> Early Reposter Index (EaR-index)
    - cosine_eigenvector      -> Coordination Centrality
    - cosine_max              -> Coordination Edge Weight
    - time_aware_influential  -> Time Aware Influence Score (TAI-score)
    - tash_index              -> Time Aware Social H-index (TASH-index)

    Args:
        reshare_data: DataFrame with reshare data
        credibility_threshold: Threshold for low-credibility content
        verbose: Whether to show progress bar

    Returns:
        Dictionary mapping user_id to list of ranking scores
    """
    rankers = [
        lambda x: amp.repost_count_ranker(x, credibility_threshold=credibility_threshold),
        lambda x: amp.early_reposter_ranker(x, credibility_threshold=credibility_threshold),
        lambda x: coo.cosine_eigenvector_ranker(x, credibility_threshold=credibility_threshold),
        lambda x: coo.cosine_max_ranker(x, credibility_threshold=credibility_threshold),
        lambda x: sup.time_aware_influential_ranker(x, credibility_threshold=credibility_threshold),
        lambda x: sup.tash_index_ranker(x, credibility_threshold=credibility_threshold),
    ]

    result: Dict[str, List[float]] = {}

    for ranker in tqdm(rankers, disable=not verbose):
        ranking = ranker(reshare_data)

        for user_id, score in ranking:
            if user_id not in result:
                result[user_id] = []
            result[user_id].append(score)

    return result


def extract_all_features(
    reshare_data: pd.DataFrame,
    credibility_threshold: float = 39.0,
    verbose: bool = True,
) -> List[Tuple[str, List[float]]]:
    """
    Extract all features for machine learning.

    Currently this is exactly the six archetype features described in the
    paper (see :func:`extract_archetype_features`).

    Args:
        reshare_data: DataFrame with reshare data
        credibility_threshold: Threshold for low-credibility content
        verbose: Whether to show progress

    Returns:
        List of (user_id, feature_vector) tuples sorted by user_id
    """
    if verbose:
        print("Extracting archetype features...")
    archetype_features = extract_archetype_features(
        reshare_data,
        credibility_threshold=credibility_threshold,
        verbose=verbose,
    )

    return sorted(archetype_features.items(), key=lambda x: x[0])


def prepare_training_data(
    feature_vectors: Dict[str, List[float]],
    labels: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Prepare feature vectors and labels for model training.

    Args:
        feature_vectors: Dictionary mapping user_id to feature vector
        labels: Dictionary mapping user_id to target label

    Returns:
        Tuple of (X_train, y_train, user_id_to_index_mapping)
    """
    X_train, y_train = [], []
    user_to_idx: Dict[str, int] = {}

    for i, (user_id, features) in enumerate(feature_vectors.items()):
        user_to_idx[user_id] = i
        X_train.append(features)
        y_train.append(labels.get(user_id, 0))

    return np.array(X_train), np.array(y_train), user_to_idx


def create_training_dataset(
    train_features_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    credibility_threshold: float = 39.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Create a complete training dataset.

    Args:
        train_features_df: DataFrame for feature extraction
        train_labels_df: DataFrame for label generation
        credibility_threshold: Threshold for low-credibility content
        verbose: Whether to show progress

    Returns:
        Tuple of (X_train, y_train, user_id_to_index_mapping)
    """
    # Extract features from training period
    feature_vectors = dict(extract_all_features(
        train_features_df,
        credibility_threshold=credibility_threshold,
        verbose=verbose,
    ))

    # Generate labels from evaluation period
    gt_ranking = amp.node_strength_ranker(
        train_labels_df,
        outgoing_only=False,
    )
    labels = dict(gt_ranking)

    return prepare_training_data(feature_vectors, labels)


def random_forest_ranker(
    train_data: pd.DataFrame,
    train_tail_ratio: float = 0.2,
    credibility_threshold: float = 39.0,
    random_seed: int = 42,
    external_model: Optional[RandomForestRegressor] = None,
    use_log_input: bool = False,
    use_log_target: bool = False,
    **kwargs,
) -> List[Tuple]:
    """
    Machine learning based ranker using Random Forest.

    Trains a Random Forest regressor to predict user scores based on
    features extracted from multiple ranking methods.

    Args:
        train_data: DataFrame with reshare data
        train_tail_ratio: Fraction of data to use for label generation
        credibility_threshold: Threshold for low-credibility content
        random_seed: Random seed for reproducibility
        external_model: Pre-trained model to use instead of training new one
        use_log_input: Whether to log-transform input features
        use_log_target: Whether to log-transform target labels

    Returns:
        List of (user_id, predicted_score) tuples sorted descending
    """
    # Sort training data by time so the internal split is always chronological.
    train_sorted = train_data.sort_values("time_delta", kind="stable").reset_index(drop=True)

    # Extract features from the FULL training period.
    # Using train_data (not just a head slice) for both training and prediction
    # eliminates the distribution shift that arises when the model is trained on
    # a feature scale derived from a shorter period than the one used for prediction.
    all_features = extract_all_features(
        train_sorted,
        credibility_threshold=credibility_threshold,
        verbose=False,
    )
    all_features_dict = dict(all_features)

    if external_model is None:
        # Create and configure model
        model = RandomForestRegressor(
            random_state=random_seed,
            n_jobs=-1,
            criterion="squared_error",
            bootstrap=True,
            max_depth=5,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=200,
        )

        # Labels come from the LAST tail_ratio fraction (chronologically).
        # This represents user importance right before the test horizon — the
        # most predictive signal for test-period behavior.
        _, train_tail = split_data(train_sorted, tail_ratio=train_tail_ratio)

        labels = dict(amp.node_strength_ranker(
            train_tail,
            credibility_threshold=credibility_threshold,
            outgoing_only=False,
        ))

        X_train, y_train, _ = prepare_training_data(all_features_dict, labels)

        if use_log_input:
            X_train = np.log1p(X_train)
        if use_log_target:
            y_train = np.log1p(y_train)

        model.fit(X_train, y_train)
    else:
        model = external_model

    # Predict scores using the same feature vectors extracted above (no distribution shift).
    user_ids = [uid for uid, _ in all_features]
    X_test = np.array([all_features_dict[uid] for uid in user_ids])

    if use_log_input:
        X_test = np.log1p(X_test)

    predictions = model.predict(X_test)

    ranking = [(uid, float(pred)) for uid, pred in zip(user_ids, predictions)]

    # Ensure all training users are covered (add any user with score 0 at the bottom).
    add_unranked_users(train_data, ranking)

    return sorted(ranking, key=lambda x: x[1], reverse=True)
