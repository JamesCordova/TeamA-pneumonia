"""
Feature selection and importance utilities.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def select_relevant_features(
    features: pd.DataFrame,
    importance_threshold: float = 0.01,
    method: str = "variance",
) -> pd.DataFrame:
    """
    Filter features by variance or pairwise correlation.

    Args:
        features: Feature DataFrame.
        importance_threshold:
            variance  — minimum variance a feature must have to be kept.
            correlation — maximum allowed absolute pairwise correlation (r > threshold
                          causes one of the pair to be dropped).
        method: 'variance' or 'correlation'.

    Returns:
        DataFrame with selected features.
    """
    if not 0 <= importance_threshold <= 1:
        raise ValueError(f"importance_threshold must be in [0,1], got {importance_threshold}")
    if method not in ("variance", "correlation"):
        raise ValueError(f"method must be 'variance' or 'correlation', got {method}")

    original_cols = list(features.columns)

    if method == "variance":
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=importance_threshold)
        selector.fit(features)
        selected = features.loc[:, selector.get_support()]

    else:  # correlation
        corr = features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = [c for c in upper.columns if (upper[c] > importance_threshold).any()]
        selected = features.drop(columns=to_drop)

    dropped = [c for c in original_cols if c not in selected.columns]
    if dropped:
        logger.info(
            f"select_relevant_features ({method}): "
            f"dropped {len(dropped)} features: {dropped}"
        )
    else:
        logger.info(
            f"select_relevant_features ({method}): "
            f"all {len(original_cols)} features retained"
        )

    return selected


def get_feature_importance(
    model,
    feature_names: List[str],
    method: str = "default",
) -> Dict[str, float]:
    """
    Extract normalized feature importance scores from a fitted model.

    Supports RandomForestModel (._rf) and XGBoostModel (._xgb) wrappers,
    as well as bare sklearn estimators with feature_importances_.

    Args:
        model: Fitted model wrapper or sklearn estimator.
        feature_names: Feature column names in training order.
        method: Currently only 'default' (model-native importances).

    Returns:
        Dict mapping feature_name -> importance, sorted descending,
        normalized to sum to 1.0.
    """
    if method != "default":
        raise ValueError(f"Only 'default' importance method is supported, got {method}")

    # Unwrap project model wrappers to their underlying estimator
    if hasattr(model, '_rf'):
        estimator = model._rf
    elif hasattr(model, '_xgb'):
        estimator = model._xgb
    else:
        estimator = model

    if not hasattr(estimator, 'feature_importances_'):
        raise AttributeError(
            f"{type(estimator).__name__} has no feature_importances_. "
            "Ensure the model is fitted."
        )

    raw = np.array(estimator.feature_importances_)

    if len(raw) != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"importances has {len(raw)}"
        )

    total = raw.sum()
    normalized = (raw / total).tolist() if total > 0 else raw.tolist()

    result = dict(sorted(zip(feature_names, normalized), key=lambda x: x[1], reverse=True))

    top5 = list(result.items())[:5]
    logger.info(f"get_feature_importance — top 5: {top5}")
    return result
