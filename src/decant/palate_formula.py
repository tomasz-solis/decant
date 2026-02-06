"""
Centralized Palate Formula

SINGLE SOURCE OF TRUTH for palate score calculations.
Eliminates code duplication between app.py, predictor.py, and palate_engine.py.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union
from decant.constants import AlgorithmConstants, ColumnNames
from decant.utils import safe_divide, logger


def calculate_palate_features(
    wine_data: Union[Dict[str, float], pd.Series],
    use_constants: bool = True
) -> Dict[str, float]:
    """
    Calculate derived palate features from raw wine features.

    This is the SINGLE SOURCE OF TRUTH for palate formula.
    All other code should call this function instead of duplicating logic.

    Args:
        wine_data: Dict or Series with acidity, minerality, body keys
        use_constants: Whether to use constants (True) or raw values (False)

    Returns:
        Dict with structure_score, acidity_body_ratio, palate_score

    Formula:
        structure_score = acidity + minerality
        acidity_body_ratio = acidity / (body + ε)
        palate_score = structure_score + (acidity_body_ratio * weight)

    Where:
        ε = ACIDITY_BODY_EPSILON (prevents division by zero)
        weight = ACIDITY_BODY_WEIGHT (emphasis on ratio)
    """
    # Extract values (works for both dict and Series)
    if isinstance(wine_data, pd.Series):
        acidity = wine_data.get(ColumnNames.ACIDITY, 0)
        minerality = wine_data.get(ColumnNames.MINERALITY, 0)
        body = wine_data.get(ColumnNames.BODY, 0)
    else:  # dict
        acidity = wine_data.get(ColumnNames.ACIDITY, 0)
        minerality = wine_data.get(ColumnNames.MINERALITY, 0)
        body = wine_data.get(ColumnNames.BODY, 0)

    # Use constants or raw values
    epsilon = AlgorithmConstants.ACIDITY_BODY_EPSILON if use_constants else 0.1
    weight = AlgorithmConstants.ACIDITY_BODY_WEIGHT if use_constants else 2.0

    # Calculate derived features
    structure_score = acidity + minerality

    acidity_body_ratio = safe_divide(
        acidity,
        body + epsilon,
        default=acidity  # Fallback to just acidity if body is 0
    )

    palate_score = structure_score + (acidity_body_ratio * weight)

    return {
        ColumnNames.STRUCTURE_SCORE: structure_score,
        ColumnNames.ACIDITY_BODY_RATIO: acidity_body_ratio,
        ColumnNames.PALATE_SCORE: palate_score
    }


def add_palate_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add palate features to entire DataFrame.

    Args:
        df: DataFrame with wine data

    Returns:
        DataFrame with added columns: structure_score, acidity_body_ratio, palate_score
    """
    if len(df) == 0:
        return df

    df = df.copy()

    # Vectorized calculation for entire dataframe
    df[ColumnNames.STRUCTURE_SCORE] = (
        df[ColumnNames.ACIDITY] + df[ColumnNames.MINERALITY]
    )

    df[ColumnNames.ACIDITY_BODY_RATIO] = df.apply(
        lambda row: safe_divide(
            row[ColumnNames.ACIDITY],
            row[ColumnNames.BODY] + AlgorithmConstants.ACIDITY_BODY_EPSILON,
            default=row[ColumnNames.ACIDITY]
        ),
        axis=1
    )

    df[ColumnNames.PALATE_SCORE] = (
        df[ColumnNames.STRUCTURE_SCORE] +
        (df[ColumnNames.ACIDITY_BODY_RATIO] * AlgorithmConstants.ACIDITY_BODY_WEIGHT)
    )

    logger.debug(f"Added palate features to {len(df)} wines")

    return df


def calculate_wine_similarity(
    wine_features: Dict[str, float],
    target_features: Dict[str, float]
) -> float:
    """
    Calculate similarity between two wines using unified Palate Formula.

    Uses Euclidean distance weighted by palate formula features.

    Args:
        wine_features: Features of wine in history
        target_features: Features of target wine to compare

    Returns:
        Euclidean distance (lower = more similar)
    """
    # Calculate palate features for both wines
    wine_palate = calculate_palate_features(wine_features)
    target_palate = calculate_palate_features(target_features)

    # Create feature vectors including both raw and derived features
    wine_vec = np.array([
        wine_features.get(ColumnNames.ACIDITY, 0),
        wine_features.get(ColumnNames.MINERALITY, 0),
        wine_features.get(ColumnNames.FRUITINESS, 0),
        wine_features.get(ColumnNames.TANNIN, 0),
        wine_features.get(ColumnNames.BODY, 0),
        wine_palate[ColumnNames.STRUCTURE_SCORE] / 2,  # Normalize (0-20 → 0-10)
        wine_palate[ColumnNames.ACIDITY_BODY_RATIO],
        wine_palate[ColumnNames.PALATE_SCORE] / 3  # Normalize
    ])

    target_vec = np.array([
        target_features.get(ColumnNames.ACIDITY, 0),
        target_features.get(ColumnNames.MINERALITY, 0),
        target_features.get(ColumnNames.FRUITINESS, 0),
        target_features.get(ColumnNames.TANNIN, 0),
        target_features.get(ColumnNames.BODY, 0),
        target_palate[ColumnNames.STRUCTURE_SCORE] / 2,
        target_palate[ColumnNames.ACIDITY_BODY_RATIO],
        target_palate[ColumnNames.PALATE_SCORE] / 3
    ])

    return np.linalg.norm(wine_vec - target_vec)


# Export key functions
__all__ = [
    'calculate_palate_features',
    'add_palate_features_to_dataframe',
    'calculate_wine_similarity'
]
