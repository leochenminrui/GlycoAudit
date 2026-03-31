"""
Feature Selection and Management

This module provides clean feature set selection for ablation studies:
- SSV-only: Shape/Structure/Volume features (8 columns)
- GCV-only: Graph/Conformation/Variability features (8 columns)
- SSV+GCV: All features combined (16 columns)
"""

from __future__ import annotations

from typing import List

import pandas as pd


# Feature column definitions
SSV_FEATURES = [
    'n_atoms',
    'n_residues',
    'radius_of_gyration',
    'max_pair_distance',
    'compactness',
    'branch_proxy',
    'terminal_proxy',
    'exposure_proxy',
]

GCV_FEATURES = [
    'contact_density',
    'long_range_contact_fraction',
    'mean_residue_neighbor_count',
    'sd_residue_neighbor_count',
    'torsion_diversity',
    'graph_laplacian_spectral_gap',
    'core_periphery_ratio',
    'max_contact_distance_seq',
]

FEATURE_SETS = {
    'ssv': SSV_FEATURES,
    'gcv': GCV_FEATURES,
    'ssv+gcv': SSV_FEATURES + GCV_FEATURES,
}


def list_feature_columns(feature_set: str) -> List[str]:
    """
    Get list of feature columns for a given feature set.

    Args:
        feature_set: One of 'ssv', 'gcv', 'ssv+gcv'

    Returns:
        List of feature column names

    Raises:
        ValueError: If feature_set is not recognized
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Must be one of: {list(FEATURE_SETS.keys())}"
        )
    return FEATURE_SETS[feature_set].copy()


def select_feature_columns(
    df_ssv: pd.DataFrame,
    df_gcv: pd.DataFrame,
    feature_set: str
) -> pd.DataFrame:
    """
    Select and merge feature columns based on feature set.

    Args:
        df_ssv: DataFrame with SSV features
        df_gcv: DataFrame with GCV features
        feature_set: One of 'ssv', 'gcv', 'ssv+gcv'

    Returns:
        DataFrame with selected features + glytoucan_id column

    Raises:
        ValueError: If feature_set is not recognized
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Must be one of: {list(FEATURE_SETS.keys())}"
        )

    # Start with glytoucan_id from SSV (both should have same IDs)
    df_result = df_ssv[['glytoucan_id']].copy()

    # Add selected features
    if feature_set in ['ssv', 'ssv+gcv']:
        for col in SSV_FEATURES:
            df_result[col] = df_ssv[col]

    if feature_set in ['gcv', 'ssv+gcv']:
        for col in GCV_FEATURES:
            df_result[col] = df_gcv[col]

    # Validate column count
    expected_count = len(FEATURE_SETS[feature_set])
    actual_count = len(df_result.columns) - 1  # Exclude glytoucan_id
    assert actual_count == expected_count, (
        f"Feature selection failed: expected {expected_count} columns, "
        f"got {actual_count} for feature_set='{feature_set}'"
    )

    return df_result


def get_feature_set_description(feature_set: str) -> str:
    """
    Get human-readable description of feature set.

    Args:
        feature_set: One of 'ssv', 'gcv', 'ssv+gcv'

    Returns:
        Description string
    """
    descriptions = {
        'ssv': 'Shape/Structure/Volume features (8 cols)',
        'gcv': 'Graph/Conformation/Variability features (8 cols)',
        'ssv+gcv': 'All features combined (16 cols)',
    }
    return descriptions.get(feature_set, 'Unknown feature set')


def get_feature_count(feature_set: str) -> int:
    """
    Get number of features in a feature set.

    Args:
        feature_set: One of 'ssv', 'gcv', 'ssv+gcv'

    Returns:
        Number of features
    """
    return len(list_feature_columns(feature_set))
