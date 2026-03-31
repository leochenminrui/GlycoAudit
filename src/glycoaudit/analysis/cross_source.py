"""
Cross-Database Generalization Evaluation

Evaluates generalization across different binding databases (CFG, SugarBind, etc.)
to measure domain shift and assess if patterns learned from one source transfer
to another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class CrossSourceManifest:
    """
    Manifest for cross-source evaluation.

    Attributes:
        source_train: Training data source
        source_test: Test data source
        train_glycans: Glycans in training set
        test_glycans: Glycans in test set
        train_agents: Agents in training set
        test_agents: Agents in test set
        glycan_overlap: Glycans present in both sources
        agent_overlap: Agents present in both sources
        seed: Random seed
        metadata: Additional metadata
    """
    source_train: str
    source_test: str
    train_glycans: List[str]
    test_glycans: List[str]
    train_agents: List[str]
    test_agents: List[str]
    glycan_overlap: List[str]
    agent_overlap: List[str]
    seed: int = 42
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'source_train': self.source_train,
            'source_test': self.source_test,
            'n_train_glycans': len(self.train_glycans),
            'n_test_glycans': len(self.test_glycans),
            'n_train_agents': len(self.train_agents),
            'n_test_agents': len(self.test_agents),
            'n_glycan_overlap': len(self.glycan_overlap),
            'n_agent_overlap': len(self.agent_overlap),
            'train_glycans': self.train_glycans,
            'test_glycans': self.test_glycans,
            'train_agents': self.train_agents,
            'test_agents': self.test_agents,
            'glycan_overlap': self.glycan_overlap,
            'agent_overlap': self.agent_overlap,
            'seed': self.seed,
            'metadata': self.metadata or {}
        }

    def save(self, path: Path):
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CrossSourceManifest':
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            source_train=data['source_train'],
            source_test=data['source_test'],
            train_glycans=data['train_glycans'],
            test_glycans=data['test_glycans'],
            train_agents=data['train_agents'],
            test_agents=data['test_agents'],
            glycan_overlap=data['glycan_overlap'],
            agent_overlap=data['agent_overlap'],
            seed=data['seed'],
            metadata=data.get('metadata')
        )


class CrossSourceSplitter:
    """
    Creates train/test splits across different data sources.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def create_cross_source_split(
        self,
        df_labels: pd.DataFrame,
        source_train: str,
        source_test: str,
        min_pos_per_agent: int = 2,
        require_overlap: bool = True
    ) -> CrossSourceManifest:
        """
        Create a cross-source train/test split.

        Args:
            df_labels: DataFrame with binding labels (must have 'data_source' column)
            source_train: Training data source (e.g., 'CFG', 'sugarbind')
            source_test: Test data source
            min_pos_per_agent: Minimum positives per agent to include
            require_overlap: If True, only include overlapping glycans in test

        Returns:
            CrossSourceManifest object
        """
        # Filter by source
        train_labels = df_labels[df_labels['data_source'] == source_train].copy()
        test_labels = df_labels[df_labels['data_source'] == source_test].copy()

        if len(train_labels) == 0:
            raise ValueError(f"No labels found for training source: {source_train}")
        if len(test_labels) == 0:
            raise ValueError(f"No labels found for test source: {source_test}")

        # Get glycans and agents
        train_glycans_all = set(train_labels['glytoucan_id'].unique())
        test_glycans_all = set(test_labels['glytoucan_id'].unique())

        train_agents_all = set(train_labels['agent_id'].unique())
        test_agents_all = set(test_labels['agent_id'].unique())

        # Compute overlap
        glycan_overlap = train_glycans_all & test_glycans_all
        agent_overlap = train_agents_all & test_agents_all

        # For test: optionally restrict to overlapping glycans
        if require_overlap and len(glycan_overlap) > 0:
            test_glycans = list(glycan_overlap)
            test_labels = test_labels[test_labels['glytoucan_id'].isin(glycan_overlap)]
        else:
            test_glycans = list(test_glycans_all)

        train_glycans = list(train_glycans_all)

        # Filter agents by min_pos
        train_agent_counts = train_labels.groupby('agent_id').size()
        valid_train_agents = train_agent_counts[train_agent_counts >= min_pos_per_agent].index.tolist()

        test_agent_counts = test_labels.groupby('agent_id').size()
        valid_test_agents = test_agent_counts[test_agent_counts >= min_pos_per_agent].index.tolist()

        metadata = {
            'source_train': source_train,
            'source_test': source_test,
            'min_pos_per_agent': min_pos_per_agent,
            'require_overlap': require_overlap,
            'n_train_pairs': len(train_labels),
            'n_test_pairs': len(test_labels),
            'description': f'Cross-source evaluation: train on {source_train}, test on {source_test}'
        }

        return CrossSourceManifest(
            source_train=source_train,
            source_test=source_test,
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=valid_train_agents,
            test_agents=valid_test_agents,
            glycan_overlap=list(glycan_overlap),
            agent_overlap=list(agent_overlap),
            seed=self.seed,
            metadata=metadata
        )

    def get_available_sources(self, df_labels: pd.DataFrame, min_pairs: int = 50) -> List[str]:
        """
        Get list of data sources with sufficient data.

        Args:
            df_labels: DataFrame with binding labels
            min_pairs: Minimum number of pairs required

        Returns:
            List of source names
        """
        source_counts = df_labels['data_source'].value_counts()
        return source_counts[source_counts >= min_pairs].index.tolist()

    def create_all_pairwise_splits(
        self,
        df_labels: pd.DataFrame,
        min_pairs: int = 100,
        min_pos_per_agent: int = 2,
        require_overlap: bool = True
    ) -> List[CrossSourceManifest]:
        """
        Create all viable pairwise cross-source splits.

        Args:
            df_labels: DataFrame with binding labels
            min_pairs: Minimum pairs per source to consider
            min_pos_per_agent: Minimum positives per agent
            require_overlap: Require glycan overlap for test set

        Returns:
            List of CrossSourceManifest objects
        """
        sources = self.get_available_sources(df_labels, min_pairs)

        manifests = []
        for i, source_a in enumerate(sources):
            for source_b in sources[i+1:]:
                # Try both directions
                try:
                    manifest_ab = self.create_cross_source_split(
                        df_labels, source_a, source_b,
                        min_pos_per_agent, require_overlap
                    )
                    if len(manifest_ab.test_glycans) >= 10:  # Minimum test set size
                        manifests.append(manifest_ab)
                except (ValueError, KeyError):
                    pass

                try:
                    manifest_ba = self.create_cross_source_split(
                        df_labels, source_b, source_a,
                        min_pos_per_agent, require_overlap
                    )
                    if len(manifest_ba.test_glycans) >= 10:
                        manifests.append(manifest_ba)
                except (ValueError, KeyError):
                    pass

        return manifests


def compute_generalization_gap(
    in_source_metrics: Dict[str, float],
    out_source_metrics: Dict[str, float],
    metric_names: List[str] = ['mrr', 'recall@5', 'recall@10', 'auprc_lb']
) -> Dict[str, float]:
    """
    Compute generalization gap between in-source and out-of-source performance.

    Args:
        in_source_metrics: Metrics evaluated on same source as training
        out_source_metrics: Metrics evaluated on different source
        metric_names: Names of metrics to compute gaps for

    Returns:
        Dictionary of gap values (positive = degradation, negative = improvement)
    """
    gaps = {}
    for metric in metric_names:
        if metric in in_source_metrics and metric in out_source_metrics:
            in_val = in_source_metrics[metric]
            out_val = out_source_metrics[metric]

            # Gap = in_source - out_source (positive means degradation)
            gap = in_val - out_val
            gaps[f'{metric}_gap'] = gap
            gaps[f'{metric}_gap_pct'] = (gap / in_val * 100) if in_val != 0 else 0.0

    return gaps


def analyze_source_characteristics(
    df_labels: pd.DataFrame,
    df_features: pd.DataFrame,
    source_name: str
) -> Dict:
    """
    Analyze characteristics of a data source.

    Args:
        df_labels: DataFrame with binding labels
        df_features: DataFrame with glycan features
        source_name: Name of the source to analyze

    Returns:
        Dictionary of characteristics
    """
    source_labels = df_labels[df_labels['data_source'] == source_name]

    glycan_ids = source_labels['glytoucan_id'].unique()
    source_features = df_features[df_features['glytoucan_id'].isin(glycan_ids)]

    characteristics = {
        'source': source_name,
        'n_pairs': len(source_labels),
        'n_glycans': len(glycan_ids),
        'n_agents': source_labels['agent_id'].nunique(),
        'pairs_per_glycan': len(source_labels) / len(glycan_ids) if len(glycan_ids) > 0 else 0,
        'pairs_per_agent': len(source_labels) / source_labels['agent_id'].nunique() if source_labels['agent_id'].nunique() > 0 else 0,
    }

    # Feature distributions (if available)
    if len(source_features) > 0:
        feature_cols = ['n_residues', 'radius_of_gyration', 'compactness',
                       'branch_proxy', 'terminal_proxy']
        for col in feature_cols:
            if col in source_features.columns:
                values = source_features[col].dropna()
                if len(values) > 0:
                    characteristics[f'{col}_mean'] = float(values.mean())
                    characteristics[f'{col}_std'] = float(values.std())

    return characteristics


def compute_bootstrap_confidence_intervals(
    per_agent_metrics: pd.DataFrame,
    metric_names: List[str] = ['mrr', 'recall@5', 'recall@10', 'auprc_lb'],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for cross-source metrics.

    Args:
        per_agent_metrics: DataFrame with per-agent metrics
        metric_names: List of metrics to compute CIs for
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        seed: Random seed

    Returns:
        Dictionary with CI bounds for each metric
    """
    rng = np.random.default_rng(seed)
    n_agents = len(per_agent_metrics)

    if n_agents == 0:
        return {}

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {}

    for metric in metric_names:
        if metric not in per_agent_metrics.columns:
            continue

        # Get metric values
        values = per_agent_metrics[metric].dropna().values

        if len(values) == 0:
            continue

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            bootstrap_means.append(sample.mean())

        bootstrap_means = np.array(bootstrap_means)

        # Compute CIs
        results[metric] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'ci_lower': float(np.percentile(bootstrap_means, lower_percentile)),
            'ci_upper': float(np.percentile(bootstrap_means, upper_percentile)),
            'bootstrap_std': float(bootstrap_means.std()),
            'n_agents': len(values),
        }

    return results


def compare_source_overlap_distributions(
    manifest: CrossSourceManifest,
    df_features: pd.DataFrame,
    glytoucan_path: Path = None
) -> Dict:
    """
    Compare structural distribution of overlapping glycans vs full sets.

    This analysis helps understand if cross-source evaluation is testing
    on a representative subset or a biased sample.

    Args:
        manifest: CrossSourceManifest with overlap information
        df_features: DataFrame with glycan features
        glytoucan_path: Path to GlyTouCan data for structure keys

    Returns:
        Dictionary with distribution comparisons
    """
    train_set = set(manifest.train_glycans)
    test_set = set(manifest.test_glycans)
    overlap_set = set(manifest.glycan_overlap)

    # Feature-based comparison
    train_features = df_features[df_features['glytoucan_id'].isin(train_set)]
    test_features = df_features[df_features['glytoucan_id'].isin(test_set)]
    overlap_features = df_features[df_features['glytoucan_id'].isin(overlap_set)]

    feature_cols = ['n_residues', 'branch_proxy', 'terminal_proxy', 'compactness']

    comparison = {
        'n_train': len(train_set),
        'n_test': len(test_set),
        'n_overlap': len(overlap_set),
        'overlap_fraction_of_test': len(overlap_set) / len(test_set) if test_set else 0,
        'feature_distributions': {}
    }

    for col in feature_cols:
        if col in train_features.columns:
            comparison['feature_distributions'][col] = {
                'train_mean': float(train_features[col].mean()) if len(train_features) > 0 else 0,
                'test_mean': float(test_features[col].mean()) if len(test_features) > 0 else 0,
                'overlap_mean': float(overlap_features[col].mean()) if len(overlap_features) > 0 else 0,
            }

    # Structure key-based comparison (if available)
    if glytoucan_path and glytoucan_path.exists():
        try:
            from analysis.structure_keys import StructureKeyGenerator

            generator = StructureKeyGenerator(glytoucan_path)

            train_scaffolds = set()
            test_scaffolds = set()
            overlap_scaffolds = set()

            for gid in train_set:
                train_scaffolds.add(generator.compute_scaffold_key(gid))
            for gid in test_set:
                test_scaffolds.add(generator.compute_scaffold_key(gid))
            for gid in overlap_set:
                overlap_scaffolds.add(generator.compute_scaffold_key(gid))

            comparison['scaffold_analysis'] = {
                'n_train_scaffolds': len(train_scaffolds),
                'n_test_scaffolds': len(test_scaffolds),
                'n_overlap_scaffolds': len(overlap_scaffolds),
                'scaffold_coverage_of_test': len(overlap_scaffolds) / len(test_scaffolds) if test_scaffolds else 0,
            }

        except Exception as e:
            comparison['scaffold_analysis'] = {'error': str(e)}

    return comparison
