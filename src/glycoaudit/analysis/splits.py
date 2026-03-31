"""
Structure-Aware Splitting Strategies for Glycan Binding Benchmark

This module implements various split protocols to evaluate generalization:
1. IID (random) split - baseline
2. Agent holdout - test on unseen agents
3. Scaffold holdout - test on unseen glycan topologies
4. Terminal motif holdout - test on unseen terminal patterns
5. Linkage holdout - test on unseen linkage patterns

All splits are deterministic with fixed seeds and include sanity checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Import structure key generator for TRUE structure-aware splits
try:
    from analysis.structure_keys import StructureKeyGenerator, load_structure_keys_for_dataset
    STRUCTURE_KEYS_AVAILABLE = True
except ImportError:
    STRUCTURE_KEYS_AVAILABLE = False
    print("[Warning] structure_keys module not available, TRUE splits disabled")


@dataclass
class SplitManifest:
    """
    Manifest describing a train/test split.

    Attributes:
        split_type: Type of split (iid, agent_holdout, scaffold_holdout, etc.)
        train_glycans: Set of training glycan IDs
        test_glycans: Set of test glycan IDs
        train_agents: Set of training agent IDs (for agent holdout)
        test_agents: Set of test agent IDs (for agent holdout)
        group_key: Description of grouping variable (for structure-aware splits)
        n_train_groups: Number of unique groups in train
        n_test_groups: Number of unique groups in test
        seed: Random seed used
        metadata: Additional metadata
    """
    split_type: str
    train_glycans: List[str]
    test_glycans: List[str]
    train_agents: Optional[List[str]] = None
    test_agents: Optional[List[str]] = None
    group_key: Optional[str] = None
    n_train_groups: Optional[int] = None
    n_test_groups: Optional[int] = None
    seed: int = 42
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'split_type': self.split_type,
            'n_train_glycans': len(self.train_glycans),
            'n_test_glycans': len(self.test_glycans),
            'n_train_agents': len(self.train_agents) if self.train_agents else None,
            'n_test_agents': len(self.test_agents) if self.test_agents else None,
            'train_glycans': self.train_glycans,
            'test_glycans': self.test_glycans,
            'train_agents': self.train_agents,
            'test_agents': self.test_agents,
            'group_key': self.group_key,
            'n_train_groups': self.n_train_groups,
            'n_test_groups': self.n_test_groups,
            'seed': self.seed,
            'metadata': self.metadata or {}
        }

    def save(self, path: Path):
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SplitManifest':
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            split_type=data['split_type'],
            train_glycans=data['train_glycans'],
            test_glycans=data['test_glycans'],
            train_agents=data.get('train_agents'),
            test_agents=data.get('test_agents'),
            group_key=data.get('group_key'),
            n_train_groups=data.get('n_train_groups'),
            n_test_groups=data.get('n_test_groups'),
            seed=data['seed'],
            metadata=data.get('metadata')
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate split integrity.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check no overlap in glycans (except for agent holdout where overlap is expected)
        if self.split_type != 'agent_holdout':
            train_set = set(self.train_glycans)
            test_set = set(self.test_glycans)
            overlap = train_set & test_set
            if overlap:
                issues.append(f"Glycan overlap: {len(overlap)} glycans in both train and test")

        # Check no overlap in agents (if agent holdout)
        if self.split_type == 'agent_holdout' and self.train_agents and self.test_agents:
            train_agent_set = set(self.train_agents)
            test_agent_set = set(self.test_agents)
            agent_overlap = train_agent_set & test_agent_set
            if agent_overlap:
                issues.append(f"Agent overlap: {len(agent_overlap)} agents in both train and test")

        # Check non-empty
        if not self.train_glycans:
            issues.append("Empty train set")
        if not self.test_glycans:
            issues.append("Empty test set")

        return len(issues) == 0, issues


class GlycanSplitter:
    """
    Factory class for creating various glycan-agent split strategies.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def iid_split(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        test_size: float = 0.2,
        min_train_pos: int = 2
    ) -> SplitManifest:
        """
        IID (random) split of glycans.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            test_size: Fraction for test set
            min_train_pos: Minimum positives per agent in train

        Returns:
            SplitManifest object
        """
        glycan_ids = df_features['glytoucan_id'].values
        n_test = max(1, int(len(glycan_ids) * test_size))

        # Shuffle and split
        shuffled_ids = glycan_ids.copy()
        self.rng.shuffle(shuffled_ids)

        test_glycans = shuffled_ids[:n_test].tolist()
        train_glycans = shuffled_ids[n_test:].tolist()

        # Build agent lists (all agents in both sets for IID)
        all_agents = df_labels['agent_id'].unique().tolist()

        metadata = {
            'test_size': test_size,
            'min_train_pos': min_train_pos,
            'description': 'IID random split of glycans'
        }

        return SplitManifest(
            split_type='iid',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            seed=self.seed,
            metadata=metadata
        )

    def agent_holdout_split(
        self,
        df_labels: pd.DataFrame,
        test_size: float = 0.2,
        min_pos: int = 2
    ) -> SplitManifest:
        """
        Agent holdout split - test on unseen agents.

        Args:
            df_labels: DataFrame with binding labels
            test_size: Fraction of agents for test
            min_pos: Minimum positives per agent to include

        Returns:
            SplitManifest object
        """
        # Count positives per agent
        agent_counts = df_labels.groupby('agent_id').size()
        valid_agents = agent_counts[agent_counts >= min_pos].index.values

        # Shuffle and split agents
        shuffled_agents = valid_agents.copy()
        self.rng.shuffle(shuffled_agents)

        n_test = max(1, int(len(valid_agents) * test_size))
        test_agents = shuffled_agents[:n_test].tolist()
        train_agents = shuffled_agents[n_test:].tolist()

        # Get glycans for each agent set
        train_glycans = df_labels[df_labels['agent_id'].isin(train_agents)]['glytoucan_id'].unique().tolist()
        test_glycans = df_labels[df_labels['agent_id'].isin(test_agents)]['glytoucan_id'].unique().tolist()

        metadata = {
            'test_size': test_size,
            'min_pos': min_pos,
            'n_agents_excluded': len(agent_counts) - len(valid_agents),
            'description': 'Agent holdout split - agents are disjoint between train and test'
        }

        return SplitManifest(
            split_type='agent_holdout',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=train_agents,
            test_agents=test_agents,
            seed=self.seed,
            metadata=metadata
        )

    def _compute_scaffold_key(self, row: pd.Series, terciles: np.ndarray) -> str:
        """
        Compute scaffold key from glycan features.

        Scaffold = size_bin + branch_proxy + terminal_proxy
        This creates a topology signature.
        """
        n_res = row['n_residues']
        if n_res <= terciles[0]:
            size_bin = 'small'
        elif n_res <= terciles[1]:
            size_bin = 'medium'
        else:
            size_bin = 'large'

        branch = int(row['branch_proxy']) if not pd.isna(row['branch_proxy']) else 0
        terminal = int(row['terminal_proxy']) if not pd.isna(row['terminal_proxy']) else 0

        return f"{size_bin}_b{branch}_t{terminal}"

    def scaffold_holdout_split(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        n_folds: int = 5,
        test_fold: int = 0
    ) -> SplitManifest:
        """
        Scaffold-based split - test on unseen glycan topologies.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            n_folds: Number of cross-validation folds
            test_fold: Which fold to use as test (0 to n_folds-1)

        Returns:
            SplitManifest object
        """
        # Compute scaffold keys
        terciles = df_features['n_residues'].quantile([0.33, 0.67]).values
        df_features = df_features.copy()
        df_features['scaffold'] = df_features.apply(
            lambda row: self._compute_scaffold_key(row, terciles),
            axis=1
        )

        # Group K-fold split
        glycan_ids = df_features['glytoucan_id'].values
        scaffolds = df_features['scaffold'].values

        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(glycan_ids, groups=scaffolds))

        if test_fold >= len(splits):
            test_fold = 0

        train_idx, test_idx = splits[test_fold]

        train_glycans = glycan_ids[train_idx].tolist()
        test_glycans = glycan_ids[test_idx].tolist()

        # Get scaffolds per set
        train_scaffolds = set(scaffolds[train_idx])
        test_scaffolds = set(scaffolds[test_idx])

        # All agents available in both sets
        all_agents = df_labels['agent_id'].unique().tolist()

        metadata = {
            'n_folds': n_folds,
            'test_fold': test_fold,
            'scaffold_definition': 'size_bin + branch_proxy + terminal_proxy',
            'tercile_bounds': terciles.tolist(),
            'n_unique_scaffolds': len(set(scaffolds)),
            'scaffold_overlap': len(train_scaffolds & test_scaffolds),
            'description': 'Scaffold-based split to prevent topology leakage'
        }

        return SplitManifest(
            split_type='scaffold_holdout',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            group_key='scaffold',
            n_train_groups=len(train_scaffolds),
            n_test_groups=len(test_scaffolds),
            seed=self.seed,
            metadata=metadata
        )

    def _compute_terminal_motif_key(self, row: pd.Series) -> str:
        """
        Compute terminal motif key from glycan features.

        Uses terminal_proxy + exposure_proxy as proxy for terminal pattern.
        """
        terminal = int(row['terminal_proxy']) if not pd.isna(row['terminal_proxy']) else 0
        exposure = row['exposure_proxy'] if not pd.isna(row['exposure_proxy']) else 0.0

        # Bin exposure into quartiles
        if exposure < 14:
            exp_bin = 'low'
        elif exposure < 16:
            exp_bin = 'med'
        else:
            exp_bin = 'high'

        return f"t{terminal}_{exp_bin}"

    def terminal_motif_holdout_split(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        n_folds: int = 4,
        test_fold: int = 0
    ) -> SplitManifest:
        """
        Terminal motif split - test on unseen terminal patterns.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            n_folds: Number of cross-validation folds
            test_fold: Which fold to use as test

        Returns:
            SplitManifest object
        """
        # Compute terminal motif keys
        df_features = df_features.copy()
        df_features['terminal_motif'] = df_features.apply(
            self._compute_terminal_motif_key, axis=1
        )

        glycan_ids = df_features['glytoucan_id'].values
        motifs = df_features['terminal_motif'].values

        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(glycan_ids, groups=motifs))

        if test_fold >= len(splits):
            test_fold = 0

        train_idx, test_idx = splits[test_fold]

        train_glycans = glycan_ids[train_idx].tolist()
        test_glycans = glycan_ids[test_idx].tolist()

        train_motifs = set(motifs[train_idx])
        test_motifs = set(motifs[test_idx])

        all_agents = df_labels['agent_id'].unique().tolist()

        metadata = {
            'n_folds': n_folds,
            'test_fold': test_fold,
            'motif_definition': 'terminal_proxy + binned_exposure_proxy',
            'n_unique_motifs': len(set(motifs)),
            'motif_overlap': len(train_motifs & test_motifs),
            'description': 'Terminal motif split to test on unseen terminal patterns'
        }

        return SplitManifest(
            split_type='terminal_motif_holdout',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            group_key='terminal_motif',
            n_train_groups=len(train_motifs),
            n_test_groups=len(test_motifs),
            seed=self.seed,
            metadata=metadata
        )

    def _compute_linkage_key(self, row: pd.Series) -> str:
        """
        Compute linkage pattern key from available features.

        Since explicit linkage info isn't available, use:
        - compactness (proxy for linkage tightness)
        - branch_proxy (branching pattern affects linkages)
        """
        compactness = row['compactness'] if not pd.isna(row['compactness']) else 0.0
        branch = int(row['branch_proxy']) if not pd.isna(row['branch_proxy']) else 0

        # Bin compactness
        if compactness < 0.33:
            comp_bin = 'loose'
        elif compactness < 0.36:
            comp_bin = 'medium'
        else:
            comp_bin = 'tight'

        return f"b{branch}_{comp_bin}"

    def linkage_holdout_split(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        n_folds: int = 4,
        test_fold: int = 0
    ) -> SplitManifest:
        """
        Linkage pattern split - test on unseen linkage patterns.

        Note: This is a proxy since explicit linkage annotations aren't available.
        Uses compactness + branching as linkage pattern proxy.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            n_folds: Number of cross-validation folds
            test_fold: Which fold to use as test

        Returns:
            SplitManifest object
        """
        # Compute linkage keys
        df_features = df_features.copy()
        df_features['linkage_pattern'] = df_features.apply(
            self._compute_linkage_key, axis=1
        )

        glycan_ids = df_features['glytoucan_id'].values
        linkages = df_features['linkage_pattern'].values

        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(glycan_ids, groups=linkages))

        if test_fold >= len(splits):
            test_fold = 0

        train_idx, test_idx = splits[test_fold]

        train_glycans = glycan_ids[train_idx].tolist()
        test_glycans = glycan_ids[test_idx].tolist()

        train_linkages = set(linkages[train_idx])
        test_linkages = set(linkages[test_idx])

        all_agents = df_labels['agent_id'].unique().tolist()

        metadata = {
            'n_folds': n_folds,
            'test_fold': test_fold,
            'linkage_definition': 'branch_proxy + binned_compactness (PROXY)',
            'n_unique_linkage_patterns': len(set(linkages)),
            'linkage_overlap': len(train_linkages & test_linkages),
            'limitation': 'Proxy-based - no explicit linkage annotations available',
            'description': 'Linkage pattern split using structural proxies'
        }

        return SplitManifest(
            split_type='linkage_holdout',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            group_key='linkage_pattern',
            n_train_groups=len(train_linkages),
            n_test_groups=len(test_linkages),
            seed=self.seed,
            metadata=metadata
        )

    # =========================================================================
    # TRUE STRUCTURE-AWARE SPLITS (v2.1)
    # =========================================================================

    def scaffold_holdout_split_true(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        n_folds: int = 5,
        test_fold: int = 0,
        glytoucan_path: Path = Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json"),
        save_group_stats: bool = False,
        group_stats_output_path: Optional[Path] = None,
        group_top10_csv_path: Optional[Path] = None
    ) -> SplitManifest:
        """
        TRUE scaffold holdout split using canonical WURCS/IUPAC-based keys.

        This addresses the limitation of proxy-based scaffold splits by using
        actual glycan structural representations (WURCS sequences) to define
        scaffold groups.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            n_folds: Number of cross-validation folds
            test_fold: Which fold to use as test
            glytoucan_path: Path to GlyTouCan data with sequences
            save_group_stats: If True, save scaffold group distribution statistics
            group_stats_output_path: Path to save group stats JSON
            group_top10_csv_path: Path to save top-10 groups CSV

        Returns:
            SplitManifest object
        """
        if not STRUCTURE_KEYS_AVAILABLE:
            raise ImportError("structure_keys module required for TRUE splits")

        # Load structure keys
        df_with_keys, generator = load_structure_keys_for_dataset(
            df_features, glytoucan_path, return_generator=True
        )

        glycan_ids = df_with_keys['glytoucan_id'].values
        scaffold_keys = df_with_keys['scaffold_key_true'].values

        # Analyze scaffold group distribution
        if save_group_stats:
            group_dist_stats = generator.analyze_scaffold_group_distribution(
                scaffold_keys.tolist(),
                glycan_ids.tolist()
            )

            # Save statistics JSON
            if group_stats_output_path:
                import json
                with open(group_stats_output_path, 'w') as f:
                    json.dump(group_dist_stats, f, indent=2)
                print(f"[ScaffoldDist] Group distribution stats saved to {group_stats_output_path}")

            # Save top-10 CSV
            if group_top10_csv_path and 'top_10_groups' in group_dist_stats:
                top_10_df = pd.DataFrame(group_dist_stats['top_10_groups'])
                top_10_df.to_csv(group_top10_csv_path, index=False)
                print(f"[ScaffoldDist] Top-10 groups saved to {group_top10_csv_path}")

        # Group K-fold split
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(glycan_ids, groups=scaffold_keys))

        if test_fold >= len(splits):
            test_fold = 0

        train_idx, test_idx = splits[test_fold]

        train_glycans = glycan_ids[train_idx].tolist()
        test_glycans = glycan_ids[test_idx].tolist()

        train_scaffolds = set(scaffold_keys[train_idx])
        test_scaffolds = set(scaffold_keys[test_idx])

        all_agents = df_labels['agent_id'].unique().tolist()

        # Analyze key distribution
        key_stats = generator.analyze_key_distribution(glycan_ids.tolist())

        metadata = {
            'n_folds': n_folds,
            'test_fold': test_fold,
            'scaffold_definition': 'WURCS/IUPAC canonical topology (TRUE)',
            'n_unique_scaffolds': len(set(scaffold_keys)),
            'scaffold_overlap': len(train_scaffolds & test_scaffolds),
            'key_stats': key_stats['scaffold'],
            'method': 'WURCS-based canonical hash',
            'description': 'TRUE scaffold split using WURCS sequences'
        }

        return SplitManifest(
            split_type='scaffold_holdout_true',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            group_key='scaffold_key_true',
            n_train_groups=len(train_scaffolds),
            n_test_groups=len(test_scaffolds),
            seed=self.seed,
            metadata=metadata
        )

    def terminal_motif_holdout_split_true(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        n_folds: int = 4,
        test_fold: int = 0,
        glytoucan_path: Path = Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json")
    ) -> SplitManifest:
        """
        TRUE terminal motif holdout split using WURCS/IUPAC terminal extraction.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            n_folds: Number of cross-validation folds
            test_fold: Which fold to use as test
            glytoucan_path: Path to GlyTouCan data

        Returns:
            SplitManifest object
        """
        if not STRUCTURE_KEYS_AVAILABLE:
            raise ImportError("structure_keys module required for TRUE splits")

        df_with_keys = load_structure_keys_for_dataset(df_features, glytoucan_path)

        glycan_ids = df_with_keys['glytoucan_id'].values
        terminal_keys = df_with_keys['terminal_motif_key_true'].values

        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(glycan_ids, groups=terminal_keys))

        if test_fold >= len(splits):
            test_fold = 0

        train_idx, test_idx = splits[test_fold]

        train_glycans = glycan_ids[train_idx].tolist()
        test_glycans = glycan_ids[test_idx].tolist()

        train_terminals = set(terminal_keys[train_idx])
        test_terminals = set(terminal_keys[test_idx])

        all_agents = df_labels['agent_id'].unique().tolist()

        metadata = {
            'n_folds': n_folds,
            'test_fold': test_fold,
            'terminal_definition': 'WURCS/IUPAC terminal residue extraction (TRUE)',
            'n_unique_terminals': len(set(terminal_keys)),
            'terminal_overlap': len(train_terminals & test_terminals),
            'method': 'WURCS terminal residue analysis',
            'description': 'TRUE terminal motif split from WURCS sequences'
        }

        return SplitManifest(
            split_type='terminal_motif_holdout_true',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            group_key='terminal_motif_key_true',
            n_train_groups=len(train_terminals),
            n_test_groups=len(test_terminals),
            seed=self.seed,
            metadata=metadata
        )

    def linkage_holdout_split_true(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        n_folds: int = 4,
        test_fold: int = 0,
        glytoucan_path: Path = Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json"),
        save_coverage_stats: bool = False,
        coverage_output_path: Optional[Path] = None
    ) -> SplitManifest:
        """
        TRUE linkage holdout split using WURCS/IUPAC linkage extraction.

        Args:
            df_features: DataFrame with glycan features
            df_labels: DataFrame with binding labels
            n_folds: Number of cross-validation folds
            test_fold: Which fold to use as test
            glytoucan_path: Path to GlyTouCan data
            save_coverage_stats: If True, save linkage coverage statistics
            coverage_output_path: Path to save coverage stats JSON

        Returns:
            SplitManifest object
        """
        if not STRUCTURE_KEYS_AVAILABLE:
            raise ImportError("structure_keys module required for TRUE splits")

        # Load structure keys and get generator for coverage stats
        df_with_keys, generator = load_structure_keys_for_dataset(
            df_features, glytoucan_path, return_generator=True
        )

        # Save linkage coverage statistics if requested
        if save_coverage_stats and coverage_output_path:
            coverage_stats = generator.get_linkage_key_coverage_stats()
            import json
            with open(coverage_output_path, 'w') as f:
                json.dump(coverage_stats, f, indent=2)
            print(f"[Coverage] Linkage key coverage stats saved to {coverage_output_path}")

        glycan_ids = df_with_keys['glytoucan_id'].values
        linkage_keys = df_with_keys['linkage_key_true'].values

        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(glycan_ids, groups=linkage_keys))

        if test_fold >= len(splits):
            test_fold = 0

        train_idx, test_idx = splits[test_fold]

        train_glycans = glycan_ids[train_idx].tolist()
        test_glycans = glycan_ids[test_idx].tolist()

        train_linkages = set(linkage_keys[train_idx])
        test_linkages = set(linkage_keys[test_idx])

        all_agents = df_labels['agent_id'].unique().tolist()

        metadata = {
            'n_folds': n_folds,
            'test_fold': test_fold,
            'linkage_definition': 'WURCS/IUPAC glycosidic linkage extraction (TRUE)',
            'n_unique_linkages': len(set(linkage_keys)),
            'linkage_overlap': len(train_linkages & test_linkages),
            'method': 'WURCS linkage pattern analysis',
            'description': 'TRUE linkage split from WURCS sequences'
        }

        return SplitManifest(
            split_type='linkage_holdout_true',
            train_glycans=train_glycans,
            test_glycans=test_glycans,
            train_agents=all_agents,
            test_agents=all_agents,
            group_key='linkage_key_true',
            n_train_groups=len(train_linkages),
            n_test_groups=len(test_linkages),
            seed=self.seed,
            metadata=metadata
        )


def compute_split_statistics(
    manifest: SplitManifest,
    df_labels: pd.DataFrame
) -> Dict:
    """
    Compute detailed statistics for a split.

    Args:
        manifest: Split manifest
        df_labels: DataFrame with binding labels

    Returns:
        Dictionary of statistics
    """
    train_set = set(manifest.train_glycans)
    test_set = set(manifest.test_glycans)

    # Filter labels
    train_labels = df_labels[df_labels['glytoucan_id'].isin(train_set)]
    test_labels = df_labels[df_labels['glytoucan_id'].isin(test_set)]

    stats = {
        'n_train_glycans': len(manifest.train_glycans),
        'n_test_glycans': len(manifest.test_glycans),
        'n_train_pairs': len(train_labels),
        'n_test_pairs': len(test_labels),
        'n_train_agents': train_labels['agent_id'].nunique(),
        'n_test_agents': test_labels['agent_id'].nunique(),
        'train_pairs_per_glycan': len(train_labels) / len(manifest.train_glycans) if manifest.train_glycans else 0,
        'test_pairs_per_glycan': len(test_labels) / len(manifest.test_glycans) if manifest.test_glycans else 0,
    }

    # Validate
    is_valid, issues = manifest.validate()
    stats['is_valid'] = is_valid
    stats['validation_issues'] = issues

    return stats
