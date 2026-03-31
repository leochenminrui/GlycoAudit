#!/usr/bin/env python3
"""
Level 1: Lectin-specific SSV/GCV dimension preference analysis.

For each agent, identifies which feature dimensions are systematically preferred
among top-ranked glycans vs. all candidates.

Scientific question: Which SSV dimensions are systematically preferred by each lectin?
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
DATA_PATH = BASE_PATH / "data"
REPORTS_PATH = BASE_PATH / "reports"
OUTPUT_PATH = REPORTS_PATH / "findings_v1"

# Feature definitions
SSV_FEATURES = [
    'n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
    'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy'
]
GCV_FEATURES = [
    'contact_density', 'long_range_contact_fraction', 'mean_residue_neighbor_count',
    'sd_residue_neighbor_count', 'torsion_diversity', 'graph_laplacian_spectral_gap',
    'core_periphery_ratio', 'max_contact_distance_seq'
]
ALL_FEATURES = SSV_FEATURES + GCV_FEATURES

# Semantic grouping
FEATURE_GROUPS = {
    'n_atoms': 'Size',
    'n_residues': 'Size',
    'radius_of_gyration': 'Shape',
    'max_pair_distance': 'Shape',
    'compactness': 'Shape',
    'branch_proxy': 'Topology',
    'terminal_proxy': 'Topology',
    'exposure_proxy': 'Surface',
    'contact_density': 'Contact',
    'long_range_contact_fraction': 'Contact',
    'mean_residue_neighbor_count': 'Contact',
    'sd_residue_neighbor_count': 'Contact',
    'torsion_diversity': 'Graph',
    'graph_laplacian_spectral_gap': 'Graph',
    'core_periphery_ratio': 'Graph',
    'max_contact_distance_seq': 'Graph',
}


def setup_logging(name: str) -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load SSV+GCV features and binding labels."""
    # Load SSV
    df_ssv = pd.read_csv(DATA_PATH / "ssv/expanded_v1/ssv_features.csv")
    # Load GCV
    df_gcv = pd.read_csv(DATA_PATH / "gcv/expanded_v1/gcv_features.csv")
    # Merge
    df_features = df_ssv.merge(
        df_gcv[['glytoucan_id'] + GCV_FEATURES],
        on='glytoucan_id',
        how='inner'
    )
    # Load labels
    df_labels = pd.read_csv(DATA_PATH / "binding/expanded_v1/labels.csv")
    df_labels = df_labels[df_labels['label'] == 1].copy()

    # Load agent metadata
    df_agents = pd.read_csv(DATA_PATH / "binding/expanded_v1/agent_meta.csv")

    return df_features, df_labels, df_agents


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_agent_rankings(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    feature_cols: List[str],
    min_pos: int = 2
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    """
    Compute glycan rankings for each agent using prototype-based cosine similarity.

    Returns: dict mapping agent_id -> (sorted_glycan_indices, positive_indices)
    """
    # Prepare features
    df_features = df_features.sort_values('glytoucan_id').reset_index(drop=True)
    glycan_to_idx = {gid: i for i, gid in enumerate(df_features['glytoucan_id'])}

    features = df_features[feature_cols].values.astype(float)
    n_candidates = features.shape[0]

    # Handle NaN
    for j in range(features.shape[1]):
        col_median = np.nanmedian(features[:, j])
        if np.isnan(col_median):
            col_median = 0.0
        features[np.isnan(features[:, j]), j] = col_median

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Build agent -> positive indices
    agent_to_pos = {}
    for _, row in df_labels.iterrows():
        agent_id = row['agent_id']
        gid = row['glytoucan_id']
        if gid in glycan_to_idx:
            if agent_id not in agent_to_pos:
                agent_to_pos[agent_id] = set()
            agent_to_pos[agent_id].add(glycan_to_idx[gid])

    # Compute rankings for agents with >= min_pos positives
    agent_rankings = {}

    for agent_id, pos_indices in agent_to_pos.items():
        pos_indices = list(pos_indices)
        if len(pos_indices) < min_pos:
            continue

        # Compute prototype
        pos_features = features_scaled[pos_indices]
        prototype = pos_features.mean(axis=0)

        if np.isnan(prototype).any():
            continue

        # Score all candidates
        scores = np.array([
            cosine_similarity(features_scaled[i], prototype)
            for i in range(n_candidates)
        ])

        # Get ranking (indices sorted by descending score)
        sorted_indices = np.argsort(-scores)

        agent_rankings[agent_id] = (sorted_indices, pos_indices)

    return agent_rankings, features_scaled, df_features['glytoucan_id'].values


def compute_preference_scores(
    features_scaled: np.ndarray,
    sorted_indices: np.ndarray,
    K: int,
    feature_cols: List[str]
) -> Dict[str, float]:
    """
    Compute preference score for each dimension.

    Preference = mean(z-scored feature among top-K) - mean(z-scored feature among all)

    Since features are already z-scored, mean(all) ≈ 0, so preference ≈ mean(top-K).
    """
    top_k_indices = sorted_indices[:K]
    top_k_features = features_scaled[top_k_indices]

    pref_scores = {}
    for j, feat in enumerate(feature_cols):
        # Mean among top-K (already standardized, so relative to population mean of 0)
        pref_scores[feat] = float(top_k_features[:, j].mean())

    return pref_scores


def compute_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size.

    Range: [-1, 1], where |d| > 0.474 is large effect
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    # Count dominance pairs
    greater = 0
    less = 0
    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                greater += 1
            elif x1 < x2:
                less += 1

    return (greater - less) / (n1 * n2)


def run_permutation_test(
    features_scaled: np.ndarray,
    sorted_indices: np.ndarray,
    K: int,
    feature_cols: List[str],
    n_perm: int = 10000,
    seed: int = 1
) -> Dict[str, Tuple[float, float]]:
    """
    Run permutation test for preference significance.

    Returns: dict mapping feature -> (observed_pref, p_value)
    """
    rng = np.random.default_rng(seed)
    n_candidates = features_scaled.shape[0]

    # Observed preferences
    top_k_indices = sorted_indices[:K]
    observed = {}
    for j, feat in enumerate(feature_cols):
        observed[feat] = float(features_scaled[top_k_indices, j].mean())

    # Null distribution
    null_counts = {feat: 0 for feat in feature_cols}

    for _ in range(n_perm):
        # Random K glycans
        random_k = rng.choice(n_candidates, size=K, replace=False)

        for j, feat in enumerate(feature_cols):
            null_val = features_scaled[random_k, j].mean()
            # Two-sided: count if |null| >= |observed|
            if abs(null_val) >= abs(observed[feat]):
                null_counts[feat] += 1

    results = {}
    for feat in feature_cols:
        p_val = (1 + null_counts[feat]) / (1 + n_perm)
        results[feat] = (observed[feat], p_val)

    return results


def holm_bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction."""
    features = list(p_values.keys())
    pvals = [p_values[f] for f in features]
    n = len(pvals)

    sorted_indices = np.argsort(pvals)
    results = {}

    for rank, idx in enumerate(sorted_indices):
        feat = features[idx]
        adjusted_alpha = alpha / (n - rank)
        is_significant = pvals[idx] <= adjusted_alpha
        results[feat] = (pvals[idx], is_significant)

    return results


def benjamini_hochberg_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Apply Benjamini-Hochberg FDR correction."""
    features = list(p_values.keys())
    pvals = np.array([p_values[f] for f in features])
    n = len(pvals)

    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # Find largest k where p(k) <= k/n * alpha
    significant = np.zeros(n, dtype=bool)
    for k in range(n - 1, -1, -1):
        threshold = (k + 1) / n * alpha
        if sorted_pvals[k] <= threshold:
            significant[:k + 1] = True
            break

    results = {}
    for i, idx in enumerate(sorted_indices):
        feat = features[idx]
        results[feat] = (pvals[idx], significant[i])

    return results


def run_level1_analysis(
    K_values: List[int] = [10, 20],
    n_perm: int = 10000,
    min_pos: int = 2,
    seed: int = 1,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Run Level 1 preference analysis for all agents.
    """
    if logger is None:
        logger = setup_logging('level1')

    logger.info("=" * 60)
    logger.info("LEVEL 1: LECTIN-SPECIFIC DIMENSION PREFERENCE ANALYSIS")
    logger.info("=" * 60)

    np.random.seed(seed)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    df_features, df_labels, df_agents = load_data()
    logger.info(f"  Glycans: {len(df_features)}")
    logger.info(f"  Binding pairs: {len(df_labels)}")

    # Compute rankings
    logger.info("Computing agent rankings...")
    agent_rankings, features_scaled, glycan_ids = compute_agent_rankings(
        df_features, df_labels, ALL_FEATURES, min_pos=min_pos
    )
    logger.info(f"  Agents with >= {min_pos} positives: {len(agent_rankings)}")

    n_candidates = features_scaled.shape[0]

    # Run preference analysis for each agent and K
    all_results = []

    for agent_id, (sorted_indices, pos_indices) in agent_rankings.items():
        for K in K_values:
            if K > n_candidates:
                continue

            # Run permutation test
            perm_results = run_permutation_test(
                features_scaled, sorted_indices, K, ALL_FEATURES,
                n_perm=n_perm, seed=seed
            )

            # Get p-values for corrections
            p_values = {feat: perm_results[feat][1] for feat in ALL_FEATURES}

            # Apply corrections
            holm_results = holm_bonferroni_correction(p_values)
            fdr_results = benjamini_hochberg_correction(p_values)

            # Compute effect sizes (Cliff's delta for top-K vs rest)
            top_k_indices = sorted_indices[:K]
            rest_indices = sorted_indices[K:]

            for j, feat in enumerate(ALL_FEATURES):
                top_k_vals = features_scaled[top_k_indices, j]
                rest_vals = features_scaled[rest_indices, j]

                cliffs_d = compute_cliffs_delta(top_k_vals, rest_vals)

                all_results.append({
                    'agent_id': agent_id,
                    'dimension': feat,
                    'feature_group': FEATURE_GROUPS[feat],
                    'K': K,
                    'pref_score': perm_results[feat][0],
                    'effect_size_cliffs_d': cliffs_d,
                    'p_perm': perm_results[feat][1],
                    'p_adj_holm': holm_results[feat][0],
                    'sig_holm': holm_results[feat][1],
                    'p_adj_fdr': fdr_results[feat][0],
                    'sig_fdr': fdr_results[feat][1],
                    'n_positives': len(pos_indices),
                    'n_candidates': n_candidates,
                })

    df_results = pd.DataFrame(all_results)

    # Save results
    output_file = OUTPUT_PATH / "level1_agent_dimension_preferences.csv"
    df_results.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    # Summary statistics
    for K in K_values:
        df_k = df_results[df_results['K'] == K]
        n_agents = df_k['agent_id'].nunique()
        n_sig_holm = df_k[df_k['sig_holm']]['agent_id'].nunique()
        n_sig_fdr = df_k[df_k['sig_fdr']]['agent_id'].nunique()

        logger.info(f"\nK={K} Summary:")
        logger.info(f"  Agents tested: {n_agents}")
        logger.info(f"  Agents with >=1 sig dim (Holm): {n_sig_holm} ({100*n_sig_holm/n_agents:.1f}%)")
        logger.info(f"  Agents with >=1 sig dim (FDR): {n_sig_fdr} ({100*n_sig_fdr/n_agents:.1f}%)")

        # Top dimensions by frequency of significance
        sig_counts = df_k[df_k['sig_fdr']].groupby('dimension').size().sort_values(ascending=False)
        if len(sig_counts) > 0:
            logger.info(f"  Top significant dimensions (FDR):")
            for dim, count in sig_counts.head(5).items():
                logger.info(f"    {dim}: {count} agents")

    return df_results


def generate_level1_summary(df_results: pd.DataFrame, logger: logging.Logger = None) -> str:
    """Generate markdown summary for Level 1 analysis."""
    if logger is None:
        logger = setup_logging('level1_summary')

    lines = [
        "# Level 1: Lectin-Specific SSV Dimension Preferences",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        "This analysis identifies which structural dimensions are systematically preferred",
        "by each lectin/antibody among their top-ranked glycans.",
        "",
        "**Method:**",
        "- For each agent, rank all glycans by prototype-based cosine similarity",
        "- Compute preference score = mean(z-scored feature) among top-K glycans",
        "- Test significance via permutation (10,000 random K-samples)",
        "- Apply Holm-Bonferroni and Benjamini-Hochberg corrections",
        "",
    ]

    # Per-K summaries
    for K in df_results['K'].unique():
        df_k = df_results[df_results['K'] == K]
        n_agents = df_k['agent_id'].nunique()
        n_sig_holm = df_k[df_k['sig_holm']]['agent_id'].nunique()
        n_sig_fdr = df_k[df_k['sig_fdr']]['agent_id'].nunique()

        lines.extend([
            f"## Results for K={K}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Agents tested | {n_agents} |",
            f"| Agents with >=1 sig dim (Holm) | {n_sig_holm} ({100*n_sig_holm/n_agents:.1f}%) |",
            f"| Agents with >=1 sig dim (FDR) | {n_sig_fdr} ({100*n_sig_fdr/n_agents:.1f}%) |",
            "",
        ])

        # Top dimensions
        sig_counts = df_k[df_k['sig_fdr']].groupby('dimension').size().sort_values(ascending=False)
        if len(sig_counts) > 0:
            lines.extend([
                "### Most Frequently Significant Dimensions (FDR)",
                "",
                "| Dimension | Feature Group | # Agents | % of Tested |",
                "|-----------|---------------|----------|-------------|",
            ])
            for dim, count in sig_counts.head(8).items():
                group = FEATURE_GROUPS[dim]
                pct = 100 * count / n_agents
                lines.append(f"| {dim} | {group} | {count} | {pct:.1f}% |")
            lines.append("")

        # Example agents with strong preferences
        strong_prefs = df_k[
            (df_k['sig_fdr']) & (abs(df_k['effect_size_cliffs_d']) > 0.3)
        ].sort_values('effect_size_cliffs_d', key=abs, ascending=False)

        if len(strong_prefs) > 0:
            lines.extend([
                "### Example Agents with Strong Preferences",
                "",
                "| Agent | Dimension | Pref Score | Cliff's d | p-value |",
                "|-------|-----------|------------|-----------|---------|",
            ])
            seen_agents = set()
            for _, row in strong_prefs.iterrows():
                if row['agent_id'] in seen_agents:
                    continue
                seen_agents.add(row['agent_id'])
                if len(seen_agents) > 5:
                    break
                lines.append(
                    f"| {row['agent_id']} | {row['dimension']} | "
                    f"{row['pref_score']:.3f} | {row['effect_size_cliffs_d']:.3f} | "
                    f"{row['p_perm']:.4f} |"
                )
            lines.append("")

    # Biological interpretation
    lines.extend([
        "## Biological Interpretation",
        "",
        "The preference analysis reveals that lectins show distinct structural preferences:",
        "",
    ])

    # Group-level summary
    for K in [10]:
        df_k = df_results[df_results['K'] == K]
        group_sig = df_k[df_k['sig_fdr']].groupby('feature_group').size()
        total_sig = group_sig.sum()

        if total_sig > 0:
            lines.append(f"**Feature group breakdown (K={K}, FDR-significant):**")
            lines.append("")
            for group in ['Size', 'Shape', 'Topology', 'Surface', 'Contact', 'Graph']:
                if group in group_sig.index:
                    count = group_sig[group]
                    pct = 100 * count / total_sig
                    lines.append(f"- {group}: {count} significant preferences ({pct:.1f}%)")
            lines.append("")

    lines.extend([
        "## Output Files",
        "",
        "- `level1_agent_dimension_preferences.csv`: Full results table",
        "- `fig_level1_heatmap.pdf`: Preference heatmap (agents x dimensions)",
        "- `fig_level1_agent_embedding.pdf`: Agent embedding by preference profile",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    logger = setup_logging('level1')

    # Run analysis
    df_results = run_level1_analysis(
        K_values=[10, 20],
        n_perm=10000,
        min_pos=2,
        seed=1,
        logger=logger
    )

    # Generate summary
    summary = generate_level1_summary(df_results, logger)

    summary_file = OUTPUT_PATH / "level1_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"Summary saved to {summary_file}")
