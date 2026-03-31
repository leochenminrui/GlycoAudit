#!/usr/bin/env python3
"""
E5_v2: Scaffold Split Null Baselines and Statistical Significance

This script extends E5 scaffold split analysis with:
1. Random baseline null: 1000 repeats of random scoring within each fold
2. Permutation baseline null: 1000 repeats of shuffled glycan-feature mapping
3. Empirical one-sided p-values per metric
4. Holm-Bonferroni correction across metrics

Output: benchmark_release_v1/E5_split_robustness_v2/
"""

from __future__ import annotations

import json
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
BENCHMARK_PATH = BASE_PATH / "benchmark_release_v1"
DATA_PATH = BASE_PATH / "data"
LOGS_PATH = BENCHMARK_PATH / "logs"

# Input paths
SSV_TABLE_PATH = DATA_PATH / "ssv/targeted_sugarbind_v0/ssv_table.csv"
LABELS_PATH = DATA_PATH / "binding/sugarbind_v0/labels.csv"
E5_EXISTING_PATH = BENCHMARK_PATH / "E5_split_robustness"

# Output path
OUT_DIR = BENCHMARK_PATH / "E5_split_robustness_v2"

# Feature columns
FEATURE_COLS = [
    'n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
    'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy'
]

SEED = 42
N_RANDOM_REPEATS = 1000
N_PERM_REPEATS = 1000
N_FOLDS = 5


def setup_logging() -> logging.Logger:
    """Setup logging."""
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_PATH / f"e5_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger('e5_v2')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_ranking_metrics(ranks: List[int], n_candidates: int, ks: List[int]) -> Dict[str, float]:
    """Compute ranking metrics for a single agent."""
    if not ranks:
        return {}

    ranks = np.array(ranks)
    n_pos = len(ranks)

    metrics = {
        'mrr': 1.0 / ranks.min(),
        'mean_rank': float(ranks.mean()),
    }

    for k in ks:
        metrics[f'recall@{k}'] = float((ranks <= k).sum() / n_pos)

    return metrics


def compute_auprc_lb(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    """Compute AUPRC lower bound treating unlabeled as negatives."""
    if positive_mask.sum() == 0 or positive_mask.sum() == len(positive_mask):
        return np.nan
    return average_precision_score(positive_mask.astype(int), scores)


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction to p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = [p_values[i] for i in sorted_indices]

    results = [None] * n
    for rank, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (n - rank)
        is_significant = sorted_p[rank] <= adjusted_alpha
        results[idx] = (p_values[idx], is_significant)

    return results


def evaluate_fold_with_features(
    features_scaled: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    agent_to_pos_indices: Dict,
    df_ssv: pd.DataFrame,
    ks: List[int]
) -> Tuple[List[Dict], int]:
    """Evaluate agents in a fold with given features."""
    train_glycans = set(df_ssv.iloc[train_idx]['glytoucan_id'])
    test_glycans = set(df_ssv.iloc[test_idx]['glytoucan_id'])

    agent_metrics = []

    for agent_id, pos_indices in agent_to_pos_indices.items():
        # Split positives into train/test
        train_pos = [idx for idx in pos_indices
                    if df_ssv.iloc[idx]['glytoucan_id'] in train_glycans]
        test_pos = [idx for idx in pos_indices
                   if df_ssv.iloc[idx]['glytoucan_id'] in test_glycans]

        if len(train_pos) < 1 or len(test_pos) < 1:
            continue

        # Compute prototype from train positives
        train_features = features_scaled[train_pos]
        prototype = train_features.mean(axis=0)

        if np.isnan(prototype).any():
            continue

        # Score test glycans only
        test_scores = np.array([cosine_similarity(features_scaled[i], prototype)
                               for i in test_idx])

        # Rank within test set
        sorted_indices = np.argsort(-test_scores)
        rank_of_test_idx = np.empty(len(test_idx), dtype=int)
        rank_of_test_idx[sorted_indices] = np.arange(1, len(test_idx) + 1)

        # Get ranks of test positives
        test_pos_in_test = [list(test_idx).index(idx) for idx in test_pos if idx in test_idx]
        if not test_pos_in_test:
            continue

        pos_ranks = [rank_of_test_idx[i] for i in test_pos_in_test]

        # Compute metrics
        metrics = compute_ranking_metrics(pos_ranks, len(test_idx), ks)

        # AUPRC-LB
        positive_mask = np.zeros(len(test_idx), dtype=bool)
        for idx in test_pos:
            if idx in test_idx:
                test_pos_idx = list(test_idx).index(idx)
                positive_mask[test_pos_idx] = True
        metrics['auprc_lb'] = compute_auprc_lb(test_scores, positive_mask)

        agent_metrics.append(metrics)

    return agent_metrics, len(agent_metrics)


def main():
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("E5_v2: SCAFFOLD SPLIT NULL BASELINES AND SIGNIFICANCE")
    logger.info("=" * 70)

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load data
    if not SSV_TABLE_PATH.exists():
        logger.error(f"SSV table not found: {SSV_TABLE_PATH}")
        sys.exit(1)

    df_ssv = pd.read_csv(SSV_TABLE_PATH)
    df_labels = pd.read_csv(LABELS_PATH)
    df_labels = df_labels[df_labels['label'] == 1].copy()

    logger.info(f"SSV glycans: {len(df_ssv)}")
    logger.info(f"Positive labels: {len(df_labels)}")

    # Intersect
    ssv_glycans = set(df_ssv['glytoucan_id'])
    df_labels_filtered = df_labels[df_labels['glytoucan_id'].isin(ssv_glycans)].copy()

    df_ssv = df_ssv.sort_values('glytoucan_id').reset_index(drop=True)
    glycan_to_idx = {gid: i for i, gid in enumerate(df_ssv['glytoucan_id'])}
    n_candidates = len(df_ssv)

    # Build agent -> positive indices
    agent_to_pos_indices = {}
    for _, row in df_labels_filtered.iterrows():
        agent_id = row['agent_id']
        gid = row['glytoucan_id']
        if gid in glycan_to_idx:
            if agent_id not in agent_to_pos_indices:
                agent_to_pos_indices[agent_id] = []
            agent_to_pos_indices[agent_id].append(glycan_to_idx[gid])

    logger.info(f"Agents with positives: {len(agent_to_pos_indices)}")

    # Prepare features
    features = df_ssv[FEATURE_COLS].values.astype(float)

    for j in range(features.shape[1]):
        col_median = np.nanmedian(features[:, j])
        features[np.isnan(features[:, j]), j] = col_median

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create scaffold groups
    n_res_terciles = df_ssv['n_residues'].quantile([0.33, 0.67]).values

    def get_scaffold(row):
        n_res = row['n_residues']
        if n_res <= n_res_terciles[0]:
            size_bin = 'small'
        elif n_res <= n_res_terciles[1]:
            size_bin = 'medium'
        else:
            size_bin = 'large'
        return f"{size_bin}_{int(row['branch_proxy'])}_{int(row['terminal_proxy'])}"

    df_ssv['scaffold'] = df_ssv.apply(get_scaffold, axis=1)
    scaffolds = df_ssv['scaffold'].values

    ks = [1, 3, 5, 10, 20]

    # GroupKFold by scaffold
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Storage for results
    observed_by_fold = []
    random_null_by_fold = []
    perm_null_by_fold = []

    logger.info(f"Running {N_FOLDS}-fold scaffold split with {N_RANDOM_REPEATS} random + {N_PERM_REPEATS} perm repeats...")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np.arange(n_candidates), groups=scaffolds)):
        logger.info(f"\nFold {fold_idx + 1}/{N_FOLDS}: {len(train_idx)} train, {len(test_idx)} test glycans")

        # ===== OBSERVED METRICS =====
        agent_metrics, n_agents = evaluate_fold_with_features(
            features_scaled, train_idx, test_idx, agent_to_pos_indices, df_ssv, ks
        )

        if agent_metrics:
            df_agents = pd.DataFrame(agent_metrics)
            observed_by_fold.append({
                'fold': fold_idx,
                'n_train_glycans': len(train_idx),
                'n_test_glycans': len(test_idx),
                'n_agents_evaluated': n_agents,
                'mrr': df_agents['mrr'].mean(),
                'recall@5': df_agents['recall@5'].mean(),
                'auprc_lb': df_agents['auprc_lb'].mean(),
                'mean_rank': df_agents['mean_rank'].mean(),
            })
            logger.info(f"  Observed: MRR={df_agents['mrr'].mean():.4f}, Recall@5={df_agents['recall@5'].mean():.4f}")

        # ===== RANDOM BASELINE =====
        logger.info(f"  Running {N_RANDOM_REPEATS} random null repeats...")
        for rep in range(N_RANDOM_REPEATS):
            if rep > 0 and rep % 250 == 0:
                logger.info(f"    Random: {rep}/{N_RANDOM_REPEATS}")

            # Random scores for test set
            rep_agent_metrics = []

            train_glycans = set(df_ssv.iloc[train_idx]['glytoucan_id'])
            test_glycans = set(df_ssv.iloc[test_idx]['glytoucan_id'])

            for agent_id, pos_indices in agent_to_pos_indices.items():
                train_pos = [idx for idx in pos_indices
                            if df_ssv.iloc[idx]['glytoucan_id'] in train_glycans]
                test_pos = [idx for idx in pos_indices
                           if df_ssv.iloc[idx]['glytoucan_id'] in test_glycans]

                if len(train_pos) < 1 or len(test_pos) < 1:
                    continue

                # Random scores
                test_scores = rng.random(len(test_idx))

                sorted_indices = np.argsort(-test_scores)
                rank_of_test_idx = np.empty(len(test_idx), dtype=int)
                rank_of_test_idx[sorted_indices] = np.arange(1, len(test_idx) + 1)

                test_pos_in_test = [list(test_idx).index(idx) for idx in test_pos if idx in test_idx]
                if not test_pos_in_test:
                    continue

                pos_ranks = [rank_of_test_idx[i] for i in test_pos_in_test]
                metrics = compute_ranking_metrics(pos_ranks, len(test_idx), ks)

                positive_mask = np.zeros(len(test_idx), dtype=bool)
                for idx in test_pos:
                    if idx in test_idx:
                        test_pos_idx = list(test_idx).index(idx)
                        positive_mask[test_pos_idx] = True
                metrics['auprc_lb'] = compute_auprc_lb(test_scores, positive_mask)

                rep_agent_metrics.append(metrics)

            if rep_agent_metrics:
                df_rep = pd.DataFrame(rep_agent_metrics)
                random_null_by_fold.append({
                    'fold': fold_idx,
                    'repeat': rep,
                    'mode': 'random',
                    'mrr': df_rep['mrr'].mean(),
                    'recall@5': df_rep['recall@5'].mean(),
                    'auprc_lb': df_rep['auprc_lb'].mean(),
                    'mean_rank': df_rep['mean_rank'].mean(),
                })

        # ===== PERMUTATION BASELINE =====
        logger.info(f"  Running {N_PERM_REPEATS} permutation null repeats...")
        for rep in range(N_PERM_REPEATS):
            if rep > 0 and rep % 250 == 0:
                logger.info(f"    Permutation: {rep}/{N_PERM_REPEATS}")

            # Permute features globally
            perm_indices = rng.permutation(n_candidates)
            features_perm = features_scaled[perm_indices]

            agent_metrics_perm, _ = evaluate_fold_with_features(
                features_perm, train_idx, test_idx, agent_to_pos_indices, df_ssv, ks
            )

            if agent_metrics_perm:
                df_rep = pd.DataFrame(agent_metrics_perm)
                perm_null_by_fold.append({
                    'fold': fold_idx,
                    'repeat': rep,
                    'mode': 'permutation',
                    'mrr': df_rep['mrr'].mean(),
                    'recall@5': df_rep['recall@5'].mean(),
                    'auprc_lb': df_rep['auprc_lb'].mean(),
                    'mean_rank': df_rep['mean_rank'].mean(),
                })

    # Convert to DataFrames
    df_observed = pd.DataFrame(observed_by_fold)
    df_random = pd.DataFrame(random_null_by_fold)
    df_perm = pd.DataFrame(perm_null_by_fold)

    # Save per-fold observed metrics
    df_observed.to_csv(OUT_DIR / "scaffold_split_observed_by_fold.csv", index=False)

    # Save null distributions by fold
    df_null_combined = pd.concat([df_random, df_perm], ignore_index=True)
    df_null_combined.to_csv(OUT_DIR / "scaffold_split_null_by_fold.csv", index=False)

    # Compute aggregate statistics across folds
    logger.info("\n" + "=" * 50)
    logger.info("AGGREGATE STATISTICS")
    logger.info("=" * 50)

    # Observed aggregates
    obs_mrr_mean = df_observed['mrr'].mean()
    obs_mrr_std = df_observed['mrr'].std()
    obs_recall5_mean = df_observed['recall@5'].mean()
    obs_recall5_std = df_observed['recall@5'].std()
    obs_auprc_mean = df_observed['auprc_lb'].mean()
    obs_auprc_std = df_observed['auprc_lb'].std()

    logger.info(f"Observed MRR: {obs_mrr_mean:.4f} +/- {obs_mrr_std:.4f}")
    logger.info(f"Observed Recall@5: {obs_recall5_mean:.4f} +/- {obs_recall5_std:.4f}")
    logger.info(f"Observed AUPRC-LB: {obs_auprc_mean:.4f} +/- {obs_auprc_std:.4f}")

    # Null aggregates (pool across folds)
    random_mrr_mean = df_random['mrr'].mean()
    random_mrr_std = df_random['mrr'].std()
    random_recall5_mean = df_random['recall@5'].mean()
    random_recall5_std = df_random['recall@5'].std()

    perm_mrr_mean = df_perm['mrr'].mean()
    perm_mrr_std = df_perm['mrr'].std()
    perm_recall5_mean = df_perm['recall@5'].mean()
    perm_recall5_std = df_perm['recall@5'].std()

    logger.info(f"Random null MRR: {random_mrr_mean:.4f} +/- {random_mrr_std:.4f}")
    logger.info(f"Random null Recall@5: {random_recall5_mean:.4f} +/- {random_recall5_std:.4f}")
    logger.info(f"Perm null MRR: {perm_mrr_mean:.4f} +/- {perm_mrr_std:.4f}")
    logger.info(f"Perm null Recall@5: {perm_recall5_mean:.4f} +/- {perm_recall5_std:.4f}")

    # Save aggregate summary
    aggregate_rows = [
        {'mode': 'observed_mean', 'mrr': obs_mrr_mean, 'recall@5': obs_recall5_mean,
         'auprc_lb': obs_auprc_mean, 'mean_rank': df_observed['mean_rank'].mean()},
        {'mode': 'observed_std', 'mrr': obs_mrr_std, 'recall@5': obs_recall5_std,
         'auprc_lb': obs_auprc_std, 'mean_rank': df_observed['mean_rank'].std()},
        {'mode': 'random_mean', 'mrr': random_mrr_mean, 'recall@5': random_recall5_mean,
         'auprc_lb': df_random['auprc_lb'].mean(), 'mean_rank': df_random['mean_rank'].mean()},
        {'mode': 'random_std', 'mrr': random_mrr_std, 'recall@5': random_recall5_std,
         'auprc_lb': df_random['auprc_lb'].std(), 'mean_rank': df_random['mean_rank'].std()},
        {'mode': 'permutation_mean', 'mrr': perm_mrr_mean, 'recall@5': perm_recall5_mean,
         'auprc_lb': df_perm['auprc_lb'].mean(), 'mean_rank': df_perm['mean_rank'].mean()},
        {'mode': 'permutation_std', 'mrr': perm_mrr_std, 'recall@5': perm_recall5_std,
         'auprc_lb': df_perm['auprc_lb'].std(), 'mean_rank': df_perm['mean_rank'].std()},
    ]
    df_aggregate = pd.DataFrame(aggregate_rows)
    df_aggregate.to_csv(OUT_DIR / "scaffold_split_aggregate.csv", index=False)

    # Compute p-values
    def compute_pvalue(observed, null_vals, greater_is_better=True):
        n = len(null_vals)
        if greater_is_better:
            n_extreme = (null_vals >= observed).sum()
        else:
            n_extreme = (null_vals <= observed).sum()
        return (1 + n_extreme) / (1 + n)

    pvalues = {}

    # P-values vs random
    pvalues['mrr_vs_random'] = compute_pvalue(obs_mrr_mean, df_random['mrr'].values, True)
    pvalues['recall@5_vs_random'] = compute_pvalue(obs_recall5_mean, df_random['recall@5'].values, True)
    pvalues['auprc_lb_vs_random'] = compute_pvalue(obs_auprc_mean, df_random['auprc_lb'].values, True)

    # P-values vs permutation
    pvalues['mrr_vs_perm'] = compute_pvalue(obs_mrr_mean, df_perm['mrr'].values, True)
    pvalues['recall@5_vs_perm'] = compute_pvalue(obs_recall5_mean, df_perm['recall@5'].values, True)
    pvalues['auprc_lb_vs_perm'] = compute_pvalue(obs_auprc_mean, df_perm['auprc_lb'].values, True)

    logger.info("\nP-values:")
    for k, v in pvalues.items():
        logger.info(f"  {k}: {v:.6f}")

    # Holm-Bonferroni correction
    pval_list_perm = [pvalues['mrr_vs_perm'], pvalues['recall@5_vs_perm'], pvalues['auprc_lb_vs_perm']]
    holm_results = holm_bonferroni_correction(pval_list_perm)

    pvalues['mrr_holm_significant'] = holm_results[0][1]
    pvalues['recall@5_holm_significant'] = holm_results[1][1]
    pvalues['auprc_lb_holm_significant'] = holm_results[2][1]

    logger.info("\nHolm-Bonferroni correction (vs perm):")
    logger.info(f"  MRR significant: {pvalues['mrr_holm_significant']}")
    logger.info(f"  Recall@5 significant: {pvalues['recall@5_holm_significant']}")
    logger.info(f"  AUPRC-LB significant: {pvalues['auprc_lb_holm_significant']}")

    # Convert bools to Python bools for JSON
    pvalues_json = {k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v))
                    for k, v in pvalues.items()}

    with open(OUT_DIR / "pvalues.json", 'w') as f:
        json.dump(pvalues_json, f, indent=2)

    # Generate figures
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Scaffold split degradation
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compare with standard E2 results (MRR=0.62)
        standard_mrr = 0.6198571594195698
        standard_recall5 = 0.5365853658536586

        metrics_names = ['MRR', 'Recall@5']
        standard_vals = [standard_mrr, standard_recall5]
        scaffold_vals = [obs_mrr_mean, obs_recall5_mean]
        scaffold_errs = [obs_mrr_std, obs_recall5_std]

        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, standard_vals, width, label='Standard (E2)', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, scaffold_vals, width, yerr=scaffold_errs, capsize=5,
                      label='Scaffold Split', color='darkorange', alpha=0.8)

        ax.set_ylabel('Metric Value')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_title(f'Standard vs Scaffold Split Performance\n'
                     f'({N_FOLDS}-fold CV, n={n_candidates} glycans)')
        ax.set_ylim(0, 0.8)

        # Add degradation annotations
        for i, (std, scaf) in enumerate(zip(standard_vals, scaffold_vals)):
            pct_drop = (std - scaf) / std * 100
            ax.annotate(f'{pct_drop:.1f}% drop', xy=(i + width/2, scaf + scaffold_errs[i] + 0.02),
                       ha='center', fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(fig_dir / "scaffold_split_degradation.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "scaffold_split_degradation.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 2: Null distribution overlay
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MRR
        ax = axes[0]
        ax.hist(df_random['mrr'].values, bins=30, alpha=0.5, color='steelblue',
               label='Random', density=True)
        ax.hist(df_perm['mrr'].values, bins=30, alpha=0.5, color='darkorange',
               label='Permutation', density=True)
        ax.axvline(obs_mrr_mean, color='red', linewidth=2, linestyle='--',
                  label=f'Observed: {obs_mrr_mean:.3f}')
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
                        obs_mrr_mean - obs_mrr_std, obs_mrr_mean + obs_mrr_std,
                        color='red', alpha=0.2)
        ax.set_xlabel('MRR')
        ax.set_ylabel('Density')
        ax.set_title(f'MRR Null Distributions\n(p vs perm = {pvalues["mrr_vs_perm"]:.4f})')
        ax.legend()

        # Recall@5
        ax = axes[1]
        ax.hist(df_random['recall@5'].values, bins=30, alpha=0.5, color='steelblue',
               label='Random', density=True)
        ax.hist(df_perm['recall@5'].values, bins=30, alpha=0.5, color='darkorange',
               label='Permutation', density=True)
        ax.axvline(obs_recall5_mean, color='red', linewidth=2, linestyle='--',
                  label=f'Observed: {obs_recall5_mean:.3f}')
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
                        obs_recall5_mean - obs_recall5_std, obs_recall5_mean + obs_recall5_std,
                        color='red', alpha=0.2)
        ax.set_xlabel('Recall@5')
        ax.set_ylabel('Density')
        ax.set_title(f'Recall@5 Null Distributions\n(p vs perm = {pvalues["recall@5_vs_perm"]:.4f})')
        ax.legend()

        plt.suptitle(f'Scaffold Split: Observed vs Null ({N_PERM_REPEATS} repeats)', fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / "scaffold_split_null_overlay.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "scaffold_split_null_overlay.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Figures saved to {fig_dir}")

    except ImportError as e:
        logger.warning(f"matplotlib not available, skipping figures: {e}")

    # Save metadata
    ssv_hash = compute_file_hash(SSV_TABLE_PATH)
    labels_hash = compute_file_hash(LABELS_PATH)

    metadata = {
        'experiment': 'E5_split_robustness_v2',
        'seed': SEED,
        'n_folds': N_FOLDS,
        'n_random_repeats': N_RANDOM_REPEATS,
        'n_perm_repeats': N_PERM_REPEATS,
        'n_candidates': n_candidates,
        'n_agents': len(agent_to_pos_indices),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'code_version': 'run_e5_scaffold_split_nulls_v2.py',
        'input_files': {
            'ssv_table': str(SSV_TABLE_PATH),
            'ssv_table_sha256': ssv_hash,
            'labels': str(LABELS_PATH),
            'labels_sha256': labels_hash
        },
        'outputs': [
            'scaffold_split_observed_by_fold.csv',
            'scaffold_split_null_by_fold.csv',
            'scaffold_split_aggregate.csv',
            'pvalues.json',
            'figures/scaffold_split_degradation.pdf',
            'figures/scaffold_split_null_overlay.pdf'
        ],
        'observed_summary': {
            'mrr_mean': float(obs_mrr_mean),
            'mrr_std': float(obs_mrr_std),
            'recall@5_mean': float(obs_recall5_mean),
            'recall@5_std': float(obs_recall5_std),
            'auprc_lb_mean': float(obs_auprc_mean),
            'auprc_lb_std': float(obs_auprc_std)
        },
        'null_summary': {
            'random_mrr_mean': float(random_mrr_mean),
            'random_recall@5_mean': float(random_recall5_mean),
            'perm_mrr_mean': float(perm_mrr_mean),
            'perm_recall@5_mean': float(perm_recall5_mean)
        },
        'pvalues': pvalues_json
    }

    with open(OUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Copy scaffold rules from existing E5
    scaffold_rules_path = E5_EXISTING_PATH / "scaffold_rules.json"
    if scaffold_rules_path.exists():
        import shutil
        shutil.copy(scaffold_rules_path, OUT_DIR / "scaffold_rules.json")
        logger.info("Copied scaffold_rules.json from existing E5")

    logger.info("\n" + "=" * 70)
    logger.info("E5_v2 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {OUT_DIR}")
    logger.info(f"Observed MRR: {obs_mrr_mean:.4f} +/- {obs_mrr_std:.4f}")
    logger.info(f"Observed Recall@5: {obs_recall5_mean:.4f} +/- {obs_recall5_std:.4f}")
    logger.info(f"P-value MRR vs perm: {pvalues['mrr_vs_perm']:.6f}")
    logger.info(f"P-value Recall@5 vs perm: {pvalues['recall@5_vs_perm']:.6f}")
    logger.info(f"All Holm-significant: {all([pvalues['mrr_holm_significant'], pvalues['recall@5_holm_significant']])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
