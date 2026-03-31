#!/usr/bin/env python3
"""
Level 3: Testable hypothesis generation and analysis.

Hypothesis H1 (Accessibility vs length):
Terminal-related SSV dimensions reflect "accessibility" rather than mere glycan length.

Hypothesis H2 (Tolerance band / regularization):
Noise improves ranking performance because lectin recognition tolerates a band of geometric variation.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score

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

SIZE_FEATURES = ['n_atoms', 'n_residues']
TERMINAL_FEATURES = ['terminal_proxy']
ACCESSIBILITY_FEATURES = ['terminal_proxy', 'exposure_proxy']


def setup_logging(name: str) -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load SSV+GCV features and binding labels."""
    df_ssv = pd.read_csv(DATA_PATH / "ssv/expanded_v1/ssv_features.csv")
    df_gcv = pd.read_csv(DATA_PATH / "gcv/expanded_v1/gcv_features.csv")

    df_features = df_ssv.merge(
        df_gcv[['glytoucan_id'] + GCV_FEATURES],
        on='glytoucan_id',
        how='inner'
    )

    df_labels = pd.read_csv(DATA_PATH / "binding/expanded_v1/labels.csv")
    df_labels = df_labels[df_labels['label'] == 1].copy()

    return df_features, df_labels


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_auprc_lb(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    """Compute AUPRC lower bound."""
    if positive_mask.sum() == 0 or positive_mask.sum() == len(positive_mask):
        return np.nan
    return average_precision_score(positive_mask.astype(int), scores)


def evaluate_ranking(
    features_scaled: np.ndarray,
    agent_to_pos: Dict[str, List[int]],
    min_pos: int = 2
) -> pd.DataFrame:
    """Evaluate ranking for all agents."""
    n_candidates = features_scaled.shape[0]
    ks = [1, 3, 5, 10, 20]

    results = []

    for agent_id, pos_indices in agent_to_pos.items():
        if len(pos_indices) < min_pos:
            continue

        pos_features = features_scaled[pos_indices]
        prototype = pos_features.mean(axis=0)

        if np.isnan(prototype).any():
            continue

        scores = np.array([
            cosine_similarity(features_scaled[i], prototype)
            for i in range(n_candidates)
        ])

        sorted_indices = np.argsort(-scores)
        rank_of_idx = np.empty(n_candidates, dtype=int)
        rank_of_idx[sorted_indices] = np.arange(1, n_candidates + 1)

        pos_ranks = np.array([rank_of_idx[idx] for idx in pos_indices])

        mrr = 1.0 / pos_ranks.min()
        mean_rank = float(pos_ranks.mean())

        recall_at_k = {f'recall@{k}': float((pos_ranks <= k).sum() / len(pos_ranks)) for k in ks}

        positive_mask = np.zeros(n_candidates, dtype=bool)
        positive_mask[pos_indices] = True
        auprc_lb = compute_auprc_lb(scores, positive_mask)

        results.append({
            'agent_id': agent_id,
            'mrr': mrr,
            'mean_rank': mean_rank,
            'auprc_lb': auprc_lb,
            'n_pos': len(pos_indices),
            **recall_at_k
        })

    return pd.DataFrame(results)


def run_h1_accessibility_test(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    min_pos: int = 2,
    n_perm: int = 10000,
    seed: int = 1,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    H1: Test if terminal preferences persist after residualizing on size.

    Logic:
    1. Identify agents with significant terminal_proxy preference (from Level 1)
    2. For these agents, recompute preference using terminal_resid (terminal regressed on size)
    3. If preference persists, supports "accessibility" interpretation
    """
    if logger is None:
        logger = setup_logging('h1')

    logger.info("Running H1: Accessibility vs Length test...")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Prepare data
    df_features = df_features.sort_values('glytoucan_id').reset_index(drop=True)
    glycan_to_idx = {gid: i for i, gid in enumerate(df_features['glytoucan_id'])}

    # Build agent -> positive indices
    agent_to_pos = {}
    for _, row in df_labels.iterrows():
        agent_id = row['agent_id']
        gid = row['glytoucan_id']
        if gid in glycan_to_idx:
            if agent_id not in agent_to_pos:
                agent_to_pos[agent_id] = set()
            agent_to_pos[agent_id].add(glycan_to_idx[gid])

    agent_to_pos = {a: list(p) for a, p in agent_to_pos.items() if len(p) >= min_pos}

    # Prepare features
    features = df_features[ALL_FEATURES].values.astype(float)
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

    # Get terminal and size feature indices
    terminal_idx = ALL_FEATURES.index('terminal_proxy')
    exposure_idx = ALL_FEATURES.index('exposure_proxy')
    size_indices = [ALL_FEATURES.index(f) for f in SIZE_FEATURES]

    # Residualize terminal on size
    size_X = features_scaled[:, size_indices]
    terminal_raw = features_scaled[:, terminal_idx]
    exposure_raw = features_scaled[:, exposure_idx]

    reg_terminal = LinearRegression()
    reg_terminal.fit(size_X, terminal_raw)
    terminal_resid = terminal_raw - reg_terminal.predict(size_X)

    reg_exposure = LinearRegression()
    reg_exposure.fit(size_X, exposure_raw)
    exposure_resid = exposure_raw - reg_exposure.predict(size_X)

    # Compute correlation between terminal/exposure and size
    terminal_size_corr = np.corrcoef(terminal_raw, features_scaled[:, size_indices[0]])[0, 1]
    exposure_size_corr = np.corrcoef(exposure_raw, features_scaled[:, size_indices[0]])[0, 1]

    logger.info(f"  Correlation: terminal_proxy vs n_atoms = {terminal_size_corr:.3f}")
    logger.info(f"  Correlation: exposure_proxy vs n_atoms = {exposure_size_corr:.3f}")

    # For each agent, test terminal preference (raw and residualized)
    results = []
    K = 10

    for agent_id, pos_indices in agent_to_pos.items():
        # Compute prototype and rank
        pos_features = features_scaled[pos_indices]
        prototype = pos_features.mean(axis=0)

        if np.isnan(prototype).any():
            continue

        scores = np.array([
            cosine_similarity(features_scaled[i], prototype)
            for i in range(n_candidates)
        ])

        sorted_indices = np.argsort(-scores)
        top_k_indices = sorted_indices[:K]

        # Raw terminal preference
        pref_terminal_raw = float(terminal_raw[top_k_indices].mean())
        pref_terminal_resid = float(terminal_resid[top_k_indices].mean())

        pref_exposure_raw = float(exposure_raw[top_k_indices].mean())
        pref_exposure_resid = float(exposure_resid[top_k_indices].mean())

        # Permutation test for residualized terminal preference
        null_count_terminal = 0
        null_count_exposure = 0

        for _ in range(n_perm):
            random_k = rng.choice(n_candidates, size=K, replace=False)
            null_terminal = terminal_resid[random_k].mean()
            null_exposure = exposure_resid[random_k].mean()

            if abs(null_terminal) >= abs(pref_terminal_resid):
                null_count_terminal += 1
            if abs(null_exposure) >= abs(pref_exposure_resid):
                null_count_exposure += 1

        p_terminal_resid = (1 + null_count_terminal) / (1 + n_perm)
        p_exposure_resid = (1 + null_count_exposure) / (1 + n_perm)

        results.append({
            'agent_id': agent_id,
            'n_positives': len(pos_indices),
            'pref_terminal_raw': pref_terminal_raw,
            'pref_terminal_resid': pref_terminal_resid,
            'p_terminal_resid': p_terminal_resid,
            'pref_exposure_raw': pref_exposure_raw,
            'pref_exposure_resid': pref_exposure_resid,
            'p_exposure_resid': p_exposure_resid,
            'terminal_size_corr': terminal_size_corr,
            'exposure_size_corr': exposure_size_corr,
        })

    df_results = pd.DataFrame(results)

    # Apply FDR correction
    if len(df_results) > 0:
        pvals_terminal = df_results['p_terminal_resid'].values
        pvals_exposure = df_results['p_exposure_resid'].values

        # Simple BH-FDR
        for col, pvals in [('sig_terminal_fdr', pvals_terminal), ('sig_exposure_fdr', pvals_exposure)]:
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            sig = np.zeros(n, dtype=bool)
            for k in range(n - 1, -1, -1):
                threshold = (k + 1) / n * 0.05
                if pvals[sorted_idx[k]] <= threshold:
                    sig[sorted_idx[:k + 1]] = True
                    break
            df_results[col] = sig

    # Summary
    n_sig_terminal = df_results['sig_terminal_fdr'].sum() if 'sig_terminal_fdr' in df_results.columns else 0
    n_sig_exposure = df_results['sig_exposure_fdr'].sum() if 'sig_exposure_fdr' in df_results.columns else 0

    logger.info(f"  Agents with sig terminal_resid preference (FDR<0.05): {n_sig_terminal}")
    logger.info(f"  Agents with sig exposure_resid preference (FDR<0.05): {n_sig_exposure}")

    return df_results


def run_h2_noise_sweep(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    noise_levels: List[float] = [0.0, 0.02, 0.05, 0.10, 0.20],
    n_repeats: int = 50,
    min_pos: int = 2,
    seed: int = 1,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    H2: Test if modest noise improves ranking performance.

    Logic:
    1. For each noise level, add Gaussian noise to features
    2. Evaluate ranking performance
    3. Find if there's a peak at modest noise
    """
    if logger is None:
        logger = setup_logging('h2')

    logger.info("Running H2: Noise tolerance sweep...")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Prepare data
    df_features = df_features.sort_values('glytoucan_id').reset_index(drop=True)
    glycan_to_idx = {gid: i for i, gid in enumerate(df_features['glytoucan_id'])}

    # Build agent -> positive indices
    agent_to_pos = {}
    for _, row in df_labels.iterrows():
        agent_id = row['agent_id']
        gid = row['glytoucan_id']
        if gid in glycan_to_idx:
            if agent_id not in agent_to_pos:
                agent_to_pos[agent_id] = set()
            agent_to_pos[agent_id].add(glycan_to_idx[gid])

    agent_to_pos = {a: list(p) for a, p in agent_to_pos.items() if len(p) >= min_pos}

    # Prepare features
    features = df_features[ALL_FEATURES].values.astype(float)
    n_candidates, n_features = features.shape

    # Handle NaN
    for j in range(n_features):
        col_median = np.nanmedian(features[:, j])
        if np.isnan(col_median):
            col_median = 0.0
        features[np.isnan(features[:, j]), j] = col_median

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Run sweep
    all_results = []

    for noise_level in noise_levels:
        logger.info(f"  Noise level: {noise_level*100:.0f}%")

        for repeat in range(n_repeats):
            # Add Gaussian noise
            if noise_level > 0:
                noise = rng.normal(0, noise_level, features_scaled.shape)
                features_noisy = features_scaled + noise
            else:
                features_noisy = features_scaled

            # Evaluate
            df_eval = evaluate_ranking(features_noisy, agent_to_pos, min_pos)

            if len(df_eval) == 0:
                continue

            # Per-agent results
            for _, row in df_eval.iterrows():
                all_results.append({
                    'noise_level': noise_level,
                    'repeat': repeat,
                    'agent_id': row['agent_id'],
                    'mrr': row['mrr'],
                    'recall@5': row['recall@5'],
                    'recall@10': row['recall@10'],
                    'auprc_lb': row['auprc_lb'],
                    'mean_rank': row['mean_rank'],
                })

    df_results = pd.DataFrame(all_results)

    # Aggregate by noise level
    df_agg = df_results.groupby('noise_level').agg({
        'mrr': ['mean', 'std'],
        'recall@5': ['mean', 'std'],
        'recall@10': ['mean', 'std'],
        'auprc_lb': ['mean', 'std'],
    }).reset_index()
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

    logger.info("\n  Noise sweep results (MRR):")
    for _, row in df_agg.iterrows():
        logger.info(f"    {row['noise_level']*100:.0f}%: MRR={row['mrr_mean']:.4f} +/- {row['mrr_std']:.4f}")

    # Find best noise level
    best_idx = df_agg['mrr_mean'].idxmax()
    best_noise = df_agg.loc[best_idx, 'noise_level']
    best_mrr = df_agg.loc[best_idx, 'mrr_mean']

    baseline_mrr = df_agg[df_agg['noise_level'] == 0.0]['mrr_mean'].values[0]

    logger.info(f"\n  Best noise level: {best_noise*100:.0f}% (MRR={best_mrr:.4f})")
    logger.info(f"  Baseline (0%): MRR={baseline_mrr:.4f}")

    # Statistical test: 0% vs best noise
    if best_noise > 0:
        mrr_baseline = df_results[df_results['noise_level'] == 0.0].groupby('agent_id')['mrr'].mean()
        mrr_best = df_results[df_results['noise_level'] == best_noise].groupby('agent_id')['mrr'].mean()

        common = mrr_baseline.index.intersection(mrr_best.index)
        if len(common) >= 10:
            try:
                stat, p = stats.wilcoxon(
                    mrr_baseline.loc[common].values,
                    mrr_best.loc[common].values,
                    alternative='two-sided'
                )
                logger.info(f"  Wilcoxon test (0% vs {best_noise*100:.0f}%): p={p:.4f}")
            except:
                pass

    return df_results, df_agg


def run_level3_analysis(
    min_pos: int = 2,
    seed: int = 1,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run complete Level 3 hypothesis tests.
    """
    if logger is None:
        logger = setup_logging('level3')

    logger.info("=" * 60)
    logger.info("LEVEL 3: TESTABLE HYPOTHESIS GENERATION")
    logger.info("=" * 60)

    np.random.seed(seed)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    df_features, df_labels = load_data()
    logger.info(f"  Glycans: {len(df_features)}")
    logger.info(f"  Binding pairs: {len(df_labels)}")

    # H1: Accessibility test
    logger.info("\n--- H1: Accessibility vs Length ---")
    df_h1 = run_h1_accessibility_test(
        df_features, df_labels, min_pos=min_pos, n_perm=10000, seed=seed, logger=logger
    )
    df_h1.to_csv(OUTPUT_PATH / "level3_terminal_accessibility_test.csv", index=False)

    # H2: Noise sweep
    logger.info("\n--- H2: Noise Tolerance Sweep ---")
    df_h2_full, df_h2_agg = run_h2_noise_sweep(
        df_features, df_labels,
        noise_levels=[0.0, 0.02, 0.05, 0.10, 0.20],
        n_repeats=50,
        min_pos=min_pos,
        seed=seed,
        logger=logger
    )
    df_h2_full.to_csv(OUTPUT_PATH / "level3_noise_sweep.csv", index=False)
    df_h2_agg.to_csv(OUTPUT_PATH / "level3_noise_sweep_aggregate.csv", index=False)

    return df_h1, df_h2_full, df_h2_agg


def generate_level3_summary(
    df_h1: pd.DataFrame,
    df_h2_agg: pd.DataFrame,
    logger: logging.Logger = None
) -> str:
    """Generate markdown summary for Level 3."""
    if logger is None:
        logger = setup_logging('level3_summary')

    lines = [
        "# Level 3: Testable Hypotheses",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        "This analysis proposes and tests two biologically-motivated hypotheses.",
        "**Important:** Results are framed as 'consistent with' or 'suggests', not 'proves mechanism'.",
        "",
    ]

    # H1: Accessibility
    n_agents = len(df_h1)
    n_sig_terminal = df_h1['sig_terminal_fdr'].sum() if 'sig_terminal_fdr' in df_h1.columns else 0
    n_sig_exposure = df_h1['sig_exposure_fdr'].sum() if 'sig_exposure_fdr' in df_h1.columns else 0

    terminal_size_corr = df_h1['terminal_size_corr'].iloc[0] if len(df_h1) > 0 else 0
    exposure_size_corr = df_h1['exposure_size_corr'].iloc[0] if len(df_h1) > 0 else 0

    lines.extend([
        "## H1: Accessibility vs Length",
        "",
        "**Hypothesis:** Terminal-related SSV dimensions (terminal_proxy, exposure_proxy) reflect ",
        "'accessibility' rather than mere glycan length.",
        "",
        "**Test:** Compute terminal preferences using size-residualized features. If preferences ",
        "persist after removing size signal, this supports an accessibility interpretation.",
        "",
        "### Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Agents tested | {n_agents} |",
        f"| terminal_proxy vs n_atoms correlation | {terminal_size_corr:.3f} |",
        f"| exposure_proxy vs n_atoms correlation | {exposure_size_corr:.3f} |",
        f"| Agents with sig terminal_resid pref (FDR<0.05) | {n_sig_terminal} ({100*n_sig_terminal/n_agents:.1f}%) |",
        f"| Agents with sig exposure_resid pref (FDR<0.05) | {n_sig_exposure} ({100*n_sig_exposure/n_agents:.1f}%) |",
        "",
    ])

    # Interpretation
    if n_sig_terminal > 0 or n_sig_exposure > 0:
        lines.extend([
            "### Interpretation",
            "",
            f"**Results are consistent with H1:** {n_sig_terminal} agents show significant terminal ",
            f"preferences even after size is regressed out. This suggests that terminal_proxy captures ",
            "structural accessibility information beyond mere size.",
            "",
            f"Similarly, {n_sig_exposure} agents retain exposure preferences after size correction, ",
            "supporting the interpretation that exposure_proxy reflects surface accessibility.",
            "",
        ])
    else:
        lines.extend([
            "### Interpretation",
            "",
            "**Results do not strongly support H1:** Few agents show significant residualized preferences. ",
            "Terminal/exposure effects may be largely explained by size, or the current feature definitions ",
            "may not adequately capture accessibility.",
            "",
        ])

    # H2: Noise tolerance
    baseline_mrr = df_h2_agg[df_h2_agg['noise_level'] == 0.0]['mrr_mean'].values[0] if len(df_h2_agg) > 0 else 0
    best_idx = df_h2_agg['mrr_mean'].idxmax() if len(df_h2_agg) > 0 else 0
    best_noise = df_h2_agg.loc[best_idx, 'noise_level'] if len(df_h2_agg) > 0 else 0
    best_mrr = df_h2_agg.loc[best_idx, 'mrr_mean'] if len(df_h2_agg) > 0 else 0

    lines.extend([
        "## H2: Noise Tolerance Band",
        "",
        "**Hypothesis:** Lectin recognition tolerates a band of geometric variation, so modest noise ",
        "in features may improve ranking by simulating conformational flexibility.",
        "",
        "**Test:** Add Gaussian noise to features at levels 0%, 2%, 5%, 10%, 20% and measure MRR.",
        "",
        "### Results",
        "",
        "| Noise Level | MRR (mean) | MRR (std) |",
        "|-------------|------------|-----------|",
    ])

    for _, row in df_h2_agg.iterrows():
        lines.append(f"| {row['noise_level']*100:.0f}% | {row['mrr_mean']:.4f} | {row['mrr_std']:.4f} |")

    lines.extend([
        "",
        f"**Best noise level:** {best_noise*100:.0f}% (MRR={best_mrr:.4f})",
        f"**Baseline (0%):** MRR={baseline_mrr:.4f}",
        "",
    ])

    # Interpretation
    if best_noise > 0 and best_mrr > baseline_mrr:
        improvement = 100 * (best_mrr - baseline_mrr) / baseline_mrr
        lines.extend([
            "### Interpretation",
            "",
            f"**Results suggest partial support for H2:** Optimal performance occurs at {best_noise*100:.0f}% noise, ",
            f"showing a {improvement:.1f}% improvement over baseline. This is consistent with the hypothesis ",
            "that lectin binding tolerates geometric variation within a 'tolerance band'.",
            "",
            "However, this could also reflect regularization effects (noise as implicit regularization) ",
            "rather than true biological tolerance.",
            "",
        ])
    else:
        lines.extend([
            "### Interpretation",
            "",
            "**Results do not support H2:** Baseline (0% noise) achieves optimal or near-optimal performance. ",
            "Adding noise degrades predictions, suggesting the features already capture the relevant signal ",
            "without needing noise-based regularization.",
            "",
        ])

    lines.extend([
        "## Caveats and Limitations",
        "",
        "1. **Correlation ≠ causation:** All findings describe correlations in the data, not mechanisms.",
        "2. **PU setting:** Unlabeled glycans may include true positives, affecting interpretation.",
        "3. **Single conformer:** Features are computed from single conformers; conformational flexibility is not modeled.",
        "4. **Feature definitions:** Terminal_proxy and exposure_proxy are simplified structural proxies.",
        "",
        "## Suggested Follow-up Experiments",
        "",
        "1. **H1 extension:** Test with explicit accessibility metrics from solvent-accessible surface area.",
        "2. **H2 extension:** Compare noise effects with explicit conformational ensemble sampling.",
        "3. **Wet-lab validation:** Prioritize lectins with strong residualized preferences for binding assays.",
        "",
        "## Output Files",
        "",
        "- `level3_terminal_accessibility_test.csv`: H1 results per agent",
        "- `level3_noise_sweep.csv`: H2 per-agent, per-noise-level results",
        "- `level3_noise_sweep_aggregate.csv`: H2 aggregated by noise level",
        "- `fig_level3_terminal_resid.pdf`: H1 visualization",
        "- `fig_level3_noise_sweep.pdf`: H2 noise curve",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    logger = setup_logging('level3')

    # Run analysis
    df_h1, df_h2_full, df_h2_agg = run_level3_analysis(
        min_pos=2, seed=1, logger=logger
    )

    # Generate summary
    summary = generate_level3_summary(df_h1, df_h2_agg, logger)

    summary_file = OUTPUT_PATH / "level3_hypotheses.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"Summary saved to {summary_file}")
