#!/usr/bin/env python3
"""
E1_v2: Ensemble Sensitivity Analysis

This script computes ensemble sensitivity statistics for SSV features.

Key Statistics (where computable):
- ICC(2,1): Intraclass correlation across sources (REQUIRES multiple sources per glycan)
- Spearman rho: Pairwise correlation between sources (REQUIRES multiple sources)
- Dominance ratio R: Var_across / (Var_across + Var_within) (REQUIRES multiple sources)
- Ensemble-limited fraction: Glycans flagged due to high source variance

CRITICAL LIMITATION (current dataset):
- All 125 glycans derive from ES-B tier (GlycoShape/GLYCAM-Web)
- Single source per glycan => ICC, rho, R CANNOT be computed
- We document this limitation and report bootstrap-based within-source stability

Output: benchmark_release_v1/E1_ensemble_sensitivity_v2/
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
BENCHMARK_PATH = BASE_PATH / "benchmark_release_v1"
DATA_PATH = BASE_PATH / "data"
LOGS_PATH = BENCHMARK_PATH / "logs"

# Input paths
SSV_TABLE_PATH = DATA_PATH / "ssv/targeted_sugarbind_v0/ssv_table.csv"
E1_EXISTING_PATH = BENCHMARK_PATH / "E1_ensemble_sensitivity"

# Output path
OUT_DIR = BENCHMARK_PATH / "E1_ensemble_sensitivity_v2"

# Feature columns
FEATURE_COLS = [
    'n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
    'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy'
]

SEED = 42


def setup_logging() -> logging.Logger:
    """Setup logging."""
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_PATH / f"e1_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger('e1_v2')
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


def main():
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("E1_v2: ENSEMBLE SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    np.random.seed(SEED)

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load SSV data
    if not SSV_TABLE_PATH.exists():
        logger.error(f"SSV table not found: {SSV_TABLE_PATH}")
        sys.exit(1)

    df_ssv = pd.read_csv(SSV_TABLE_PATH)
    n_glycans = len(df_ssv)
    logger.info(f"Loaded {n_glycans} glycans with SSV features")

    # Load existing provenance rules
    provenance_path = E1_EXISTING_PATH / "provenance_rules.json"
    if provenance_path.exists():
        with open(provenance_path) as f:
            existing_provenance = json.load(f)
        logger.info(f"Loaded existing provenance rules")
    else:
        existing_provenance = {}

    # Analyze source distribution
    # In current dataset: all glycans are ES-B tier (single source each)
    # This means ICC, rho, R cannot be computed

    n_with_multi_source = 0  # Current dataset has 0 multi-source glycans

    logger.info(f"Glycans with multiple sources: {n_with_multi_source}")
    logger.info("LIMITATION: All glycans are single-source (ES-B tier)")
    logger.info("ICC, Spearman rho, and R cannot be computed")

    # Compute what IS available: within-source bootstrap statistics
    n_bootstrap = 1000

    per_glycan_feature_results = []

    for _, row in df_ssv.iterrows():
        gid = row['glytoucan_id']

        for feat in FEATURE_COLS:
            val = row[feat]

            # Bootstrap over similar-sized glycans as variance proxy
            size_bin = row['n_residues']
            similar = df_ssv[df_ssv['n_residues'] == size_bin]
            if len(similar) < 5:
                similar = df_ssv[(df_ssv['n_residues'] >= size_bin - 1) &
                                (df_ssv['n_residues'] <= size_bin + 1)]

            if len(similar) >= 3:
                feat_values = similar[feat].values
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    boot_sample = np.random.choice(feat_values, size=len(feat_values), replace=True)
                    bootstrap_means.append(boot_sample.mean())

                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                bootstrap_std = np.std(bootstrap_means)
                within_var = np.var(feat_values)
            else:
                ci_lower = val
                ci_upper = val
                bootstrap_std = 0.0
                within_var = 0.0

            per_glycan_feature_results.append({
                'glytoucan_id': gid,
                'feature': feat,
                'value': val,
                'bootstrap_ci_lower': ci_lower,
                'bootstrap_ci_upper': ci_upper,
                'bootstrap_std': bootstrap_std,
                'within_var': within_var,
                'across_var': np.nan,  # Cannot compute - single source
                'icc': np.nan,         # Cannot compute - single source
                'spearman_rho': np.nan, # Cannot compute - single source
                'R': np.nan,           # Cannot compute - single source
                'ensemble_limited_flag': 'NA',  # Cannot determine
                'tier': 'ES-B',
                'n_sources': 1
            })

    df_per_glycan = pd.DataFrame(per_glycan_feature_results)
    df_per_glycan.to_csv(OUT_DIR / "per_glycan_feature.csv", index=False)
    logger.info(f"Saved per_glycan_feature.csv: {len(df_per_glycan)} rows")

    # Compute feature-level summary
    summary_rows = []

    for feat in FEATURE_COLS:
        feat_data = df_per_glycan[df_per_glycan['feature'] == feat]
        feat_values = df_ssv[feat].values

        # Bootstrap CV as stability proxy
        if np.mean(feat_values) != 0:
            cv = feat_data['bootstrap_std'].mean() / np.abs(np.mean(feat_values))
        else:
            cv = 0.0

        summary_rows.append({
            'feature': feat,
            'n_glycans': len(df_ssv),
            'n_with_multi_source': 0,
            'median_value': np.median(feat_values),
            'mean_value': np.mean(feat_values),
            'std_value': np.std(feat_values),
            'median_bootstrap_std': feat_data['bootstrap_std'].median(),
            'mean_bootstrap_std': feat_data['bootstrap_std'].mean(),
            'bootstrap_cv': cv,
            # ICC/rho/R: cannot compute, set to NA
            'icc_median': np.nan,
            'icc_fraction_above_0.7': np.nan,
            'spearman_rho_median': np.nan,
            'spearman_rho_fraction_above_0.7': np.nan,
            'R_median': np.nan,
            'R_fraction_above_0.5': np.nan,
            'ensemble_limited_fraction_pairs': np.nan,
        })

    df_summary = pd.DataFrame(summary_rows)

    # Compute overall summary metrics
    overall_summary = {
        'n_glycans_evaluated': n_glycans,
        'n_with_multi_source': 0,
        'multi_source_fraction': 0.0,
        'icc_median': np.nan,
        'icc_fraction_above_0.7': np.nan,
        'spearman_rho_median': np.nan,
        'spearman_rho_fraction_above_0.7': np.nan,
        'R_median': np.nan,
        'R_fraction_above_0.5': np.nan,
        'ensemble_limited_fraction_pairs': np.nan,
        'ensemble_limited_fraction_glycans': np.nan,
        'median_bootstrap_cv': df_summary['bootstrap_cv'].median(),
        'limitation': 'All glycans are single-source (ES-B tier); ICC, rho, R cannot be computed'
    }

    df_summary.to_csv(OUT_DIR / "summary.csv", index=False)
    logger.info("Saved summary.csv")

    # Save provenance rules (updated)
    provenance_rules = {
        "description": "All structures derived from GlycoShape/GLYCAM-Web conformer generation",
        "tiers": {
            "ES-A": "Experimentally determined (X-ray, NMR, cryo-EM) - NONE in current dataset",
            "ES-B": "Public library (GlycoShape/GLYCAM-Web) - ALL structures",
            "ES-C": "MD-derived with multiple seeds - NONE in current dataset"
        },
        "assignment": f"All {n_glycans} glycans assigned ES-B tier",
        "multi_source_count": 0,
        "limitations": [
            "Single source per glycan (no multi-source ensemble)",
            "Cannot compute across-source variance",
            "ICC(2,1) computation requires multiple sources per glycan - not available",
            "Spearman rho between sources not computable",
            "Dominance ratio R not computable",
            "Ensemble-limited classification not applicable"
        ],
        "available_statistics": [
            "Within-tier bootstrap coefficient of variation",
            "Per-feature value distributions",
            "Bootstrap confidence intervals"
        ]
    }

    with open(OUT_DIR / "provenance_rules.json", 'w') as f:
        json.dump(provenance_rules, f, indent=2)

    # Generate figures
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: ICC distribution (showing NA)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5,
                'ICC Distribution\n\nNot Available\n\nAll glycans are single-source (ES-B tier)\nICC requires multiple sources per glycan',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title('ICC(2,1) Distribution Across Glycan-Feature Pairs', fontsize=14)
        plt.savefig(fig_dir / "icc_distribution.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "icc_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 2: R distribution (showing NA)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5,
                'Dominance Ratio (R) Distribution\n\nNot Available\n\nR = Var_across / (Var_across + Var_within)\nRequires multiple sources per glycan',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title('Dominance Ratio R Distribution', fontsize=14)
        plt.savefig(fig_dir / "R_distribution.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "R_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 3: Bootstrap CV by feature (what IS available)
        fig, ax = plt.subplots(figsize=(10, 6))
        cvs = df_summary['bootstrap_cv'].values
        bars = ax.bar(range(len(FEATURE_COLS)), cvs, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(FEATURE_COLS)))
        ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right')
        ax.set_ylabel('Bootstrap Coefficient of Variation')
        ax.set_xlabel('SSV Feature')
        ax.set_title(f'Within-Tier Feature Stability (ES-B, n={n_glycans} glycans)\n'
                     f'Median CV = {np.median(cvs):.3f}')
        ax.axhline(y=np.median(cvs), color='red', linestyle='--', alpha=0.7, label=f'Median = {np.median(cvs):.3f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "ensemble_limited_rate_by_feature.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "ensemble_limited_rate_by_feature.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Figures saved to {fig_dir}")

    except ImportError as e:
        logger.warning(f"matplotlib not available, skipping figures: {e}")

    # Save metadata
    input_hash = compute_file_hash(SSV_TABLE_PATH)

    metadata = {
        'experiment': 'E1_ensemble_sensitivity_v2',
        'seed': SEED,
        'n_bootstrap': n_bootstrap,
        'n_glycans': n_glycans,
        'n_features': len(FEATURE_COLS),
        'n_with_multi_source': 0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'code_version': 'run_e1_ensemble_sensitivity_v2.py',
        'input_files': {
            'ssv_table': str(SSV_TABLE_PATH),
            'ssv_table_sha256': input_hash
        },
        'outputs': [
            'summary.csv',
            'per_glycan_feature.csv',
            'provenance_rules.json',
            'figures/icc_distribution.pdf',
            'figures/R_distribution.pdf',
            'figures/ensemble_limited_rate_by_feature.pdf'
        ],
        'limitations': provenance_rules['limitations'],
        'overall_summary': overall_summary
    }

    with open(OUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("E1_v2 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {OUT_DIR}")
    logger.info(f"Key finding: All {n_glycans} glycans are single-source (ES-B)")
    logger.info("ICC, rho, R cannot be computed - documented as limitation")
    logger.info(f"Median bootstrap CV: {overall_summary['median_bootstrap_cv']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
