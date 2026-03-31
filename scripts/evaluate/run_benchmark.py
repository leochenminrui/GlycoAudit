#!/usr/bin/env python3
"""
Paper Benchmark v2.1 + Minor Revisions - Full Suite

Includes:
1. TRUE structure-aware splits (v2.1)
2. Bootstrap CIs for cross-source (v2.1)
3. Linkage coverage stats (minor rev task 1)
4. Scaffold distribution stats (minor rev task 2)
5. Feature ablation baselines (minor rev task 3) ⭐ NEW

Usage:
    python scripts/run_paper_benchmark_v2_1_minorrev.py --seed 42 --outdir outputs/bench_v2_1_minorrev_full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.splits import GlycanSplitter
from analysis.cross_source import (
    CrossSourceSplitter,
    compute_bootstrap_confidence_intervals,
    compare_source_overlap_distributions
)
from analysis.features import select_feature_columns, get_feature_set_description

# Import v2 runner functions
sys.path.insert(0, str(Path(__file__).parent))
from run_paper_benchmark_v2 import (
    setup_logging, evaluate_ranking, run_pu_sensitivity_evaluation
)

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")


def load_data_with_feature_set(feature_set='ssv+gcv'):
    """Load data with specified feature set."""
    ssv_path = BASE_PATH / "data/ssv/expanded_v1/ssv_features.csv"
    gcv_path = BASE_PATH / "data/gcv/expanded_v1/gcv_features.csv"
    labels_path = BASE_PATH / "data/binding/expanded_v1/labels.csv"

    df_ssv = pd.read_csv(ssv_path)
    df_gcv = pd.read_csv(gcv_path)
    df_labels = pd.read_csv(labels_path)

    # Select features based on feature_set
    df_features = select_feature_columns(df_ssv, df_gcv, feature_set)

    print(f"[Data] Loaded {len(df_features)} glycans with feature_set='{feature_set}'")
    print(f"[Data] Feature columns: {len(df_features.columns) - 1} ({get_feature_set_description(feature_set)})")

    return df_features, df_labels


def prepare_features_from_df(df_features):
    """Prepare features from DataFrame."""
    # Get feature columns (all except glytoucan_id)
    feature_cols = [col for col in df_features.columns if col != 'glytoucan_id']

    # Extract features as numpy array
    features = df_features[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create glycan -> index mapping
    glycan_to_idx = {gid: i for i, gid in enumerate(df_features['glytoucan_id'])}

    return features_scaled, glycan_to_idx


def run_split_evaluation(
    run_name,
    manifest,
    df_features,
    df_labels,
    features_scaled,
    glycan_to_idx,
    output_dir,
    logger
):
    """Run evaluation for a split."""
    logger.info(f"Evaluating {run_name}...")

    # Save manifest
    manifest.save(output_dir / "split_manifest.json")

    # Build agent -> positives mapping
    agent_to_positives = {}
    for _, row in df_labels.iterrows():
        agent_id = row['agent_id']
        gid = row['glytoucan_id']
        if agent_id not in agent_to_positives:
            agent_to_positives[agent_id] = []
        agent_to_positives[agent_id].append(gid)

    # Filter to test agents/glycans
    test_agents = manifest.test_agents
    test_glycan_set = set(manifest.test_glycans)

    # Evaluate
    df_metrics = evaluate_ranking(
        features_scaled,
        glycan_to_idx,
        agent_to_positives,
        test_agents,
        test_glycan_set,
        logger=logger
    )

    df_metrics.to_csv(output_dir / "per_agent_metrics.csv", index=False)

    # Aggregate metrics
    agg_metrics = {
        'n_agents_evaluated': len(df_metrics),
        'mrr': df_metrics['mrr'].mean() if len(df_metrics) > 0 else 0.0,
        'recall@5': df_metrics['recall@5'].mean() if len(df_metrics) > 0 else 0.0,
        'recall@10': df_metrics['recall@10'].mean() if len(df_metrics) > 0 else 0.0,
        'auprc_lb': df_metrics['auprc_lb'].mean() if len(df_metrics) > 0 else 0.0,
    }

    with open(output_dir / "aggregate_metrics.json", 'w') as f:
        json.dump(agg_metrics, f, indent=2)

    logger.info(f"  MRR: {agg_metrics['mrr']:.4f}")
    logger.info(f"  Recall@5: {agg_metrics['recall@5']:.4f}")
    logger.info(f"  Recall@10: {agg_metrics['recall@10']:.4f}")

    return agg_metrics


def run_cross_source_with_bootstrap(
    manifest,
    df_features,
    df_labels,
    features_scaled,
    glycan_to_idx,
    output_dir,
    seed,
    logger
):
    """Run cross-source evaluation with bootstrap CIs."""
    logger.info(f"Cross-source: {manifest.source_train} → {manifest.source_test}")

    manifest.save(output_dir / "cross_source_manifest.json")

    # Build agent -> positives per source
    test_labels = df_labels[df_labels['data_source'] == manifest.source_test]
    agent_to_positives_test = {}
    for _, row in test_labels.iterrows():
        agent_id = row['agent_id']
        gid = row['glytoucan_id']
        if agent_id not in agent_to_positives_test:
            agent_to_positives_test[agent_id] = []
        agent_to_positives_test[agent_id].append(gid)

    # Evaluate
    test_glycan_set = set(manifest.test_glycans)
    df_metrics = evaluate_ranking(
        features_scaled,
        glycan_to_idx,
        agent_to_positives_test,
        manifest.test_agents,
        test_glycan_set,
        logger=logger
    )

    df_metrics.to_csv(output_dir / "per_agent_metrics.csv", index=False)

    # Aggregate metrics
    agg_metrics = {
        'source_train': manifest.source_train,
        'source_test': manifest.source_test,
        'n_agents_evaluated': len(df_metrics),
        'mrr': df_metrics['mrr'].mean() if len(df_metrics) > 0 else 0.0,
        'recall@5': df_metrics['recall@5'].mean() if len(df_metrics) > 0 else 0.0,
        'recall@10': df_metrics['recall@10'].mean() if len(df_metrics) > 0 else 0.0,
        'auprc_lb': df_metrics['auprc_lb'].mean() if len(df_metrics) > 0 else 0.0,
    }

    # Bootstrap CIs
    if len(df_metrics) > 0:
        bootstrap_cis = compute_bootstrap_confidence_intervals(
            df_metrics, n_bootstrap=1000, confidence_level=0.95, seed=seed
        )
        with open(output_dir / "bootstrap_cis.json", 'w') as f:
            json.dump(bootstrap_cis, f, indent=2)

        # Add CIs to aggregate
        for metric in ['mrr', 'recall@5', 'recall@10', 'auprc_lb']:
            if metric in bootstrap_cis:
                agg_metrics[f'{metric}_ci_lower'] = bootstrap_cis[metric]['ci_lower']
                agg_metrics[f'{metric}_ci_upper'] = bootstrap_cis[metric]['ci_upper']

    with open(output_dir / "aggregate_metrics.json", 'w') as f:
        json.dump(agg_metrics, f, indent=2)

    return agg_metrics


def run_feature_ablation_suite(
    output_base,
    feature_set,
    seed,
    main_logger,
    skip_scaffold=False
):
    """
    Run feature ablation experiments.

    Args:
        output_base: Base output directory
        feature_set: 'ssv', 'gcv', or 'ssv+gcv'
        seed: Random seed
        main_logger: Logger instance
        skip_scaffold: If True, skip scaffold holdout (faster)

    Returns:
        List of result dictionaries
    """
    results = []

    # Load data with specified feature set
    df_features, df_labels = load_data_with_feature_set(feature_set)
    features_scaled, glycan_to_idx = prepare_features_from_df(df_features)

    # Initialize splitters
    glycan_splitter = GlycanSplitter(seed=seed)
    cross_source_splitter = CrossSourceSplitter(seed=seed)

    # 1. IID Split
    main_logger.info("\n" + "=" * 70)
    main_logger.info(f"IID (feature_set={feature_set})")
    main_logger.info("=" * 70)

    output_dir = output_base / f"iid__{feature_set}"
    output_dir.mkdir(exist_ok=True, parents=True)

    manifest = glycan_splitter.iid_split(df_features, df_labels, test_size=0.2)
    metrics = run_split_evaluation(
        f'iid__{feature_set}', manifest, df_features, df_labels,
        features_scaled, glycan_to_idx, output_dir, main_logger
    )
    results.append({'run_name': f'iid__{feature_set}', 'feature_set': feature_set, **metrics})

    # 2. TRUE Scaffold Holdout (optional)
    if not skip_scaffold:
        main_logger.info("\n" + "=" * 70)
        main_logger.info(f"TRUE SCAFFOLD HOLDOUT (feature_set={feature_set})")
        main_logger.info("=" * 70)

        output_dir = output_base / f"scaffold_holdout_true__{feature_set}"
        output_dir.mkdir(exist_ok=True)

        try:
            # Only save group stats for ssv+gcv (avoid duplication)
            save_stats = (feature_set == 'ssv+gcv')
            manifest = glycan_splitter.scaffold_holdout_split_true(
                df_features, df_labels, n_folds=5, test_fold=0,
                save_group_stats=save_stats,
                group_stats_output_path=output_base / "scaffold_group_stats.json" if save_stats else None,
                group_top10_csv_path=output_base / "scaffold_group_top10.csv" if save_stats else None
            )
            metrics = run_split_evaluation(
                f'scaffold_holdout_true__{feature_set}', manifest, df_features, df_labels,
                features_scaled, glycan_to_idx, output_dir, main_logger
            )
            results.append({'run_name': f'scaffold_holdout_true__{feature_set}', 'feature_set': feature_set, **metrics})
        except Exception as e:
            main_logger.error(f"TRUE scaffold failed for {feature_set}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark v2.1 + Minor Revisions")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--outdir', type=Path, default=None, help="Output directory")
    parser.add_argument('--skip-pu', action='store_true', help="Skip PU sensitivity")
    parser.add_argument('--skip-cross-source', action='store_true', help="Skip cross-source")
    parser.add_argument('--skip-scaffold-ablations', action='store_true', help="Skip scaffold holdout for ablations (faster)")
    args = parser.parse_args()

    # Set output directory
    if args.outdir:
        OUTPUT_BASE = args.outdir
    else:
        OUTPUT_BASE = BASE_PATH / "outputs/bench_v2_1_minorrev_full"

    OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

    # Setup logging
    main_logger = setup_logging("benchmark_v2_1_minorrev", OUTPUT_BASE)
    main_logger.info("=" * 70)
    main_logger.info("BENCHMARK v2.1 + MINOR REVISIONS - FULL SUITE")
    main_logger.info("=" * 70)
    main_logger.info(f"Output: {OUTPUT_BASE}")
    main_logger.info(f"Seed: {args.seed}")

    start_time = time.time()
    all_results = []

    # =========================================================================
    # PART A: FEATURE ABLATION BASELINES (Task 3)
    # =========================================================================

    main_logger.info("\n" + "=" * 70)
    main_logger.info("PART A: FEATURE ABLATION BASELINES")
    main_logger.info("=" * 70)

    for feature_set in ['ssv', 'gcv', 'ssv+gcv']:
        results = run_feature_ablation_suite(
            OUTPUT_BASE,
            feature_set,
            args.seed,
            main_logger,
            skip_scaffold=args.skip_scaffold_ablations
        )
        all_results.extend(results)

    # =========================================================================
    # PART B: FULL v2.1 SUITE (with ssv+gcv only)
    # =========================================================================

    main_logger.info("\n" + "=" * 70)
    main_logger.info("PART B: FULL v2.1 SUITE (ssv+gcv)")
    main_logger.info("=" * 70)

    # Load full feature set
    df_features, df_labels = load_data_with_feature_set('ssv+gcv')
    features_scaled, glycan_to_idx = prepare_features_from_df(df_features)

    glycan_splitter = GlycanSplitter(seed=args.seed)
    cross_source_splitter = CrossSourceSplitter(seed=args.seed)

    # Agent Holdout
    main_logger.info("\n" + "=" * 70)
    main_logger.info("AGENT HOLDOUT")
    main_logger.info("=" * 70)

    output_dir = OUTPUT_BASE / "agent_holdout"
    output_dir.mkdir(exist_ok=True)

    manifest = glycan_splitter.agent_holdout_split(df_labels, test_size=0.2)
    metrics = run_split_evaluation(
        'agent_holdout', manifest, df_features, df_labels,
        features_scaled, glycan_to_idx, output_dir, main_logger
    )
    all_results.append({'run_name': 'agent_holdout', 'feature_set': 'ssv+gcv', **metrics})

    # Terminal Motif Holdout
    main_logger.info("\n" + "=" * 70)
    main_logger.info("TRUE TERMINAL MOTIF HOLDOUT")
    main_logger.info("=" * 70)

    output_dir = OUTPUT_BASE / "terminal_motif_holdout_true"
    output_dir.mkdir(exist_ok=True)

    try:
        manifest = glycan_splitter.terminal_motif_holdout_split_true(
            df_features, df_labels, n_folds=4, test_fold=0
        )
        metrics = run_split_evaluation(
            'terminal_motif_holdout_true', manifest, df_features, df_labels,
            features_scaled, glycan_to_idx, output_dir, main_logger
        )
        all_results.append({'run_name': 'terminal_motif_holdout_true', 'feature_set': 'ssv+gcv', **metrics})
    except Exception as e:
        main_logger.error(f"TRUE terminal motif failed: {e}")

    # Linkage Holdout
    main_logger.info("\n" + "=" * 70)
    main_logger.info("TRUE LINKAGE HOLDOUT")
    main_logger.info("=" * 70)

    output_dir = OUTPUT_BASE / "linkage_holdout_true"
    output_dir.mkdir(exist_ok=True)

    try:
        manifest = glycan_splitter.linkage_holdout_split_true(
            df_features, df_labels, n_folds=4, test_fold=0,
            save_coverage_stats=True,
            coverage_output_path=OUTPUT_BASE / "linkage_key_coverage.json"
        )
        metrics = run_split_evaluation(
            'linkage_holdout_true', manifest, df_features, df_labels,
            features_scaled, glycan_to_idx, output_dir, main_logger
        )
        all_results.append({'run_name': 'linkage_holdout_true', 'feature_set': 'ssv+gcv', **metrics})
    except Exception as e:
        main_logger.error(f"TRUE linkage failed: {e}")

    # Cross-Source
    if not args.skip_cross_source:
        # CFG → SugarBind
        main_logger.info("\n" + "=" * 70)
        main_logger.info("CROSS-SOURCE: CFG → SugarBind")
        main_logger.info("=" * 70)

        output_dir = OUTPUT_BASE / "cross_source_cfg_to_sugarbind"
        output_dir.mkdir(exist_ok=True)

        try:
            manifest = cross_source_splitter.create_cross_source_split(
                df_labels, 'CFG', 'sugarbind', min_pos_per_agent=2, require_overlap=True
            )
            metrics = run_cross_source_with_bootstrap(
                manifest, df_features, df_labels,
                features_scaled, glycan_to_idx, output_dir, args.seed, main_logger
            )
            all_results.append({'run_name': 'cross_source_cfg_to_sugarbind', 'feature_set': 'ssv+gcv', **metrics})
        except Exception as e:
            main_logger.warning(f"CFG→SugarBind failed: {e}")

        # SugarBind → CFG
        main_logger.info("\n" + "=" * 70)
        main_logger.info("CROSS-SOURCE: SugarBind → CFG")
        main_logger.info("=" * 70)

        output_dir = OUTPUT_BASE / "cross_source_sugarbind_to_cfg"
        output_dir.mkdir(exist_ok=True)

        try:
            manifest = cross_source_splitter.create_cross_source_split(
                df_labels, 'sugarbind', 'CFG', min_pos_per_agent=2, require_overlap=True
            )
            metrics = run_cross_source_with_bootstrap(
                manifest, df_features, df_labels,
                features_scaled, glycan_to_idx, output_dir, args.seed, main_logger
            )
            all_results.append({'run_name': 'cross_source_sugarbind_to_cfg', 'feature_set': 'ssv+gcv', **metrics})
        except Exception as e:
            main_logger.warning(f"SugarBind→CFG failed: {e}")

    # PU Sensitivity
    if not args.skip_pu:
        main_logger.info("\n" + "=" * 70)
        main_logger.info("PU SENSITIVITY")
        main_logger.info("=" * 70)

        output_dir = OUTPUT_BASE / "pu_sensitivity"
        output_dir.mkdir(exist_ok=True)

        try:
            run_pu_sensitivity_evaluation(
                df_features, df_labels, features_scaled, glycan_to_idx,
                output_dir, args.seed, main_logger
            )
        except Exception as e:
            main_logger.error(f"PU sensitivity failed: {e}")

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    main_logger.info("\n" + "=" * 70)
    main_logger.info("SAVING SUMMARY")
    main_logger.info("=" * 70)

    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_BASE / "comparison_summary.csv", index=False)
    main_logger.info(f"Saved: {OUTPUT_BASE / 'comparison_summary.csv'}")

    elapsed = time.time() - start_time
    main_logger.info(f"\n✓ Benchmark complete in {elapsed/60:.1f} minutes")
    main_logger.info(f"Results: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
