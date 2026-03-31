#!/usr/bin/env python3
"""
Build Benchmark Release Summary

Aggregates all experiment results into:
- benchmark_summary.csv
- benchmark_summary.json

Reads from:
- E1_ensemble_sensitivity_v2/
- E2_pu_ranking/
- E4_runtime_scaling/
- E5_split_robustness_v2/
- E6_size_control/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Configuration
BENCHMARK_PATH = Path("/home/minrui/glyco/public_glyco_mirror/benchmark_release_v1")


def load_json(filepath: Path) -> Dict:
    """Load JSON file safely."""
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        return json.load(f)


def load_csv_first_row(filepath: Path, column: str) -> Any:
    """Load a value from first row of CSV."""
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    if len(df) > 0 and column in df.columns:
        return df[column].iloc[0]
    return None


def main():
    print("=" * 70)
    print("BUILDING BENCHMARK RELEASE SUMMARY")
    print("=" * 70)

    metrics = []

    # ========== E1: Ensemble Sensitivity (v2 preferred, fallback to v1) ==========
    e1_v2_path = BENCHMARK_PATH / "E1_ensemble_sensitivity_v2"
    e1_v1_path = BENCHMARK_PATH / "E1_ensemble_sensitivity"

    if e1_v2_path.exists():
        e1_path = e1_v2_path
        e1_source = "E1_ensemble_sensitivity_v2"
        print(f"Using E1_v2: {e1_v2_path}")
    else:
        e1_path = e1_v1_path
        e1_source = "E1_ensemble_sensitivity"
        print(f"Using E1_v1: {e1_v1_path}")

    # Load E1 summary
    e1_summary = e1_path / "summary.csv"
    if e1_summary.exists():
        df_e1 = pd.read_csv(e1_summary)
        n_glycans = df_e1['n_glycans'].iloc[0] if 'n_glycans' in df_e1.columns else 125

        metrics.append({
            'metric': 'n_glycans_evaluated',
            'value': int(n_glycans),
            'source_file': f"{e1_source}/summary.csv",
            'experiment': 'E1'
        })

        # Check for ICC/rho/R
        if 'n_with_multi_source' in df_e1.columns:
            n_multi = df_e1['n_with_multi_source'].iloc[0]
            metrics.append({
                'metric': 'n_glycans_with_multi_source',
                'value': int(n_multi) if pd.notna(n_multi) else 0,
                'source_file': f"{e1_source}/summary.csv",
                'experiment': 'E1'
            })

        # Bootstrap CV
        if 'bootstrap_cv' in df_e1.columns:
            median_cv = df_e1['bootstrap_cv'].median()
            metrics.append({
                'metric': 'median_bootstrap_cv',
                'value': float(median_cv),
                'source_file': f"{e1_source}/summary.csv",
                'experiment': 'E1'
            })

    # Load E1 metadata
    e1_meta = load_json(e1_path / "metadata.json")
    if e1_meta and 'overall_summary' in e1_meta:
        summary = e1_meta['overall_summary']
        if 'icc_median' in summary and pd.notna(summary.get('icc_median')):
            metrics.append({
                'metric': 'icc_median',
                'value': summary['icc_median'],
                'source_file': f"{e1_source}/metadata.json",
                'experiment': 'E1'
            })
        if 'R_median' in summary and pd.notna(summary.get('R_median')):
            metrics.append({
                'metric': 'R_median',
                'value': summary['R_median'],
                'source_file': f"{e1_source}/metadata.json",
                'experiment': 'E1'
            })
        if 'spearman_rho_median' in summary and pd.notna(summary.get('spearman_rho_median')):
            metrics.append({
                'metric': 'spearman_rho_median',
                'value': summary['spearman_rho_median'],
                'source_file': f"{e1_source}/metadata.json",
                'experiment': 'E1'
            })

    # ========== E2: PU Ranking ==========
    e2_path = BENCHMARK_PATH / "E2_pu_ranking"
    print(f"Loading E2: {e2_path}")

    e2_agg = e2_path / "aggregate_metrics.csv"
    if e2_agg.exists():
        df_e2 = pd.read_csv(e2_agg)
        observed = df_e2[df_e2['mode'] == 'observed'].iloc[0] if len(df_e2[df_e2['mode'] == 'observed']) > 0 else None

        if observed is not None:
            for col in ['mrr', 'recall@5', 'auprc_lb', 'mean_rank']:
                if col in df_e2.columns:
                    metrics.append({
                        'metric': f'observed_{col.replace("@", "_at_")}',
                        'value': float(observed[col]),
                        'source_file': 'E2_pu_ranking/aggregate_metrics.csv',
                        'experiment': 'E2'
                    })

        # Random baseline
        random_mean = df_e2[df_e2['mode'] == 'random_mean']
        random_std = df_e2[df_e2['mode'] == 'random_std']
        if len(random_mean) > 0:
            for col in ['mrr', 'recall@5', 'auprc_lb']:
                metrics.append({
                    'metric': f'random_baseline_{col.replace("@", "_at_")}_mean',
                    'value': float(random_mean[col].iloc[0]),
                    'source_file': 'E2_pu_ranking/aggregate_metrics.csv',
                    'experiment': 'E2'
                })
                if len(random_std) > 0:
                    metrics.append({
                        'metric': f'random_baseline_{col.replace("@", "_at_")}_std',
                        'value': float(random_std[col].iloc[0]),
                        'source_file': 'E2_pu_ranking/aggregate_metrics.csv',
                        'experiment': 'E2'
                    })

        # Permutation baseline
        perm_mean = df_e2[df_e2['mode'] == 'permutation_mean']
        perm_std = df_e2[df_e2['mode'] == 'permutation_std']
        if len(perm_mean) > 0:
            for col in ['mrr', 'recall@5']:
                metrics.append({
                    'metric': f'permutation_baseline_{col.replace("@", "_at_")}_mean',
                    'value': float(perm_mean[col].iloc[0]),
                    'source_file': 'E2_pu_ranking/aggregate_metrics.csv',
                    'experiment': 'E2'
                })
                if len(perm_std) > 0:
                    metrics.append({
                        'metric': f'permutation_baseline_{col.replace("@", "_at_")}_std',
                        'value': float(perm_std[col].iloc[0]),
                        'source_file': 'E2_pu_ranking/aggregate_metrics.csv',
                        'experiment': 'E2'
                    })

    # E2 p-values
    e2_pvalues = load_json(e2_path / "pvalues.json")
    if e2_pvalues:
        for key in ['mrr_vs_random', 'mrr_vs_perm', 'recall@5_vs_perm']:
            if key in e2_pvalues:
                metrics.append({
                    'metric': f'pvalue_{key.replace("@", "_at_")}',
                    'value': float(e2_pvalues[key]),
                    'source_file': 'E2_pu_ranking/pvalues.json',
                    'experiment': 'E2'
                })

        for key in ['mrr_holm_significant', 'recall@5_holm_significant', 'auprc_lb_holm_significant']:
            if key in e2_pvalues:
                metrics.append({
                    'metric': f'holm_corrected_{key.replace("@", "_at_").replace("_holm_significant", "_significant")}',
                    'value': bool(e2_pvalues[key]),
                    'source_file': 'E2_pu_ranking/pvalues.json',
                    'experiment': 'E2'
                })

    # ========== E4: Runtime Scaling ==========
    e4_path = BENCHMARK_PATH / "E4_runtime_scaling"
    print(f"Loading E4: {e4_path}")

    e4_runtime = e4_path / "runtime_table.csv"
    if e4_runtime.exists():
        df_e4 = pd.read_csv(e4_runtime)
        for config in ['optimized', 'kdtree']:
            config_data = df_e4[(df_e4['configuration'] == config) & (df_e4['n_glycans'] == 125)]
            if len(config_data) > 0:
                spg = config_data['seconds_per_glycan'].iloc[0]
                metrics.append({
                    'metric': f'runtime_{config}_seconds_per_glycan_n125',
                    'value': float(spg),
                    'source_file': 'E4_runtime_scaling/runtime_table.csv',
                    'experiment': 'E4'
                })

    # ========== E5: Split Robustness (v2 preferred) ==========
    e5_v2_path = BENCHMARK_PATH / "E5_split_robustness_v2"
    e5_v1_path = BENCHMARK_PATH / "E5_split_robustness"

    if e5_v2_path.exists():
        e5_path = e5_v2_path
        e5_source = "E5_split_robustness_v2"
        print(f"Using E5_v2: {e5_v2_path}")
    else:
        e5_path = e5_v1_path
        e5_source = "E5_split_robustness"
        print(f"Using E5_v1: {e5_v1_path}")

    # Load E5 scaffold split metrics
    if e5_source == "E5_split_robustness_v2":
        e5_agg = e5_path / "scaffold_split_aggregate.csv"
        if e5_agg.exists():
            df_e5 = pd.read_csv(e5_agg)
            obs_mean = df_e5[df_e5['mode'] == 'observed_mean']
            obs_std = df_e5[df_e5['mode'] == 'observed_std']

            if len(obs_mean) > 0:
                metrics.append({
                    'metric': 'scaffold_split_mrr_fold_mean',
                    'value': float(obs_mean['mrr'].iloc[0]),
                    'source_file': f'{e5_source}/scaffold_split_aggregate.csv',
                    'experiment': 'E5'
                })
                metrics.append({
                    'metric': 'scaffold_split_recall_at_5_fold_mean',
                    'value': float(obs_mean['recall@5'].iloc[0]),
                    'source_file': f'{e5_source}/scaffold_split_aggregate.csv',
                    'experiment': 'E5'
                })

            if len(obs_std) > 0:
                metrics.append({
                    'metric': 'scaffold_split_mrr_fold_std',
                    'value': float(obs_std['mrr'].iloc[0]),
                    'source_file': f'{e5_source}/scaffold_split_aggregate.csv',
                    'experiment': 'E5'
                })
                metrics.append({
                    'metric': 'scaffold_split_recall_at_5_fold_std',
                    'value': float(obs_std['recall@5'].iloc[0]),
                    'source_file': f'{e5_source}/scaffold_split_aggregate.csv',
                    'experiment': 'E5'
                })

        # E5 p-values
        e5_pvalues = load_json(e5_path / "pvalues.json")
        if e5_pvalues:
            for key in ['mrr_vs_perm', 'recall@5_vs_perm', 'mrr_vs_random', 'recall@5_vs_random']:
                if key in e5_pvalues:
                    metrics.append({
                        'metric': f'scaffold_pvalue_{key.replace("@", "_at_")}',
                        'value': float(e5_pvalues[key]),
                        'source_file': f'{e5_source}/pvalues.json',
                        'experiment': 'E5'
                    })

            for key in ['mrr_holm_significant', 'recall@5_holm_significant']:
                if key in e5_pvalues:
                    metrics.append({
                        'metric': f'scaffold_holm_{key.replace("@", "_at_").replace("_holm_significant", "_significant")}',
                        'value': bool(e5_pvalues[key]),
                        'source_file': f'{e5_source}/pvalues.json',
                        'experiment': 'E5'
                    })

    else:
        # Fallback to v1
        e5_scaffold = e5_path / "scaffold_split_metrics.csv"
        if e5_scaffold.exists():
            df_e5 = pd.read_csv(e5_scaffold)
            metrics.append({
                'metric': 'scaffold_split_mrr_fold_mean',
                'value': float(df_e5['mrr_mean'].mean()),
                'source_file': 'E5_split_robustness/scaffold_split_metrics.csv',
                'experiment': 'E5'
            })
            metrics.append({
                'metric': 'scaffold_split_recall_at_5_fold_mean',
                'value': float(df_e5['recall@5_mean'].mean()),
                'source_file': 'E5_split_robustness/scaffold_split_metrics.csv',
                'experiment': 'E5'
            })

    # ========== E6: Size Control ==========
    e6_path = BENCHMARK_PATH / "E6_size_control"
    print(f"Loading E6: {e6_path}")

    e6_summary = load_json(e6_path / "summary.json")
    if e6_summary:
        # Residualized metrics
        if 'residualized' in e6_summary:
            for key in ['mrr', 'recall@5', 'auprc_lb']:
                if key in e6_summary['residualized']:
                    metrics.append({
                        'metric': f'residualized_{key.replace("@", "_at_")}',
                        'value': float(e6_summary['residualized'][key]),
                        'source_file': 'E6_size_control/summary.json',
                        'experiment': 'E6'
                    })

        # Size-matched null
        if 'size_matched_null_mean' in e6_summary:
            for key in ['mrr', 'recall@5']:
                if key in e6_summary['size_matched_null_mean']:
                    metrics.append({
                        'metric': f'size_matched_null_{key.replace("@", "_at_")}_mean',
                        'value': float(e6_summary['size_matched_null_mean'][key]),
                        'source_file': 'E6_size_control/summary.json',
                        'experiment': 'E6'
                    })

        # P-values
        if 'pvalues' in e6_summary:
            for key in ['mrr_vs_size_matched', 'recall@5_vs_size_matched']:
                if key in e6_summary['pvalues']:
                    metrics.append({
                        'metric': f'pvalue_{key.replace("@", "_at_")}',
                        'value': float(e6_summary['pvalues'][key]),
                        'source_file': 'E6_size_control/summary.json',
                        'experiment': 'E6'
                    })

    # ========== Build output files ==========
    df_summary = pd.DataFrame(metrics)
    df_summary.to_csv(BENCHMARK_PATH / "benchmark_summary.csv", index=False)
    print(f"\nSaved benchmark_summary.csv: {len(metrics)} metrics")

    # JSON version
    json_output = {
        'benchmark_version': 'v1.1',
        'generation_note': 'Values extracted verbatim from computed result files. No post-hoc modification.',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'metrics': [
            {
                'metric_name': m['metric'],
                'value': m['value'],
                'source_file': m['source_file'],
                'experiment_id': m['experiment']
            }
            for m in metrics
        ]
    }

    with open(BENCHMARK_PATH / "benchmark_summary.json", 'w') as f:
        json.dump(json_output, f, indent=2, default=str)
    print("Saved benchmark_summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY BUILD COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
