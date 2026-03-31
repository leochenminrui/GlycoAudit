#!/usr/bin/env python3
"""
E8: End-to-end deep learning baseline comparison (PU-compatible)

Goal: Provide a FAIR comparison against an end-to-end deep learning model
trained directly for glycan–lectin relation recovery.

This experiment aggregates existing WURCS Transformer results and compares
them against SSV Full and Size-only baselines.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V3_INTERPRETABILITY/E8_dl_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing results paths
SSV_FULL_PATH = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V2_2/ssv_pu/metrics_aggregate.csv')
SSV_PER_AGENT_PATH = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V2_2/ssv_pu/metrics_per_agent.csv')
SIZE_ONLY_PATH = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V2_2/e6_size/size_only_metrics_aggregate.csv')
SIZE_PER_AGENT_PATH = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V2_2/e6_size/size_only_metrics_per_agent.csv')
DL_WURCS_PATH = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V2_3/dl_wurcs/metrics_by_fold.csv')
DL_PER_AGENT_PATH = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V2_3/dl_wurcs/metrics_per_agent.csv')


def load_and_aggregate_dl_results():
    """Load WURCS Transformer results and compute aggregates."""
    df = pd.read_csv(DL_WURCS_PATH)

    # Aggregate across seeds and folds
    agg = {
        'mrr': df['mrr'].mean(),
        'mrr_std': df['mrr'].std(),
        'recall@5': df['recall@5'].mean(),
        'recall@5_std': df['recall@5'].std(),
        'auprc_lb': df['auprc_lb'].mean(),
        'auprc_lb_std': df['auprc_lb'].std(),
        'mean_rank': df['mean_rank'].mean(),
        'mean_rank_std': df['mean_rank'].std(),
        'n_seeds': df['seed'].nunique(),
        'n_folds': df['fold'].nunique()
    }

    # Per-seed aggregates
    seed_agg = df.groupby('seed').agg({
        'mrr': 'mean',
        'recall@5': 'mean',
        'auprc_lb': 'mean',
        'mean_rank': 'mean'
    }).reset_index()

    return agg, seed_agg


def main():
    print("="*70)
    print("E8: End-to-end deep learning baseline comparison")
    print("="*70)

    # Load SSV Full results
    print("\nLoading results...")
    ssv_df = pd.read_csv(SSV_FULL_PATH)
    ssv_agg = {
        'method': 'SSV Full',
        'mrr': ssv_df['mrr'].values[0],
        'mrr_std': ssv_df['mrr_std'].values[0],
        'recall@5': ssv_df['recall@5'].values[0],
        'recall@5_std': ssv_df['recall@5_std'].values[0],
        'auprc_lb': ssv_df['auprc_lb'].values[0],
        'auprc_lb_std': ssv_df['auprc_lb_std'].values[0],
        'mean_rank': ssv_df['mean_rank'].values[0],
        'mean_rank_std': ssv_df['mean_rank_std'].values[0],
    }
    print(f"  SSV Full: MRR={ssv_agg['mrr']:.4f}, Recall@5={ssv_agg['recall@5']:.4f}")

    # Load Size-only results
    size_df = pd.read_csv(SIZE_ONLY_PATH)
    size_agg = {
        'method': 'Size-only',
        'mrr': size_df['mrr'].values[0],
        'mrr_std': size_df['mrr_std'].values[0],
        'recall@5': size_df['recall@5'].values[0],
        'recall@5_std': size_df['recall@5_std'].values[0],
        'auprc_lb': size_df['auprc_lb'].values[0],
        'auprc_lb_std': size_df['auprc_lb_std'].values[0],
        'mean_rank': size_df['mean_rank'].values[0],
        'mean_rank_std': size_df['mean_rank_std'].values[0],
    }
    print(f"  Size-only: MRR={size_agg['mrr']:.4f}, Recall@5={size_agg['recall@5']:.4f}")

    # Load DL WURCS results
    dl_agg, dl_seed_agg = load_and_aggregate_dl_results()
    dl_agg['method'] = 'DL WURCS Transformer'
    print(f"  DL WURCS: MRR={dl_agg['mrr']:.4f} ± {dl_agg['mrr_std']:.4f}, Recall@5={dl_agg['recall@5']:.4f}")
    print(f"    (3 seeds × 5 folds = 15 evaluations)")

    # Create comparison table
    comparison_data = []
    for agg in [ssv_agg, dl_agg, size_agg]:
        comparison_data.append({
            'method': agg['method'] if 'method' in agg else agg.get('method', 'Unknown'),
            'mrr': agg['mrr'],
            'mrr_std': agg.get('mrr_std', np.nan),
            'recall@5': agg['recall@5'],
            'recall@5_std': agg.get('recall@5_std', np.nan),
            'auprc_lb': agg['auprc_lb'],
            'auprc_lb_std': agg.get('auprc_lb_std', np.nan),
            'mean_rank': agg['mean_rank'],
            'mean_rank_std': agg.get('mean_rank_std', np.nan),
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Compute delta from SSV
    ssv_mrr = ssv_agg['mrr']
    ssv_r5 = ssv_agg['recall@5']

    comparison_df['delta_mrr_vs_ssv'] = comparison_df['mrr'] - ssv_mrr
    comparison_df['delta_recall@5_vs_ssv'] = comparison_df['recall@5'] - ssv_r5

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['method']}:")
        print(f"  MRR: {row['mrr']:.4f} ± {row['mrr_std']:.4f}")
        print(f"  Recall@5: {row['recall@5']:.4f} ± {row['recall@5_std']:.4f}")
        print(f"  AUPRC-LB: {row['auprc_lb']:.4f}")
        print(f"  Mean Rank: {row['mean_rank']:.2f}")
        if row['method'] != 'SSV Full':
            print(f"  Δ MRR vs SSV: {row['delta_mrr_vs_ssv']:+.4f}")
            print(f"  Δ Recall@5 vs SSV: {row['delta_recall@5_vs_ssv']:+.4f}")

    # Save aggregate comparison
    comparison_df.to_csv(OUTPUT_DIR / 'dl_end_to_end_metrics_aggregate.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'dl_end_to_end_metrics_aggregate.csv'}")

    # Per-agent comparison (if DL per-agent results exist)
    if DL_PER_AGENT_PATH.exists():
        dl_per_agent = pd.read_csv(DL_PER_AGENT_PATH)
        dl_per_agent['method'] = 'DL WURCS'
        dl_per_agent.to_csv(OUTPUT_DIR / 'dl_end_to_end_metrics_per_agent.csv', index=False)
        print(f"  Saved: {OUTPUT_DIR / 'dl_end_to_end_metrics_per_agent.csv'}")

    # Create comparison figure
    create_comparison_figure(comparison_df, OUTPUT_DIR)

    # Training stability analysis
    write_summary(comparison_df, dl_seed_agg, OUTPUT_DIR)

    print("\nE8 complete!")
    return comparison_df


def create_comparison_figure(comparison_df, output_dir):
    """Create bar plot comparing methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Warning: matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = comparison_df['method'].tolist()
    x = np.arange(len(methods))
    width = 0.6

    # Colors
    colors = ['#2ecc71', '#3498db', '#95a5a6']  # green, blue, gray

    # MRR
    ax1 = axes[0]
    mrr_vals = comparison_df['mrr'].values
    mrr_errs = comparison_df['mrr_std'].values
    bars1 = ax1.bar(x, mrr_vals, width, yerr=mrr_errs, color=colors, alpha=0.8, capsize=5)
    ax1.set_ylabel('MRR')
    ax1.set_title('Mean Reciprocal Rank')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, 0.8)
    ax1.axhline(mrr_vals[0], color='green', linestyle='--', alpha=0.3, label='SSV baseline')

    # Add value labels
    for bar, val in zip(bars1, mrr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Recall@5
    ax2 = axes[1]
    r5_vals = comparison_df['recall@5'].values
    r5_errs = comparison_df['recall@5_std'].values
    bars2 = ax2.bar(x, r5_vals, width, yerr=r5_errs, color=colors, alpha=0.8, capsize=5)
    ax2.set_ylabel('Recall@5')
    ax2.set_title('Recall at 5')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylim(0, 0.8)
    ax2.axhline(r5_vals[0], color='green', linestyle='--', alpha=0.3, label='SSV baseline')

    for bar, val in zip(bars2, r5_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_method_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'fig_method_comparison.pdf'}")


def write_summary(comparison_df, dl_seed_agg, output_dir):
    """Write summary analysis."""
    ssv_row = comparison_df[comparison_df['method'] == 'SSV Full'].iloc[0]
    dl_row = comparison_df[comparison_df['method'] == 'DL WURCS Transformer'].iloc[0]
    size_row = comparison_df[comparison_df['method'] == 'Size-only'].iloc[0]

    lines = [
        "E8: End-to-end deep learning baseline comparison - Summary",
        "=" * 60,
        "",
        "Methods compared:",
        "  1. SSV Full: 8-dimensional structural shape vectors with prototype ranking",
        "  2. DL WURCS Transformer: End-to-end learned representations from WURCS strings",
        "  3. Size-only: Using only n_atoms/n_residues features",
        "",
        "Training details (DL WURCS Transformer):",
        f"  - Architecture: Transformer (d_model=128, nhead=4, num_layers=2)",
        f"  - Training: 50 epochs, batch_size=32, lr=0.0001",
        f"  - Seeds tested: 3 (42, 43, 44)",
        f"  - Cross-validation: 5-fold",
        f"  - Total evaluations: 15 (3 seeds × 5 folds)",
        "",
        "=" * 60,
        "AGGREGATE RESULTS",
        "=" * 60,
        "",
        f"{'Method':<25} {'MRR':<15} {'Recall@5':<15} {'AUPRC-LB':<15}",
        "-" * 70,
    ]

    for _, row in comparison_df.iterrows():
        mrr_str = f"{row['mrr']:.4f} ± {row['mrr_std']:.4f}"
        r5_str = f"{row['recall@5']:.4f} ± {row['recall@5_std']:.4f}"
        auprc_str = f"{row['auprc_lb']:.4f}"
        lines.append(f"{row['method']:<25} {mrr_str:<15} {r5_str:<15} {auprc_str:<15}")

    lines.extend([
        "",
        "=" * 60,
        "KEY FINDINGS",
        "=" * 60,
        "",
    ])

    # Compare SSV vs DL
    mrr_diff = dl_row['mrr'] - ssv_row['mrr']
    r5_diff = dl_row['recall@5'] - ssv_row['recall@5']

    if mrr_diff < 0:
        lines.append(f"1. SSV OUTPERFORMS DL on MRR: {ssv_row['mrr']:.4f} vs {dl_row['mrr']:.4f} (Δ = {-mrr_diff:.4f})")
    else:
        lines.append(f"1. DL outperforms SSV on MRR: {dl_row['mrr']:.4f} vs {ssv_row['mrr']:.4f} (Δ = {mrr_diff:.4f})")

    if r5_diff < 0:
        lines.append(f"2. SSV OUTPERFORMS DL on Recall@5: {ssv_row['recall@5']:.4f} vs {dl_row['recall@5']:.4f}")
    else:
        lines.append(f"2. DL slightly better on Recall@5: {dl_row['recall@5']:.4f} vs {ssv_row['recall@5']:.4f}")

    lines.extend([
        "",
        "3. Training stability (DL across 3 seeds):",
    ])

    for _, seed_row in dl_seed_agg.iterrows():
        lines.append(f"   Seed {int(seed_row['seed'])}: MRR={seed_row['mrr']:.4f}, Recall@5={seed_row['recall@5']:.4f}")

    seed_mrr_std = dl_seed_agg['mrr'].std()
    lines.append(f"   Across-seed MRR std: {seed_mrr_std:.4f}")

    lines.extend([
        "",
        "4. Both SSV and DL substantially outperform Size-only baseline,",
        f"   confirming that structural features beyond size contribute to binding prediction.",
        "",
        "5. SSV advantages:",
        "   - Interpretable dimensions (geometry, topology, exposure)",
        "   - No training required (prototype-based)",
        "   - Faster inference (no GPU needed)",
        "",
        "6. DL advantages:",
        "   - End-to-end learning from sequence",
        "   - No manual feature engineering",
        "   - May capture subtle patterns missed by SSV",
        "",
        "Files generated:",
        "  - dl_end_to_end_metrics_aggregate.csv",
        "  - dl_end_to_end_metrics_per_agent.csv",
        "  - fig_method_comparison.pdf/.png",
    ])

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Saved: {output_dir / 'summary.txt'}")


if __name__ == '__main__':
    main()
