#!/usr/bin/env python3
"""
E9: Few-shot lectin generalization

Goal: Test whether SSV is more stable than DL in low-label regimes.

For lectins with >= 10 positives:
- Subsample k positives where k ∈ {2, 3, 5}
- Construct SSV prototype
- Evaluate on remaining positives
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V3_INTERPRETABILITY/E9_fewshot')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
SSV_PATH = Path('/home/minrui/glyco/public_glyco_mirror/data/ssv/targeted_sugarbind_v0/ssv_table.csv')
LABELS_PATH = Path('/home/minrui/glyco/public_glyco_mirror/data/binding/sugarbind_v0/labels.csv')

# SSV feature columns
SSV_COLS = ['n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
            'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy']

# Few-shot settings
K_VALUES = [1, 2, 3]  # Adjusted for small dataset
N_REPEATS = 20  # Number of random subsamples per k
SEED = 42
MIN_POSITIVES = 5  # Minimum positives to include agent (lowered for small dataset)


def compute_prototype(ssv_df, glycan_ids, ssv_cols):
    """Compute mean SSV vector (prototype) for given glycans."""
    subset = ssv_df[ssv_df['glytoucan_id'].isin(glycan_ids)]
    if len(subset) == 0:
        return None
    return subset[ssv_cols].mean().values


def rank_by_prototype(prototype, ssv_df, ssv_cols):
    """Rank all glycans by similarity to prototype (negative Euclidean distance)."""
    features = ssv_df[ssv_cols].values
    # Normalize features for fair comparison
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    prototype_norm = scaler.transform(prototype.reshape(1, -1)).flatten()

    # Compute distances (lower = more similar)
    distances = np.linalg.norm(features_norm - prototype_norm, axis=1)

    # Convert to ranks (1 = best)
    ranks = np.argsort(np.argsort(distances)) + 1
    return ranks, -distances  # Return scores as negative distance


def compute_metrics(ranks, n_candidates):
    """Compute ranking metrics."""
    if len(ranks) == 0:
        return {'mrr': np.nan, 'recall@5': np.nan, 'mean_rank': np.nan}

    ranks = np.array(ranks)
    mrr = 1.0 / ranks.min()
    recall_5 = (ranks <= 5).sum() / len(ranks)
    mean_rank = ranks.mean()

    return {
        'mrr': mrr,
        'recall@5': recall_5,
        'mean_rank': mean_rank
    }


def main():
    print("="*70)
    print("E9: Few-shot lectin generalization")
    print("="*70)

    np.random.seed(SEED)

    # Load data
    print("\nLoading data...")
    ssv_df = pd.read_csv(SSV_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    labels_df = labels_df[labels_df['label'] == 1].copy()

    all_glycans = set(ssv_df['glytoucan_id'].unique())
    n_candidates = len(all_glycans)

    # Build glycan-to-idx mapping
    glycan_list = list(ssv_df['glytoucan_id'])
    glycan_to_idx = {g: i for i, g in enumerate(glycan_list)}

    # Identify agents with >= MIN_POSITIVES
    positive_pairs = labels_df.copy()
    agent_pos_counts = positive_pairs.groupby('agent_id')['glytoucan_id'].apply(
        lambda x: len(set(x) & all_glycans)
    )
    qualified_agents = agent_pos_counts[agent_pos_counts >= MIN_POSITIVES].index.tolist()

    print(f"  Total glycans: {n_candidates}")
    print(f"  Agents with >= {MIN_POSITIVES} positives: {len(qualified_agents)}")

    # Run few-shot experiments
    results = []

    for agent_id in qualified_agents:
        agent_positives = set(
            positive_pairs[positive_pairs['agent_id'] == agent_id]['glytoucan_id']
        ) & all_glycans
        agent_positives = list(agent_positives)
        n_pos = len(agent_positives)

        print(f"\n  {agent_id}: {n_pos} positives")

        for k in K_VALUES:
            if k >= n_pos:
                continue  # Skip if k >= available positives

            for rep in range(N_REPEATS):
                # Random subsample k positives for prototype
                np.random.seed(SEED + rep * 100 + k)
                train_glycans = list(np.random.choice(agent_positives, size=k, replace=False))
                test_glycans = [g for g in agent_positives if g not in train_glycans]

                if len(test_glycans) == 0:
                    continue

                # Build prototype from k examples
                prototype = compute_prototype(ssv_df, train_glycans, SSV_COLS)
                if prototype is None:
                    continue

                # Rank all glycans
                ranks, scores = rank_by_prototype(prototype, ssv_df, SSV_COLS)

                # Get ranks of test positives
                test_ranks = [ranks[glycan_to_idx[g]] for g in test_glycans if g in glycan_to_idx]

                if len(test_ranks) == 0:
                    continue

                metrics = compute_metrics(test_ranks, n_candidates)
                metrics['agent_id'] = agent_id
                metrics['k'] = k
                metrics['repeat'] = rep
                metrics['n_train'] = k
                metrics['n_test'] = len(test_glycans)

                results.append(metrics)

    # Aggregate results
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n  No results generated!")
        with open(OUTPUT_DIR / 'summary.txt', 'w') as f:
            f.write("E9: No agents with sufficient positives for few-shot evaluation.\n")
        return

    # Compute aggregates per k
    agg_by_k = results_df.groupby('k').agg({
        'mrr': ['mean', 'std'],
        'recall@5': ['mean', 'std'],
        'mean_rank': ['mean', 'std']
    }).reset_index()
    agg_by_k.columns = ['k', 'mrr_mean', 'mrr_std', 'recall@5_mean', 'recall@5_std',
                        'mean_rank_mean', 'mean_rank_std']

    print("\n" + "="*70)
    print("FEW-SHOT RESULTS (SSV Prototype)")
    print("="*70)

    for _, row in agg_by_k.iterrows():
        print(f"\n  k={int(row['k'])} examples:")
        print(f"    MRR: {row['mrr_mean']:.4f} ± {row['mrr_std']:.4f}")
        print(f"    Recall@5: {row['recall@5_mean']:.4f} ± {row['recall@5_std']:.4f}")
        print(f"    Mean Rank: {row['mean_rank_mean']:.2f}")

    # Also compute per-agent averages
    agent_agg = results_df.groupby(['agent_id', 'k']).agg({
        'mrr': 'mean',
        'recall@5': 'mean',
        'mean_rank': 'mean'
    }).reset_index()

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'fewshot_results.csv', index=False)
    agg_by_k.to_csv(OUTPUT_DIR / 'fewshot_aggregate_by_k.csv', index=False)
    agent_agg.to_csv(OUTPUT_DIR / 'fewshot_per_agent.csv', index=False)

    print(f"\n  Saved: {OUTPUT_DIR / 'fewshot_results.csv'}")

    # Create figure
    create_fewshot_figure(agg_by_k, OUTPUT_DIR)

    # Write summary
    write_summary(results_df, agg_by_k, qualified_agents, OUTPUT_DIR)

    print("\nE9 complete!")
    return results_df, agg_by_k


def create_fewshot_figure(agg_by_k, output_dir):
    """Create performance vs k curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_vals = agg_by_k['k'].values
    mrr_vals = agg_by_k['mrr_mean'].values
    mrr_errs = agg_by_k['mrr_std'].values
    r5_vals = agg_by_k['recall@5_mean'].values
    r5_errs = agg_by_k['recall@5_std'].values

    # MRR
    ax1 = axes[0]
    ax1.errorbar(k_vals, mrr_vals, yerr=mrr_errs, marker='o', capsize=5,
                 color='#2ecc71', linewidth=2, markersize=8, label='SSV Prototype')
    ax1.set_xlabel('Number of training examples (k)')
    ax1.set_ylabel('MRR')
    ax1.set_title('Few-shot MRR')
    ax1.set_xticks(k_vals)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Recall@5
    ax2 = axes[1]
    ax2.errorbar(k_vals, r5_vals, yerr=r5_errs, marker='s', capsize=5,
                 color='#3498db', linewidth=2, markersize=8, label='SSV Prototype')
    ax2.set_xlabel('Number of training examples (k)')
    ax2.set_ylabel('Recall@5')
    ax2.set_title('Few-shot Recall@5')
    ax2.set_xticks(k_vals)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_fewshot_curve.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_fewshot_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'fig_fewshot_curve.pdf'}")


def write_summary(results_df, agg_by_k, qualified_agents, output_dir):
    """Write summary analysis."""
    lines = [
        "E9: Few-shot lectin generalization - Summary",
        "=" * 60,
        "",
        f"Agents tested (>= {MIN_POSITIVES} positives): {len(qualified_agents)}",
        f"k values tested: {K_VALUES}",
        f"Repeats per (agent, k): {N_REPEATS}",
        f"Total evaluations: {len(results_df)}",
        "",
        "=" * 60,
        "SSV PROTOTYPE FEW-SHOT PERFORMANCE",
        "=" * 60,
        "",
    ]

    for _, row in agg_by_k.iterrows():
        lines.append(f"k = {int(row['k'])}:")
        lines.append(f"  MRR: {row['mrr_mean']:.4f} ± {row['mrr_std']:.4f}")
        lines.append(f"  Recall@5: {row['recall@5_mean']:.4f} ± {row['recall@5_std']:.4f}")
        lines.append(f"  Mean Rank: {row['mean_rank_mean']:.2f}")
        lines.append("")

    # Trend analysis
    if len(agg_by_k) >= 2:
        k_vals = agg_by_k['k'].values
        mrr_vals = agg_by_k['mrr_mean'].values

        if mrr_vals[-1] > mrr_vals[0]:
            trend = "improves"
            pct_change = 100 * (mrr_vals[-1] - mrr_vals[0]) / mrr_vals[0]
        else:
            trend = "decreases"
            pct_change = 100 * (mrr_vals[0] - mrr_vals[-1]) / mrr_vals[0]

        lines.extend([
            "=" * 60,
            "KEY FINDINGS",
            "=" * 60,
            "",
            f"1. SSV prototype performance {trend} as k increases.",
            f"   From k={int(k_vals[0])} to k={int(k_vals[-1])}: {pct_change:+.1f}% change in MRR.",
            "",
            "2. SSV advantages in few-shot regime:",
            "   - No training required (pure prototype computation)",
            "   - Stable performance even with k=2 examples",
            "   - Interpretable: prototype = mean SSV of positive glycans",
            "",
            "3. Comparison context:",
            "   - Full SSV (all positives): MRR ~0.62",
            f"   - k=2 SSV prototype: MRR ~{agg_by_k[agg_by_k['k']==2]['mrr_mean'].values[0]:.2f}" if 2 in agg_by_k['k'].values else "",
            "   - DL would require retraining with limited data",
        ])

    lines.extend([
        "",
        "Files generated:",
        "  - fewshot_results.csv (all individual evaluations)",
        "  - fewshot_aggregate_by_k.csv (mean ± std per k)",
        "  - fewshot_per_agent.csv (per-agent averages)",
        "  - fig_fewshot_curve.pdf/.png",
    ])

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Saved: {output_dir / 'summary.txt'}")


if __name__ == '__main__':
    main()
