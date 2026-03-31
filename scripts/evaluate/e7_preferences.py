#!/usr/bin/env python3
"""
E7: Lectin-specific SSV preference profiling (interpretability)

Goal: Demonstrate that individual SSV dimensions correspond to specific
lectin binding preferences, not just generic ranking features.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path('/home/minrui/glyco/public_glyco_mirror/results/RELEASE_RUN_V3_INTERPRETABILITY/E7_ssv_preferences')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
SSV_PATH = Path('/home/minrui/glyco/public_glyco_mirror/data/ssv/targeted_sugarbind_v0/ssv_table.csv')
LABELS_PATH = Path('/home/minrui/glyco/public_glyco_mirror/data/binding/sugarbind_v0/labels.csv')

# SSV dimensions to analyze
SSV_DIMENSIONS = [
    'n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
    'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy'
]

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan

    # Count concordant and discordant pairs
    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1

    delta = (more - less) / (n_x * n_y)
    return delta

def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    n_x, n_y = len(x), len(y)
    if n_x < 2 or n_y < 2:
        return np.nan

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))

    if pooled_std == 0:
        return np.nan

    d = (np.mean(x) - np.mean(y)) / pooled_std
    return d

def holm_correction(pvalues):
    """Apply Holm-Bonferroni correction."""
    n = len(pvalues)
    if n == 0:
        return []

    # Sort p-values and track original indices
    sorted_indices = np.argsort(pvalues)
    sorted_pvals = np.array(pvalues)[sorted_indices]

    # Apply Holm correction
    corrected = np.zeros(n)
    for i, p in enumerate(sorted_pvals):
        corrected[sorted_indices[i]] = min(1.0, p * (n - i))

    # Ensure monotonicity
    for i in range(1, n):
        if corrected[sorted_indices[i]] < corrected[sorted_indices[i-1]]:
            corrected[sorted_indices[i]] = corrected[sorted_indices[i-1]]

    return corrected.tolist()

def main():
    print("="*70)
    print("E7: Lectin-specific SSV preference profiling")
    print("="*70)

    # Load data
    print("\nLoading data...")
    ssv_df = pd.read_csv(SSV_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    print(f"  SSV table: {len(ssv_df)} glycans")
    print(f"  Labels: {len(labels_df)} binding annotations")
    print(f"  Agent column identified: 'agent_id'")

    # Get all glycans with SSV features
    all_glycans = set(ssv_df['glytoucan_id'].unique())
    print(f"  Total glycans with SSV: {len(all_glycans)}")

    # Get positive pairs per agent
    positive_pairs = labels_df[labels_df['label'] == 1].copy()
    agents = positive_pairs['agent_id'].unique()
    print(f"  Total agents: {len(agents)}")

    # Filter agents with >= 5 positives
    agent_counts = positive_pairs.groupby('agent_id')['glytoucan_id'].nunique()
    qualified_agents = agent_counts[agent_counts >= 5].index.tolist()
    print(f"  Agents with >= 5 positives: {len(qualified_agents)}")

    # Prepare results
    results = []

    for agent_id in qualified_agents:
        # Get positive glycans for this agent
        pos_glycans = positive_pairs[positive_pairs['agent_id'] == agent_id]['glytoucan_id'].unique()
        pos_glycans = [g for g in pos_glycans if g in all_glycans]

        if len(pos_glycans) < 5:
            continue

        # Background glycans (all glycans except positives)
        bg_glycans = [g for g in all_glycans if g not in pos_glycans]

        # Get SSV values
        pos_ssv = ssv_df[ssv_df['glytoucan_id'].isin(pos_glycans)]
        bg_ssv = ssv_df[ssv_df['glytoucan_id'].isin(bg_glycans)]

        # Test each dimension
        agent_pvals = []
        agent_results = []

        for dim in SSV_DIMENSIONS:
            pos_vals = pos_ssv[dim].dropna().values
            bg_vals = bg_ssv[dim].dropna().values

            if len(pos_vals) < 3 or len(bg_vals) < 3:
                continue

            # Compute effect sizes
            cliff_d = cliffs_delta(pos_vals, bg_vals)
            cohen_d = cohens_d(pos_vals, bg_vals)

            # Mann-Whitney U test
            try:
                stat, pval = stats.mannwhitneyu(pos_vals, bg_vals, alternative='two-sided')
            except:
                pval = 1.0

            # Direction
            if np.mean(pos_vals) > np.mean(bg_vals):
                direction = 'higher'
            else:
                direction = 'lower'

            agent_pvals.append(pval)
            agent_results.append({
                'agent_id': agent_id,
                'ssv_dimension': dim,
                'cliff_delta': cliff_d,
                'cohens_d': cohen_d,
                'p_raw': pval,
                'n_pos': len(pos_vals),
                'n_bg': len(bg_vals),
                'pos_mean': np.mean(pos_vals),
                'bg_mean': np.mean(bg_vals),
                'direction': direction
            })

        # Apply Holm correction within agent
        if agent_pvals:
            corrected = holm_correction(agent_pvals)
            for i, res in enumerate(agent_results):
                res['p_holm'] = corrected[i]
            results.extend(agent_results)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Identify significant preferences
    sig_mask = (results_df['p_holm'] < 0.05) & (results_df['cliff_delta'].abs() >= 0.33)
    sig_df = results_df[sig_mask].copy()

    print(f"\n--- Results ---")
    print(f"  Total tests: {len(results_df)}")
    print(f"  Significant preferences (p_holm < 0.05, |delta| >= 0.33): {len(sig_df)}")

    # Count unique agents with significant preferences
    agents_with_sig = sig_df['agent_id'].nunique() if len(sig_df) > 0 else 0
    print(f"  Agents with >= 1 significant preference: {agents_with_sig}")

    # Most frequently enriched dimensions
    if len(sig_df) > 0:
        dim_counts = sig_df['ssv_dimension'].value_counts()
        print(f"\n  Most frequently enriched dimensions:")
        for dim, count in dim_counts.head(5).items():
            print(f"    {dim}: {count} agents")

    # Save full results
    results_df.to_csv(OUTPUT_DIR / 'lectin_dimension_preferences.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'lectin_dimension_preferences.csv'}")

    # Save significant only
    sig_df.to_csv(OUTPUT_DIR / 'lectin_dimension_preferences_significant.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'lectin_dimension_preferences_significant.csv'}")

    # Create heatmap
    create_heatmap(results_df, sig_df, OUTPUT_DIR)

    # Write summary
    write_summary(results_df, sig_df, qualified_agents, OUTPUT_DIR)

    print("\nE7 complete!")
    return results_df, sig_df

def create_heatmap(results_df, sig_df, output_dir):
    """Create heatmap of SSV preferences."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  Warning: matplotlib/seaborn not available, skipping heatmap")
        return

    if len(sig_df) == 0:
        print("  No significant preferences to plot")
        return

    # Get agents with significant preferences
    sig_agents = sig_df['agent_id'].unique()

    # Create pivot table for Cliff's delta
    pivot_df = results_df[results_df['agent_id'].isin(sig_agents)].pivot_table(
        index='agent_id',
        columns='ssv_dimension',
        values='cliff_delta',
        aggfunc='first'
    )

    # Create significance mask
    sig_pivot = sig_df.pivot_table(
        index='agent_id',
        columns='ssv_dimension',
        values='cliff_delta',
        aggfunc='first'
    )

    # Limit to top agents by number of significant dimensions
    agent_sig_counts = sig_df.groupby('agent_id').size().sort_values(ascending=False)
    top_agents = agent_sig_counts.head(30).index.tolist()

    if len(top_agents) == 0:
        print("  No agents to plot")
        return

    pivot_subset = pivot_df.loc[pivot_df.index.isin(top_agents)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(top_agents) * 0.4)))

    # Plot heatmap
    sns.heatmap(
        pivot_subset,
        center=0,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        annot=False,
        cbar_kws={'label': "Cliff's delta"},
        ax=ax
    )

    # Mark significant cells
    for i, agent in enumerate(pivot_subset.index):
        for j, dim in enumerate(pivot_subset.columns):
            if agent in sig_pivot.index and dim in sig_pivot.columns:
                if not pd.isna(sig_pivot.loc[agent, dim]):
                    ax.plot(j + 0.5, i + 0.5, 'k*', markersize=8)

    ax.set_xlabel('SSV Dimension')
    ax.set_ylabel('Agent (Lectin/Antibody)')
    ax.set_title('SSV Preference Profiles\n(* = significant: p_holm < 0.05, |delta| >= 0.33)')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=8)
    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'fig_ssv_preference_heatmap.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_ssv_preference_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'fig_ssv_preference_heatmap.pdf'}")

def write_summary(results_df, sig_df, qualified_agents, output_dir):
    """Write summary statistics."""
    summary_lines = [
        "E7: Lectin-specific SSV preference profiling - Summary",
        "=" * 60,
        "",
        f"Agents tested (>= 5 positives): {len(qualified_agents)}",
        f"Total dimension tests: {len(results_df)}",
        f"Significant preferences (p_holm < 0.05, |Cliff's delta| >= 0.33): {len(sig_df)}",
        "",
    ]

    if len(sig_df) > 0:
        agents_with_sig = sig_df['agent_id'].nunique()
        summary_lines.append(f"Agents with >= 1 significant preference: {agents_with_sig} ({100*agents_with_sig/len(qualified_agents):.1f}%)")
        summary_lines.append("")
        summary_lines.append("Most frequently enriched dimensions:")

        dim_counts = sig_df['ssv_dimension'].value_counts()
        for dim, count in dim_counts.items():
            pct = 100 * count / agents_with_sig
            summary_lines.append(f"  {dim}: {count} agents ({pct:.1f}%)")

        summary_lines.append("")
        summary_lines.append("Direction breakdown:")
        dir_counts = sig_df.groupby(['ssv_dimension', 'direction']).size().unstack(fill_value=0)
        for dim in dir_counts.index:
            higher = dir_counts.loc[dim, 'higher'] if 'higher' in dir_counts.columns else 0
            lower = dir_counts.loc[dim, 'lower'] if 'lower' in dir_counts.columns else 0
            summary_lines.append(f"  {dim}: higher={higher}, lower={lower}")

        summary_lines.append("")
        summary_lines.append("Top agents by number of significant dimensions:")
        agent_sig_counts = sig_df.groupby('agent_id').size().sort_values(ascending=False)
        for agent, count in agent_sig_counts.head(10).items():
            dims = sig_df[sig_df['agent_id'] == agent]['ssv_dimension'].tolist()
            summary_lines.append(f"  {agent}: {count} dims ({', '.join(dims)})")
    else:
        summary_lines.append("No significant preferences detected at threshold.")

    summary_lines.append("")
    summary_lines.append("Files generated:")
    summary_lines.append("  - lectin_dimension_preferences.csv")
    summary_lines.append("  - lectin_dimension_preferences_significant.csv")
    summary_lines.append("  - fig_ssv_preference_heatmap.pdf/.png")

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"  Saved: {output_dir / 'summary.txt'}")

if __name__ == '__main__':
    main()
