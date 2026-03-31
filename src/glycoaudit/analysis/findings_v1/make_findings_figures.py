#!/usr/bin/env python3
"""
Generate publication-ready figures for Findings v1 analysis.

Outputs:
- fig_level1_heatmap.pdf/.png: Preference heatmap (agents x dimensions)
- fig_level1_agent_embedding.pdf/.png: Agent embedding by preference profile
- fig_level2_ablation.pdf/.png: Ablation bar plot
- fig_level2_raw_vs_resid.pdf/.png: Scatter raw vs residualized
- fig_level3_terminal_resid.pdf/.png: Terminal preference paired plot
- fig_level3_noise_sweep.pdf/.png: Noise sweep curve
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
REPORTS_PATH = BASE_PATH / "reports"
OUTPUT_PATH = REPORTS_PATH / "findings_v1"

# Feature groups for coloring
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

GROUP_COLORS = {
    'Size': '#E41A1C',
    'Shape': '#377EB8',
    'Topology': '#4DAF4A',
    'Surface': '#984EA3',
    'Contact': '#FF7F00',
    'Graph': '#A65628',
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


def generate_level1_heatmap(
    df_prefs: pd.DataFrame,
    K: int = 10,
    max_agents: int = 50,
    logger: logging.Logger = None
) -> None:
    """
    Generate preference heatmap (agents x dimensions).
    """
    if logger is None:
        logger = setup_logging('fig_level1_heatmap')

    logger.info(f"Generating Level 1 heatmap (K={K})...")

    # Filter to K
    df_k = df_prefs[df_prefs['K'] == K].copy()

    # Pivot: agents x dimensions
    df_pivot = df_k.pivot(index='agent_id', columns='dimension', values='pref_score')

    # Select top agents by total absolute preference
    abs_sum = df_pivot.abs().sum(axis=1)
    top_agents = abs_sum.nlargest(max_agents).index
    df_pivot = df_pivot.loc[top_agents]

    # Order dimensions by group
    ordered_dims = []
    for group in ['Size', 'Shape', 'Topology', 'Surface', 'Contact', 'Graph']:
        for dim in df_pivot.columns:
            if FEATURE_GROUPS.get(dim) == group and dim not in ordered_dims:
                ordered_dims.append(dim)
    df_pivot = df_pivot[ordered_dims]

    # Get significance mask
    df_sig = df_k.pivot(index='agent_id', columns='dimension', values='sig_fdr')
    df_sig = df_sig.loc[top_agents, ordered_dims]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, min(max_agents * 0.25 + 2, 16)))

    # Normalize colormap
    vmax = max(abs(df_pivot.values.min()), abs(df_pivot.values.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Heatmap
    im = ax.imshow(df_pivot.values, cmap='RdBu_r', norm=norm, aspect='auto')

    # Mark significant cells
    for i in range(len(df_pivot)):
        for j in range(len(df_pivot.columns)):
            if df_sig.iloc[i, j]:
                ax.scatter(j, i, marker='*', color='black', s=30, zorder=10)

    # Axis labels
    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_xticklabels(df_pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(df_pivot)))
    ax.set_yticklabels(df_pivot.index, fontsize=7)

    # Color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Preference Score (z-scored)', fontsize=10)

    # Legend for significance
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='* = FDR < 0.05'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Color group labels
    group_positions = {}
    for j, dim in enumerate(df_pivot.columns):
        group = FEATURE_GROUPS.get(dim, 'Other')
        if group not in group_positions:
            group_positions[group] = []
        group_positions[group].append(j)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([np.mean(pos) for pos in group_positions.values()])
    ax2.set_xticklabels(group_positions.keys(), fontsize=9)

    ax.set_xlabel('Feature Dimension', fontsize=11)
    ax.set_ylabel('Agent', fontsize=11)
    ax.set_title(f'Level 1: Agent Dimension Preferences (K={K}, top {max_agents} agents)', fontsize=12)

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_PATH / f"fig_level1_heatmap.{ext}", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved fig_level1_heatmap.pdf/.png")


def generate_level1_embedding(
    df_prefs: pd.DataFrame,
    K: int = 10,
    n_clusters: int = 5,
    logger: logging.Logger = None
) -> None:
    """
    Generate agent embedding by preference profile.
    """
    if logger is None:
        logger = setup_logging('fig_level1_embedding')

    logger.info(f"Generating Level 1 agent embedding (K={K})...")

    if not HAS_SKLEARN:
        logger.warning("  sklearn not available, skipping embedding")
        return

    # Filter to K
    df_k = df_prefs[df_prefs['K'] == K].copy()

    # Pivot: agents x dimensions
    df_pivot = df_k.pivot(index='agent_id', columns='dimension', values='pref_score')
    df_pivot = df_pivot.fillna(0)

    if len(df_pivot) < 10:
        logger.warning("  Too few agents for embedding")
        return

    # Use PCA or UMAP
    X = df_pivot.values

    if HAS_UMAP and len(df_pivot) >= 15:
        reducer = umap.UMAP(n_neighbors=min(15, len(df_pivot) - 1), min_dist=0.1, random_state=1)
        coords = reducer.fit_transform(X)
        method = 'UMAP'
    else:
        pca = PCA(n_components=2, random_state=1)
        coords = pca.fit_transform(X)
        method = 'PCA'

    # Cluster
    kmeans = KMeans(n_clusters=min(n_clusters, len(df_pivot)), random_state=1, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=clusters, cmap='Set1', s=50, alpha=0.7
    )

    # Label some agents
    n_label = min(10, len(df_pivot))
    # Label agents with highest total preference magnitude
    total_pref = df_pivot.abs().sum(axis=1)
    top_agents = total_pref.nlargest(n_label).index

    for agent in top_agents:
        idx = df_pivot.index.get_loc(agent)
        ax.annotate(
            agent[:20] + '...' if len(agent) > 20 else agent,
            (coords[idx, 0], coords[idx, 1]),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords='offset points'
        )

    ax.set_xlabel(f'{method} 1', fontsize=11)
    ax.set_ylabel(f'{method} 2', fontsize=11)
    ax.set_title(f'Level 1: Agent Embedding by Preference Profile (K={K})', fontsize=12)

    # Legend for clusters
    handles, labels = scatter.legend_elements()
    ax.legend(handles, [f'Cluster {i}' for i in range(len(handles))],
              loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_PATH / f"fig_level1_agent_embedding.{ext}", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved fig_level1_agent_embedding.pdf/.png")


def generate_level2_ablation_plot(
    df_stats: pd.DataFrame,
    logger: logging.Logger = None
) -> None:
    """
    Generate ablation bar plot with confidence intervals.
    """
    if logger is None:
        logger = setup_logging('fig_level2_ablation')

    logger.info("Generating Level 2 ablation plot...")

    # Filter to MRR metric
    df_mrr = df_stats[df_stats['metric'] == 'mrr'].copy()

    if len(df_mrr) == 0:
        logger.warning("  No MRR data available")
        return

    # Separate drop and only conditions
    df_drop = df_mrr[df_mrr['condition'].str.startswith('drop_')].copy()
    df_only = df_mrr[df_mrr['condition'].str.startswith('only_')].copy()

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot drop conditions
    ax = axes[0]
    blocks = df_drop['block'].values
    deltas = df_drop['median_delta'].values
    ci_low = df_drop['ci_low'].values
    ci_high = df_drop['ci_high'].values
    sig = df_drop['sig_holm'].values if 'sig_holm' in df_drop.columns else [False] * len(df_drop)

    colors = ['red' if d < 0 else 'green' for d in deltas]

    x = np.arange(len(blocks))
    bars = ax.bar(x, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(x, deltas, yerr=[deltas - ci_low, ci_high - deltas],
                fmt='none', color='black', capsize=3)

    # Mark significant
    for i, is_sig in enumerate(sig):
        if is_sig:
            ax.annotate('*', (x[i], deltas[i] + (ci_high[i] - deltas[i]) + 0.005),
                       ha='center', fontsize=14, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('Block_', '') for b in blocks], rotation=45, ha='right')
    ax.set_xlabel('Dropped Block', fontsize=11)
    ax.set_ylabel('MRR Delta (vs. all features)', fontsize=11)
    ax.set_title('Effect of Dropping Feature Blocks', fontsize=12)

    # Plot only conditions
    ax = axes[1]
    blocks = df_only['block'].values
    deltas = df_only['median_delta'].values
    ci_low = df_only['ci_low'].values
    ci_high = df_only['ci_high'].values

    x = np.arange(len(blocks))
    bars = ax.bar(x, deltas, color='steelblue', alpha=0.7, edgecolor='black')
    ax.errorbar(x, deltas, yerr=[deltas - ci_low, ci_high - deltas],
                fmt='none', color='black', capsize=3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('Block_', '') for b in blocks], rotation=45, ha='right')
    ax.set_xlabel('Block Used Alone', fontsize=11)
    ax.set_ylabel('MRR Delta (vs. all features)', fontsize=11)
    ax.set_title('Performance Using Only Single Block', fontsize=12)

    plt.suptitle('Level 2: Feature Block Ablation Analysis', fontsize=14)
    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_PATH / f"fig_level2_ablation.{ext}", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("  Saved fig_level2_ablation.pdf/.png")


def generate_level2_scatter(
    df_deltas: pd.DataFrame,
    logger: logging.Logger = None
) -> None:
    """
    Generate scatter plot of raw vs residualized MRR.
    """
    if logger is None:
        logger = setup_logging('fig_level2_scatter')

    logger.info("Generating Level 2 raw vs residualized scatter...")

    # Filter to MRR
    df_mrr = df_deltas[df_deltas['metric'] == 'mrr'].copy()

    if len(df_mrr) == 0:
        logger.warning("  No MRR data available")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    x = df_mrr['val_raw'].values
    y = df_mrr['val_residualized'].values

    ax.scatter(x, y, alpha=0.5, s=30, c='steelblue', edgecolors='white', linewidth=0.5)

    # y=x line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y = x')

    # Fit line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), 'r-', alpha=0.7, linewidth=1.5, label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    # Stats
    from scipy import stats as scipy_stats
    r, p_val = scipy_stats.pearsonr(x, y)
    n_above = (y > x).sum()
    n_below = (y < x).sum()

    ax.text(0.05, 0.95, f'r = {r:.3f}\n{n_above} above / {n_below} below diagonal',
            transform=ax.transAxes, fontsize=10, va='top')

    ax.set_xlabel('Raw Features MRR', fontsize=11)
    ax.set_ylabel('Residualized Features MRR', fontsize=11)
    ax.set_title('Level 2: Raw vs Size-Residualized Performance', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_PATH / f"fig_level2_raw_vs_resid.{ext}", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("  Saved fig_level2_raw_vs_resid.pdf/.png")


def generate_level3_terminal_plot(
    df_h1: pd.DataFrame,
    logger: logging.Logger = None
) -> None:
    """
    Generate terminal preference raw vs residualized paired plot.
    """
    if logger is None:
        logger = setup_logging('fig_level3_terminal')

    logger.info("Generating Level 3 terminal preference plot...")

    if len(df_h1) == 0:
        logger.warning("  No H1 data available")
        return

    # Select agents with significant raw terminal preference
    df_sig = df_h1[df_h1['pref_terminal_raw'].abs() > 0.3].copy()
    df_sig = df_sig.nlargest(30, 'pref_terminal_raw', keep='first')

    if len(df_sig) < 5:
        df_sig = df_h1.nlargest(30, 'pref_terminal_raw', keep='first')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    x = df_sig['pref_terminal_raw'].values
    y = df_sig['pref_terminal_resid'].values
    sig = df_sig['sig_terminal_fdr'].values if 'sig_terminal_fdr' in df_sig.columns else [False] * len(df_sig)

    # Plot points
    for i in range(len(x)):
        color = 'red' if sig[i] else 'gray'
        alpha = 0.8 if sig[i] else 0.4
        ax.scatter(x[i], y[i], c=color, s=50, alpha=alpha, edgecolors='white', linewidth=0.5)

    # y=x line
    lims = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y = x (no change)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Legend
    legend_elements = [
        plt.scatter([], [], c='red', s=50, label='Sig after residualization (FDR<0.05)'),
        plt.scatter([], [], c='gray', s=50, alpha=0.4, label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.set_xlabel('Terminal Preference (Raw)', fontsize=11)
    ax.set_ylabel('Terminal Preference (Size-Residualized)', fontsize=11)
    ax.set_title('H1: Terminal Preferences Before/After Size Correction', fontsize=12)

    n_persist = sig.sum()
    ax.text(0.05, 0.95, f'{n_persist}/{len(df_sig)} preferences persist\nafter size correction',
            transform=ax.transAxes, fontsize=10, va='top')

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_PATH / f"fig_level3_terminal_resid.{ext}", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("  Saved fig_level3_terminal_resid.pdf/.png")


def generate_level3_noise_plot(
    df_agg: pd.DataFrame,
    df_full: pd.DataFrame = None,
    logger: logging.Logger = None
) -> None:
    """
    Generate noise sweep curve and boxplot.
    """
    if logger is None:
        logger = setup_logging('fig_level3_noise')

    logger.info("Generating Level 3 noise sweep plot...")

    if len(df_agg) == 0:
        logger.warning("  No noise sweep data available")
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: line plot with error bands
    ax = axes[0]
    noise_levels = df_agg['noise_level'].values * 100
    mrr_mean = df_agg['mrr_mean'].values
    mrr_std = df_agg['mrr_std'].values

    ax.plot(noise_levels, mrr_mean, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(noise_levels, mrr_mean - mrr_std, mrr_mean + mrr_std,
                    alpha=0.3, color='steelblue')

    # Mark best
    best_idx = np.argmax(mrr_mean)
    ax.scatter([noise_levels[best_idx]], [mrr_mean[best_idx]],
               s=150, marker='*', color='red', zorder=10, label=f'Best: {noise_levels[best_idx]:.0f}%')

    ax.set_xlabel('Noise Level (%)', fontsize=11)
    ax.set_ylabel('Mean MRR', fontsize=11)
    ax.set_title('H2: MRR vs Noise Level (macro average)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: boxplot of per-agent MRR by noise level
    ax = axes[1]

    if df_full is not None and len(df_full) > 0:
        # Aggregate per agent-noise level
        df_agent_agg = df_full.groupby(['noise_level', 'agent_id'])['mrr'].mean().reset_index()

        noise_levels_unique = sorted(df_agent_agg['noise_level'].unique())
        data_for_boxplot = [
            df_agent_agg[df_agent_agg['noise_level'] == nl]['mrr'].values
            for nl in noise_levels_unique
        ]

        bp = ax.boxplot(data_for_boxplot, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)

        ax.set_xticklabels([f'{nl*100:.0f}%' for nl in noise_levels_unique])
        ax.set_xlabel('Noise Level', fontsize=11)
        ax.set_ylabel('MRR (per agent)', fontsize=11)
        ax.set_title('H2: Per-Agent MRR Distribution', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No per-agent data available',
                transform=ax.transAxes, ha='center', va='center')

    plt.suptitle('Level 3: Noise Tolerance Analysis', fontsize=14)
    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_PATH / f"fig_level3_noise_sweep.{ext}", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("  Saved fig_level3_noise_sweep.pdf/.png")


def generate_all_figures(logger: logging.Logger = None) -> None:
    """
    Generate all figures from saved analysis results.
    """
    if logger is None:
        logger = setup_logging('make_figures')

    logger.info("=" * 60)
    logger.info("GENERATING PUBLICATION-READY FIGURES")
    logger.info("=" * 60)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Level 1 figures
    prefs_file = OUTPUT_PATH / "level1_agent_dimension_preferences.csv"
    if prefs_file.exists():
        df_prefs = pd.read_csv(prefs_file)
        generate_level1_heatmap(df_prefs, K=10, max_agents=50, logger=logger)
        generate_level1_embedding(df_prefs, K=10, logger=logger)
    else:
        logger.warning(f"Level 1 data not found: {prefs_file}")

    # Level 2 figures
    ablation_stats_file = OUTPUT_PATH / "level2_block_ablation_stats.csv"
    resid_deltas_file = OUTPUT_PATH / "level2_residualization_deltas.csv"

    if ablation_stats_file.exists():
        df_stats = pd.read_csv(ablation_stats_file)
        generate_level2_ablation_plot(df_stats, logger=logger)
    else:
        logger.warning(f"Level 2 ablation data not found: {ablation_stats_file}")

    if resid_deltas_file.exists():
        df_deltas = pd.read_csv(resid_deltas_file)
        generate_level2_scatter(df_deltas, logger=logger)
    else:
        logger.warning(f"Level 2 residualization data not found: {resid_deltas_file}")

    # Level 3 figures
    h1_file = OUTPUT_PATH / "level3_terminal_accessibility_test.csv"
    h2_file = OUTPUT_PATH / "level3_noise_sweep.csv"
    h2_agg_file = OUTPUT_PATH / "level3_noise_sweep_aggregate.csv"

    if h1_file.exists():
        df_h1 = pd.read_csv(h1_file)
        generate_level3_terminal_plot(df_h1, logger=logger)
    else:
        logger.warning(f"Level 3 H1 data not found: {h1_file}")

    if h2_agg_file.exists():
        df_h2_agg = pd.read_csv(h2_agg_file)
        df_h2_full = pd.read_csv(h2_file) if h2_file.exists() else None
        generate_level3_noise_plot(df_h2_agg, df_h2_full, logger=logger)
    else:
        logger.warning(f"Level 3 H2 data not found: {h2_agg_file}")

    logger.info("\nAll figures generated successfully!")


if __name__ == "__main__":
    logger = setup_logging('make_figures')
    generate_all_figures(logger)
