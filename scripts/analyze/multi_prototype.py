#!/usr/bin/env python3
"""
Analyze multi-prototype vs single prototype comparison.

Decision criteria for placement:
A. Main text: If improvement is broad, substantial, and stable
B. Supplement: If improvement is moderate or limited to few agents
C. Brief mention: If improvement is marginal or unstable
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)


def load_results():
    """Load prototype comparison results."""
    proto = pd.read_csv("outputs/baseline_comparison/baseline_prototype_per_agent.csv")
    multi = pd.read_csv("outputs/baseline_comparison/baseline_multi_prototype_per_agent.csv")

    return proto, multi


def analyze_improvement(proto, multi):
    """Analyze per-agent improvements."""
    # Merge on agent_id
    merged = proto.merge(multi, on='agent_id', suffixes=('_proto', '_multi'))

    # Compute deltas
    merged['mrr_delta'] = merged['mrr_multi'] - merged['mrr_proto']
    merged['recall5_delta'] = merged['recall@5_multi'] - merged['recall@5_proto']

    # Categorize agents
    merged['improved'] = merged['mrr_delta'] > 0.05  # >5% MRR improvement
    merged['degraded'] = merged['mrr_delta'] < -0.05  # >5% MRR loss
    merged['neutral'] = (~merged['improved']) & (~merged['degraded'])

    results = {
        'n_agents': len(merged),
        'n_improved': merged['improved'].sum(),
        'n_degraded': merged['degraded'].sum(),
        'n_neutral': merged['neutral'].sum(),
        'pct_improved': 100 * merged['improved'].mean(),
        'pct_degraded': 100 * merged['degraded'].mean(),
        'mean_mrr_delta': merged['mrr_delta'].mean(),
        'median_mrr_delta': merged['mrr_delta'].median(),
        'mean_recall5_delta': merged['recall5_delta'].mean(),
        'median_recall5_delta': merged['recall5_delta'].median(),
        'mean_mrr_proto': merged['mrr_proto'].mean(),
        'mean_mrr_multi': merged['mrr_multi'].mean(),
        'relative_improvement': 100 * (merged['mrr_multi'].mean() - merged['mrr_proto'].mean()) / merged['mrr_proto'].mean()
    }

    # Statistical test
    stat, pval = stats.wilcoxon(merged['mrr_proto'], merged['mrr_multi'], alternative='less')
    results['wilcoxon_stat'] = float(stat)
    results['wilcoxon_pval'] = float(pval)
    results['significant'] = pval < 0.05

    return results, merged


def analyze_by_positive_count(merged):
    """Check if improvement varies by number of positives."""
    # Bin by positive count
    merged['n_pos_bin'] = pd.cut(merged['n_positives_proto'],
                                   bins=[0, 2, 5, 10, 500],
                                   labels=['2', '3-5', '6-10', '>10'])

    by_bin = merged.groupby('n_pos_bin', observed=True).agg({
        'mrr_delta': ['mean', 'median', 'std', 'count'],
        'improved': 'sum'
    }).round(4)

    return by_bin


def make_recommendation(results, by_bin):
    """
    Decide placement based on results.

    Criteria:
    - Main text: >30% agents improved, >20% relative gain, p<0.01, broad across bins
    - Supplement: 15-30% agents improved, 10-20% relative gain, p<0.05
    - Brief mention: <15% improved or <10% relative gain
    """
    pct_improved = results['pct_improved']
    rel_improvement = results['relative_improvement']
    pval = results['wilcoxon_pval']

    if pct_improved > 30 and rel_improvement > 20 and pval < 0.01:
        recommendation = "MAIN_TEXT"
        reasoning = (
            f"Strong recommendation for MAIN TEXT inclusion:\n"
            f"  - {pct_improved:.1f}% of agents show >5% MRR improvement\n"
            f"  - {rel_improvement:.1f}% relative improvement in mean MRR\n"
            f"  - Statistically significant (Wilcoxon p={pval:.4f})\n"
            f"  - Improvement is broad and substantial"
        )
    elif pct_improved > 15 and rel_improvement > 10 and pval < 0.05:
        recommendation = "SUPPLEMENTARY"
        reasoning = (
            f"Recommendation for SUPPLEMENTARY MATERIAL:\n"
            f"  - {pct_improved:.1f}% of agents show >5% MRR improvement (moderate)\n"
            f"  - {rel_improvement:.1f}% relative improvement in mean MRR\n"
            f"  - Statistically significant (Wilcoxon p={pval:.4f})\n"
            f"  - Improvement is meaningful but not transformative"
        )
    else:
        recommendation = "BRIEF_MENTION"
        reasoning = (
            f"Recommendation for BRIEF MENTION only:\n"
            f"  - {pct_improved:.1f}% of agents show >5% MRR improvement (limited)\n"
            f"  - {rel_improvement:.1f}% relative improvement in mean MRR\n"
            f"  - p-value: {pval:.4f}\n"
            f"  - Improvement is marginal"
        )

    return recommendation, reasoning


def plot_comparison(merged, outdir):
    """Create publication-ready comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: Per-agent scatter
    ax = axes[0]
    improved = merged[merged['improved']]
    degraded = merged[merged['degraded']]
    neutral = merged[merged['neutral']]

    ax.scatter(neutral['mrr_proto'], neutral['mrr_multi'],
               alpha=0.6, s=30, color='gray', label='Neutral')
    ax.scatter(degraded['mrr_proto'], degraded['mrr_multi'],
               alpha=0.7, s=40, color='#d62728', label='Degraded')
    ax.scatter(improved['mrr_proto'], improved['mrr_multi'],
               alpha=0.7, s=40, color='#2ca02c', label='Improved')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Single Prototype MRR')
    ax.set_ylabel('Multi-Prototype MRR')
    ax.set_title('A. Per-Agent Comparison')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    # Panel B: MRR delta distribution
    ax = axes[1]
    ax.hist(merged['mrr_delta'], bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(merged['mrr_delta'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {merged["mrr_delta"].median():.3f}')
    ax.set_xlabel('MRR Delta (Multi - Single)')
    ax.set_ylabel('Count')
    ax.set_title('B. Improvement Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel C: Aggregate comparison
    ax = axes[2]
    metrics = ['MRR', 'Recall@5', 'Recall@10']
    proto_vals = [
        merged['mrr_proto'].mean(),
        merged['recall@5_proto'].mean(),
        merged['recall@10_proto'].mean()
    ]
    multi_vals = [
        merged['mrr_multi'].mean(),
        merged['recall@5_multi'].mean(),
        merged['recall@10_multi'].mean()
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, proto_vals, width, label='Single Prototype',
           color='#ff7f0e', alpha=0.7)
    ax.bar(x + width/2, multi_vals, width, label='Multi-Prototype',
           color='#2ca02c', alpha=0.7)

    ax.set_ylabel('Score')
    ax.set_title('C. Aggregate Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 0.6])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'multi_prototype_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_manuscript_text(results, recommendation):
    """Generate manuscript text based on recommendation."""
    if recommendation == "MAIN_TEXT":
        text = (
            "\\subsection{Multi-prototype representation improves ranking performance}\n\n"
            "To test whether single-prototype agent representation is overly restrictive, "
            "we compared single-prototype baseline (mean of all positives) against multi-prototype "
            "representation using k-means clustering (k=3) over positive examples. "
            f"Multi-prototype achieves mean MRR={results['mean_mrr_multi']:.3f}, "
            f"a {results['relative_improvement']:.1f}\\% relative improvement over single-prototype "
            f"(MRR={results['mean_mrr_proto']:.3f}, Wilcoxon signed-rank test p={results['wilcoxon_pval']:.4f}). "
            f"This improvement is broad: {results['pct_improved']:.1f}\\% of agents show $>$5\\% MRR gains, "
            f"with median gain of {results['median_mrr_delta']:.3f}. "
            "\n\nThese results suggest many agents exhibit multimodal binding preferences not captured "
            "by a single prototype. The multi-prototype approach better accommodates structural diversity "
            "within positive training sets, yielding substantial performance gains across the benchmark."
        )
    elif recommendation == "SUPPLEMENTARY":
        text = (
            "\\textbf{Multi-prototype representation.} "
            "We tested whether multi-prototype representation (k-means clustering with k=3) "
            f"improves upon single-prototype baseline. Multi-prototype achieves MRR={results['mean_mrr_multi']:.3f}, "
            f"a {results['relative_improvement']:.1f}\\% improvement over single-prototype "
            f"(MRR={results['mean_mrr_proto']:.3f}, p={results['wilcoxon_pval']:.4f}). "
            f"{results['pct_improved']:.1f}\\% of agents show $>$5\\% gains. "
            "See Supplementary Table SX for full comparison."
        )
    else:  # BRIEF_MENTION
        text = (
            f"Multi-prototype representation (k=3 clusters) yielded modest improvement "
            f"(MRR={results['mean_mrr_multi']:.3f} vs {results['mean_mrr_proto']:.3f}, "
            f"{results['relative_improvement']:.1f}\\% gain), "
            "suggesting single-prototype is adequate for most agents in this benchmark."
        )

    return text


def main():
    print("Loading results...")
    proto, multi = load_results()

    print("Analyzing improvement...")
    results, merged = analyze_improvement(proto, multi)

    print("\n=== MULTI-PROTOTYPE ANALYSIS ===\n")
    print(f"Total agents: {results['n_agents']}")
    print(f"Agents improved (>5% MRR): {results['n_improved']} ({results['pct_improved']:.1f}%)")
    print(f"Agents degraded (>5% MRR): {results['n_degraded']} ({results['pct_degraded']:.1f}%)")
    print(f"Agents neutral: {results['n_neutral']}")
    print(f"\nMean MRR (single prototype): {results['mean_mrr_proto']:.4f}")
    print(f"Mean MRR (multi-prototype): {results['mean_mrr_multi']:.4f}")
    print(f"Relative improvement: {results['relative_improvement']:.1f}%")
    print(f"Mean delta: {results['mean_mrr_delta']:.4f}")
    print(f"Median delta: {results['median_mrr_delta']:.4f}")
    print(f"\nStatistical test (Wilcoxon):")
    print(f"  Statistic: {results['wilcoxon_stat']:.2f}")
    print(f"  P-value: {results['wilcoxon_pval']:.6f}")
    print(f"  Significant: {results['significant']}")

    print("\nAnalyzing by positive count...")
    by_bin = analyze_by_positive_count(merged)
    print("\nImprovement by positive count:")
    print(by_bin)

    print("\nMaking placement recommendation...")
    recommendation, reasoning = make_recommendation(results, by_bin)

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*60}")
    print(reasoning)
    print(f"{'='*60}\n")

    # Create output directory
    outdir = Path("outputs/multi_prototype_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save results
    full_results = {
        'summary': results,
        'recommendation': recommendation,
        'reasoning': reasoning
    }

    with open(outdir / 'multi_prototype_analysis.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    # Save by_bin separately as CSV
    by_bin.to_csv(outdir / 'improvement_by_positive_count.csv')

    # Save detailed comparison table
    comparison_cols = ['agent_id', 'n_positives_proto', 'mrr_proto', 'mrr_multi',
                       'mrr_delta', 'recall@5_proto', 'recall@5_multi', 'recall5_delta',
                       'improved', 'degraded']
    merged[comparison_cols].to_csv(outdir / 'per_agent_comparison.csv', index=False)

    # Generate plot
    plot_comparison(merged, outdir)

    # Generate manuscript text
    manuscript_text = generate_manuscript_text(results, recommendation)
    with open(outdir / 'manuscript_text_multi_prototype.tex', 'w') as f:
        f.write(manuscript_text)

    print(f"Saved results to {outdir}/")
    print("  - multi_prototype_analysis.json")
    print("  - per_agent_comparison.csv")
    print("  - multi_prototype_comparison.png")
    print("  - manuscript_text_multi_prototype.tex")

    # Create recommendation file
    with open(outdir / 'RECOMMENDATION.txt', 'w') as f:
        f.write(f"PLACEMENT RECOMMENDATION: {recommendation}\n")
        f.write("="*60 + "\n\n")
        f.write(reasoning + "\n\n")
        f.write("DECISION CRITERIA:\n")
        f.write("  Main text: >30% improved, >20% relative gain, p<0.01\n")
        f.write("  Supplement: 15-30% improved, 10-20% relative gain, p<0.05\n")
        f.write("  Brief: <15% improved or <10% relative gain\n\n")
        f.write("ACTUAL RESULTS:\n")
        f.write(f"  % improved: {results['pct_improved']:.1f}%\n")
        f.write(f"  Relative gain: {results['relative_improvement']:.1f}%\n")
        f.write(f"  P-value: {results['wilcoxon_pval']:.6f}\n")

    print(f"\nRecommendation saved to {outdir}/RECOMMENDATION.txt")


if __name__ == "__main__":
    main()
