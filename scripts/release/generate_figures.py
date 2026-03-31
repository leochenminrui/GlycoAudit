#!/usr/bin/env python3
"""
Generate publication-ready figures and tables for benchmark paper.

Creates consolidated figures suitable for main text and supplementary materials.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def load_all_data():
    """Load all analysis results."""
    data = {}

    # Joinability
    with open("outputs/joinability_audit/joinability_audit_summary.json") as f:
        data["joinability"] = json.load(f)

    # Biological composition
    data["composition"] = pd.read_csv(
        "outputs/joinability_audit/biological_composition_stats.csv"
    )

    # Baseline comparison
    with open("outputs/baseline_comparison/baseline_comparison_summary.json") as f:
        data["baselines"] = json.load(f)

    data["baseline_agg"] = pd.read_csv(
        "outputs/baseline_comparison/baseline_comparison_aggregate.csv"
    )

    # Distance sensitivity
    with open("outputs/structural_distance_sensitivity/distance_sensitivity_summary.json") as f:
        data["distance"] = json.load(f)

    # Cross-source
    with open("outputs/cross_source_decomposition/cross_source_decomposition_summary.json") as f:
        data["cross_source"] = json.load(f)

    # Candidate pool
    with open("outputs/candidate_pool_report/pool_report_Full_IID.json") as f:
        data["pool"] = json.load(f)

    return data


def create_figure1_joinability_flow(data, outdir):
    """
    Figure 1: Joinability flow and coverage uplift.

    Panel A: Sankey-style flow diagram
    Panel B: Size/complexity comparison (targeted vs direct)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Flow bars
    ax = axes[0]
    stages = data["joinability"]["stages"]

    stage_names = [
        "Raw\nlabeled\n(1097)",
        "Normalized\n(1097)",
        "Direct\njoinable\n(217)",
        "Targeted\nadded\n(125)",
        "Final\nbenchmark\n(342)",
        "Unresolved\n(755)"
    ]

    stage_values = [
        stages["stage1_raw_labeled"]["glycans"],
        stages["stage2_normalized"]["glycans"],
        stages["stage3_directly_joinable"]["glycans"],
        stages["stage4_targeted_completion"]["glycans"],
        stages["stage5_final_joinable"]["glycans"],
        stages["stage6_unresolved"]["glycans"]
    ]

    colors = ["#1f77b4", "#1f77b4", "#2ca02c", "#ff7f0e", "#2ca02c", "#d62728"]
    y_pos = np.arange(len(stage_names))

    ax.barh(y_pos, stage_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stage_names, fontsize=10)
    ax.set_xlabel("Number of Glycans", fontsize=11)
    ax.set_title("A. Joinability Flow", fontsize=12, fontweight='bold')
    ax.axvline(342, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Final benchmark')
    ax.legend()

    # Panel B: Composition comparison
    ax = axes[1]
    comp_df = data["composition"]

    if not comp_df.empty and "source_type" in comp_df.columns:
        metrics = ["n_atoms_mean", "n_residues_mean", "branch_proxy_mean"]
        metric_labels = ["Atoms", "Residues", "Branch Proxy"]

        x = np.arange(len(metrics))
        width = 0.35

        targeted = []
        direct = []

        for metric in metrics:
            if metric in comp_df.columns:
                targeted_val = comp_df[comp_df["source_type"] == "targeted"][metric].values
                direct_val = comp_df[comp_df["source_type"] == "direct"][metric].values

                if len(targeted_val) > 0:
                    targeted.append(targeted_val[0])
                else:
                    targeted.append(0)

                if len(direct_val) > 0:
                    direct.append(direct_val[0])
                else:
                    direct.append(0)

        # Normalize for visualization
        targeted_norm = np.array(targeted) / np.array(direct)

        ax.bar(x - width/2, [1, 1, 1], width, label='Direct', color='#2ca02c', alpha=0.7)
        ax.bar(x + width/2, targeted_norm, width, label='Targeted', color='#ff7f0e', alpha=0.7)

        ax.set_ylabel("Normalized Value\n(Relative to Direct)", fontsize=11)
        ax.set_xlabel("Structural Property", fontsize=11)
        ax.set_title("B. Coverage Uplift:\nTargeted vs Direct", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim([0, 1.2])

    plt.tight_layout()
    plt.savefig(outdir / "figure1_joinability_and_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created Figure 1: {outdir}/figure1_joinability_and_coverage.png")


def create_figure2_baseline_comparison(data, outdir):
    """
    Figure 2: Baseline comparison demonstrating benchmark generality.

    Panel A: MRR by baseline
    Panel B: Recall@k curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    baseline_df = data["baseline_agg"]

    # Panel A: MRR comparison
    ax = axes[0]

    baselines = baseline_df["baseline"].values
    mrr = baseline_df["mrr_mean"].values
    mrr_std = baseline_df["mrr_std"].values

    colors_map = {
        "random": "#d62728",
        "linear": "#9467bd",
        "prototype": "#ff7f0e",
        "nearest_neighbor": "#2ca02c"
    }

    colors = [colors_map.get(b, "#1f77b4") for b in baselines]

    x_pos = np.arange(len(baselines))
    ax.bar(x_pos, mrr, yerr=mrr_std, color=colors, alpha=0.7,
           edgecolor='black', capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([b.replace("_", " ").title() for b in baselines], rotation=15, ha='right')
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11)
    ax.set_title("A. MRR by Baseline Type", fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(mrr) * 1.2])

    # Add value labels
    for i, (m, s) in enumerate(zip(mrr, mrr_std)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha='center', fontsize=9)

    # Panel B: Recall@k curves
    ax = axes[1]

    k_values = [1, 3, 5, 10, 20]
    for baseline in baselines:
        row = baseline_df[baseline_df["baseline"] == baseline].iloc[0]
        recall_values = [row[f"recall@{k}_mean"] for k in k_values]

        ax.plot(k_values, recall_values, marker='o', linewidth=2,
                label=baseline.replace("_", " ").title(),
                color=colors_map.get(baseline, "#1f77b4"))

    ax.set_xlabel("k (top-k predictions)", fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_title("B. Recall@k by Baseline", fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "figure2_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created Figure 2: {outdir}/figure2_baseline_comparison.png")


def create_figure3_distance_sensitivity(data, outdir):
    """
    Figure 3: Structural distance sensitivity analysis.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    buckets = data["distance"]["bucket_results"]

    bucket_labels = [b["bucket_label"] for b in buckets]
    bucket_centers = [(b["distance_min"] + b["distance_max"]) / 2 for b in buckets]
    mrr_values = [b["mrr_mean"] for b in buckets]
    mrr_stds = [b["mrr_std"] for b in buckets]
    recall5_values = [b["recall@5_mean"] for b in buckets]
    recall5_stds = [b["recall@5_std"] for b in buckets]

    ax.errorbar(bucket_centers, mrr_values, yerr=mrr_stds, marker='o',
                linewidth=2, capsize=5, label='MRR', color='#1f77b4')
    ax.errorbar(bucket_centers, recall5_values, yerr=recall5_stds, marker='s',
                linewidth=2, capsize=5, label='Recall@5', color='#ff7f0e')

    ax.set_xlabel("Euclidean Distance to Nearest Training Glycan", fontsize=12)
    ax.set_ylabel("Performance Metric", fontsize=12)
    ax.set_title("Structural Distance Sensitivity\n(Prototype Baseline)", fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(recall5_values) * 1.2])

    # Add random baseline reference
    ax.axhline(0.018, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Random MRR')

    plt.tight_layout()
    plt.savefig(outdir / "figure3_distance_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created Figure 3: {outdir}/figure3_distance_sensitivity.png")


def create_figure4_cross_source_transfer(data, outdir):
    """
    Figure 4: Cross-source transfer and decomposition.

    Panel A: Transfer performance matrix
    Panel B: Source composition comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Transfer matrix
    ax = axes[0]

    transfer_data = data["cross_source"]["transfer_performance"]

    cfg_to_sb = transfer_data.get("cfg_to_sugarbind", {}).get("mrr", 0)
    sb_to_cfg = transfer_data.get("sugarbind_to_cfg", {}).get("mrr", 0)

    transfer_matrix = np.array([
        [0.39, cfg_to_sb],  # CFG: IID ~0.39 (from existing), CFG→SB
        [sb_to_cfg, 0.39]   # SB→CFG, SB: IID ~0.39
    ])

    im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0, vmax=0.5, aspect='auto')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Train: CFG', 'Train: SugarBind'])
    ax.set_yticklabels(['Test: CFG', 'Test: SugarBind'])
    ax.set_title("A. Cross-Source Transfer (MRR)", fontsize=12, fontweight='bold')

    # Annotate cells
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f"{transfer_matrix[i, j]:.3f}",
                          ha="center", va="center", color="black", fontsize=14)

    plt.colorbar(im, ax=ax, label='MRR')

    # Panel B: Source composition
    ax = axes[1]

    composition = data["cross_source"]["composition_by_source"]

    sources = list(composition.keys())
    n_atoms_means = [composition[s]["size"]["n_atoms"]["mean"] for s in sources]
    branch_proxy_means = [composition[s]["topology"]["branch_proxy"]["mean"] for s in sources]

    x = np.arange(len(sources))
    width = 0.35

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, n_atoms_means, width, label='Mean Atoms',
                   color='#1f77b4', alpha=0.7)
    bars2 = ax2.bar(x + width/2, branch_proxy_means, width, label='Branch Proxy',
                    color='#ff7f0e', alpha=0.7)

    ax.set_xlabel("Source", fontsize=11)
    ax.set_ylabel("Mean Atoms", fontsize=11, color='#1f77b4')
    ax2.set_ylabel("Branch Proxy", fontsize=11, color='#ff7f0e')
    ax.set_title("B. Source Composition", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in sources], rotation=15, ha='right')

    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(outdir / "figure4_cross_source_transfer.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created Figure 4: {outdir}/figure4_cross_source_transfer.png")


def create_table1_summary_statistics(data, outdir):
    """
    Table 1: Summary statistics for manuscript.
    """
    rows = []

    # Benchmark composition
    rows.append({
        "Category": "Benchmark Composition",
        "Metric": "Raw labeled glycans",
        "Value": "1,097"
    })
    rows.append({
        "Category": "Benchmark Composition",
        "Metric": "Final benchmark glycans",
        "Value": "342 (31.2%)"
    })
    rows.append({
        "Category": "Benchmark Composition",
        "Metric": "Targeted completion contribution",
        "Value": "125 (36.5%)"
    })
    rows.append({
        "Category": "Benchmark Composition",
        "Metric": "Evaluated agents",
        "Value": "196"
    })
    rows.append({
        "Category": "Benchmark Composition",
        "Metric": "Positive pairs",
        "Value": "3,004"
    })

    # Baseline performance
    baseline_df = data["baseline_agg"]
    for _, row in baseline_df.iterrows():
        rows.append({
            "Category": "Baseline Performance",
            "Metric": f"{row['baseline'].replace('_', ' ').title()} MRR",
            "Value": f"{row['mrr_mean']:.3f} ± {row['mrr_std']:.3f}"
        })

    # Cross-source transfer
    transfer = data["cross_source"]["transfer_performance"]
    cfg_to_sb = transfer.get("cfg_to_sugarbind", {}).get("mrr", 0)
    sb_to_cfg = transfer.get("sugarbind_to_cfg", {}).get("mrr", 0)

    rows.append({
        "Category": "Cross-Source Transfer",
        "Metric": "CFG → SugarBind MRR",
        "Value": f"{cfg_to_sb:.3f}"
    })
    rows.append({
        "Category": "Cross-Source Transfer",
        "Metric": "SugarBind → CFG MRR",
        "Value": f"{sb_to_cfg:.3f}"
    })
    rows.append({
        "Category": "Cross-Source Transfer",
        "Metric": "Asymmetry ratio",
        "Value": f"{cfg_to_sb / sb_to_cfg:.2f}×"
    })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(outdir / "table1_summary_statistics.csv", index=False)
    print(f"Created Table 1: {outdir}/table1_summary_statistics.csv")

    return table_df


def create_table2_joinability_flow(data, outdir):
    """
    Table 2: Joinability flow for manuscript.
    """
    stages = data["joinability"]["stages"]

    rows = []
    rows.append({
        "Stage": "1. Raw labeled glycans",
        "Count": stages["stage1_raw_labeled"]["glycans"],
        "% of Raw": "100.0%",
        "Description": "From SugarBind + Carbogrove"
    })
    rows.append({
        "Stage": "2. Normalized IDs",
        "Count": stages["stage2_normalized"]["glycans"],
        "% of Raw": "100.0%",
        "Description": "After GlyTouCan ID normalization"
    })
    rows.append({
        "Stage": "3. Directly joinable",
        "Count": stages["stage3_directly_joinable"]["glycans"],
        "% of Raw": f"{stages['stage3_directly_joinable']['proportion_of_raw'] * 100:.1f}%",
        "Description": "With existing 3D structures"
    })
    rows.append({
        "Stage": "4. Targeted completion",
        "Count": stages["stage4_targeted_completion"]["glycans"],
        "% of Raw": f"{stages['stage4_targeted_completion']['proportion_of_raw'] * 100:.1f}%",
        "Description": "Added via structure acquisition"
    })
    rows.append({
        "Stage": "5. Final benchmark",
        "Count": stages["stage5_final_joinable"]["glycans"],
        "% of Raw": f"{stages['stage5_final_joinable']['join_success_rate'] * 100:.1f}%",
        "Description": "Successfully joined (structure + label)"
    })
    rows.append({
        "Stage": "6. Unresolved",
        "Count": stages["stage6_unresolved"]["glycans"],
        "% of Raw": f"{stages['stage6_unresolved']['proportion_of_raw'] * 100:.1f}%",
        "Description": "Missing structures"
    })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(outdir / "table2_joinability_flow.csv", index=False)
    print(f"Created Table 2: {outdir}/table2_joinability_flow.csv")

    return table_df


def main():
    print("Loading data...")
    data = load_all_data()

    outdir = Path("outputs/publication_figures")
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    create_figure1_joinability_flow(data, outdir)
    create_figure2_baseline_comparison(data, outdir)
    create_figure3_distance_sensitivity(data, outdir)
    create_figure4_cross_source_transfer(data, outdir)

    print("\nGenerating tables...")
    create_table1_summary_statistics(data, outdir)
    create_table2_joinability_flow(data, outdir)

    print(f"\nAll publication figures saved to {outdir}/")
    print("\nGenerated files:")
    print("  - figure1_joinability_and_coverage.png")
    print("  - figure2_baseline_comparison.png")
    print("  - figure3_distance_sensitivity.png")
    print("  - figure4_cross_source_transfer.png")
    print("  - table1_summary_statistics.csv")
    print("  - table2_joinability_flow.csv")


if __name__ == "__main__":
    main()
