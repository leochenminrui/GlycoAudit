#!/usr/bin/env python3
"""
Cross-source transfer decomposition analysis.

Compares CFG and SugarBind sources on:
- Glycan size and structural composition
- Agent overlap
- Annotation density
- Motif distributions (basic)
- Transfer performance breakdown

Provides deeper context for the observed asymmetric transfer.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data():
    """Load required datasets."""
    # Features (SSV only for now, can add GCV if needed)
    ssv = pd.read_csv("data/ssv/expanded_v1/ssv_features.csv")

    # Labels (includes data_source)
    labels = pd.read_csv("data/binding/expanded_v1/labels.csv")

    # Agent metadata
    agent_meta = pd.read_csv("data/binding/expanded_v1/agent_meta.csv")

    # Source is in ssv_features.csv source_dir column
    # Extract data source from source_dir (targeted_sugarbind or carbogrove_glycoshape)
    ssv["data_source"] = ssv["source_dir"]

    return ssv, labels, agent_meta


def analyze_source_composition(ssv):
    """Analyze structural composition by source."""
    df = ssv.copy()

    # Structural features
    size_features = ["n_atoms", "n_residues"]
    geometry_features = ["radius_of_gyration", "max_pair_distance", "compactness"]
    topology_features = ["branch_proxy", "terminal_proxy"]

    results = {}
    for source in df["data_source"].dropna().unique():
        source_df = df[df["data_source"] == source]

        stats_dict = {
            "n_glycans": len(source_df),
            "size": {},
            "geometry": {},
            "topology": {}
        }

        # Size
        for feat in size_features:
            stats_dict["size"][feat] = {
                "mean": float(source_df[feat].mean()),
                "std": float(source_df[feat].std()),
                "median": float(source_df[feat].median())
            }

        # Geometry
        for feat in geometry_features:
            stats_dict["geometry"][feat] = {
                "mean": float(source_df[feat].mean()),
                "std": float(source_df[feat].std()),
                "median": float(source_df[feat].median())
            }

        # Topology
        for feat in topology_features:
            stats_dict["topology"][feat] = {
                "mean": float(source_df[feat].mean()),
                "std": float(source_df[feat].std()),
                "median": float(source_df[feat].median())
            }

        results[source] = stats_dict

    # Statistical comparisons
    comparisons = {}
    sources = list(results.keys())
    if len(sources) >= 2:
        source_a = sources[0]
        source_b = sources[1]

        df_a = df[df["data_source"] == source_a]
        df_b = df[df["data_source"] == source_b]

        for feat in size_features + geometry_features + topology_features:
            try:
                stat, pval = stats.mannwhitneyu(df_a[feat].dropna(),
                                                 df_b[feat].dropna(),
                                                 alternative="two-sided")
                comparisons[feat] = {
                    "test": "Mann-Whitney U",
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant": bool(pval < 0.05),
                    "source_a": str(source_a),
                    "source_b": str(source_b),
                    "mean_a": float(df_a[feat].mean()),
                    "mean_b": float(df_b[feat].mean())
                }
            except Exception as e:
                comparisons[feat] = {"error": str(e)}

    return results, comparisons, df


def analyze_agent_overlap(labels, agent_meta):
    """Analyze agent overlap between sources."""
    # Use data_source column from labels (binding data source)
    labels_with_source = labels.copy()

    agents_by_source = labels_with_source.groupby("data_source")["agent_id"].nunique()

    # Find shared agents
    source_agents = {}
    for source in labels_with_source["data_source"].unique():
        source_agents[source] = set(labels_with_source[
            labels_with_source["data_source"] == source
        ]["agent_id"].unique())

    overlap_stats = {
        "agents_per_source": agents_by_source.to_dict(),
        "total_unique_agents": len(labels["agent_id"].unique())
    }

    # Pairwise overlaps
    sources = list(source_agents.keys())
    if len(sources) >= 2:
        for i, source_a in enumerate(sources):
            for source_b in sources[i+1:]:
                shared = source_agents[source_a] & source_agents[source_b]
                overlap_stats[f"shared_{source_a}_{source_b}"] = len(shared)
                overlap_stats[f"jaccard_{source_a}_{source_b}"] = (
                    len(shared) / len(source_agents[source_a] | source_agents[source_b])
                    if len(source_agents[source_a] | source_agents[source_b]) > 0 else 0.0
                )

    return overlap_stats


def analyze_annotation_density(labels, agent_meta, ssv):
    """Analyze annotation density by source."""
    # Merge to get source for agents (from agent_meta)
    # Note: agent_meta has agent_type, not data_source usually
    # Labels already have data_source column

    # Get glycan source from SSV
    glycan_source = ssv[["glytoucan_id", "data_source"]].drop_duplicates()

    labels_with_glycan_source = labels.merge(
        glycan_source,
        on="glytoucan_id",
        how="left",
        suffixes=("_label", "_glycan")
    )

    # Use data_source from labels (this is the binding data source)
    labels_glycan_source = labels_with_glycan_source

    # Count by source
    results = {}

    # Binding label source (from data_source_label)
    label_density = labels_glycan_source.groupby("data_source_label").agg({
        "glytoucan_id": "nunique",
        "agent_id": "nunique"
    }).rename(columns={
        "glytoucan_id": "unique_glycans",
        "agent_id": "unique_agents"
    })
    label_density["pairs"] = labels_glycan_source.groupby("data_source_label").size()
    label_density["pairs_per_agent"] = label_density["pairs"] / label_density["unique_agents"]
    label_density["pairs_per_glycan"] = label_density["pairs"] / label_density["unique_glycans"]

    results["by_label_source"] = label_density.to_dict(orient="index")

    # Glycan source (from structure source)
    glycan_density = labels_glycan_source.groupby("data_source_glycan").agg({
        "glytoucan_id": "nunique",
        "agent_id": "nunique"
    }).rename(columns={
        "glytoucan_id": "unique_glycans",
        "agent_id": "unique_agents"
    })
    glycan_density["pairs"] = labels_glycan_source.groupby("data_source_glycan").size()
    glycan_density["pairs_per_agent"] = glycan_density["pairs"] / glycan_density["unique_agents"]
    glycan_density["pairs_per_glycan"] = glycan_density["pairs"] / glycan_density["unique_glycans"]

    results["by_glycan_source"] = glycan_density.to_dict(orient="index")

    return results


def load_cross_source_results():
    """Load existing cross-source transfer results."""
    base_dir = Path("outputs/bench_v2_1_minorrev_full")

    results = {}

    # CFG to SugarBind
    cfg_to_sb_file = base_dir / "cross_source_cfg_to_sugarbind" / "aggregate_metrics.json"
    if cfg_to_sb_file.exists():
        with open(cfg_to_sb_file) as f:
            results["cfg_to_sugarbind"] = json.load(f)

    # SugarBind to CFG
    sb_to_cfg_file = base_dir / "cross_source_sugarbind_to_cfg" / "aggregate_metrics.json"
    if sb_to_cfg_file.exists():
        with open(sb_to_cfg_file) as f:
            results["sugarbind_to_cfg"] = json.load(f)

    # Bootstrap CIs
    cfg_to_sb_ci_file = base_dir / "cross_source_cfg_to_sugarbind" / "bootstrap_cis.json"
    if cfg_to_sb_ci_file.exists():
        with open(cfg_to_sb_ci_file) as f:
            results["cfg_to_sugarbind_ci"] = json.load(f)

    sb_to_cfg_ci_file = base_dir / "cross_source_sugarbind_to_cfg" / "bootstrap_cis.json"
    if sb_to_cfg_ci_file.exists():
        with open(sb_to_cfg_ci_file) as f:
            results["sugarbind_to_cfg_ci"] = json.load(f)

    return results


def plot_source_comparison(composition_df, outdir):
    """Create comparison plots for sources."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Size comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    features_to_plot = [
        ("n_atoms", "Number of Atoms"),
        ("n_residues", "Number of Residues"),
        ("branch_proxy", "Branch Proxy")
    ]

    for ax, (feat, label) in zip(axes, features_to_plot):
        for source in composition_df["data_source"].unique():
            source_data = composition_df[composition_df["data_source"] == source]
            ax.hist(source_data[feat], alpha=0.6, label=source, bins=20)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"{label} by Source")

    plt.tight_layout()
    plt.savefig(outdir / "source_structural_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Loading data...")
    ssv, labels, agent_meta = load_data()

    print("Analyzing source composition...")
    composition_results, composition_comparisons, composition_df = analyze_source_composition(ssv)

    print("Analyzing agent overlap...")
    overlap_results = analyze_agent_overlap(labels, agent_meta)

    print("Analyzing annotation density...")
    density_results = analyze_annotation_density(labels, agent_meta, ssv)

    print("Loading existing transfer results...")
    transfer_results = load_cross_source_results()

    # Create output directory
    outdir = Path("outputs/cross_source_decomposition")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save results
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "composition_by_source": composition_results,
        "composition_statistical_comparisons": composition_comparisons,
        "agent_overlap": overlap_results,
        "annotation_density": density_results,
        "transfer_performance": transfer_results
    }

    with open(outdir / "cross_source_decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {outdir}/cross_source_decomposition_summary.json")

    # Create comparison table
    comparison_rows = []
    for feat, comp in composition_comparisons.items():
        if "error" not in comp:
            comparison_rows.append({
                "feature": feat,
                "source_a": comp["source_a"],
                "mean_a": comp["mean_a"],
                "source_b": comp["source_b"],
                "mean_b": comp["mean_b"],
                "p_value": comp["p_value"],
                "significant": comp["significant"]
            })

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(outdir / "source_comparison_table.csv", index=False)
        print(f"Saved comparison table to {outdir}/source_comparison_table.csv")

    # Plot
    if "data_source" in composition_df.columns:
        print("Creating plots...")
        plot_source_comparison(composition_df, outdir)
        print(f"Saved plots to {outdir}/")

    print("\n=== Cross-Source Decomposition Summary ===")
    print(f"\nSource Composition:")
    for source, stats in composition_results.items():
        print(f"\n{source}:")
        print(f"  N glycans: {stats['n_glycans']}")
        print(f"  Mean atoms: {stats['size']['n_atoms']['mean']:.1f}")
        print(f"  Mean residues: {stats['size']['n_residues']['mean']:.1f}")
        print(f"  Mean branch_proxy: {stats['topology']['branch_proxy']['mean']:.3f}")

    print(f"\nAgent Overlap:")
    for k, v in overlap_results.items():
        print(f"  {k}: {v}")

    print(f"\nAnnotation Density (by label source):")
    for source, stats in density_results["by_label_source"].items():
        print(f"\n{source}:")
        print(f"  Pairs per agent: {stats['pairs_per_agent']:.1f}")
        print(f"  Pairs per glycan: {stats['pairs_per_glycan']:.1f}")

    if transfer_results:
        print(f"\nTransfer Performance:")
        if "cfg_to_sugarbind" in transfer_results:
            mrr = transfer_results["cfg_to_sugarbind"].get("mrr", "N/A")
            print(f"  CFG → SugarBind: MRR = {mrr}")
        if "sugarbind_to_cfg" in transfer_results:
            mrr = transfer_results["sugarbind_to_cfg"].get("mrr", "N/A")
            print(f"  SugarBind → CFG: MRR = {mrr}")

    print("\nDone!")


if __name__ == "__main__":
    main()
