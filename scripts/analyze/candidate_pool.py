#!/usr/bin/env python3
"""
Candidate Pool Diagnostics: IID vs Scaffold Split Comparison

Analyzes whether IID vs scaffold split performance differences stem from:
1. Candidate pool size differences
2. Positive density shifts (positives per agent)
3. Source composition changes (CFG vs SugarBind)
4. Nearest-train distance distributions
5. Performance vs structural distance relationships

This is a critical analysis for T1-1 in the major revision roadmap.

Usage:
    python scripts/analyze_candidate_pool_diagnostics.py \\
        --benchmark_dir outputs/bench_v2_1_minorrev_full \\
        --iid_split iid__ssv \\
        --scaffold_split scaffold_holdout_true__ssv \\
        --output_dir outputs/candidate_pool_diagnostics

Author: Major Revision Team
Date: 2026-03-31
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_split_manifest(benchmark_dir: Path, split_name: str) -> dict:
    """Load split manifest JSON."""
    manifest_path = benchmark_dir / split_name / "split_manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_full_data(data_dir: Path = Path("data/joined/ssv_binding_targeted_v0")) -> pd.DataFrame:
    """Load full joined dataset."""
    return pd.read_csv(data_dir / "full_joined.csv")


def compute_candidate_pool_stats(split_manifest: dict, full_data: pd.DataFrame) -> dict:
    """
    Extract candidate pool statistics from split manifest.

    Returns statistics on:
    - Number of test glycans
    - Number of agents evaluated
    - Positives per agent distribution
    - Source composition in train vs test
    """
    # Handle both index-based and glycan-list-based manifests
    if "test_indices" in split_manifest:
        test_indices = split_manifest["test_indices"]
        train_indices = split_manifest["train_indices"]
        test_data = full_data.iloc[test_indices].copy()
        train_data = full_data.iloc[train_indices].copy()
    elif "test_glycans" in split_manifest:
        test_glycans = split_manifest["test_glycans"]
        train_glycans = split_manifest["train_glycans"]
        test_data = full_data[full_data["glytoucan_id"].isin(test_glycans)].copy()
        train_data = full_data[full_data["glytoucan_id"].isin(train_glycans)].copy()
    else:
        raise KeyError("Split manifest must contain either 'test_indices' or 'test_glycans'")

    # Test set statistics
    n_test_glycans = test_data["glytoucan_id"].nunique()
    n_test_agents = test_data["agent_id"].nunique()

    # Positives per agent in test set
    test_positives = test_data[test_data["label"] == 1].groupby("agent_id").size()
    positives_stats = {
        "mean": float(test_positives.mean()) if len(test_positives) > 0 else 0.0,
        "median": float(test_positives.median()) if len(test_positives) > 0 else 0.0,
        "std": float(test_positives.std()) if len(test_positives) > 0 else 0.0,
        "min": int(test_positives.min()) if len(test_positives) > 0 else 0,
        "max": int(test_positives.max()) if len(test_positives) > 0 else 0
    }

    # Source composition
    if "source" in test_data.columns:
        source_test = test_data["source"].value_counts(normalize=True).to_dict()
        source_train = train_data["source"].value_counts(normalize=True).to_dict()
    else:
        source_test = {}
        source_train = {}

    return {
        "n_test_glycans": n_test_glycans,
        "n_test_agents": n_test_agents,
        "n_train_glycans": train_data["glytoucan_id"].nunique(),
        "n_train_agents": train_data["agent_id"].nunique(),
        "positives_per_agent": positives_stats,
        "source_composition_test": source_test,
        "source_composition_train": source_train
    }


def compute_nearest_train_distances(
    train_features: np.ndarray,
    test_features: np.ndarray,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    For each test sample, compute distance to nearest train sample.

    Args:
        train_features: (n_train, n_features)
        test_features: (n_test, n_features)
        metric: "euclidean", "cityblock", or "cosine"

    Returns:
        distances: (n_test,) array of nearest-neighbor distances
    """
    if len(train_features) == 0 or len(test_features) == 0:
        return np.array([])

    dist_matrix = cdist(test_features, train_features, metric=metric)
    nearest_distances = dist_matrix.min(axis=1)

    return nearest_distances


def compare_distance_distributions(
    distances_iid: np.ndarray,
    distances_scaffold: np.ndarray
) -> dict:
    """
    Statistical comparison of nearest-train distance distributions.

    Uses Mann-Whitney U test to test if scaffold test set is farther from train.
    """
    if len(distances_iid) == 0 or len(distances_scaffold) == 0:
        return {}

    u_stat, p_value = stats.mannwhitneyu(
        distances_scaffold, distances_iid, alternative="greater"
    )

    return {
        "iid_mean": float(np.mean(distances_iid)),
        "iid_median": float(np.median(distances_iid)),
        "iid_std": float(np.std(distances_iid)),
        "scaffold_mean": float(np.mean(distances_scaffold)),
        "scaffold_median": float(np.median(distances_scaffold)),
        "scaffold_std": float(np.std(distances_scaffold)),
        "mann_whitney_u": float(u_stat),
        "p_value": float(p_value),
        "interpretation": (
            "Scaffold test set is significantly farther from train"
            if p_value < 0.05
            else "No significant distance difference"
        )
    }


def bin_performance_by_distance(
    test_data: pd.DataFrame,
    distances: np.ndarray,
    per_agent_metrics: pd.DataFrame,
    n_bins: int = 4
) -> pd.DataFrame:
    """
    Bin test glycans by nearest-train distance and compute aggregate performance per bin.

    Note: This is a simplified version that bins by distance and reports
    aggregate metrics. A full version would re-evaluate per-agent rankings within each bin.

    Returns:
        bin_stats: DataFrame with [bin_id, distance_min, distance_max, bin_center,
                                    n_glycans, n_agents, mean_mrr, mean_recall@5]
    """
    if len(distances) == 0:
        return pd.DataFrame()

    test_data = test_data.copy()

    # Assign distances to test data (one per unique glycan)
    unique_glycans = test_data[["glytoucan_id"]].drop_duplicates().reset_index(drop=True)
    if len(unique_glycans) != len(distances):
        logger.warning(
            f"Mismatch: {len(unique_glycans)} unique glycans, {len(distances)} distances. "
            "Using first N distances."
        )
        distances = distances[:len(unique_glycans)]

    unique_glycans["distance"] = distances

    # Merge back to full test_data
    test_data = test_data.merge(unique_glycans, on="glytoucan_id", how="left")

    # Quantile binning
    try:
        test_data["distance_bin"] = pd.qcut(test_data["distance"], q=n_bins, labels=False, duplicates="drop")
    except ValueError as e:
        logger.warning(f"Binning failed: {e}. Using uniform bins.")
        test_data["distance_bin"] = pd.cut(test_data["distance"], bins=n_bins, labels=False)

    # Aggregate stats per bin
    bin_stats = []
    for bin_id in sorted(test_data["distance_bin"].dropna().unique()):
        bin_data = test_data[test_data["distance_bin"] == bin_id]

        n_glycans = bin_data["glytoucan_id"].nunique()
        n_agents = bin_data["agent_id"].nunique()

        # For simplicity, report average per-agent MRR for agents with glycans in this bin
        # (A full implementation would re-rank candidates per agent within this bin)
        agents_in_bin = bin_data["agent_id"].unique()
        bin_metrics = per_agent_metrics[per_agent_metrics["agent_id"].isin(agents_in_bin)]

        bin_stats.append({
            "bin_id": int(bin_id),
            "distance_min": float(bin_data["distance"].min()),
            "distance_max": float(bin_data["distance"].max()),
            "bin_center": float(bin_data["distance"].median()),
            "n_glycans": n_glycans,
            "n_agents": n_agents,
            "mean_mrr": float(bin_metrics["mrr"].mean()) if len(bin_metrics) > 0 else np.nan,
            "mean_recall@5": float(bin_metrics["recall@5"].mean()) if len(bin_metrics) > 0 else np.nan
        })

    return pd.DataFrame(bin_stats)


def plot_performance_vs_distance(
    iid_binned: pd.DataFrame,
    scaffold_binned: pd.DataFrame,
    output_path: Path
):
    """
    Plot performance (MRR) vs nearest-train distance for IID and scaffold splits.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if not iid_binned.empty:
        ax.plot(
            iid_binned["bin_center"], iid_binned["mean_mrr"],
            marker='o', linestyle='-', linewidth=2, markersize=8,
            label="IID split", color="C0"
        )

    if not scaffold_binned.empty:
        ax.plot(
            scaffold_binned["bin_center"], scaffold_binned["mean_mrr"],
            marker='s', linestyle='-', linewidth=2, markersize=8,
            label="Scaffold split", color="C1"
        )

    ax.set_xlabel("Nearest-train distance (bin center)", fontsize=12)
    ax.set_ylabel("Mean MRR", fontsize=12)
    ax.set_title("Performance vs Structural Distance from Training Set", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved performance vs distance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Candidate pool diagnostics for IID vs scaffold splits")
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=Path("outputs/bench_v2_1_minorrev_full"),
        help="Benchmark output directory"
    )
    parser.add_argument(
        "--iid_split",
        type=str,
        default="iid__ssv",
        help="IID split subdirectory name"
    )
    parser.add_argument(
        "--scaffold_split",
        type=str,
        default="scaffold_holdout_true__ssv",
        help="Scaffold split subdirectory name"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/joined/ssv_binding_targeted_v0"),
        help="Path to full joined dataset"
    )
    parser.add_argument(
        "--feature_prefix",
        type=str,
        default="ssv_",
        help="Prefix of feature columns"
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine", "cityblock"],
        help="Distance metric for nearest-train calculation"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/candidate_pool_diagnostics"),
        help="Output directory"
    )
    parser.add_argument(
        "--n_distance_bins",
        type=int,
        default=4,
        help="Number of bins for distance-binned performance"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CANDIDATE POOL DIAGNOSTICS: IID vs Scaffold Split")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading full dataset from {args.data_dir}")
    full_data = load_full_data(args.data_dir)

    logger.info(f"Loading IID split manifest from {args.benchmark_dir / args.iid_split}")
    iid_manifest = load_split_manifest(args.benchmark_dir, args.iid_split)

    logger.info(f"Loading scaffold split manifest from {args.benchmark_dir / args.scaffold_split}")
    scaffold_manifest = load_split_manifest(args.benchmark_dir, args.scaffold_split)

    # === ANALYSIS 1: Candidate pool statistics ===
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS 1: Candidate Pool Statistics")
    logger.info("=" * 80)

    iid_stats = compute_candidate_pool_stats(iid_manifest, full_data)
    scaffold_stats = compute_candidate_pool_stats(scaffold_manifest, full_data)

    pool_comparison = {
        "iid": iid_stats,
        "scaffold": scaffold_stats
    }

    # Create comparison table
    pool_table = pd.DataFrame({
        "Metric": [
            "n_test_glycans",
            "n_test_agents",
            "positives_per_agent_mean",
            "positives_per_agent_median",
            "positives_per_agent_std"
        ],
        "IID": [
            iid_stats["n_test_glycans"],
            iid_stats["n_test_agents"],
            iid_stats["positives_per_agent"]["mean"],
            iid_stats["positives_per_agent"]["median"],
            iid_stats["positives_per_agent"]["std"]
        ],
        "Scaffold": [
            scaffold_stats["n_test_glycans"],
            scaffold_stats["n_test_agents"],
            scaffold_stats["positives_per_agent"]["mean"],
            scaffold_stats["positives_per_agent"]["median"],
            scaffold_stats["positives_per_agent"]["std"]
        ]
    })
    pool_table["Difference"] = pool_table["Scaffold"] - pool_table["IID"]

    logger.info("\nCandidate Pool Comparison:")
    logger.info("\n" + pool_table.to_string(index=False))

    pool_table.to_csv(args.output_dir / "candidate_pool_comparison.csv", index=False)

    # === ANALYSIS 2: Source composition shift ===
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS 2: Source Composition Shift")
    logger.info("=" * 80)

    source_composition = []
    for split_name, stats in [("IID", iid_stats), ("Scaffold", scaffold_stats)]:
        for source, proportion in stats["source_composition_test"].items():
            source_composition.append({
                "split": split_name,
                "split_subset": "test",
                "source": source,
                "proportion": proportion
            })

    source_df = pd.DataFrame(source_composition)
    if not source_df.empty:
        logger.info("\n" + source_df.to_string(index=False))
        source_df.to_csv(args.output_dir / "source_composition_shift.csv", index=False)
    else:
        logger.warning("No source composition data available")

    # === ANALYSIS 3: Nearest-train distance distribution ===
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS 3: Nearest-Train Distance Distribution")
    logger.info("=" * 80)

    # Extract feature columns
    feature_cols = [col for col in full_data.columns if col.startswith(args.feature_prefix)]
    if not feature_cols:
        logger.error(f"No feature columns found with prefix '{args.feature_prefix}'")
        return

    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

    # Get feature matrices
    train_iid = full_data.iloc[iid_manifest["train_indices"]]
    test_iid = full_data.iloc[iid_manifest["test_indices"]]
    train_scaffold = full_data.iloc[scaffold_manifest["train_indices"]]
    test_scaffold = full_data.iloc[scaffold_manifest["test_indices"]]

    # Aggregate features by glycan (mean over all pairs)
    def aggregate_by_glycan(data, feature_cols):
        return data.groupby("glytoucan_id")[feature_cols].mean().reset_index()

    train_iid_agg = aggregate_by_glycan(train_iid, feature_cols)
    test_iid_agg = aggregate_by_glycan(test_iid, feature_cols)
    train_scaffold_agg = aggregate_by_glycan(train_scaffold, feature_cols)
    test_scaffold_agg = aggregate_by_glycan(test_scaffold, feature_cols)

    # Compute distances
    logger.info(f"Computing nearest-train distances (metric: {args.distance_metric})")

    distances_iid = compute_nearest_train_distances(
        train_iid_agg[feature_cols].values,
        test_iid_agg[feature_cols].values,
        metric=args.distance_metric
    )

    distances_scaffold = compute_nearest_train_distances(
        train_scaffold_agg[feature_cols].values,
        test_scaffold_agg[feature_cols].values,
        metric=args.distance_metric
    )

    # Statistical comparison
    distance_comparison = compare_distance_distributions(distances_iid, distances_scaffold)

    logger.info("\nNearest-Train Distance Statistics:")
    for key, value in distance_comparison.items():
        logger.info(f"  {key}: {value}")

    with open(args.output_dir / "nearest_train_distance_stats.json", "w") as f:
        json.dump(distance_comparison, f, indent=2)

    # === ANALYSIS 4: Performance vs distance binning ===
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS 4: Performance vs Distance (Binned)")
    logger.info("=" * 80)

    # Load per-agent metrics
    iid_per_agent = pd.read_csv(args.benchmark_dir / args.iid_split / "per_agent_metrics.csv")
    scaffold_per_agent = pd.read_csv(args.benchmark_dir / args.scaffold_split / "per_agent_metrics.csv")

    iid_binned = bin_performance_by_distance(
        test_iid, distances_iid, iid_per_agent, n_bins=args.n_distance_bins
    )
    scaffold_binned = bin_performance_by_distance(
        test_scaffold, distances_scaffold, scaffold_per_agent, n_bins=args.n_distance_bins
    )

    logger.info("\nIID Split - Performance vs Distance:")
    if not iid_binned.empty:
        logger.info("\n" + iid_binned.to_string(index=False))
        iid_binned.to_csv(args.output_dir / "performance_vs_distance_iid.csv", index=False)
    else:
        logger.warning("No IID binned results")

    logger.info("\nScaffold Split - Performance vs Distance:")
    if not scaffold_binned.empty:
        logger.info("\n" + scaffold_binned.to_string(index=False))
        scaffold_binned.to_csv(args.output_dir / "performance_vs_distance_scaffold.csv", index=False)
    else:
        logger.warning("No scaffold binned results")

    # === FIGURE: Performance vs distance ===
    plot_performance_vs_distance(
        iid_binned,
        scaffold_binned,
        args.output_dir / "performance_vs_distance_comparison.png"
    )

    # === Summary JSON ===
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "candidate_pool_comparison": pool_comparison,
        "distance_comparison": distance_comparison,
        "interpretation": {
            "pool_size_difference": (
                f"Scaffold has {scaffold_stats['n_test_glycans'] - iid_stats['n_test_glycans']} "
                f"more test glycans"
            ),
            "positive_density_difference": (
                f"Scaffold has {scaffold_stats['positives_per_agent']['mean'] - iid_stats['positives_per_agent']['mean']:.2f} "
                f"more positives per agent (mean)"
            ),
            "distance_difference": distance_comparison.get("interpretation", "N/A")
        }
    }

    with open(args.output_dir / "candidate_pool_diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED: Candidate Pool Diagnostics")
    logger.info(f"All results saved to {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
