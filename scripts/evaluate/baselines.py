#!/usr/bin/env python3
"""
Baselines and permutation tests for PU ranking evaluation.

Adds statistical grounding to the binding ranking evaluation by comparing
observed performance against:
1. Random baseline: Random scores for each glycan (no structure information)
2. Permutation test: Shuffle feature matrix to break glycan-feature correspondence

This provides p-values to assess whether the ranking performance is statistically
significant compared to chance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# Import shared functions from the main evaluation script
# We import these to ensure consistency with the observed evaluation
from evaluate_binding_ranking_pu import (
    load_ssv_features,
    load_labels,
    cosine_similarity,
    compute_ranking_metrics,
    compute_auprc_lb,
)


def setup_logging(log_path: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("binding_ranking_baselines")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baselines and permutation tests for PU ranking evaluation"
    )
    parser.add_argument(
        "--ssv",
        type=Path,
        default=Path("data/ssv/targeted_sugarbind_v0/ssv_table.csv"),
        help="Path to SSV feature table",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/binding/sugarbind_v0/labels.csv"),
        help="Path to binding labels (positive-only)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/binding_ranking_v0_baselines"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--min_pos",
        type=int,
        default=2,
        help="Minimum positives per agent to include in evaluation",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Primary K value for Recall@K reporting",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=1000,
        help="Number of random baseline repeats",
    )
    parser.add_argument(
        "--perm-repeats",
        type=int,
        default=1000,
        help="Number of permutation test repeats",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: use 200 repeats for quick sanity check",
    )
    return parser.parse_args()


def evaluate_agent_with_scores(
    agent_id: str,
    positive_indices: List[int],
    scores: np.ndarray,
    n_candidates: int,
    ks: List[int]
) -> Dict[str, float]:
    """
    Evaluate ranking for a single agent given pre-computed scores.

    Args:
        agent_id: Agent identifier
        positive_indices: Indices of positive glycans in the candidate list
        scores: Score array (one per candidate glycan)
        n_candidates: Total number of candidates
        ks: K values for Recall@K

    Returns:
        Dictionary of metrics
    """
    if not positive_indices:
        return {}

    n_pos = len(positive_indices)

    # Rank candidates (descending score, 1-indexed)
    sorted_indices = np.argsort(-scores)
    rank_of_idx = np.empty(n_candidates, dtype=int)
    rank_of_idx[sorted_indices] = np.arange(1, n_candidates + 1)

    # Get ranks of positive glycans
    pos_ranks = [rank_of_idx[idx] for idx in positive_indices]

    # Compute ranking metrics
    metrics = compute_ranking_metrics(pos_ranks, n_candidates, ks)
    metrics["n_pos_in_universe"] = n_pos

    # Compute AUPRC lower bound
    positive_mask = np.zeros(n_candidates, dtype=bool)
    positive_mask[positive_indices] = True
    metrics["auprc_lb"] = compute_auprc_lb(scores, positive_mask)

    return metrics


def compute_prototype_scores(
    positive_indices: List[int],
    features_scaled: np.ndarray
) -> np.ndarray:
    """Compute prototype-based cosine similarity scores."""
    n_candidates = features_scaled.shape[0]

    # Compute prototype (mean of positive feature vectors)
    pos_features = features_scaled[positive_indices]
    prototype = pos_features.mean(axis=0)

    # Check for NaN prototype
    if np.isnan(prototype).any():
        return np.full(n_candidates, np.nan)

    # Score all candidates by cosine similarity to prototype
    scores = np.array([
        cosine_similarity(features_scaled[i], prototype)
        for i in range(n_candidates)
    ])

    return scores


def run_observed_evaluation(
    agents_info: List[Tuple[str, List[int]]],
    features_scaled: np.ndarray,
    ks: List[int],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run observed (prototype-based) evaluation.

    Returns:
        Tuple of (per-agent metrics DataFrame, aggregate metrics dict)
    """
    logger.info("Running observed evaluation (prototype-based scoring)...")

    n_candidates = features_scaled.shape[0]
    all_metrics = []

    for agent_id, pos_indices in agents_info:
        scores = compute_prototype_scores(pos_indices, features_scaled)
        if np.isnan(scores).any():
            logger.warning(f"Agent {agent_id}: scores contain NaN, skipping")
            continue

        metrics = evaluate_agent_with_scores(
            agent_id, pos_indices, scores, n_candidates, ks
        )
        if metrics:
            metrics["agent_id"] = agent_id
            all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)

    # Compute macro-average
    agg = {}
    for col in ["mrr", "mean_rank", "auprc_lb"] + [f"recall@{k}" for k in ks]:
        if col in df_metrics.columns:
            agg[col] = df_metrics[col].mean()

    logger.info(f"Observed evaluation: {len(all_metrics)} agents, MRR={agg.get('mrr', 0):.4f}")

    return df_metrics, agg


def run_random_baseline(
    agents_info: List[Tuple[str, List[int]]],
    n_candidates: int,
    ks: List[int],
    n_repeats: int,
    rng: np.random.Generator,
    logger: logging.Logger
) -> List[Dict[str, float]]:
    """
    Run random baseline evaluation.

    For each repeat, generate random scores for all candidate glycans
    and compute aggregate metrics.

    Returns:
        List of aggregate metrics dicts (one per repeat)
    """
    logger.info(f"Running random baseline ({n_repeats} repeats)...")

    null_distribution = []

    for rep in range(n_repeats):
        if rep > 0 and rep % 500 == 0:
            logger.info(f"  Random baseline: {rep}/{n_repeats} repeats done")

        rep_metrics = []

        for agent_id, pos_indices in agents_info:
            # Generate random scores for this agent
            scores = rng.random(n_candidates)

            metrics = evaluate_agent_with_scores(
                agent_id, pos_indices, scores, n_candidates, ks
            )
            if metrics:
                rep_metrics.append(metrics)

        if rep_metrics:
            # Compute macro-average for this repeat
            df_rep = pd.DataFrame(rep_metrics)
            agg = {"repeat": rep, "mode": "random"}
            for col in ["mrr", "mean_rank", "auprc_lb"] + [f"recall@{k}" for k in ks]:
                if col in df_rep.columns:
                    agg[col] = df_rep[col].mean()
            null_distribution.append(agg)

    logger.info(f"Random baseline complete: {len(null_distribution)} repeats")
    return null_distribution


def run_permutation_test(
    agents_info: List[Tuple[str, List[int]]],
    features_scaled: np.ndarray,
    ks: List[int],
    n_repeats: int,
    rng: np.random.Generator,
    logger: logging.Logger
) -> List[Dict[str, float]]:
    """
    Run permutation test.

    For each repeat, permute feature matrix rows (global permutation)
    and run the same prototype-based scoring.

    Returns:
        List of aggregate metrics dicts (one per repeat)
    """
    logger.info(f"Running permutation test ({n_repeats} repeats)...")

    n_candidates = features_scaled.shape[0]
    null_distribution = []

    for rep in range(n_repeats):
        if rep > 0 and rep % 500 == 0:
            logger.info(f"  Permutation test: {rep}/{n_repeats} repeats done")

        # Permute rows of feature matrix
        perm_indices = rng.permutation(n_candidates)
        features_permuted = features_scaled[perm_indices]

        rep_metrics = []

        for agent_id, pos_indices in agents_info:
            scores = compute_prototype_scores(pos_indices, features_permuted)
            if np.isnan(scores).any():
                continue

            metrics = evaluate_agent_with_scores(
                agent_id, pos_indices, scores, n_candidates, ks
            )
            if metrics:
                rep_metrics.append(metrics)

        if rep_metrics:
            # Compute macro-average for this repeat
            df_rep = pd.DataFrame(rep_metrics)
            agg = {"repeat": rep, "mode": "permute_features"}
            for col in ["mrr", "mean_rank", "auprc_lb"] + [f"recall@{k}" for k in ks]:
                if col in df_rep.columns:
                    agg[col] = df_rep[col].mean()
            null_distribution.append(agg)

    logger.info(f"Permutation test complete: {len(null_distribution)} repeats")
    return null_distribution


def compute_pvalue(observed: float, null_dist: np.ndarray, greater_is_better: bool = True) -> float:
    """
    Compute one-sided p-value.

    p = (1 + #repeats where baseline_metric >= observed_metric) / (1 + n_repeats)
    for greater-is-better metrics (MRR, Recall@K, AUPRC-LB)

    For lower-is-better (mean_rank), invert the comparison.
    """
    n_repeats = len(null_dist)
    if n_repeats == 0:
        return np.nan

    if greater_is_better:
        n_extreme = (null_dist >= observed).sum()
    else:
        n_extreme = (null_dist <= observed).sum()

    return (1 + n_extreme) / (1 + n_repeats)


def main():
    args = parse_args()

    # Apply fast mode
    if args.fast:
        args.random_repeats = 200
        args.perm_repeats = 200

    # Setup
    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/binding_ranking_baselines.log")
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("PU Ranking Evaluation: Baselines & Permutation Tests")
    logger.info("=" * 60)

    # Set random seed
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # For sklearn compatibility

    # Load data
    df_ssv, feature_cols = load_ssv_features(args.ssv)
    df_labels = load_labels(args.labels)

    # Intersect: keep only labels where glycan is in SSV
    ssv_glycans = set(df_ssv["glytoucan_id"])
    df_labels_filtered = df_labels[df_labels["glytoucan_id"].isin(ssv_glycans)].copy()

    logger.info(f"After intersection with SSV universe:")
    logger.info(f"  {len(df_labels_filtered)} positive pairs retained")
    logger.info(f"  {df_labels_filtered['glytoucan_id'].nunique()} unique glycans")
    logger.info(f"  {df_labels_filtered['agent_id'].nunique()} unique agents")

    # Sort glycans for deterministic ordering
    df_ssv = df_ssv.sort_values("glytoucan_id").reset_index(drop=True)
    glycan_to_idx = {gid: i for i, gid in enumerate(df_ssv["glytoucan_id"])}

    # Build agent -> positive indices mapping
    agent_to_pos_indices: Dict[str, List[int]] = {}
    for _, row in df_labels_filtered.iterrows():
        agent_id = row["agent_id"]
        gid = row["glytoucan_id"]
        if gid in glycan_to_idx:
            if agent_id not in agent_to_pos_indices:
                agent_to_pos_indices[agent_id] = []
            agent_to_pos_indices[agent_id].append(glycan_to_idx[gid])

    # Filter to agents with >= min_pos positives and sort for determinism
    agents_info = [
        (agent_id, sorted(pos_indices))
        for agent_id, pos_indices in sorted(agent_to_pos_indices.items())
        if len(pos_indices) >= args.min_pos
    ]

    logger.info(f"Agents with >= {args.min_pos} positives: {len(agents_info)}")

    if not agents_info:
        logger.error("No agents met criteria. Exiting.")
        return

    # Prepare feature matrix
    features = df_ssv[feature_cols].values.astype(float)
    n_candidates = features.shape[0]

    # Handle missing values
    n_missing = np.isnan(features).sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} missing values found, imputing with column median")
        for j in range(features.shape[1]):
            col_median = np.nanmedian(features[:, j])
            features[np.isnan(features[:, j]), j] = col_median

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K values for evaluation
    ks = [1, 3, 5, 10, 20]
    primary_k = args.k

    # Run observed evaluation
    start_time = time.time()
    df_observed, agg_observed = run_observed_evaluation(
        agents_info, features_scaled, ks, logger
    )

    # Run random baseline
    null_random = run_random_baseline(
        agents_info, n_candidates, ks, args.random_repeats, rng, logger
    )

    # Run permutation test
    null_perm = run_permutation_test(
        agents_info, features_scaled, ks, args.perm_repeats, rng, logger
    )

    elapsed = time.time() - start_time
    logger.info(f"Total evaluation time: {elapsed:.1f}s")

    # Combine null distributions
    df_null = pd.DataFrame(null_random + null_perm)

    # Compute p-values
    metrics_to_test = ["mrr", f"recall@{primary_k}", "auprc_lb"]
    greater_is_better = {"mrr": True, f"recall@{primary_k}": True, "auprc_lb": True, "mean_rank": False}

    pvalues = {}
    for metric in metrics_to_test:
        obs_val = agg_observed.get(metric, np.nan)

        # Random baseline p-value
        null_random_vals = df_null[df_null["mode"] == "random"][metric].values
        pvalues[f"{metric}_vs_random"] = compute_pvalue(
            obs_val, null_random_vals, greater_is_better.get(metric, True)
        )

        # Permutation test p-value
        null_perm_vals = df_null[df_null["mode"] == "permute_features"][metric].values
        pvalues[f"{metric}_vs_perm"] = compute_pvalue(
            obs_val, null_perm_vals, greater_is_better.get(metric, True)
        )

    # Save config
    config = {
        "ssv_path": str(args.ssv),
        "labels_path": str(args.labels),
        "min_pos": args.min_pos,
        "primary_k": primary_k,
        "random_repeats": args.random_repeats,
        "perm_repeats": args.perm_repeats,
        "seed": args.seed,
        "fast_mode": args.fast,
        "n_agents_evaluated": len(agents_info),
        "n_candidates": n_candidates,
        "feature_cols": feature_cols,
        "elapsed_seconds": elapsed,
    }

    with open(args.outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save null distributions
    df_null.to_csv(args.outdir / "null_distributions.csv", index=False)

    # Save aggregate metrics
    agg_rows = []

    # Observed
    obs_row = {"mode": "observed", **agg_observed}
    agg_rows.append(obs_row)

    # Random baseline summary
    df_random = df_null[df_null["mode"] == "random"]
    random_row = {
        "mode": "random_mean",
        **{col: df_random[col].mean() for col in df_random.columns if col not in ["repeat", "mode"]}
    }
    agg_rows.append(random_row)

    # Permutation test summary
    df_perm = df_null[df_null["mode"] == "permute_features"]
    perm_row = {
        "mode": "permute_mean",
        **{col: df_perm[col].mean() for col in df_perm.columns if col not in ["repeat", "mode"]}
    }
    agg_rows.append(perm_row)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(args.outdir / "aggregate_metrics.csv", index=False)

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = args.outdir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for metric, label in [("mrr", "MRR"), (f"recall@{primary_k}", f"Recall@{primary_k}")]:
            obs_val = agg_observed.get(metric, np.nan)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Random baseline
            ax = axes[0]
            null_vals = df_random[metric].values
            ax.hist(null_vals, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
            ax.axvline(obs_val, color="red", linewidth=2, label=f"Observed: {obs_val:.4f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Random Baseline (p={pvalues[f'{metric}_vs_random']:.4f})")
            ax.legend()

            # Permutation test
            ax = axes[1]
            null_vals = df_perm[metric].values
            ax.hist(null_vals, bins=30, alpha=0.7, color="darkorange", edgecolor="white")
            ax.axvline(obs_val, color="red", linewidth=2, label=f"Observed: {obs_val:.4f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Permutation Test (p={pvalues[f'{metric}_vs_perm']:.4f})")
            ax.legend()

            plt.tight_layout()
            plt.savefig(plots_dir / f"{metric.replace('@', '_at_')}_null_dist.png", dpi=150)
            plt.close()

        logger.info(f"Plots saved to {plots_dir}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")

    # Generate summary report
    summary_lines = [
        "# Binding Ranking Evaluation: Baselines & Permutation Tests",
        "",
        "This report provides statistical grounding for the PU ranking evaluation",
        "by comparing observed performance against random and permutation baselines.",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| SSV file | `{args.ssv}` |",
        f"| Labels file | `{args.labels}` |",
        f"| Min positives per agent | {args.min_pos} |",
        f"| Primary K for Recall@K | {primary_k} |",
        f"| Random repeats | {args.random_repeats} |",
        f"| Permutation repeats | {args.perm_repeats} |",
        f"| Seed | {args.seed} |",
        f"| Fast mode | {args.fast} |",
        "",
        "## Dataset Statistics",
        "",
        f"| Statistic | Value |",
        f"|-----------|-------|",
        f"| Candidate glycans | {n_candidates} |",
        f"| Agents evaluated | {len(agents_info)} |",
        f"| Feature columns | {len(feature_cols)} |",
        "",
        "## Baselines Explained",
        "",
        "### Random Baseline",
        "",
        "For each repeat, we generate **random scores** for all candidate glycans",
        "(independently for each agent) and compute ranking metrics. This represents",
        "performance expected when the model has **no predictive power**.",
        "",
        "### Permutation Test",
        "",
        "For each repeat, we **shuffle the feature matrix rows** (permute glycan-to-feature",
        "correspondence globally) and run the same prototype-based scoring. This breaks",
        "the structure-binding relationship while preserving feature distributions.",
        "",
        "This tests whether the observed ranking performance is due to the actual",
        "glycan features rather than statistical artifacts.",
        "",
        "## Results",
        "",
        "### Aggregate Metrics Comparison",
        "",
        "| Mode | MRR | Recall@{} | AUPRC-LB | Mean Rank |".format(primary_k),
        "|------|-----|-----------|----------|-----------|",
    ]

    for _, row in df_agg.iterrows():
        mode = row["mode"]
        mrr = row.get("mrr", np.nan)
        recall = row.get(f"recall@{primary_k}", np.nan)
        auprc = row.get("auprc_lb", np.nan)
        mean_rank = row.get("mean_rank", np.nan)
        summary_lines.append(
            f"| {mode} | {mrr:.4f} | {recall:.4f} | {auprc:.4f} | {mean_rank:.2f} |"
        )

    summary_lines.extend([
        "",
        "### P-Values (one-sided, greater is better)",
        "",
        "| Metric | vs Random | vs Permutation | Interpretation |",
        "|--------|-----------|----------------|----------------|",
    ])

    for metric in metrics_to_test:
        p_random = pvalues[f"{metric}_vs_random"]
        p_perm = pvalues[f"{metric}_vs_perm"]

        # Interpretation
        if p_perm < 0.01:
            interp = "Highly significant"
        elif p_perm < 0.05:
            interp = "Significant"
        elif p_perm < 0.10:
            interp = "Marginally significant"
        else:
            interp = "Not significant"

        summary_lines.append(
            f"| {metric} | {p_random:.4f} | {p_perm:.4f} | {interp} |"
        )

    summary_lines.extend([
        "",
        "**Interpretation of p-values:**",
        "",
        "- p < 0.01: Highly significant (observed >> baseline)",
        "- p < 0.05: Significant at 95% confidence level",
        "- p < 0.10: Marginally significant",
        "- p >= 0.10: Not statistically significant",
        "",
        "**Note**: P-values are computed as:",
        "```",
        "p = (1 + #repeats where baseline >= observed) / (1 + n_repeats)",
        "```",
        "",
        "## Null Distribution Statistics",
        "",
        "### Random Baseline",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|--------|------|-----|-----|-----|",
    ])

    for metric in ["mrr", f"recall@{primary_k}", "auprc_lb"]:
        vals = df_random[metric].values
        summary_lines.append(
            f"| {metric} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} |"
        )

    summary_lines.extend([
        "",
        "### Permutation Test",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|--------|------|-----|-----|-----|",
    ])

    for metric in ["mrr", f"recall@{primary_k}", "auprc_lb"]:
        vals = df_perm[metric].values
        summary_lines.append(
            f"| {metric} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} |"
        )

    summary_lines.extend([
        "",
        "## Files Generated",
        "",
        "- `config.json`: Run configuration and parameters",
        "- `aggregate_metrics.csv`: Summary metrics for observed, random, and permutation",
        "- `null_distributions.csv`: Per-repeat metrics for null distributions",
        "- `plots/`: Histograms showing null distributions with observed values",
        "",
        "## Conclusion",
        "",
    ])

    # Add conclusion based on p-values
    p_mrr = pvalues[f"mrr_vs_perm"]
    p_recall = pvalues[f"recall@{primary_k}_vs_perm"]

    if p_mrr < 0.05 and p_recall < 0.05:
        summary_lines.append(
            "The ranking performance is **statistically significant** (p < 0.05 for both MRR and Recall@K)."
        )
        summary_lines.append(
            "The SSV features contain meaningful structural information for predicting glycan-agent binding."
        )
    elif p_mrr < 0.10 or p_recall < 0.10:
        summary_lines.append(
            "The ranking performance shows **marginal significance** (0.05 < p < 0.10)."
        )
        summary_lines.append(
            "More data or features may be needed to establish strong predictive power."
        )
    else:
        summary_lines.append(
            "The ranking performance is **not statistically significant** (p >= 0.10)."
        )
        summary_lines.append(
            "The current SSV features may not adequately capture binding-relevant structure."
        )

    summary_lines.extend([
        "",
        "---",
        "",
        f"*Generated with seed={args.seed}, {args.random_repeats} random repeats, {args.perm_repeats} permutation repeats*",
        f"*Elapsed time: {elapsed:.1f}s*",
    ])

    summary_text = "\n".join(summary_lines)
    (args.outdir / "summary.md").write_text(summary_text)

    # Print summary to console
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Observed MRR: {agg_observed['mrr']:.4f}")
    logger.info(f"Observed Recall@{primary_k}: {agg_observed[f'recall@{primary_k}']:.4f}")
    logger.info(f"Observed AUPRC-LB: {agg_observed['auprc_lb']:.4f}")
    logger.info("")
    logger.info("P-values (vs permutation test):")
    logger.info(f"  MRR: {pvalues['mrr_vs_perm']:.4f}")
    logger.info(f"  Recall@{primary_k}: {pvalues[f'recall@{primary_k}_vs_perm']:.4f}")
    logger.info(f"  AUPRC-LB: {pvalues['auprc_lb_vs_perm']:.4f}")
    logger.info("=" * 60)
    logger.info(f"Report saved to {args.outdir / 'summary.md'}")


if __name__ == "__main__":
    main()
