#!/usr/bin/env python3
"""
Ranking-based binding evaluation under positive-unlabeled (PU) setting.

Since binding labels are positive-only (all label=1), standard binary classification
metrics (AUC/accuracy) are invalid. This script evaluates ranking quality:

"Given an agent and a candidate set of glycans with SSV features, can we rank
glycans such that the known positives for that agent appear near the top?"

Scoring method (prototype-based):
- For each agent, compute prototype = mean(SSV features of positive glycans)
- Score each candidate glycan by cosine similarity to prototype
- Rank glycans descending by score

Metrics:
- MRR (Mean Reciprocal Rank)
- Recall@K for K in {1,3,5,10,20}
- Mean/Median rank of positives
- AUPRC-LB (lower bound PR-AUC treating unlabeled as negatives)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binding ranking under PU setting"
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
        default=Path("reports/binding_ranking_v0"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--min_pos",
        type=int,
        default=2,
        help="Minimum positives per agent to include in evaluation",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for Recall@K",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_ssv_features(ssv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load SSV table and identify feature columns."""
    df = pd.read_csv(ssv_path)

    # Identify feature columns: numeric, excluding identifiers and metadata
    exclude_cols = {
        "glytoucan_id", "candidate_id", "n_candidates", "is_aggregated",
        "source_file", "label"
    }
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.lower().startswith("label"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    print(f"[INFO] Loaded SSV table: {len(df)} rows, {len(feature_cols)} feature columns")
    print(f"[INFO] Feature columns: {feature_cols}")

    return df, feature_cols


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load binding labels (positive-only)."""
    df = pd.read_csv(labels_path)
    # Keep only positive labels (should be all)
    df = df[df["label"] == 1].copy()
    print(f"[INFO] Loaded labels: {len(df)} positive pairs")
    print(f"[INFO] Unique glycans in labels: {df['glytoucan_id'].nunique()}")
    print(f"[INFO] Unique agents in labels: {df['agent_id'].nunique()}")
    return df


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_ranking_metrics(
    ranks: List[int],
    n_candidates: int,
    ks: List[int]
) -> Dict[str, float]:
    """
    Compute ranking metrics for a single agent.

    Args:
        ranks: List of ranks (1-indexed) of positive glycans
        n_candidates: Total number of candidate glycans
        ks: K values for Recall@K

    Returns:
        Dictionary of metrics
    """
    if not ranks:
        return {}

    ranks = np.array(ranks)
    n_pos = len(ranks)

    metrics = {}

    # MRR: 1 / (best rank among positives)
    best_rank = ranks.min()
    metrics["mrr"] = 1.0 / best_rank

    # Mean and median rank
    metrics["mean_rank"] = float(ranks.mean())
    metrics["median_rank"] = float(np.median(ranks))

    # Recall@K
    for k in ks:
        recall_at_k = (ranks <= k).sum() / n_pos
        metrics[f"recall@{k}"] = float(recall_at_k)

    return metrics


def compute_auprc_lb(
    scores: np.ndarray,
    positive_mask: np.ndarray
) -> float:
    """
    Compute AUPRC lower bound treating unlabeled as negatives.

    This is a LOWER BOUND because some "negatives" may actually be
    undiscovered positives.
    """
    if positive_mask.sum() == 0 or positive_mask.sum() == len(positive_mask):
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return average_precision_score(positive_mask.astype(int), scores)


def evaluate_agent(
    agent_id: str,
    positive_glycans: set,
    df_ssv: pd.DataFrame,
    feature_cols: List[str],
    features_scaled: np.ndarray,
    glycan_to_idx: Dict[str, int],
    ks: List[int]
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate ranking for a single agent using prototype-based scoring.

    Returns:
        Tuple of (metrics dict, top-k results dataframe)
    """
    # Get indices of positive glycans that are in our candidate universe
    pos_indices = []
    for gid in positive_glycans:
        if gid in glycan_to_idx:
            pos_indices.append(glycan_to_idx[gid])

    if not pos_indices:
        return {}, pd.DataFrame()

    n_pos = len(pos_indices)
    n_candidates = len(df_ssv)

    # Compute prototype (mean of positive feature vectors)
    pos_features = features_scaled[pos_indices]
    prototype = pos_features.mean(axis=0)

    # Check for NaN prototype
    if np.isnan(prototype).any():
        print(f"[WARN] Agent {agent_id}: prototype contains NaN, skipping")
        return {}, pd.DataFrame()

    # Score all candidates by cosine similarity to prototype
    scores = np.array([
        cosine_similarity(features_scaled[i], prototype)
        for i in range(n_candidates)
    ])

    # Rank candidates (descending score, 1-indexed)
    sorted_indices = np.argsort(-scores)
    rank_of_idx = np.empty(n_candidates, dtype=int)
    rank_of_idx[sorted_indices] = np.arange(1, n_candidates + 1)

    # Get ranks of positive glycans
    pos_ranks = [rank_of_idx[idx] for idx in pos_indices]

    # Compute ranking metrics
    metrics = compute_ranking_metrics(pos_ranks, n_candidates, ks)
    metrics["n_pos_in_universe"] = n_pos

    # Compute AUPRC lower bound
    positive_mask = np.zeros(n_candidates, dtype=bool)
    positive_mask[pos_indices] = True
    metrics["auprc_lb"] = compute_auprc_lb(scores, positive_mask)

    # Build top-K results for inspection
    max_k = max(ks)
    top_k_indices = sorted_indices[:max_k]
    top_k_data = []
    for rank, idx in enumerate(top_k_indices, 1):
        gid = df_ssv.iloc[idx]["glytoucan_id"]
        top_k_data.append({
            "agent_id": agent_id,
            "rank": rank,
            "glytoucan_id": gid,
            "score": scores[idx],
            "is_positive": gid in positive_glycans
        })

    return metrics, pd.DataFrame(top_k_data)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("PU Ranking Evaluation for Binding Prediction")
    print("=" * 60)

    # Load data
    df_ssv, feature_cols = load_ssv_features(args.ssv)
    df_labels = load_labels(args.labels)

    # Intersect: keep only labels where glycan is in SSV
    ssv_glycans = set(df_ssv["glytoucan_id"])
    df_labels_filtered = df_labels[df_labels["glytoucan_id"].isin(ssv_glycans)].copy()

    print(f"[INFO] After intersection with SSV universe:")
    print(f"       {len(df_labels_filtered)} positive pairs retained")
    print(f"       {df_labels_filtered['glytoucan_id'].nunique()} unique glycans")
    print(f"       {df_labels_filtered['agent_id'].nunique()} unique agents")

    # Build agent -> positive glycans mapping
    agent_to_positives: Dict[str, set] = {}
    for _, row in df_labels_filtered.iterrows():
        agent_id = row["agent_id"]
        gid = row["glytoucan_id"]
        if agent_id not in agent_to_positives:
            agent_to_positives[agent_id] = set()
        agent_to_positives[agent_id].add(gid)

    # Count agents by number of positives
    agents_with_min_pos = [
        a for a, pos in agent_to_positives.items() if len(pos) >= args.min_pos
    ]
    print(f"[INFO] Agents with >= {args.min_pos} positives: {len(agents_with_min_pos)}")

    # Prepare feature matrix
    features = df_ssv[feature_cols].values.astype(float)

    # Check for missing values
    n_missing = np.isnan(features).sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} missing values found, imputing with column median")
        for j in range(features.shape[1]):
            col_median = np.nanmedian(features[:, j])
            features[np.isnan(features[:, j]), j] = col_median

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Build glycan -> index mapping
    glycan_to_idx = {gid: i for i, gid in enumerate(df_ssv["glytoucan_id"])}

    # Evaluate each agent
    print(f"\n[INFO] Evaluating {len(agents_with_min_pos)} agents...")

    all_metrics = []
    all_top_k = []

    for agent_id in agents_with_min_pos:
        positives = agent_to_positives[agent_id]
        metrics, top_k_df = evaluate_agent(
            agent_id=agent_id,
            positive_glycans=positives,
            df_ssv=df_ssv,
            feature_cols=feature_cols,
            features_scaled=features_scaled,
            glycan_to_idx=glycan_to_idx,
            ks=args.ks
        )

        if metrics:
            metrics["agent_id"] = agent_id
            all_metrics.append(metrics)
            all_top_k.append(top_k_df)

    print(f"[INFO] Successfully evaluated {len(all_metrics)} agents")

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not all_metrics:
        # No agents met criteria
        summary = f"""# Binding Ranking Evaluation (PU Setting)

## Status: No Agents Evaluated

No agents had >= {args.min_pos} positive glycans within the candidate universe.

### Dataset Statistics
- Candidate glycans (with SSV features): {len(df_ssv)}
- Total agents in labels: {df_labels['agent_id'].nunique()}
- Agents with >= {args.min_pos} positives in universe: 0
"""
        (args.outdir / "summary.md").write_text(summary)
        print(f"[INFO] Report written to {args.outdir / 'summary.md'}")
        return

    # Build per-agent metrics DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    cols_order = ["agent_id", "n_pos_in_universe", "mrr", "mean_rank", "median_rank", "auprc_lb"]
    cols_order += [f"recall@{k}" for k in args.ks]
    df_metrics = df_metrics[cols_order]

    # Compute aggregate metrics
    macro_metrics = {}
    for col in df_metrics.columns:
        if col in ["agent_id"]:
            continue
        macro_metrics[col] = df_metrics[col].mean()

    # Micro-average Recall@K (weighted by n_pos)
    total_pos = df_metrics["n_pos_in_universe"].sum()
    micro_recall = {}
    for k in args.ks:
        col = f"recall@{k}"
        weighted_sum = (df_metrics[col] * df_metrics["n_pos_in_universe"]).sum()
        micro_recall[col] = weighted_sum / total_pos

    # Save per-agent metrics
    df_metrics.to_csv(args.outdir / "per_agent_metrics.csv", index=False)
    print(f"[INFO] Per-agent metrics saved to {args.outdir / 'per_agent_metrics.csv'}")

    # Save top-K examples
    df_top_k = pd.concat(all_top_k, ignore_index=True)
    df_top_k.to_csv(args.outdir / "per_agent_topk_examples.csv", index=False)
    print(f"[INFO] Top-K examples saved to {args.outdir / 'per_agent_topk_examples.csv'}")

    # Generate Recall@K plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        ks_sorted = sorted(args.ks)
        macro_recalls = [macro_metrics[f"recall@{k}"] for k in ks_sorted]
        micro_recalls = [micro_recall[f"recall@{k}"] for k in ks_sorted]

        ax.plot(ks_sorted, macro_recalls, "o-", label="Macro-avg", linewidth=2, markersize=8)
        ax.plot(ks_sorted, micro_recalls, "s--", label="Micro-avg", linewidth=2, markersize=8)

        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel("Recall@K", fontsize=12)
        ax.set_title("Recall@K Curve (PU Ranking Evaluation)", fontsize=14)
        ax.set_xticks(ks_sorted)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(args.outdir / "recall_at_k.png", dpi=150)
        plt.close()
        print(f"[INFO] Plot saved to {args.outdir / 'recall_at_k.png'}")
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot generation")

    # Build summary report
    summary_lines = [
        "# Binding Ranking Evaluation (Positive-Unlabeled Setting)",
        "",
        "**Important**: This evaluation operates under a **positive-unlabeled (PU)** setting.",
        "Labels contain only positive binding interactions; we cannot assume unlabeled pairs are negative.",
        "Standard binary classification metrics (ROC-AUC, accuracy) are **not valid** here.",
        "",
        "## Method",
        "",
        "**Scoring**: Prototype-based cosine similarity",
        "- For each agent, compute prototype = mean(standardized SSV features of positive glycans)",
        "- Score each candidate glycan by cosine similarity to prototype",
        "- Rank glycans descending by score",
        "",
        "## Dataset Statistics",
        "",
        f"| Statistic | Value |",
        f"|-----------|-------|",
        f"| Candidate glycans (with SSV) | {len(df_ssv)} |",
        f"| Total agents in labels | {df_labels['agent_id'].nunique()} |",
        f"| Positive pairs (glycan-agent) | {len(df_labels)} |",
        f"| Positive pairs in candidate universe | {len(df_labels_filtered)} |",
        f"| Agents with >= {args.min_pos} positives in universe | {len(agents_with_min_pos)} |",
        f"| Agents successfully evaluated | {len(all_metrics)} |",
        "",
        "## Feature Columns Used",
        "",
        "```",
        ", ".join(feature_cols),
        "```",
        "",
        f"Total features: {len(feature_cols)}",
        "",
        "## Aggregate Metrics",
        "",
        "### Macro-Average (unweighted mean across agents)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MRR | {macro_metrics['mrr']:.4f} |",
        f"| Mean Rank | {macro_metrics['mean_rank']:.2f} |",
        f"| Median Rank | {macro_metrics['median_rank']:.2f} |",
        f"| AUPRC-LB | {macro_metrics['auprc_lb']:.4f} |",
    ]

    for k in args.ks:
        summary_lines.append(f"| Recall@{k} | {macro_metrics[f'recall@{k}']:.4f} |")

    summary_lines.extend([
        "",
        "### Micro-Average Recall@K (weighted by number of positives)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    for k in args.ks:
        summary_lines.append(f"| Recall@{k} | {micro_recall[f'recall@{k}']:.4f} |")

    summary_lines.extend([
        "",
        "## Distribution of Per-Agent Metrics",
        "",
        "| Metric | Min | 25% | Median | 75% | Max |",
        "|--------|-----|-----|--------|-----|-----|",
    ])

    for metric_name in ["mrr", "mean_rank", "auprc_lb"]:
        desc = df_metrics[metric_name].describe()
        summary_lines.append(
            f"| {metric_name} | {desc['min']:.3f} | {desc['25%']:.3f} | "
            f"{desc['50%']:.3f} | {desc['75%']:.3f} | {desc['max']:.3f} |"
        )

    summary_lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- **MRR (Mean Reciprocal Rank)**: Average of 1/(best rank of any positive). Higher is better.",
        "- **Recall@K**: Fraction of positives ranked in top K. Higher is better.",
        "- **Mean/Median Rank**: Average/median rank of positive glycans. Lower is better.",
        "- **AUPRC-LB**: Area under Precision-Recall curve, treating all unlabeled as negatives.",
        "  This is a **lower bound** because some unlabeled pairs may be undiscovered positives.",
        "",
        "## Limitations",
        "",
        "1. **Positive-only labels**: We cannot compute true negatives, so standard AUC/accuracy are invalid.",
        "2. **Prototype method**: Simple mean-based prototype may not capture complex binding patterns.",
        "3. **Small candidate universe**: Only 61 glycans with SSV features limits evaluation power.",
        "4. **No agent features**: Ranking uses only glycan features; agent-specific features could improve results.",
        "5. **Lower bound estimates**: AUPRC-LB underestimates true performance if unlabeled positives exist.",
        "",
        "## Files Generated",
        "",
        "- `per_agent_metrics.csv`: Per-agent ranking metrics",
        "- `per_agent_topk_examples.csv`: Top-K ranked glycans per agent for inspection",
        "- `recall_at_k.png`: Recall@K curve plot",
        "",
        f"---",
        f"",
        f"*Generated with seed={args.seed}*",
    ])

    summary_text = "\n".join(summary_lines)
    (args.outdir / "summary.md").write_text(summary_text)
    print(f"[INFO] Summary report saved to {args.outdir / 'summary.md'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Evaluated {len(all_metrics)} agents")
    print(f"Macro-avg MRR: {macro_metrics['mrr']:.4f}")
    print(f"Macro-avg Recall@5: {macro_metrics['recall@5']:.4f}")
    print(f"Macro-avg AUPRC-LB: {macro_metrics['auprc_lb']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
