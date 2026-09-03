"""Sensitivity evaluation using explicit reported-nonbinding evidence.

This is secondary to the primary PU ranking. It uses only positive and
``reported_nonbinding`` records that share declared assay contexts. Inconclusive,
unmeasured, and insufficient-context records never become negatives implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .pu_ranking import (
    DEFAULT_AGENT_COLUMN,
    DEFAULT_EVIDENCE_COLUMN,
    DEFAULT_ID_COLUMN,
    _clean_matrix,
    _cosine_scores,
    infer_feature_columns,
)


@dataclass(frozen=True)
class ExplicitNonbindingResult:
    per_fold: pd.DataFrame
    per_agent: pd.DataFrame
    aggregate: dict[str, float | int | str]


def _binary_metrics(scores: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return {}

    comparisons = (positives[:, None] > negatives[None, :]).astype(float)
    comparisons += 0.5 * (positives[:, None] == negatives[None, :])
    auroc = float(comparisons.mean())

    order = np.lexsort((ids.astype(str), -scores))
    ordered_labels = labels[order]
    positive_positions = np.flatnonzero(ordered_labels == 1) + 1
    precisions = np.arange(1, len(positive_positions) + 1, dtype=float) / positive_positions
    average_precision = float(precisions.mean())
    reciprocal_rank = float(1.0 / positive_positions.min())
    recall_at_5 = float(np.mean(positive_positions <= 5))
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "mrr": reciprocal_rank,
        "recall@5": recall_at_5,
    }


def _bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    if len(values) == 0 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def evaluate_explicit_nonbinding(
    features: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    context_columns: Sequence[str] = ("experiment_id",),
    id_column: str = DEFAULT_ID_COLUMN,
    agent_column: str = DEFAULT_AGENT_COLUMN,
    evidence_column: str = DEFAULT_EVIDENCE_COLUMN,
    n_splits: int = 5,
    minimum_positives: int = 2,
    seed: int = 42,
    n_bootstrap: int = 2000,
) -> ExplicitNonbindingResult:
    """Evaluate positives against assay-context-matched reported non-binding records."""

    required_evidence = {id_column, agent_column, evidence_column, *context_columns}
    missing = sorted(required_evidence - set(evidence.columns))
    if missing:
        raise ValueError(f"Evidence table is missing columns: {missing}")
    if id_column not in features.columns:
        raise ValueError(f"Feature table is missing {id_column}")
    if features[id_column].duplicated().any():
        raise ValueError("Feature table must contain one row per glycan")
    if minimum_positives < 2:
        raise ValueError("minimum_positives must be at least 2")

    feature_columns = list(feature_columns or infer_feature_columns(features, id_column=id_column))
    feature_table = features.copy()
    feature_table[id_column] = feature_table[id_column].astype(str)
    feature_table = feature_table.sort_values(id_column).reset_index(drop=True)
    glycan_ids = feature_table[id_column].to_numpy(dtype=str)
    index_by_glycan = {glycan: index for index, glycan in enumerate(glycan_ids)}
    raw_matrix = feature_table[feature_columns].to_numpy(dtype=float)

    selected = evidence.copy()
    selected[evidence_column] = selected[evidence_column].astype(str).str.lower()
    selected = selected[
        selected[evidence_column].isin({"positive", "reported_nonbinding"})
    ].copy()
    selected[id_column] = selected[id_column].astype(str)
    selected[agent_column] = selected[agent_column].astype(str)
    selected = selected[selected[id_column].isin(index_by_glycan)]
    selected["_context_key"] = selected[list(context_columns)].astype(str).agg("|".join, axis=1)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    all_indices = np.arange(len(feature_table), dtype=int)

    for agent_id, agent_evidence in selected.groupby(agent_column, sort=True):
        positive_contexts = set(
            agent_evidence.loc[
                agent_evidence[evidence_column] == "positive", "_context_key"
            ]
        )
        nonbinding_contexts = set(
            agent_evidence.loc[
                agent_evidence[evidence_column] == "reported_nonbinding", "_context_key"
            ]
        )
        matched_contexts = positive_contexts & nonbinding_contexts
        matched = agent_evidence[agent_evidence["_context_key"].isin(matched_contexts)].copy()
        if matched.empty:
            continue

        conflict_counts = matched.groupby([id_column, "_context_key"])[evidence_column].nunique()
        conflicts = set(conflict_counts[conflict_counts > 1].index.tolist())
        if conflicts:
            conflict_mask = matched.apply(
                lambda row: (row[id_column], row["_context_key"]) in conflicts,
                axis=1,
            )
            matched = matched[~conflict_mask]

        positives = matched[matched[evidence_column] == "positive"].drop_duplicates(
            [id_column, "_context_key"]
        )
        nonbinding = matched[
            matched[evidence_column] == "reported_nonbinding"
        ].drop_duplicates([id_column, "_context_key"])
        positive_glycans = sorted(positives[id_column].unique().tolist())
        if len(positive_glycans) < minimum_positives or nonbinding.empty:
            continue

        shuffled = np.asarray(positive_glycans, dtype=str)
        rng.shuffle(shuffled)
        fold_count = min(n_splits, len(shuffled))
        folds = [fold for fold in np.array_split(shuffled, fold_count) if len(fold)]

        for fold_index, held_out_glycans in enumerate(folds):
            held_out_set = set(held_out_glycans.tolist())
            train_glycans = [glycan for glycan in positive_glycans if glycan not in held_out_set]
            if not train_glycans:
                continue
            test_positive = positives[positives[id_column].isin(held_out_set)].copy()
            candidate_nonbinding = nonbinding[~nonbinding[id_column].isin(train_glycans)].copy()
            removed_nonbinding = len(nonbinding) - len(candidate_nonbinding)
            if test_positive.empty or candidate_nonbinding.empty:
                continue

            candidate_records = pd.concat([test_positive, candidate_nonbinding], ignore_index=True)
            candidate_records["_binary_label"] = (
                candidate_records[evidence_column] == "positive"
            ).astype(int)
            candidate_records["_record_key"] = (
                candidate_records[id_column].astype(str)
                + "|"
                + candidate_records["_context_key"].astype(str)
            )

            candidate_glycans = set(candidate_records[id_column])
            fit_rows = np.asarray(
                [index for index, glycan in enumerate(glycan_ids) if glycan not in candidate_glycans],
                dtype=int,
            )
            if len(fit_rows) == 0:
                continue
            scaled = _clean_matrix(raw_matrix, fit_rows)
            train_indices = np.asarray([index_by_glycan[glycan] for glycan in train_glycans])
            prototype = scaled[train_indices].mean(axis=0)

            candidate_indices = np.asarray(
                [index_by_glycan[glycan] for glycan in candidate_records[id_column]],
                dtype=int,
            )
            scores = _cosine_scores(scaled[candidate_indices], prototype)
            binary_labels = candidate_records["_binary_label"].to_numpy(dtype=int)
            metrics = _binary_metrics(
                scores,
                binary_labels,
                candidate_records["_record_key"].to_numpy(dtype=str),
            )
            if not metrics:
                continue
            rows.append(
                {
                    "agent_id": agent_id,
                    "fold": fold_index,
                    "n_matched_contexts": len(matched_contexts),
                    "n_conflicting_context_records_excluded": len(conflicts),
                    "n_train_positive_glycans": len(train_glycans),
                    "n_test_positive_records": int(binary_labels.sum()),
                    "n_reported_nonbinding_records": int((binary_labels == 0).sum()),
                    "n_removed_nonbinding_train_glycan": removed_nonbinding,
                    "n_preprocessing_fit_glycans": len(fit_rows),
                    **metrics,
                }
            )

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        raise ValueError(
            "No agents had sufficient context-matched positive and reported-nonbinding evidence"
        )
    metric_columns = ["auroc", "average_precision", "mrr", "recall@5"]
    per_agent = per_fold.groupby("agent_id", as_index=False).agg(
        {
            **{metric: "mean" for metric in metric_columns},
            "fold": "count",
            "n_train_positive_glycans": "min",
            "n_test_positive_records": "sum",
            "n_reported_nonbinding_records": "sum",
        }
    ).rename(columns={"fold": "n_folds_evaluated"})

    aggregate: dict[str, float | int | str] = {
        "protocol": "context-matched explicit positive vs reported_nonbinding sensitivity",
        "prototype_scope": "training positives only",
        "candidate_scope": "held-out positives plus context-matched reported_nonbinding records",
        "preprocessing_scope": "feature universe excluding evaluated candidate glycans",
        "n_agents_evaluated": len(per_agent),
        "n_agent_folds": len(per_fold),
        "context_columns": ",".join(context_columns),
        "minimum_positives": minimum_positives,
        "seed": seed,
    }
    for offset, metric in enumerate(metric_columns):
        values = per_agent[metric].dropna().to_numpy(dtype=float)
        aggregate[metric] = float(values.mean())
        lower, upper = _bootstrap_ci(values, seed + offset, n_bootstrap)
        aggregate[f"{metric}_ci_lower"] = lower
        aggregate[f"{metric}_ci_upper"] = upper
    return ExplicitNonbindingResult(per_fold, per_agent, aggregate)
