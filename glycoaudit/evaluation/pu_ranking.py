"""Leakage-controlled positive--unlabeled ranking evaluation.

The original prototype diagnostic used all known positives both to construct an
agent prototype and to evaluate the same prototype. That self-retrieval design
does not estimate recovery of unseen relations. This module instead holds out
known positives before prototype construction and records the complete candidate
pool and preprocessing scope for every fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMN = "glytoucan_id"
DEFAULT_AGENT_COLUMN = "agent_id"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_EVIDENCE_COLUMN = "evidence_status"
ALLOWED_EVIDENCE_STATUSES = {
    "positive",
    "reported_nonbinding",
    "inconclusive",
    "unmeasured",
    "insufficient_context",
}
NON_FEATURE_COLUMNS = {
    DEFAULT_ID_COLUMN,
    "candidate_id",
    "model_id",
    "n_models",
    "contact_cutoff_angstrom",
    "source_file",
    "source_dir",
    "data_source",
    "acquisition_source",
    "structure_source",
    "label",
}


@dataclass(frozen=True)
class EvaluationResult:
    """Tables produced by a leakage-controlled evaluation run."""

    per_fold: pd.DataFrame
    per_agent: pd.DataFrame
    aggregate: dict[str, float | int | str]


@dataclass(frozen=True)
class NullControlResult:
    """Empirical random and feature-permutation null distributions."""

    distributions: pd.DataFrame
    pvalues: dict[str, float | int]


def evaluate_leave_one_agent_out_popularity(
    labels: pd.DataFrame,
    candidate_ids: Sequence[str],
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    agent_column: str = DEFAULT_AGENT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    evidence_column: str = DEFAULT_EVIDENCE_COLUMN,
    minimum_positives: int = 2,
    ks: Sequence[int] = (5, 10),
    seed: int = 42,
    n_bootstrap: int = 2000,
    fold_manifest: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Evaluate a leave-one-agent-out glycan-popularity control.

    For target agent ``a``, a glycan's score is the number of distinct positive
    relations involving that glycan among agents other than ``a``. The score
    reference pool is the complete strictly joined set (765 relations from 227
    agents); evaluation remains restricted to the released fold population (697
    relations from 159 agents). Every relation for the target agent is excluded
    from score construction. The
    target agent's training positives are then removed from each fold's candidate
    set exactly as in :func:`evaluate_positive_kfold`.  Exact score ties are
    resolved by ascending glycan identifier through :func:`_ranks`.
    """

    if minimum_positives < 2:
        raise ValueError("minimum_positives must be at least 2 for held-out evaluation")

    candidate_array = np.asarray(sorted({str(value) for value in candidate_ids}), dtype=str)
    if len(candidate_array) == 0:
        raise ValueError("candidate_ids must contain at least one glycan")

    positives = positive_pairs(
        labels,
        id_column=id_column,
        agent_column=agent_column,
        label_column=label_column,
        evidence_column=evidence_column,
    )
    positives = positives[positives[id_column].isin(set(candidate_array))].copy()

    if fold_manifest is None:
        raise ValueError(
            "A released fold manifest is required for the popularity control"
        )
    required_manifest_columns = {agent_column, id_column, "fold"}
    missing_manifest_columns = required_manifest_columns - set(fold_manifest.columns)
    if missing_manifest_columns:
        raise ValueError(
            f"Fold manifest missing columns: {sorted(missing_manifest_columns)}"
        )
    manifest = fold_manifest[[agent_column, id_column, "fold"]].copy()
    manifest[agent_column] = manifest[agent_column].astype(str)
    manifest[id_column] = manifest[id_column].astype(str)
    if manifest.duplicated([agent_column, id_column]).any():
        raise ValueError("Fold manifest assigns an agent--glycan positive more than once")

    global_counts = positives.groupby(id_column)[agent_column].nunique()
    rows: list[dict[str, float | int | str]] = []

    for agent_id, agent_frame in positives.groupby(agent_column, sort=True):
        agent_positive_ids = sorted(agent_frame[id_column].unique().tolist())
        if len(agent_positive_ids) < minimum_positives:
            continue

        agent_manifest = manifest[manifest[agent_column] == str(agent_id)]
        assigned_ids = set(agent_manifest[id_column])
        if assigned_ids != set(agent_positive_ids):
            raise ValueError(
                f"Fold manifest does not exactly cover positives for agent {agent_id}: "
                f"missing={sorted(set(agent_positive_ids) - assigned_ids)}, "
                f"extra={sorted(assigned_ids - set(agent_positive_ids))}"
            )

        target_positive_set = set(agent_positive_ids)
        scores_all = np.asarray(
            [
                int(global_counts.get(glycan_id, 0))
                - int(glycan_id in target_positive_set)
                for glycan_id in candidate_array
            ],
            dtype=float,
        )

        for fold_value, group in agent_manifest.groupby("fold", sort=True):
            test_positive_ids = sorted(group[id_column].astype(str).tolist())
            test_positive_set = set(test_positive_ids)
            train_positive_ids = sorted(target_positive_set - test_positive_set)
            if not train_positive_ids:
                continue

            candidate_mask = ~np.isin(candidate_array, np.asarray(train_positive_ids))
            fold_candidate_ids = candidate_array[candidate_mask]
            fold_scores = scores_all[candidate_mask]
            fold_ranks = _ranks(fold_scores, fold_candidate_ids)
            rank_by_id = {
                glycan_id: int(rank)
                for glycan_id, rank in zip(fold_candidate_ids.tolist(), fold_ranks.tolist())
            }
            held_out_ranks = np.asarray(
                [rank_by_id[glycan_id] for glycan_id in test_positive_ids], dtype=int
            )
            rows.append(
                {
                    "agent_id": str(agent_id),
                    "fold": int(fold_value),
                    "n_train_positives": len(train_positive_ids),
                    "n_test_positives": len(test_positive_ids),
                    "n_candidates": len(fold_candidate_ids),
                    "preprocessing_fit_glycans": 0,
                    "target_agent_relations_used_for_scores": 0,
                    **_fold_metrics(held_out_ranks, ks),
                }
            )

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        raise ValueError("No agents were evaluable for the popularity control")
    per_agent = _agent_summary(per_fold, ks)

    aggregate: dict[str, float | int | str] = {
        "protocol": "leave-one-agent-out glycan popularity",
        "score_definition": "distinct positive relations for each glycan among all other agents",
        "target_agent_information": "all target-agent relations excluded from score construction",
        "candidate_policy": "held-out positives plus unlabeled glycans; training positives excluded",
        "tie_policy": "descending popularity score, then ascending glycan identifier",
        "split_source": "released fold manifest",
        "n_glycans": len(candidate_array),
        "n_scoring_agents": int(positives[agent_column].nunique()),
        "n_scoring_relations": int(len(positives)),
        "n_agents_evaluated": len(per_agent),
        "n_agent_folds": len(per_fold),
        "minimum_positives": minimum_positives,
        "seed": seed,
    }
    metrics = ["mrr", "mean_rank", "known_positive_ap", *[f"recall@{k}" for k in ks]]
    for offset, metric in enumerate(metrics):
        aggregate[metric] = float(per_agent[metric].mean())
        if n_bootstrap > 0:
            lower, upper = _bootstrap_agent_ci(
                per_agent,
                metric,
                seed=seed + offset,
                n_bootstrap=n_bootstrap,
            )
            aggregate[f"{metric}_ci_lower"] = lower
            aggregate[f"{metric}_ci_upper"] = upper

    return EvaluationResult(per_fold=per_fold, per_agent=per_agent, aggregate=aggregate)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""

    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues must be a one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must be finite values between zero and one")
    if len(values) == 0:
        return []

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    multipliers = np.arange(len(values), 0, -1, dtype=float)
    adjusted_ordered = np.maximum.accumulate(ordered * multipliers)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted.tolist()


def infer_feature_columns(
    features: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    excluded: Iterable[str] = NON_FEATURE_COLUMNS,
) -> list[str]:
    """Return numeric descriptor columns after removing identifiers/metadata."""

    excluded_columns = set(excluded) | {id_column}
    columns = [
        column
        for column in features.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not columns:
        raise ValueError("No numeric feature columns were found in the feature table")
    return columns


def validate_inputs(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    agent_column: str = DEFAULT_AGENT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    evidence_column: str = DEFAULT_EVIDENCE_COLUMN,
) -> None:
    """Validate the minimum schema and uniqueness assumptions."""

    missing_feature_columns = {id_column} - set(features.columns)
    missing_label_columns = {id_column, agent_column} - set(labels.columns)
    if missing_feature_columns:
        raise ValueError(f"Feature table is missing columns: {sorted(missing_feature_columns)}")
    if missing_label_columns:
        raise ValueError(f"Label table is missing columns: {sorted(missing_label_columns)}")
    if features[id_column].isna().any():
        raise ValueError(f"Feature table contains missing {id_column} values")
    duplicated = features.loc[features[id_column].duplicated(), id_column].astype(str).tolist()
    if duplicated:
        preview = ", ".join(duplicated[:5])
        raise ValueError(f"Feature table must contain one row per glycan; duplicates include {preview}")
    if label_column in labels.columns:
        observed = set(labels[label_column].dropna().unique().tolist())
        if not observed.issubset({0, 1, False, True}):
            raise ValueError(
                f"{label_column} must be binary when present; observed values include {sorted(observed)[:5]}"
            )
    elif evidence_column in labels.columns:
        observed_statuses = set(
            labels[evidence_column].dropna().astype(str).str.lower().unique().tolist()
        )
        unexpected = observed_statuses - ALLOWED_EVIDENCE_STATUSES
        if unexpected:
            raise ValueError(
                f"{evidence_column} contains unsupported states: {sorted(unexpected)}"
            )
    else:
        raise ValueError(
            f"Evidence table must contain either {label_column!r} or {evidence_column!r}"
        )


def positive_pairs(
    labels: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    agent_column: str = DEFAULT_AGENT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    evidence_column: str = DEFAULT_EVIDENCE_COLUMN,
) -> pd.DataFrame:
    """Return unique explicitly positive agent--glycan relations."""

    selected = labels.copy()
    if label_column in selected.columns:
        selected = selected[selected[label_column] == 1]
    else:
        selected = selected[
            selected[evidence_column].astype(str).str.lower() == "positive"
        ]
    selected = selected[[agent_column, id_column]].dropna().drop_duplicates()
    selected[agent_column] = selected[agent_column].astype(str)
    selected[id_column] = selected[id_column].astype(str)
    return selected


def _clean_matrix(matrix: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Median-impute and standardize using only declared preprocessing rows."""

    values = matrix.astype(float, copy=True)
    values[~np.isfinite(values)] = np.nan
    fit_values = values[fit_rows]
    medians = np.nanmedian(fit_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing_rows, missing_columns = np.where(np.isnan(values))
    values[missing_rows, missing_columns] = medians[missing_columns]

    means = values[fit_rows].mean(axis=0)
    standard_deviations = values[fit_rows].std(axis=0)
    standard_deviations = np.where(standard_deviations > 0, standard_deviations, 1.0)
    return (values - means) / standard_deviations


def _tfidf_matrix(matrix: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Apply TF-IDF with document frequencies fitted only on declared rows."""
    values = matrix.astype(float, copy=True)
    values[~np.isfinite(values)] = 0.0
    values = np.where(values > 0, values, 0.0)
    fit_values = values[fit_rows]
    document_frequency = np.sum(fit_values > 0, axis=0)
    inverse_document_frequency = np.log(
        (1.0 + len(fit_rows)) / (1.0 + document_frequency)
    ) + 1.0
    transformed = values * inverse_document_frequency
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return transformed / norms


def _preprocess_matrix(
    matrix: np.ndarray, fit_rows: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "standardize":
        return _clean_matrix(matrix, fit_rows)
    if mode == "tfidf":
        return _tfidf_matrix(matrix, fit_rows)
    if mode == "none":
        values = matrix.astype(float, copy=True)
        values[~np.isfinite(values)] = 0.0
        return values
    raise ValueError(f"Unsupported preprocessing mode: {mode}")


def _cosine_scores(candidate_matrix: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    # Some BLAS builds report stale floating-point status flags for the
    # high-dimensional, finite WURCS matrices. Keep the released arithmetic
    # unchanged while preventing those spurious runtime warnings.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        prototype_norm = np.linalg.norm(prototype)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        denominator = candidate_norms * prototype_norm
        scores = np.zeros(len(candidate_matrix), dtype=float)
        valid = denominator > 0
        scores[valid] = candidate_matrix[valid] @ prototype / denominator[valid]
    return scores


def _ranks(scores: np.ndarray, candidate_ids: Sequence[str]) -> np.ndarray:
    """Return deterministic 1-indexed ranks, breaking score ties by glycan ID."""

    ids = np.asarray(candidate_ids, dtype=str)
    order = np.lexsort((ids, -scores))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _known_positive_average_precision(ranks: np.ndarray) -> float:
    """Average precision for the held-out known-positive relevance set.

    Unlabeled candidates are treated as nonrelevant for this calculation. The
    statistic is not a formal lower bound on average precision under complete
    labels because unobserved positives can change average precision in either
    direction.
    """

    if len(ranks) == 0:
        return float("nan")
    ordered = np.sort(ranks)
    precisions = np.arange(1, len(ordered) + 1, dtype=float) / ordered
    return float(precisions.mean())


def _fold_metrics(test_ranks: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    if len(test_ranks) == 0:
        return {}
    metrics: dict[str, float] = {
        "mrr": float(1.0 / test_ranks.min()),
        "mean_rank": float(test_ranks.mean()),
        "known_positive_ap": _known_positive_average_precision(test_ranks),
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(test_ranks <= k))
    return metrics


def _agent_summary(per_fold: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    metric_columns = [
        "mrr",
        "mean_rank",
        "known_positive_ap",
        *[f"recall@{k}" for k in ks],
    ]
    aggregations: dict[str, str] = {column: "mean" for column in metric_columns}
    aggregations.update(
        {"fold": "count", "n_train_positives": "min", "n_test_positives": "sum"}
    )
    summary = per_fold.groupby("agent_id", as_index=False).agg(aggregations).rename(
        columns={
            "fold": "n_folds_evaluated",
            "n_train_positives": "minimum_train_positives",
            "n_test_positives": "n_held_out_positive_evaluations",
        }
    )
    summary["total_known_positives"] = summary["n_held_out_positive_evaluations"]
    summary["positive_support_stratum"] = pd.cut(
        summary["total_known_positives"],
        bins=[1, 2, 5, 10, float("inf")],
        labels=["2", "3-5", "6-10", ">10"],
        right=True,
    ).astype(str)
    return summary


def _bootstrap_agent_ci(
    per_agent: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = per_agent[metric].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def evaluate_positive_kfold(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
    agent_column: str = DEFAULT_AGENT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    n_splits: int = 5,
    minimum_positives: int = 2,
    ks: Sequence[int] = (5, 10),
    seed: int = 42,
    n_bootstrap: int = 2000,
    feature_permutation_seed: int | None = None,
    preprocessing: str = "standardize",
    fold_manifest: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Evaluate unseen positive relations with agent-stratified positive folds.

    For every agent, known positives are partitioned into folds. The prototype
    is constructed only from training positives. Training positives are removed
    from the candidate pool, and held-out positives are ranked against unlabeled
    glycans. Feature imputation and scaling are fitted without the held-out
    positive glycans for that fold.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if minimum_positives < 2:
        raise ValueError("minimum_positives must be at least 2 for held-out evaluation")

    validate_inputs(
        features,
        labels,
        id_column=id_column,
        agent_column=agent_column,
        label_column=label_column,
    )
    feature_columns = list(
        feature_columns or infer_feature_columns(features, id_column=id_column)
    )

    feature_table = features.copy()
    feature_table[id_column] = feature_table[id_column].astype(str)
    feature_table = feature_table.sort_values(id_column).reset_index(drop=True)
    glycan_ids = feature_table[id_column].to_numpy(dtype=str)
    glycan_to_index = {glycan_id: index for index, glycan_id in enumerate(glycan_ids)}
    raw_matrix = feature_table[feature_columns].to_numpy(dtype=float)
    if feature_permutation_seed is not None:
        permutation_rng = np.random.default_rng(feature_permutation_seed)
        raw_matrix = raw_matrix[permutation_rng.permutation(len(raw_matrix))]

    positives = positive_pairs(
        labels,
        id_column=id_column,
        agent_column=agent_column,
        label_column=label_column,
    )
    positives = positives[positives[id_column].isin(glycan_to_index)]

    if fold_manifest is not None:
        required_manifest_columns = {agent_column, id_column, "fold"}
        missing_manifest_columns = required_manifest_columns - set(fold_manifest.columns)
        if missing_manifest_columns:
            raise ValueError(
                f"Fold manifest missing columns: {sorted(missing_manifest_columns)}"
            )
        fold_manifest = fold_manifest[[agent_column, id_column, "fold"]].copy()
        fold_manifest[agent_column] = fold_manifest[agent_column].astype(str)
        fold_manifest[id_column] = fold_manifest[id_column].astype(str)
        if fold_manifest.duplicated([agent_column, id_column]).any():
            raise ValueError("Fold manifest assigns an agent--glycan positive more than once")
        positive_index = set(zip(positives[agent_column], positives[id_column]))
        manifest_index = set(zip(fold_manifest[agent_column], fold_manifest[id_column]))
        unexpected_manifest_pairs = sorted(manifest_index - positive_index)
        if unexpected_manifest_pairs:
            raise ValueError(
                "Fold manifest contains relations not present as joined positives: "
                f"{unexpected_manifest_pairs[:5]}"
            )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    all_indices = np.arange(len(feature_table), dtype=int)

    for agent_id, agent_frame in positives.groupby(agent_column, sort=True):
        agent_positive_ids = sorted(agent_frame[id_column].unique().tolist())
        if len(agent_positive_ids) < minimum_positives:
            continue

        if fold_manifest is not None:
            agent_manifest = fold_manifest[
                fold_manifest[agent_column].astype(str) == str(agent_id)
            ]
            assigned_ids = set(agent_manifest[id_column].astype(str))
            if assigned_ids != set(agent_positive_ids):
                raise ValueError(
                    f"Fold manifest does not exactly cover positives for agent {agent_id}: "
                    f"missing={sorted(set(agent_positive_ids) - assigned_ids)}, "
                    f"extra={sorted(assigned_ids - set(agent_positive_ids))}"
                )
            positive_folds = [
                group[id_column].astype(str).to_numpy()
                for _, group in agent_manifest.groupby("fold", sort=True)
                if len(group)
            ]
        else:
            shuffled = np.asarray(agent_positive_ids, dtype=str)
            rng.shuffle(shuffled)
            fold_count = min(n_splits, len(shuffled))
            positive_folds = [
                fold for fold in np.array_split(shuffled, fold_count) if len(fold)
            ]

        for fold_index, test_positive_ids in enumerate(positive_folds):
            test_positive_set = set(test_positive_ids.tolist())
            train_positive_ids = [
                glycan_id for glycan_id in agent_positive_ids if glycan_id not in test_positive_set
            ]
            if not train_positive_ids:
                continue

            train_positive_indices = np.asarray(
                [glycan_to_index[glycan_id] for glycan_id in train_positive_ids], dtype=int
            )
            test_positive_indices = np.asarray(
                [glycan_to_index[glycan_id] for glycan_id in test_positive_ids], dtype=int
            )

            preprocessing_rows = np.setdiff1d(all_indices, test_positive_indices)
            scaled_matrix = _preprocess_matrix(
                raw_matrix, preprocessing_rows, preprocessing
            )
            prototype = scaled_matrix[train_positive_indices].mean(axis=0)

            candidate_indices = np.setdiff1d(all_indices, train_positive_indices)
            candidate_ids = glycan_ids[candidate_indices]
            scores = _cosine_scores(scaled_matrix[candidate_indices], prototype)
            candidate_ranks = _ranks(scores, candidate_ids)
            rank_by_index = {
                int(index): int(rank)
                for index, rank in zip(candidate_indices.tolist(), candidate_ranks.tolist())
            }
            held_out_ranks = np.asarray(
                [rank_by_index[int(index)] for index in test_positive_indices], dtype=int
            )

            rows.append(
                {
                    "agent_id": str(agent_id),
                    "fold": fold_index,
                    "n_train_positives": len(train_positive_indices),
                    "n_test_positives": len(test_positive_indices),
                    "n_candidates": len(candidate_indices),
                    "preprocessing_fit_glycans": len(preprocessing_rows),
                    **_fold_metrics(held_out_ranks, ks),
                }
            )

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        raise ValueError(
            "No agents were evaluable. Check the feature/label join and minimum-positive threshold."
        )
    per_agent = _agent_summary(per_fold, ks)

    aggregate: dict[str, float | int | str] = {
        "protocol": "agent-stratified positive holdout",
        "prototype_scope": "training positives only",
        "candidate_policy": "held-out positives plus unlabeled glycans; training positives excluded",
        "preprocessing_scope": "all feature rows except held-out positive glycans",
        "preprocessing_mode": preprocessing,
        "split_source": "released fold manifest" if fold_manifest is not None else "seeded runtime assignment",
        "n_glycans": len(feature_table),
        "n_agents_evaluated": len(per_agent),
        "n_agent_folds": len(per_fold),
        "minimum_positives": minimum_positives,
        "requested_splits": n_splits,
        "seed": seed,
        "feature_permutation_seed": (
            feature_permutation_seed if feature_permutation_seed is not None else "none"
        ),
    }
    metrics = ["mrr", "mean_rank", "known_positive_ap", *[f"recall@{k}" for k in ks]]
    for offset, metric in enumerate(metrics):
        aggregate[metric] = float(per_agent[metric].mean())
        if n_bootstrap > 0:
            lower, upper = _bootstrap_agent_ci(
                per_agent,
                metric,
                seed=seed + offset,
                n_bootstrap=n_bootstrap,
            )
            aggregate[f"{metric}_ci_lower"] = lower
            aggregate[f"{metric}_ci_upper"] = upper

    return EvaluationResult(per_fold=per_fold, per_agent=per_agent, aggregate=aggregate)


def _aggregate_simulated_folds(per_fold: pd.DataFrame, ks: Sequence[int]) -> dict[str, float]:
    per_agent = _agent_summary(per_fold, ks)
    metrics = ["mrr", "mean_rank", "known_positive_ap", *[f"recall@{k}" for k in ks]]
    return {metric: float(per_agent[metric].mean()) for metric in metrics}


def run_null_controls(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    observed: EvaluationResult,
    *,
    feature_columns: Sequence[str] | None = None,
    n_splits: int = 5,
    minimum_positives: int = 2,
    ks: Sequence[int] = (5, 10),
    seed: int = 42,
    n_repeats: int = 200,
    preprocessing: str = "standardize",
    fold_manifest: pd.DataFrame | None = None,
) -> NullControlResult:
    """Generate random-ranking and feature-permutation null distributions."""

    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    rng = np.random.default_rng(seed + 10_000)
    rows: list[dict[str, float | int | str]] = []

    for repeat in range(n_repeats):
        random_fold_rows = []
        for fold_row in observed.per_fold.to_dict(orient="records"):
            n_candidates = int(fold_row["n_candidates"])
            n_test = int(fold_row["n_test_positives"])
            simulated_ranks = rng.choice(
                np.arange(1, n_candidates + 1, dtype=int), size=n_test, replace=False
            )
            random_fold_rows.append(
                {
                    **fold_row,
                    **_fold_metrics(simulated_ranks, ks),
                }
            )
        random_metrics = _aggregate_simulated_folds(pd.DataFrame(random_fold_rows), ks)
        rows.append({"mode": "random", "repeat": repeat, **random_metrics})

        permuted = evaluate_positive_kfold(
            features,
            labels,
            feature_columns=feature_columns,
            n_splits=n_splits,
            minimum_positives=minimum_positives,
            ks=ks,
            seed=seed,
            n_bootstrap=0,
            feature_permutation_seed=seed + 20_000 + repeat,
            preprocessing=preprocessing,
            fold_manifest=fold_manifest,
        )
        permuted_metrics = {
            metric: permuted.aggregate[metric]
            for metric in random_metrics
        }
        rows.append({"mode": "feature_permutation", "repeat": repeat, **permuted_metrics})

    distributions = pd.DataFrame(rows)
    pvalues: dict[str, float | int] = {"n_repeats": n_repeats}
    metric_names = ["mrr", "known_positive_ap", *[f"recall@{k}" for k in ks]]
    for mode in ["random", "feature_permutation"]:
        mode_frame = distributions[distributions["mode"] == mode]
        for metric in metric_names:
            observed_value = float(observed.aggregate[metric])
            exceedances = int(np.sum(mode_frame[metric].to_numpy(dtype=float) >= observed_value))
            pvalues[f"{metric}_vs_{mode}"] = float((exceedances + 1) / (n_repeats + 1))

    return NullControlResult(distributions=distributions, pvalues=pvalues)
