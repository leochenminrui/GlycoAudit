"""Leakage-controlled cross-source transfer evaluation and size controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .pu_ranking import (
    DEFAULT_AGENT_COLUMN,
    DEFAULT_EVIDENCE_COLUMN,
    DEFAULT_ID_COLUMN,
    _bootstrap_agent_ci,
    _clean_matrix,
    _cosine_scores,
    _fold_metrics,
    _ranks,
    infer_feature_columns,
)


@dataclass(frozen=True)
class CrossSourceResult:
    per_agent: pd.DataFrame
    candidate_audit: pd.DataFrame
    aggregate: dict[str, float | int | str]
    balance: pd.DataFrame


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def _candidate_ids(
    manifest: pd.DataFrame,
    *,
    direction: str,
    source: str,
    agent_id: str | None,
    id_column: str,
    source_column: str,
) -> list[str]:
    selected = manifest.copy()
    if "direction" in selected.columns:
        selected = selected[selected["direction"].astype(str) == direction]
    selected = selected[selected[source_column].astype(str) == source]
    if "is_candidate" in selected.columns:
        selected = selected[_as_bool(selected["is_candidate"])]
    if "agent_id" in selected.columns and agent_id is not None:
        agent_values = selected["agent_id"].fillna("").astype(str)
        selected = selected[(agent_values == "") | (agent_values == agent_id)]
    return sorted(selected[id_column].dropna().astype(str).unique().tolist())


def _impute(matrix: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    values = matrix.astype(float, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values[fit_rows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing_rows, missing_columns = np.where(np.isnan(values))
    values[missing_rows, missing_columns] = medians[missing_columns]
    return values


def _residualized_matrix(
    feature_table: pd.DataFrame,
    feature_columns: Sequence[str],
    size_columns: Sequence[str],
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    non_size_columns = [column for column in feature_columns if column not in size_columns]
    present_size = [column for column in size_columns if column in feature_table.columns]
    if not non_size_columns:
        raise ValueError("No non-size features remain for residualization")
    if not present_size:
        raise ValueError("None of the declared size columns are present")

    response = _impute(feature_table[non_size_columns].to_numpy(dtype=float), fit_rows)
    size = _impute(feature_table[present_size].to_numpy(dtype=float), fit_rows)
    design = np.column_stack([np.ones(len(size)), size])
    coefficients, *_ = np.linalg.lstsq(design[fit_rows], response[fit_rows], rcond=None)
    residuals = response - design @ coefficients
    return _clean_matrix(residuals, fit_rows), [f"residual_{column}" for column in non_size_columns]


def greedy_size_match(
    features: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    size_columns: Sequence[str] = ("n_atoms", "n_residues"),
    id_column: str = DEFAULT_ID_COLUMN,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Create deterministic equal-size source subsets by nearest size distance."""

    present_size = [column for column in size_columns if column in features.columns]
    if not present_size:
        raise ValueError("Size matching requires at least one declared size column")
    table = features[[id_column, *present_size]].copy()
    table[id_column] = table[id_column].astype(str)
    table = table.drop_duplicates(id_column).set_index(id_column)
    train = [glycan for glycan in train_ids if glycan in table.index]
    test = [glycan for glycan in test_ids if glycan in table.index]
    if not train or not test:
        raise ValueError("Size matching requires non-empty train and test glycan sets")

    combined = table.loc[train + test, present_size].apply(pd.to_numeric, errors="coerce")
    combined = combined.fillna(combined.median()).fillna(0.0)
    standard_deviation = combined.std(ddof=0).replace(0, 1.0)
    standardized = (combined - combined.mean()) / standard_deviation

    if len(train) <= len(test):
        anchors, pool, anchor_is_train = train, test, True
    else:
        anchors, pool, anchor_is_train = test, train, False
    unused = set(pool)
    pairs = []
    for anchor in sorted(anchors):
        anchor_vector = standardized.loc[anchor].to_numpy(dtype=float)
        distances = [
            (
                float(np.linalg.norm(anchor_vector - standardized.loc[candidate].to_numpy(dtype=float))),
                candidate,
            )
            for candidate in unused
        ]
        distance, matched = min(distances, key=lambda value: (value[0], value[1]))
        unused.remove(matched)
        train_id, test_id = (anchor, matched) if anchor_is_train else (matched, anchor)
        pairs.append({"train_glycan": train_id, "test_glycan": test_id, "size_distance": distance})

    pair_table = pd.DataFrame(pairs)
    return (
        sorted(pair_table["train_glycan"].tolist()),
        sorted(pair_table["test_glycan"].tolist()),
        pair_table,
    )


def _balance_table(
    features: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    size_columns: Sequence[str],
    id_column: str,
) -> pd.DataFrame:
    rows = []
    for source_role, ids in [("train", train_ids), ("test", test_ids)]:
        subset = features[features[id_column].astype(str).isin(ids)]
        for column in size_columns:
            if column not in subset.columns:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            rows.append(
                {
                    "source_role": source_role,
                    "feature": column,
                    "n": len(values),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def evaluate_cross_source(
    features: pd.DataFrame,
    evidence: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    *,
    train_source: str,
    test_source: str,
    feature_columns: Sequence[str] | None = None,
    size_columns: Sequence[str] = ("n_atoms", "n_residues"),
    representation_mode: str = "full",
    id_column: str = DEFAULT_ID_COLUMN,
    agent_column: str = DEFAULT_AGENT_COLUMN,
    evidence_column: str = DEFAULT_EVIDENCE_COLUMN,
    source_column: str = "data_source",
    minimum_train_positives: int = 2,
    minimum_test_positives: int = 1,
    exclude_train_glycan_overlap: bool = True,
    ks: Sequence[int] = (5, 10),
    seed: int = 42,
    n_bootstrap: int = 2000,
) -> CrossSourceResult:
    """Evaluate transfer with prototypes and preprocessing fitted on train source only."""

    direction = f"{train_source}->{test_source}"
    required_evidence = {id_column, agent_column, evidence_column, source_column}
    required_manifest = {id_column, source_column}
    missing_evidence = sorted(required_evidence - set(evidence.columns))
    missing_manifest = sorted(required_manifest - set(candidate_manifest.columns))
    if missing_evidence:
        raise ValueError(f"Evidence table missing columns: {missing_evidence}")
    if missing_manifest:
        raise ValueError(f"Candidate manifest missing columns: {missing_manifest}")
    if features[id_column].duplicated().any():
        raise ValueError("Feature table must contain one row per glycan")

    feature_columns = list(feature_columns or infer_feature_columns(features, id_column=id_column))
    feature_table = features.copy()
    feature_table[id_column] = feature_table[id_column].astype(str)
    feature_table = feature_table.sort_values(id_column).reset_index(drop=True)
    glycan_ids = feature_table[id_column].to_numpy(dtype=str)
    index_by_glycan = {glycan: index for index, glycan in enumerate(glycan_ids)}

    normalized_evidence = evidence.copy()
    normalized_evidence[id_column] = normalized_evidence[id_column].astype(str)
    normalized_evidence[agent_column] = normalized_evidence[agent_column].astype(str)
    normalized_evidence[evidence_column] = (
        normalized_evidence[evidence_column].astype(str).str.lower()
    )
    positives = normalized_evidence[
        normalized_evidence[evidence_column] == "positive"
    ].drop_duplicates([source_column, agent_column, id_column])

    global_train_ids = _candidate_ids(
        candidate_manifest,
        direction=direction,
        source=train_source,
        agent_id=None,
        id_column=id_column,
        source_column=source_column,
    )
    global_test_ids = _candidate_ids(
        candidate_manifest,
        direction=direction,
        source=test_source,
        agent_id=None,
        id_column=id_column,
        source_column=source_column,
    )
    global_train_ids = [glycan for glycan in global_train_ids if glycan in index_by_glycan]
    global_test_ids = [glycan for glycan in global_test_ids if glycan in index_by_glycan]
    if not global_train_ids or not global_test_ids:
        raise ValueError("Candidate manifest yielded an empty train or test source universe")

    match_table = pd.DataFrame()
    if representation_mode == "size_matched":
        global_train_ids, global_test_ids, match_table = greedy_size_match(
            feature_table,
            global_train_ids,
            global_test_ids,
            size_columns=size_columns,
            id_column=id_column,
        )

    train_fit_rows = np.asarray(
        [index_by_glycan[glycan] for glycan in global_train_ids], dtype=int
    )
    if representation_mode in {"full", "size_matched"}:
        matrix = _clean_matrix(feature_table[feature_columns].to_numpy(dtype=float), train_fit_rows)
        used_feature_columns = feature_columns
    elif representation_mode == "size_excluded":
        used_feature_columns = [column for column in feature_columns if column not in size_columns]
        if not used_feature_columns:
            raise ValueError("No features remain after excluding size columns")
        matrix = _clean_matrix(
            feature_table[used_feature_columns].to_numpy(dtype=float), train_fit_rows
        )
    elif representation_mode == "residualized":
        matrix, used_feature_columns = _residualized_matrix(
            feature_table, feature_columns, size_columns, train_fit_rows
        )
    else:
        raise ValueError(
            "representation_mode must be one of full, size_excluded, residualized, size_matched"
        )

    train_positive_counts = (
        positives[positives[source_column].astype(str) == train_source]
        .groupby(agent_column)[id_column].nunique()
    )
    test_positive_counts = (
        positives[positives[source_column].astype(str) == test_source]
        .groupby(agent_column)[id_column].nunique()
    )
    shared_agents = sorted(
        set(train_positive_counts[train_positive_counts >= minimum_train_positives].index)
        & set(test_positive_counts[test_positive_counts >= minimum_test_positives].index)
    )

    metric_rows = []
    candidate_rows = []
    for agent_id in shared_agents:
        train_candidates = _candidate_ids(
            candidate_manifest,
            direction=direction,
            source=train_source,
            agent_id=agent_id,
            id_column=id_column,
            source_column=source_column,
        )
        test_candidates = _candidate_ids(
            candidate_manifest,
            direction=direction,
            source=test_source,
            agent_id=agent_id,
            id_column=id_column,
            source_column=source_column,
        )
        train_universe = set(train_candidates) & set(global_train_ids)
        test_universe = set(test_candidates) & set(global_test_ids)
        agent_train_positives = sorted(
            set(
                positives.loc[
                    (positives[source_column].astype(str) == train_source)
                    & (positives[agent_column] == agent_id),
                    id_column,
                ]
            )
            & train_universe
        )
        test_positive_ids = set(
            positives.loc[
                (positives[source_column].astype(str) == test_source)
                & (positives[agent_column] == agent_id),
                id_column,
            ]
        ) & test_universe

        overlap = set(agent_train_positives) & test_universe
        if exclude_train_glycan_overlap:
            test_universe -= set(agent_train_positives)
            test_positive_ids -= set(agent_train_positives)
        if (
            len(agent_train_positives) < minimum_train_positives
            or len(test_positive_ids) < minimum_test_positives
            or not test_universe
        ):
            continue

        train_indices = np.asarray(
            [index_by_glycan[glycan] for glycan in agent_train_positives], dtype=int
        )
        candidate_ids = sorted(glycan for glycan in test_universe if glycan in index_by_glycan)
        candidate_indices = np.asarray(
            [index_by_glycan[glycan] for glycan in candidate_ids], dtype=int
        )
        prototype = matrix[train_indices].mean(axis=0)
        scores = _cosine_scores(matrix[candidate_indices], prototype)
        ranks = _ranks(scores, candidate_ids)
        rank_by_id = dict(zip(candidate_ids, ranks.tolist()))
        held_out_ranks = np.asarray(
            [rank_by_id[glycan] for glycan in sorted(test_positive_ids)], dtype=int
        )
        metrics = _fold_metrics(held_out_ranks, ks)
        if not metrics:
            continue
        metric_rows.append(
            {
                "agent_id": agent_id,
                "n_train_positives": len(agent_train_positives),
                "n_test_positives": len(test_positive_ids),
                "n_test_candidates": len(candidate_ids),
                "n_train_overlap_excluded": len(overlap) if exclude_train_glycan_overlap else 0,
                **metrics,
            }
        )
        for glycan, score, rank in zip(candidate_ids, scores.tolist(), ranks.tolist()):
            candidate_rows.append(
                {
                    "direction": direction,
                    "representation_mode": representation_mode,
                    "agent_id": agent_id,
                    id_column: glycan,
                    "is_test_positive": glycan in test_positive_ids,
                    "score": score,
                    "rank": rank,
                    "was_train_positive": glycan in set(agent_train_positives),
                }
            )

    per_agent = pd.DataFrame(metric_rows)
    if per_agent.empty:
        raise ValueError("No shared agents remained evaluable under the declared source protocol")
    candidate_audit = pd.DataFrame(candidate_rows)
    balance = _balance_table(
        feature_table,
        global_train_ids,
        global_test_ids,
        size_columns,
        id_column,
    )
    if not match_table.empty:
        match_summary = {
            "n_size_matched_pairs": len(match_table),
            "mean_size_match_distance": float(match_table["size_distance"].mean()),
        }
    else:
        match_summary = {"n_size_matched_pairs": 0, "mean_size_match_distance": "not_applicable"}

    aggregate: dict[str, float | int | str] = {
        "protocol": "training-source-only prototype transfer",
        "direction": direction,
        "representation_mode": representation_mode,
        "prototype_scope": f"positive evidence from {train_source} only",
        "preprocessing_scope": f"candidate universe from {train_source} only",
        "candidate_scope": f"frozen {test_source} candidate manifest",
        "exclude_train_glycan_overlap": str(exclude_train_glycan_overlap).lower(),
        "n_train_source_glycans": len(global_train_ids),
        "n_test_source_glycans": len(global_test_ids),
        "n_agents_evaluated": len(per_agent),
        "minimum_train_positives": minimum_train_positives,
        "minimum_test_positives": minimum_test_positives,
        "feature_columns": ",".join(used_feature_columns),
        "seed": seed,
        **match_summary,
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
    return CrossSourceResult(per_agent, candidate_audit, aggregate, balance)
