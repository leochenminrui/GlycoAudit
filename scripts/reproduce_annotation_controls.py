#!/usr/bin/env python3
"""Reproduce the reported annotation-degree analyses.

The script reads the public dataset and writes diagnostic artifacts outside the
checksummed data directory. It adds only analyses
that can be derived from the released evidence, folds, candidates, and agent
metadata: a leave-one-agent-out popularity control, the 765-to-697 count funnel,
and a non-speculative agent-identifier audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glycoaudit.evaluation import evaluate_leave_one_agent_out_popularity
from glycoaudit.evaluation.pu_ranking import (
    _cosine_scores,
    _known_positive_average_precision,
    _preprocess_matrix,
    _ranks,
)


SEED = 20260901
RANDOM_TIE_REPEATS = 1_000


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def candidate_ids(release: Path) -> list[str]:
    candidates = pd.read_csv(release / "candidate_manifest.csv", dtype=str)
    included = candidates["included_in_universe"].str.lower().isin({"1", "true", "yes"})
    return sorted(candidates.loc[included, "glytoucan_id"].astype(str).unique())


def positive_pairs(evidence: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return (
        evidence[
            evidence["evidence_status"].astype(str).str.lower().eq("positive")
            & evidence["glytoucan_id"].astype(str).isin(ids)
        ][["agent_id", "glytoucan_id"]]
        .drop_duplicates()
        .astype(str)
    )


def fold_metrics(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mrr": float(1.0 / ranks.min()),
        "recall@5": float(np.mean(ranks <= 5)),
        "known_positive_ap": _known_positive_average_precision(ranks),
    }


def equal_agent_metrics(per_fold: pd.DataFrame) -> dict[str, float]:
    metrics = ["mrr", "recall@5", "known_positive_ap"]
    per_agent = per_fold.groupby("agent_id", as_index=False)[metrics].mean()
    return {metric: float(per_agent[metric].mean()) for metric in metrics}


def popularity_fold_inputs(
    evidence: pd.DataFrame,
    folds: pd.DataFrame,
    ids: list[str],
) -> list[dict[str, object]]:
    positives = positive_pairs(evidence, set(ids))
    global_counts = positives.groupby("glytoucan_id")["agent_id"].nunique()
    candidate_array = np.asarray(sorted(ids), dtype=str)
    inputs: list[dict[str, object]] = []
    for agent_id, agent_frame in positives.groupby("agent_id", sort=True):
        target = set(agent_frame["glytoucan_id"].astype(str))
        if len(target) < 2:
            continue
        agent_folds = folds[folds["agent_id"].astype(str).eq(str(agent_id))]
        if set(agent_folds["glytoucan_id"].astype(str)) != target:
            raise ValueError(f"Fold manifest mismatch for popularity audit agent {agent_id}")
        all_scores = np.asarray(
            [int(global_counts.get(glycan_id, 0)) - int(glycan_id in target) for glycan_id in candidate_array],
            dtype=float,
        )
        for fold, frame in agent_folds.groupby("fold", sort=True):
            held_out = sorted(frame["glytoucan_id"].astype(str))
            training = sorted(target - set(held_out))
            if not training:
                continue
            mask = ~np.isin(candidate_array, np.asarray(training, dtype=str))
            inputs.append(
                {
                    "agent_id": str(agent_id),
                    "fold": int(fold),
                    "candidate_ids": candidate_array[mask],
                    "scores": all_scores[mask],
                    "held_out_ids": held_out,
                }
            )
    return inputs


def popularity_tie_sensitivity(
    evidence: pd.DataFrame,
    folds: pd.DataFrame,
    ids: list[str],
    out: Path,
    wurcs_metrics: dict[str, float],
) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    fold_inputs = popularity_fold_inputs(evidence, folds, ids)
    tie_rows: list[dict[str, object]] = []
    for item in fold_inputs:
        scores = np.asarray(item["scores"], dtype=float)
        counts = pd.Series(scores).value_counts().sort_index(ascending=False)
        tied = counts[counts > 1]
        descending = np.sort(scores)[::-1]
        tie_rows.append(
            {
                "agent_id": item["agent_id"],
                "fold": item["fold"],
                "n_candidates": len(scores),
                "n_tied_score_groups": int(len(tied)),
                "largest_tied_group": int(tied.max()) if len(tied) else 1,
                "highest_score_position_tied": bool(counts.iloc[0] > 1),
                "top5_boundary_cuts_tied_group": bool(
                    len(descending) > 5 and descending[4] == descending[5]
                ),
                "candidates_in_ties": int(tied.sum()),
                "fraction_candidates_in_ties": float(tied.sum() / len(scores)),
            }
        )
    tie_table = pd.DataFrame(tie_rows)
    tie_table.to_csv(out / "tie_statistics_per_agent_fold.csv", index=False)

    random_rows: list[dict[str, float | int]] = []
    for repeat in range(RANDOM_TIE_REPEATS):
        rng = np.random.default_rng(SEED + repeat)
        fold_rows: list[dict[str, float | str]] = []
        for item in fold_inputs:
            candidate_ids_array = np.asarray(item["candidate_ids"], dtype=str)
            scores = np.asarray(item["scores"], dtype=float)
            random_secondary = rng.random(len(scores))
            order = np.lexsort((candidate_ids_array, random_secondary, -scores))
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            rank_by_id = dict(zip(candidate_ids_array, ranks.tolist()))
            held_out_ranks = np.asarray(
                [rank_by_id[glycan_id] for glycan_id in item["held_out_ids"]], dtype=int
            )
            fold_rows.append(
                {
                    "agent_id": str(item["agent_id"]),
                    **fold_metrics(held_out_ranks),
                }
            )
        random_rows.append({"repeat": repeat, "seed": SEED + repeat, **equal_agent_metrics(pd.DataFrame(fold_rows))})
    random_table = pd.DataFrame(random_rows)
    random_table.to_csv(out / "random_tie_metrics_by_seed.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    conclusion_rules = {
        "mrr": "popularity > WURCS",
        "known_positive_ap": "popularity > WURCS",
        "recall@5": "WURCS > popularity",
    }
    for metric, rule in conclusion_rules.items():
        values = random_table[metric].to_numpy(dtype=float)
        if metric == "recall@5":
            stable = values < wurcs_metrics[metric]
        else:
            stable = values > wurcs_metrics[metric]
        metric_rows.append(
            {
                "metric": metric,
                "conclusion_tested": rule,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "percentile_2_5": float(np.quantile(values, 0.025)),
                "percentile_97_5": float(np.quantile(values, 0.975)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "wurcs_value": float(wurcs_metrics[metric]),
                "stable_seed_count": int(stable.sum()),
                "stable_seed_fraction": float(stable.mean()),
            }
        )
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(out / "random_tie_summary.csv", index=False)
    summary = {
        "agent_folds": len(tie_table),
        "folds_with_any_tie": int((tie_table["n_tied_score_groups"] > 0).sum()),
        "folds_with_highest_score_tied": int(tie_table["highest_score_position_tied"].sum()),
        "folds_with_top5_boundary_tie": int(tie_table["top5_boundary_cuts_tied_group"].sum()),
        "median_fraction_candidates_in_ties": float(tie_table["fraction_candidates_in_ties"].median()),
        "maximum_tied_group": int(tie_table["largest_tied_group"].max()),
        "random_tie_repeats": RANDOM_TIE_REPEATS,
        "seed_range": [SEED, SEED + RANDOM_TIE_REPEATS - 1],
        "metric_summary": metric_rows,
        "all_three_conclusions_stable_for_all_seeds": bool(
            metric_table["stable_seed_fraction"].eq(1.0).all()
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def duplicate_group_members(agent_table: pd.DataFrame, key: str) -> set[str]:
    exploded: list[dict[str, str]] = []
    for row in agent_table.itertuples(index=False):
        raw = getattr(row, key)
        if pd.isna(raw):
            values: list[str] = []
        elif key == "external_uniprot_ids":
            values = str(raw).split(";")
        else:
            values = [str(raw)]
        for value in (value for value in values if value and value.lower() != "nan"):
            exploded.append({"agent_id": str(row.agent_id), "group": value})
    if not exploded:
        return set()
    frame = pd.DataFrame(exploded)
    duplicated_groups = set(
        frame.groupby("group")["agent_id"].nunique().loc[lambda x: x > 1].index
    )
    return set(frame.loc[frame["group"].isin(duplicated_groups), "agent_id"])


def entity_exclusion_sensitivity(
    release: Path,
    evidence: pd.DataFrame,
    folds: pd.DataFrame,
    ids: list[str],
    out: Path,
) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    agents = pd.read_csv(
        out.parent / "agent_identifier_audit/agent_identifier_table.csv", dtype=str
    )
    evaluated = agents[agents["evaluated_in_primary_protocol"].str.lower().eq("true")].copy()
    name_members = duplicate_group_members(evaluated, "normalized_name")
    uniprot_members = duplicate_group_members(evaluated, "external_uniprot_ids")
    scenarios = {
        "exclude_shared_uniprot_group_members": uniprot_members,
        "exclude_exact_normalized_name_duplicate_members": name_members,
        "exclude_union_of_known_duplicate_members": name_members | uniprot_members,
    }
    wurcs = pd.read_csv(release / "results/wurcs/metrics_per_agent.csv").set_index("agent_id")
    rows: list[dict[str, object]] = []
    for scenario, excluded in scenarios.items():
        filtered_evidence = evidence[~evidence["agent_id"].astype(str).isin(excluded)].copy()
        filtered_folds = folds[~folds["agent_id"].astype(str).isin(excluded)].copy()
        result = evaluate_leave_one_agent_out_popularity(
            filtered_evidence,
            ids,
            minimum_positives=2,
            ks=(5, 10),
            seed=42,
            n_bootstrap=2_000,
            fold_manifest=filtered_folds,
        )
        popularity_agents = result.per_agent.set_index("agent_id")
        shared = popularity_agents.index.intersection(wurcs.index)
        row: dict[str, object] = {
            "scenario": scenario,
            "excluded_agent_identifiers": len(excluded),
            "remaining_evaluable_agents": len(result.per_agent),
            "remaining_agent_folds": len(result.per_fold),
            "remaining_held_out_positive_relations": int(result.per_agent["n_held_out_positive_evaluations"].sum()),
            "mrr": result.aggregate["mrr"],
            "mrr_ci_lower": result.aggregate["mrr_ci_lower"],
            "mrr_ci_upper": result.aggregate["mrr_ci_upper"],
            "recall@5": result.aggregate["recall@5"],
            "recall@5_ci_lower": result.aggregate["recall@5_ci_lower"],
            "recall@5_ci_upper": result.aggregate["recall@5_ci_upper"],
            "known_positive_ap": result.aggregate["known_positive_ap"],
            "known_positive_ap_ci_lower": result.aggregate["known_positive_ap_ci_lower"],
            "known_positive_ap_ci_upper": result.aggregate["known_positive_ap_ci_upper"],
        }
        rng = np.random.default_rng(SEED)
        for metric in ["mrr", "recall@5", "known_positive_ap"]:
            differences = (
                wurcs.loc[shared, metric].astype(float)
                - popularity_agents.loc[shared, metric].astype(float)
            ).to_numpy()
            boot = rng.choice(differences, size=(10_000, len(differences)), replace=True).mean(axis=1)
            prefix = metric.replace("@", "_at_")
            row[f"wurcs_{prefix}"] = float(wurcs.loc[shared, metric].astype(float).mean())
            row[f"wurcs_minus_popularity_{prefix}"] = float(differences.mean())
            row[f"wurcs_minus_popularity_{prefix}_ci_lower"] = float(np.quantile(boot, 0.025))
            row[f"wurcs_minus_popularity_{prefix}_ci_upper"] = float(np.quantile(boot, 0.975))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "entity_exclusion_sensitivity.csv", index=False)
    summary = {
        "scenarios": rows,
        "known_duplicate_definition": "all members of groups sharing an exact normalized name or recorded external UniProt mediator identifier",
        "manual_approximate_alias_inference": False,
        "scope_warning": "These exclusions remove only duplicate groups identifiable from released metadata and do not establish complete biological-entity independence.",
    }
    write_json(out / "summary.json", summary)
    return summary


def popularity_stratified_diagnostic(
    release: Path,
    evidence: pd.DataFrame,
    folds: pd.DataFrame,
    ids: list[str],
    out: Path,
) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(release / "wurcs_features.csv")
    features["glytoucan_id"] = features["glytoucan_id"].astype(str)
    features = features[features["glytoucan_id"].isin(ids)].sort_values("glytoucan_id").reset_index(drop=True)
    feature_columns = [column for column in features.columns if column.startswith("wurcs_hash_")]
    glycan_ids = features["glytoucan_id"].to_numpy(dtype=str)
    glycan_to_index = {glycan_id: index for index, glycan_id in enumerate(glycan_ids)}
    raw_matrix = features[feature_columns].to_numpy(dtype=float)
    all_indices = np.arange(len(features), dtype=int)

    positives = positive_pairs(evidence, set(ids))
    global_counts = positives.groupby("glytoucan_id")["agent_id"].nunique()
    rows: list[dict[str, object]] = []
    for agent_id, agent_frame in positives.groupby("agent_id", sort=True):
        target = set(agent_frame["glytoucan_id"].astype(str))
        if len(target) < 2:
            continue
        agent_folds = folds[folds["agent_id"].astype(str).eq(str(agent_id))]
        for fold, frame in agent_folds.groupby("fold", sort=True):
            held_out = sorted(frame["glytoucan_id"].astype(str))
            training = sorted(target - set(held_out))
            test_indices = np.asarray([glycan_to_index[glycan_id] for glycan_id in held_out], dtype=int)
            train_indices = np.asarray([glycan_to_index[glycan_id] for glycan_id in training], dtype=int)
            preprocessing_rows = np.setdiff1d(all_indices, test_indices)
            scaled = _preprocess_matrix(raw_matrix, preprocessing_rows, "tfidf")
            prototype = scaled[train_indices].mean(axis=0)
            candidate_indices = np.setdiff1d(all_indices, train_indices)
            candidate_ids_array = glycan_ids[candidate_indices]
            wurcs_ranks = _ranks(_cosine_scores(scaled[candidate_indices], prototype), candidate_ids_array)
            popularity_scores = np.asarray(
                [int(global_counts.get(glycan_id, 0)) - int(glycan_id in target) for glycan_id in candidate_ids_array],
                dtype=float,
            )
            popularity_ranks = _ranks(popularity_scores, candidate_ids_array)
            wurcs_by_id = dict(zip(candidate_ids_array, wurcs_ranks.tolist()))
            popularity_by_id = dict(zip(candidate_ids_array, popularity_ranks.tolist()))
            wurcs_order = {glycan_id: index for index, glycan_id in enumerate(sorted(held_out, key=lambda x: wurcs_by_id[x]), 1)}
            popularity_order = {glycan_id: index for index, glycan_id in enumerate(sorted(held_out, key=lambda x: popularity_by_id[x]), 1)}
            n_held_out = len(held_out)
            for glycan_id in held_out:
                degree = int(global_counts.get(glycan_id, 0)) - 1
                wrank = int(wurcs_by_id[glycan_id])
                prank = int(popularity_by_id[glycan_id])
                rows.append(
                    {
                        "agent_id": str(agent_id),
                        "fold": int(fold),
                        "glytoucan_id": glycan_id,
                        "held_out_relation_id": f"{agent_id}::{glycan_id}",
                        "n_held_out_positives_in_fold": n_held_out,
                        "leave_one_agent_out_annotation_degree": degree,
                        "wurcs_rank": wrank,
                        "popularity_rank": prank,
                        "wurcs_ordered_positive_index_within_fold": wurcs_order[glycan_id],
                        "popularity_ordered_positive_index_within_fold": popularity_order[glycan_id],
                        "wurcs_top1": int(wrank == 1),
                        "popularity_top1": int(prank == 1),
                        "wurcs_recall_at_5": int(wrank <= 5),
                        "popularity_recall_at_5": int(prank <= 5),
                        "wurcs_reciprocal_rank": 1.0 / wrank,
                        "popularity_reciprocal_rank": 1.0 / prank,
                        "wurcs_ap_known_fold_contribution": wurcs_order[glycan_id] / (n_held_out * wrank),
                        "popularity_ap_known_fold_contribution": popularity_order[glycan_id] / (n_held_out * prank),
                    }
                )
    per_positive = pd.DataFrame(rows)
    q1, q2 = per_positive["leave_one_agent_out_annotation_degree"].quantile([1 / 3, 2 / 3]).tolist()
    lower = int(np.floor(q1))
    upper = int(np.floor(q2))
    per_positive["degree_stratum"] = np.select(
        [
            per_positive["leave_one_agent_out_annotation_degree"] <= lower,
            per_positive["leave_one_agent_out_annotation_degree"] <= upper,
        ],
        ["low", "middle"],
        default="high",
    )
    long_rows: list[pd.DataFrame] = []
    for method in ["wurcs", "popularity"]:
        long_rows.append(
            per_positive[
                [
                    "held_out_relation_id",
                    "agent_id",
                    "fold",
                    "glytoucan_id",
                    "n_held_out_positives_in_fold",
                    "leave_one_agent_out_annotation_degree",
                    "degree_stratum",
                    f"{method}_rank",
                    f"{method}_ordered_positive_index_within_fold",
                    f"{method}_top1",
                    f"{method}_recall_at_5",
                    f"{method}_reciprocal_rank",
                    f"{method}_ap_known_fold_contribution",
                ]
            ].rename(
                columns={
                    "agent_id": "target_agent_id",
                    "leave_one_agent_out_annotation_degree": "annotation_degree_non_target_agents",
                    f"{method}_rank": "held_out_rank",
                    f"{method}_ordered_positive_index_within_fold": "ordered_positive_index_within_fold",
                    f"{method}_top1": "top1",
                    f"{method}_recall_at_5": "recall_at_5",
                    f"{method}_reciprocal_rank": "reciprocal_rank",
                    f"{method}_ap_known_fold_contribution": "ap_known_fold_contribution",
                }
            ).assign(method=method)
        )
    per_positive_long = pd.concat(long_rows, ignore_index=True)[
        [
            "held_out_relation_id",
            "target_agent_id",
            "fold",
            "method",
            "glytoucan_id",
            "held_out_rank",
            "ordered_positive_index_within_fold",
            "n_held_out_positives_in_fold",
            "annotation_degree_non_target_agents",
            "degree_stratum",
            "top1",
            "recall_at_5",
            "reciprocal_rank",
            "ap_known_fold_contribution",
        ]
    ]
    per_positive_long.to_csv(out / "held_out_positive_ranks_by_degree.csv", index=False)

    fold_index = per_positive[["agent_id", "fold"]].drop_duplicates()
    stratum_index = pd.DataFrame({"degree_stratum": ["low", "middle", "high"]})
    fold_stratum_index = fold_index.merge(stratum_index, how="cross")
    exact_contributions: dict[tuple[str, str], float] = {}
    overall_ap: dict[str, float] = {}
    for method in ["wurcs", "popularity"]:
        fold_stratum = (
            per_positive.groupby(["agent_id", "fold", "degree_stratum"], as_index=False)[
                f"{method}_ap_known_fold_contribution"
            ]
            .sum()
            .rename(columns={f"{method}_ap_known_fold_contribution": "contribution"})
        )
        complete = fold_stratum_index.merge(
            fold_stratum,
            on=["agent_id", "fold", "degree_stratum"],
            how="left",
        ).fillna({"contribution": 0.0})
        per_agent_stratum = complete.groupby(
            ["agent_id", "degree_stratum"], as_index=False
        )["contribution"].mean()
        for stratum, value in per_agent_stratum.groupby("degree_stratum")[
            "contribution"
        ].mean().items():
            exact_contributions[(method, stratum)] = float(value)
        per_fold_ap = per_positive.groupby(["agent_id", "fold"], as_index=False)[
            f"{method}_ap_known_fold_contribution"
        ].sum()
        overall_ap[method] = float(
            per_fold_ap.groupby("agent_id")[
                f"{method}_ap_known_fold_contribution"
            ].mean().mean()
        )
        contribution_sum = sum(
            exact_contributions[(method, stratum)]
            for stratum in ["low", "middle", "high"]
        )
        if not np.isclose(contribution_sum, overall_ap[method], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{method} degree contributions do not decompose overall AP_known: "
                f"{contribution_sum} != {overall_ap[method]}"
            )

    released_overall = {
        "wurcs": float(
            json.loads((release / "results/wurcs/aggregate_metrics.json").read_text())[
                "known_positive_ap"
            ]
        ),
        "popularity": float(
            json.loads(
                (out.parent / "popularity_baseline/aggregate_metrics.json").read_text()
            )["known_positive_ap"]
        ),
    }
    for method in ["wurcs", "popularity"]:
        if not np.isclose(overall_ap[method], released_overall[method], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{method} reconstructed AP_known does not match released aggregate: "
                f"{overall_ap[method]} != {released_overall[method]}"
            )

    summary_rows: list[dict[str, object]] = []
    for stratum in ["low", "middle", "high"]:
        frame = per_positive[per_positive["degree_stratum"].eq(stratum)]
        for method in ["wurcs", "popularity"]:
            agent_values = frame.groupby("agent_id", as_index=False).agg(
                mean_rank=(f"{method}_rank", "mean"),
                top1=(f"{method}_top1", "mean"),
                recall_at_5=(f"{method}_recall_at_5", "mean"),
                reciprocal_rank=(f"{method}_reciprocal_rank", "mean"),
            )
            summary_rows.append(
                {
                    "degree_stratum": stratum,
                    "method": method,
                    "degree_min": int(frame["leave_one_agent_out_annotation_degree"].min()),
                    "degree_max": int(frame["leave_one_agent_out_annotation_degree"].max()),
                    "held_out_positive_relations": len(frame),
                    "agents_with_stratum_positives": len(agent_values),
                    **{column: float(agent_values[column].mean()) for column in ["mean_rank", "top1", "recall_at_5", "reciprocal_rank"]},
                    "ap_known_contribution_to_overall_equal_agent_mean": exact_contributions[(method, stratum)],
                }
            )
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(out / "degree_stratified_method_summary.csv", index=False)
    pivot = summary_table.pivot(index="degree_stratum", columns="method")
    contrasts = []
    for stratum in ["low", "middle", "high"]:
        contrasts.append(
            {
                "degree_stratum": stratum,
                "wurcs_minus_popularity_recall_at_5": float(pivot.loc[stratum, ("recall_at_5", "wurcs")] - pivot.loc[stratum, ("recall_at_5", "popularity")]),
                "wurcs_minus_popularity_top1": float(pivot.loc[stratum, ("top1", "wurcs")] - pivot.loc[stratum, ("top1", "popularity")]),
                "wurcs_minus_popularity_reciprocal_rank": float(pivot.loc[stratum, ("reciprocal_rank", "wurcs")] - pivot.loc[stratum, ("reciprocal_rank", "popularity")]),
                "wurcs_minus_popularity_ap_known_contribution_to_overall_equal_agent_mean": float(pivot.loc[stratum, ("ap_known_contribution_to_overall_equal_agent_mean", "wurcs")] - pivot.loc[stratum, ("ap_known_contribution_to_overall_equal_agent_mean", "popularity")]),
                "wurcs_minus_popularity_mean_rank": float(pivot.loc[stratum, ("mean_rank", "wurcs")] - pivot.loc[stratum, ("mean_rank", "popularity")]),
            }
        )
    pd.DataFrame(contrasts).to_csv(out / "degree_stratified_contrasts.csv", index=False)
    table_s8 = summary_table.rename(
        columns={
            "held_out_positive_relations": "held_out_positives_n",
            "agents_with_stratum_positives": "contributing_agents_n",
            "reciprocal_rank": "rr",
        }
    )[
        [
            "degree_stratum",
            "held_out_positives_n",
            "contributing_agents_n",
            "method",
            "mean_rank",
            "top1",
            "recall_at_5",
            "rr",
            "ap_known_contribution_to_overall_equal_agent_mean",
        ]
    ]
    table_s8.to_csv(out / "supplementary_table_s8.csv", index=False)

    summary = {
        "quantile_rule": "Empirical tertile cut points of leave-one-agent-out degree over the 697 held-out positives; integer ties remain in the lower bin.",
        "low_degree_max": lower,
        "middle_degree_max": upper,
        "held_out_positive_relations": len(per_positive),
        "evaluated_agents": int(per_positive["agent_id"].nunique()),
        "agent_folds": int(len(fold_index)),
        "degree_definition": "number of distinct positive relations for the glycan among non-target agents only",
        "ap_known_contribution_definition": "For ordered held-out positive k in agent a, fold f: k / (m_{a,f} * r_{a,f,(k)}); sum within stratum and fold, average folds within agent with zero for absent strata, then average agents equally.",
        "overall_ap_known": overall_ap,
        "released_overall_ap_known": released_overall,
        "ap_known_decomposition_tolerance": 1e-12,
        "method_summary": summary_rows,
        "contrasts": contrasts,
    }
    write_json(out / "summary.json", summary)
    return summary


def popularity_audit(release: Path, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(release / "evidence_records.csv")
    folds = pd.read_csv(release / "positive_holdout_folds.csv")
    ids = candidate_ids(release)
    result = evaluate_leave_one_agent_out_popularity(
        evidence,
        ids,
        minimum_positives=2,
        ks=(5, 10),
        seed=42,
        n_bootstrap=2000,
        fold_manifest=folds,
    )
    result.per_fold.to_csv(out / "metrics_per_agent_fold.csv", index=False)
    result.per_agent.to_csv(out / "metrics_per_agent.csv", index=False)
    write_json(out / "aggregate_metrics.json", result.aggregate)

    positives = positive_pairs(evidence, set(ids))
    evaluable_agents = sorted(result.per_agent["agent_id"].astype(str).unique())
    global_counts = positives.groupby("glytoucan_id")["agent_id"].nunique()
    score_rows: list[dict[str, object]] = []
    for agent_id in evaluable_agents:
        target = set(
            positives.loc[positives["agent_id"].eq(agent_id), "glytoucan_id"].astype(str)
        )
        for glycan_id in ids:
            score_rows.append(
                {
                    "agent_id": agent_id,
                    "glytoucan_id": glycan_id,
                    "target_agent_known_positive": glycan_id in target,
                    "all_agent_positive_count": int(global_counts.get(glycan_id, 0)),
                    "target_agent_contribution": 0,
                    "leave_one_agent_out_popularity": int(global_counts.get(glycan_id, 0))
                    - int(glycan_id in target),
                }
            )
    score_table = pd.DataFrame(score_rows)
    score_table.to_csv(out / "popularity_scores_by_target_agent.csv", index=False)

    wurcs = pd.read_csv(release / "results/wurcs/metrics_per_agent.csv").set_index("agent_id")
    popularity = result.per_agent.set_index("agent_id")
    shared = wurcs.index.intersection(popularity.index)
    rng = np.random.default_rng(SEED)
    paired_rows = []
    bootstrap_rows = []
    for metric in ["mrr", "recall@5", "known_positive_ap"]:
        difference = (
            wurcs.loc[shared, metric].astype(float)
            - popularity.loc[shared, metric].astype(float)
        ).to_numpy()
        samples = rng.choice(
            difference, size=(10_000, len(difference)), replace=True
        ).mean(axis=1)
        paired_rows.append(
            {
                "comparison": "WURCS minus leave-one-agent-out glycan popularity",
                "metric": metric,
                "n_paired_agents": len(difference),
                "mean_paired_difference": float(difference.mean()),
                "bootstrap_ci_lower": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(samples, 0.975)),
                "bootstrap_replicates": 10_000,
                "seed": SEED,
                "inferential_p_value_reported": False,
            }
        )
        bootstrap_rows.extend(
            {
                "metric": metric,
                "replicate": int(index),
                "mean_paired_difference": float(value),
            }
            for index, value in enumerate(samples)
        )
    pd.DataFrame(paired_rows).to_csv(out / "wurcs_vs_popularity_paired.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        out / "wurcs_vs_popularity_bootstrap_replicates.csv", index=False
    )

    assert len(result.per_agent) == 159
    assert len(result.per_fold) == 645
    assert int(result.per_agent["n_held_out_positive_evaluations"].sum()) == 697
    assert len(score_table) == 6996
    assert score_table["target_agent_contribution"].eq(0).all()
    summary = {
        "definition": "For target agent a, count each candidate glycan's distinct positive relations among agents other than a.",
        "candidate_universe": len(ids),
        "evaluable_agents": len(result.per_agent),
        "known_positive_relations": int(
            result.per_agent["n_held_out_positive_evaluations"].sum()
        ),
        "agent_folds": len(result.per_fold),
        "score_rows": len(score_table),
        "target_agent_information_used": 0,
        "tie_policy": "descending popularity score, then ascending GlyTouCan accession",
        "metrics": result.aggregate,
        "paired_comparison": paired_rows,
    }
    write_json(out / "summary.json", summary)
    return summary


def count_funnel(release: Path, out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(release / "evidence_records.csv")
    identifiers = pd.read_csv(release / "glycan_identifier_map.csv")
    excluded = pd.read_csv(release / "excluded_records.csv")
    folds = pd.read_csv(release / "positive_holdout_folds.csv")
    candidates = pd.read_csv(release / "candidate_manifest.csv")
    ids = set(candidate_ids(release))
    positives = positive_pairs(evidence, ids)
    support = positives.groupby("agent_id")["glytoucan_id"].nunique()

    resolved_records = int(evidence["glytoucan_id"].notna().sum())
    resolved_glycans = int(
        identifiers.loc[identifiers["glytoucan_id"].notna(), "glytoucan_id"].nunique()
    )
    joined_records = int(evidence["glytoucan_id"].astype(str).isin(ids).sum())
    joined_relations = int(len(positives))
    evaluable_agents = int((support >= 2).sum())
    evaluable_relations = int(support[support >= 2].sum())
    insufficient_agents = int((support < 2).sum())
    insufficient_relations = int(support[support < 2].sum())

    rows = [
        {"stage": "source_records", "count": len(evidence), "unit": "records", "excluded_from_previous": 0, "exclusion_reason": "none"},
        {"stage": "resolved_records", "count": resolved_records, "unit": "records", "excluded_from_previous": len(evidence) - resolved_records, "exclusion_reason": "unresolved source identifier"},
        {"stage": "resolved_glycans", "count": resolved_glycans, "unit": "unique glycans", "excluded_from_previous": 0, "exclusion_reason": "unit transition"},
        {"stage": "strictly_joined_records", "count": joined_records, "unit": "records", "excluded_from_previous": resolved_records - joined_records, "exclusion_reason": "resolved glycan failed strict structure contract"},
        {"stage": "strictly_joined_glycans", "count": len(ids), "unit": "unique glycans", "excluded_from_previous": resolved_glycans - len(ids), "exclusion_reason": "no GlycoShape record, requested branch absent, or WURCS mismatch"},
        {"stage": "canonical_joined_relations", "count": joined_relations, "unit": "agent-glycan relations", "excluded_from_previous": 0, "exclusion_reason": "deduplicated from joined records"},
        {"stage": "evaluable_agents", "count": evaluable_agents, "unit": "agent identifiers", "excluded_from_previous": insufficient_agents, "exclusion_reason": "fewer than two joined positive relations"},
        {"stage": "evaluable_relations", "count": evaluable_relations, "unit": "agent-glycan relations", "excluded_from_previous": insufficient_relations, "exclusion_reason": "relations assigned to agents with fewer than two joined positives"},
        {"stage": "candidate_glycans", "count": len(ids), "unit": "glycans", "excluded_from_previous": 0, "exclusion_reason": "fixed common representation universe"},
        {"stage": "candidate_rows", "count": len(candidates), "unit": "agent-glycan candidate rows", "excluded_from_previous": 0, "exclusion_reason": "159 by 44 Cartesian product"},
        {"stage": "agent_folds", "count": folds[["agent_id", "fold"]].drop_duplicates().shape[0], "unit": "agent-folds", "excluded_from_previous": 0, "exclusion_reason": "up to five predefined folds per evaluable agent"},
    ]
    failure_counts = (
        excluded.groupby(["exclusion_stage", "exclusion_reason"], dropna=False)
        .size()
        .reset_index(name="record_count")
    )
    failure_counts.to_csv(out / "record_exclusions_by_reason.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "count_funnel.csv", index=False)

    counts = {
        "source_records": int(len(evidence)),
        "resolved_records": resolved_records,
        "resolved_glycans": resolved_glycans,
        "strictly_joined_records": joined_records,
        "strictly_joined_glycans": int(len(ids)),
        "canonical_joined_relations": joined_relations,
        "evaluable_agents": evaluable_agents,
        "evaluable_relations": evaluable_relations,
        "relations_excluded_for_insufficient_agent_support": insufficient_relations,
        "agents_excluded_for_insufficient_support": insufficient_agents,
        "candidate_glycans": int(len(ids)),
        "candidate_rows": int(len(candidates)),
        "agent_folds": int(folds[["agent_id", "fold"]].drop_duplicates().shape[0]),
    }
    assert counts["source_records"] == 1263
    assert counts["resolved_records"] == 1258
    assert counts["resolved_glycans"] == 173
    assert counts["strictly_joined_records"] == 801
    assert counts["strictly_joined_glycans"] == 44
    assert counts["canonical_joined_relations"] == 765
    assert counts["evaluable_agents"] == 159
    assert counts["evaluable_relations"] == 697
    assert counts["relations_excluded_for_insufficient_agent_support"] == 68
    assert counts["agent_folds"] == 645
    write_json(out / "count_funnel.json", counts)
    return counts


def normalized_name(values: pd.Series) -> pd.Series:
    return values.fillna("").str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)


def agent_identifier_audit(release: Path, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(release / "evidence_records.csv", dtype=str)
    folds = pd.read_csv(release / "positive_holdout_folds.csv", dtype=str)
    evaluated = set(folds["agent_id"].astype(str))

    columns = [
        "agent_id",
        "agent_name",
        "agent_url",
        "binding_mediator_id",
        "binding_mediator_name",
        "binding_mediator_uniprot",
    ]
    source = evidence[columns].drop_duplicates()
    agents = (
        source.groupby("agent_id", as_index=False)
        .agg(
            agent_name=("agent_name", "first"),
            agent_url=("agent_url", "first"),
            binding_mediator_ids=("binding_mediator_id", lambda x: ";".join(sorted(set(x.dropna())))),
            binding_mediator_names=("binding_mediator_name", lambda x: ";".join(sorted(set(x.dropna())))),
            external_uniprot_ids=("binding_mediator_uniprot", lambda x: ";".join(sorted(set(x.dropna())))),
        )
    )
    agents["normalized_name"] = normalized_name(agents["agent_name"])
    agents["evaluated_in_primary_protocol"] = agents["agent_id"].isin(evaluated)
    agents["has_external_uniprot_identifier"] = agents["external_uniprot_ids"].ne("")
    agents["defensible_biological_entity_mapping"] = np.where(
        agents["has_external_uniprot_identifier"],
        "partial_external_identifier_available",
        "not_defensibly_mappable_from_released_metadata",
    )
    agents.to_csv(out / "agent_identifier_table.csv", index=False)

    name_groups = (
        agents.groupby("normalized_name")
        .agg(
            duplicate_group_size=("agent_id", "nunique"),
            agent_ids=("agent_id", lambda x: ";".join(sorted(set(x)))),
            display_names=("agent_name", lambda x: ";".join(sorted(set(x.dropna())))),
        )
        .reset_index()
        .query("duplicate_group_size > 1")
        .sort_values(["duplicate_group_size", "normalized_name"], ascending=[False, True])
    )
    name_groups.to_csv(out / "exact_normalized_name_duplicate_groups.csv", index=False)

    external_rows = []
    for row in agents.itertuples(index=False):
        for external_id in filter(None, row.external_uniprot_ids.split(";")):
            external_rows.append(
                {
                    "agent_id": row.agent_id,
                    "external_uniprot_id": external_id,
                    "evaluated_in_primary_protocol": row.evaluated_in_primary_protocol,
                }
            )
    external = pd.DataFrame(external_rows)
    if len(external):
        external_groups = (
            external.groupby("external_uniprot_id")
            .agg(
                duplicate_group_size=("agent_id", "nunique"),
                agent_ids=("agent_id", lambda x: ";".join(sorted(set(x)))),
            )
            .reset_index()
            .query("duplicate_group_size > 1")
            .sort_values(["duplicate_group_size", "external_uniprot_id"], ascending=[False, True])
        )
    else:
        external_groups = pd.DataFrame(
            columns=["external_uniprot_id", "duplicate_group_size", "agent_ids"]
        )
    external_groups.to_csv(out / "shared_external_identifier_groups.csv", index=False)

    def scope_summary(frame: pd.DataFrame) -> dict[str, int]:
        groups = frame.groupby("normalized_name")["agent_id"].nunique()
        if len(external):
            scoped_external = external[
                external["agent_id"].isin(set(frame["agent_id"]))
            ]
            ext_groups = scoped_external.groupby("external_uniprot_id")["agent_id"].nunique()
        else:
            ext_groups = pd.Series(dtype=int)
        return {
            "agent_identifiers": int(len(frame)),
            "with_external_uniprot_identifier": int(
                frame["has_external_uniprot_identifier"].sum()
            ),
            "not_defensibly_mappable": int(
                (~frame["has_external_uniprot_identifier"]).sum()
            ),
            "exact_normalized_name_duplicate_groups": int((groups > 1).sum()),
            "maximum_normalized_name_duplicate_group_size": int(groups.max()),
            "shared_external_identifier_groups": int((ext_groups > 1).sum()),
            "maximum_shared_external_identifier_group_size": int(ext_groups.max())
            if len(ext_groups)
            else 0,
        }

    summary = {
        "source_scope": scope_summary(agents),
        "primary_evaluation_scope": scope_summary(
            agents[agents["evaluated_in_primary_protocol"]]
        ),
        "normalization": "lowercase alphanumeric exact match only",
        "manual_alias_inference": False,
        "automatic_merging": False,
        "cluster_bootstrap_status": "NOT_ESTIMABLE",
        "cluster_bootstrap_reason": "External UniProt identifiers cover only a subset and identify recorded binding mediators rather than a complete, uniform biological-entity key for every SugarBind agent identifier.",
        "interpretation": "Equal weighting remains at the SugarBind agent-identifier level and should not be interpreted as independent unique biological entities.",
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT / "data",
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "results"
    )
    args = parser.parse_args()
    release = args.release_dir.resolve()
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(release / "evidence_records.csv")
    folds = pd.read_csv(release / "positive_holdout_folds.csv")
    ids = candidate_ids(release)
    popularity = popularity_audit(release, out / "popularity_baseline")
    identifier_audit = agent_identifier_audit(release, out / "agent_identifier_audit")
    wurcs_aggregate = json.loads(
        (release / "results/wurcs/aggregate_metrics.json").read_text()
    )
    summary = {
        "dataset": "glycoaudit_data",
        "popularity": popularity,
        "popularity_tie_sensitivity": popularity_tie_sensitivity(
            evidence,
            folds,
            ids,
            out / "popularity_tie_sensitivity",
            {
                "mrr": float(wurcs_aggregate["mrr"]),
                "recall@5": float(wurcs_aggregate["recall@5"]),
                "known_positive_ap": float(wurcs_aggregate["known_positive_ap"]),
            },
        ),
        "popularity_entity_exclusion_sensitivity": entity_exclusion_sensitivity(
            release,
            evidence,
            folds,
            ids,
            out / "popularity_entity_exclusion_sensitivity",
        ),
        "popularity_stratified_diagnostic": popularity_stratified_diagnostic(
            release,
            evidence,
            folds,
            ids,
            out / "popularity_stratified_diagnostic",
        ),
        "count_funnel": count_funnel(release, out / "count_funnel"),
        "agent_identifier_audit": identifier_audit,
    }
    write_json(out / "analysis_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
