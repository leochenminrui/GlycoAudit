#!/usr/bin/env python3
"""Validate the public GlycoAudit dataset."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "RELEASE_MANIFEST.json",
    "CHECKSUMS.sha256",
    "CFG_DATA_STATUS.json",
    "audit/release_flow.csv",
    "evidence_records.csv",
    "excluded_records.csv",
    "cfg_assay_thresholds.csv",
    "structure_manifest.csv",
    "glycan_identifier_map.csv",
    "glycan_structure_map.csv",
    "joinability_record_audit.csv",
    "joinability_audit.csv",
    "candidate_manifest.csv",
    "positive_holdout_folds.csv",
    "scaffold_folds.csv",
    "cross_source_candidate_manifest.csv",
    "ssv_features.csv",
    "gcv_features.csv",
    "wurcs_features.csv",
    "positive_holdout_protocol.json",
    "sources/cfg_experiment_catalogue.csv",
    "results/ssv/aggregate_metrics.json",
    "results/ssv/metrics_per_agent.csv",
    "results/gcv_4_5/aggregate_metrics.json",
    "results/gcv_4_5/metrics_per_agent.csv",
    "results/wurcs/aggregate_metrics.json",
    "results/wurcs/metrics_per_agent.csv",
    "results/summary/primary_results.csv",
}

EVIDENCE_COLUMNS = {
    "record_id",
    "source",
    "agent_id",
    "glycan_source_id",
    "glytoucan_id",
    "evidence_status",
}
JOIN_COLUMNS = {"glytoucan_id", "joinable"}
RECORD_JOIN_COLUMNS = {
    "record_id",
    "agent_id",
    "glycan_source_id",
    "resolved_glytoucan_id",
    "eligible_evidence",
    "joinable",
    "failure_category",
}
CANDIDATE_COLUMNS = {
    "candidate_pool_id",
    "agent_id",
    "glytoucan_id",
    "included_in_universe",
    "known_positive",
}
FOLD_COLUMNS = {
    "agent_id",
    "glytoucan_id",
    "fold",
    "fold_count_for_agent",
    "total_known_positives",
    "assignment_role",
    "seed",
}
EXCLUSION_COLUMNS = {
    "record_id",
    "exclusion_stage",
    "exclusion_reason",
    "resolved_identifier",
}
STRUCTURE_COLUMNS = {"glytoucan_id", "status", "model_count"}
ALLOWED_EVIDENCE = {
    "positive",
    "reported_nonbinding",
    "inconclusive",
    "unmeasured",
    "insufficient_context",
}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_author_required(value: object) -> bool:
    if value == "AUTHOR_REQUIRED":
        return True
    if isinstance(value, dict):
        return any(_contains_author_required(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_author_required(item) for item in value)
    return False


def validate(release_dir: Path) -> dict[str, object]:
    missing = sorted(name for name in REQUIRED_FILES if not (release_dir / name).exists())
    if missing:
        raise ValueError(f"Missing required release files: {missing}")

    manifest = json.loads((release_dir / "RELEASE_MANIFEST.json").read_text())
    if _contains_author_required(manifest):
        raise ValueError("Release manifest still contains AUTHOR_REQUIRED")
    protocol = json.loads((release_dir / "positive_holdout_protocol.json").read_text())
    if _contains_author_required(protocol):
        raise ValueError("Positive-holdout protocol still contains AUTHOR_REQUIRED")

    evidence = pd.read_csv(release_dir / "evidence_records.csv")
    excluded = pd.read_csv(release_dir / "excluded_records.csv")
    record_joins = pd.read_csv(release_dir / "joinability_record_audit.csv")
    joins = pd.read_csv(release_dir / "joinability_audit.csv")
    candidates = pd.read_csv(release_dir / "candidate_manifest.csv")
    folds = pd.read_csv(release_dir / "positive_holdout_folds.csv")
    structures = pd.read_csv(release_dir / "structure_manifest.csv")
    release_flow = pd.read_csv(release_dir / "audit/release_flow.csv")
    cfg_catalogue = pd.read_csv(release_dir / "sources/cfg_experiment_catalogue.csv")
    feature_tables: dict[str, pd.DataFrame] = {}
    for feature_name in ["ssv_features.csv", "gcv_features.csv", "wurcs_features.csv"]:
        feature_table = pd.read_csv(release_dir / feature_name)
        if "glytoucan_id" not in feature_table.columns or len(feature_table.columns) < 2:
            raise ValueError(
                f"{feature_name} must contain glytoucan_id and at least one representation column"
            )
        if feature_table["glytoucan_id"].astype(str).duplicated().any():
            raise ValueError(f"{feature_name} contains duplicate glycan identifiers")
        feature_tables[feature_name] = feature_table
    missing_evidence_columns = sorted(EVIDENCE_COLUMNS - set(evidence.columns))
    missing_join_columns = sorted(JOIN_COLUMNS - set(joins.columns))
    missing_record_join_columns = sorted(RECORD_JOIN_COLUMNS - set(record_joins.columns))
    missing_candidate_columns = sorted(CANDIDATE_COLUMNS - set(candidates.columns))
    missing_fold_columns = sorted(FOLD_COLUMNS - set(folds.columns))
    missing_exclusion_columns = sorted(EXCLUSION_COLUMNS - set(excluded.columns))
    missing_structure_columns = sorted(STRUCTURE_COLUMNS - set(structures.columns))
    if missing_evidence_columns:
        raise ValueError(f"Evidence table missing columns: {missing_evidence_columns}")
    if missing_join_columns:
        raise ValueError(f"Joinability table missing columns: {missing_join_columns}")
    if missing_record_join_columns:
        raise ValueError(
            f"Record joinability table missing columns: {missing_record_join_columns}"
        )
    if missing_candidate_columns:
        raise ValueError(f"Candidate manifest missing columns: {missing_candidate_columns}")
    if missing_fold_columns:
        raise ValueError(f"Positive-fold manifest missing columns: {missing_fold_columns}")
    if missing_exclusion_columns:
        raise ValueError(f"Exclusion table missing columns: {missing_exclusion_columns}")
    if missing_structure_columns:
        raise ValueError(f"Structure manifest missing columns: {missing_structure_columns}")

    if evidence["record_id"].astype(str).duplicated().any():
        raise ValueError("Evidence record_id values must be unique")
    if record_joins["record_id"].astype(str).duplicated().any():
        raise ValueError("Record joinability audit record_id values must be unique")
    evidence_ids = set(evidence["record_id"].dropna().astype(str))
    audit_ids = set(record_joins["record_id"].dropna().astype(str))
    if evidence_ids != audit_ids:
        raise ValueError(
            "Record joinability audit does not cover the evidence table: "
            f"missing={sorted(evidence_ids - audit_ids)}, extra={sorted(audit_ids - evidence_ids)}"
        )

    statuses = set(evidence["evidence_status"].dropna().astype(str).str.lower())
    unexpected = sorted(statuses - ALLOWED_EVIDENCE)
    if unexpected:
        raise ValueError(f"Unsupported evidence statuses: {unexpected}")

    positive = evidence[evidence["evidence_status"].astype(str).str.lower() == "positive"]
    eligible = evidence[
        evidence["evidence_status"].astype(str).str.lower()
        .isin(ALLOWED_EVIDENCE - {"unmeasured"})
    ]
    eligible_ids = set(eligible["record_id"].dropna().astype(str))
    audited_eligible_ids = set(
        record_joins.loc[truthy(record_joins["eligible_evidence"]), "record_id"]
        .dropna()
        .astype(str)
    )
    if eligible_ids != audited_eligible_ids:
        raise ValueError(
            "Eligible-evidence flags do not reconcile with evidence_status: "
            f"missing={sorted(eligible_ids - audited_eligible_ids)}, "
            f"extra={sorted(audited_eligible_ids - eligible_ids)}"
        )
    joinable_ids = set(
        joins.loc[truthy(joins["joinable"]), "glytoucan_id"].dropna().astype(str)
    )
    joinable_record_rows = record_joins[truthy(record_joins["joinable"])]
    if not truthy(joinable_record_rows["eligible_evidence"]).all():
        raise ValueError("A joinable record is not marked as eligible evidence")
    if joinable_record_rows["resolved_glytoucan_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("A joinable record lacks a resolved GlyTouCan identifier")
    joinable_record_glycans = set(
        joinable_record_rows["resolved_glytoucan_id"].dropna().astype(str)
    )
    if joinable_record_glycans != joinable_ids:
        raise ValueError(
            "Record- and glycan-level joinability do not reconcile: "
            f"record_only={sorted(joinable_record_glycans - joinable_ids)}, "
            f"glycan_only={sorted(joinable_ids - joinable_record_glycans)}"
        )
    unresolved_eligible = record_joins[
        truthy(record_joins["eligible_evidence"]) & ~truthy(record_joins["joinable"])
    ]
    if unresolved_eligible["failure_category"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("An unresolved eligible record lacks a failure_category")
    unresolved_ids = set(unresolved_eligible["record_id"].dropna().astype(str))
    excluded_ids = set(excluded["record_id"].dropna().astype(str))
    if unresolved_ids != excluded_ids:
        raise ValueError(
            "Excluded records do not reconcile with unresolved eligible evidence: "
            f"missing={sorted(unresolved_ids - excluded_ids)}, "
            f"extra={sorted(excluded_ids - unresolved_ids)}"
        )
    if excluded["exclusion_reason"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("An excluded record lacks an exclusion_reason")

    included_candidates = candidates[truthy(candidates["included_in_universe"])]
    if included_candidates[["agent_id", "glytoucan_id"]].astype(str).duplicated().any():
        raise ValueError("Candidate manifest contains duplicate agent--glycan pairs")
    candidate_ids = set(included_candidates["glytoucan_id"].dropna().astype(str))
    feature_id_sets = [
        set(feature_table["glytoucan_id"].dropna().astype(str))
        for feature_table in feature_tables.values()
    ]
    shared_feature_ids = set.intersection(*feature_id_sets)
    if not candidate_ids or candidate_ids != shared_feature_ids:
        raise ValueError(
            "Candidate glycan universe does not equal the shared feature intersection"
        )
    candidate_agents = set(included_candidates["agent_id"].dropna().astype(str))
    candidate_pairs = set(
        zip(
            included_candidates["agent_id"].astype(str),
            included_candidates["glytoucan_id"].astype(str),
        )
    )
    expected_candidate_pairs = set(itertools.product(candidate_agents, candidate_ids))
    if candidate_pairs != expected_candidate_pairs:
        raise ValueError(
            "Candidate manifest is not the complete agent-by-glycan Cartesian universe"
        )
    fold_pairs = set(
        zip(folds["agent_id"].astype(str), folds["glytoucan_id"].astype(str))
    )
    if folds[["agent_id", "glytoucan_id"]].astype(str).duplicated().any():
        raise ValueError("Positive-fold manifest contains duplicate relations")
    positive_pairs = set(
        zip(positive["agent_id"].astype(str), positive["glytoucan_id"].astype(str))
    )
    if not fold_pairs or not fold_pairs.issubset(positive_pairs):
        raise ValueError("Positive-fold manifest contains non-positive or no relations")
    if not set(folds["glytoucan_id"].astype(str)).issubset(candidate_ids):
        raise ValueError("Positive-fold manifest contains glycans outside the candidate universe")
    candidate_positive_pairs = set(
        zip(
            included_candidates.loc[truthy(included_candidates["known_positive"]), "agent_id"].astype(str),
            included_candidates.loc[
                truthy(included_candidates["known_positive"]), "glytoucan_id"
            ].astype(str),
        )
    )
    if fold_pairs != candidate_positive_pairs:
        raise ValueError(
            "Positive-fold manifest does not exactly equal known positives in the candidate universe"
        )
    if set(folds["agent_id"].astype(str)) != candidate_agents:
        raise ValueError("Candidate and positive-fold agent sets do not match")
    if set(folds["assignment_role"].astype(str)) != {"held_out_positive_in_assigned_fold"}:
        raise ValueError("Unexpected positive-fold assignment role")

    for agent_id, group in folds.groupby("agent_id", sort=False):
        declared_fold_counts = set(group["fold_count_for_agent"].astype(int))
        declared_support = set(group["total_known_positives"].astype(int))
        observed_folds = set(group["fold"].astype(int))
        if len(declared_fold_counts) != 1 or len(declared_support) != 1:
            raise ValueError(f"Inconsistent fold metadata for agent {agent_id}")
        fold_count = declared_fold_counts.pop()
        support = declared_support.pop()
        if support != len(group) or observed_folds != set(range(fold_count)):
            raise ValueError(f"Fold assignment does not reconcile for agent {agent_id}")

    valid_structures = structures[structures["status"].astype(str) == "ok"]
    if valid_structures["glytoucan_id"].astype(str).duplicated().any():
        raise ValueError("Structure manifest contains duplicate accepted glycans")
    if set(valid_structures["glytoucan_id"].astype(str)) != candidate_ids:
        raise ValueError("Accepted structures do not equal the candidate glycan universe")

    resolved_eligible = record_joins[
        truthy(record_joins["eligible_evidence"])
        & record_joins["resolved_glytoucan_id"].fillna("").astype(str).str.strip().ne("")
    ]
    joinable_relations = joinable_record_rows[
        ["agent_id", "resolved_glytoucan_id"]
    ].drop_duplicates()
    excluded_resolved_glycans = excluded["resolved_identifier"].dropna().astype(str)
    cfg_snapshot_dir = release_dir / "sources/cfg_catalogue_html"
    cfg_snapshot_pages = (
        len(list(cfg_snapshot_dir.glob("catalogue_page_*.html")))
        if cfg_snapshot_dir.is_dir()
        else 0
    )
    observed_counts = {
        "raw_evidence_records": int(len(evidence)),
        "source_ligand_identifiers": int(
            evidence["glycan_source_id"].dropna().astype(str).nunique()
        ),
        "source_agent_ligand_relations": int(
            evidence[["agent_id", "glycan_source_id"]]
            .dropna()
            .drop_duplicates()
            .shape[0]
        ),
        "source_agents": int(evidence["agent_id"].dropna().astype(str).nunique()),
        "eligible_evidence_records": int(len(eligible)),
        "positive_evidence_records": int(len(positive)),
        "identifier_resolved_records": int(len(resolved_eligible)),
        "identifier_resolved_glycans": int(
            resolved_eligible["resolved_glytoucan_id"].astype(str).nunique()
        ),
        "identifier_resolved_relations": int(
            resolved_eligible[["agent_id", "resolved_glytoucan_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "identifier_resolved_agents": int(
            resolved_eligible["agent_id"].astype(str).nunique()
        ),
        "joinable_evidence_records": int(len(joinable_record_rows)),
        "joinable_glycans": int(len(joinable_ids)),
        "joinable_relations": int(len(joinable_relations)),
        "joinable_agents": int(joinable_record_rows["agent_id"].astype(str).nunique()),
        "valid_structures": int(len(valid_structures)),
        "structure_models": int(valid_structures["model_count"].fillna(0).astype(int).sum()),
        "excluded_from_structure_supported_universe": int(len(excluded)),
        "excluded_resolved_glycans": int(excluded_resolved_glycans.nunique()),
        "candidate_rows": int(len(included_candidates)),
        "candidate_glycans": int(len(candidate_ids)),
        "candidate_positive_relations": int(len(fold_pairs)),
        "evaluable_agents": int(len(candidate_agents)),
        "agent_folds": int(folds[["agent_id", "fold"]].drop_duplicates().shape[0]),
        "cfg_catalogue_records": int(len(cfg_catalogue)),
        "cfg_catalogue_pages": int(cfg_snapshot_pages),
    }

    source_relation_count = int(
        evidence[["agent_id", "glycan_source_id"]]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )
    eligible_record_rows = record_joins[truthy(record_joins["eligible_evidence"])]
    unresolved_record_rows = eligible_record_rows[~truthy(eligible_record_rows["joinable"])]
    expected_flow = {
        "raw_evidence_records": (
            int(len(evidence)),
            int(evidence["glycan_source_id"].dropna().astype(str).nunique()),
            source_relation_count,
            int(evidence["agent_id"].dropna().astype(str).nunique()),
        ),
        "eligible_evidence": (
            int(len(eligible_record_rows)),
            int(eligible_record_rows["glycan_source_id"].dropna().astype(str).nunique()),
            int(
                eligible_record_rows[["agent_id", "glycan_source_id"]]
                .dropna()
                .drop_duplicates()
                .shape[0]
            ),
            int(eligible_record_rows["agent_id"].dropna().astype(str).nunique()),
        ),
        "identifier_resolved": (
            int(len(resolved_eligible)),
            int(resolved_eligible["resolved_glytoucan_id"].astype(str).nunique()),
            int(
                resolved_eligible[["agent_id", "resolved_glytoucan_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            int(resolved_eligible["agent_id"].astype(str).nunique()),
        ),
        "final_joinable": (
            int(len(joinable_record_rows)),
            int(joinable_record_rows["resolved_glytoucan_id"].astype(str).nunique()),
            int(len(joinable_relations)),
            int(joinable_record_rows["agent_id"].astype(str).nunique()),
        ),
        "excluded_or_unresolved": (
            int(len(unresolved_record_rows)),
            int(
                unresolved_record_rows["resolved_glytoucan_id"]
                .dropna()
                .astype(str)
                .nunique()
            ),
            int(
                unresolved_record_rows[["agent_id", "resolved_glytoucan_id"]]
                .dropna()
                .drop_duplicates()
                .shape[0]
            ),
            int(unresolved_record_rows["agent_id"].dropna().astype(str).nunique()),
        ),
    }
    observed_flow = {
        str(row.stage): (int(row.records), int(row.glycans), int(row.relations), int(row.agents))
        for row in release_flow.itertuples(index=False)
    }
    if observed_flow != expected_flow:
        raise ValueError(
            f"Release-flow stage counts do not reconcile: expected={expected_flow}, observed={observed_flow}"
        )

    declared_counts = manifest.get("counts", {})
    unresolved = {
        key: value for key, value in declared_counts.items() if value == "AUTHOR_REQUIRED"
    }
    if unresolved:
        raise ValueError(f"Manifest counts still require author values: {sorted(unresolved)}")
    mismatches = {
        key: {"declared": int(declared_counts[key]), "observed": observed}
        for key, observed in observed_counts.items()
        if key not in declared_counts or int(declared_counts[key]) != observed
    }
    if mismatches:
        raise ValueError(f"Manifest count mismatch: {mismatches}")

    summary = pd.read_csv(release_dir / "results/summary/primary_results.csv")
    result_map = {
        "SSV": "ssv",
        "GCV 4.5 A": "gcv_4_5",
        "WURCS fold-TFIDF": "wurcs",
    }
    if set(summary["representation"].astype(str)) != set(result_map):
        raise ValueError("Primary-results representations are incomplete or unexpected")
    for representation, directory in result_map.items():
        aggregate = json.loads(
            (release_dir / f"results/{directory}/aggregate_metrics.json").read_text()
        )
        per_agent = pd.read_csv(release_dir / f"results/{directory}/metrics_per_agent.csv")
        row = summary.loc[summary["representation"] == representation].iloc[0]
        comparisons = {
            "n_glycans": aggregate["n_glycans"],
            "n_agents": aggregate["n_agents_evaluated"],
            "n_agent_folds": aggregate["n_agent_folds"],
            "mrr": aggregate["mrr"],
            "mrr_ci_lower": aggregate["mrr_ci_lower"],
            "mrr_ci_upper": aggregate["mrr_ci_upper"],
            "recall_at_5": aggregate["recall@5"],
            "recall_at_5_ci_lower": aggregate["recall@5_ci_lower"],
            "recall_at_5_ci_upper": aggregate["recall@5_ci_upper"],
            "known_positive_ap": aggregate["known_positive_ap"],
            "known_positive_ap_ci_lower": aggregate["known_positive_ap_ci_lower"],
            "known_positive_ap_ci_upper": aggregate["known_positive_ap_ci_upper"],
        }
        for column, expected in comparisons.items():
            if abs(float(row[column]) - float(expected)) > 1e-12:
                raise ValueError(
                    f"Primary-results mismatch for {representation} {column}: "
                    f"summary={row[column]}, aggregate={expected}"
                )
        for metric in ["mrr", "mean_rank", "known_positive_ap", "recall@5"]:
            if abs(float(per_agent[metric].mean()) - float(aggregate[metric])) > 1e-12:
                raise ValueError(
                    f"Per-agent aggregation mismatch for {representation} {metric}"
                )

    files = sorted(path for path in release_dir.rglob("*") if path.is_file())
    checksums = {
        str(path.relative_to(release_dir)): sha256(path)
        for path in files
        if path.name != "CHECKSUMS.sha256"
    }
    declared_checksums = {}
    for line in (release_dir / "CHECKSUMS.sha256").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid checksum line: {line}")
        digest, relative_path = parts
        relative_path = relative_path.lstrip("*")
        declared_checksums[relative_path] = digest
    missing_checksum_entries = sorted(set(checksums) - set(declared_checksums))
    mismatched_checksums = {
        path: {"declared": declared_checksums.get(path), "observed": digest}
        for path, digest in checksums.items()
        if path in declared_checksums and declared_checksums[path] != digest
    }
    if missing_checksum_entries:
        raise ValueError(f"Files missing checksum entries: {missing_checksum_entries}")
    if mismatched_checksums:
        raise ValueError(f"Checksum mismatch: {mismatched_checksums}")
    return {
        "release_dir": str(release_dir.resolve()),
        "counts": observed_counts,
        "evidence_statuses": sorted(statuses),
        "n_files": len(files),
        "checksums": checksums,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args.release_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
