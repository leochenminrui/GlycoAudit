"""Record- and glycan-level joinability auditing for a frozen release."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


ALLOWED_EVIDENCE = {
    "positive",
    "reported_nonbinding",
    "inconclusive",
    "unmeasured",
    "insufficient_context",
}
ELIGIBLE_EVIDENCE = ALLOWED_EVIDENCE - {"unmeasured"}

EVIDENCE_REQUIRED = {
    "record_id",
    "source",
    "agent_id",
    "glycan_source_id",
    "glytoucan_id",
    "evidence_status",
}
IDENTIFIER_REQUIRED = {
    "source",
    "source_glycan_id",
    "glytoucan_id",
    "ambiguity_status",
}
STRUCTURE_REQUIRED = {
    "glytoucan_id",
    "structure_id",
    "join_method",
    "valid_structure",
}


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _identifier_index(identifier_map: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for key, group in identifier_map.groupby(["source", "source_glycan_id"], dropna=False):
        normalized_key = (_text(key[0]), _text(key[1]))
        ids = sorted({_text(value) for value in group["glytoucan_id"] if _text(value)})
        ambiguity_values = {
            _text(value).lower() for value in group["ambiguity_status"] if _text(value)
        }
        flagged = any(
            value in {"ambiguous", "conflict", "unresolved"}
            or ("ambiguous" in value and "unambiguous" not in value)
            or "conflict" in value
            or "unresolved" in value
            for value in ambiguity_values
        )
        if len(ids) == 1 and not flagged:
            index[normalized_key] = {"status": "unique", "glytoucan_id": ids[0]}
        elif len(ids) > 1 or flagged:
            index[normalized_key] = {"status": "ambiguous", "glytoucan_id": ""}
        else:
            index[normalized_key] = {"status": "unmapped", "glytoucan_id": ""}
    return index


def _structure_index(structure_map: pd.DataFrame) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for glytoucan_id, group in structure_map.groupby("glytoucan_id", dropna=False):
        gid = _text(glytoucan_id)
        if not gid:
            continue
        valid = group[group["valid_structure"].map(_truthy)]
        methods = sorted({_text(value).lower() for value in valid["join_method"] if _text(value)})
        index[gid] = {
            "structure_candidate_count": int(len(group)),
            "valid_structure_count": int(len(valid)),
            "join_methods": methods,
        }
    return index


def _direct_or_recovered(methods: list[str]) -> str:
    if not methods:
        return ""
    recovered = any(
        token in method
        for method in methods
        for token in ("recover", "generat", "target")
    )
    direct = any(
        token in method
        for method in methods
        for token in ("direct", "existing", "download")
    )
    if direct and recovered:
        return "direct_and_recovered"
    if recovered:
        return "recovered"
    if direct:
        return "direct"
    return "other_documented_method"


def build_joinability_audit(
    evidence: pd.DataFrame,
    identifier_map: pd.DataFrame,
    structure_map: pd.DataFrame,
    excluded_records: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build record, glycan, flow, and summary artifacts without dropping failures."""
    _require_columns(evidence, EVIDENCE_REQUIRED, "evidence_records")
    _require_columns(identifier_map, IDENTIFIER_REQUIRED, "glycan_identifier_map")
    _require_columns(structure_map, STRUCTURE_REQUIRED, "glycan_structure_map")

    evidence = evidence.copy()
    evidence["evidence_status"] = evidence["evidence_status"].map(_text).str.lower()
    unexpected = sorted(set(evidence["evidence_status"]) - ALLOWED_EVIDENCE)
    if unexpected:
        raise ValueError(f"Unsupported evidence statuses: {unexpected}")

    identifiers = _identifier_index(identifier_map)
    structures = _structure_index(structure_map)
    record_rows: list[dict[str, Any]] = []

    for row in evidence.to_dict(orient="records"):
        source = _text(row["source"])
        source_glycan_id = _text(row["glycan_source_id"])
        supplied_gid = _text(row["glytoucan_id"])
        mapped = identifiers.get((source, source_glycan_id), {"status": "unmapped", "glytoucan_id": ""})

        if supplied_gid:
            if mapped["status"] == "ambiguous":
                resolved_gid = ""
                mapping_status = "ambiguous_identifier_mapping"
            elif mapped["status"] == "unique" and mapped["glytoucan_id"] != supplied_gid:
                resolved_gid = ""
                mapping_status = "identifier_conflict"
            else:
                resolved_gid = supplied_gid
                mapping_status = "provided_in_evidence"
        elif mapped["status"] == "unique":
            resolved_gid = mapped["glytoucan_id"]
            mapping_status = "mapped_from_identifier_table"
        elif mapped["status"] == "ambiguous":
            resolved_gid = ""
            mapping_status = "ambiguous_identifier_mapping"
        else:
            resolved_gid = ""
            mapping_status = "unmapped_identifier"

        status = _text(row["evidence_status"]).lower()
        eligible = status in ELIGIBLE_EVIDENCE
        structure = structures.get(
            resolved_gid,
            {"structure_candidate_count": 0, "valid_structure_count": 0, "join_methods": []},
        )
        has_valid_structure = structure["valid_structure_count"] > 0
        joinable = bool(eligible and resolved_gid and has_valid_structure)

        if not eligible:
            failure = "unmeasured_relation"
        elif not resolved_gid:
            failure = mapping_status
        elif structure["structure_candidate_count"] == 0:
            failure = "no_structure_record"
        elif not has_valid_structure:
            failure = "invalid_structure"
        else:
            failure = ""

        record_rows.append(
            {
                "record_id": _text(row["record_id"]),
                "source": source,
                "agent_id": _text(row["agent_id"]),
                "glycan_source_id": source_glycan_id,
                "input_glytoucan_id": supplied_gid,
                "resolved_glytoucan_id": resolved_gid,
                "evidence_status": status,
                "eligible_evidence": eligible,
                "identifier_mapping_status": mapping_status,
                "valid_structure_count": structure["valid_structure_count"],
                "structure_candidate_count": structure["structure_candidate_count"],
                "direct_or_recovered": _direct_or_recovered(structure["join_methods"]),
                "joinable": joinable,
                "failure_category": failure,
            }
        )

    record_audit = pd.DataFrame(record_rows)
    resolved_evidence_ids = set(record_audit["resolved_glytoucan_id"].loc[lambda s: s != ""])
    all_glycan_ids = sorted(resolved_evidence_ids | set(structures))
    glycan_rows: list[dict[str, Any]] = []
    for gid in all_glycan_ids:
        records = record_audit[record_audit["resolved_glytoucan_id"] == gid]
        eligible_records = records[records["eligible_evidence"]]
        structure = structures.get(
            gid,
            {"structure_candidate_count": 0, "valid_structure_count": 0, "join_methods": []},
        )
        has_evidence = len(eligible_records) > 0
        has_structure = structure["valid_structure_count"] > 0
        joinable = bool(has_evidence and has_structure)
        if joinable:
            failure = ""
        elif not has_evidence and not has_structure:
            failure = "no_eligible_evidence_and_no_valid_structure"
        elif not has_evidence:
            failure = "no_eligible_evidence"
        elif structure["structure_candidate_count"] == 0:
            failure = "no_structure_record"
        else:
            failure = "invalid_structure"
        glycan_rows.append(
            {
                "glytoucan_id": gid,
                "has_eligible_evidence": has_evidence,
                "has_valid_structure": has_structure,
                "joinable": joinable,
                "direct_or_recovered": _direct_or_recovered(structure["join_methods"]),
                "failure_category": failure,
                "evidence_record_count": int(len(eligible_records)),
                "structure_candidate_count": structure["structure_candidate_count"],
            }
        )
    glycan_audit = pd.DataFrame(glycan_rows)

    def _flow_row(
        stage: str,
        subset: pd.DataFrame,
        rule: str,
        *,
        glycan_column: str,
        identifier_basis: str,
    ) -> dict[str, Any]:
        identified = subset[subset[glycan_column] != ""]
        return {
            "stage": stage,
            "records": int(len(subset)),
            "glycans": int(identified[glycan_column].nunique()),
            "relations": int(
                identified[["agent_id", glycan_column]].drop_duplicates().shape[0]
            ),
            "agents": int(subset["agent_id"].loc[lambda s: s != ""].nunique()),
            "identifier_basis": identifier_basis,
            "rule": rule,
        }

    eligible_records = record_audit[record_audit["eligible_evidence"]]
    mapped_eligible = eligible_records[eligible_records["resolved_glytoucan_id"] != ""]
    joined_records = eligible_records[eligible_records["joinable"]]
    unresolved_records = eligible_records[~eligible_records["joinable"]]
    flow = pd.DataFrame(
        [
            _flow_row(
                "raw_evidence_records",
                record_audit,
                "All normalized evidence-table rows",
                glycan_column="glycan_source_id",
                identifier_basis="source ligand identifier",
            ),
            _flow_row(
                "eligible_evidence",
                eligible_records,
                "Observed statuses except unmeasured",
                glycan_column="glycan_source_id",
                identifier_basis="source ligand identifier",
            ),
            _flow_row(
                "identifier_resolved",
                mapped_eligible,
                "Unique normalized GlyTouCan identifier",
                glycan_column="resolved_glytoucan_id",
                identifier_basis="resolved GlyTouCan accession",
            ),
            _flow_row(
                "final_joinable",
                joined_records,
                "Eligible evidence plus at least one valid structure",
                glycan_column="resolved_glytoucan_id",
                identifier_basis="resolved GlyTouCan accession",
            ),
            _flow_row(
                "excluded_or_unresolved",
                unresolved_records,
                "Eligible records failing mapping or structure join",
                glycan_column="resolved_glytoucan_id",
                identifier_basis="resolved GlyTouCan accession; unmapped records retained in record and agent counts",
            ),
        ]
    )

    exclusion_counts = Counter(record_audit["failure_category"].loc[lambda s: s != ""])
    supplied_exclusion_counts: dict[str, int] = {}
    if excluded_records is not None and not excluded_records.empty:
        if "exclusion_reason" not in excluded_records.columns:
            raise ValueError("excluded_records missing required column: exclusion_reason")
        supplied_exclusion_counts = {
            _text(key): int(value)
            for key, value in excluded_records["exclusion_reason"].map(_text).value_counts().items()
        }

    summary = {
        "raw_evidence_records": int(len(record_audit)),
        "eligible_evidence_records": int(len(eligible_records)),
        "resolved_eligible_records": int(len(mapped_eligible)),
        "joinable_evidence_records": int(len(joined_records)),
        "unique_resolved_glycans": int(len(resolved_evidence_ids)),
        "joinable_glycans": int(glycan_audit["joinable"].sum()) if not glycan_audit.empty else 0,
        "record_failure_counts": dict(sorted(exclusion_counts.items())),
        "supplied_exclusion_reason_counts": dict(sorted(supplied_exclusion_counts.items())),
        "by_source": {},
    }
    for source, group in record_audit.groupby("source", dropna=False):
        summary["by_source"][_text(source)] = {
            "raw_records": int(len(group)),
            "eligible_records": int(group["eligible_evidence"].sum()),
            "resolved_eligible_records": int(
                ((group["eligible_evidence"]) & (group["resolved_glytoucan_id"] != "")).sum()
            ),
            "joinable_records": int(group["joinable"].sum()),
            "joinable_glycans": int(
                group.loc[group["joinable"], "resolved_glytoucan_id"].nunique()
            ),
        }

    return {
        "record_audit": record_audit,
        "glycan_audit": glycan_audit,
        "release_flow": flow,
        "summary": summary,
    }
