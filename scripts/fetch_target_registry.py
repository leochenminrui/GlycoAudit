#!/usr/bin/env python3
"""Fetch GlyTouCan sequences for the accessions present in reconstructed evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


SPARQL_ENDPOINT = "https://ts.glytoucan.org/sparql"
ACCESSION_PATTERN = re.compile(r"^G[0-9A-Z]{7}$")
QUERY_TEMPLATE = """
PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>

SELECT DISTINCT ?accession ?wurcs ?glycoct
WHERE {{
  VALUES ?accession {{ {values} }}
  ?saccharide glytoucan:has_primary_id ?accession .
  OPTIONAL {{
    ?saccharide glycan:has_glycosequence ?wurcs_seq .
    ?wurcs_seq glycan:has_sequence ?wurcs ;
               glycan:in_carbohydrate_format glycan:carbohydrate_format_wurcs .
  }}
  OPTIONAL {{
    ?saccharide glycan:has_glycosequence ?glycoct_seq .
    ?glycoct_seq glycan:has_sequence ?glycoct ;
                 glycan:in_carbohydrate_format glycan:carbohydrate_format_glycoct .
  }}
}}
ORDER BY ?accession
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_value(binding: dict[str, Any], key: str) -> str:
    value = binding.get(key, {})
    return str(value.get("value", "")) if isinstance(value, dict) else ""


def collapse_registry_bindings(
    requested_ids: list[str], bindings: list[dict[str, Any]], retrieved_at: str
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, set[str]]] = {
        accession: {"wurcs": set(), "glycoct": set()} for accession in requested_ids
    }
    for binding in bindings:
        accession = _binding_value(binding, "accession")
        if accession not in grouped:
            continue
        wurcs = _binding_value(binding, "wurcs")
        glycoct = _binding_value(binding, "glycoct")
        if wurcs:
            grouped[accession]["wurcs"].add(wurcs)
        if glycoct:
            grouped[accession]["glycoct"].add(glycoct)

    rows: list[dict[str, Any]] = []
    for accession in requested_ids:
        wurcs_values = sorted(grouped[accession]["wurcs"])
        glycoct_values = sorted(grouped[accession]["glycoct"])
        found = bool(wurcs_values or glycoct_values)
        rows.append(
            {
                "glytoucan_id": accession,
                "wurcs": " | ".join(wurcs_values),
                "glycoct": " | ".join(glycoct_values),
                "wurcs_count": len(wurcs_values),
                "glycoct_count": len(glycoct_values),
                "registry_status": (
                    "not_returned"
                    if not found
                    else "ambiguous_multiple_sequences"
                    if len(wurcs_values) > 1 or len(glycoct_values) > 1
                    else "resolved"
                ),
                "source_url": f"https://glytoucan.org/Structures/Glycans/{accession}",
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def build_identifier_map(evidence: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "glycan_source_id", "glytoucan_id", "retrieval_url"}
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"Evidence table missing identifier-map columns: {sorted(missing)}")
    rows = []
    for key, group in evidence.groupby(["source", "glycan_source_id"], dropna=False):
        values = sorted(
            {
                str(value).strip()
                for value in group["glytoucan_id"]
                if pd.notna(value) and str(value).strip()
            }
        )
        rows.append(
            {
                "source": str(key[0]),
                "source_glycan_id": str(key[1]),
                "glytoucan_id": values[0] if len(values) == 1 else "",
                "mapping_method": "SugarBind sb:has_structure URI",
                "ambiguity_status": (
                    "unmapped" if not values else "unambiguous" if len(values) == 1 else "ambiguous"
                ),
                "source_record_count": len(group),
                "source_url": " | ".join(
                    sorted(
                        {
                            str(value).strip()
                            for value in group["retrieval_url"]
                            if pd.notna(value) and str(value).strip()
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "source_glycan_id"])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = pd.read_csv(args.evidence, keep_default_na=False)
    identifiers = sorted(
        {
            str(value).strip()
            for value in evidence["glytoucan_id"]
            if str(value).strip()
        }
    )
    invalid = sorted(value for value in identifiers if not ACCESSION_PATTERN.fullmatch(value))
    if invalid:
        raise ValueError(f"Invalid GlyTouCan accessions in evidence: {invalid}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.outdir / "raw_sparql"
    raw_dir.mkdir(exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    all_bindings: list[dict[str, Any]] = []
    queries: list[str] = []

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "GlycoAudit/1.0",
    }
    with httpx.Client(timeout=args.timeout, follow_redirects=True, headers=headers) as client:
        for index in range(0, len(identifiers), args.batch_size):
            batch = identifiers[index : index + args.batch_size]
            values = " ".join(f'"{accession}"' for accession in batch)
            query = QUERY_TEMPLATE.format(values=values)
            queries.append(query)
            response = client.post(SPARQL_ENDPOINT, data={"query": query})
            response.raise_for_status()
            payload = response.json()
            (raw_dir / f"batch_{index // args.batch_size + 1:03d}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            all_bindings.extend(payload.get("results", {}).get("bindings", []))

    registry_rows = collapse_registry_bindings(identifiers, all_bindings, retrieved_at)
    registry_path = args.outdir / "registry_table.csv"
    write_csv(registry_path, registry_rows, list(registry_rows[0].keys()))

    identifier_map = build_identifier_map(evidence)
    identifier_map_path = args.outdir / "glycan_identifier_map.csv"
    identifier_map.to_csv(identifier_map_path, index=False)

    excluded = [
        {
            "glytoucan_id": row["glytoucan_id"],
            "exclusion_stage": "targeted GlyTouCan registry query",
            "exclusion_reason": row["registry_status"],
            "details": row["source_url"],
        }
        for row in registry_rows
        if row["registry_status"] != "resolved"
    ]
    excluded_path = args.outdir / "registry_excluded_records.csv"
    write_csv(
        excluded_path,
        excluded,
        ["glytoucan_id", "exclusion_stage", "exclusion_reason", "details"],
    )

    query_path = args.outdir / "queries.sparql"
    query_path.write_text("\n\n# ---- batch ----\n\n".join(queries), encoding="utf-8")
    output_files = [registry_path, identifier_map_path, excluded_path, query_path]
    output_files.extend(sorted(raw_dir.glob("*.json")))
    metadata = {
        "source": "GlyTouCan targeted SPARQL",
        "endpoint": SPARQL_ENDPOINT,
        "retrieved_at": retrieved_at,
        "requested_accessions": len(identifiers),
        "resolved_accessions": sum(row["registry_status"] == "resolved" for row in registry_rows),
        "ambiguous_accessions": sum(
            row["registry_status"] == "ambiguous_multiple_sequences" for row in registry_rows
        ),
        "not_returned_accessions": sum(
            row["registry_status"] == "not_returned" for row in registry_rows
        ),
        "checksums": {str(path.relative_to(args.outdir)): sha256(path) for path in output_files},
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
