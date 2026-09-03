#!/usr/bin/env python3
"""Fetch exact-accession GlycoShape structures for a targeted GlyTouCan table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


API_BASE = "https://glycoshape.org/api"
PDB_ID_PATTERN = re.compile(r"^REMARK\s+GlyTouCan ID:\s*(G[0-9A-Z]{7})\s*$", re.M)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def find_exact_branch(metadata: dict[str, Any], requested_id: str) -> tuple[str, dict[str, Any]]:
    """Return the exact alpha/beta metadata branch; archetype is not an exact PDB route."""
    for branch_name in ("alpha", "beta"):
        branch = metadata.get(branch_name)
        if isinstance(branch, dict) and str(branch.get("glytoucan", "")) == requested_id:
            return branch_name, branch
    archetype = metadata.get("archetype")
    if isinstance(archetype, dict) and str(archetype.get("glytoucan", "")) == requested_id:
        return "archetype", archetype
    return "", {}


def normalize_wurcs_link_order(value: str) -> str:
    """Normalize ordering of independent WURCS linkage clauses for comparison."""
    parts = str(value or "").strip().split("/")
    if len(parts) < 2:
        return str(value or "").strip()
    linkages = [token for token in parts[-1].split("_") if token]
    parts[-1] = "_".join(sorted(linkages))
    return "/".join(parts)


def validate_pdb(
    pdb_text: str,
    requested_id: str,
    requested_wurcs: str,
    branch_name: str,
    branch: dict[str, Any],
) -> tuple[bool, str, str]:
    if "ATOM" not in pdb_text or "MODEL" not in pdb_text:
        return False, "invalid_or_empty_pdb", ""
    match = PDB_ID_PATTERN.search(pdb_text)
    returned_id = match.group(1) if match else ""
    if not returned_id:
        return False, "pdb_missing_glytoucan_remark", ""
    if returned_id != requested_id:
        return False, "pdb_identifier_mismatch", returned_id
    if branch_name not in {"alpha", "beta"}:
        return False, "archetype_has_no_exact_anomeric_pdb_route", returned_id
    branch_wurcs = str(branch.get("wurcs", ""))
    if (
        requested_wurcs
        and branch_wurcs
        and normalize_wurcs_link_order(requested_wurcs)
        != normalize_wurcs_link_order(branch_wurcs)
    ):
        return False, "registry_glycoshape_wurcs_mismatch", returned_id
    return True, "", returned_id


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = pd.read_csv(args.registry, keep_default_na=False)
    registry = registry[registry["registry_status"] == "resolved"].copy()
    if args.max_items is not None:
        registry = registry.head(args.max_items)

    args.outdir.mkdir(parents=True, exist_ok=True)
    structures_dir = args.outdir / "structures"
    structures_dir.mkdir(exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []

    headers = {"User-Agent": "GlycoAudit/1.0"}
    with httpx.Client(timeout=args.timeout, follow_redirects=True, headers=headers) as client:
        for number, row in enumerate(registry.to_dict(orient="records"), start=1):
            requested_id = str(row["glytoucan_id"])
            requested_wurcs = str(row.get("wurcs", ""))
            glycan_dir = structures_dir / requested_id
            glycan_dir.mkdir(exist_ok=True)
            metadata_url = f"{API_BASE}/glycan/{requested_id}"
            metadata_path = glycan_dir / "glycoshape_metadata.json"
            pdb_path = glycan_dir / "structure_1.pdb"
            status = "failed"
            reason = ""
            branch_name = ""
            branch: dict[str, Any] = {}
            pdb_url = ""
            returned_id = ""
            pdb_sha256 = ""
            pdb_size = 0
            model_count = 0
            error_detail = ""

            try:
                metadata_response = client.get(metadata_url)
                metadata_response.raise_for_status()
                metadata = metadata_response.json()
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                branch_name, branch = find_exact_branch(metadata, requested_id)
                if not branch_name:
                    reason = "requested_accession_not_exposed_by_glycoshape_metadata"
                elif branch_name == "archetype":
                    reason = "archetype_has_no_exact_anomeric_pdb_route"
                elif not branch.get("ID"):
                    reason = "glycoshape_branch_missing_structure_id"
                else:
                    pdb_url = (
                        f"{API_BASE}/structure/PDB/{branch['ID']}"
                        f"?level=1&anomer={branch_name}&format=pdb"
                    )
                    pdb_response = client.get(pdb_url)
                    pdb_response.raise_for_status()
                    pdb_text = pdb_response.text
                    valid, reason, returned_id = validate_pdb(
                        pdb_text,
                        requested_id,
                        requested_wurcs,
                        branch_name,
                        branch,
                    )
                    if valid:
                        pdb_path.write_text(pdb_text, encoding="utf-8")
                        pdb_bytes = pdb_text.encode("utf-8")
                        pdb_sha256 = sha256_bytes(pdb_bytes)
                        pdb_size = len(pdb_bytes)
                        model_count = len(re.findall(r"^MODEL\s+", pdb_text, flags=re.M))
                        status = "ok"
            except httpx.HTTPStatusError as exc:
                reason = reason or f"glycoshape_http_{exc.response.status_code}"
                error_detail = str(exc)
            except Exception as exc:
                reason = reason or "glycoshape_request_failed"
                error_detail = str(exc)

            metadata_record = {
                "requested_glytoucan_id": requested_id,
                "requested_wurcs": requested_wurcs,
                "glycoshape_branch": branch_name,
                "glycoshape_structure_id": branch.get("ID", ""),
                "glycoshape_branch_glytoucan_id": branch.get("glytoucan", ""),
                "glycoshape_branch_wurcs": branch.get("wurcs", ""),
                "pdb_returned_glytoucan_id": returned_id,
                "metadata_url": metadata_url,
                "pdb_url": pdb_url,
                "status": status,
                "failure_reason": reason,
                "error_detail": error_detail,
                "model_count": model_count,
                "pdb_sha256": pdb_sha256,
                "retrieved_at": retrieved_at,
                "license": "CC BY-NC-ND 4.0 for academic use; as stated in GlycoShape PDB REMARK",
            }
            (glycan_dir / "meta.json").write_text(
                json.dumps(metadata_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            structure_id = f"GlycoShape:{branch.get('ID', '')}:{branch_name}" if status == "ok" else ""
            manifest_rows.append(
                {
                    "glytoucan_id": requested_id,
                    "structure_id": structure_id,
                    "candidate_id": "level_1_multimodel" if status == "ok" else "",
                    "source": "GlycoShape",
                    "source_accession": branch.get("ID", ""),
                    "source_url": pdb_url or metadata_url,
                    "snapshot_date": retrieved_at[:10],
                    "local_path": str(pdb_path) if status == "ok" else "",
                    "sha256": pdb_sha256,
                    "bytes": pdb_size,
                    "model_count": model_count,
                    "forcefield": "GLYCAM_06j" if status == "ok" else "",
                    "ensemble_level": "GlycoShape level_1 multi-model PDB" if status == "ok" else "",
                    "status": status,
                    "failure_reason": reason,
                }
            )
            map_rows.append(
                {
                    "glytoucan_id": requested_id,
                    "structure_id": structure_id,
                    "join_method": "direct_exact_glycoshape_accession_and_wurcs" if status == "ok" else "failed_exact_join",
                    "valid_structure": status == "ok",
                    "requested_accession": requested_id,
                    "returned_accession": returned_id,
                    "glycoshape_branch": branch_name,
                    "sequence_match": (
                        bool(
                            requested_wurcs
                            and normalize_wurcs_link_order(requested_wurcs)
                            == normalize_wurcs_link_order(str(branch.get("wurcs", "")))
                        )
                        if branch
                        else False
                    ),
                    "model_count": model_count,
                    "failure_reason": reason,
                }
            )
            print(
                f"[{number}/{len(registry)}] {requested_id}: {status}"
                + (f" ({reason})" if reason else ""),
                flush=True,
            )
            if args.delay > 0:
                time.sleep(args.delay)

    structure_columns = [
        "glytoucan_id",
        "structure_id",
        "candidate_id",
        "source",
        "source_accession",
        "source_url",
        "snapshot_date",
        "local_path",
        "sha256",
        "bytes",
        "model_count",
        "forcefield",
        "ensemble_level",
        "status",
        "failure_reason",
    ]
    map_columns = [
        "glytoucan_id",
        "structure_id",
        "join_method",
        "valid_structure",
        "requested_accession",
        "returned_accession",
        "glycoshape_branch",
        "sequence_match",
        "model_count",
        "failure_reason",
    ]
    write_csv(args.outdir / "structure_manifest.csv", manifest_rows, structure_columns)
    write_csv(args.outdir / "glycan_structure_map.csv", map_rows, map_columns)

    summary = {
        "retrieved_at": retrieved_at,
        "requested": len(manifest_rows),
        "valid_exact_structures": sum(row["status"] == "ok" for row in manifest_rows),
        "failed_exact_structures": sum(row["status"] != "ok" for row in manifest_rows),
        "failure_reasons": {},
        "total_models": sum(int(row["model_count"]) for row in manifest_rows),
    }
    for row in manifest_rows:
        if row["failure_reason"]:
            summary["failure_reasons"][row["failure_reason"]] = (
                summary["failure_reasons"].get(row["failure_reason"], 0) + 1
            )
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
