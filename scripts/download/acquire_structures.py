#!/usr/bin/env python3
"""
Acquire Targeted Structures for SugarBind Glycans

This script acquires 3D structures for glycans in the targeted manifest.
It tries multiple routes in priority order:

Route 0: Skip if structure already exists
Route 1: Download from GlycoShape API (if glycan is in available list)
Route 2: Copy from library mapping (if WURCS match exists)
Route 3: Generate from WURCS (placeholder - requires glycosylator)

Input:
    data/targets/sugarbind_glycans.csv - Target glycans with sequences

Output:
    data/raw/structures/targeted_sugarbind/<glytoucan_id>/
        structure_1.pdb
        meta.json (method, source, sha256, status, reason)
"""

import csv
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
MANIFEST_FILE = BASE_PATH / "data/targets/sugarbind_glycans.csv"
GLYCOSHAPE_AVAILABLE = BASE_PATH / "data/raw/glycoshape/_glycoshape_available.json"
EXISTING_STRUCTURES = BASE_PATH / "data/raw/glycoshape/structures"
LIBRARY_MAPPING = BASE_PATH / "data/raw/glycoshape/library_mapping_to_glytoucan.csv"

OUTPUT_DIR = BASE_PATH / "data/raw/structures/targeted_sugarbind"
DOWNLOADS_LOG = BASE_PATH / "data/meta/downloads.jsonl"
REPORTS_DIR = BASE_PATH / "reports"

# GlycoShape API
API_BASES = [
    "https://glycoshape.org/api",
    "https://glycoshape.io/api",
]

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between API requests
MAX_RETRIES = 3
TIMEOUT = 60


@dataclass
class AcquisitionResult:
    """Result of structure acquisition attempt."""
    glytoucan_id: str
    status: str  # ok, failed, skipped
    method: str  # existing, glycoshape_api, library_copy, generation, none
    reason: str
    source_url: Optional[str] = None
    structure_path: Optional[str] = None
    sha256: Optional[str] = None
    file_size: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def log_download(result: AcquisitionResult) -> None:
    """Append acquisition result to downloads.jsonl."""
    DOWNLOADS_LOG.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "source": "targeted_structures",
        "source_item_id": result.glytoucan_id,
        "item_type": "structure",
        "url": result.source_url or "",
        "local_path": result.structure_path or "",
        "sha256": result.sha256 or "",
        "bytes": result.file_size,
        "downloaded_at": result.timestamp,
        "status": result.status,
        "error": result.reason if result.status == "failed" else None,
        "extra_json": {
            "method": result.method,
            "reason": result.reason,
        },
    }

    with open(DOWNLOADS_LOG, 'a') as f:
        f.write(json.dumps(record) + '\n')


def load_manifest(filepath: Path) -> pd.DataFrame:
    """Load targeted glycan manifest."""
    return pd.read_csv(filepath)


def load_glycoshape_available(filepath: Path) -> set:
    """Load set of GlyTouCan IDs available in GlycoShape."""
    if not filepath.exists():
        return set()
    with open(filepath) as f:
        return set(json.load(f))


def load_library_mapping(filepath: Path) -> dict:
    """Load library mapping as dict: glytoucan_id -> library_entry."""
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    mapping = {}
    for _, row in df.iterrows():
        mapping[row["glytoucan_id"]] = {
            "library_entry": row["library_entry"],
            "structure_path": row["structure_path"],
        }
    return mapping


def download_from_glycoshape(
    glytoucan_id: str,
    output_dir: Path,
) -> AcquisitionResult:
    """
    Download structure from GlycoShape API.

    Tries multiple API bases and structure endpoints.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_path = output_dir / "structure_1.pdb"

    # Try each API base
    for api_base in API_BASES:
        # Try different structure endpoints (correct endpoint is /api/pdb/{id})
        endpoints = [
            f"{api_base}/pdb/{glytoucan_id}",
            f"{api_base}/structure/{glytoucan_id}/pdb",
            f"{api_base}/download/{glytoucan_id}/pdb",
        ]

        for endpoint in endpoints:
            try:
                time.sleep(REQUEST_DELAY)

                with httpx.Client(timeout=TIMEOUT) as client:
                    response = client.get(
                        endpoint,
                        headers={"User-Agent": "public-glyco-mirror/1.0"},
                        follow_redirects=True,
                    )

                    if response.status_code == 200:
                        content = response.text

                        # Validate PDB content
                        if "ATOM" in content or "HETATM" in content:
                            # Save structure
                            with open(structure_path, 'w') as f:
                                f.write(content)

                            sha256 = compute_sha256(structure_path)
                            file_size = structure_path.stat().st_size

                            # Save meta.json
                            meta = {
                                "glytoucan_id": glytoucan_id,
                                "method": "glycoshape_api",
                                "source_url": endpoint,
                                "sha256": sha256,
                                "file_size": file_size,
                                "acquired_at": timestamp,
                                "status": "ok",
                            }
                            with open(output_dir / "meta.json", 'w') as f:
                                json.dump(meta, f, indent=2)

                            return AcquisitionResult(
                                glytoucan_id=glytoucan_id,
                                status="ok",
                                method="glycoshape_api",
                                reason="downloaded_from_glycoshape",
                                source_url=endpoint,
                                structure_path=str(structure_path),
                                sha256=sha256,
                                file_size=file_size,
                                timestamp=timestamp,
                            )

            except Exception as e:
                continue  # Try next endpoint

    # All endpoints failed
    meta = {
        "glytoucan_id": glytoucan_id,
        "method": "glycoshape_api",
        "status": "failed",
        "reason": "api_download_failed",
        "attempted_at": timestamp,
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return AcquisitionResult(
        glytoucan_id=glytoucan_id,
        status="failed",
        method="glycoshape_api",
        reason="api_download_failed",
        timestamp=timestamp,
    )


def copy_existing_structure(
    glytoucan_id: str,
    source_dir: Path,
    output_dir: Path,
) -> AcquisitionResult:
    """Copy structure from existing structures directory."""
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find PDB file in source
    source_glycan_dir = source_dir / glytoucan_id
    pdb_files = list(source_glycan_dir.glob("*.pdb"))

    if not pdb_files:
        return AcquisitionResult(
            glytoucan_id=glytoucan_id,
            status="failed",
            method="existing",
            reason="no_pdb_in_source",
            timestamp=timestamp,
        )

    # Copy first PDB file
    source_pdb = pdb_files[0]
    dest_pdb = output_dir / "structure_1.pdb"

    shutil.copy2(source_pdb, dest_pdb)

    sha256 = compute_sha256(dest_pdb)
    file_size = dest_pdb.stat().st_size

    # Save meta.json
    meta = {
        "glytoucan_id": glytoucan_id,
        "method": "existing_copy",
        "source_path": str(source_pdb),
        "sha256": sha256,
        "file_size": file_size,
        "acquired_at": timestamp,
        "status": "ok",
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return AcquisitionResult(
        glytoucan_id=glytoucan_id,
        status="ok",
        method="existing",
        reason="copied_from_existing",
        source_url=str(source_pdb),
        structure_path=str(dest_pdb),
        sha256=sha256,
        file_size=file_size,
        timestamp=timestamp,
    )


def copy_from_library_mapping(
    glytoucan_id: str,
    library_info: dict,
    output_dir: Path,
) -> AcquisitionResult:
    """Copy structure from library mapping."""
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(library_info["structure_path"])

    if not source_path.exists():
        return AcquisitionResult(
            glytoucan_id=glytoucan_id,
            status="failed",
            method="library_copy",
            reason="library_structure_not_found",
            timestamp=timestamp,
        )

    dest_pdb = output_dir / "structure_1.pdb"
    shutil.copy2(source_path, dest_pdb)

    sha256 = compute_sha256(dest_pdb)
    file_size = dest_pdb.stat().st_size

    # Save meta.json
    meta = {
        "glytoucan_id": glytoucan_id,
        "method": "library_copy",
        "library_entry": library_info["library_entry"],
        "source_path": str(source_path),
        "sha256": sha256,
        "file_size": file_size,
        "acquired_at": timestamp,
        "status": "ok",
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return AcquisitionResult(
        glytoucan_id=glytoucan_id,
        status="ok",
        method="library_copy",
        reason="copied_from_library",
        source_url=str(source_path),
        structure_path=str(dest_pdb),
        sha256=sha256,
        file_size=file_size,
        timestamp=timestamp,
    )


def acquire_structure(
    glytoucan_id: str,
    manifest_row: pd.Series,
    glycoshape_available: set,
    library_mapping: dict,
    dry_run: bool = False,
) -> AcquisitionResult:
    """
    Acquire structure for a glycan using priority routes.

    Route 0: Skip if already acquired
    Route 1: Download from GlycoShape API
    Route 2: Copy from library mapping
    Route 3: Generate from WURCS (not implemented)
    """
    output_dir = OUTPUT_DIR / glytoucan_id
    timestamp = datetime.now(timezone.utc).isoformat()

    # Route 0: Check if already acquired
    meta_file = output_dir / "meta.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("status") == "ok":
                return AcquisitionResult(
                    glytoucan_id=glytoucan_id,
                    status="skipped",
                    method=meta.get("method", "existing"),
                    reason="already_acquired",
                    structure_path=str(output_dir / "structure_1.pdb"),
                    sha256=meta.get("sha256"),
                    file_size=meta.get("file_size", 0),
                    timestamp=timestamp,
                )
        except:
            pass  # Continue with acquisition

    if dry_run:
        # Determine what method would be used
        if manifest_row.get("has_existing_structure"):
            method = "existing"
        elif glytoucan_id in glycoshape_available:
            method = "glycoshape_api"
        elif glytoucan_id in library_mapping:
            method = "library_copy"
        else:
            method = "generation"

        return AcquisitionResult(
            glytoucan_id=glytoucan_id,
            status="dry_run",
            method=method,
            reason="would_acquire",
            timestamp=timestamp,
        )

    # Route 1: Copy from existing structures
    if manifest_row.get("has_existing_structure"):
        result = copy_existing_structure(glytoucan_id, EXISTING_STRUCTURES, output_dir)
        if result.status == "ok":
            return result

    # Route 2: Download from GlycoShape API
    if glytoucan_id in glycoshape_available:
        result = download_from_glycoshape(glytoucan_id, output_dir)
        if result.status == "ok":
            return result

    # Route 3: Copy from library mapping
    if glytoucan_id in library_mapping:
        result = copy_from_library_mapping(
            glytoucan_id,
            library_mapping[glytoucan_id],
            output_dir,
        )
        if result.status == "ok":
            return result

    # Route 4: Generation from WURCS (placeholder)
    wurcs = manifest_row.get("wurcs", "")
    if wurcs:
        # Save meta for future generation
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "glytoucan_id": glytoucan_id,
            "method": "generation",
            "wurcs": wurcs,
            "status": "failed",
            "reason": "generation_not_implemented",
            "attempted_at": timestamp,
        }
        with open(output_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        return AcquisitionResult(
            glytoucan_id=glytoucan_id,
            status="failed",
            method="generation",
            reason="generation_not_implemented",
            timestamp=timestamp,
        )

    # No route available
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "glytoucan_id": glytoucan_id,
        "method": "none",
        "status": "failed",
        "reason": "no_acquisition_route",
        "attempted_at": timestamp,
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return AcquisitionResult(
        glytoucan_id=glytoucan_id,
        status="failed",
        method="none",
        reason="no_acquisition_route",
        timestamp=timestamp,
    )


def generate_acquisition_report(
    results: list[AcquisitionResult],
    output_path: Path,
) -> None:
    """Generate acquisition summary report."""
    now = datetime.now(timezone.utc).isoformat()

    # Count by status and method
    status_counts = {}
    method_counts = {}
    reason_counts = {}

    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        method_counts[r.method] = method_counts.get(r.method, 0) + 1
        reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1

    total = len(results)
    ok_count = status_counts.get("ok", 0)
    skipped_count = status_counts.get("skipped", 0)
    failed_count = status_counts.get("failed", 0)

    report = f"""# Targeted Structure Acquisition Report

**Generated:** {now}

---

## Summary

| Status | Count | % |
|--------|-------|---|
| OK (new) | {ok_count} | {100*ok_count/total:.1f}% |
| Skipped (existing) | {skipped_count} | {100*skipped_count/total:.1f}% |
| Failed | {failed_count} | {100*failed_count/total:.1f}% |
| **Total** | {total} | 100% |

**Coverage:** {ok_count + skipped_count} / {total} = {100*(ok_count+skipped_count)/total:.1f}%

---

## By Method

| Method | Count | % |
|--------|-------|---|
"""
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        report += f"| {method} | {count} | {100*count/total:.1f}% |\n"

    report += """

---

## Failure Reasons

| Reason | Count |
|--------|-------|
"""
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        if "fail" in reason.lower() or "not_" in reason.lower():
            report += f"| {reason} | {count} |\n"

    # List failed glycans
    failed = [r for r in results if r.status == "failed"]
    if failed:
        report += f"""

---

## Failed Glycans ({len(failed)})

| GlyTouCan ID | Method | Reason |
|--------------|--------|--------|
"""
        for r in failed[:50]:
            report += f"| {r.glytoucan_id} | {r.method} | {r.reason} |\n"

        if len(failed) > 50:
            report += f"\n... and {len(failed) - 50} more\n"

    report += """

---

## Output Files

| Location | Description |
|----------|-------------|
| `data/raw/structures/targeted_sugarbind/` | Acquired structures |
| `data/raw/structures/targeted_sugarbind/<id>/meta.json` | Per-glycan metadata |
| `data/meta/downloads.jsonl` | Download log |

"""

    output_path.write_text(report)
    print(f"Report saved: {output_path}")


def main(
    max_items: Optional[int] = None,
    dry_run: bool = False,
    only_status: bool = False,
) -> dict:
    """
    Main acquisition function.

    Args:
        max_items: Maximum items to process
        dry_run: If True, don't actually download
        only_status: If True, only report current status

    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("TARGETED STRUCTURE ACQUISITION")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    manifest = load_manifest(MANIFEST_FILE)
    glycoshape_available = load_glycoshape_available(GLYCOSHAPE_AVAILABLE)
    library_mapping = load_library_mapping(LIBRARY_MAPPING)

    print(f"Target glycans: {len(manifest)}")
    print(f"GlycoShape available: {len(glycoshape_available)}")
    print(f"Library mappings: {len(library_mapping)}")
    print(f"Dry run: {dry_run}")

    if only_status:
        # Just report current status
        ok = 0
        failed = 0
        pending = 0

        for _, row in manifest.iterrows():
            gtc_id = row["glytoucan_id"]
            meta_file = OUTPUT_DIR / gtc_id / "meta.json"

            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    if meta.get("status") == "ok":
                        ok += 1
                    else:
                        failed += 1
                except:
                    pending += 1
            else:
                pending += 1

        print("\n" + "-" * 40)
        print("CURRENT STATUS")
        print("-" * 40)
        print(f"OK: {ok}")
        print(f"Failed: {failed}")
        print(f"Pending: {pending}")
        print(f"Coverage: {100*ok/len(manifest):.1f}%")

        return {"ok": ok, "failed": failed, "pending": pending}

    # Process each glycan
    results = []
    processed = 0

    for idx, row in manifest.iterrows():
        if max_items and processed >= max_items:
            break

        gtc_id = row["glytoucan_id"]
        print(f"\n[{processed+1}/{len(manifest)}] {gtc_id}")

        result = acquire_structure(
            glytoucan_id=gtc_id,
            manifest_row=row,
            glycoshape_available=glycoshape_available,
            library_mapping=library_mapping,
            dry_run=dry_run,
        )

        results.append(result)

        # Log result
        if not dry_run:
            log_download(result)

        print(f"  Status: {result.status} | Method: {result.method} | Reason: {result.reason}")
        processed += 1

    # Generate report
    report_path = REPORTS_DIR / "targeted_acquisition_summary.md"
    generate_acquisition_report(results, report_path)

    # Summary stats
    stats = {
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "failed": sum(1 for r in results if r.status == "failed"),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"OK: {stats['ok']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Coverage: {100*(stats['ok']+stats['skipped'])/stats['total']:.1f}%")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Acquire targeted structures")
    parser.add_argument("--max-items", type=int, help="Max items to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually download")
    parser.add_argument("--only-status", action="store_true", help="Only report status")

    args = parser.parse_args()
    main(max_items=args.max_items, dry_run=args.dry_run, only_status=args.only_status)
