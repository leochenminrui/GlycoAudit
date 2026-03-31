"""GlycoShape source connector.

GlycoShape provides 3D structures of glycans.
Website: https://glycoshape.io/
API: https://glycoshape.io/api/

This connector downloads glycan 3D structures in PDB format and associated metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import MirrorConfig
from ..io import (
    DownloadRecord,
    DownloadStatus,
    HTTPClient,
    ItemType,
    make_download_record,
    save_json_response,
    save_text_response,
)
from ..logging import get_logger
from . import BaseSource, SourceItem

logger = get_logger("glycoshape")


# API endpoints - try multiple bases (glycoshape.org is current, glycoshape.io is legacy)
API_BASES = [
    "https://glycoshape.org/api",
    "https://glycoshape.io/api",
]
API_BASE = API_BASES[0]  # Default to current
GLYCAN_LIST_URL = f"{API_BASE}/glycan"
GLYCAN_DETAIL_URL = f"{API_BASE}/glycan"  # + /{glycan_id}
STRUCTURE_URL = f"{API_BASE}/structure"  # + /{glycan_id}

# GitHub raw data fallback (bulk downloads)
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Ojas-Singh/GlycoShape/main"


class GlycoShapeSource(BaseSource):
    """Source connector for GlycoShape 3D structure database."""

    SOURCE_NAME = "glycoshape"
    CATEGORY = "B"  # Glycan 3D structures

    def list_items(self) -> Iterator[SourceItem]:
        """List all glycan structures from GlycoShape.

        Priority order:
        1. Manual manifest (if exists)
        2. API endpoints (tries multiple bases)
        3. Save instructions for manual access
        """
        if not self.is_enabled:
            logger.info("GlycoShape source is disabled")
            return

        # Priority 1: Check for manual manifest
        manifest_path = self.data_dir / "manifest.csv"
        if manifest_path.exists():
            logger.info(f"Using manual manifest: {manifest_path}")
            yield from self._load_manifest(manifest_path)
            return

        logger.info("Fetching glycan list from GlycoShape API...")

        # Priority 2: Try API endpoints with multiple bases
        for api_base in API_BASES:
            glycan_list_url = f"{api_base}/glycan"
            try:
                logger.info(f"Trying API base: {api_base}")
                response = self.http.get_json(glycan_list_url)

                # The response format may vary - handle both list and dict
                glycans = []
                if isinstance(response, list):
                    glycans = response
                elif isinstance(response, dict):
                    glycans = response.get("data", response.get("glycans", []))

                if glycans:
                    logger.info(f"Found {len(glycans)} glycans from {api_base}")

                    for glycan in glycans:
                        # Handle different response formats
                        if isinstance(glycan, str):
                            glycan_id = glycan
                            metadata = {}
                        else:
                            glycan_id = glycan.get("id") or glycan.get("glycan_id", "")
                            metadata = glycan

                        if not glycan_id:
                            continue

                        yield SourceItem(
                            source=self.SOURCE_NAME,
                            item_id=glycan_id,
                            url=f"{api_base}/glycan/{glycan_id}",
                            item_type="structure",
                            metadata=metadata,
                        )
                    return  # Successfully got glycans, exit

            except Exception as e:
                logger.warning(f"API base {api_base} failed: {e}")
                continue

        # Priority 3: All APIs failed - save instructions
        logger.error("All GlycoShape API endpoints failed")
        self._save_access_instructions()

    def _load_manifest(self, manifest_path: Path) -> Iterator[SourceItem]:
        """Load glycans from a manual manifest CSV file.

        Expected columns: glytoucan_id, wurcs (optional), glycoct (optional), pdb_url (optional)
        """
        import csv

        try:
            with open(manifest_path, newline="") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    glycan_id = row.get("glytoucan_id", row.get("id", "")).strip()
                    if not glycan_id:
                        continue

                    metadata = {
                        "wurcs": row.get("wurcs", ""),
                        "glycoct": row.get("glycoct", ""),
                        "source": "manifest",
                    }

                    # Use provided PDB URL or construct from glycan ID
                    url = row.get("pdb_url", row.get("url", ""))
                    if not url:
                        url = f"manifest://{glycan_id}"

                    yield SourceItem(
                        source=self.SOURCE_NAME,
                        item_id=glycan_id,
                        url=url,
                        item_type="structure",
                        metadata=metadata,
                    )
                    count += 1

                logger.info(f"Loaded {count} glycans from manifest")

        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")

    def _save_access_instructions(self) -> None:
        """Save instructions for manual data access when API fails."""
        instructions = """
GlycoShape Data Access Instructions
====================================

The GlycoShape API is currently unavailable. To obtain 3D glycan structures:

OPTION 1: Manual Manifest Mode (Recommended)
---------------------------------------------
1. Create a CSV file: data/raw/glycoshape/manifest.csv
2. Add columns: glytoucan_id, wurcs, glycoct, pdb_url
3. Populate with glycans from GlyTouCan registry
4. Re-run the mirror

Example manifest.csv:
```
glytoucan_id,wurcs,glycoct,pdb_url
G00028MO,WURCS=2.0/...,RES 1b:...,
G00055MO,WURCS=2.0/...,RES 1b:...,
```

OPTION 2: Use GlyTouCan + External Tools
-----------------------------------------
1. The GlyTouCan registry is already downloaded (260k+ glycans)
2. Use glycan sequences (WURCS/GlycoCT) with:
   - GLYCAM-Web (https://glycam.org) for 3D generation
   - Re-Glyco tool for protein glycosylation

OPTION 3: Direct Download from GlycoShape
------------------------------------------
1. Visit: https://glycoshape.org
2. Search for glycans and download PDB files
3. Place in: data/raw/glycoshape/<glycan_id>/structure.pdb

OPTION 4: GitHub Bulk Data
---------------------------
1. Clone: https://github.com/Ojas-Singh/GlycoShape
2. Data may be in the repository or linked locations
3. Copy PDB files to the appropriate directories

Contact: ojas.singh.2023@mumail.ie (GlycoShape maintainer)

Generated: {timestamp}
"""
        from datetime import datetime

        instructions_path = self.data_dir / "ACCESS_INSTRUCTIONS.txt"
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(instructions_path, "w") as f:
            f.write(instructions.format(timestamp=datetime.now().isoformat()))

        logger.info(f"Saved access instructions: {instructions_path}")

    def _list_items_fallback(self) -> Iterator[SourceItem]:
        """Fallback method if main API fails."""
        logger.warning("Trying fallback glycan listing method...")

        # Try fetching the main page data
        try:
            # Some APIs use different pagination
            page = 1
            while True:
                url = f"{GLYCAN_LIST_URL}?page={page}&limit=100"
                response = self.http.get_json(url)

                items = response if isinstance(response, list) else response.get("items", [])
                if not items:
                    break

                for item in items:
                    glycan_id = item.get("id", "") if isinstance(item, dict) else str(item)
                    if glycan_id:
                        yield SourceItem(
                            source=self.SOURCE_NAME,
                            item_id=glycan_id,
                            url=f"{GLYCAN_DETAIL_URL}/{glycan_id}",
                            item_type="structure",
                            metadata=item if isinstance(item, dict) else {},
                        )

                page += 1
                if len(items) < 100:
                    break

        except Exception as e:
            logger.error(f"Fallback listing also failed: {e}")

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download glycan structure and metadata.

        For each glycan, we download:
        1. API metadata response (JSON)
        2. 3D structure file(s) (PDB format)

        In manifest mode (url starts with "manifest://"):
        - Only save metadata from manifest
        - Download PDB only if pdb_url was provided

        Args:
            item: Source item to download.

        Returns:
            List of download records.
        """
        records = []
        glycan_id = item.item_id

        # Create directory for this glycan
        glycan_dir = self.data_dir / glycan_id
        glycan_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        metadata_path = glycan_dir / "metadata.json"
        if metadata_path.exists():
            logger.debug(f"Already downloaded: {glycan_id}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.STRUCTURE,
                    url=item.url,
                    local_path=metadata_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        # Handle manifest mode
        if item.url.startswith("manifest://"):
            return self._download_from_manifest(item, glycan_dir, metadata_path)

        try:
            # Fetch glycan metadata from API
            metadata_url = f"{GLYCAN_DETAIL_URL}/{glycan_id}"
            metadata = self.http.get_json(metadata_url)

            # Save metadata
            sha256, size = save_json_response(metadata, metadata_path)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"{glycan_id}/metadata",
                    item_type=ItemType.METADATA,
                    url=metadata_url,
                    local_path=metadata_path,
                    sha256=sha256,
                    size=size,
                )
            )

            # Download structure file(s)
            structure_records = self._download_structures(glycan_id, glycan_dir, metadata)
            records.extend(structure_records)

        except Exception as e:
            logger.error(f"Failed to download {glycan_id}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.STRUCTURE,
                    url=item.url,
                    local_path=glycan_dir,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _download_structures(
        self,
        glycan_id: str,
        glycan_dir: Path,
        metadata: dict[str, Any],
    ) -> list[DownloadRecord]:
        """Download 3D structure files for a glycan.

        Args:
            glycan_id: Glycan identifier.
            glycan_dir: Directory to save files.
            metadata: Glycan metadata.

        Returns:
            List of download records.
        """
        records = []

        # Try different structure endpoints
        structure_urls = [
            (f"{STRUCTURE_URL}/{glycan_id}/pdb", "structure.pdb"),
            (f"{STRUCTURE_URL}/{glycan_id}", "structure_data.json"),
            (f"{API_BASE}/download/{glycan_id}/pdb", "structure.pdb"),
        ]

        # Check if metadata provides structure URLs
        if "pdb_url" in metadata:
            structure_urls.insert(0, (metadata["pdb_url"], "structure.pdb"))
        if "structures" in metadata:
            for i, struct in enumerate(metadata.get("structures", [])):
                if isinstance(struct, dict) and "url" in struct:
                    fmt = struct.get("format", "pdb").lower()
                    structure_urls.append((struct["url"], f"structure_{i}.{fmt}"))

        for url, filename in structure_urls:
            try:
                filepath = glycan_dir / filename

                if filename.endswith(".pdb"):
                    # Download PDB as text
                    content = self.http.get_text(url)
                    if content and "ATOM" in content:  # Validate PDB content
                        sha256, size = save_text_response(content, filepath)
                        records.append(
                            make_download_record(
                                source=self.SOURCE_NAME,
                                item_id=f"{glycan_id}/{filename}",
                                item_type=ItemType.STRUCTURE,
                                url=url,
                                local_path=filepath,
                                sha256=sha256,
                                size=size,
                            )
                        )
                        logger.debug(f"Downloaded structure: {filepath}")
                        break  # Successfully got PDB, don't try other URLs
                else:
                    # Download as JSON
                    content = self.http.get_json(url)
                    sha256, size = save_json_response(content, filepath)
                    records.append(
                        make_download_record(
                            source=self.SOURCE_NAME,
                            item_id=f"{glycan_id}/{filename}",
                            item_type=ItemType.STRUCTURE,
                            url=url,
                            local_path=filepath,
                            sha256=sha256,
                            size=size,
                        )
                    )

            except Exception as e:
                logger.debug(f"Structure download failed for {url}: {e}")
                continue

        if not records:
            logger.warning(f"No structure files found for {glycan_id}")

        return records

    def _download_from_manifest(
        self,
        item: SourceItem,
        glycan_dir: Path,
        metadata_path: Path,
    ) -> list[DownloadRecord]:
        """Handle download for manifest-sourced items.

        In manifest mode, we save the metadata from the manifest
        and optionally download PDB if a URL was provided.

        Args:
            item: Source item with manifest data.
            glycan_dir: Directory for this glycan.
            metadata_path: Path to save metadata.

        Returns:
            List of download records.
        """
        records = []
        glycan_id = item.item_id

        # Build metadata from manifest
        metadata = {
            "id": glycan_id,
            "source": "manifest",
            "wurcs": item.metadata.get("wurcs", ""),
            "glycoct": item.metadata.get("glycoct", ""),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save metadata
        sha256, size = save_json_response(metadata, metadata_path)
        records.append(
            make_download_record(
                source=self.SOURCE_NAME,
                item_id=f"{glycan_id}/metadata",
                item_type=ItemType.METADATA,
                url=item.url,
                local_path=metadata_path,
                sha256=sha256,
                size=size,
                extra={"source": "manifest"},
            )
        )

        logger.info(f"Saved manifest entry: {glycan_id}")
        return records

    def download_bulk_catalog(self) -> list[DownloadRecord]:
        """Download the complete glycan catalog.

        Creates a single file with all glycan metadata.
        """
        records = []

        catalog_path = self.data_dir / "catalog.json"
        logger.info("Downloading GlycoShape catalog...")

        try:
            response = self.http.get_json(GLYCAN_LIST_URL)

            sha256, size = save_json_response(response, catalog_path)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="catalog",
                    item_type=ItemType.METADATA,
                    url=GLYCAN_LIST_URL,
                    local_path=catalog_path,
                    sha256=sha256,
                    size=size,
                )
            )

            logger.info(f"Catalog downloaded: {catalog_path}")

        except Exception as e:
            logger.error(f"Catalog download failed: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="catalog",
                    item_type=ItemType.METADATA,
                    url=GLYCAN_LIST_URL,
                    local_path=catalog_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records
