"""GlyGen source connector.

GlyGen is a data integration and dissemination project for glycosciences.
Website: https://www.glygen.org/
API: https://api.glygen.org/

This connector uses the GlyGen REST API to download:
- Glycan data and annotations
- Protein-glycan interactions
- Glycan structure metadata

API Documentation: https://api.glygen.org/
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
)
from ..logging import get_logger
from . import BaseSource, SourceItem

logger = get_logger("glygen")


# API endpoints
API_BASE = "https://api.glygen.org"

# Main data endpoints
ENDPOINTS = {
    "glycan_list": f"{API_BASE}/glycan/list",
    "glycan_detail": f"{API_BASE}/glycan/detail",
    "protein_list": f"{API_BASE}/protein/list",
    "protein_detail": f"{API_BASE}/protein/detail",
    "glycan_search": f"{API_BASE}/glycan/search",
    "glycoprotein_search": f"{API_BASE}/glycoprotein/search",
}

# Data types to fetch
DATA_TYPES = [
    "glycan",
    "protein",
    "motif",
    "pathway",
]


class GlyGenSource(BaseSource):
    """Source connector for GlyGen data integration platform."""

    SOURCE_NAME = "glygen"
    CATEGORY = "C"  # Lectin-glycan binding / integrative metadata

    def list_items(self) -> Iterator[SourceItem]:
        """List available data from GlyGen.

        GlyGen provides various data types. This connector fetches:
        1. Glycan list and details
        2. Protein-glycan interactions
        3. Glycan annotations

        Yields:
            SourceItem instances.
        """
        if not self.is_enabled:
            logger.info("GlyGen source is disabled")
            return

        logger.info("Fetching GlyGen data catalog...")

        # First, yield catalog items (bulk downloads)
        yield from self._list_catalog_items()

        # Then, yield individual glycan items (if mode allows)
        mode = self.source_config.mode
        if mode in ("full", "detailed"):
            yield from self._list_glycan_items()

    def _list_catalog_items(self) -> Iterator[SourceItem]:
        """List catalog-level data items."""
        # These are bulk data dumps available from GlyGen

        catalog_items = [
            {
                "id": "glycan_list",
                "url": ENDPOINTS["glycan_list"],
                "name": "Complete glycan list",
                "type": "bulk_export",
            },
            {
                "id": "motif_list",
                "url": f"{API_BASE}/motif/list",
                "name": "Glycan motif list",
                "type": "metadata",
            },
        ]

        for item in catalog_items:
            yield SourceItem(
                source=self.SOURCE_NAME,
                item_id=item["id"],
                url=item["url"],
                item_type=item["type"],
                metadata={"name": item["name"]},
            )

    def _list_glycan_items(self) -> Iterator[SourceItem]:
        """List individual glycan items for detailed download."""
        logger.info("Fetching glycan list from GlyGen API...")

        try:
            # The list endpoint requires a POST with query parameters
            # First, get a sample to understand the format
            response = self._search_glycans(limit=1000, offset=0)

            glycans = response.get("results", [])
            total = response.get("pagination", {}).get("total_count", len(glycans))

            logger.info(f"Found {total} glycans in GlyGen")

            # Paginate through all glycans
            offset = 0
            while offset < total:
                if offset > 0:
                    response = self._search_glycans(limit=1000, offset=offset)
                    glycans = response.get("results", [])

                for glycan in glycans:
                    glytoucan_ac = glycan.get("glytoucan_ac", "")
                    if not glytoucan_ac:
                        continue

                    yield SourceItem(
                        source=self.SOURCE_NAME,
                        item_id=glytoucan_ac,
                        url=f"{ENDPOINTS['glycan_detail']}/{glytoucan_ac}",
                        item_type="registry_record",
                        metadata=glycan,
                    )

                offset += len(glycans)
                if len(glycans) < 1000:
                    break

        except Exception as e:
            logger.error(f"Failed to list glycans: {e}")

    def _search_glycans(
        self,
        limit: int = 100,
        offset: int = 0,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search glycans using the API.

        Args:
            limit: Maximum results to return.
            offset: Pagination offset.
            query: Optional query parameters.

        Returns:
            API response dictionary.
        """
        url = f"{API_BASE}/glycan/search"

        payload = {
            "query": query or {},
            "pagination": {
                "offset": offset,
                "limit": limit,
                "sort": "glytoucan_ac",
                "order": "asc",
            },
        }

        # GlyGen API uses POST for searches
        response = self.http._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download a GlyGen data item.

        Args:
            item: Source item to download.

        Returns:
            List of download records.
        """
        records = []
        item_id = item.item_id

        # Determine if this is a catalog item or individual glycan
        if item.item_type in ("bulk_export", "metadata"):
            return self._download_catalog_item(item)
        else:
            return self._download_glycan_item(item)

    def _download_catalog_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download a catalog-level data item."""
        records = []
        item_id = item.item_id

        filepath = self.data_dir / f"{item_id}.json"

        # Check if already downloaded
        if filepath.exists():
            logger.debug(f"Already downloaded: {item_id}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=item_id,
                    item_type=ItemType.BULK_EXPORT,
                    url=item.url,
                    local_path=filepath,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        try:
            # For list endpoints, use POST with empty query
            if "list" in item.url or "search" in item.url:
                response = self._fetch_all_pages(item.url)
            else:
                response = self.http.get_json(item.url)

            sha256, size = save_json_response(response, filepath)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=item_id,
                    item_type=ItemType.BULK_EXPORT,
                    url=item.url,
                    local_path=filepath,
                    sha256=sha256,
                    size=size,
                )
            )

            logger.info(f"Downloaded catalog item: {item_id}")

        except Exception as e:
            error_str = str(e)
            # Check for temporary server errors (502, 503, 504) - mark as retry_later
            is_temporary = any(code in error_str for code in ["502", "503", "504", "timeout", "Timeout"])

            if is_temporary:
                logger.warning(f"GlyGen temporarily unavailable for {item_id}: {e}")
                logger.info("GlyGen unavailable due to upstream 502; safe to retry later.")
                self._save_retry_instructions()
                records.append(
                    make_download_record(
                        source=self.SOURCE_NAME,
                        item_id=item_id,
                        item_type=ItemType.BULK_EXPORT,
                        url=item.url,
                        local_path=filepath,
                        sha256="",
                        size=0,
                        status=DownloadStatus.FAILED,
                        error=f"RETRY_LATER: {error_str}",
                        extra={"retry_later": True, "reason": "upstream_502"},
                    )
                )
            else:
                logger.error(f"Failed to download {item_id}: {e}")
                records.append(
                    make_download_record(
                        source=self.SOURCE_NAME,
                        item_id=item_id,
                        item_type=ItemType.BULK_EXPORT,
                        url=item.url,
                        local_path=filepath,
                        sha256="",
                        size=0,
                        status=DownloadStatus.FAILED,
                        error=error_str,
                    )
                )

        return records

    def _save_retry_instructions(self) -> None:
        """Save instructions when GlyGen is temporarily unavailable."""
        instructions = """
GlyGen Data Access - Temporarily Unavailable
=============================================

The GlyGen API returned a 502 Proxy Error, indicating temporary server issues.

STATUS: RETRY_LATER
--------------------

This is a temporary issue. The GlyGen API is generally reliable.

TO RETRY:
---------

1. Wait a few hours and run:
   python scripts/run_stage.py --config configs/mirror.yaml --stage 1

2. Or retry just GlyGen:
   python scripts/mirror_category.py --config configs/mirror.yaml --category glygen

MANUAL ACCESS:
--------------

If the API remains unavailable:
1. Visit: https://www.glygen.org/
2. Use the web interface to search and download data
3. Place files in: data/raw/glygen/

API DOCUMENTATION:
------------------

GlyGen API: https://api.glygen.org/
Web Interface: https://www.glygen.org/

Generated: {timestamp}
"""
        from datetime import datetime

        instructions_path = self.data_dir / "RETRY_INSTRUCTIONS.txt"
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(instructions_path, "w") as f:
            f.write(instructions.format(timestamp=datetime.now().isoformat()))

    def _download_glycan_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download an individual glycan record."""
        records = []
        glycan_id = item.item_id

        # Create directory for this glycan
        glycan_dir = self.data_dir / "glycans" / glycan_id
        glycan_dir.mkdir(parents=True, exist_ok=True)

        detail_path = glycan_dir / "detail.json"

        # Check if already downloaded
        if detail_path.exists():
            logger.debug(f"Already downloaded: {glycan_id}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.REGISTRY_RECORD,
                    url=item.url,
                    local_path=detail_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        try:
            # Fetch glycan detail
            # GlyGen detail endpoint uses POST
            url = f"{ENDPOINTS['glycan_detail']}/{glycan_id}"

            try:
                response = self.http.get_json(url)
            except Exception:
                # Try POST if GET fails
                response = self.http._client.post(
                    ENDPOINTS["glycan_detail"],
                    json={"glytoucan_ac": glycan_id},
                ).json()

            sha256, size = save_json_response(response, detail_path)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.REGISTRY_RECORD,
                    url=url,
                    local_path=detail_path,
                    sha256=sha256,
                    size=size,
                    extra=self._extract_glycan_metadata(response),
                )
            )

        except Exception as e:
            logger.error(f"Failed to download glycan {glycan_id}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.REGISTRY_RECORD,
                    url=item.url,
                    local_path=detail_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _fetch_all_pages(self, url: str) -> dict[str, Any]:
        """Fetch all pages from a paginated endpoint.

        Args:
            url: API endpoint URL.

        Returns:
            Combined results from all pages.
        """
        all_results = []
        offset = 0
        limit = 500
        total = None

        while True:
            payload = {
                "query": {},
                "pagination": {
                    "offset": offset,
                    "limit": limit,
                    "sort": "glytoucan_ac",
                    "order": "asc",
                },
            }

            response = self.http._client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            all_results.extend(results)

            if total is None:
                total = data.get("pagination", {}).get("total_count", len(results))
                logger.info(f"Fetching {total} records from {url}")

            offset += len(results)

            if len(results) < limit or offset >= total:
                break

        return {
            "results": all_results,
            "total_count": len(all_results),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_glycan_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract key metadata from glycan detail response."""
        return {
            "glytoucan_ac": response.get("glytoucan_ac", ""),
            "mass": response.get("mass"),
            "mass_pme": response.get("mass_pme"),
            "species": [s.get("name") for s in response.get("species", [])],
            "motifs": [m.get("name") for m in response.get("motifs", [])],
            "glycoprotein_count": len(response.get("glycoproteins", [])),
        }

    def download_query_snapshot(
        self,
        query_name: str,
        query: dict[str, Any],
    ) -> list[DownloadRecord]:
        """Download results of a specific query.

        Useful for reproducible snapshots of specific data subsets.

        Args:
            query_name: Name for the query (used in filename).
            query: Query parameters for the search.

        Returns:
            List of download records.
        """
        records = []

        filepath = self.data_dir / "queries" / f"{query_name}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Run the query
            url = f"{API_BASE}/glycan/search"
            all_results = []
            offset = 0

            while True:
                payload = {
                    "query": query,
                    "pagination": {"offset": offset, "limit": 500},
                }

                response = self.http._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                all_results.extend(results)
                offset += len(results)

                if len(results) < 500:
                    break

            # Save results
            output = {
                "query": query,
                "query_name": query_name,
                "total_results": len(all_results),
                "results": all_results,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }

            sha256, size = save_json_response(output, filepath)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"query/{query_name}",
                    item_type=ItemType.API_RESPONSE,
                    url=url,
                    local_path=filepath,
                    sha256=sha256,
                    size=size,
                    extra={"query": query, "result_count": len(all_results)},
                )
            )

        except Exception as e:
            logger.error(f"Query {query_name} failed: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"query/{query_name}",
                    item_type=ItemType.API_RESPONSE,
                    url=f"{API_BASE}/glycan/search",
                    local_path=filepath,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def download_api_snapshot(self) -> list[DownloadRecord]:
        """Download a snapshot of key GlyGen API data.

        Creates a comprehensive snapshot including:
        - Glycan list with basic info
        - Motif list
        - Species summary

        Returns:
            List of download records.
        """
        records = []

        snapshot_dir = self.data_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")

        # Download glycan list
        try:
            glycan_data = self._fetch_all_pages(ENDPOINTS["glycan_list"])
            filepath = snapshot_dir / f"glycan_list_{timestamp}.json"
            sha256, size = save_json_response(glycan_data, filepath)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"snapshot/glycan_list_{timestamp}",
                    item_type=ItemType.BULK_EXPORT,
                    url=ENDPOINTS["glycan_list"],
                    local_path=filepath,
                    sha256=sha256,
                    size=size,
                )
            )
            logger.info(f"Downloaded glycan list: {len(glycan_data.get('results', []))} entries")
        except Exception as e:
            logger.error(f"Failed to download glycan list: {e}")

        # Log the API query for reproducibility
        query_log_path = snapshot_dir / f"api_log_{timestamp}.json"
        query_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoints_queried": list(ENDPOINTS.keys()),
            "base_url": API_BASE,
        }
        save_json_response(query_log, query_log_path)

        return records
