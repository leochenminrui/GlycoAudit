"""GlyTouCan source connector.

GlyTouCan is the international glycan structure repository.
Official API: https://api.glytoucan.org/
SPARQL endpoint: https://ts.glytoucan.org/sparql

This connector uses the SPARQL endpoint for bulk queries and the REST API
for individual record retrieval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from SPARQLWrapper import JSON, SPARQLWrapper

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

logger = get_logger("glytoucan")


# SPARQL endpoint
SPARQL_ENDPOINT = "https://ts.glytoucan.org/sparql"

# REST API base URL
API_BASE = "https://api.glytoucan.org"

# SPARQL query to get all glycan entries with their sequences
BULK_QUERY = """
PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>

SELECT DISTINCT ?accession ?wurcs ?glycoct
WHERE {
    ?saccharide glytoucan:has_primary_id ?accession .

    OPTIONAL {
        ?saccharide glycan:has_glycosequence ?wurcs_seq .
        ?wurcs_seq glycan:has_sequence ?wurcs .
        ?wurcs_seq glycan:in_carbohydrate_format glycan:carbohydrate_format_wurcs .
    }

    OPTIONAL {
        ?saccharide glycan:has_glycosequence ?glycoct_seq .
        ?glycoct_seq glycan:has_sequence ?glycoct .
        ?glycoct_seq glycan:in_carbohydrate_format glycan:carbohydrate_format_glycoct .
    }
}
ORDER BY ?accession
"""

# Query to get count
COUNT_QUERY = """
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>

SELECT (COUNT(DISTINCT ?accession) as ?count)
WHERE {
    ?saccharide glytoucan:has_primary_id ?accession .
}
"""


class GlyTouCanSource(BaseSource):
    """Source connector for GlyTouCan glycan repository."""

    SOURCE_NAME = "glytoucan"
    CATEGORY = "A"  # Glycan ID/sequence registry

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        super().__init__(config, http_client)
        self.sparql = SPARQLWrapper(SPARQL_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", config.user_agent)

    def _run_sparql_query(self, query: str) -> dict[str, Any]:
        """Run a SPARQL query and return results.

        Args:
            query: SPARQL query string.

        Returns:
            Query results as dictionary.
        """
        self.sparql.setQuery(query)
        return self.sparql.query().convert()

    def get_total_count(self) -> int:
        """Get total number of glycan entries."""
        try:
            results = self._run_sparql_query(COUNT_QUERY)
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                return int(bindings[0]["count"]["value"])
        except Exception as e:
            logger.warning(f"Failed to get count: {e}")
        return 0

    def list_items(self) -> Iterator[SourceItem]:
        """List all glycan entries from GlyTouCan.

        Uses SPARQL bulk query to fetch all entries efficiently.
        In bulk mode, yields a single item that triggers bulk export download.
        """
        if not self.is_enabled:
            logger.info("GlyTouCan source is disabled")
            return

        mode = self.source_config.mode
        logger.info(f"Listing GlyTouCan items (mode: {mode})")

        if mode in ("bulk", "bulk_or_api"):
            # In bulk mode, yield single item representing bulk export
            # The actual SPARQL query happens in download_item
            yield SourceItem(
                source=self.SOURCE_NAME,
                item_id="bulk_export",
                url=SPARQL_ENDPOINT,
                item_type="bulk_export",
                metadata={"mode": "bulk"},
            )
        else:
            # Use paginated API (fallback)
            yield from self._list_items_api()

    def _list_items_bulk(self) -> Iterator[SourceItem]:
        """List items using bulk SPARQL query."""
        logger.info("Fetching all entries via SPARQL bulk query...")

        try:
            results = self._run_sparql_query(BULK_QUERY)
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return

        bindings = results.get("results", {}).get("bindings", [])
        logger.info(f"Found {len(bindings)} entries")

        for binding in bindings:
            accession = binding.get("accession", {}).get("value", "")
            if not accession:
                continue

            wurcs = binding.get("wurcs", {}).get("value", "")
            glycoct = binding.get("glycoct", {}).get("value", "")

            yield SourceItem(
                source=self.SOURCE_NAME,
                item_id=accession,
                url=f"{API_BASE}/glycan/{accession}",
                item_type="registry_record",
                metadata={
                    "wurcs": wurcs,
                    "glycoct": glycoct,
                },
            )

    def _list_items_api(self) -> Iterator[SourceItem]:
        """List items using REST API (paginated)."""
        logger.info("Fetching entries via REST API...")

        # The GlyTouCan API provides a list endpoint
        # This is a fallback if SPARQL is unavailable
        page = 1
        per_page = 100

        while True:
            try:
                url = f"{API_BASE}/glycan/list?page={page}&per_page={per_page}"
                response = self.http.get_json(url)

                glycans = response.get("data", [])
                if not glycans:
                    break

                for glycan in glycans:
                    accession = glycan.get("accession", "")
                    if not accession:
                        continue

                    yield SourceItem(
                        source=self.SOURCE_NAME,
                        item_id=accession,
                        url=f"{API_BASE}/glycan/{accession}",
                        item_type="registry_record",
                        metadata=glycan,
                    )

                if len(glycans) < per_page:
                    break

                page += 1

            except Exception as e:
                logger.error(f"API request failed at page {page}: {e}")
                break

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download a glycan record.

        For bulk_export items, downloads the complete SPARQL export.
        For individual items, fetches from REST API.

        Args:
            item: Source item to download.

        Returns:
            List of download records.
        """
        # Handle bulk export mode
        if item.item_id == "bulk_export":
            return self.download_bulk_export()

        records = []
        accession = item.item_id

        # Create directory for this entry
        entry_dir = self.data_dir / accession
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        record_path = entry_dir / "record.json"
        if record_path.exists():
            logger.debug(f"Already downloaded: {accession}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.REGISTRY_RECORD,
                    url=item.url,
                    local_path=record_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        try:
            # Fetch full record from API
            url = f"{API_BASE}/glycan/{accession}"
            response = self.http.get_json(url)

            # Save raw API response
            api_path = entry_dir / "api_response.json"
            sha256, size = save_json_response(response, api_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"{accession}/api_response",
                    item_type=ItemType.API_RESPONSE,
                    url=url,
                    local_path=api_path,
                    sha256=sha256,
                    size=size,
                )
            )

            # Create normalized record
            normalized = self._normalize_record(accession, response, item.metadata)
            sha256, size = save_json_response(normalized, record_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.REGISTRY_RECORD,
                    url=url,
                    local_path=record_path,
                    sha256=sha256,
                    size=size,
                    extra=normalized,
                )
            )

        except Exception as e:
            logger.error(f"Failed to download {accession}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.REGISTRY_RECORD,
                    url=item.url,
                    local_path=record_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _normalize_record(
        self,
        accession: str,
        api_response: dict[str, Any],
        sparql_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize a glycan record to standard format.

        Args:
            accession: GlyTouCan accession.
            api_response: Raw API response.
            sparql_metadata: Metadata from SPARQL query.

        Returns:
            Normalized record dictionary.
        """
        # Extract sequences from various sources
        wurcs = sparql_metadata.get("wurcs", "")
        glycoct = sparql_metadata.get("glycoct", "")

        # Try to get from API response if not in SPARQL
        if not wurcs and "wurcs" in api_response:
            wurcs = api_response["wurcs"]
        if not glycoct and "glycoct" in api_response:
            glycoct = api_response["glycoct"]

        # Extract synonyms
        synonyms = api_response.get("synonyms", [])
        if isinstance(synonyms, str):
            synonyms = [synonyms]

        return {
            "id": accession,
            "wurcs": wurcs,
            "glycoct": glycoct,
            "synonyms": synonyms,
            "mass": api_response.get("mass"),
            "motifs": api_response.get("motifs", []),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_url": f"https://glytoucan.org/Structures/Glycans/{accession}",
        }

    def download_bulk_export(self) -> list[DownloadRecord]:
        """Download the complete SPARQL bulk export.

        This creates a single file with all glycan entries.
        """
        records = []

        bulk_path = self.data_dir / "bulk_export.json"
        logger.info("Downloading bulk SPARQL export...")

        try:
            results = self._run_sparql_query(BULK_QUERY)

            sha256, size = save_json_response(results, bulk_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="bulk_export",
                    item_type=ItemType.BULK_EXPORT,
                    url=SPARQL_ENDPOINT,
                    local_path=bulk_path,
                    sha256=sha256,
                    size=size,
                )
            )

            # Also create a normalized table
            bindings = results.get("results", {}).get("bindings", [])
            normalized_records = []

            for binding in bindings:
                accession = binding.get("accession", {}).get("value", "")
                if accession:
                    normalized_records.append({
                        "id": accession,
                        "wurcs": binding.get("wurcs", {}).get("value", ""),
                        "glycoct": binding.get("glycoct", {}).get("value", ""),
                    })

            table_path = self.data_dir / "registry_table.json"
            sha256, size = save_json_response(normalized_records, table_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="registry_table",
                    item_type=ItemType.METADATA,
                    url=SPARQL_ENDPOINT,
                    local_path=table_path,
                    sha256=sha256,
                    size=size,
                )
            )

            logger.info(f"Bulk export complete: {len(normalized_records)} entries")

        except Exception as e:
            logger.error(f"Bulk export failed: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="bulk_export",
                    item_type=ItemType.BULK_EXPORT,
                    url=SPARQL_ENDPOINT,
                    local_path=bulk_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records
