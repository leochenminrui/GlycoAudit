"""SugarBind RDF source connector via GlyCosmos SPARQL.

SugarBind is a database of pathogen-glycan binding interactions.
Data is accessed via the GlyCosmos SPARQL endpoint.

Endpoint: https://ts.glycosmos.org/sparql
Documentation: https://glycosmos.org/sparql

This connector queries SugarBind lectin-glycan binding data and extracts:
- GlyTouCan accession (glycan identifier)
- Protein/lectin identifier
- Pathogen information
- Evidence references
"""

from __future__ import annotations

import csv
import json
import time
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

logger = get_logger("sugarbind")


# GlyCosmos SPARQL endpoint
SPARQL_ENDPOINT = "https://ts.glycosmos.org/sparql"

# Rate limiting: 0.5 requests per second = 2 seconds between requests
REQUEST_DELAY = 2.0

# Pagination settings
PAGE_SIZE = 1000

# SPARQL query to get lectin-glycan binding data from GlycoEpitope via GlyCosmos
# Uses the GlyTouCan partner graph to link glycoepitopes to GlyTouCan IDs
GLYCOEPITOPE_BINDING_QUERY = """
PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ge: <http://www.glycoepitope.jp/epitopes/glycoepitope.owl#>
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>

SELECT DISTINCT ?glytoucan_id ?epitope_id ?epitope_label ?binding_type ?agent_label
WHERE {{
    # Get GlyTouCan linked to glycoepitope via partner graph
    GRAPH <http://rdf.glytoucan.org/partner/glycoepitope> {{
        ?saccharide glycan:has_resource_entry ?entry .
        ?entry dcterms:identifier ?epitope_id .
    }}

    # Get glytoucan ID
    GRAPH <http://rdf.glytoucan.org/core> {{
        ?saccharide glytoucan:has_primary_id ?glytoucan_id .
    }}

    # Build epitope URI from ID and get binding info
    BIND(IRI(CONCAT("http://www.glycoepitope.jp/epitopes/", ?epitope_id)) AS ?epitope_uri)

    GRAPH <http://rdf.glycoinfo.org/glycoepitope> {{
        ?epitope_uri rdfs:label ?epitope_label .
        {{
            ?epitope_uri ge:has_antibody ?agent .
            BIND("antibody" AS ?binding_type)
        }}
        UNION
        {{
            ?epitope_uri ge:has_affinity_to ?agent .
            BIND("receptor" AS ?binding_type)
        }}
        OPTIONAL {{ ?agent rdfs:label ?agent_label . }}
    }}
}}
ORDER BY ?glytoucan_id
LIMIT {limit}
OFFSET {offset}
"""

# Count query for binding entries
COUNT_QUERY = """
PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ge: <http://www.glycoepitope.jp/epitopes/glycoepitope.owl#>
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>

SELECT (COUNT(DISTINCT ?pair) as ?count)
WHERE {
    GRAPH <http://rdf.glytoucan.org/partner/glycoepitope> {
        ?saccharide glycan:has_resource_entry ?entry .
        ?entry dcterms:identifier ?epitope_id .
    }

    GRAPH <http://rdf.glytoucan.org/core> {
        ?saccharide glytoucan:has_primary_id ?glytoucan_id .
    }

    BIND(IRI(CONCAT("http://www.glycoepitope.jp/epitopes/", ?epitope_id)) AS ?epitope_uri)

    GRAPH <http://rdf.glycoinfo.org/glycoepitope> {
        {
            ?epitope_uri ge:has_antibody ?agent .
        }
        UNION
        {
            ?epitope_uri ge:has_affinity_to ?agent .
        }
    }

    BIND(CONCAT(?glytoucan_id, "_", STR(?agent)) AS ?pair)
}
"""


class SugarBindRDFSource(BaseSource):
    """Source connector for SugarBind via GlyCosmos SPARQL."""

    SOURCE_NAME = "sugarbind"
    CATEGORY = "C"  # Binding data

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        super().__init__(config, http_client)
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            sleep_time = REQUEST_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _run_sparql_query(
        self,
        query: str,
        accept: str = "application/sparql-results+json",
    ) -> dict[str, Any]:
        """Run a SPARQL query against GlyCosmos endpoint.

        Args:
            query: SPARQL query string.
            accept: Accept header for response format.

        Returns:
            Query results as dictionary.
        """
        self._rate_limit()

        headers = {
            "Accept": accept,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # Use POST for longer queries
        response = self.http.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            headers=headers,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"SPARQL query failed: HTTP {response.status_code}: {response.text[:500]}"
            )

        return response.json()

    def _run_sparql_query_csv(self, query: str) -> str:
        """Run a SPARQL query and get CSV response.

        Args:
            query: SPARQL query string.

        Returns:
            CSV content as string.
        """
        self._rate_limit()

        headers = {
            "Accept": "text/csv",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = self.http.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            headers=headers,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"SPARQL query failed: HTTP {response.status_code}: {response.text[:500]}"
            )

        return response.text

    def get_total_count(self) -> int:
        """Get total number of SugarBind binding entries."""
        try:
            results = self._run_sparql_query(COUNT_QUERY)
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                return int(bindings[0]["count"]["value"])
        except Exception as e:
            logger.warning(f"Failed to get count: {e}")
        return 0

    def list_items(self) -> Iterator[SourceItem]:
        """List SugarBind binding entries.

        In bulk mode, yields a single item that triggers bulk download.
        Otherwise yields individual binding entries.
        """
        if not self.is_enabled:
            logger.info("SugarBind source is disabled")
            return

        mode = self.source_config.mode if self.source_config else "bulk"
        logger.info(f"Listing SugarBind items (mode: {mode})")

        if mode in ("bulk", "bulk_or_api"):
            # In bulk mode, yield single item for bulk export
            yield SourceItem(
                source=self.SOURCE_NAME,
                item_id="bulk_export",
                url=SPARQL_ENDPOINT,
                item_type="bulk_export",
                metadata={"mode": "bulk"},
            )
        else:
            # Paginated mode - yield individual binding entries
            yield from self._list_items_paginated()

    def _list_items_paginated(self) -> Iterator[SourceItem]:
        """List items with SPARQL pagination."""
        logger.info("Fetching SugarBind entries via paginated SPARQL...")

        offset = 0
        total_items = 0

        while True:
            query = SUGARBIND_QUERY_V2.format(limit=PAGE_SIZE, offset=offset)

            try:
                results = self._run_sparql_query(query)
            except Exception as e:
                logger.error(f"SPARQL query failed at offset {offset}: {e}")
                break

            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                logger.info(f"No more results at offset {offset}")
                break

            logger.info(f"Processing {len(bindings)} bindings at offset {offset}")

            for binding in bindings:
                glytoucan_id = binding.get("glytoucan_id", {}).get("value", "")
                if not glytoucan_id:
                    continue

                # Create unique item ID from binding info
                agent_name = binding.get("agent_name", {}).get("value", "unknown")
                item_id = f"{glytoucan_id}_{agent_name}".replace(" ", "_")[:100]

                metadata = {
                    "glytoucan_id": glytoucan_id,
                    "agent_name": agent_name,
                    "agent_type": binding.get("agent_type", {}).get("value", ""),
                    "interaction_type": binding.get("interaction_type", {}).get("value", ""),
                    "reference": binding.get("reference", {}).get("value", ""),
                }

                yield SourceItem(
                    source=self.SOURCE_NAME,
                    item_id=item_id,
                    url=f"{SPARQL_ENDPOINT}#binding/{item_id}",
                    item_type="binding_record",
                    metadata=metadata,
                )
                total_items += 1

            if len(bindings) < PAGE_SIZE:
                break

            offset += PAGE_SIZE

        logger.info(f"Total items listed: {total_items}")

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download a SugarBind binding record.

        For bulk_export items, downloads the complete SPARQL export.
        For individual items, saves the binding metadata.

        Args:
            item: Source item to download.

        Returns:
            List of download records.
        """
        if item.item_id == "bulk_export":
            return self.download_bulk_export()

        records = []
        item_id = item.item_id

        # Create directory for binding records
        binding_dir = self.data_dir / "bindings"
        binding_dir.mkdir(parents=True, exist_ok=True)

        # Save individual binding record
        record_path = binding_dir / f"{item_id}.json"

        if record_path.exists():
            logger.debug(f"Already downloaded: {item_id}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=item_id,
                    item_type=ItemType.BINDING_RECORD,
                    url=item.url,
                    local_path=record_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        try:
            record = {
                "id": item_id,
                **item.metadata,
                "source": "sugarbind",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }

            sha256, size = save_json_response(record, record_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=item_id,
                    item_type=ItemType.BINDING_RECORD,
                    url=item.url,
                    local_path=record_path,
                    sha256=sha256,
                    size=size,
                )
            )

        except Exception as e:
            logger.error(f"Failed to save {item_id}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=item_id,
                    item_type=ItemType.BINDING_RECORD,
                    url=item.url,
                    local_path=record_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def download_bulk_export(self) -> list[DownloadRecord]:
        """Download the complete GlycoEpitope binding dataset via SPARQL.

        This creates consolidated output files:
        - bulk_export.json: Raw SPARQL results
        - binding_table.csv: Normalized binding table
        """
        records = []

        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading GlycoEpitope binding data via SPARQL...")

        all_bindings = []
        offset = 0
        page_count = 0

        while True:
            query = GLYCOEPITOPE_BINDING_QUERY.format(limit=PAGE_SIZE, offset=offset)
            page_count += 1

            logger.info(f"Fetching page {page_count} (offset {offset})...")

            try:
                results = self._run_sparql_query(query)
            except Exception as e:
                logger.error(f"SPARQL query failed at offset {offset}: {e}")
                break

            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                logger.info(f"No more results at offset {offset}")
                break

            all_bindings.extend(bindings)
            logger.info(f"Fetched {len(bindings)} bindings (total: {len(all_bindings)})")

            if len(bindings) < PAGE_SIZE:
                break

            offset += PAGE_SIZE

        if not all_bindings:
            logger.error("No GlycoEpitope binding data found")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="bulk_export",
                    item_type=ItemType.BULK_EXPORT,
                    url=SPARQL_ENDPOINT,
                    local_path=self.data_dir / "bulk_export.json",
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error="No data retrieved",
                )
            ]

        # Save raw SPARQL results
        bulk_path = self.data_dir / "bulk_export.json"
        bulk_data = {
            "head": {"vars": list(all_bindings[0].keys()) if all_bindings else []},
            "results": {"bindings": all_bindings},
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source": "glycosmos_sugarbind",
            "endpoint": SPARQL_ENDPOINT,
        }

        try:
            sha256, size = save_json_response(bulk_data, bulk_path)
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
            logger.info(f"Saved bulk export: {bulk_path}")

        except Exception as e:
            logger.error(f"Failed to save bulk export: {e}")
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

        # Create normalized binding table (CSV)
        table_path = self.data_dir / "binding_table.csv"
        try:
            normalized = self._normalize_bindings(all_bindings)
            self._save_binding_table(normalized, table_path)

            sha256, size = self._compute_file_hash(table_path)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id="binding_table",
                    item_type=ItemType.BINDING_RECORD,
                    url=SPARQL_ENDPOINT,
                    local_path=table_path,
                    sha256=sha256,
                    size=size,
                    extra={"rows": len(normalized)},
                )
            )
            logger.info(f"Saved binding table: {table_path} ({len(normalized)} rows)")

        except Exception as e:
            logger.error(f"Failed to save binding table: {e}")

        # Summary
        unique_glycans = len(set(
            b.get("glytoucan_id", {}).get("value", "")
            for b in all_bindings
            if b.get("glytoucan_id", {}).get("value", "")
        ))
        logger.info(
            f"Bulk export complete: {len(all_bindings)} bindings, "
            f"{unique_glycans} unique glycans"
        )

        return records

    def _try_alternative_queries(self) -> list[dict]:
        """Try alternative SPARQL query structures."""
        # Try simpler query to explore what's available
        explore_query = """
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?graph
        WHERE {
            GRAPH ?graph {
                ?s glycan:has_glycan ?o .
            }
        }
        LIMIT 20
        """

        try:
            logger.info("Exploring available graphs...")
            results = self._run_sparql_query(explore_query)
            bindings = results.get("results", {}).get("bindings", [])
            graphs = [b.get("graph", {}).get("value", "") for b in bindings]
            logger.info(f"Found graphs: {graphs}")
        except Exception as e:
            logger.warning(f"Graph exploration failed: {e}")

        # Try direct SugarBind endpoint exploration
        sugarbind_explore = """
        PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>

        SELECT ?s ?p ?glytoucan_id
        WHERE {
            ?s glycan:has_glycan ?glycan_uri .
            ?glycan_uri glytoucan:has_primary_id ?glytoucan_id .
            ?s ?p ?o .
        }
        LIMIT 100
        """

        try:
            logger.info("Exploring SugarBind structure...")
            results = self._run_sparql_query(sugarbind_explore)
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                logger.info(f"Found {len(bindings)} binding-related triples")
                return bindings
        except Exception as e:
            logger.warning(f"SugarBind exploration failed: {e}")

        return []

    def _normalize_bindings(self, bindings: list[dict]) -> list[dict]:
        """Normalize SPARQL bindings to flat records.

        Args:
            bindings: Raw SPARQL result bindings.

        Returns:
            List of normalized binding records.
        """
        normalized = []

        for binding in bindings:
            record = {}

            # Extract values from SPARQL binding format
            for key, value_obj in binding.items():
                if isinstance(value_obj, dict) and "value" in value_obj:
                    record[key] = value_obj["value"]
                else:
                    record[key] = str(value_obj) if value_obj else ""

            if record.get("glytoucan_id"):
                normalized.append(record)

        return normalized

    def _save_binding_table(self, records: list[dict], path: Path) -> None:
        """Save binding records as CSV.

        Args:
            records: List of binding records.
            path: Output path.
        """
        if not records:
            return

        # Determine all columns
        all_keys = set()
        for record in records:
            all_keys.update(record.keys())

        # Order columns with important ones first
        priority_cols = ["glytoucan_id", "agent_name", "agent_type", "interaction_type", "reference"]
        columns = [c for c in priority_cols if c in all_keys]
        columns.extend(sorted(c for c in all_keys if c not in priority_cols))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    def _compute_file_hash(self, path: Path) -> tuple[str, int]:
        """Compute SHA256 hash and size of a file.

        Args:
            path: File path.

        Returns:
            Tuple of (sha256_hex, size_bytes).
        """
        import hashlib

        sha256 = hashlib.sha256()
        size = 0

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
                size += len(chunk)

        return sha256.hexdigest(), size
