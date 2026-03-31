"""GLYCAM-Web source connector.

GLYCAM-Web is a web-based tool for generating glycan 3D structures.
Website: https://glycam.org/
API documentation: https://glycam.org/json-api/

IMPORTANT: GLYCAM-Web primarily operates as an interactive tool that generates
structures on-demand. Full programmatic access may require authentication or
session management. This connector implements:

1. "api" mode: Use the JSON API for available endpoints
2. "manual" mode: User provides a CSV manifest of URLs to download
3. "stub" mode: Skip with instructions (default if auth needed)

Environment variables:
- GLYCAM_AUTH_TOKEN: Authentication token if required
- GLYCAM_SESSION_COOKIE: Session cookie if using manual auth
"""

from __future__ import annotations

import csv
import os
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

logger = get_logger("glycam_web")


# API endpoints
API_BASE = "https://glycam.org"
JSON_API = f"{API_BASE}/json-api"

# Known public endpoints that don't require auth
PUBLIC_ENDPOINTS = [
    "/json-api/glycan/sequence/",  # Convert sequence to structure
]

AUTH_INSTRUCTIONS = """
================================================================================
GLYCAM-Web Authentication Required
================================================================================

GLYCAM-Web's structure generation API may require authentication or session
management for certain operations. To use this connector:

Option 1: Use the JSON API (if available for your use case)
-----------------------------------------------------------
Some endpoints may work without authentication. The connector will attempt
these first.

Option 2: Manual Manifest Mode
------------------------------
1. Create a CSV file with the following columns:
   - url: The full URL to download
   - local_filename: Desired local filename
   - glycan_id: Optional glycan identifier

2. Save it as: data/raw/glycam_web/manifest.csv

3. Set in your config:
   sources:
     glycam_web:
       enabled: true
       mode: "manifest"

Option 3: Export Session Cookies
--------------------------------
1. Log into GLYCAM-Web in your browser
2. Export cookies (e.g., using a browser extension)
3. Set environment variable:
   export GLYCAM_SESSION_COOKIE="your_session_cookie_here"

4. Re-run the mirror

Option 4: API Token (if supported)
----------------------------------
If GLYCAM-Web provides API tokens:
1. Obtain a token from your account settings
2. Set environment variable:
   export GLYCAM_AUTH_TOKEN="your_token_here"

3. Re-run the mirror

For questions, consult: https://glycam.org/docs/

================================================================================
"""


class GlycamWebSource(BaseSource):
    """Source connector for GLYCAM-Web structure generation service."""

    SOURCE_NAME = "glycam_web"
    CATEGORY = "B"  # Glycan 3D structures

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        super().__init__(config, http_client)
        self.auth_token = config.glycam_auth_token
        self._session_cookie = os.environ.get("GLYCAM_SESSION_COOKIE")

    def requires_auth(self) -> bool:
        """Check if authentication is required."""
        mode = self.source_config.mode
        if mode == "manifest":
            return False  # User provides URLs directly
        if mode == "api":
            # Try to detect if auth is needed
            return self.auth_token is None and self._session_cookie is None
        return True

    def auth_instructions(self) -> str:
        """Get authentication instructions."""
        return AUTH_INSTRUCTIONS

    def list_items(self) -> Iterator[SourceItem]:
        """List items for download.

        Depending on mode:
        - "api": Try to list from API
        - "manifest": Read from user-provided manifest
        - "stub": Skip with instructions
        """
        if not self.is_enabled:
            logger.info("GLYCAM-Web source is disabled")
            return

        mode = self.source_config.mode

        if mode == "manifest":
            yield from self._list_items_manifest()
        elif mode == "api":
            yield from self._list_items_api()
        elif mode in ("stub", "manual_or_api"):
            # Check if auth is available
            if self.requires_auth():
                logger.warning("GLYCAM-Web requires authentication")
                self._print_auth_instructions()
                return
            yield from self._list_items_api()
        else:
            logger.warning(f"Unknown mode: {mode}")
            self._print_auth_instructions()

    def _print_auth_instructions(self) -> None:
        """Print authentication instructions to console."""
        print(AUTH_INSTRUCTIONS)
        logger.info("Skipping GLYCAM-Web (auth required)")

        # Save instructions to file
        instructions_path = self.data_dir / "AUTH_INSTRUCTIONS.txt"
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(instructions_path, "w") as f:
            f.write(AUTH_INSTRUCTIONS)

    def _list_items_manifest(self) -> Iterator[SourceItem]:
        """List items from user-provided manifest."""
        manifest_path = self.data_dir / "manifest.csv"

        if not manifest_path.exists():
            logger.error(f"Manifest file not found: {manifest_path}")
            logger.info("Create a manifest.csv with columns: url, local_filename, glycan_id")
            return

        logger.info(f"Reading manifest from {manifest_path}")

        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "")
                if not url:
                    continue

                glycan_id = row.get("glycan_id", "") or row.get("id", "") or url.split("/")[-1]
                local_filename = row.get("local_filename", f"{glycan_id}.pdb")

                yield SourceItem(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    url=url,
                    item_type="structure",
                    metadata={
                        "local_filename": local_filename,
                        "source": "manifest",
                    },
                )

    def _list_items_api(self) -> Iterator[SourceItem]:
        """Try to list items from GLYCAM-Web API."""
        logger.info("Attempting to access GLYCAM-Web API...")

        # GLYCAM-Web doesn't have a simple list endpoint
        # The API is primarily for on-demand structure generation
        # We'll try to access any available catalog endpoints

        catalog_endpoints = [
            f"{JSON_API}/structures",
            f"{JSON_API}/glycans",
            f"{JSON_API}/catalog",
            f"{API_BASE}/api/v1/structures",
        ]

        headers = self._get_auth_headers()

        for endpoint in catalog_endpoints:
            try:
                if headers:
                    response = self.http.get_json(endpoint, headers=headers)
                else:
                    response = self.http.get_json(endpoint)

                # If we get here, the endpoint works
                items = response if isinstance(response, list) else response.get("data", [])

                for item in items:
                    if isinstance(item, dict):
                        item_id = item.get("id", item.get("name", ""))
                        url = item.get("url", f"{endpoint}/{item_id}")
                    else:
                        item_id = str(item)
                        url = f"{endpoint}/{item_id}"

                    if item_id:
                        yield SourceItem(
                            source=self.SOURCE_NAME,
                            item_id=item_id,
                            url=url,
                            item_type="structure",
                            metadata=item if isinstance(item, dict) else {},
                        )

                logger.info(f"Found items at {endpoint}")
                return

            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue

        # No catalog endpoint worked
        logger.warning("No accessible catalog endpoints found")
        self._print_auth_instructions()

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        headers = {}

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self._session_cookie:
            headers["Cookie"] = self._session_cookie

        return headers

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download a structure from GLYCAM-Web.

        Args:
            item: Source item to download.

        Returns:
            List of download records.
        """
        records = []
        glycan_id = item.item_id

        # Create directory
        glycan_dir = self.data_dir / glycan_id
        glycan_dir.mkdir(parents=True, exist_ok=True)

        local_filename = item.metadata.get("local_filename", f"{glycan_id}.pdb")
        filepath = glycan_dir / local_filename

        # Check if already downloaded
        if filepath.exists():
            logger.debug(f"Already downloaded: {glycan_id}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.STRUCTURE,
                    url=item.url,
                    local_path=filepath,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        try:
            headers = self._get_auth_headers()

            if local_filename.endswith(".pdb"):
                if headers:
                    content = self.http.get_text(item.url, headers=headers)
                else:
                    content = self.http.get_text(item.url)

                sha256, size = save_text_response(content, filepath)
            else:
                if headers:
                    content = self.http.get_json(item.url, headers=headers)
                else:
                    content = self.http.get_json(item.url)

                sha256, size = save_json_response(content, filepath)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.STRUCTURE,
                    url=item.url,
                    local_path=filepath,
                    sha256=sha256,
                    size=size,
                )
            )

        except Exception as e:
            logger.error(f"Failed to download {glycan_id}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=glycan_id,
                    item_type=ItemType.STRUCTURE,
                    url=item.url,
                    local_path=filepath,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def generate_structure(
        self,
        sequence: str,
        output_format: str = "pdb",
    ) -> SourceItem | None:
        """Request structure generation from GLYCAM-Web API.

        This uses the JSON API to generate a structure from a sequence.

        Args:
            sequence: Glycan sequence (e.g., in GLYCAM condensed notation)
            output_format: Output format (pdb, mol2, etc.)

        Returns:
            SourceItem for the generated structure, or None if failed.
        """
        logger.info(f"Requesting structure generation for: {sequence[:50]}...")

        try:
            # The GLYCAM JSON API endpoint for structure generation
            url = f"{JSON_API}/glycan/sequence/{sequence}"

            headers = self._get_auth_headers()
            if headers:
                response = self.http.get_json(url, headers=headers)
            else:
                response = self.http.get_json(url)

            # Extract structure URL from response
            structure_url = response.get("pdb_url") or response.get("structure_url")

            if structure_url:
                return SourceItem(
                    source=self.SOURCE_NAME,
                    item_id=sequence.replace("/", "_")[:50],  # Sanitize for filename
                    url=structure_url,
                    item_type="structure",
                    metadata={
                        "sequence": sequence,
                        "format": output_format,
                        "api_response": response,
                    },
                )

        except Exception as e:
            logger.error(f"Structure generation failed: {e}")

        return None

    def create_manifest_template(self) -> Path:
        """Create a template manifest file for manual downloads.

        Returns:
            Path to the created template.
        """
        manifest_path = self.data_dir / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "local_filename", "glycan_id"])
            writer.writeheader()
            # Write example rows
            writer.writerow({
                "url": "https://glycam.org/example/structure1.pdb",
                "local_filename": "example_structure.pdb",
                "glycan_id": "example_001",
            })

        logger.info(f"Created manifest template: {manifest_path}")
        return manifest_path
