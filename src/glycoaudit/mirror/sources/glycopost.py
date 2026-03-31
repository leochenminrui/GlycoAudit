"""GlycoPOST source connector.

GlycoPOST is a repository for glycomics mass spectrometry data.
Website: https://glycopost.glycosmos.org/
API documentation: Limited; this connector uses observed endpoints.

This connector implements project-level mirroring:
1. User provides project accessions in config
2. Downloads project metadata
3. Optionally downloads raw files based on size policy

Note: The GlycoPOST API is not fully documented. This connector stores
HTML/JSON snapshots for debugging and may require updates if the site changes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup

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

logger = get_logger("glycopost")


# Website/API URLs
GLYCOPOST_BASE = "https://glycopost.glycosmos.org"
PROJECT_URL = f"{GLYCOPOST_BASE}/preview"  # + /{accession}
API_BASE = f"{GLYCOPOST_BASE}/api"

# Known API endpoints (discovered through observation)
ENDPOINTS = {
    "project_list": f"{API_BASE}/projects",
    "project_detail": f"{API_BASE}/project",  # + /{accession}
    "file_list": f"{API_BASE}/files",  # + /{accession}
}


class GlycoPostSource(BaseSource):
    """Source connector for GlycoPOST MS data repository.

    This is a PARTIAL STUB connector. The GlycoPOST API is not fully documented,
    so this connector attempts multiple strategies:
    1. Try the REST API if available
    2. Fall back to HTML scraping
    3. If all fails, save error details and instructions

    The connector never fails silently - it always logs clear errors.
    """

    SOURCE_NAME = "glycopost"
    CATEGORY = "D"  # MS datasets

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        super().__init__(config, http_client)
        self._project_accessions = self.source_config.project_accessions
        self._download_policy = self.source_config.ms_download_policy
        self._size_limit_mb = self.source_config.ms_size_limit_mb
        self._confirm_large = self.source_config.confirm_large_download
        self._snapshots_dir = self.data_dir / "_snapshots"
        self._api_available = None  # Will be determined on first request

    def list_items(self) -> Iterator[SourceItem]:
        """List projects for download.

        Uses project accessions from config, or fetches available projects.
        """
        if not self.is_enabled:
            logger.info("GlycoPOST source is disabled")
            return

        logger.info("Preparing GlycoPOST project list...")

        # If specific accessions provided, use those
        if self._project_accessions:
            logger.info(f"Using {len(self._project_accessions)} configured project accessions")
            for accession in self._project_accessions:
                yield SourceItem(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    url=f"{PROJECT_URL}/{accession}",
                    item_type="ms_project",
                    metadata={"source": "config"},
                )
            return

        # Otherwise, try to list available projects
        yield from self._list_available_projects()

    def _list_available_projects(self) -> Iterator[SourceItem]:
        """Try to list available projects from GlycoPOST.

        NOTE: This is a partial stub. GlycoPOST API is not fully documented.
        If this fails, users should provide project_accessions in config.
        """
        logger.info("Attempting to fetch GlycoPOST project list...")
        logger.warning("NOTE: GlycoPOST API is not fully documented. "
                      "Consider using project_accessions in config for reliability.")

        # Ensure snapshots directory exists
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Try API endpoint first
        api_error = None
        try:
            logger.debug(f"Trying API: {ENDPOINTS['project_list']}")
            response = self.http.get_json(ENDPOINTS["project_list"])
            self._api_available = True
            projects = response if isinstance(response, list) else response.get("data", [])

            logger.info(f"API returned {len(projects)} projects")
            for proj in projects:
                accession = proj.get("accession", proj.get("id", ""))
                if accession:
                    yield SourceItem(
                        source=self.SOURCE_NAME,
                        item_id=accession,
                        url=f"{PROJECT_URL}/{accession}",
                        item_type="ms_project",
                        metadata=proj,
                    )
            return

        except Exception as e:
            api_error = str(e)
            self._api_available = False
            logger.warning(f"GlycoPOST API endpoint failed: {e}")
            logger.info("Falling back to HTML scraping...")

        # Fallback: try to scrape project list from website
        scrape_error = None
        try:
            html = self.http.get_text(GLYCOPOST_BASE)

            # Save HTML snapshot for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = self._snapshots_dir / f"homepage_{timestamp}.html"
            save_text_response(html, snapshot_path)
            logger.debug(f"Saved homepage snapshot: {snapshot_path}")

            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(html, "html.parser")

            # Look for project links
            found_projects = []
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                match = re.search(r"/preview/(GPST\d+)", href)
                if match:
                    accession = match.group(1)
                    if accession not in found_projects:
                        found_projects.append(accession)
                        yield SourceItem(
                            source=self.SOURCE_NAME,
                            item_id=accession,
                            url=f"{PROJECT_URL}/{accession}",
                            item_type="ms_project",
                            metadata={"source": "scraped"},
                        )

            if found_projects:
                logger.info(f"Found {len(found_projects)} projects via scraping")
                return
            else:
                scrape_error = "No project links found in page"

        except Exception as e:
            scrape_error = str(e)
            logger.error(f"HTML scraping also failed: {e}")

        # Both methods failed - save detailed error info
        logger.error("="*60)
        logger.error("GLYCOPOST CONNECTOR FAILED")
        logger.error("="*60)
        logger.error(f"API error: {api_error}")
        logger.error(f"Scrape error: {scrape_error}")
        logger.error("Please provide project_accessions in config")
        logger.error("="*60)

        self._save_instructions(api_error=api_error, scrape_error=scrape_error)

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download project metadata and optionally files.

        Args:
            item: Source item (project) to download.

        Returns:
            List of download records.
        """
        records = []
        accession = item.item_id

        # Create project directory
        project_dir = self.data_dir / accession
        project_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        metadata_path = project_dir / "metadata.json"
        if metadata_path.exists():
            logger.debug(f"Already downloaded: {accession}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.MS_PROJECT,
                    url=item.url,
                    local_path=metadata_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        # Download project metadata
        metadata_records = self._download_project_metadata(accession, project_dir, item.url)
        records.extend(metadata_records)

        # Optionally download files based on policy
        if self._download_policy != "metadata_only":
            file_records = self._download_project_files(accession, project_dir)
            records.extend(file_records)

        return records

    def _download_project_metadata(
        self,
        accession: str,
        project_dir: Path,
        project_url: str,
    ) -> list[DownloadRecord]:
        """Download project metadata."""
        records = []

        try:
            # Try API first
            metadata = self._fetch_project_api(accession)

            if not metadata:
                # Fallback to scraping
                metadata = self._scrape_project_page(accession, project_url, project_dir)

            if metadata:
                metadata_path = project_dir / "metadata.json"
                sha256, size = save_json_response(metadata, metadata_path)

                records.append(
                    make_download_record(
                        source=self.SOURCE_NAME,
                        item_id=accession,
                        item_type=ItemType.MS_PROJECT,
                        url=project_url,
                        local_path=metadata_path,
                        sha256=sha256,
                        size=size,
                        extra=self._extract_project_summary(metadata),
                    )
                )
            else:
                raise ValueError("Could not retrieve project metadata")

        except Exception as e:
            logger.error(f"Failed to download metadata for {accession}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.MS_PROJECT,
                    url=project_url,
                    local_path=project_dir / "metadata.json",
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _fetch_project_api(self, accession: str) -> dict[str, Any] | None:
        """Fetch project metadata from API."""
        try:
            url = f"{ENDPOINTS['project_detail']}/{accession}"
            response = self.http.get_json(url)
            return response
        except Exception as e:
            logger.debug(f"API fetch failed for {accession}: {e}")
            return None

    def _scrape_project_page(
        self,
        accession: str,
        url: str,
        project_dir: Path,
    ) -> dict[str, Any] | None:
        """Scrape project metadata from HTML page."""
        try:
            html = self.http.get_text(url)

            # Save HTML snapshot
            snapshot_path = project_dir / "page.html"
            save_text_response(html, snapshot_path)

            soup = BeautifulSoup(html, "lxml")

            # Extract metadata from page
            metadata = {
                "accession": accession,
                "url": url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }

            # Try to find title
            title_elem = soup.find("h1") or soup.find("title")
            if title_elem:
                metadata["title"] = title_elem.get_text(strip=True)

            # Look for description
            desc_elem = soup.find("div", class_="description") or soup.find(
                "p", class_="description"
            )
            if desc_elem:
                metadata["description"] = desc_elem.get_text(strip=True)

            # Look for file listing
            files = []
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if any(ext in href.lower() for ext in [".raw", ".mzml", ".mzxml", ".mgf"]):
                    files.append({
                        "name": link.get_text(strip=True),
                        "url": href,
                    })

            if files:
                metadata["files"] = files
                metadata["file_count"] = len(files)

            return metadata

        except Exception as e:
            logger.error(f"Failed to scrape project page: {e}")
            return None

    def _download_project_files(
        self,
        accession: str,
        project_dir: Path,
    ) -> list[DownloadRecord]:
        """Download project data files."""
        records = []

        # Get file list
        file_list = self._get_project_files(accession)

        if not file_list:
            logger.warning(f"No files found for project {accession}")
            return records

        # Calculate total size
        total_size_mb = sum(f.get("size_mb", 0) for f in file_list)

        # Check download policy
        if self._download_policy == "all":
            if not self._confirm_large and total_size_mb > self._size_limit_mb:
                logger.warning(
                    f"Project {accession} has {total_size_mb:.1f} MB of files. "
                    f"Set confirm_large_download: true to download all files."
                )
                return records
        elif self._download_policy == "small_files_only":
            file_list = [f for f in file_list if f.get("size_mb", 0) <= self._size_limit_mb]

        # Download files
        files_dir = project_dir / "files"
        files_dir.mkdir(exist_ok=True)

        for file_info in file_list:
            url = file_info.get("url", "")
            filename = file_info.get("name", url.split("/")[-1])

            if not url:
                continue

            filepath = files_dir / filename

            if filepath.exists():
                logger.debug(f"File already exists: {filename}")
                continue

            try:
                success, record = self.http.download_file(
                    url=url,
                    dest_path=filepath,
                    source=self.SOURCE_NAME,
                    item_id=f"{accession}/{filename}",
                )
                record.item_type = ItemType.MS_FILE.value
                records.append(record)

            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                records.append(
                    make_download_record(
                        source=self.SOURCE_NAME,
                        item_id=f"{accession}/{filename}",
                        item_type=ItemType.MS_FILE,
                        url=url,
                        local_path=filepath,
                        sha256="",
                        size=0,
                        status=DownloadStatus.FAILED,
                        error=str(e),
                    )
                )

        return records

    def _get_project_files(self, accession: str) -> list[dict[str, Any]]:
        """Get list of files for a project."""
        # Try API
        try:
            url = f"{ENDPOINTS['file_list']}/{accession}"
            response = self.http.get_json(url)
            files = response if isinstance(response, list) else response.get("files", [])
            return files
        except Exception:
            pass

        # Try reading from cached metadata
        metadata_path = self.data_dir / accession / "metadata.json"
        if metadata_path.exists():
            try:
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
                    return metadata.get("files", [])
            except Exception:
                pass

        return []

    def _extract_project_summary(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract summary from project metadata."""
        return {
            "title": metadata.get("title", ""),
            "file_count": metadata.get("file_count", 0),
            "species": metadata.get("species", []),
            "instrument": metadata.get("instrument", ""),
        }

    def _save_instructions(
        self,
        api_error: str | None = None,
        scrape_error: str | None = None,
    ) -> None:
        """Save instructions for manual project configuration."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        instructions = f"""
GlycoPOST Project Configuration
===============================

Generated: {timestamp}

STATUS: PARTIAL STUB CONNECTOR
------------------------------
The GlycoPOST API is not fully documented. This connector attempted:
1. REST API: {api_error or 'Not attempted'}
2. HTML Scraping: {scrape_error or 'Not attempted'}

RECOMMENDED ACTION: Provide project accessions manually in config.

HOW TO CONFIGURE
----------------
Add project accessions to configs/mirror.yaml:

```yaml
sources:
  glycopost:
    enabled: true
    mode: "project_download"
    project_accessions:
      - "GPST000001"
      - "GPST000002"
    ms_download_policy: "metadata_only"  # or "small_files_only" or "all"
    ms_size_limit_mb: 100  # for small_files_only
    confirm_large_download: false  # set true to download all files > limit
```

FINDING PROJECT ACCESSIONS
--------------------------
1. Visit https://glycopost.glycosmos.org/
2. Browse or search for projects
3. Note the accession number (e.g., GPST000123)
4. Add to project_accessions list in config

DOWNLOAD POLICIES
-----------------
- metadata_only: Only download project metadata (default, safe)
- small_files_only: Download files smaller than ms_size_limit_mb
- all: Download all files (requires confirm_large_download: true)

DEBUGGING
---------
HTML snapshots are saved in: {self._snapshots_dir}
Check these files if you need to debug the connector.
"""

        path = self.data_dir / "CONFIGURATION.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(instructions)

        logger.info(f"Saved configuration instructions: {path}")
