"""PRIDE source connector.

PRIDE is the PRoteomics IDEntifications database at EMBL-EBI.
Website: https://www.ebi.ac.uk/pride/
API: https://www.ebi.ac.uk/pride/ws/archive/v2/

This connector implements project-level mirroring for proteomics/glycoproteomics data:
1. User provides project accessions in config
2. Downloads project metadata via the well-documented REST API
3. Optionally downloads raw files based on size policy

API Documentation: https://www.ebi.ac.uk/pride/ws/archive/v2/swagger-ui.html
"""

from __future__ import annotations

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

logger = get_logger("pride")


# API endpoints
API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"

ENDPOINTS = {
    "projects": f"{API_BASE}/projects",
    "project": f"{API_BASE}/projects",  # + /{accession}
    "files": f"{API_BASE}/files",
    "project_files": f"{API_BASE}/projects",  # + /{accession}/files
}


class PrideSource(BaseSource):
    """Source connector for PRIDE proteomics repository."""

    SOURCE_NAME = "pride"
    CATEGORY = "D"  # MS datasets

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        super().__init__(config, http_client)
        self._project_accessions = self.source_config.project_accessions
        self._download_policy = self.source_config.ms_download_policy
        self._size_limit_mb = self.source_config.ms_size_limit_mb
        self._confirm_large = self.source_config.confirm_large_download

    def list_items(self) -> Iterator[SourceItem]:
        """List projects for download.

        Uses project accessions from config. For glycoproteomics,
        users should search PRIDE and add relevant accessions.
        """
        if not self.is_enabled:
            logger.info("PRIDE source is disabled")
            return

        logger.info("Preparing PRIDE project list...")

        if not self._project_accessions:
            logger.info("No project accessions configured for PRIDE")
            self._save_search_instructions()
            return

        logger.info(f"Processing {len(self._project_accessions)} PRIDE projects")

        for accession in self._project_accessions:
            yield SourceItem(
                source=self.SOURCE_NAME,
                item_id=accession,
                url=f"{ENDPOINTS['project']}/{accession}",
                item_type="ms_project",
                metadata={"source": "config"},
            )

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
        metadata_records = self._download_project_metadata(accession, project_dir)
        records.extend(metadata_records)

        # Download file metadata
        file_meta_records = self._download_file_metadata(accession, project_dir)
        records.extend(file_meta_records)

        # Optionally download files
        if self._download_policy != "metadata_only":
            file_records = self._download_project_files(accession, project_dir)
            records.extend(file_records)

        return records

    def _download_project_metadata(
        self,
        accession: str,
        project_dir: Path,
    ) -> list[DownloadRecord]:
        """Download project metadata from PRIDE API."""
        records = []

        url = f"{ENDPOINTS['project']}/{accession}"

        try:
            response = self.http.get_json(url)

            # Save metadata
            metadata_path = project_dir / "metadata.json"
            sha256, size = save_json_response(response, metadata_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.MS_PROJECT,
                    url=url,
                    local_path=metadata_path,
                    sha256=sha256,
                    size=size,
                    extra=self._extract_project_summary(response),
                )
            )

            logger.info(f"Downloaded PRIDE project metadata: {accession}")

        except Exception as e:
            logger.error(f"Failed to download metadata for {accession}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=accession,
                    item_type=ItemType.MS_PROJECT,
                    url=url,
                    local_path=project_dir / "metadata.json",
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _download_file_metadata(
        self,
        accession: str,
        project_dir: Path,
    ) -> list[DownloadRecord]:
        """Download file listing for a project."""
        records = []

        url = f"{ENDPOINTS['project_files']}/{accession}/files"

        try:
            # PRIDE API returns paginated results
            all_files = []
            page = 0
            page_size = 100

            while True:
                paginated_url = f"{url}?pageSize={page_size}&page={page}"
                response = self.http.get_json(paginated_url)

                # Handle the response structure
                if isinstance(response, list):
                    files = response
                elif isinstance(response, dict):
                    files = response.get("_embedded", {}).get("files", [])
                    if not files:
                        files = response.get("list", [])
                else:
                    break

                all_files.extend(files)

                if len(files) < page_size:
                    break

                page += 1

            # Save file listing
            files_path = project_dir / "files.json"
            file_data = {
                "accession": accession,
                "file_count": len(all_files),
                "files": all_files,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            sha256, size = save_json_response(file_data, files_path)

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"{accession}/files",
                    item_type=ItemType.METADATA,
                    url=url,
                    local_path=files_path,
                    sha256=sha256,
                    size=size,
                    extra={"file_count": len(all_files)},
                )
            )

            logger.info(f"Found {len(all_files)} files for project {accession}")

        except Exception as e:
            logger.error(f"Failed to get file list for {accession}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"{accession}/files",
                    item_type=ItemType.METADATA,
                    url=url,
                    local_path=project_dir / "files.json",
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _download_project_files(
        self,
        accession: str,
        project_dir: Path,
    ) -> list[DownloadRecord]:
        """Download project data files."""
        records = []

        # Load file list
        files_path = project_dir / "files.json"
        if not files_path.exists():
            logger.warning(f"No file list for {accession}")
            return records

        try:
            import json

            with open(files_path) as f:
                file_data = json.load(f)

            files = file_data.get("files", [])
        except Exception as e:
            logger.error(f"Failed to load file list: {e}")
            return records

        # Calculate total size
        total_size_mb = sum(self._get_file_size_mb(f) for f in files)

        logger.info(f"Project {accession}: {len(files)} files, {total_size_mb:.1f} MB total")

        # Apply download policy
        if self._download_policy == "all":
            if not self._confirm_large and total_size_mb > 1000:  # > 1GB
                logger.warning(
                    f"Project {accession} has {total_size_mb:.1f} MB of files. "
                    f"Set confirm_large_download: true to download."
                )
                return records
            files_to_download = files
        elif self._download_policy == "small_files_only":
            files_to_download = [
                f for f in files if self._get_file_size_mb(f) <= self._size_limit_mb
            ]
            logger.info(f"Downloading {len(files_to_download)} files <= {self._size_limit_mb} MB")
        else:
            return records

        # Download files
        files_dir = project_dir / "files"
        files_dir.mkdir(exist_ok=True)

        for file_info in files_to_download:
            file_records = self._download_file(accession, files_dir, file_info)
            records.extend(file_records)

        return records

    def _download_file(
        self,
        accession: str,
        files_dir: Path,
        file_info: dict[str, Any],
    ) -> list[DownloadRecord]:
        """Download a single file."""
        records = []

        # Extract file URL and name
        url = file_info.get("publicFileLocations", [{}])[0].get("value", "")
        if not url:
            url = file_info.get("downloadLink", "")
        if not url:
            # Try to construct URL from accession and filename
            filename = file_info.get("fileName", "")
            if filename:
                url = f"https://ftp.pride.ebi.ac.uk/pride/data/archive/{accession}/{filename}"

        filename = file_info.get("fileName", url.split("/")[-1])

        if not url or not filename:
            return records

        filepath = files_dir / filename

        if filepath.exists():
            logger.debug(f"File already exists: {filename}")
            return records

        try:
            success, record = self.http.download_file(
                url=url,
                dest_path=filepath,
                source=self.SOURCE_NAME,
                item_id=f"{accession}/{filename}",
            )
            record.item_type = ItemType.MS_FILE.value
            record.extra_json = {
                "file_type": file_info.get("fileCategory", {}).get("value", ""),
                "original_name": file_info.get("fileName", ""),
            }
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

    def _get_file_size_mb(self, file_info: dict[str, Any]) -> float:
        """Get file size in MB."""
        size_bytes = file_info.get("fileSizeBytes", 0)
        if not size_bytes:
            size_bytes = file_info.get("size", 0)
        return size_bytes / (1024 * 1024)

    def _extract_project_summary(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract summary from project metadata."""
        return {
            "title": metadata.get("title", ""),
            "description": metadata.get("projectDescription", "")[:200],
            "submission_date": metadata.get("submissionDate", ""),
            "publication_date": metadata.get("publicationDate", ""),
            "keywords": metadata.get("keywords", []),
            "sample_processing": metadata.get("sampleProcessingProtocol", "")[:100],
            "data_processing": metadata.get("dataProcessingProtocol", "")[:100],
        }

    def search_projects(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search PRIDE for projects.

        Useful for finding glycoproteomics datasets.

        Args:
            query: Search query (e.g., "glycoproteomics", "N-glycan")
            max_results: Maximum results to return.

        Returns:
            List of project metadata dictionaries.
        """
        url = f"{ENDPOINTS['projects']}"
        params = {
            "keyword": query,
            "pageSize": min(max_results, 100),
            "page": 0,
        }

        all_results = []

        try:
            while len(all_results) < max_results:
                response = self.http.get_json(url, params=params)

                # Handle response structure
                if isinstance(response, dict):
                    projects = response.get("_embedded", {}).get("compactprojects", [])
                    if not projects:
                        projects = response.get("list", [])
                else:
                    projects = response

                all_results.extend(projects)

                if len(projects) < params["pageSize"]:
                    break

                params["page"] += 1

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return all_results[:max_results]

    def _save_search_instructions(self) -> None:
        """Save instructions for finding relevant PRIDE projects."""
        instructions = """
PRIDE Project Configuration
===========================

To download PRIDE proteomics/glycoproteomics projects:

1. Search for relevant projects:
   - Visit: https://www.ebi.ac.uk/pride/
   - Search for: "glycoproteomics", "N-glycan", "O-glycan", "glycopeptide"
   - Note the project accessions (e.g., PXD012345)

2. Add accessions to config:

configs/mirror.yaml:
```yaml
sources:
  pride:
    enabled: true
    mode: "project_download"
    project_accessions:
      - "PXD012345"
      - "PXD067890"
    ms_download_policy: "metadata_only"  # or "small_files_only" or "all"
    ms_size_limit_mb: 100
    confirm_large_download: false
```

Example glycoproteomics-related projects:
- Search "glycoproteomics" on PRIDE to find relevant datasets
- Search by specific glycan types or analysis methods

API Documentation:
- https://www.ebi.ac.uk/pride/ws/archive/v2/swagger-ui.html

Download policies:
- metadata_only: Only download project/file metadata (default)
- small_files_only: Download files smaller than ms_size_limit_mb
- all: Download all files (may be very large!)
"""

        path = self.data_dir / "SEARCH_INSTRUCTIONS.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(instructions)

        logger.info(f"Saved search instructions: {path}")
