"""CFG Glycan Array source connector.

The Consortium for Functional Glycomics (CFG) provides glycan microarray data.
Website: https://www.functionalglycomics.org/
Data: https://www.functionalglycomics.org/glycan-array/

This connector:
1. Fetches the experiment listing pages
2. Extracts individual experiment data download links
3. Downloads per-experiment Excel/CSV files
4. Extracts experiment metadata (lectin, array version, date)

Note: The CFG website structure may change. This connector stores HTML snapshots
for debugging and audit purposes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

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

logger = get_logger("cfg_array")


# Website URLs
CFG_BASE = "https://www.functionalglycomics.org"
ARRAY_BASE = f"{CFG_BASE}/glycan-array"
EXPERIMENT_LIST = f"{ARRAY_BASE}/primscreen"

# Alternative/backup URLs
ALT_BASES = [
    "https://ncfg.hms.harvard.edu",  # New CFG location
    "http://www.functionalglycomics.org",  # HTTP fallback
]


class CFGArraySource(BaseSource):
    """Source connector for CFG Glycan Array experiments."""

    SOURCE_NAME = "cfg"
    CATEGORY = "C"  # Lectin-glycan binding (microarray)

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        super().__init__(config, http_client)
        self._base_url = CFG_BASE
        self._cached_experiments: list[dict[str, Any]] = []
        self._snapshots_dir = self.data_dir / "_snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    def list_items(self) -> Iterator[SourceItem]:
        """List all glycan array experiments.

        Fetches experiment listing pages and extracts links.
        """
        if not self.is_enabled:
            logger.info("CFG Array source is disabled")
            return

        logger.info("Fetching CFG glycan array experiment list...")

        # Try to fetch experiment listing
        experiments = self._fetch_experiment_list()

        if not experiments:
            logger.warning("No experiments found. CFG website may have changed.")
            logger.info("Saving instructions for manual access...")
            self._save_access_instructions()
            return

        logger.info(f"Found {len(experiments)} experiments")

        for exp in experiments:
            yield SourceItem(
                source=self.SOURCE_NAME,
                item_id=exp["id"],
                url=exp["url"],
                item_type="microarray",
                metadata=exp,
            )

    def _fetch_experiment_list(self) -> list[dict[str, Any]]:
        """Fetch the list of experiments from CFG website."""
        experiments = []

        # Check for manual manifest first
        manifest_path = self.data_dir / "manual_manifest.csv"
        if manifest_path.exists():
            logger.info(f"Using manual manifest: {manifest_path}")
            return self._load_manual_manifest(manifest_path)

        # Try main URL first, then alternatives
        urls_to_try = [EXPERIMENT_LIST] + [
            f"{base}/glycan-array/primscreen" for base in ALT_BASES
        ]

        # Also try additional known paths
        for base in [CFG_BASE] + ALT_BASES:
            urls_to_try.extend([
                f"{base}/glycan-array/",
                f"{base}/glycan-array/data",
                f"{base}/data/glycan-array",
            ])

        html_content = None
        working_url = None
        last_error = None

        for url in urls_to_try:
            try:
                logger.debug(f"Trying: {url}")
                html_content = self.http.get_text(url)
                working_url = url
                self._base_url = urlparse(url).scheme + "://" + urlparse(url).netloc
                logger.info(f"Successfully accessed: {url}")
                break
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Failed to fetch {url}: {e}")
                continue

        if not html_content:
            logger.error(f"Could not access CFG experiment listing. Last error: {last_error}")
            logger.error("Tried URLs: " + ", ".join(urls_to_try[:3]) + "...")
            return experiments

        # Save snapshot with URL info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self._snapshots_dir / f"experiment_list_{timestamp}.html"
        save_text_response(html_content, snapshot_path)
        logger.info(f"Saved HTML snapshot: {snapshot_path}")

        # Save URL info for debugging
        url_info_path = self._snapshots_dir / f"experiment_list_{timestamp}_url.txt"
        with open(url_info_path, "w") as f:
            f.write(f"URL: {working_url}\n")
            f.write(f"Base: {self._base_url}\n")
            f.write(f"Timestamp: {timestamp}\n")

        # Parse HTML
        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception as e:
            logger.error(f"Failed to parse HTML: {e}")
            # Try with html.parser as fallback
            try:
                soup = BeautifulSoup(html_content, "html.parser")
            except Exception as e2:
                logger.error(f"Fallback parser also failed: {e2}")
                return experiments

        # Try multiple strategies to find experiment links
        experiments = self._parse_experiment_links_v1(soup, working_url)
        logger.debug(f"Strategy v1 found {len(experiments)} experiments")

        if not experiments:
            experiments = self._parse_experiment_links_v2(soup, working_url)
            logger.debug(f"Strategy v2 found {len(experiments)} experiments")

        if not experiments:
            experiments = self._parse_experiment_links_v3(soup, working_url)
            logger.debug(f"Strategy v3 found {len(experiments)} experiments")

        if not experiments:
            logger.warning("No experiments found with any parsing strategy")
            logger.warning(f"Check the HTML snapshot at: {snapshot_path}")
            logger.info("You can create a manual manifest at: " + str(manifest_path))

        self._cached_experiments = experiments
        return experiments

    def _load_manual_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        """Load experiments from a manual manifest CSV file.

        Expected columns:
        - experiment_id (required): Unique identifier
        - download_url or url (required): Direct URL to data file
        - lectin_name (optional): Name of the lectin/protein
        - array_version (optional): Array version used
        - notes (optional): Additional notes
        """
        import csv

        experiments = []
        skipped = 0
        try:
            with open(manifest_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    exp_id = row.get("experiment_id", row.get("id", "")).strip()
                    url = row.get("download_url", row.get("url", "")).strip()

                    if not exp_id:
                        logger.warning(f"Skipping row with missing experiment_id: {row}")
                        skipped += 1
                        continue

                    if not url:
                        logger.warning(f"Skipping {exp_id}: missing download_url")
                        skipped += 1
                        continue

                    experiments.append({
                        "id": exp_id,
                        "url": url,
                        "name": row.get("lectin_name", row.get("name", exp_id)),
                        "lectin_name": row.get("lectin_name", ""),
                        "array_version": row.get("array_version", ""),
                        "notes": row.get("notes", ""),
                        "source": "manual_manifest",
                    })

        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")

        logger.info(f"Loaded {len(experiments)} experiments from manifest (skipped {skipped})")
        return experiments

    def _parse_experiment_links_v1(
        self, soup: BeautifulSoup, base_url: str
    ) -> list[dict[str, Any]]:
        """Parse experiment links - Strategy 1: Table-based layout."""
        experiments = []

        # Look for tables with experiment data
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue

                # Look for experiment ID pattern
                for cell in cells:
                    link = cell.find("a", href=True)
                    if link:
                        href = link.get("href", "")
                        text = link.get_text(strip=True)

                        # Check if this looks like an experiment link
                        if self._is_experiment_link(href, text):
                            exp_id = self._extract_experiment_id(href, text)
                            if exp_id and exp_id not in [e["id"] for e in experiments]:
                                experiments.append({
                                    "id": exp_id,
                                    "url": urljoin(base_url, href),
                                    "name": text,
                                    "row_data": [c.get_text(strip=True) for c in cells],
                                })

        return experiments

    def _parse_experiment_links_v2(
        self, soup: BeautifulSoup, base_url: str
    ) -> list[dict[str, Any]]:
        """Parse experiment links - Strategy 2: List-based layout."""
        experiments = []

        # Look for lists of experiments
        for ul in soup.find_all(["ul", "ol"]):
            for li in ul.find_all("li"):
                link = li.find("a", href=True)
                if link:
                    href = link.get("href", "")
                    text = link.get_text(strip=True)

                    if self._is_experiment_link(href, text):
                        exp_id = self._extract_experiment_id(href, text)
                        if exp_id:
                            experiments.append({
                                "id": exp_id,
                                "url": urljoin(base_url, href),
                                "name": text,
                            })

        return experiments

    def _parse_experiment_links_v3(
        self, soup: BeautifulSoup, base_url: str
    ) -> list[dict[str, Any]]:
        """Parse experiment links - Strategy 3: Any link with experiment patterns."""
        experiments = []

        # Find all links
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if self._is_experiment_link(href, text):
                exp_id = self._extract_experiment_id(href, text)
                if exp_id and exp_id not in [e["id"] for e in experiments]:
                    experiments.append({
                        "id": exp_id,
                        "url": urljoin(base_url, href),
                        "name": text,
                    })

        return experiments

    def _is_experiment_link(self, href: str, text: str) -> bool:
        """Check if a link appears to be an experiment link."""
        patterns = [
            r"primscreen/\d+",
            r"experiment[_/]\d+",
            r"exp[_/]?\d+",
            r"request[_/]?\d+",
            r"cfg\d+",
            r"pa\d+",  # Primary Array numbers
        ]

        combined = f"{href} {text}".lower()
        return any(re.search(p, combined, re.IGNORECASE) for p in patterns)

    def _extract_experiment_id(self, href: str, text: str) -> str | None:
        """Extract experiment ID from link."""
        # Try common patterns
        patterns = [
            r"primscreen/(\d+)",
            r"experiment[_/](\d+)",
            r"exp[_/]?(\d+)",
            r"request[_/]?(\d+)",
            r"(cfg\d+)",
            r"(pa\d+)",
        ]

        combined = f"{href} {text}"
        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback: use any number found
        numbers = re.findall(r"\d+", combined)
        if numbers:
            return f"exp_{numbers[0]}"

        return None

    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download experiment data.

        For each experiment:
        1. Fetch the experiment detail page
        2. Extract data download links
        3. Download data files (Excel/CSV)
        4. Extract metadata

        Args:
            item: Source item to download.

        Returns:
            List of download records.
        """
        records = []
        exp_id = item.item_id

        # Create directory for this experiment
        exp_dir = self.data_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        metadata_path = exp_dir / "metadata.json"
        if metadata_path.exists():
            logger.debug(f"Already downloaded: {exp_id}")
            return [
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=exp_id,
                    item_type=ItemType.MICROARRAY,
                    url=item.url,
                    local_path=metadata_path,
                    sha256="",
                    size=0,
                    status=DownloadStatus.SKIPPED,
                    extra={"reason": "already_exists"},
                )
            ]

        try:
            # Fetch experiment detail page
            logger.debug(f"Fetching experiment page: {item.url}")
            html_content = self.http.get_text(item.url)

            # Save HTML snapshot
            html_path = exp_dir / "page.html"
            save_text_response(html_content, html_path)
            logger.debug(f"Saved experiment page: {html_path}")

            # Parse for data files and metadata
            try:
                soup = BeautifulSoup(html_content, "lxml")
            except Exception:
                soup = BeautifulSoup(html_content, "html.parser")

            data_files = self._find_data_files(soup, item.url)
            metadata = self._extract_metadata(soup, item)
            metadata["data_files_found"] = len(data_files)
            metadata["html_snapshot"] = str(html_path)

            # Save metadata
            sha256, size = save_json_response(metadata, metadata_path)
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"{exp_id}/metadata",
                    item_type=ItemType.METADATA,
                    url=item.url,
                    local_path=metadata_path,
                    sha256=sha256,
                    size=size,
                )
            )

            # Download data files
            for file_info in data_files:
                file_records = self._download_data_file(exp_id, exp_dir, file_info)
                records.extend(file_records)

            if not data_files:
                logger.warning(f"No data files found for experiment {exp_id}")
                # Record as skipped with reason
                records.append(
                    make_download_record(
                        source=self.SOURCE_NAME,
                        item_id=f"{exp_id}/data",
                        item_type=ItemType.MICROARRAY,
                        url=item.url,
                        local_path=exp_dir,
                        sha256="",
                        size=0,
                        status=DownloadStatus.SKIPPED,
                        extra={"reason": "no_data_files_found", "html_snapshot": str(html_path)},
                    )
                )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to download experiment {exp_id}: {error_msg}")

            # Save partial metadata if possible
            try:
                partial_metadata = {
                    "experiment_id": exp_id,
                    "source_url": item.url,
                    "error": error_msg,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }
                save_json_response(partial_metadata, metadata_path)
            except Exception:
                pass

            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=exp_id,
                    item_type=ItemType.MICROARRAY,
                    url=item.url,
                    local_path=exp_dir,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=error_msg,
                )
            )

        return records

    def _find_data_files(
        self, soup: BeautifulSoup, base_url: str
    ) -> list[dict[str, str]]:
        """Find data file download links in experiment page."""
        files = []

        # Look for download links
        download_patterns = [
            r"\.xls[x]?$",
            r"\.csv$",
            r"\.tsv$",
            r"\.txt$",
            r"download",
            r"export",
            r"data",
        ]

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True).lower()

            for pattern in download_patterns:
                if re.search(pattern, href, re.IGNORECASE) or re.search(
                    pattern, text, re.IGNORECASE
                ):
                    full_url = urljoin(base_url, href)

                    # Determine filename
                    filename = href.split("/")[-1].split("?")[0]
                    if not filename or not any(
                        filename.endswith(ext)
                        for ext in [".xls", ".xlsx", ".csv", ".tsv", ".txt"]
                    ):
                        filename = f"data_{len(files)}.xlsx"

                    files.append({
                        "url": full_url,
                        "filename": filename,
                        "link_text": text,
                    })
                    break

        return files

    def _extract_metadata(
        self, soup: BeautifulSoup, item: SourceItem
    ) -> dict[str, Any]:
        """Extract experiment metadata from page."""
        metadata = {
            "experiment_id": item.item_id,
            "source_url": item.url,
            "name": item.metadata.get("name", ""),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Try to extract common metadata fields
        text_content = soup.get_text()

        # Look for lectin name
        lectin_patterns = [
            r"lectin[:\s]+([A-Za-z0-9\-]+)",
            r"protein[:\s]+([A-Za-z0-9\-]+)",
            r"sample[:\s]+([A-Za-z0-9\-]+)",
        ]
        for pattern in lectin_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                metadata["lectin"] = match.group(1)
                break

        # Look for array version
        version_patterns = [
            r"array\s+v(?:ersion)?\s*([0-9.]+)",
            r"v([0-9.]+)\s+array",
            r"version\s*([0-9.]+)",
        ]
        for pattern in version_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                metadata["array_version"] = match.group(1)
                break

        # Look for date
        date_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{2,4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                metadata["experiment_date"] = match.group(1)
                break

        return metadata

    def _download_data_file(
        self,
        exp_id: str,
        exp_dir: Path,
        file_info: dict[str, str],
    ) -> list[DownloadRecord]:
        """Download a data file."""
        records = []

        url = file_info["url"]
        filename = file_info["filename"]
        filepath = exp_dir / filename

        try:
            # Download file
            success, record = self.http.download_file(
                url=url,
                dest_path=filepath,
                source=self.SOURCE_NAME,
                item_id=f"{exp_id}/{filename}",
            )

            # Update record with correct item type
            record.item_type = ItemType.MICROARRAY.value
            records.append(record)

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            records.append(
                make_download_record(
                    source=self.SOURCE_NAME,
                    item_id=f"{exp_id}/{filename}",
                    item_type=ItemType.MICROARRAY,
                    url=url,
                    local_path=filepath,
                    sha256="",
                    size=0,
                    status=DownloadStatus.FAILED,
                    error=str(e),
                )
            )

        return records

    def _save_access_instructions(self) -> None:
        """Save instructions for manual data access."""
        from datetime import datetime

        instructions = """
CFG Glycan Array Data Access (RECOMMENDED: Manual Manifest Mode)
=================================================================

The CFG website structure has changed. Auto-parsing is no longer reliable.

RECOMMENDED: Manual Manifest Mode
----------------------------------

1. Create: data/raw/cfg/manual_manifest.csv

2. CSV Format (with header row):
   ```
   experiment_id,download_url,lectin_name
   primscreen_001,https://ncfg.hms.harvard.edu/.../data.xlsx,ConA
   primscreen_002,https://ncfg.hms.harvard.edu/.../data.xlsx,WGA
   ```

3. Required columns:
   - experiment_id: Unique identifier for the experiment
   - download_url: Direct URL to the data file (Excel/CSV)

4. Optional columns:
   - lectin_name: Name of the lectin/protein tested
   - array_version: Version of the glycan array used
   - notes: Any additional notes

5. Re-run the mirror:
   python scripts/run_stage.py --config configs/mirror.yaml --stage 2

HOW TO FIND EXPERIMENT DATA
----------------------------

1. Visit: https://ncfg.hms.harvard.edu/glycan-array/primscreen
   (or: https://www.functionalglycomics.org/glycan-array/)

2. Browse available experiments

3. Right-click on download links and copy URLs

4. Add each experiment to your manual_manifest.csv

HTML SNAPSHOTS FOR REFERENCE
-----------------------------

HTML snapshots of the experiment listing page are saved in:
  data/raw/cfg/_snapshots/

These can help you understand the current website structure.

CONTACT
--------

CFG Data Questions: cfg-info@bidmc.harvard.edu
NCFG at Harvard: https://ncfg.hms.harvard.edu/

Generated: {timestamp}
"""

        instructions_path = self.data_dir / "ACCESS_INSTRUCTIONS.txt"
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(instructions_path, "w") as f:
            f.write(instructions.format(timestamp=datetime.now().isoformat()))

        logger.info(f"Saved access instructions: {instructions_path}")
