"""I/O utilities for the glyco mirror."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .checksum import compute_sha256, compute_sha256_bytes
from .logging import DownloadLogger, get_logger
from .rate_limit import SyncRateLimiter


class ItemType(str, Enum):
    """Type of downloaded item."""

    REGISTRY_RECORD = "registry_record"
    STRUCTURE = "structure"
    MICROARRAY = "microarray"
    MS_PROJECT = "ms_project"
    MS_FILE = "ms_file"
    METADATA = "metadata"
    API_RESPONSE = "api_response"
    BULK_EXPORT = "bulk_export"
    BINDING_RECORD = "binding_record"


class DownloadStatus(str, Enum):
    """Status of a download attempt."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class DownloadRecord:
    """Record of a single download attempt."""

    source: str
    source_item_id: str
    item_type: ItemType | str
    url: str
    local_path: str
    sha256: str
    bytes: int
    downloaded_at: str
    status: DownloadStatus | str
    error: str | None = None
    extra_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert enums to strings
        if isinstance(d["item_type"], ItemType):
            d["item_type"] = d["item_type"].value
        if isinstance(d["status"], DownloadStatus):
            d["status"] = d["status"].value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownloadRecord:
        """Create from dictionary.

        Handles legacy formats:
        - glytoucan_id -> source_item_id
        - timestamp -> downloaded_at
        - file_path -> local_path
        - file_size -> bytes
        - failure_reason -> error
        """
        # Handle legacy field names
        source_item_id = data.get("source_item_id") or data.get("glytoucan_id", "")
        downloaded_at = data.get("downloaded_at") or data.get("timestamp", "")
        local_path = data.get("local_path") or data.get("file_path", "")
        byte_count = data.get("bytes") or data.get("file_size") or 0
        error = data.get("error") or data.get("failure_reason")
        url = data.get("url", "")
        sha256 = data.get("sha256") or ""
        item_type = data.get("item_type", "unknown")
        status = data.get("status", "unknown")

        return cls(
            source=data["source"],
            source_item_id=source_item_id,
            item_type=item_type,
            url=url,
            local_path=local_path or "",
            sha256=sha256 or "",
            bytes=byte_count or 0,
            downloaded_at=downloaded_at,
            status=status,
            error=error,
            extra_json=data.get("extra_json", {}),
        )


class DownloadLog:
    """Append-only log of download records."""

    def __init__(self, log_path: Path):
        """Initialize the download log.

        Args:
            log_path: Path to the downloads.jsonl file.
        """
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DownloadRecord) -> None:
        """Append a record to the log."""
        with open(self.log_path, "a") as f:
            json.dump(record.to_dict(), f)
            f.write("\n")

    def iter_records(self) -> Iterator[DownloadRecord]:
        """Iterate over all records in the log."""
        if not self.log_path.exists():
            return

        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield DownloadRecord.from_dict(json.loads(line))


class HTTPClient:
    """HTTP client with rate limiting and retries."""

    def __init__(
        self,
        user_agent: str,
        timeout: int = 60,
        max_retries: int = 5,
        requests_per_second: float = 1.0,
    ):
        """Initialize the HTTP client.

        Args:
            user_agent: User agent string.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            requests_per_second: Rate limit.
        """
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = SyncRateLimiter(requests_per_second)
        self.logger = get_logger("http")
        self.download_logger = DownloadLogger(self.logger)

        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _make_retry_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a request with retries."""

        @retry(
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, max=60),
            reraise=True,
        )
        def _do_request() -> httpx.Response:
            self.rate_limiter.wait()
            host = urlparse(url).netloc
            self.rate_limiter.wait_for_host(host)

            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        return _do_request()

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a GET request with retries."""
        return self._make_retry_request("GET", url, **kwargs)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """Make a GET request and return JSON."""
        response = self.get(url, **kwargs)
        return response.json()

    def get_text(self, url: str, **kwargs: Any) -> str:
        """Make a GET request and return text."""
        response = self.get(url, **kwargs)
        return response.text

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a POST request with retries."""
        return self._make_retry_request("POST", url, **kwargs)

    def post_json(self, url: str, **kwargs: Any) -> Any:
        """Make a POST request and return JSON."""
        response = self.post(url, **kwargs)
        return response.json()

    def download_file(
        self,
        url: str,
        dest_path: Path,
        source: str,
        item_id: str,
        expected_sha256: str | None = None,
    ) -> tuple[bool, DownloadRecord]:
        """Download a file to disk atomically.

        Uses atomic download: writes to temp file, then moves to destination.
        If file exists with matching checksum, skips download.

        Args:
            url: URL to download.
            dest_path: Destination path.
            source: Source name for logging.
            item_id: Item ID for logging.
            expected_sha256: Expected checksum for verification/skip.

        Returns:
            Tuple of (success, download_record).
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists with correct checksum (idempotency)
        if expected_sha256 and dest_path.exists():
            existing_sha256 = compute_sha256(dest_path)
            if existing_sha256 == expected_sha256:
                self.logger.debug(f"File exists with correct checksum: {dest_path}")
                return True, DownloadRecord(
                    source=source,
                    source_item_id=item_id,
                    item_type=ItemType.STRUCTURE,
                    url=url,
                    local_path=str(dest_path),
                    sha256=existing_sha256,
                    bytes=dest_path.stat().st_size,
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                    status=DownloadStatus.SKIPPED,
                    extra_json={"reason": "already_exists_with_matching_checksum"},
                )

        self.download_logger.download_start(source, item_id, url)

        try:
            response = self._make_retry_request("GET", url)

            # Compute checksum
            sha256 = compute_sha256_bytes(response.content)
            size = len(response.content)

            # Atomic write: write to temp file, then move
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            try:
                with open(tmp_path, "wb") as f:
                    f.write(response.content)

                # Atomic move
                shutil.move(str(tmp_path), str(dest_path))
            except Exception:
                # Clean up temp file on error
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

            record = DownloadRecord(
                source=source,
                source_item_id=item_id,
                item_type=ItemType.STRUCTURE,
                url=url,
                local_path=str(dest_path),
                sha256=sha256,
                bytes=size,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                status=DownloadStatus.OK,
            )

            self.download_logger.download_success(source, item_id, dest_path, size)
            return True, record

        except RetryError as e:
            error_msg = str(e.last_attempt.exception()) if e.last_attempt.exception() else str(e)
            record = DownloadRecord(
                source=source,
                source_item_id=item_id,
                item_type=ItemType.STRUCTURE,
                url=url,
                local_path=str(dest_path),
                sha256="",
                bytes=0,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                status=DownloadStatus.FAILED,
                error=error_msg,
            )
            self.download_logger.download_fail(source, item_id, error_msg)
            return False, record

        except Exception as e:
            error_msg = str(e)
            record = DownloadRecord(
                source=source,
                source_item_id=item_id,
                item_type=ItemType.STRUCTURE,
                url=url,
                local_path=str(dest_path),
                sha256="",
                bytes=0,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                status=DownloadStatus.FAILED,
                error=error_msg,
            )
            self.download_logger.download_fail(source, item_id, error_msg)
            return False, record

    def download_bytes(self, url: str) -> tuple[bytes, int]:
        """Download content as bytes.

        Args:
            url: URL to download.

        Returns:
            Tuple of (content, status_code).
        """
        self.rate_limiter.wait()
        response = self._client.get(url)
        return response.content, response.status_code


def save_json_response(data: Any, dest_path: Path) -> tuple[str, int]:
    """Save JSON data to file.

    Args:
        data: JSON-serializable data.
        dest_path: Destination path.

    Returns:
        Tuple of (sha256, size).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(data, indent=2, ensure_ascii=False)
    content_bytes = content.encode("utf-8")

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)

    return compute_sha256_bytes(content_bytes), len(content_bytes)


def save_text_response(text: str, dest_path: Path) -> tuple[str, int]:
    """Save text to file.

    Args:
        text: Text content.
        dest_path: Destination path.

    Returns:
        Tuple of (sha256, size).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    content_bytes = text.encode("utf-8")

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)

    return compute_sha256_bytes(content_bytes), len(content_bytes)


def make_download_record(
    source: str,
    item_id: str,
    item_type: ItemType,
    url: str,
    local_path: Path,
    sha256: str,
    size: int,
    status: DownloadStatus = DownloadStatus.OK,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> DownloadRecord:
    """Create a download record.

    Args:
        source: Source name.
        item_id: Item ID.
        item_type: Type of item.
        url: Source URL.
        local_path: Local file path.
        sha256: SHA256 checksum.
        size: File size in bytes.
        status: Download status.
        error: Error message if failed.
        extra: Extra metadata.

    Returns:
        DownloadRecord instance.
    """
    return DownloadRecord(
        source=source,
        source_item_id=item_id,
        item_type=item_type,
        url=url,
        local_path=str(local_path),
        sha256=sha256,
        bytes=size,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        error=error,
        extra_json=extra or {},
    )
