"""Checksum utilities for the glyco mirror."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

# Chunk size for reading files (64KB)
CHUNK_SIZE = 65536


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hexadecimal SHA256 hash.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_sha256_streaming(stream: BinaryIO) -> str:
    """Compute SHA256 hash from a stream.

    Args:
        stream: Binary stream to read from.

    Returns:
        Hexadecimal SHA256 hash.
    """
    sha256 = hashlib.sha256()

    while chunk := stream.read(CHUNK_SIZE):
        sha256.update(chunk)

    return sha256.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Hexadecimal SHA256 hash.
    """
    return hashlib.sha256(data).hexdigest()


class ChecksumManager:
    """Manages checksums for downloaded files."""

    def __init__(self, checksum_file: Path):
        """Initialize the checksum manager.

        Args:
            checksum_file: Path to the sha256sums.txt file.
        """
        self.checksum_file = checksum_file
        self._cache: dict[str, str] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing checksums from file."""
        if not self.checksum_file.exists():
            return

        with open(self.checksum_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    sha256, path = parts
                    # Handle both " *" (binary) and "  " (text) prefixes
                    path = path.lstrip(" *")
                    self._cache[path] = sha256

    def get_checksum(self, relative_path: str) -> str | None:
        """Get cached checksum for a file.

        Args:
            relative_path: Relative path from data root.

        Returns:
            Cached SHA256 or None if not found.
        """
        return self._cache.get(relative_path)

    def add_checksum(self, relative_path: str, sha256: str) -> None:
        """Add or update checksum for a file.

        Args:
            relative_path: Relative path from data root.
            sha256: SHA256 hash.
        """
        self._cache[relative_path] = sha256
        self._append_to_file(relative_path, sha256)

    def _append_to_file(self, relative_path: str, sha256: str) -> None:
        """Append a checksum entry to the file."""
        self.checksum_file.parent.mkdir(parents=True, exist_ok=True)

        # Use binary mode indicator matching sha256sum format
        with open(self.checksum_file, "a") as f:
            f.write(f"{sha256}  {relative_path}\n")

    def verify_file(self, file_path: Path, relative_path: str) -> tuple[bool, str | None]:
        """Verify a file's checksum.

        Args:
            file_path: Absolute path to the file.
            relative_path: Relative path for cache lookup.

        Returns:
            Tuple of (is_valid, computed_hash).
        """
        if not file_path.exists():
            return False, None

        computed = compute_sha256(file_path)
        expected = self._cache.get(relative_path)

        if expected is None:
            # No cached checksum, add it
            self.add_checksum(relative_path, computed)
            return True, computed

        return computed == expected, computed

    def rewrite_file(self) -> None:
        """Rewrite the checksum file with current cache (deduplicates)."""
        self.checksum_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.checksum_file, "w") as f:
            f.write("# SHA256 checksums for glyco mirror\n")
            f.write("# Format: sha256  relative_path\n\n")

            for path in sorted(self._cache.keys()):
                sha256 = self._cache[path]
                f.write(f"{sha256}  {path}\n")

    def file_exists_with_checksum(
        self, file_path: Path, relative_path: str
    ) -> bool:
        """Check if file exists and has valid checksum (for resuming).

        Args:
            file_path: Absolute path to the file.
            relative_path: Relative path for cache lookup.

        Returns:
            True if file exists with matching checksum.
        """
        if not file_path.exists():
            return False

        expected = self._cache.get(relative_path)
        if expected is None:
            return False

        computed = compute_sha256(file_path)
        return computed == expected
