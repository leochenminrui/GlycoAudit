"""Index management for the glyco mirror."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import DownloadLog, DownloadRecord


class SourceManifest:
    """Manages per-source manifest files."""

    def __init__(self, manifest_dir: Path, source_name: str):
        """Initialize the manifest.

        Args:
            manifest_dir: Directory for manifest files.
            source_name: Name of the source.
        """
        self.manifest_path = manifest_dir / f"{source_name}.csv"
        self.source_name = source_name
        self._records: list[dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing manifest if it exists."""
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self._records = list(reader)

    def add_record(self, record: DownloadRecord) -> None:
        """Add a record to the manifest."""
        self._records.append(record.to_dict())

    def save(self) -> None:
        """Save the manifest to disk."""
        if not self._records:
            return

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all keys
        all_keys = set()
        for record in self._records:
            all_keys.update(record.keys())

        # Define column order
        column_order = [
            "source",
            "source_item_id",
            "item_type",
            "url",
            "local_path",
            "sha256",
            "bytes",
            "downloaded_at",
            "status",
            "error",
        ]
        # Add any extra keys not in the order
        fieldnames = column_order + [k for k in sorted(all_keys) if k not in column_order]

        with open(self.manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self._records:
                # Convert extra_json to string if present
                if "extra_json" in record and isinstance(record["extra_json"], dict):
                    record["extra_json"] = json.dumps(record["extra_json"])
                writer.writerow(record)

    def get_existing_ids(self) -> set[str]:
        """Get set of already downloaded item IDs."""
        return {r["source_item_id"] for r in self._records if r.get("status") == "ok"}


class UnifiedIndex:
    """Manages the unified index across all sources."""

    def __init__(self, meta_dir: Path):
        """Initialize the unified index.

        Args:
            meta_dir: Directory for metadata files.
        """
        self.meta_dir = meta_dir
        self.csv_path = meta_dir / "unified_index.csv"
        self.parquet_path = meta_dir / "unified_index.parquet"

    def build_from_download_log(self, download_log: DownloadLog) -> pd.DataFrame:
        """Build unified index from the download log.

        Args:
            download_log: The download log to read from.

        Returns:
            DataFrame with the unified index.
        """
        records = list(download_log.iter_records())

        if not records:
            return pd.DataFrame()

        # Convert to dictionaries
        data = [r.to_dict() for r in records]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Ensure consistent column order
        column_order = [
            "source",
            "source_item_id",
            "item_type",
            "url",
            "local_path",
            "sha256",
            "bytes",
            "downloaded_at",
            "status",
            "error",
            "extra_json",
        ]
        # Add columns in order, then any extras
        existing_cols = [c for c in column_order if c in df.columns]
        extra_cols = [c for c in df.columns if c not in column_order]
        df = df[existing_cols + extra_cols]

        return df

    def build_from_manifests(self, manifest_dir: Path) -> pd.DataFrame:
        """Build unified index from source manifests.

        Args:
            manifest_dir: Directory containing manifest files.

        Returns:
            DataFrame with the unified index.
        """
        all_records = []

        for manifest_path in manifest_dir.glob("*.csv"):
            if not manifest_path.is_file():
                continue

            df = pd.read_csv(manifest_path)
            all_records.append(df)

        if not all_records:
            return pd.DataFrame()

        return pd.concat(all_records, ignore_index=True)

    def save(self, df: pd.DataFrame) -> None:
        """Save the unified index.

        Args:
            df: DataFrame to save.
        """
        if df.empty:
            return

        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV (always works)
        df.to_csv(self.csv_path, index=False)

        # Try to save as Parquet (requires pyarrow)
        try:
            df.to_parquet(self.parquet_path, index=False)
        except ImportError:
            # pyarrow not installed, skip parquet
            pass
        except Exception as e:
            # Handle other parquet errors (e.g., unsupported types)
            import logging
            logging.getLogger("glyco_mirror").warning(f"Could not save parquet: {e}")

    def load(self) -> pd.DataFrame | None:
        """Load the unified index.

        Returns:
            DataFrame or None if not found.
        """
        if self.parquet_path.exists():
            return pd.read_parquet(self.parquet_path)
        elif self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return None

    def get_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get summary statistics from the index.

        Args:
            df: Unified index DataFrame.

        Returns:
            Summary statistics.
        """
        if df.empty:
            return {"total_files": 0, "sources": {}}

        summary = {
            "total_files": len(df),
            "total_bytes": int(df["bytes"].sum()) if "bytes" in df.columns else 0,
            "sources": {},
        }

        for source in df["source"].unique():
            source_df = df[df["source"] == source]
            status_counts = source_df["status"].value_counts().to_dict()

            summary["sources"][source] = {
                "total": len(source_df),
                "ok": status_counts.get("ok", 0),
                "failed": status_counts.get("failed", 0),
                "skipped": status_counts.get("skipped", 0),
                "bytes": int(source_df["bytes"].sum()) if "bytes" in source_df.columns else 0,
            }

        return summary


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def group_errors(df: pd.DataFrame, top_n: int = 10) -> list[tuple[str, int]]:
    """Group and count error messages.

    Args:
        df: DataFrame with 'error' column.
        top_n: Number of top errors to return.

    Returns:
        List of (error_message, count) tuples.
    """
    if df.empty or "error" not in df.columns:
        return []

    failed_df = df[df["status"] == "failed"]
    if failed_df.empty:
        return []

    # Normalize error messages (truncate, remove specific details)
    def normalize_error(err: str | None) -> str:
        if not err or pd.isna(err):
            return "Unknown error"
        err = str(err)
        # Truncate long errors
        if len(err) > 100:
            err = err[:100] + "..."
        return err

    error_counts = failed_df["error"].apply(normalize_error).value_counts()
    return list(error_counts.head(top_n).items())


def print_summary(summary: dict[str, Any], error_groups: list[tuple[str, int]] | None = None) -> None:
    """Print summary statistics to console.

    Args:
        summary: Summary statistics from get_summary().
        error_groups: Optional grouped error counts.
    """
    print("\n" + "=" * 60)
    print("MIRROR SUMMARY")
    print("=" * 60)

    print(f"\nTotal files: {summary['total_files']}")
    print(f"Total size: {format_size(summary['total_bytes'])}")

    print("\nBy source:")
    print("-" * 60)

    for source, stats in sorted(summary["sources"].items()):
        print(f"\n  {source}:")
        print(f"    Total:   {stats['total']}")
        print(f"    OK:      {stats['ok']}")
        print(f"    Failed:  {stats['failed']}")
        print(f"    Skipped: {stats['skipped']}")
        print(f"    Size:    {format_size(stats['bytes'])}")

    # Print top errors if provided
    if error_groups:
        print("\n" + "-" * 60)
        print("TOP FAILURE REASONS:")
        print("-" * 60)
        for error, count in error_groups:
            print(f"  [{count:4d}] {error}")

    print("\n" + "=" * 60)


def print_run_summary(
    config_path: Path,
    meta_dir: Path,
    checksum_dir: Path,
    log_dir: Path,
) -> None:
    """Print summary of output locations.

    Args:
        config_path: Path to config file.
        meta_dir: Path to metadata directory.
        checksum_dir: Path to checksum directory.
        log_dir: Path to log directory.
    """
    print("\n" + "-" * 60)
    print("OUTPUT LOCATIONS:")
    print("-" * 60)
    print(f"  Config:     {config_path}")
    print(f"  Manifests:  {meta_dir / 'source_manifests'}")
    print(f"  Index:      {meta_dir / 'unified_index.csv'}")
    print(f"  Downloads:  {meta_dir / 'downloads.jsonl'}")
    print(f"  Checksums:  {checksum_dir / 'sha256sums.txt'}")
    print(f"  Logs:       {log_dir}")
    print("-" * 60)
