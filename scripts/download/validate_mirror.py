#!/usr/bin/env python3
"""
Validate mirror integrity by checking files and checksums.

Usage:
    python scripts/validate_mirror.py --config configs/mirror.yaml
    python scripts/validate_mirror.py --config configs/mirror.yaml --fix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirror.checksum import ChecksumManager, compute_sha256
from mirror.config import MirrorConfig
from mirror.index import UnifiedIndex, format_size


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate mirror integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate all files
    python scripts/validate_mirror.py --config configs/mirror.yaml

    # Validate specific source
    python scripts/validate_mirror.py --config configs/mirror.yaml --source glytoucan

    # Fix checksums file (update with current values)
    python scripts/validate_mirror.py --config configs/mirror.yaml --fix

    # Show detailed output
    python scripts/validate_mirror.py --config configs/mirror.yaml --verbose
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to mirror configuration YAML file",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Validate only specific source",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Update checksums file with current values",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only check file existence, skip checksum verification",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = MirrorConfig.from_yaml(config_path)

    print("=" * 60)
    print("VALIDATING MIRROR")
    print("=" * 60)

    # Load unified index
    unified_index = UnifiedIndex(config.meta_dir)
    df = unified_index.load()

    if df is None or df.empty:
        print("Error: No unified index found. Run build_unified_index.py first.")
        return 1

    # Filter by source if specified
    if args.source:
        df = df[df["source"] == args.source]
        if df.empty:
            print(f"No records found for source: {args.source}")
            return 1

    print(f"Validating {len(df)} records...")

    # Initialize checksum manager
    checksum_file = config.checksum_dir / "sha256sums.txt"
    checksum_manager = ChecksumManager(checksum_file)

    # Track results
    results = {
        "ok": [],
        "missing": [],
        "checksum_mismatch": [],
        "no_checksum": [],
    }

    source_stats: dict[str, dict[str, int]] = {}

    for _, row in df.iterrows():
        source = row["source"]
        item_id = row["source_item_id"]
        local_path = Path(row["local_path"])
        expected_sha256 = row.get("sha256", "")
        status = row.get("status", "ok")

        # Initialize source stats
        if source not in source_stats:
            source_stats[source] = {
                "total": 0,
                "ok": 0,
                "missing": 0,
                "checksum_mismatch": 0,
                "no_checksum": 0,
                "bytes": 0,
            }

        source_stats[source]["total"] += 1

        # Skip failed downloads
        if status == "failed":
            continue

        # Check file existence
        if not local_path.exists():
            results["missing"].append({
                "source": source,
                "item_id": item_id,
                "path": str(local_path),
            })
            source_stats[source]["missing"] += 1
            if args.verbose:
                print(f"MISSING: {local_path}")
            continue

        # Get file size
        file_size = local_path.stat().st_size
        source_stats[source]["bytes"] += file_size

        # Quick mode - skip checksum
        if args.quick:
            results["ok"].append(str(local_path))
            source_stats[source]["ok"] += 1
            continue

        # Verify checksum
        if not expected_sha256:
            results["no_checksum"].append({
                "source": source,
                "item_id": item_id,
                "path": str(local_path),
            })
            source_stats[source]["no_checksum"] += 1

            if args.fix:
                # Compute and add checksum
                computed = compute_sha256(local_path)
                relative_path = str(local_path.relative_to(config.root_dir.parent))
                checksum_manager.add_checksum(relative_path, computed)

            continue

        # Compute actual checksum
        computed_sha256 = compute_sha256(local_path)

        if computed_sha256 == expected_sha256:
            results["ok"].append(str(local_path))
            source_stats[source]["ok"] += 1
            if args.verbose:
                print(f"OK: {local_path}")
        else:
            results["checksum_mismatch"].append({
                "source": source,
                "item_id": item_id,
                "path": str(local_path),
                "expected": expected_sha256,
                "actual": computed_sha256,
            })
            source_stats[source]["checksum_mismatch"] += 1
            if args.verbose:
                print(f"MISMATCH: {local_path}")
                print(f"  Expected: {expected_sha256}")
                print(f"  Actual:   {computed_sha256}")

    # Fix mode - rewrite checksums
    if args.fix:
        print("\nRewriting checksums file...")
        checksum_manager.rewrite_file()
        print(f"Updated: {checksum_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal records: {len(df)}")
    print(f"OK: {len(results['ok'])}")
    print(f"Missing: {len(results['missing'])}")
    print(f"Checksum mismatch: {len(results['checksum_mismatch'])}")
    print(f"No checksum: {len(results['no_checksum'])}")

    # Per-source summary
    print("\n" + "-" * 60)
    print("BY SOURCE:")
    print("-" * 60)

    for source, stats in sorted(source_stats.items()):
        print(f"\n{source}:")
        print(f"  Total:     {stats['total']}")
        print(f"  OK:        {stats['ok']}")
        print(f"  Missing:   {stats['missing']}")
        print(f"  Mismatch:  {stats['checksum_mismatch']}")
        print(f"  No hash:   {stats['no_checksum']}")
        print(f"  Size:      {format_size(stats['bytes'])}")

    # Print details of problems
    if results["missing"] and args.verbose:
        print("\n" + "-" * 60)
        print("MISSING FILES:")
        print("-" * 60)
        for item in results["missing"][:20]:
            print(f"  [{item['source']}] {item['item_id']}")
            print(f"    {item['path']}")
        if len(results["missing"]) > 20:
            print(f"  ... and {len(results['missing']) - 20} more")

    if results["checksum_mismatch"] and args.verbose:
        print("\n" + "-" * 60)
        print("CHECKSUM MISMATCHES:")
        print("-" * 60)
        for item in results["checksum_mismatch"][:10]:
            print(f"  [{item['source']}] {item['item_id']}")
            print(f"    Expected: {item['expected']}")
            print(f"    Actual:   {item['actual']}")
        if len(results["checksum_mismatch"]) > 10:
            print(f"  ... and {len(results['checksum_mismatch']) - 10} more")

    # Return code
    if results["missing"] or results["checksum_mismatch"]:
        print("\nValidation FAILED")
        return 1
    else:
        print("\nValidation PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
