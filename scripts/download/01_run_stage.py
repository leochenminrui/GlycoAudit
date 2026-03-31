#!/usr/bin/env python3
"""
Run mirror in staged mode for safer, more controlled downloads.

Stages:
    1: GlyTouCan + GlycoShape + GlyGen + PRIDE (metadata_only)
    2: CFG (slow mode with configurable rate limiting)
    3: SugarBind + GlycoPOST (binding data via SPARQL + MS metadata)
    4: Optional large MS files (only if config allows)

Usage:
    python scripts/run_stage.py --config configs/mirror.yaml --stage 1
    python scripts/run_stage.py --config configs/mirror.yaml --stage 1 --dry-run
    python scripts/run_stage.py --config configs/mirror.yaml --stage 2 --cfg-max-experiments 10
    python scripts/run_stage.py --config configs/mirror.yaml --stage 3 --max-items 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirror.config import MirrorConfig, SourceConfig
from mirror.index import (
    SourceManifest,
    UnifiedIndex,
    format_size,
    group_errors,
    print_run_summary,
    print_summary,
)
from mirror.io import DownloadLog, HTTPClient
from mirror.logging import DownloadLogger, get_logger, setup_logging
from mirror.sources import get_source, SOURCES

# Stage definitions
STAGES = {
    1: {
        "name": "Core registries and structures",
        "sources": ["glytoucan", "glycoshape", "glygen", "pride"],
        "description": "GlyTouCan + GlycoShape + GlyGen + PRIDE (metadata only)",
    },
    2: {
        "name": "CFG Glycan Array",
        "sources": ["cfg"],
        "description": "CFG glycan array experiments (slow mode)",
    },
    3: {
        "name": "SugarBind + GlycoPOST",
        "sources": ["sugarbind", "glycopost"],
        "description": "SugarBind binding data via GlyCosmos SPARQL + GlycoPOST MS repository",
    },
    4: {
        "name": "Targeted Structures",
        "sources": [],  # Uses custom acquisition script
        "description": "Acquire structures for SugarBind binding glycans",
        "targeted_structures": True,
    },
    5: {
        "name": "Large MS files",
        "sources": ["pride", "glycopost"],
        "description": "Download actual MS files (requires confirm_large_download)",
        "ms_files": True,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run glyco mirror in staged mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
    1: GlyTouCan + GlycoShape + GlyGen + PRIDE (metadata_only)
    2: CFG Glycan Array (slow mode)
    3: SugarBind + GlycoPOST (binding data + MS metadata)
    4: Large MS files (requires config flag)

Examples:
    # Dry run stage 1
    python scripts/run_stage.py --config configs/mirror.yaml --stage 1 --dry-run

    # Run stage 1
    python scripts/run_stage.py --config configs/mirror.yaml --stage 1

    # Run stage 2 with limits
    python scripts/run_stage.py --config configs/mirror.yaml --stage 2 \\
        --cfg-max-experiments 50 --cfg-sleep 2.0

    # Run all stages sequentially
    python scripts/run_stage.py --config configs/mirror.yaml --stage all
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
        "--stage",
        "-s",
        type=str,
        required=True,
        help="Stage number (1-4) or 'all'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List items without downloading",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum items to download per source",
    )

    # CFG-specific options
    parser.add_argument(
        "--cfg-max-experiments",
        type=int,
        default=None,
        help="Maximum CFG experiments to download (stage 2)",
    )
    parser.add_argument(
        "--cfg-sleep",
        type=float,
        default=None,
        help="Sleep between CFG experiment downloads (seconds)",
    )

    # Status filter
    parser.add_argument(
        "--only-status",
        type=str,
        choices=["failed", "skipped", "ok"],
        default=None,
        help="Only process items with this status from previous runs",
    )

    # Time filter
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only process items added since this date (YYYY-MM-DD)",
    )

    return parser.parse_args()


def get_stages_to_run(stage_arg: str) -> list[int]:
    """Parse stage argument and return list of stages to run."""
    if stage_arg.lower() == "all":
        return [1, 2, 3]  # Stage 4 requires explicit confirmation

    try:
        stage_num = int(stage_arg)
        if stage_num not in STAGES:
            print(f"Error: Invalid stage number: {stage_num}")
            print(f"Valid stages: {list(STAGES.keys())}")
            sys.exit(1)
        return [stage_num]
    except ValueError:
        print(f"Error: Invalid stage: {stage_arg}")
        print("Use a number (1-4) or 'all'")
        sys.exit(1)


def run_stage(
    stage_num: int,
    config: MirrorConfig,
    args: argparse.Namespace,
    download_log: DownloadLog,
    logger,
) -> dict[str, int]:
    """Run a single stage.

    Returns:
        Stats dictionary with ok/failed/skipped/bytes counts.
    """
    stage = STAGES[stage_num]
    stats = {"ok": 0, "failed": 0, "skipped": 0, "bytes": 0, "errors": []}

    logger.info("=" * 60)
    logger.info(f"STAGE {stage_num}: {stage['name']}")
    logger.info(f"Description: {stage['description']}")
    logger.info("=" * 60)

    # Check if this is targeted structures stage
    if stage.get("targeted_structures"):
        return run_targeted_structures(config, args, download_log, logger)

    # Check if this is MS file stage
    if stage.get("ms_files"):
        # Verify that confirm_large_download is set
        can_download = False
        for source_name in stage["sources"]:
            if source_name in config.sources:
                src_cfg = config.sources[source_name]
                if src_cfg.confirm_large_download and src_cfg.ms_download_policy == "all":
                    can_download = True
                    break

        if not can_download:
            logger.warning("Stage 5 requires confirm_large_download: true in config")
            logger.warning("Skipping stage 5")
            return stats

    # Process each source in this stage
    with HTTPClient(
        user_agent=config.user_agent,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
        requests_per_second=config.requests_per_second,
    ) as http:
        for source_name in stage["sources"]:
            if source_name not in SOURCES:
                logger.warning(f"Unknown source: {source_name}")
                continue

            if not config.is_source_enabled(source_name):
                logger.info(f"Source {source_name} is disabled, skipping")
                continue

            source_stats = run_source(
                source_name=source_name,
                config=config,
                http=http,
                args=args,
                download_log=download_log,
                logger=logger,
                stage_num=stage_num,
            )

            stats["ok"] += source_stats["ok"]
            stats["failed"] += source_stats["failed"]
            stats["skipped"] += source_stats["skipped"]
            stats["bytes"] += source_stats["bytes"]
            stats["errors"].extend(source_stats.get("errors", []))

    return stats


def run_source(
    source_name: str,
    config: MirrorConfig,
    http: HTTPClient,
    args: argparse.Namespace,
    download_log: DownloadLog,
    logger,
    stage_num: int,
) -> dict[str, int]:
    """Run a single source."""
    stats = {"ok": 0, "failed": 0, "skipped": 0, "bytes": 0, "errors": []}

    logger.info("-" * 60)
    logger.info(f"Processing source: {source_name}")
    logger.info("-" * 60)

    try:
        source = get_source(source_name, config, http)
    except ValueError as e:
        logger.error(str(e))
        stats["failed"] += 1
        return stats

    # Check authentication
    if source.requires_auth():
        logger.warning(f"Source {source_name} requires authentication")
        print(source.auth_instructions())
        stats["skipped"] += 1
        return stats

    # Initialize manifest
    manifest = SourceManifest(
        config.meta_dir / "source_manifests",
        source.SOURCE_NAME,
    )

    # Get existing IDs for resumability
    existing_ids = manifest.get_existing_ids()
    logger.info(f"Found {len(existing_ids)} existing items")

    # Determine max items
    max_items = args.max_items
    if source_name == "cfg" and args.cfg_max_experiments:
        max_items = args.cfg_max_experiments

    # CFG sleep between experiments
    cfg_sleep = args.cfg_sleep if source_name == "cfg" else None

    dl_logger = DownloadLogger(logger)
    dl_logger.source_start(source.SOURCE_NAME)
    items_processed = 0

    try:
        for item in source.list_items():
            # Skip if already downloaded
            if item.item_id in existing_ids:
                logger.debug(f"Skipping existing: {item.item_id}")
                stats["skipped"] += 1
                continue

            if args.dry_run:
                logger.info(f"[DRY RUN] Would download: {item.item_id}")
                items_processed += 1
                if max_items and items_processed >= max_items:
                    break
                continue

            # Download
            try:
                records = source.download_item(item)

                for record in records:
                    download_log.append(record)
                    manifest.add_record(record)

                    if record.status == "ok":
                        stats["ok"] += 1
                        stats["bytes"] += record.bytes
                    elif record.status == "failed":
                        stats["failed"] += 1
                        if record.error:
                            stats["errors"].append(record.error)
                    else:
                        stats["skipped"] += 1

            except Exception as e:
                logger.error(f"Error downloading {item.item_id}: {e}")
                stats["failed"] += 1
                stats["errors"].append(str(e))

            items_processed += 1

            # CFG sleep
            if cfg_sleep:
                time.sleep(cfg_sleep)

            if max_items and items_processed >= max_items:
                logger.info(f"Reached max items limit ({max_items})")
                break

        # Save manifest
        if not args.dry_run:
            manifest.save()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        if not args.dry_run:
            manifest.save()
        raise

    dl_logger.source_complete(
        source.SOURCE_NAME,
        stats["ok"],
        stats["failed"],
        stats["skipped"],
    )

    return stats


def run_targeted_structures(
    config: MirrorConfig,
    args: argparse.Namespace,
    download_log: DownloadLog,
    logger,
) -> dict[str, int]:
    """
    Run targeted structure acquisition.

    This imports and calls the acquire_targeted_structures module.
    """
    stats = {"ok": 0, "failed": 0, "skipped": 0, "bytes": 0, "errors": []}

    logger.info("Running targeted structure acquisition...")

    # Import the acquisition module
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))

    try:
        from acquire_targeted_structures import main as acquire_main

        # Run acquisition
        result = acquire_main(
            max_items=args.max_items,
            dry_run=args.dry_run,
            only_status=(args.only_status == "ok"),  # Status check mode
        )

        stats["ok"] = result.get("ok", 0)
        stats["skipped"] = result.get("skipped", 0)
        stats["failed"] = result.get("failed", 0)

        logger.info(f"Targeted acquisition complete: {stats['ok']} ok, {stats['failed']} failed")

    except ImportError as e:
        logger.error(f"Failed to import acquire_targeted_structures: {e}")
        stats["failed"] += 1
        stats["errors"].append(str(e))
    except Exception as e:
        logger.error(f"Error during targeted acquisition: {e}")
        stats["failed"] += 1
        stats["errors"].append(str(e))

    return stats


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = MirrorConfig.from_yaml(config_path)
    config.ensure_directories()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(config.log_dir, level=level)

    # Get stages to run
    stages = get_stages_to_run(args.stage)

    logger.info("=" * 60)
    logger.info("GLYCO DATA MIRROR - STAGED RUN")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Stages: {stages}")
    logger.info(f"Dry run: {args.dry_run}")

    # Initialize download log
    download_log = DownloadLog(config.meta_dir / "downloads.jsonl")

    # Track overall stats
    total_stats = {"ok": 0, "failed": 0, "skipped": 0, "bytes": 0, "errors": []}
    start_time = time.time()

    try:
        for stage_num in stages:
            stage_stats = run_stage(
                stage_num=stage_num,
                config=config,
                args=args,
                download_log=download_log,
                logger=logger,
            )

            total_stats["ok"] += stage_stats["ok"]
            total_stats["failed"] += stage_stats["failed"]
            total_stats["skipped"] += stage_stats["skipped"]
            total_stats["bytes"] += stage_stats["bytes"]
            total_stats["errors"].extend(stage_stats.get("errors", []))

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")

    # Calculate duration
    duration = time.time() - start_time
    duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"

    # Print final summary
    print("\n" + "=" * 60)
    print("STAGED RUN COMPLETE")
    print("=" * 60)
    print(f"\nStages run: {stages}")
    print(f"Duration: {duration_str}")
    print(f"\nResults:")
    print(f"  OK:      {total_stats['ok']}")
    print(f"  Failed:  {total_stats['failed']}")
    print(f"  Skipped: {total_stats['skipped']}")
    print(f"  Size:    {format_size(total_stats['bytes'])}")

    # Group and print errors
    if total_stats["errors"]:
        error_counts = Counter(total_stats["errors"])
        print("\n" + "-" * 60)
        print("TOP FAILURE REASONS:")
        print("-" * 60)
        for error, count in error_counts.most_common(10):
            # Truncate long errors
            if len(error) > 80:
                error = error[:80] + "..."
            print(f"  [{count:4d}] {error}")

    # Build unified index
    if not args.dry_run:
        logger.info("Building unified index...")
        unified_index = UnifiedIndex(config.meta_dir)
        df = unified_index.build_from_download_log(download_log)
        unified_index.save(df)
        logger.info(f"Unified index saved: {unified_index.csv_path}")

    # Print output locations
    print_run_summary(
        config_path=config_path,
        meta_dir=config.meta_dir,
        checksum_dir=config.checksum_dir,
        log_dir=config.log_dir,
    )

    return 0 if total_stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
