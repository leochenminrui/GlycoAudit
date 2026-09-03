#!/usr/bin/env python3
"""Compute a fixed, auditable character-ngram hash representation of WURCS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_wurcs(value: str) -> str:
    parts = str(value or "").strip().split("/")
    if len(parts) < 2:
        return str(value or "").strip()
    parts[-1] = "_".join(sorted(token for token in parts[-1].split("_") if token))
    return "/".join(parts)


def character_ngrams(value: str, minimum: int, maximum: int) -> list[str]:
    normalized = normalize_wurcs(value)
    return [
        normalized[index : index + size]
        for size in range(minimum, maximum + 1)
        for index in range(max(0, len(normalized) - size + 1))
    ]


def feature_bucket(ngram: str, n_features: int) -> int:
    digest = hashlib.sha256(ngram.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_features


def compute_rows(
    registry: pd.DataFrame,
    *,
    minimum: int,
    maximum: int,
    n_features: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ngram_documents: dict[str, set[str]] = defaultdict(set)
    for record in registry.to_dict(orient="records"):
        glytoucan_id = str(record["glytoucan_id"])
        ngrams = character_ngrams(str(record["wurcs"]), minimum, maximum)
        bucket_counts = Counter(feature_bucket(ngram, n_features) for ngram in ngrams)
        row: dict[str, Any] = {"glytoucan_id": glytoucan_id}
        for bucket in range(n_features):
            row[f"wurcs_hash_{bucket:04d}"] = bucket_counts.get(bucket, 0)
        rows.append(row)
        for ngram in set(ngrams):
            ngram_documents[ngram].add(glytoucan_id)

    by_bucket: dict[int, list[str]] = defaultdict(list)
    for ngram in ngram_documents:
        by_bucket[feature_bucket(ngram, n_features)].append(ngram)
    mapping_rows: list[dict[str, Any]] = []
    for bucket, ngrams in sorted(by_bucket.items()):
        collision_count = len(ngrams)
        for ngram in sorted(ngrams):
            mapping_rows.append(
                {
                    "feature_column": f"wurcs_hash_{bucket:04d}",
                    "bucket": bucket,
                    "ngram": ngram,
                    "ngram_length": len(ngram),
                    "document_frequency": len(ngram_documents[ngram]),
                    "observed_ngrams_in_bucket": collision_count,
                }
            )
    return rows, mapping_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("Invalid n-gram range")
    if args.n_features < 2:
        raise ValueError("n_features must be at least 2")
    registry = pd.read_csv(args.registry, keep_default_na=False)
    registry = registry[
        (registry["registry_status"] == "resolved") & registry["wurcs"].astype(str).ne("")
    ].sort_values("glytoucan_id")
    rows, mapping = compute_rows(
        registry,
        minimum=args.ngram_min,
        maximum=args.ngram_max,
        n_features=args.n_features,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_columns = [f"wurcs_hash_{bucket:04d}" for bucket in range(args.n_features)]
    write_csv(args.outdir / "wurcs_features.csv", rows, ["glytoucan_id", *feature_columns])
    write_csv(
        args.outdir / "wurcs_hash_mapping.csv",
        mapping,
        [
            "feature_column",
            "bucket",
            "ngram",
            "ngram_length",
            "document_frequency",
            "observed_ngrams_in_bucket",
        ],
    )
    sequence_rows = [
        {
            "glytoucan_id": str(record["glytoucan_id"]),
            "wurcs": str(record["wurcs"]),
            "normalized_wurcs": normalize_wurcs(str(record["wurcs"])),
        }
        for record in registry.to_dict(orient="records")
    ]
    write_csv(
        args.outdir / "wurcs_sequences.csv",
        sequence_rows,
        ["glytoucan_id", "wurcs", "normalized_wurcs"],
    )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "representation": "fixed SHA-256 character-ngram hashing followed by fold-fitted TF-IDF",
        "ngram_range": [args.ngram_min, args.ngram_max],
        "n_features": args.n_features,
        "n_glycans": len(rows),
        "hash": "unsigned first-64-bit SHA-256 modulo n_features",
        "normalization": "sort independent clauses in final WURCS linkage section",
        "leakage_control": (
            "Hash buckets are fixed without fitting; inverse-document frequencies "
            "must be fitted inside each evaluation fold excluding held-out positives."
        ),
        "observed_unique_ngrams": len(mapping),
        "occupied_buckets": len({row['bucket'] for row in mapping}),
    }
    (args.outdir / "wurcs_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
