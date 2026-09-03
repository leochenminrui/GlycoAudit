#!/usr/bin/env python3
"""Run the leakage-controlled GlycoAudit PU-ranking benchmark.

This is the supported public entry point for the primary prototype diagnostic.
Known positives used to construct an agent prototype are never used to evaluate
that same prototype.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from glycoaudit.config import get_data_dir, get_outputs_dir
from glycoaudit.evaluation import (
    evaluate_positive_kfold,
    infer_feature_columns,
    run_null_controls,
)


def portable_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled positive--unlabeled relation-recovery evaluation"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=get_data_dir() / "ssv_features.csv",
        help="One-row-per-glycan feature table",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=get_data_dir() / "evidence_records.csv",
        help=(
            "Agent--glycan evidence table; either label=1 or "
            "evidence_status=positive identifies explicit positives"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=get_outputs_dir() / "pu_ranking",
        help="Directory for protocol, per-fold, per-agent, and aggregate outputs",
    )
    parser.add_argument("--feature-columns", nargs="+", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-positives", type=int, default=2)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--preprocessing",
        choices=["standardize", "tfidf", "none"],
        default="standardize",
        help="Fold-fitted feature preprocessing; use tfidf for released WURCS n-gram counts",
    )
    parser.add_argument(
        "--null-repeats",
        type=int,
        default=0,
        help="Random-ranking and feature-permutation repeats; 0 skips null controls",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=None,
        help="Released agent_id,glytoucan_id,fold assignment table",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=None,
        help="Released candidate universe; features are restricted to included glytoucan_id values",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.features.exists():
        raise SystemExit(
            f"Feature table not found: {args.features}\n"
            "Download the frozen data release or pass --features explicitly."
        )
    if not args.labels.exists():
        raise SystemExit(
            f"Label table not found: {args.labels}\n"
            "Download the frozen data release or pass --labels explicitly."
        )

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    fold_manifest = pd.read_csv(args.fold_manifest) if args.fold_manifest else None
    if args.candidate_manifest:
        candidate_manifest = pd.read_csv(args.candidate_manifest)
        if "glytoucan_id" not in candidate_manifest.columns:
            raise ValueError("Candidate manifest must contain glytoucan_id")
        if "included_in_universe" in candidate_manifest.columns:
            included = candidate_manifest["included_in_universe"].astype(str).str.lower().isin(
                {"1", "true", "yes"}
            )
            candidate_manifest = candidate_manifest[included]
        candidate_ids = set(candidate_manifest["glytoucan_id"].dropna().astype(str))
        features = features[features["glytoucan_id"].astype(str).isin(candidate_ids)].copy()
        if not len(features):
            raise ValueError("Candidate manifest leaves no feature rows")
    selected_feature_columns = list(
        args.feature_columns or infer_feature_columns(features)
    )
    result = evaluate_positive_kfold(
        features,
        labels,
        feature_columns=selected_feature_columns,
        n_splits=args.n_splits,
        minimum_positives=args.min_positives,
        ks=args.ks,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        preprocessing=args.preprocessing,
        fold_manifest=fold_manifest,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    result.per_fold.to_csv(args.outdir / "metrics_per_agent_fold.csv", index=False)
    result.per_agent.to_csv(args.outdir / "metrics_per_agent.csv", index=False)
    with (args.outdir / "aggregate_metrics.json").open("w") as handle:
        json.dump(result.aggregate, handle, indent=2, sort_keys=True)

    if args.null_repeats > 0:
        null_result = run_null_controls(
            features,
            labels,
            result,
            feature_columns=selected_feature_columns,
            n_splits=args.n_splits,
            minimum_positives=args.min_positives,
            ks=args.ks,
            seed=args.seed,
            n_repeats=args.null_repeats,
            preprocessing=args.preprocessing,
            fold_manifest=fold_manifest,
        )
        null_result.distributions.to_csv(args.outdir / "null_distributions.csv", index=False)
        with (args.outdir / "null_pvalues.json").open("w") as handle:
            json.dump(null_result.pvalues, handle, indent=2, sort_keys=True)

    protocol = {
        "features": portable_path(args.features),
        "labels": portable_path(args.labels),
        "feature_columns": selected_feature_columns,
        "n_splits": args.n_splits,
        "minimum_positives": args.min_positives,
        "ks": args.ks,
        "bootstrap_replicates": args.bootstrap,
        "null_repeats": args.null_repeats,
        "seed": args.seed,
        "fold_manifest": portable_path(args.fold_manifest) if args.fold_manifest else None,
        "candidate_manifest": (
            portable_path(args.candidate_manifest) if args.candidate_manifest else None
        ),
        "preprocessing": args.preprocessing,
        "positive_evidence_rule": (
            "label == 1" if "label" in labels.columns
            else "evidence_status == positive"
        ),
        "leakage_control": {
            "prototype": "training positives only",
            "evaluation": "held-out positives only",
            "candidate_pool": "training positives excluded",
            "preprocessing": (
                f"{args.preprocessing} fitted with held-out positive features excluded"
            ),
        },
    }
    with (args.outdir / "protocol.json").open("w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)

    print(json.dumps(result.aggregate, indent=2, sort_keys=True))
    print(f"Results written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
