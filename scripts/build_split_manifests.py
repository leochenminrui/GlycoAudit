#!/usr/bin/env python3
"""Freeze candidate, positive-holdout, and WURCS linkage-scaffold manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EDGE_PATTERN = re.compile(r"([a-z?]+)([0-9?]+)-([a-z?]+)([0-9?]+)", re.I)


def linkage_scaffold(wurcs: str) -> tuple[str, str]:
    """Return a residue-agnostic linkage-topology key and its readable rule string."""
    parts = str(wurcs or "").strip().split("/")
    if len(parts) < 5:
        readable = "unparsed:" + str(wurcs or "")
    else:
        counts = parts[1].split(",")
        residue_count = counts[1] if len(counts) > 1 else "?"
        edges = []
        for token in parts[-1].split("_"):
            match = EDGE_PATTERN.fullmatch(token)
            if match:
                left_node, left_position, right_node, right_position = match.groups()
                edges.append(
                    f"{left_node.lower()}{left_position}-{right_node.lower()}{right_position}"
                )
            elif token:
                edges.append(token)
        readable = f"n={residue_count}|edges={'_'.join(sorted(edges))}"
    key = "wurcs_linkage_" + hashlib.sha256(readable.encode("utf-8")).hexdigest()[:16]
    return key, readable


def assign_scaffold_folds(scaffolds: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    group_sizes = (
        scaffolds.groupby("scaffold_key", as_index=False)
        .size()
        .sort_values(["size", "scaffold_key"], ascending=[False, True])
    )
    fold_loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for row in group_sizes.to_dict(orient="records"):
        fold = min(range(n_folds), key=lambda index: (fold_loads[index], index))
        assignment[str(row["scaffold_key"])] = fold
        fold_loads[fold] += int(row["size"])
    result = scaffolds.copy()
    result["fold"] = result["scaffold_key"].map(assignment)
    result["scaffold_size"] = result["scaffold_key"].map(
        group_sizes.set_index("scaffold_key")["size"].to_dict()
    )
    return result.sort_values(["fold", "scaffold_key", "glytoucan_id"])


def build_positive_folds(
    positives: pd.DataFrame,
    *,
    n_splits: int,
    minimum_positives: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for agent_id, group in positives.groupby("agent_id", sort=True):
        glycan_ids = sorted(group["glytoucan_id"].astype(str).unique().tolist())
        if len(glycan_ids) < minimum_positives:
            continue
        shuffled = np.asarray(glycan_ids, dtype=str)
        rng.shuffle(shuffled)
        fold_count = min(n_splits, len(shuffled))
        for fold, held_out in enumerate(np.array_split(shuffled, fold_count)):
            for glytoucan_id in held_out.tolist():
                rows.append(
                    {
                        "agent_id": str(agent_id),
                        "glytoucan_id": str(glytoucan_id),
                        "fold": fold,
                        "fold_count_for_agent": fold_count,
                        "total_known_positives": len(glycan_ids),
                        "assignment_role": "held_out_positive_in_assigned_fold",
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--feature-table", action="append", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-positives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = pd.read_csv(args.evidence, keep_default_na=False)
    registry = pd.read_csv(args.registry, keep_default_na=False)
    feature_sets = []
    for path in args.feature_table:
        table = pd.read_csv(path, usecols=["glytoucan_id"], keep_default_na=False)
        feature_sets.append(set(table["glytoucan_id"].astype(str)))
    candidate_ids = sorted(set.intersection(*feature_sets)) if feature_sets else []
    if not candidate_ids:
        raise ValueError("The declared feature intersection is empty")

    positive = evidence[
        evidence["evidence_status"].astype(str).str.lower().eq("positive")
        & evidence["glytoucan_id"].astype(str).isin(candidate_ids)
    ][["agent_id", "glytoucan_id"]].drop_duplicates()
    counts = positive.groupby("agent_id")["glytoucan_id"].nunique()
    evaluable_agents = sorted(counts[counts >= args.min_positives].index.astype(str))
    positive = positive[positive["agent_id"].astype(str).isin(evaluable_agents)].copy()

    positive_pairs = set(zip(positive["agent_id"].astype(str), positive["glytoucan_id"].astype(str)))
    candidate_rows = []
    for agent_id in evaluable_agents:
        for glytoucan_id in candidate_ids:
            known_positive = (agent_id, glytoucan_id) in positive_pairs
            candidate_rows.append(
                {
                    "candidate_pool_id": "primary_shared_feature_intersection",
                    "agent_id": agent_id,
                    "glytoucan_id": glytoucan_id,
                    "known_evidence_status": "positive" if known_positive else "unlabeled",
                    "known_positive": known_positive,
                    "included_in_universe": True,
                    "training_positive_policy": "removed from candidate pool within its evaluation fold",
                    "candidate_source": "intersection of declared release feature tables",
                }
            )

    fold_manifest = build_positive_folds(
        positive,
        n_splits=args.n_splits,
        minimum_positives=args.min_positives,
        seed=args.seed,
    )

    registry = registry[registry["glytoucan_id"].astype(str).isin(candidate_ids)]
    scaffold_rows = []
    for row in registry.to_dict(orient="records"):
        key, readable = linkage_scaffold(str(row["wurcs"]))
        scaffold_rows.append(
            {
                "glytoucan_id": str(row["glytoucan_id"]),
                "scaffold_key": key,
                "scaffold_definition": readable,
                "scaffold_rule": "residue-agnostic WURCS linkage topology with attachment positions retained",
            }
        )
    scaffold_manifest = assign_scaffold_folds(pd.DataFrame(scaffold_rows), args.n_splits)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.outdir / "candidate_manifest.csv",
        candidate_rows,
        list(candidate_rows[0].keys()),
    )
    fold_manifest.to_csv(args.outdir / "positive_holdout_folds.csv", index=False)
    scaffold_manifest.to_csv(args.outdir / "scaffold_folds.csv", index=False)

    cross_source_columns = [
        "direction",
        "agent_id",
        "glytoucan_id",
        "train_source",
        "test_source",
        "included",
        "reason",
    ]
    write_csv(args.outdir / "cross_source_candidate_manifest.csv", [], cross_source_columns)
    (args.outdir / "cross_source_status.json").write_text(
        json.dumps(
            {
                "status": "not_evaluable",
                "reason": (
                    "The original CFG record-level RFU files and threshold rules are unavailable; "
                    "no cross-source result is generated from catalogue metadata alone."
                ),
                "exclusion_note": (
                    "Cross-source outputs that mixed CarboGrove provenance with CFG labeling "
                    "are excluded."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    protocol = {
        "protocol": "agent-stratified positive holdout",
        "candidate_pool_id": "primary_shared_feature_intersection",
        "feature_tables": [path.name for path in args.feature_table],
        "candidate_glycans": len(candidate_ids),
        "evaluable_agents": len(evaluable_agents),
        "positive_relations": len(positive),
        "requested_splits": args.n_splits,
        "minimum_positives": args.min_positives,
        "seed": args.seed,
        "prototype_scope": "training positives only",
        "candidate_policy": "training positives removed; held-out positives plus unlabeled remain",
        "fold_manifest": "positive_holdout_folds.csv",
        "scaffold_manifest": "scaffold_folds.csv",
    }
    (args.outdir / "positive_holdout_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
