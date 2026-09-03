#!/usr/bin/env python3
"""Validate the open-source package against the reference numerical contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def positive_pairs() -> pd.DataFrame:
    evidence = pd.read_csv(DATA / "evidence_records.csv", dtype=str)
    candidates = pd.read_csv(DATA / "candidate_manifest.csv", dtype=str)
    ids = set(
        candidates.loc[
            candidates["included_in_universe"].str.lower().isin({"1", "true", "yes"}),
            "glytoucan_id",
        ]
    )
    return evidence[
        evidence["evidence_status"].str.lower().eq("positive")
        & evidence["glytoucan_id"].isin(ids)
    ][["agent_id", "glytoucan_id"]].drop_duplicates()


def validate_counts(contract: dict) -> None:
    expected = contract["benchmark"]
    evidence = pd.read_csv(DATA / "evidence_records.csv", dtype=str)
    identifiers = pd.read_csv(DATA / "glycan_identifier_map.csv", dtype=str)
    folds = pd.read_csv(DATA / "positive_holdout_folds.csv", dtype=str)
    candidates = pd.read_csv(DATA / "candidate_manifest.csv", dtype=str)
    pairs = positive_pairs()
    support = pairs.groupby("agent_id")["glytoucan_id"].nunique()
    actual = {
        "source_records": len(evidence),
        "resolved_records": int(evidence["glytoucan_id"].notna().sum()),
        "resolved_glycans": int(identifiers["glytoucan_id"].dropna().nunique()),
        "strictly_joined_records": int(evidence["glytoucan_id"].isin(set(pairs["glytoucan_id"])).sum()),
        "strictly_joined_glycans": int(pairs["glytoucan_id"].nunique()),
        "canonical_joined_relations": len(pairs),
        "popularity_scoring_agents": int(pairs["agent_id"].nunique()),
        "popularity_scoring_relations": len(pairs),
        "evaluated_agents": int((support >= 2).sum()),
        "evaluated_relations": int(support[support >= 2].sum()),
        "agent_folds": int(folds[["agent_id", "fold"]].drop_duplicates().shape[0]),
        "candidate_rows": len(candidates),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise AssertionError(f"{key}: expected {value}, found {actual[key]}")


def validate_primary(contract: dict) -> None:
    locations = {
        "WURCS": DATA / "results/wurcs/aggregate_metrics.json",
        "SSV": DATA / "results/ssv/aggregate_metrics.json",
        "GCV_4_5": DATA / "results/gcv_4_5/aggregate_metrics.json",
        "POPULARITY": RESULTS / "popularity_baseline/aggregate_metrics.json",
    }
    fields = {"mrr": "mrr", "recall_at_5": "recall@5", "known_positive_ap": "known_positive_ap"}
    for name, path in locations.items():
        actual = read_json(path)
        for contract_field, result_field in fields.items():
            close(actual[result_field], contract["primary_results"][name][contract_field], f"{name}.{contract_field}")
    popularity = read_json(locations["POPULARITY"])
    if popularity.get("n_scoring_agents", 227) != 227 or popularity.get("n_scoring_relations", 765) != 765:
        raise AssertionError("Popularity scoring pool must contain 227 agents and 765 relations")


def validate_prototype(contract: dict) -> None:
    path = RESULTS / "prototype_stability/prototype_perturbation_per_fold.csv"
    frame = pd.read_csv(path)
    if "absolute_fold_mean_held_out_rank_change" in frame:
        values = frame.groupby("representation")["absolute_fold_mean_held_out_rank_change"].mean()
    else:
        values = frame.assign(
            absolute_fold_mean_held_out_rank_change=frame["mean_held_out_rank_change"].abs()
        ).groupby("representation")["absolute_fold_mean_held_out_rank_change"].mean()
    for representation in ["WURCS", "SSV", "GCV_4_5"]:
        close(values[representation], contract["prototype_perturbation"][representation], f"prototype.{representation}")


def validate_scope() -> None:
    forbidden_suffixes = {".tex", ".pdf", ".aux", ".bbl", ".blg"}
    forbidden_parts = {"tests", "figure_source_data", "reviewers"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = {part.lower() for part in path.relative_to(ROOT).parts[:-1]}
        if path.suffix.lower() in forbidden_suffixes:
            raise AssertionError(f"Forbidden publication artifact: {path.relative_to(ROOT)}")
        if path.relative_to(ROOT).parts[0].lower() == "manuscript":
            raise AssertionError(f"Forbidden manuscript directory: {path.relative_to(ROOT)}")
        if relative_parts & forbidden_parts:
            raise AssertionError(f"Forbidden directory content: {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".log":
            raise AssertionError(f"Internal log retained: {path.relative_to(ROOT)}")


def main() -> int:
    contract = read_json(ROOT / "REFERENCE_RESULTS.json")
    validate_counts(contract)
    validate_primary(contract)
    validate_prototype(contract)
    validate_scope()
    print("PASS: open-source package matches the reference result contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
