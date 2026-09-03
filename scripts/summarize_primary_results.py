#!/usr/bin/env python3
"""Rebuild the reported summary tables from the public dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from glycoaudit.evaluation import evaluate_positive_kfold, holm_adjust


REPRESENTATIONS = {
    "SSV": ("ssv", "standardize"),
    "GCV 4.5 A": ("gcv_4_5", "standardize"),
    "WURCS fold-TFIDF": ("wurcs", "tfidf"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Public data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/primary_summary",
        help="Directory for regenerated summary tables.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"glytoucan_id", "model_id", "source_dir", "source_file"}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_primary_summary(root: Path) -> pd.DataFrame:
    rows = []
    for label, (directory, _) in REPRESENTATIONS.items():
        metrics = read_json(root / "results" / directory / "aggregate_metrics.json")
        pvalues = read_json(root / "results" / directory / "null_pvalues.json")
        rows.append(
            {
                "representation": label,
                "n_glycans": metrics["n_glycans"],
                "n_agents": metrics["n_agents_evaluated"],
                "n_agent_folds": metrics["n_agent_folds"],
                "mrr": metrics["mrr"],
                "mrr_ci_lower": metrics["mrr_ci_lower"],
                "mrr_ci_upper": metrics["mrr_ci_upper"],
                "recall_at_5": metrics["recall@5"],
                "recall_at_5_ci_lower": metrics["recall@5_ci_lower"],
                "recall_at_5_ci_upper": metrics["recall@5_ci_upper"],
                "known_positive_ap": metrics["known_positive_ap"],
                "known_positive_ap_ci_lower": metrics["known_positive_ap_ci_lower"],
                "known_positive_ap_ci_upper": metrics["known_positive_ap_ci_upper"],
                "mrr_p_vs_random": pvalues["mrr_vs_random"],
                "mrr_p_vs_feature_permutation": pvalues[
                    "mrr_vs_feature_permutation"
                ],
                "recall_at_5_p_vs_random": pvalues["recall@5_vs_random"],
                "recall_at_5_p_vs_feature_permutation": pvalues[
                    "recall@5_vs_feature_permutation"
                ],
                "null_repeats": pvalues["n_repeats"],
            }
        )
    frame = pd.DataFrame(rows)
    family_columns = [
        "mrr_p_vs_feature_permutation",
        "recall_at_5_p_vs_feature_permutation",
    ]
    family = [
        float(row[column])
        for _, row in frame.iterrows()
        for column in family_columns
    ]
    adjusted = holm_adjust(family)
    index = 0
    for row_index in frame.index:
        for column in family_columns:
            adjusted_column = column.replace("_p_", "_holm_p_")
            frame.loc[row_index, adjusted_column] = adjusted[index]
            index += 1
    return frame


def build_null_summary(root: Path) -> pd.DataFrame:
    rows = []
    for label, (directory, _) in REPRESENTATIONS.items():
        frame = pd.read_csv(root / "results" / directory / "null_distributions.csv")
        for mode, group in frame.groupby("mode", sort=True):
            rows.append(
                {
                    "representation": label,
                    "null_mode": mode,
                    "n_repeats": len(group),
                    "mrr_mean": group["mrr"].mean(),
                    "mrr_sd": group["mrr"].std(ddof=1),
                    "recall_at_5_mean": group["recall@5"].mean(),
                    "recall_at_5_sd": group["recall@5"].std(ddof=1),
                    "known_positive_ap_mean": group["known_positive_ap"].mean(),
                    "known_positive_ap_sd": group["known_positive_ap"].std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def build_support_strata(root: Path) -> pd.DataFrame:
    rows = []
    order = {"2": 0, "3-5": 1, "6-10": 2, ">10": 3}
    for label, (directory, _) in REPRESENTATIONS.items():
        frame = pd.read_csv(root / "results" / directory / "metrics_per_agent.csv")
        for stratum, group in frame.groupby("positive_support_stratum", sort=False):
            rows.append(
                {
                    "representation": label,
                    "support_stratum": stratum,
                    "support_order": order[str(stratum)],
                    "n_agents": len(group),
                    "mrr": group["mrr"].mean(),
                    "recall_at_5": group["recall@5"].mean(),
                    "known_positive_ap": group["known_positive_ap"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["representation", "support_order"])


def build_minimum_support_summary(root: Path) -> pd.DataFrame:
    rows = []
    for label, (directory, _) in REPRESENTATIONS.items():
        for minimum in (2, 3, 5):
            suffix = "" if minimum == 2 else f"_min{minimum}"
            metrics = read_json(
                root / "results" / f"{directory}{suffix}" / "aggregate_metrics.json"
            )
            rows.append(
                {
                    "representation": label,
                    "minimum_positive_support": minimum,
                    "n_agents": metrics["n_agents_evaluated"],
                    "n_agent_folds": metrics["n_agent_folds"],
                    "mrr": metrics["mrr"],
                    "recall_at_5": metrics["recall@5"],
                    "known_positive_ap": metrics["known_positive_ap"],
                }
            )
    return pd.DataFrame(rows)


def build_gcv_cutoff_summary(root: Path) -> pd.DataFrame:
    rows = []
    for cutoff, directory in [(4.0, "gcv_4_0"), (4.5, "gcv_4_5"), (5.0, "gcv_5_0")]:
        metrics = read_json(root / "results" / directory / "aggregate_metrics.json")
        rows.append(
            {
                "contact_cutoff_angstrom": cutoff,
                "n_agents": metrics["n_agents_evaluated"],
                "mrr": metrics["mrr"],
                "mrr_ci_lower": metrics["mrr_ci_lower"],
                "mrr_ci_upper": metrics["mrr_ci_upper"],
                "recall_at_5": metrics["recall@5"],
                "recall_at_5_ci_lower": metrics["recall@5_ci_lower"],
                "recall_at_5_ci_upper": metrics["recall@5_ci_upper"],
                "known_positive_ap": metrics["known_positive_ap"],
            }
        )
    return pd.DataFrame(rows)


def first_existing(root: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    choices = ", ".join(relative_paths)
    raise FileNotFoundError(f"None of the required paths exists under {root}: {choices}")


def build_model_sensitivity(root: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        first_existing(root, "evidence_records.csv", "raw/sugarbind/evidence_records.csv")
    )
    fold_manifest = pd.read_csv(
        first_existing(
            root,
            "positive_holdout_folds.csv",
            "derived/manifests/positive_holdout_folds.csv",
        )
    )
    candidates = pd.read_csv(
        first_existing(
            root,
            "candidate_manifest.csv",
            "derived/manifests/candidate_manifest.csv",
        )
    )
    candidate_ids = set(
        candidates.loc[
            candidates["included_in_universe"].astype(str).str.lower().isin(
                {"1", "true", "yes"}
            ),
            "glytoucan_id",
        ].astype(str)
    )
    input_tables = {
        "SSV": first_existing(
            root,
            "features/ssv_per_model.csv",
            "derived/features/ssv/ssv_per_model.csv",
        ),
        "GCV 4.5 A": first_existing(
            root,
            "features/gcv_4_5_per_model.csv",
            "derived/features/gcv_4_5/gcv_per_model.csv",
        ),
    }
    rows = []
    for label, path in input_tables.items():
        frame = pd.read_csv(path)
        frame = frame[frame["glytoucan_id"].astype(str).isin(candidate_ids)]
        columns = feature_columns(frame)
        for model_id, model_frame in frame.groupby("model_id", sort=True):
            result = evaluate_positive_kfold(
                model_frame[["glytoucan_id", *columns]],
                labels,
                feature_columns=columns,
                n_splits=5,
                minimum_positives=2,
                ks=(5, 10),
                seed=42,
                n_bootstrap=0,
                preprocessing="standardize",
                fold_manifest=fold_manifest,
            )
            rows.append(
                {
                    "representation": label,
                    "model_id": int(model_id),
                    "n_glycans": result.aggregate["n_glycans"],
                    "n_agents": result.aggregate["n_agents_evaluated"],
                    "mrr": result.aggregate["mrr"],
                    "recall_at_5": result.aggregate["recall@5"],
                    "known_positive_ap": result.aggregate["known_positive_ap"],
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = args.release_dir.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "primary_results.csv": build_primary_summary(root),
        "null_controls.csv": build_null_summary(root),
        "support_strata.csv": build_support_strata(root),
        "minimum_support_sensitivity.csv": build_minimum_support_summary(root),
        "gcv_cutoff_sensitivity.csv": build_gcv_cutoff_summary(root),
        "model_sensitivity.csv": build_model_sensitivity(root),
    }
    for name, frame in outputs.items():
        frame.to_csv(outdir / name, index=False)

    summary = {
        "dataset": "glycoaudit_data",
        "status": "complete",
        "source_scope": {
            "quantitative_evidence": ["SugarBind"],
            "metadata_only": ["CFG experiment catalogue"],
            "cfg_quantitative_status": "unrecoverable_original_rfu_and_thresholds",
        },
        "outputs": sorted(outputs),
    }
    (outdir / "summary_status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(outputs)} summary tables to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
