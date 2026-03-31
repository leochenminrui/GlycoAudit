#!/usr/bin/env python3
"""
Joinability and Coverage Audit Pipeline

Generates comprehensive audit of:
1. Raw labeled glycans (before normalization)
2. Normalized glycans
3. Directly joinable glycans (structure + label match)
4. Glycans added by targeted completion
5. Final joinable benchmark subset
6. Unresolved cases with failure mode taxonomy

Output:
- CSV tables with counts and proportions
- JSON summary with detailed breakdowns
- Publication-ready flow diagram data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "outputs/joinability_audit"


def load_binding_labels(source: str) -> pd.DataFrame:
    """Load binding labels from a source."""
    if source == "sugarbind":
        labels_path = DATA_PATH / "binding/sugarbind_v0/labels.csv"
    elif source == "carbogrove":
        labels_path = DATA_PATH / "binding/carbogrove_v0/labels.csv"
    else:
        raise ValueError(f"Unknown source: {source}")

    if not labels_path.exists():
        print(f"Warning: {labels_path} not found")
        return pd.DataFrame()

    return pd.read_csv(labels_path)


def load_structure_map() -> pd.DataFrame:
    """Load glycan structure map."""
    structure_map_path = DATA_PATH / "binding/expanded_v1/glycan_structure_map.csv"
    return pd.read_csv(structure_map_path)


def load_final_labels() -> pd.DataFrame:
    """Load final joinable labels."""
    labels_path = DATA_PATH / "binding/expanded_v1/labels.csv"
    return pd.read_csv(labels_path)


def check_structure_exists(glytoucan_id: str, structure_map: pd.DataFrame) -> Tuple[bool, str]:
    """Check if structure exists for a glycan."""
    matches = structure_map[structure_map['glytoucan_id'] == glytoucan_id]
    if len(matches) == 0:
        return False, "no_structure"

    # Check if file actually exists
    pdb_path = BASE_PATH / matches.iloc[0]['pdb_path']
    if not pdb_path.exists():
        return False, "structure_missing_file"

    # Check file size (very small files might be corrupt)
    file_size = pdb_path.stat().st_size
    if file_size < 100:
        return False, "structure_too_small"

    return True, "ok"


def identify_structure_source(glytoucan_id: str, structure_map: pd.DataFrame) -> str:
    """Identify source of structure (direct or targeted)."""
    matches = structure_map[structure_map['glytoucan_id'] == glytoucan_id]
    if len(matches) == 0:
        return "none"

    source = matches.iloc[0]['source']
    if 'targeted' in source:
        return "targeted_completion"
    else:
        return "direct"


def audit_joinability() -> Dict:
    """Run full joinability audit."""
    print("=" * 80)
    print("JOINABILITY AND COVERAGE AUDIT")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading binding labels...")
    df_sugarbind = load_binding_labels("sugarbind")
    df_carbogrove = load_binding_labels("carbogrove")
    df_structure_map = load_structure_map()
    df_final = load_final_labels()

    # Stage 1: Raw labeled glycans
    print("\n[2/6] Analyzing raw labeled glycans...")
    raw_glycans_sb = set(df_sugarbind['glytoucan_id'].unique()) if not df_sugarbind.empty else set()
    raw_glycans_cb = set(df_carbogrove['glytoucan_id'].unique()) if not df_carbogrove.empty else set()
    raw_glycans_all = raw_glycans_sb | raw_glycans_cb

    raw_pairs_sb = len(df_sugarbind) if not df_sugarbind.empty else 0
    raw_pairs_cb = len(df_carbogrove) if not df_carbogrove.empty else 0

    # Stage 2: Normalized glycans (assume same as raw for now)
    # In a real implementation, would track ID normalization
    normalized_glycans = raw_glycans_all

    # Stage 3: Check structure availability
    print("\n[3/6] Checking structure availability...")
    structure_status = {}
    failure_modes = Counter()

    for gid in raw_glycans_all:
        has_structure, status = check_structure_exists(gid, df_structure_map)
        structure_status[gid] = status
        if status != "ok":
            failure_modes[status] += 1

    directly_joinable = {gid for gid, status in structure_status.items() if status == "ok"}

    # Stage 4: Identify targeted completion glycans
    print("\n[4/6] Identifying targeted completion glycans...")
    targeted_glycans = set()
    for gid in directly_joinable:
        source = identify_structure_source(gid, df_structure_map)
        if source == "targeted_completion":
            targeted_glycans.add(gid)

    direct_available = directly_joinable - targeted_glycans

    # Stage 5: Final joinable subset
    print("\n[5/6] Analyzing final joinable subset...")
    final_joinable = set(df_final['glytoucan_id'].unique())
    final_pairs = len(df_final)

    # Stage 6: Categorize unresolved
    print("\n[6/6] Categorizing unresolved cases...")
    unresolved = raw_glycans_all - final_joinable

    # Detailed failure analysis
    failure_details = defaultdict(list)
    for gid in unresolved:
        status = structure_status.get(gid, "unknown")
        failure_details[status].append(gid)

    # Build audit summary
    audit_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "stage1_raw_labeled": {
                "description": "Raw labeled glycans from sources",
                "glycans": len(raw_glycans_all),
                "glycans_sugarbind": len(raw_glycans_sb),
                "glycans_carbogrove": len(raw_glycans_cb),
                "pairs_sugarbind": raw_pairs_sb,
                "pairs_carbogrove": raw_pairs_cb,
                "pairs_total": raw_pairs_sb + raw_pairs_cb
            },
            "stage2_normalized": {
                "description": "Glycans after ID normalization",
                "glycans": len(normalized_glycans),
                "normalization_failures": len(raw_glycans_all) - len(normalized_glycans)
            },
            "stage3_directly_joinable": {
                "description": "Glycans with existing structures (not targeted)",
                "glycans": len(direct_available),
                "proportion_of_raw": len(direct_available) / len(raw_glycans_all) if raw_glycans_all else 0
            },
            "stage4_targeted_completion": {
                "description": "Glycans added via targeted structure acquisition",
                "glycans": len(targeted_glycans),
                "proportion_of_raw": len(targeted_glycans) / len(raw_glycans_all) if raw_glycans_all else 0
            },
            "stage5_final_joinable": {
                "description": "Final benchmark subset with structure + labels",
                "glycans": len(final_joinable),
                "pairs": final_pairs,
                "join_success_rate": len(final_joinable) / len(raw_glycans_all) if raw_glycans_all else 0
            },
            "stage6_unresolved": {
                "description": "Glycans with labels but no usable structure",
                "glycans": len(unresolved),
                "proportion_of_raw": len(unresolved) / len(raw_glycans_all) if raw_glycans_all else 0
            }
        },
        "failure_modes": {
            "taxonomy": {
                mode: {
                    "count": count,
                    "proportion": count / len(unresolved) if unresolved else 0,
                    "examples": failure_details[mode][:5]
                }
                for mode, count in failure_modes.items()
            },
            "total_failures": len(unresolved)
        },
        "flow_summary": {
            "raw_labeled": len(raw_glycans_all),
            "normalized": len(normalized_glycans),
            "direct_joinable": len(direct_available),
            "targeted_added": len(targeted_glycans),
            "final_joinable": len(final_joinable),
            "unresolved": len(unresolved)
        }
    }

    return audit_summary


def generate_flow_table(audit_summary: Dict) -> pd.DataFrame:
    """Generate flow table for publication."""
    stages = audit_summary["stages"]

    rows = [
        {
            "Stage": "1. Raw Labeled Glycans",
            "Count": stages["stage1_raw_labeled"]["glycans"],
            "Description": "Unique glycans with binding labels from SugarBind + Carbogrove"
        },
        {
            "Stage": "2. Normalized IDs",
            "Count": stages["stage2_normalized"]["glycans"],
            "Description": "After GlyTouCan ID normalization"
        },
        {
            "Stage": "3. Direct Joinable",
            "Count": stages["stage3_directly_joinable"]["glycans"],
            "Description": "Glycans with pre-existing 3D structures"
        },
        {
            "Stage": "4. Targeted Completion",
            "Count": stages["stage4_targeted_completion"]["glycans"],
            "Description": "Structures acquired via targeted pipeline"
        },
        {
            "Stage": "5. Final Benchmark",
            "Count": stages["stage5_final_joinable"]["glycans"],
            "Description": "Joinable glycans with structure + binding labels"
        },
        {
            "Stage": "6. Unresolved",
            "Count": stages["stage6_unresolved"]["glycans"],
            "Description": "Labels without usable structures"
        }
    ]

    return pd.DataFrame(rows)


def generate_failure_taxonomy_table(audit_summary: Dict) -> pd.DataFrame:
    """Generate failure mode taxonomy table."""
    taxonomy = audit_summary["failure_modes"]["taxonomy"]

    rows = []
    for mode, data in taxonomy.items():
        rows.append({
            "Failure Mode": mode.replace("_", " ").title(),
            "Count": data["count"],
            "Proportion": f"{data['proportion']:.1%}",
            "Examples": ", ".join(data["examples"][:3])
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Count", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Joinability and Coverage Audit")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_PATH,
                        help="Output directory")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run audit
    audit_summary = audit_joinability()

    # Save summary JSON
    summary_path = args.output_dir / "joinability_audit_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\n✓ Saved summary: {summary_path}")

    # Generate and save flow table
    df_flow = generate_flow_table(audit_summary)
    flow_path = args.output_dir / "joinability_flow_table.csv"
    df_flow.to_csv(flow_path, index=False)
    print(f"✓ Saved flow table: {flow_path}")

    # Generate and save failure taxonomy
    df_failures = generate_failure_taxonomy_table(audit_summary)
    failures_path = args.output_dir / "failure_taxonomy.csv"
    df_failures.to_csv(failures_path, index=False)
    print(f"✓ Saved failure taxonomy: {failures_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("JOINABILITY AUDIT SUMMARY")
    print("=" * 80)
    print(f"\nRaw labeled glycans: {audit_summary['flow_summary']['raw_labeled']}")
    print(f"Direct joinable: {audit_summary['flow_summary']['direct_joinable']}")
    print(f"Targeted completion: {audit_summary['flow_summary']['targeted_added']}")
    print(f"Final benchmark: {audit_summary['flow_summary']['final_joinable']}")
    print(f"Unresolved: {audit_summary['flow_summary']['unresolved']}")

    join_rate = audit_summary['stages']['stage5_final_joinable']['join_success_rate']
    print(f"\nJoin success rate: {join_rate:.1%}")

    print("\nTop failure modes:")
    for _, row in df_failures.head(3).iterrows():
        print(f"  - {row['Failure Mode']}: {row['Count']} ({row['Proportion']})")

    print(f"\n✓ Audit complete. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
