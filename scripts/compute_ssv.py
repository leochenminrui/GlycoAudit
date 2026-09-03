#!/usr/bin/env python3
"""
Compute Structural Shape Vector (SSV) features from an explicit structure set.

One or more structure roots must be supplied with ``--structure-dir``. Requiring
explicit inputs prevents an accidental rerun against historical local layouts.
"""

import csv
import argparse
import io
import json
import logging
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

# Suppress BioPython warnings
warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glycoaudit.config import get_project_root

BASE_PATH = get_project_root()

STRUCTURE_DIRS = []

OUTPUT_DIR = BASE_PATH / "outputs/features/ssv"
REPORTS_DIR = BASE_PATH / "reports"
LOGS_DIR = BASE_PATH / "logs"

NEIGHBOR_CUTOFF = 4.0  # Angstroms for exposure calculation
SAMPLE_SIZE = 500  # Max atoms for pairwise distance sampling
SAMPLE_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SSV descriptors from glycan structures")
    parser.add_argument(
        "--structure-dir",
        action="append",
        type=Path,
        dest="structure_dirs",
        required=True,
        help="Structure root containing one directory per GlyTouCan ID; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR)
    return parser.parse_args()


@dataclass
class SSVFeatures:
    """SSV v0 feature set for a single structure."""
    glytoucan_id: str
    model_id: int
    n_atoms: int
    n_residues: int
    radius_of_gyration: float
    max_pair_distance: float
    compactness: float
    branch_proxy: int
    terminal_proxy: int
    exposure_proxy: float
    source_dir: str
    source_file: str


@dataclass
class CoverageFlag:
    """Coverage status for a glycan."""
    glytoucan_id: str
    status: str  # OK or FAILED
    failure_reason: str
    source_dir: str


def setup_logging():
    """Configure logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "compute_ssv.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def find_all_structures() -> dict:
    """Find all valid structures across all structure directories."""
    structures = {}

    for struct_dir in STRUCTURE_DIRS:
        if not struct_dir.exists():
            continue

        source_name = struct_dir.name

        for glycan_dir in struct_dir.iterdir():
            if not glycan_dir.is_dir():
                continue

            glytoucan_id = glycan_dir.name

            # Skip if already found from higher-priority source
            if glytoucan_id in structures:
                continue

            # Check for PDB file
            pdb_file = glycan_dir / "structure_1.pdb"
            if not pdb_file.exists() or pdb_file.stat().st_size < 100:
                continue

            # Check meta.json for status (if exists)
            meta_file = glycan_dir / "meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    if meta.get("status") == "failed":
                        continue
                except:
                    pass

            structures[glytoucan_id] = {
                "pdb_path": pdb_file,
                "source_dir": source_name
            }

    return structures


def parse_models(filepath: Path, logger) -> list[object]:
    """Parse every explicit MODEL block, working around multi-model parser truncation."""
    try:
        parser = PDBParser(QUIET=True)
        text = filepath.read_text()
        blocks = re.findall(r"^MODEL\s+.*?^ENDMDL\s*$", text, flags=re.M | re.S)
        if not blocks:
            structure = parser.get_structure('glycan', str(filepath))
            return list(structure.get_models())
        models = []
        for index, block in enumerate(blocks, start=1):
            parsed = parser.get_structure(
                f'glycan_model_{index}', io.StringIO(block + "\nEND\n")
            )
            parsed_models = list(parsed.get_models())
            if parsed_models:
                models.append(parsed_models[0])
        return models
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return []


def get_atom_coords(structure) -> np.ndarray:
    """Extract all atom coordinates from structure."""
    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.coord)
    return np.array(coords)


def compute_radius_of_gyration(coords: np.ndarray) -> float:
    """Compute radius of gyration."""
    if len(coords) == 0:
        return 0.0
    center = coords.mean(axis=0)
    distances_sq = np.sum((coords - center) ** 2, axis=1)
    return np.sqrt(np.mean(distances_sq))


def compute_max_pair_distance(coords: np.ndarray, sample_size: int = SAMPLE_SIZE) -> float:
    """Compute maximum pairwise distance with sampling for large structures."""
    n = len(coords)
    if n <= 2:
        if n == 2:
            return np.linalg.norm(coords[0] - coords[1])
        return 0.0

    if n <= sample_size:
        max_dist = 0.0
        for i in range(n):
            dists = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
            if len(dists) > 0:
                max_dist = max(max_dist, dists.max())
        return max_dist

    rng = np.random.default_rng(SAMPLE_SEED)
    indices = rng.choice(n, size=sample_size, replace=False)
    sampled = coords[indices]
    max_dist = 0.0
    for i in range(len(sampled)):
        dists = np.linalg.norm(sampled[i+1:] - sampled[i], axis=1)
        if len(dists) > 0:
            max_dist = max(max_dist, dists.max())
    return max_dist


def build_residue_connectivity(structure) -> dict:
    """Build residue connectivity graph based on inter-residue bonds."""
    BOND_CUTOFF = 2.0

    residue_atoms = defaultdict(list)
    for residue in structure.get_residues():
        chain = residue.get_parent()
        res_id = f"{chain.id}_{residue.id[1]}_{residue.resname}"
        for atom in residue:
            residue_atoms[res_id].append(atom.coord)

    residue_ids = list(residue_atoms.keys())
    connectivity = defaultdict(set)

    for i, res_i in enumerate(residue_ids):
        coords_i = np.array(residue_atoms[res_i])
        for j, res_j in enumerate(residue_ids[i+1:], i+1):
            coords_j = np.array(residue_atoms[res_j])

            for ci in coords_i:
                dists = np.linalg.norm(coords_j - ci, axis=1)
                if np.any(dists < BOND_CUTOFF):
                    connectivity[res_i].add(res_j)
                    connectivity[res_j].add(res_i)
                    break

    return {k: list(v) for k, v in connectivity.items()}


def compute_exposure_proxy(coords: np.ndarray, cutoff: float = NEIGHBOR_CUTOFF) -> float:
    """Compute mean neighbor count within cutoff distance."""
    n = len(coords)
    if n <= 1:
        return 0.0

    if n > 1000:
        rng = np.random.default_rng(SAMPLE_SEED)
        sample_indices = rng.choice(n, size=1000, replace=False)
        sample_coords = coords[sample_indices]
    else:
        sample_coords = coords

    neighbor_counts = []
    for coord in sample_coords:
        dists = np.linalg.norm(coords - coord, axis=1)
        count = np.sum((dists > 0) & (dists < cutoff))
        neighbor_counts.append(count)

    return np.mean(neighbor_counts)


def compute_ssv_features(structure, filepath: Path, glytoucan_id: str,
                         source_dir: str, logger, model_id: int = 0) -> Optional[SSVFeatures]:
    """Compute all SSV v0 features for a structure."""
    try:
        atoms = list(structure.get_atoms())
        n_atoms = len(atoms)

        if n_atoms == 0:
            logger.warning(f"No atoms in {filepath}")
            return None

        residues = set()
        for residue in structure.get_residues():
            chain = residue.get_parent()
            res_id = (chain.id, residue.id[1], residue.resname)
            residues.add(res_id)
        n_residues = len(residues)

        coords = get_atom_coords(structure)

        rg = compute_radius_of_gyration(coords)
        max_dist = compute_max_pair_distance(coords)
        compactness = rg / max_dist if max_dist > 0 else 0.0

        connectivity = build_residue_connectivity(structure)

        branch_proxy = 0
        terminal_proxy = 0
        for res_id, neighbors in connectivity.items():
            degree = len(neighbors)
            if degree >= 3:
                branch_proxy += 1
            if degree == 1:
                terminal_proxy += 1

        if len(connectivity) == 0 and n_residues > 0:
            terminal_proxy = min(2, n_residues)
            branch_proxy = 0

        exposure = compute_exposure_proxy(coords)

        return SSVFeatures(
            glytoucan_id=glytoucan_id,
            model_id=model_id,
            n_atoms=n_atoms,
            n_residues=n_residues,
            radius_of_gyration=round(rg, 3),
            max_pair_distance=round(max_dist, 3),
            compactness=round(compactness, 4),
            branch_proxy=branch_proxy,
            terminal_proxy=terminal_proxy,
            exposure_proxy=round(exposure, 3),
            source_dir=source_dir,
            source_file=str(filepath.relative_to(BASE_PATH)),
        )

    except Exception as e:
        logger.error(f"Error computing features for {filepath}: {e}")
        return None


def main():
    global STRUCTURE_DIRS, OUTPUT_DIR, REPORTS_DIR, LOGS_DIR
    args = parse_args()
    if args.structure_dirs:
        STRUCTURE_DIRS = [path.resolve() for path in args.structure_dirs]
    OUTPUT_DIR = args.output_dir.resolve()
    REPORTS_DIR = args.reports_dir.resolve()
    LOGS_DIR = args.logs_dir.resolve()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("SSV v0 Computation for Expanded Dataset")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all structures
    structures = find_all_structures()
    logger.info(f"Total structures found: {len(structures)}")

    if not structures:
        logger.error("No valid structure files were found; no output tables were written")
        logger.error("Pass one or more --structure-dir values or install the frozen data release")
        return {"ok": 0, "failed": 0, "status": "no_structures"}

    by_source = defaultdict(int)
    for info in structures.values():
        by_source[info["source_dir"]] += 1
    for src, count in sorted(by_source.items()):
        logger.info(f"  - {src}: {count}")

    # Process each structure
    all_features = []
    per_model_features = []
    coverage_flags = []
    stats = {"ok": 0, "failed": 0}

    glycan_ids = sorted(structures.keys())
    for i, glytoucan_id in enumerate(glycan_ids):
        info = structures[glytoucan_id]
        pdb_path = info["pdb_path"]
        source_dir = info["source_dir"]

        models = parse_models(pdb_path, logger)
        if not models:
            coverage_flags.append(CoverageFlag(
                glytoucan_id=glytoucan_id,
                status="FAILED",
                failure_reason="parse_error",
                source_dir=source_dir
            ))
            stats["failed"] += 1
            continue

        model_rows = []
        for model_index, model in enumerate(models, start=1):
            model_features = compute_ssv_features(
                model,
                pdb_path,
                glytoucan_id,
                source_dir,
                logger,
                model_id=model_index,
            )
            if model_features is not None:
                model_rows.append(asdict(model_features))

        if not model_rows:
            coverage_flags.append(CoverageFlag(
                glytoucan_id=glytoucan_id,
                status="FAILED",
                failure_reason="feature_computation_error",
                source_dir=source_dir
            ))
            stats["failed"] += 1
            continue

        per_model_features.extend(model_rows)
        numeric_fields = [
            'n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
            'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy'
        ]
        aggregate_row = {
            'glytoucan_id': glytoucan_id,
            'n_models': len(model_rows),
            'source_dir': source_dir,
            'source_file': model_rows[0]['source_file'],
        }
        for field in numeric_fields:
            values = np.asarray([row[field] for row in model_rows], dtype=float)
            aggregate_row[field] = round(float(values.mean()), 6)
            aggregate_row[f'{field}_sd'] = round(float(values.std()), 6)
        all_features.append(aggregate_row)
        coverage_flags.append(CoverageFlag(
            glytoucan_id=glytoucan_id,
            status="OK",
            failure_reason="",
            source_dir=source_dir
        ))
        stats["ok"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(glycan_ids)} glycans")

    # Save features
    logger.info("\nSaving SSV features...")
    ssv_path = OUTPUT_DIR / "ssv_features.csv"

    numeric_fields = ['n_atoms', 'n_residues', 'radius_of_gyration',
                      'max_pair_distance', 'compactness', 'branch_proxy', 'terminal_proxy',
                      'exposure_proxy']
    fieldnames = ['glytoucan_id', 'n_models', *numeric_fields,
                  *[f'{field}_sd' for field in numeric_fields], 'source_dir', 'source_file']

    with open(ssv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_features:
            writer.writerow(row)

    logger.info(f"Saved: {ssv_path}")

    per_model_path = OUTPUT_DIR / "ssv_per_model.csv"
    per_model_fieldnames = ['glytoucan_id', 'model_id', *numeric_fields, 'source_dir', 'source_file']
    with open(per_model_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_model_fieldnames)
        writer.writeheader()
        writer.writerows(per_model_features)
    logger.info(f"Saved: {per_model_path}")

    # Save coverage flags
    flags_path = OUTPUT_DIR / "coverage_flags.csv"
    with open(flags_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['glytoucan_id', 'status', 'failure_reason', 'source_dir'])
        writer.writeheader()
        for flag in coverage_flags:
            writer.writerow(asdict(flag))

    logger.info(f"Saved: {flags_path}")

    # Compute statistics
    feature_stats = {}
    for field in numeric_fields:
        values = [f[field] for f in all_features if f.get(field) is not None]
        if values:
            feature_stats[field] = {
                'min': round(min(values), 3),
                'max': round(max(values), 3),
                'mean': round(np.mean(values), 3),
                'std': round(np.std(values), 3),
                'median': round(np.median(values), 3),
            }

    # Generate report
    report_path = REPORTS_DIR / "ssv_summary.md"
    with open(report_path, 'w') as f:
        f.write("# SSV v0 Computation Summary (Expanded Dataset v1)\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("---\n\n")

        f.write("## Coverage Summary\n\n")
        f.write("| Metric | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        total = stats['ok'] + stats['failed']
        f.write(f"| OK | {stats['ok']} | {100*stats['ok']/total:.1f}% |\n")
        f.write(f"| Failed | {stats['failed']} | {100*stats['failed']/total:.1f}% |\n")
        f.write(f"| **Total** | **{total}** | 100% |\n\n")

        f.write("## By Source\n\n")
        f.write("| Source | Count |\n")
        f.write("|--------|-------|\n")
        for src, count in sorted(by_source.items()):
            f.write(f"| {src} | {count} |\n")

        f.write("\n---\n\n")
        f.write("## Feature Distributions\n\n")
        f.write("Each glycan row is the mean across genuine models in the GlycoShape level-1 multi-model PDB; matching `_sd` columns and `ssv_per_model.csv` preserve ensemble dispersion.\n\n")
        f.write("| Feature | Min | Max | Mean | Std | Median |\n")
        f.write("|---------|-----|-----|------|-----|--------|\n")
        for field, s in feature_stats.items():
            f.write(f"| {field} | {s['min']} | {s['max']} | {s['mean']} | {s['std']} | {s['median']} |\n")

    logger.info(f"Saved: {report_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SSV v0 COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Glycans processed: {stats['ok']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - {ssv_path}")
    logger.info(f"  - {per_model_path}")
    logger.info(f"  - {flags_path}")
    logger.info(f"  - {report_path}")

    return stats


if __name__ == "__main__":
    result = main()
    raise SystemExit(1 if result.get("status") == "no_structures" else 0)
