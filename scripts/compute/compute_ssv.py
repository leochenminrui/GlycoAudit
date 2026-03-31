#!/usr/bin/env python3
"""
Compute SSV v0 Features for Expanded Dataset (342 glycans)

This script computes SSV features for all structures in the expanded dataset:
- data/raw/structures/targeted_sugarbind/ (125 glycans)
- data/raw/structures/carbogrove_glycoshape/ (217 glycans)

Output:
    data/ssv/expanded_v1/ssv_features.csv
    data/ssv/expanded_v1/coverage_flags.csv
    reports/ssv_expanded_v1_summary.md
"""

import csv
import json
import logging
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
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")

STRUCTURE_DIRS = [
    BASE_PATH / "data/raw/structures/targeted_sugarbind",
    BASE_PATH / "data/raw/structures/carbogrove_glycoshape",
]

OUTPUT_DIR = BASE_PATH / "data/ssv/expanded_v1"
REPORTS_DIR = BASE_PATH / "reports"
LOGS_DIR = BASE_PATH / "logs"

NEIGHBOR_CUTOFF = 4.0  # Angstroms for exposure calculation
SAMPLE_SIZE = 500  # Max atoms for pairwise distance sampling


@dataclass
class SSVFeatures:
    """SSV v0 feature set for a single structure."""
    glytoucan_id: str
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
    log_file = LOGS_DIR / "ssv_expanded_v1.log"

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


def parse_structure(filepath: Path, logger) -> Optional[object]:
    """Parse PDB file, return structure object."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('glycan', str(filepath))
        return structure
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None


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

    indices = np.random.choice(n, size=sample_size, replace=False)
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
    for model in structure:
        for chain in model:
            for residue in chain:
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
        sample_indices = np.random.choice(n, size=1000, replace=False)
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
                         source_dir: str, logger) -> Optional[SSVFeatures]:
    """Compute all SSV v0 features for a structure."""
    try:
        atoms = list(structure.get_atoms())
        n_atoms = len(atoms)

        if n_atoms == 0:
            logger.warning(f"No atoms in {filepath}")
            return None

        residues = set()
        for model in structure:
            for chain in model:
                for residue in chain:
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
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("SSV v0 Computation for Expanded Dataset")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all structures
    structures = find_all_structures()
    logger.info(f"Total structures found: {len(structures)}")

    by_source = defaultdict(int)
    for info in structures.values():
        by_source[info["source_dir"]] += 1
    for src, count in sorted(by_source.items()):
        logger.info(f"  - {src}: {count}")

    # Process each structure
    all_features = []
    coverage_flags = []
    stats = {"ok": 0, "failed": 0}

    glycan_ids = sorted(structures.keys())
    for i, glytoucan_id in enumerate(glycan_ids):
        info = structures[glytoucan_id]
        pdb_path = info["pdb_path"]
        source_dir = info["source_dir"]

        structure = parse_structure(pdb_path, logger)
        if structure is None:
            coverage_flags.append(CoverageFlag(
                glytoucan_id=glytoucan_id,
                status="FAILED",
                failure_reason="parse_error",
                source_dir=source_dir
            ))
            stats["failed"] += 1
            continue

        features = compute_ssv_features(structure, pdb_path, glytoucan_id, source_dir, logger)
        if features is None:
            coverage_flags.append(CoverageFlag(
                glytoucan_id=glytoucan_id,
                status="FAILED",
                failure_reason="feature_computation_error",
                source_dir=source_dir
            ))
            stats["failed"] += 1
            continue

        all_features.append(asdict(features))
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

    fieldnames = ['glytoucan_id', 'n_atoms', 'n_residues', 'radius_of_gyration',
                  'max_pair_distance', 'compactness', 'branch_proxy', 'terminal_proxy',
                  'exposure_proxy', 'source_dir', 'source_file']

    with open(ssv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_features:
            writer.writerow(row)

    logger.info(f"Saved: {ssv_path}")

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
    for field in ['n_atoms', 'n_residues', 'radius_of_gyration', 'max_pair_distance',
                  'compactness', 'branch_proxy', 'terminal_proxy', 'exposure_proxy']:
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
    report_path = REPORTS_DIR / "ssv_expanded_v1_summary.md"
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
    logger.info(f"  - {flags_path}")
    logger.info(f"  - {report_path}")

    return stats


if __name__ == "__main__":
    main()
