#!/usr/bin/env python3
"""
Compute GCV (GlyContact Vector) Features for Expanded Dataset (342 glycans)

This script computes GCV features (contact graph based) for all structures
in the expanded dataset.

Output:
    data/gcv/expanded_v1/gcv_features.csv
    data/gcv/expanded_v1/coverage_flags.csv
    reports/gcv_expanded_v1_summary.md
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
from typing import Optional, Dict, List, Set
import numpy as np

warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

from scipy.spatial.distance import cdist

# Configuration
BASE_PATH = Path("/home/minrui/glyco/public_glyco_mirror")

STRUCTURE_DIRS = [
    BASE_PATH / "data/raw/structures/targeted_sugarbind",
    BASE_PATH / "data/raw/structures/carbogrove_glycoshape",
]

OUTPUT_DIR = BASE_PATH / "data/gcv/expanded_v1"
REPORTS_DIR = BASE_PATH / "reports"
LOGS_DIR = BASE_PATH / "logs"

# GCV computation parameters
CONTACT_CUTOFF = 4.5  # Angstroms
NEIGHBOR_CUTOFF = 4.0  # Angstroms for neighbor count
LONG_RANGE_SEQ_DIST = 3  # Min sequence distance for "long-range"


@dataclass
class GCVFeatures:
    """GlyContact Vector features."""
    glytoucan_id: str
    contact_density: float
    long_range_contact_fraction: float
    mean_residue_neighbor_count: float
    sd_residue_neighbor_count: float
    torsion_diversity: float
    graph_laplacian_spectral_gap: float
    core_periphery_ratio: float
    max_contact_distance_seq: int
    n_residues: int
    n_contacts: int
    source_dir: str
    source_file: str


def setup_logging():
    """Configure logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "gcv_expanded_v1.log"

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
    """Find all valid structures across directories."""
    structures = {}

    for struct_dir in STRUCTURE_DIRS:
        if not struct_dir.exists():
            continue

        source_name = struct_dir.name

        for glycan_dir in struct_dir.iterdir():
            if not glycan_dir.is_dir():
                continue

            glytoucan_id = glycan_dir.name

            if glytoucan_id in structures:
                continue

            pdb_file = glycan_dir / "structure_1.pdb"
            if not pdb_file.exists() or pdb_file.stat().st_size < 100:
                continue

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


def parse_structure(filepath: Path):
    """Parse PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        return parser.get_structure('glycan', str(filepath))
    except:
        return None


def get_residue_centers(structure) -> Dict[str, np.ndarray]:
    """Get center of mass for each residue."""
    residue_centers = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = f"{chain.id}_{residue.id[1]}_{residue.resname}"
                coords = [atom.coord for atom in residue]
                if coords:
                    residue_centers[res_id] = np.mean(coords, axis=0)
    return residue_centers


def build_contact_graph(residue_centers: Dict[str, np.ndarray],
                       cutoff: float = CONTACT_CUTOFF) -> Dict[str, Set[str]]:
    """Build contact graph based on residue distances."""
    contacts = defaultdict(set)
    res_ids = list(residue_centers.keys())
    n = len(res_ids)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(residue_centers[res_ids[i]] - residue_centers[res_ids[j]])
            if dist < cutoff:
                contacts[res_ids[i]].add(res_ids[j])
                contacts[res_ids[j]].add(res_ids[i])

    return contacts


def compute_residue_neighbor_counts(structure, cutoff: float = NEIGHBOR_CUTOFF) -> List[float]:
    """Compute neighbor counts for each residue."""
    residue_atoms = defaultdict(list)
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = f"{chain.id}_{residue.id[1]}_{residue.resname}"
                for atom in residue:
                    residue_atoms[res_id].append(atom.coord)

    neighbor_counts = []
    all_residues = list(residue_atoms.keys())

    for res_id in all_residues:
        res_coords = np.array(residue_atoms[res_id])
        neighbor_count = 0

        for other_id in all_residues:
            if other_id == res_id:
                continue
            other_coords = np.array(residue_atoms[other_id])

            for coord in res_coords:
                dists = np.linalg.norm(other_coords - coord, axis=1)
                neighbor_count += np.sum(dists < cutoff)

        neighbor_counts.append(neighbor_count)

    return neighbor_counts


def compute_torsion_diversity(structure) -> float:
    """Compute conformational diversity proxy."""
    residue_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                coords = [atom.coord for atom in residue]
                if coords:
                    residue_coords.append(np.mean(coords, axis=0))

    if len(residue_coords) < 2:
        return 0.0

    coords_array = np.array(residue_coords)
    dists = cdist(coords_array, coords_array)
    upper_tri = dists[np.triu_indices(len(dists), k=1)]

    if len(upper_tri) == 0:
        return 0.0

    mean_dist = np.mean(upper_tri)
    if mean_dist == 0:
        return 0.0

    return np.std(upper_tri) / mean_dist


def compute_graph_spectral_gap(contacts: Dict[str, Set[str]], n_nodes: int) -> float:
    """Compute spectral gap of contact graph Laplacian."""
    if n_nodes == 0:
        return 0.0

    node_ids = list(contacts.keys())
    if len(node_ids) < 2:
        return 0.0

    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    adj = np.zeros((n, n))

    for node, neighbors in contacts.items():
        if node not in idx_map:
            continue
        i = idx_map[node]
        for neighbor in neighbors:
            if neighbor in idx_map:
                j = idx_map[neighbor]
                adj[i, j] = 1
                adj[j, i] = 1

    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj

    try:
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)
        if len(eigenvalues) >= 2:
            return float(eigenvalues[1])
        return 0.0
    except:
        return 0.0


def compute_core_periphery_ratio(contacts: Dict[str, Set[str]]) -> float:
    """Compute ratio of core to peripheral residues."""
    if not contacts:
        return 0.5

    degrees = [len(neighbors) for neighbors in contacts.values()]
    if not degrees:
        return 0.5

    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)

    if std_deg == 0:
        return 0.5

    core_count = sum(1 for d in degrees if d >= mean_deg + 0.5 * std_deg)
    periph_count = sum(1 for d in degrees if d <= mean_deg - 0.5 * std_deg)

    total = core_count + periph_count
    if total == 0:
        return 0.5

    return core_count / total


def get_sequence_distance(res_id1: str, res_id2: str) -> int:
    """Get sequence distance between residues."""
    try:
        parts1 = res_id1.split('_')
        parts2 = res_id2.split('_')
        seq1 = int(parts1[1])
        seq2 = int(parts2[1])
        return abs(seq2 - seq1)
    except:
        return 0


def compute_gcv_features(structure, filepath: Path, glytoucan_id: str,
                        source_dir: str) -> Optional[GCVFeatures]:
    """Compute all GCV features."""
    try:
        residue_centers = get_residue_centers(structure)
        n_residues = len(residue_centers)

        if n_residues == 0:
            return None

        contacts = build_contact_graph(residue_centers)
        n_contacts = sum(len(neighbors) for neighbors in contacts.values()) // 2

        # Contact density
        contact_density = n_contacts / n_residues if n_residues > 0 else 0.0

        # Long-range contacts
        long_range_contacts = 0
        for res1, neighbors in contacts.items():
            for res2 in neighbors:
                if get_sequence_distance(res1, res2) >= LONG_RANGE_SEQ_DIST:
                    long_range_contacts += 1
        long_range_contacts //= 2

        long_range_fraction = long_range_contacts / n_contacts if n_contacts > 0 else 0.0

        # Neighbor counts
        neighbor_counts = compute_residue_neighbor_counts(structure)
        mean_neighbor = np.mean(neighbor_counts) if neighbor_counts else 0.0
        sd_neighbor = np.std(neighbor_counts) if neighbor_counts else 0.0

        # Torsion diversity
        torsion_div = compute_torsion_diversity(structure)

        # Spectral gap
        spectral_gap = compute_graph_spectral_gap(contacts, n_residues)

        # Core-periphery ratio
        cp_ratio = compute_core_periphery_ratio(contacts)

        # Max sequence distance
        max_seq_dist = 0
        for res1, neighbors in contacts.items():
            for res2 in neighbors:
                max_seq_dist = max(max_seq_dist, get_sequence_distance(res1, res2))

        return GCVFeatures(
            glytoucan_id=glytoucan_id,
            contact_density=round(contact_density, 4),
            long_range_contact_fraction=round(long_range_fraction, 4),
            mean_residue_neighbor_count=round(mean_neighbor, 3),
            sd_residue_neighbor_count=round(sd_neighbor, 3),
            torsion_diversity=round(torsion_div, 4),
            graph_laplacian_spectral_gap=round(spectral_gap, 4),
            core_periphery_ratio=round(cp_ratio, 4),
            max_contact_distance_seq=max_seq_dist,
            n_residues=n_residues,
            n_contacts=n_contacts,
            source_dir=source_dir,
            source_file=str(filepath.relative_to(BASE_PATH)),
        )
    except Exception as e:
        return None


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("GCV Feature Computation for Expanded Dataset")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find structures
    structures = find_all_structures()
    logger.info(f"Total structures found: {len(structures)}")

    by_source = defaultdict(int)
    for info in structures.values():
        by_source[info["source_dir"]] += 1
    for src, count in sorted(by_source.items()):
        logger.info(f"  - {src}: {count}")

    # Process
    all_features = []
    stats = {"ok": 0, "failed": 0}

    glycan_ids = sorted(structures.keys())
    for i, glytoucan_id in enumerate(glycan_ids):
        info = structures[glytoucan_id]
        pdb_path = info["pdb_path"]
        source_dir = info["source_dir"]

        structure = parse_structure(pdb_path)
        if structure is None:
            stats["failed"] += 1
            continue

        features = compute_gcv_features(structure, pdb_path, glytoucan_id, source_dir)
        if features is None:
            stats["failed"] += 1
            continue

        all_features.append(asdict(features))
        stats["ok"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(glycan_ids)} glycans")

    # Save
    logger.info("\nSaving GCV features...")
    gcv_path = OUTPUT_DIR / "gcv_features.csv"

    fieldnames = ['glytoucan_id', 'contact_density', 'long_range_contact_fraction',
                  'mean_residue_neighbor_count', 'sd_residue_neighbor_count',
                  'torsion_diversity', 'graph_laplacian_spectral_gap',
                  'core_periphery_ratio', 'max_contact_distance_seq',
                  'n_residues', 'n_contacts', 'source_dir', 'source_file']

    with open(gcv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_features:
            writer.writerow(row)

    logger.info(f"Saved: {gcv_path}")

    # Compute statistics
    feature_cols = ['contact_density', 'long_range_contact_fraction',
                   'mean_residue_neighbor_count', 'sd_residue_neighbor_count',
                   'torsion_diversity', 'graph_laplacian_spectral_gap',
                   'core_periphery_ratio', 'max_contact_distance_seq']

    feature_stats = {}
    for field in feature_cols:
        values = [f[field] for f in all_features if f.get(field) is not None]
        if values:
            feature_stats[field] = {
                'min': round(min(values), 4),
                'max': round(max(values), 4),
                'mean': round(np.mean(values), 4),
                'std': round(np.std(values), 4),
                'median': round(np.median(values), 4),
            }

    # Generate report
    report_path = REPORTS_DIR / "gcv_expanded_v1_summary.md"
    with open(report_path, 'w') as f:
        f.write("# GCV Feature Computation Summary (Expanded Dataset v1)\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("---\n\n")

        f.write("## Coverage Summary\n\n")
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| OK | {stats['ok']} |\n")
        f.write(f"| Failed | {stats['failed']} |\n")
        f.write(f"| **Total** | **{stats['ok'] + stats['failed']}** |\n\n")

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

        f.write("\n---\n\n")
        f.write("## Feature Descriptions\n\n")
        f.write("- **contact_density**: Inter-residue contacts / number of residues\n")
        f.write("- **long_range_contact_fraction**: Fraction of contacts between residues >= 3 apart\n")
        f.write("- **mean_residue_neighbor_count**: Mean neighboring atoms from other residues\n")
        f.write("- **sd_residue_neighbor_count**: Std dev of neighbor counts\n")
        f.write("- **torsion_diversity**: CV of pairwise residue distances (conformational diversity)\n")
        f.write("- **graph_laplacian_spectral_gap**: Second smallest eigenvalue (graph connectivity)\n")
        f.write("- **core_periphery_ratio**: Ratio of high-degree to low-degree residues\n")
        f.write("- **max_contact_distance_seq**: Max sequence distance between contacting residues\n")

    logger.info(f"Saved: {report_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("GCV COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Glycans processed: {stats['ok']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - {gcv_path}")
    logger.info(f"  - {report_path}")

    return stats


if __name__ == "__main__":
    main()
