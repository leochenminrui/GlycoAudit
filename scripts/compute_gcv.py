#!/usr/bin/env python3
"""
Compute Glycan Contact Vector (GCV) features from an explicit structure set.

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
from typing import Optional, Dict, List, Set
import numpy as np

warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

from scipy.spatial.distance import cdist

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from glycoaudit.config import get_project_root

BASE_PATH = get_project_root()

STRUCTURE_DIRS = []

OUTPUT_DIR = BASE_PATH / "outputs/features/gcv"
REPORTS_DIR = BASE_PATH / "reports"
LOGS_DIR = BASE_PATH / "logs"

# GCV computation parameters
CONTACT_CUTOFF = 4.5  # Angstroms
NEIGHBOR_CUTOFF = 4.0  # Angstroms for neighbor count
LONG_RANGE_SEQ_DIST = 3  # Min sequence distance for "long-range"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute GCV contact-graph descriptors")
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
    parser.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF)
    return parser.parse_args()


@dataclass
class GCVFeatures:
    """GlyContact Vector features."""
    glytoucan_id: str
    model_id: int
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
    log_file = LOGS_DIR / "compute_gcv.log"

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


def parse_models(filepath: Path) -> list[object]:
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
    except:
        return []


def get_residue_centers(structure) -> Dict[str, np.ndarray]:
    """Get center of mass for each residue."""
    residue_centers = {}
    for residue in structure.get_residues():
        chain = residue.get_parent()
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
    for residue in structure.get_residues():
        chain = residue.get_parent()
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
    """Return the CV of pairwise residue-centre distances (legacy field name)."""
    residue_coords = []
    for residue in structure.get_residues():
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
                        source_dir: str, model_id: int = 0,
                        contact_cutoff: float = CONTACT_CUTOFF) -> Optional[GCVFeatures]:
    """Compute all GCV features."""
    try:
        residue_centers = get_residue_centers(structure)
        n_residues = len(residue_centers)

        if n_residues == 0:
            return None

        contacts = build_contact_graph(residue_centers, cutoff=contact_cutoff)
        n_contacts = sum(len(neighbors) for neighbors in contacts.values()) // 2

        # Contacts per residue (legacy field name: contact_density).
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

        # Pairwise residue-centre distance CV (legacy field: torsion_diversity).
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
            model_id=model_id,
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
    global STRUCTURE_DIRS, OUTPUT_DIR, REPORTS_DIR, LOGS_DIR, CONTACT_CUTOFF
    args = parse_args()
    if args.structure_dirs:
        STRUCTURE_DIRS = [path.resolve() for path in args.structure_dirs]
    OUTPUT_DIR = args.output_dir.resolve()
    REPORTS_DIR = args.reports_dir.resolve()
    LOGS_DIR = args.logs_dir.resolve()
    CONTACT_CUTOFF = args.contact_cutoff

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("GCV Feature Computation for Expanded Dataset")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find structures
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

    # Process
    all_features = []
    per_model_features = []
    stats = {"ok": 0, "failed": 0}

    glycan_ids = sorted(structures.keys())
    for i, glytoucan_id in enumerate(glycan_ids):
        info = structures[glytoucan_id]
        pdb_path = info["pdb_path"]
        source_dir = info["source_dir"]

        models = parse_models(pdb_path)
        if not models:
            stats["failed"] += 1
            continue

        model_rows = []
        for model_index, model in enumerate(models, start=1):
            model_features = compute_gcv_features(
                model,
                pdb_path,
                glytoucan_id,
                source_dir,
                model_id=model_index,
                contact_cutoff=CONTACT_CUTOFF,
            )
            if model_features is not None:
                model_rows.append(asdict(model_features))

        if not model_rows:
            stats["failed"] += 1
            continue

        per_model_features.extend(model_rows)
        numeric_fields = [
            'contact_density', 'long_range_contact_fraction',
            'mean_residue_neighbor_count', 'sd_residue_neighbor_count',
            'torsion_diversity', 'graph_laplacian_spectral_gap',
            'core_periphery_ratio', 'max_contact_distance_seq',
            'n_residues', 'n_contacts'
        ]
        aggregate_row = {
            'glytoucan_id': glytoucan_id,
            'n_models': len(model_rows),
            'contact_cutoff_angstrom': CONTACT_CUTOFF,
            'source_dir': source_dir,
            'source_file': model_rows[0]['source_file'],
        }
        for field in numeric_fields:
            values = np.asarray([row[field] for row in model_rows], dtype=float)
            aggregate_row[field] = round(float(values.mean()), 6)
            aggregate_row[f'{field}_sd'] = round(float(values.std()), 6)
        all_features.append(aggregate_row)
        stats["ok"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(glycan_ids)} glycans")

    # Save
    logger.info("\nSaving GCV features...")
    gcv_path = OUTPUT_DIR / "gcv_features.csv"

    numeric_fields = ['contact_density', 'long_range_contact_fraction',
                      'mean_residue_neighbor_count', 'sd_residue_neighbor_count',
                      'torsion_diversity', 'graph_laplacian_spectral_gap',
                      'core_periphery_ratio', 'max_contact_distance_seq',
                      'n_residues', 'n_contacts']
    fieldnames = ['glytoucan_id', 'n_models', 'contact_cutoff_angstrom', *numeric_fields,
                  *[f'{field}_sd' for field in numeric_fields], 'source_dir', 'source_file']

    with open(gcv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_features:
            writer.writerow(row)

    logger.info(f"Saved: {gcv_path}")

    per_model_path = OUTPUT_DIR / "gcv_per_model.csv"
    per_model_fieldnames = [
        'glytoucan_id', 'model_id', *numeric_fields, 'source_dir', 'source_file'
    ]
    with open(per_model_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_model_fieldnames)
        writer.writeheader()
        writer.writerows(per_model_features)
    logger.info(f"Saved: {per_model_path}")

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
    report_path = REPORTS_DIR / "gcv_summary.md"
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
        f.write(f"Contact cutoff: {CONTACT_CUTOFF:.1f} Angstrom. Each glycan row is the mean across genuine models in the GlycoShape level-1 multi-model PDB; matching `_sd` columns and `gcv_per_model.csv` preserve ensemble dispersion.\n\n")
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
        f.write("- **torsion_diversity**: CV of pairwise residue-centre distances; no torsion angles are used\n")
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
    logger.info(f"  - {per_model_path}")
    logger.info(f"  - {report_path}")

    return stats


if __name__ == "__main__":
    result = main()
    raise SystemExit(1 if result.get("status") == "no_structures" else 0)
