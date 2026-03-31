#!/usr/bin/env python3
"""
WURCS → PDB Structure Generation Module.

Generates 3D structures from WURCS sequences using glycosylator library.
Uses GlyCosmos SPARQL to retrieve IUPAC from GlyTouCan IDs, then builds structures.

This module provides deterministic, auditable structure generation with explicit
failure classification for unsupported cases.

Failure taxonomy:
- unsupported_monosaccharide: Monosaccharide not in glycosylator topology
- ambiguous_linkage_unresolvable: Linkage ambiguity that cannot be resolved
- iupac_retrieval_failed: Could not get IUPAC from GlyCosmos
- iupac_parse_error: IUPAC string could not be parsed
- structure_build_error: Error during 3D structure assembly
- optimization_failed: Structure optimization failed
- tool_limitations: General glycosylator limitation
- other: Unclassified error
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of structure generation attempt."""
    glytoucan_id: str
    success: bool
    method: str  # "wurcs_generation", "iupac_generation"
    pdb_path: Optional[str]
    n_atoms: int
    n_residues: int
    iupac: Optional[str]
    wurcs: Optional[str]
    failure_reason: Optional[str]
    failure_category: Optional[str]
    assumptions: List[str]
    generation_time_sec: float
    sha256: Optional[str]
    timestamp: str


class WURCSToPDBGenerator:
    """
    Generator for 3D structures from WURCS/GlyTouCan IDs.

    Uses glycosylator library with GlyCosmos IUPAC lookup.
    """

    # Known monosaccharide mappings that may need special handling
    MONOSACCHARIDE_ALIASES = {
        'GlcNAc': 'NAG',
        'GalNAc': 'A2G',
        'Man': 'MAN',
        'Gal': 'GAL',
        'Glc': 'GLC',
        'Fuc': 'FUC',
        'NeuAc': 'SIA',
        'Sia': 'SIA',
        'Neu5Ac': 'SIA',
        'Xyl': 'XYL',
    }

    def __init__(
        self,
        output_dir: Path,
        cache_iupac: bool = True,
        optimize_structure: bool = False,  # Skip optimization by default for speed
        api_delay: float = 0.5,  # Delay between GlyCosmos API calls
    ):
        """
        Initialize generator.

        Args:
            output_dir: Directory for output structures
            cache_iupac: Cache IUPAC lookups to avoid repeated API calls
            optimize_structure: Whether to optimize generated structures
            api_delay: Delay between API calls in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_iupac = cache_iupac
        self.optimize_structure = optimize_structure
        self.api_delay = api_delay

        # IUPAC cache
        self._iupac_cache: Dict[str, Optional[str]] = {}
        self._cache_file = self.output_dir / ".iupac_cache.json"
        self._load_cache()

        # Import glycosylator lazily to avoid import errors
        self._gly = None
        self._topology_loaded = False

    def _load_cache(self):
        """Load IUPAC cache from disk."""
        if self.cache_iupac and self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    self._iupac_cache = json.load(f)
                logger.info(f"Loaded {len(self._iupac_cache)} cached IUPAC entries")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")

    def _save_cache(self):
        """Save IUPAC cache to disk."""
        if self.cache_iupac:
            try:
                with open(self._cache_file, 'w') as f:
                    json.dump(self._iupac_cache, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")

    def _get_glycosylator(self):
        """Lazy import of glycosylator."""
        if self._gly is None:
            import glycosylator as gly
            self._gly = gly

            # Ensure sugars are loaded
            if not self._topology_loaded:
                try:
                    gly.load_sugars()
                    self._topology_loaded = True
                except Exception as e:
                    logger.warning(f"Could not load sugar topology: {e}")

        return self._gly

    def get_iupac_from_glytoucan(self, glytoucan_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get IUPAC string from GlyTouCan ID via GlyCosmos.

        Args:
            glytoucan_id: GlyTouCan accession ID

        Returns:
            Tuple of (iupac_string, error_message)
        """
        # Check cache first
        if glytoucan_id in self._iupac_cache:
            cached = self._iupac_cache[glytoucan_id]
            if cached is None:
                return None, "iupac_retrieval_failed (cached)"
            return cached, None

        try:
            gly = self._get_glycosylator()

            # Rate limiting
            time.sleep(self.api_delay)

            iupac = gly.get_iupac_from_glycosmos(glytoucan_id)

            if iupac and len(iupac.strip()) > 0:
                self._iupac_cache[glytoucan_id] = iupac
                self._save_cache()
                return iupac, None
            else:
                self._iupac_cache[glytoucan_id] = None
                self._save_cache()
                return None, "iupac_retrieval_failed: empty response"

        except Exception as e:
            self._iupac_cache[glytoucan_id] = None
            self._save_cache()
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                return None, f"iupac_retrieval_failed: {glytoucan_id} not in GlyCosmos"
            return None, f"iupac_retrieval_failed: {error_msg}"

    def _classify_error(self, error_msg: str) -> str:
        """Classify error into taxonomy category."""
        error_lower = error_msg.lower()

        if "residue" in error_lower and ("not found" in error_lower or "unknown" in error_lower):
            return "unsupported_monosaccharide"
        if "linkage" in error_lower:
            return "ambiguous_linkage_unresolvable"
        if "iupac" in error_lower and "parse" in error_lower:
            return "iupac_parse_error"
        if "iupac" in error_lower:
            return "iupac_retrieval_failed"
        if "optim" in error_lower:
            return "optimization_failed"
        if "build" in error_lower or "construct" in error_lower:
            return "structure_build_error"
        if "glycosylator" in error_lower or "topology" in error_lower:
            return "tool_limitations"
        return "other"

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_structure(
        self,
        glytoucan_id: str,
        wurcs: Optional[str] = None,
        iupac: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate 3D structure for a glycan.

        Args:
            glytoucan_id: GlyTouCan accession ID
            wurcs: Optional WURCS string (for metadata only, not used for generation)
            iupac: Optional IUPAC string (if known, skip API lookup)

        Returns:
            GenerationResult with success/failure info
        """
        start_time = time.time()
        assumptions = []

        # Create output directory for this glycan
        glycan_dir = self.output_dir / glytoucan_id
        glycan_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = glycan_dir / "structure_1.pdb"
        meta_path = glycan_dir / "meta.json"

        # Get IUPAC if not provided
        if iupac is None:
            iupac, error = self.get_iupac_from_glytoucan(glytoucan_id)
            if error:
                return self._make_failure_result(
                    glytoucan_id, wurcs, iupac, error,
                    self._classify_error(error), assumptions, start_time
                )

        # Build structure from IUPAC
        try:
            gly = self._get_glycosylator()

            # Parse IUPAC and build structure
            glycan = gly.read_iupac(glytoucan_id, iupac)

            # Count atoms and residues
            n_atoms = glycan.count_atoms()
            n_residues = len(list(glycan.residues))

            if n_atoms == 0:
                return self._make_failure_result(
                    glytoucan_id, wurcs, iupac,
                    "structure_build_error: 0 atoms generated",
                    "structure_build_error", assumptions, start_time
                )

            # Optional optimization
            if self.optimize_structure:
                try:
                    gly.mmff_optimize(glycan, max_iter=100)
                    assumptions.append("structure_optimized_mmff")
                except Exception as e:
                    assumptions.append(f"optimization_skipped: {e}")
            else:
                assumptions.append("no_optimization_applied")

            # Write PDB
            gly.write_pdb(glycan, str(pdb_path))

            # Verify output
            if not pdb_path.exists() or pdb_path.stat().st_size == 0:
                return self._make_failure_result(
                    glytoucan_id, wurcs, iupac,
                    "structure_build_error: empty PDB output",
                    "structure_build_error", assumptions, start_time
                )

            # Compute hash
            sha256 = self._compute_file_hash(pdb_path)

            # Add assumptions about linkage resolution
            assumptions.append("glycosylator_default_conformations")
            assumptions.append("iupac_condensed_format")

            result = GenerationResult(
                glytoucan_id=glytoucan_id,
                success=True,
                method="iupac_generation",
                pdb_path=str(pdb_path),
                n_atoms=n_atoms,
                n_residues=n_residues,
                iupac=iupac,
                wurcs=wurcs,
                failure_reason=None,
                failure_category=None,
                assumptions=assumptions,
                generation_time_sec=round(time.time() - start_time, 3),
                sha256=sha256,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Write meta.json
            meta = {
                "status": "ok",
                "method": "wurcs_generation",
                "iupac": iupac,
                "wurcs": wurcs,
                "n_atoms": n_atoms,
                "n_residues": n_residues,
                "assumptions": assumptions,
                "sha256": sha256,
                "timestamp": result.timestamp,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            return result

        except Exception as e:
            error_msg = str(e)
            category = self._classify_error(error_msg)

            # Write failure meta.json
            meta = {
                "status": "failed",
                "method": "wurcs_generation",
                "reason": error_msg,
                "failure_category": category,
                "iupac": iupac,
                "wurcs": wurcs,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            return self._make_failure_result(
                glytoucan_id, wurcs, iupac, error_msg, category, assumptions, start_time
            )

    def _make_failure_result(
        self,
        glytoucan_id: str,
        wurcs: Optional[str],
        iupac: Optional[str],
        error_msg: str,
        category: str,
        assumptions: List[str],
        start_time: float,
    ) -> GenerationResult:
        """Create a failure result."""
        return GenerationResult(
            glytoucan_id=glytoucan_id,
            success=False,
            method="iupac_generation",
            pdb_path=None,
            n_atoms=0,
            n_residues=0,
            iupac=iupac,
            wurcs=wurcs,
            failure_reason=error_msg,
            failure_category=category,
            assumptions=assumptions,
            generation_time_sec=round(time.time() - start_time, 3),
            sha256=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def generate_batch(
        self,
        glycans: List[Dict[str, str]],
        progress_callback=None,
    ) -> List[GenerationResult]:
        """
        Generate structures for a batch of glycans.

        Args:
            glycans: List of dicts with 'glytoucan_id' and optional 'wurcs', 'iupac'
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of GenerationResults
        """
        results = []
        total = len(glycans)

        for i, glycan in enumerate(glycans):
            glytoucan_id = glycan['glytoucan_id']
            wurcs = glycan.get('wurcs')
            iupac = glycan.get('iupac')

            result = self.generate_structure(glytoucan_id, wurcs, iupac)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results


def generate_structures_for_pending(
    manifest_path: Path,
    output_dir: Path,
    max_items: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Generate structures for glycans marked as pending in manifest.

    Args:
        manifest_path: Path to sugarbind_glycans.csv manifest
        output_dir: Output directory for structures
        max_items: Maximum number to generate (for testing)
        dry_run: If True, just count pending without generating

    Returns:
        Summary statistics dict
    """
    import csv

    # Load manifest
    pending = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            glytoucan_id = row['glytoucan_id']

            # Check if structure already exists and is valid
            meta_path = output_dir / glytoucan_id / "meta.json"
            if meta_path.exists():
                with open(meta_path) as mf:
                    meta = json.load(mf)
                if meta.get('status') == 'ok':
                    continue  # Already have good structure
                if meta.get('failure_category') in [
                    'unsupported_monosaccharide',
                    'iupac_retrieval_failed',
                ]:
                    # Permanent failure, skip
                    continue

            pending.append({
                'glytoucan_id': glytoucan_id,
                'wurcs': row.get('wurcs'),
            })

    logger.info(f"Found {len(pending)} glycans pending structure generation")

    if dry_run:
        return {
            'pending_count': len(pending),
            'dry_run': True,
        }

    if max_items:
        pending = pending[:max_items]

    # Initialize generator
    generator = WURCSToPDBGenerator(
        output_dir=output_dir,
        cache_iupac=True,
        optimize_structure=False,
    )

    # Generate with progress
    stats = {
        'total': len(pending),
        'success': 0,
        'failed': 0,
        'failure_categories': {},
    }

    def progress_cb(current, total, result):
        if result.success:
            stats['success'] += 1
            logger.info(f"[{current}/{total}] {result.glytoucan_id}: OK ({result.n_atoms} atoms)")
        else:
            stats['failed'] += 1
            cat = result.failure_category or 'unknown'
            stats['failure_categories'][cat] = stats['failure_categories'].get(cat, 0) + 1
            logger.warning(f"[{current}/{total}] {result.glytoucan_id}: FAILED ({cat})")

    results = generator.generate_batch(pending, progress_callback=progress_cb)

    # Save summary
    stats['results'] = [asdict(r) for r in results]

    return stats
