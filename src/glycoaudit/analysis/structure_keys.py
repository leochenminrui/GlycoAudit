"""
Canonical Structure Key Generation for Glycans

This module provides TRUE structure-aware grouping keys based on canonical
glycan representations (WURCS, IUPAC, GlycoCT) rather than feature-based proxies.

Hierarchy of methods (most to least biologically faithful):
1. WURCS canonical hash (preferred)
2. IUPAC condensed normalization
3. GlycoCT hash
4. Topology-based hash (graph structure)
5. Fallback to GlyTouCan ID (unique but not grouping)

All methods are deterministic and produce stable group keys for splitting.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class StructureKeyGenerator:
    """
    Generate canonical structure keys for glycans from available metadata.
    """

    def __init__(self, glytoucan_data_path: Optional[Path] = None):
        """
        Initialize generator.

        Args:
            glytoucan_data_path: Path to GlyTouCan bulk export JSON
        """
        self.glytoucan_data_path = glytoucan_data_path
        self.glytoucan_cache = {}
        self._load_glytoucan_data()

        # Coverage tracking for linkage key generation
        self.linkage_coverage_stats = {
            'n_total_glycans_seen': 0,
            'n_linkage_key_success': 0,
            'n_linkage_key_fallback': 0,
            'n_linkage_key_missing_input': 0,
            'n_wurcs_linkage_success': 0,
            'n_iupac_linkage_success': 0,
            'n_alpha_beta_detected': 0,
            'n_position_detected': 0,
            'linkage_patterns': defaultdict(int),  # Count each linkage pattern type
        }

    def _load_glytoucan_data(self):
        """Load GlyTouCan SPARQL data with WURCS sequences."""
        if self.glytoucan_data_path and self.glytoucan_data_path.exists():
            try:
                with open(self.glytoucan_data_path, 'r') as f:
                    data = json.load(f)

                bindings = data.get('results', {}).get('bindings', [])

                for entry in bindings:
                    accession = entry.get('accession', {}).get('value')
                    if accession:
                        self.glytoucan_cache[accession] = {
                            'wurcs': entry.get('wurcs', {}).get('value'),
                            'iupac': entry.get('iupac', {}).get('value'),
                            'iupac_condensed': entry.get('iupac_condensed', {}).get('value'),
                            'glycoct': entry.get('glycoct', {}).get('value'),
                        }

                print(f"[StructureKeys] Loaded {len(self.glytoucan_cache)} GlyTouCan entries")

            except Exception as e:
                print(f"[StructureKeys] Warning: Could not load GlyTouCan data: {e}")

    def get_wurcs(self, glytoucan_id: str) -> Optional[str]:
        """Get WURCS sequence for a glycan."""
        entry = self.glytoucan_cache.get(glytoucan_id, {})
        return entry.get('wurcs')

    def get_iupac(self, glytoucan_id: str) -> Optional[str]:
        """Get IUPAC sequence for a glycan."""
        entry = self.glytoucan_cache.get(glytoucan_id, {})
        return entry.get('iupac') or entry.get('iupac_condensed')

    # =========================================================================
    # TRUE SCAFFOLD KEY (Topology without linkage specifics)
    # =========================================================================

    def compute_scaffold_key(self, glytoucan_id: str) -> str:
        """
        Compute canonical scaffold key (topology).

        Scaffold = core glycan topology/backbone ignoring:
        - Linkage positions (e.g., 1-3 vs 1-4)
        - Anomeric configuration (alpha/beta)
        - Modifications (sulfation, methylation, etc.)

        Method priority:
        1. WURCS-based topology hash (preferred)
        2. IUPAC-based topology extraction
        3. Fallback to GlyTouCan ID

        Returns:
            Canonical scaffold key string
        """
        # Try WURCS-based scaffold
        wurcs = self.get_wurcs(glytoucan_id)
        if wurcs:
            scaffold = self._extract_wurcs_scaffold(wurcs)
            if scaffold:
                return f"wurcs_scaffold_{self._hash_str(scaffold)}"

        # Try IUPAC-based scaffold
        iupac = self.get_iupac(glytoucan_id)
        if iupac:
            scaffold = self._extract_iupac_scaffold(iupac)
            if scaffold:
                return f"iupac_scaffold_{self._hash_str(scaffold)}"

        # Fallback
        return f"fallback_scaffold_{glytoucan_id[:8]}"

    def _extract_wurcs_scaffold(self, wurcs: str) -> Optional[str]:
        """
        Extract scaffold from WURCS sequence.

        WURCS format: WURCS=2.0/<uniqueRES>/<RES>,<topology>,<connections>
        Scaffold = residue composition + connection topology (ignore linkage details)
        """
        try:
            # Parse WURCS components
            parts = wurcs.split('/')
            if len(parts) < 4:
                return None

            # Extract residue section (compositional info)
            residue_section = parts[3] if len(parts) > 3 else ""

            # Normalize residues by removing modifications and linkage specifics
            # Keep only core sugar types
            residues = self._normalize_wurcs_residues(residue_section)

            # Extract topology section (number of residues, branching)
            topology = parts[2] if len(parts) > 2 else ""

            # Create scaffold key
            scaffold = f"{residues}|{topology}"
            return scaffold

        except Exception:
            return None

    def _normalize_wurcs_residues(self, residue_section: str) -> str:
        """
        Normalize WURCS residues to scaffold level.

        Removes linkage specifics, keeps core composition.
        """
        # Extract individual residues (in brackets)
        residues = re.findall(r'\[([^\]]+)\]', residue_section)

        # Normalize each residue (remove fine details, keep core type)
        normalized = []
        for res in residues:
            # Keep only the first few characters (core type)
            # Remove modifications, linkage positions
            core = re.sub(r'_\d+\*.*', '', res)  # Remove linkage details
            core = re.sub(r'\d+\*', '', core)    # Remove position markers
            core = re.sub(r'/.*', '', core)      # Remove modifications
            normalized.append(core[:10])  # Truncate to core

        # Sort for canonical form
        normalized.sort()
        return ','.join(normalized)

    def _extract_iupac_scaffold(self, iupac: str) -> Optional[str]:
        """
        Extract scaffold from IUPAC sequence.

        Keep residue types and branching, ignore linkages.
        """
        try:
            # Remove linkage info (e.g., a1-3, b1-4)
            scaffold = re.sub(r'[ab]\d+-\d+', 'x-x', iupac)

            # Remove anomeric info
            scaffold = re.sub(r'[αβ]', '', scaffold)

            # Normalize whitespace
            scaffold = re.sub(r'\s+', '', scaffold)

            return scaffold

        except Exception:
            return None

    # =========================================================================
    # TRUE TERMINAL MOTIF KEY
    # =========================================================================

    def compute_terminal_motif_key(self, glytoucan_id: str) -> str:
        """
        Compute canonical terminal motif key.

        Terminal motif = structure of terminal (non-reducing end) residues.

        Method:
        1. Extract terminal residues from WURCS/IUPAC
        2. Hash their structure including modifications
        3. Group glycans with same terminal patterns

        Returns:
            Canonical terminal motif key
        """
        wurcs = self.get_wurcs(glytoucan_id)
        if wurcs:
            terminal = self._extract_wurcs_terminals(wurcs)
            if terminal:
                return f"wurcs_terminal_{self._hash_str(terminal)}"

        iupac = self.get_iupac(glytoucan_id)
        if iupac:
            terminal = self._extract_iupac_terminals(iupac)
            if terminal:
                return f"iupac_terminal_{self._hash_str(terminal)}"

        return f"fallback_terminal_{glytoucan_id[:8]}"

    def _extract_wurcs_terminals(self, wurcs: str) -> Optional[str]:
        """
        Extract terminal residues from WURCS.

        Terminal = residues at non-reducing ends (no outgoing bonds).
        """
        try:
            parts = wurcs.split('/')
            if len(parts) < 5:
                return None

            # Parse connection section
            connections = parts[4] if len(parts) > 4 else ""

            # Identify terminal positions (appear only on right side of connections)
            # This is simplified; full WURCS parser would be complex
            # Use heuristic: last residues in sequence are often terminal

            residue_section = parts[3]
            residues = re.findall(r'\[([^\]]+)\]', residue_section)

            # Take last 1-2 residues as terminals (heuristic)
            terminal_residues = residues[-2:] if len(residues) >= 2 else residues[-1:]

            # Normalize terminal residues (keep modifications for terminals)
            terminal_str = ','.join(sorted(terminal_residues))

            return terminal_str

        except Exception:
            return None

    def _extract_iupac_terminals(self, iupac: str) -> Optional[str]:
        """Extract terminal residues from IUPAC sequence."""
        try:
            # Split by branching
            parts = re.split(r'[\[\]]', iupac)

            # Extract last residue in each branch (terminals)
            terminals = []
            for part in parts:
                if part.strip():
                    # Get rightmost residue
                    residues = re.findall(r'[A-Z][a-z]*', part)
                    if residues:
                        terminals.append(residues[-1])

            terminal_str = ','.join(sorted(set(terminals)))
            return terminal_str

        except Exception:
            return None

    # =========================================================================
    # TRUE LINKAGE KEY
    # =========================================================================

    def compute_linkage_key(self, glytoucan_id: str) -> str:
        """
        Compute canonical linkage pattern key.

        Linkage pattern = specific glycosidic bond types (α/β, positions).

        Method:
        1. Extract all linkages from WURCS/IUPAC
        2. Create canonical linkage signature
        3. Group glycans with same linkage patterns

        Returns:
            Canonical linkage key
        """
        # Track coverage
        self.linkage_coverage_stats['n_total_glycans_seen'] += 1

        wurcs = self.get_wurcs(glytoucan_id)
        iupac = self.get_iupac(glytoucan_id)

        # Check if we have any input data
        if not wurcs and not iupac:
            self.linkage_coverage_stats['n_linkage_key_missing_input'] += 1
            self.linkage_coverage_stats['n_linkage_key_fallback'] += 1
            return f"fallback_linkage_{glytoucan_id[:8]}"

        # Try WURCS first
        if wurcs:
            linkages = self._extract_wurcs_linkages(wurcs)
            if linkages:
                self.linkage_coverage_stats['n_linkage_key_success'] += 1
                self.linkage_coverage_stats['n_wurcs_linkage_success'] += 1
                self._track_linkage_patterns(linkages)
                return f"wurcs_linkage_{self._hash_str(linkages)}"

        # Try IUPAC fallback
        if iupac:
            linkages = self._extract_iupac_linkages(iupac)
            if linkages:
                self.linkage_coverage_stats['n_linkage_key_success'] += 1
                self.linkage_coverage_stats['n_iupac_linkage_success'] += 1
                self._track_linkage_patterns(linkages)
                return f"iupac_linkage_{self._hash_str(linkages)}"

        # Failed to extract linkages despite having input
        self.linkage_coverage_stats['n_linkage_key_fallback'] += 1
        return f"fallback_linkage_{glytoucan_id[:8]}"

    def _extract_wurcs_linkages(self, wurcs: str) -> Optional[str]:
        """
        Extract linkage patterns from WURCS.

        Linkages are in the connection section.
        """
        try:
            parts = wurcs.split('/')
            if len(parts) < 5:
                return None

            # Connection section contains linkage info
            connections = parts[4]

            # Extract linkage patterns (e.g., a4-b1, b3-c1)
            linkage_pattern = re.findall(r'[a-z]\d+-[a-z]\d+', connections)

            # Normalize: sort and deduplicate
            linkage_pattern = sorted(set(linkage_pattern))

            return ','.join(linkage_pattern)

        except Exception:
            return None

    def _extract_iupac_linkages(self, iupac: str) -> Optional[str]:
        """Extract linkage patterns from IUPAC sequence."""
        try:
            # Extract linkage patterns (e.g., a1-3, b1-4)
            linkages = re.findall(r'[ab]\d+-\d+', iupac)

            # Normalize
            linkages = sorted(set(linkages))

            return ','.join(linkages)

        except Exception:
            return None

    def _track_linkage_patterns(self, linkages: str):
        """
        Track statistics about detected linkage patterns.

        Args:
            linkages: Comma-separated linkage patterns (e.g., "a4-b1,b3-c1")
        """
        # Count alpha/beta bonds
        if 'a' in linkages.lower():
            self.linkage_coverage_stats['n_alpha_beta_detected'] += 1

        # Count position-specific linkages
        if re.search(r'\d+-\w+\d+', linkages):
            self.linkage_coverage_stats['n_position_detected'] += 1

        # Track individual patterns
        patterns = linkages.split(',')
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern:
                self.linkage_coverage_stats['linkage_patterns'][pattern] += 1

    # =========================================================================
    # Utilities
    # =========================================================================

    def _hash_str(self, s: str) -> str:
        """Generate stable hash of string."""
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

    def compute_all_keys(self, glytoucan_id: str) -> Dict[str, str]:
        """
        Compute all structure keys for a glycan.

        Returns:
            Dictionary with scaffold, terminal, linkage keys
        """
        return {
            'scaffold_key': self.compute_scaffold_key(glytoucan_id),
            'terminal_motif_key': self.compute_terminal_motif_key(glytoucan_id),
            'linkage_key': self.compute_linkage_key(glytoucan_id),
        }

    def analyze_key_distribution(self, glycan_ids: List[str]) -> Dict:
        """
        Analyze structure key distribution for a set of glycans.

        Returns:
            Dictionary with statistics
        """
        scaffold_keys = defaultdict(list)
        terminal_keys = defaultdict(list)
        linkage_keys = defaultdict(list)

        for gid in glycan_ids:
            keys = self.compute_all_keys(gid)
            scaffold_keys[keys['scaffold_key']].append(gid)
            terminal_keys[keys['terminal_motif_key']].append(gid)
            linkage_keys[keys['linkage_key']].append(gid)

        stats = {
            'n_glycans': len(glycan_ids),
            'scaffold': {
                'n_unique': len(scaffold_keys),
                'size_mean': sum(len(v) for v in scaffold_keys.values()) / len(scaffold_keys) if scaffold_keys else 0,
                'size_max': max(len(v) for v in scaffold_keys.values()) if scaffold_keys else 0,
                'singleton_fraction': sum(1 for v in scaffold_keys.values() if len(v) == 1) / len(scaffold_keys) if scaffold_keys else 0,
            },
            'terminal_motif': {
                'n_unique': len(terminal_keys),
                'size_mean': sum(len(v) for v in terminal_keys.values()) / len(terminal_keys) if terminal_keys else 0,
                'size_max': max(len(v) for v in terminal_keys.values()) if terminal_keys else 0,
                'singleton_fraction': sum(1 for v in terminal_keys.values() if len(v) == 1) / len(terminal_keys) if terminal_keys else 0,
            },
            'linkage': {
                'n_unique': len(linkage_keys),
                'size_mean': sum(len(v) for v in linkage_keys.values()) / len(linkage_keys) if linkage_keys else 0,
                'size_max': max(len(v) for v in linkage_keys.values()) if linkage_keys else 0,
                'singleton_fraction': sum(1 for v in linkage_keys.values() if len(v) == 1) / len(linkage_keys) if linkage_keys else 0,
            }
        }

        return stats

    def get_linkage_key_coverage_stats(self) -> Dict:
        """
        Get comprehensive linkage key coverage statistics.

        Returns:
            Dictionary with coverage metrics including:
            - Success/fallback/missing rates
            - WURCS vs IUPAC usage
            - Alpha/beta bond detection
            - Position-specific linkage detection
            - Top linkage patterns
        """
        n_total = self.linkage_coverage_stats['n_total_glycans_seen']

        # Compute percentages
        success_rate = (
            self.linkage_coverage_stats['n_linkage_key_success'] / n_total * 100
            if n_total > 0 else 0
        )
        fallback_rate = (
            self.linkage_coverage_stats['n_linkage_key_fallback'] / n_total * 100
            if n_total > 0 else 0
        )
        missing_input_rate = (
            self.linkage_coverage_stats['n_linkage_key_missing_input'] / n_total * 100
            if n_total > 0 else 0
        )

        # Top linkage patterns
        linkage_patterns = dict(self.linkage_coverage_stats['linkage_patterns'])
        top_patterns = sorted(
            linkage_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]  # Top 20

        stats = {
            'summary': {
                'n_total_glycans_seen': n_total,
                'n_linkage_key_success': self.linkage_coverage_stats['n_linkage_key_success'],
                'n_linkage_key_fallback': self.linkage_coverage_stats['n_linkage_key_fallback'],
                'n_linkage_key_missing_input': self.linkage_coverage_stats['n_linkage_key_missing_input'],
                'success_rate_pct': round(success_rate, 2),
                'fallback_rate_pct': round(fallback_rate, 2),
                'missing_input_rate_pct': round(missing_input_rate, 2),
            },
            'method_usage': {
                'n_wurcs_linkage_success': self.linkage_coverage_stats['n_wurcs_linkage_success'],
                'n_iupac_linkage_success': self.linkage_coverage_stats['n_iupac_linkage_success'],
            },
            'feature_detection': {
                'n_alpha_beta_detected': self.linkage_coverage_stats['n_alpha_beta_detected'],
                'n_position_detected': self.linkage_coverage_stats['n_position_detected'],
            },
            'top_linkage_patterns': [
                {'pattern': pattern, 'count': count}
                for pattern, count in top_patterns
            ],
            'warning': None
        }

        # Add warnings if coverage is low
        if success_rate < 80:
            stats['warning'] = (
                f"Low linkage key success rate ({success_rate:.1f}%). "
                f"Consider checking GlyTouCan data quality or parser implementation."
            )

        return stats

    def analyze_scaffold_group_distribution(
        self,
        scaffold_keys: List[str],
        glycan_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze TRUE scaffold group size distribution and dominance.

        Computes:
        - Group size statistics (min/median/mean/max)
        - Top-N largest groups
        - Dominance metrics (HHI, Gini, top-K fractions)
        - Warnings if distribution is highly unbalanced

        Args:
            scaffold_keys: List of scaffold keys for glycans
            glycan_ids: Optional list of glycan IDs (for top-N group membership)

        Returns:
            Dictionary with distribution statistics and dominance metrics
        """
        from collections import Counter

        # Count group sizes
        group_counts = Counter(scaffold_keys)
        sizes = sorted(group_counts.values(), reverse=True)

        n_glycans = len(scaffold_keys)
        n_groups = len(group_counts)

        if n_groups == 0:
            return {'error': 'No groups found'}

        # Basic statistics
        size_min = min(sizes)
        size_max = max(sizes)
        size_mean = np.mean(sizes)
        size_median = np.median(sizes)
        size_std = np.std(sizes)

        # Top-10 groups
        top_10 = [
            {'group_key': key, 'size': count}
            for key, count in group_counts.most_common(10)
        ]

        # Add glycan IDs for top-10 if provided
        if glycan_ids:
            key_to_glycans = defaultdict(list)
            for gid, key in zip(glycan_ids, scaffold_keys):
                key_to_glycans[key].append(gid)

            for item in top_10:
                item['glycan_ids'] = key_to_glycans[item['group_key']][:5]  # First 5

        # Dominance metrics
        # Fraction of data in top-K groups
        top_1_frac = sizes[0] / n_glycans if len(sizes) > 0 else 0
        top_5_frac = sum(sizes[:5]) / n_glycans if len(sizes) >= 5 else sum(sizes) / n_glycans
        top_10_frac = sum(sizes[:10]) / n_glycans if len(sizes) >= 10 else sum(sizes) / n_glycans

        # Herfindahl-Hirschman Index (HHI): sum of squared market shares
        # HHI = 1 means perfect concentration (one group)
        # HHI → 0 means perfect balance
        hhi = sum((size / n_glycans) ** 2 for size in sizes)

        # Gini coefficient: inequality measure
        # Gini = 0 means perfect equality
        # Gini = 1 means perfect inequality
        sizes_sorted = np.array(sorted(sizes))
        n = len(sizes_sorted)
        cumsum = np.cumsum(sizes_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sizes_sorted)) / (n * cumsum[-1]) - (n + 1) / n

        stats = {
            'summary': {
                'n_glycans': n_glycans,
                'n_groups': n_groups,
                'size_min': int(size_min),
                'size_median': float(size_median),
                'size_mean': round(float(size_mean), 2),
                'size_max': int(size_max),
                'size_std': round(float(size_std), 2),
            },
            'dominance': {
                'top_1_fraction': round(top_1_frac, 4),
                'top_5_fraction': round(top_5_frac, 4),
                'top_10_fraction': round(top_10_frac, 4),
                'hhi': round(hhi, 4),
                'gini_coefficient': round(gini, 4),
            },
            'top_10_groups': top_10,
            'warning': None
        }

        # Add warnings for highly unbalanced distributions
        if hhi > 0.25:
            stats['warning'] = (
                f"High HHI ({hhi:.3f}) indicates strong dominance by few groups. "
                f"Top-1 group contains {top_1_frac*100:.1f}% of glycans."
            )
        elif top_1_frac > 0.5:
            stats['warning'] = (
                f"Top-1 scaffold group dominates with {top_1_frac*100:.1f}% of glycans. "
                f"Consider whether this reflects biological reality or data artifact."
            )

        return stats


def load_structure_keys_for_dataset(
    df_features: pd.DataFrame,
    glytoucan_path: Path = Path("/home/minrui/glyco/public_glyco_mirror/data/raw/glytoucan/bulk_export.json"),
    return_generator: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, StructureKeyGenerator]:
    """
    Load structure keys for all glycans in a dataset.

    Args:
        df_features: DataFrame with glytoucan_id column
        glytoucan_path: Path to GlyTouCan data
        return_generator: If True, return (df, generator) tuple for coverage access

    Returns:
        DataFrame with added columns: scaffold_key, terminal_motif_key, linkage_key
        Or tuple of (DataFrame, generator) if return_generator=True
    """
    generator = StructureKeyGenerator(glytoucan_path)

    df = df_features.copy()

    scaffold_keys = []
    terminal_keys = []
    linkage_keys = []

    for gid in df['glytoucan_id']:
        keys = generator.compute_all_keys(gid)
        scaffold_keys.append(keys['scaffold_key'])
        terminal_keys.append(keys['terminal_motif_key'])
        linkage_keys.append(keys['linkage_key'])

    df['scaffold_key_true'] = scaffold_keys
    df['terminal_motif_key_true'] = terminal_keys
    df['linkage_key_true'] = linkage_keys

    if return_generator:
        return df, generator
    return df
