# GlycoAudit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A reproducible benchmark for glycan-binding agent interaction prediction**

GlycoAudit provides a curated, semantically audited benchmark for evaluating computational methods that predict lectin-glycan and antibody-glycan binding interactions. The benchmark uses a positive-unlabeled (PU) learning framework to handle the inherent incompleteness of experimental binding data.

---

## Overview

**What is GlycoAudit?**

GlycoAudit is a resource paper that provides:

1. **Curated Benchmark Dataset** - 342 structurally resolved glycans × 196 binding agents (lectins/antibodies) with experimentally validated interactions
2. **Semantic Joinability Pipeline** - Reproducible workflow for integrating heterogeneous glycobiology data sources
3. **Transparent Baselines** - Multiple baseline methods (structural, sequence-based, learned) for binding prediction
4. **Evaluation Framework** - PU-ranking metrics designed for incomplete binding data
5. **Statistical Validation** - Null controls, multiple testing correction, and robustness checks

**Key Finding:** Structure-based features (SSV) significantly outperform random and permutation baselines (MRR=0.592 vs 0.106/0.430, p<0.001), but generalization across structural scaffolds remains challenging.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/GlycoAudit.git
cd GlycoAudit

# Install dependencies
conda env create -f environment.yml
conda activate glycoaudit

# Or use pip
pip install -r requirements.txt
```

### Minimal Example

```bash
# 1. Download data (~ 2-3 hours, rate-limited)
python scripts/download/01_run_stage.py --config configs/mirror.yaml --stage 1

# 2. Compute features (~ 10-15 minutes)
python scripts/compute/compute_ssv.py

# 3. Run benchmark (~ 30 minutes)
python scripts/evaluate/run_benchmark.py

# 4. View results
cat benchmark_results/benchmark_summary.csv
```

---

## Benchmark Statistics

| Metric | Value | Description |
|--------|-------|-------------|
| **Glycans** | 342 | Structurally resolved glycans with 3D conformations |
| **Agents** | 196 | Lectins (n=153) and antibodies (n=43) |
| **Interactions** | 1,523 | Experimentally validated binding pairs |
| **Joinability** | 31.2% | Fraction of source data successfully joined (342/1097) |
| **Data Sources** | 7 | GlyTouCan, GlycoShape, CFG, SugarBind, GlyGen, GlycoPOST, PRIDE |

### Performance

**IID Split (Random 80/20):**
- MRR: 0.592 (95% CI: [0.509, 0.672])
- Recall@5: 0.324 (vs. random 0.015)
- AUPRC-LB: 0.368 (vs. random 0.054)

**Scaffold Split (Structure-aware):**
- MRR: 0.279 ± 0.159 (significant degradation)
- Recall@5: 0.350 (still above random)

All metrics significant vs. random and permutation baselines (p < 0.001, Holm-corrected).

---

## Repository Structure

```
GlycoAudit/
├── src/glycoaudit/          # Core library
│   ├── mirror/              # Data source downloaders
│   ├── features/            # SSV/GCV feature extraction
│   ├── evaluation/          # PU-ranking framework
│   └── analysis/            # Statistical analysis
│
├── scripts/                 # Executable scripts
│   ├── download/            # Data acquisition
│   ├── compute/             # Feature computation
│   ├── evaluate/            # Benchmark experiments
│   ├── analyze/             # Result analysis
│   └── release/             # Figure/table generation
│
├── benchmark_results/       # Complete benchmark outputs
│   ├── E1_ensemble_sensitivity_v2/
│   ├── E2_pu_ranking/
│   ├── E5_split_robustness_v2/
│   ├── E6_size_control/
│   └── ... (see docs/benchmark_schema.md)
│
├── configs/                 # Configuration files
├── data/                    # Data manifests (raw data downloaded separately)
├── docs/                    # Documentation
├── examples/                # Usage examples
└── tests/                   # Basic tests
```

---

## Data Sources

GlycoAudit integrates data from major glycobiology resources:

| Source | Type | Purpose | Records |
|--------|------|---------|---------|
| **GlyTouCan** | Registry | Glycan ID normalization | 260k+ |
| **GlycoShape** | Structure | 3D conformations | Library |
| **SugarBind** | Binding | Curated lectin-glycan pairs | Literature |
| **CFG** | Binding | High-throughput glycan arrays | Experimental |
| **GlyGen** | Metadata | Cross-references | Integrated |
| **GlycoPOST** | Mass spec | Glycoproteomics metadata | Projects |
| **PRIDE** | Mass spec | Proteomics archives | Metadata |

Data acquisition is **rate-limited** and **reproducible** with full provenance tracking.

---

## Features

### SSV (Structural Shape Vectors)
8 geometric descriptors extracted from 3D PDB structures:
- `n_atoms`, `n_residues`: Size
- `radius_of_gyration`, `max_pair_distance`: Extent
- `compactness`: Shape
- `branch_proxy`, `terminal_proxy`, `exposure_proxy`: Topology

### GCV (Graph Contact Vectors)
8 graph-based descriptors:
- `contact_density`, `long_range_contact_fraction`
- `mean/sd_residue_neighbor_count`
- `torsion_diversity`, `graph_laplacian_spectral_gap`
- `core_periphery_ratio`, `max_contact_distance_seq`

---

## Experiments

The benchmark includes **12 main experiments** validating different aspects:

| Experiment | Purpose | Key Finding |
|------------|---------|-------------|
| **E1** | Ensemble sensitivity | Bootstrap CV = 0.034 (stable) |
| **E2** | PU ranking + nulls | MRR >> random/permutation |
| **E4** | Runtime scaling | <1ms per glycan |
| **E5** | Split robustness | Scaffold split degrades performance |
| **E6** | Size control | Signal robust beyond size |
| **E7** | Preference profiling | Clear SSV preferences (Cliff's δ ≥ 0.71) |
| **E8** | DL baseline | SSV competitive with transformers |
| **E9** | Few-shot | Performance degrades with k=1,2,3 |
| **E10** | GCV failure analysis | GCV weaker than SSV |
| **E11** | Multi-prototype | Helps for high-n agents |
| **Cross-source** | Transfer asymmetry | CFG→SugarBind 2× better than reverse |
| **Case studies** | Biological validation | 3 agents (success/mixed/failure) |

See `benchmark_results/README.md` for complete documentation.

---

## Reproducibility

All results are **fully reproducible** with documented provenance:

```bash
# Reproduce specific experiment
python scripts/evaluate/e2_pu_ranking.py --seed 42

# Reproduce all experiments (~ 6-8 hours)
bash scripts/reproduce_all.sh

# Verify against published results
python scripts/release/verify_results.py
```

### Random Seeds
All experiments use **seed=42** for deterministic results.

### Hardware Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Storage**: 5GB (with downloaded data: ~500MB compressed)

---

## Citation

If you use GlycoAudit in your research, please cite:

```bibtex
@article{glycoaudit2026,
  title={GlycoAudit: A Semantically Audited Benchmark for Glycan-Binding Interaction Prediction},
  author={[Author names]},
  journal={[Journal]},
  year={2026},
  note={Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Benchmark data is released under CC-BY 4.0.

---

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Quick Start Tutorial](docs/quickstart.md)** - Step-by-step walkthrough
- **[Benchmark Schema](docs/benchmark_schema.md)** - Result format specification
- **[Data Sources](docs/data_sources.md)** - Source documentation and provenance
- **[Reproducibility Guide](docs/reproducibility.md)** - How to reproduce paper results
- **[API Reference](docs/api_reference.md)** - Code documentation

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/GlycoAudit/issues)
- **Questions**: [GitHub Discussions](https://github.com/your-username/GlycoAudit/discussions)
- **Email**: [contact email]

---

## Acknowledgments

This project integrates data from multiple public glycobiology resources. We thank the maintainers of GlyTouCan, GlycoShape, CFG, SugarBind, GlyGen, GlycoPOST, and PRIDE for making their data publicly available.

---

**Status**: Manuscript in preparation | Repository under active development

