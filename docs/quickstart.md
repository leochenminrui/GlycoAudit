# Quick Start Guide

This guide will walk you through running GlycoAudit from scratch.

## Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB disk space
- Internet connection for data download

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/your-username/GlycoAudit.git
cd GlycoAudit

# Create conda environment
conda env create -f environment.yml
conda activate glycoaudit
```

### Option 2: pip + virtualenv

```bash
# Clone repository
git clone https://github.com/your-username/GlycoAudit.git
cd GlycoAudit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step Walkthrough

### Step 1: Download Data (~ 2-3 hours)

The first step downloads data from public glycobiology resources. This is **rate-limited** to be respectful of source servers.

```bash
# Stage 1: Core registries (GlyTouCan, GlycoShape, GlyGen)
python scripts/download/01_run_stage.py --config configs/mirror.yaml --stage 1

# This will download:
# - GlyTouCan glycan registry (~260k entries)
# - GlycoShape 3D structures (library subset)
# - GlyGen metadata
```

**Expected output:**
```
✓ Downloaded 342 glycan structures
✓ Retrieved 1097 binding records
✓ Validated checksums: 342/342 passed
```

**Common issues:**
- **Timeout errors**: Servers are sometimes slow. Retry with `--max-retries 5`
- **Rate limiting**: If blocked, wait 1 hour and resume. Progress is saved.

### Step 2: Compute Features (~ 10-15 minutes)

Extract structural descriptors from 3D conformations:

```bash
# Compute SSV (Structural Shape Vectors)
python scripts/compute/compute_ssv.py --output data/ssv/

# Compute GCV (Graph Contact Vectors) - optional
python scripts/compute/compute_gcv.py --output data/gcv/
```

**Expected output:**
```
Processing 342 glycans...
✓ SSV computed: 342/342 (100%)
✓ Output: data/ssv/expanded_v1/ssv_features.csv
```

### Step 3: Run Benchmark (~ 30 minutes)

Execute the main benchmark evaluation:

```bash
python scripts/evaluate/run_benchmark.py \
  --features data/ssv/expanded_v1/ssv_features.csv \
  --binding data/binding/expanded_v1/binding_labels.csv \
  --output benchmark_results/ \
  --seed 42
```

**Expected output:**
```
Running PU-ranking evaluation...
✓ IID split (80/20): MRR=0.592, Recall@5=0.324
✓ Random baseline: MRR=0.106 (p<0.001)
✓ Permutation baseline: MRR=0.430 (p<0.001)

Results saved to: benchmark_results/benchmark_summary.csv
```

### Step 4: View Results

```bash
# Summary table
cat benchmark_results/benchmark_summary.csv | column -t -s,

# Detailed metrics
cat benchmark_results/E2_pu_ranking/aggregate_metrics.csv
```

## What's Next?

### Run Individual Experiments

```bash
# Ensemble sensitivity
python scripts/evaluate/e1_ensemble.py

# Split robustness
python scripts/evaluate/e5_splits.py

# Size control
python scripts/evaluate/e6_size_control.py
```

### Analyze Results

```bash
# Cross-source transfer
python scripts/analyze/cross_source.py

# Joinability audit
python scripts/analyze/joinability.py

# Case studies
python scripts/analyze/case_studies.py
```

### Generate Figures

```bash
# Generate all publication figures
python scripts/release/generate_figures.py --output figures/
```

## Minimal Working Example (Python API)

```python
from glycoaudit.evaluation import pu_ranking
from glycoaudit.features import compute_ssv
import pandas as pd

# Load data
features = pd.read_csv("data/ssv/expanded_v1/ssv_features.csv")
labels = pd.read_csv("data/binding/expanded_v1/binding_labels.csv")

# Run evaluation
results = pu_ranking.evaluate(
    features=features,
    labels=labels,
    n_splits=5,
    seed=42
)

print(f"MRR: {results['mrr']:.3f}")
print(f"Recall@5: {results['recall_at_5']:.3f}")
```

## Troubleshooting

### Data Download Fails

```bash
# Check which stage failed
python scripts/download/01_run_stage.py --config configs/mirror.yaml --stage 1 --dry-run

# Resume from checkpoint
python scripts/download/01_run_stage.py --config configs/mirror.yaml --stage 1 --resume
```

### Feature Computation Errors

```bash
# Check structure files
ls data/raw/structures/ | wc -l  # Should be 342

# Recompute with verbose logging
python scripts/compute/compute_ssv.py --verbose --log-file ssv.log
```

### Import Errors

```bash
# Verify installation
pip list | grep -E "numpy|pandas|scipy|scikit-learn"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Performance Tuning

### Speed Up Data Download

```bash
# Increase parallelism (use cautiously!)
python scripts/download/01_run_stage.py --workers 4

# Skip validation for testing
python scripts/download/01_run_stage.py --skip-validation
```

### Speed Up Feature Computation

```bash
# Use multiprocessing
python scripts/compute/compute_ssv.py --workers 8

# Process subset for testing
python scripts/compute/compute_ssv.py --max-items 50
```

## Next Steps

- **[Reproducibility Guide](reproducibility.md)** - Reproduce paper results exactly
- **[Benchmark Schema](benchmark_schema.md)** - Understand result format
- **[Data Sources](data_sources.md)** - Learn about integrated databases
- **[API Reference](api_reference.md)** - Use GlycoAudit programmatically

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or share use cases
- **Email**: [contact email]
