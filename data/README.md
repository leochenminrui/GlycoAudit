# Data Directory

This directory contains data manifests, checksums, and sample outputs for the GlycoAudit benchmark.

## Structure

```
data/
├── README.md                 # This file
├── manifests/                # Source data manifests
│   ├── glytoucan_manifest.csv
│   ├── glycoshape_manifest.csv
│   ├── sugarbind_manifest.csv
│   └── ...
├── checksums/                # Data integrity verification
│   ├── ssv_checksums.md5
│   ├── binding_checksums.md5
│   └── ...
└── sample_outputs/           # Example outputs for testing
    ├── ssv_sample.csv        # Sample SSV features (10 glycans)
    ├── binding_sample.csv    # Sample binding labels
    └── joined_sample.csv     # Sample joined dataset
```

## Data Not Included in Repository

Due to size constraints, the following data are **NOT** included in the Git repository:

### Large Data (Download Required)

- `data/raw/` - Raw downloaded data from sources (~312MB)
  - GlyTouCan registry
  - GlycoShape PDB structures
  - CFG glycan array data
  - SugarBind binding records

- `data/ssv/` - Computed SSV features (~5MB)
- `data/gcv/` - Computed GCV features (~5MB)
- `data/binding/` - Processed binding labels (~2MB)
- `data/joined/` - Joined feature+label datasets (~8MB)

### How to Obtain Data

**Option 1: Download from source (recommended for reproducibility)**
```bash
# Download all data sources (2-3 hours, rate-limited)
python scripts/download/01_run_stage.py --config configs/mirror.yaml --stage all

# Compute features (10-15 minutes)
python scripts/compute/compute_ssv.py
python scripts/compute/compute_gcv.py
```

**Option 2: Download pre-computed archive (faster)**
```bash
# Download from Zenodo/Figshare (link in paper)
wget https://zenodo.org/record/XXXXXX/files/glycoaudit_data_v1.tar.gz

# Extract
tar -xzf glycoaudit_data_v1.tar.gz -C data/
```

**Option 3: Use sample data (for testing)**
```bash
# Sample data already included in repository
python scripts/evaluate/run_benchmark.py --use-sample-data
```

## Manifests

The `manifests/` directory contains **CSV manifests** that document:
- Source URLs
- Download timestamps
- Record counts
- Data provenance

These are included in the repository to enable reproducible downloads.

## Checksums

The `checksums/` directory contains **MD5 checksums** for verifying data integrity:

```bash
# Verify downloaded data
cd data/raw
md5sum -c ../checksums/raw_data_checksums.md5

# Verify computed features
cd data/ssv
md5sum -c ../checksums/ssv_checksums.md5
```

## Sample Outputs

The `sample_outputs/` directory contains **small examples** (~10 glycans) for:
- Quick testing without full download
- Understanding data formats
- Validating pipeline setup

## Data Provenance

All data in GlycoAudit is derived from public sources:

| Directory | Source | License | URL |
|-----------|--------|---------|-----|
| `raw/glytoucan/` | GlyTouCan | CC-BY | https://glytoucan.org |
| `raw/glycoshape/` | GlycoShape | CC-BY | https://glycoshape.io |
| `raw/sugarbind/` | SugarBind | CC-BY | https://sugarbind.expasy.org |
| `raw/cfg/` | CFG | Custom* | https://functionalglycomics.org |
| `raw/glygen/` | GlyGen | CC0 | https://glygen.org |
| `raw/glycopost/` | GlycoPOST | CC-BY | https://glycopost.glycosmos.org |
| `raw/pride/` | PRIDE | CC0 | https://www.ebi.ac.uk/pride |

*CFG data usage: Please review CFG's terms of use before redistribution.

## Data Updates

The benchmark uses data snapshots from **December 2025**. To update:

```bash
# Re-download with new timestamp
python scripts/download/01_run_stage.py --config configs/mirror.yaml --force-refresh

# Regenerate features
python scripts/compute/compute_ssv.py --force
```

Note: Results may differ slightly with updated data.

## Size Estimates

| Component | Compressed | Uncompressed |
|-----------|------------|--------------|
| Raw data | ~100MB | ~312MB |
| SSV features | ~1MB | ~5MB |
| GCV features | ~1MB | ~5MB |
| Binding labels | ~500KB | ~2MB |
| Joined datasets | ~2MB | ~8MB |
| **Total** | **~105MB** | **~332MB** |

## Questions?

See the main [README](../README.md) or [Data Sources Documentation](../docs/data_sources.md) for more information.
