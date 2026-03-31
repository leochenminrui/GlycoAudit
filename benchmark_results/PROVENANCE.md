# Benchmark Release v1 Provenance

This document traces each experiment folder to its generating script and input file checksums.

## Experiment Provenance

### E1: Ensemble Sensitivity

| Version | Script | Status |
|---------|--------|--------|
| v1 | `scripts/run_all_benchmarks.py` | Original |
| v2 | `scripts/run_e1_ensemble_sensitivity_v2.py` | Extended (ICC/rho/R analysis) |

**v2 Input Files**:
- `data/ssv/targeted_sugarbind_v0/ssv_table.csv`
- SHA256 recorded in `E1_ensemble_sensitivity_v2/metadata.json`

**v2 Key Outputs**:
- `summary.csv`: Per-feature aggregate statistics
- `per_glycan_feature.csv`: 1000 rows (125 glycans × 8 features)
- `provenance_rules.json`: ES-B tier assignment and documented limitations

**Limitation**: Single-source data (all ES-B tier) precludes ICC(2,1), Spearman rho, and dominance ratio R computation.

---

### E2: PU Ranking Evaluation

| Script | Repeats | Status |
|--------|---------|--------|
| `scripts/run_all_benchmarks.py` | 1000 | Complete |

**Input Files**:
- `data/ssv/targeted_sugarbind_v0/ssv_table.csv`
- `data/binding/sugarbind_v0/labels.csv`

**Key Outputs**:
- `aggregate_metrics.csv`: Observed MRR = 0.620, Recall@5 = 0.537
- `pvalues.json`: All p < 0.001 (Holm-corrected)
- `figures/combined_null_distributions.pdf`: Main manuscript figure

---

### E4: Runtime Scaling

| Script | Status |
|--------|--------|
| `scripts/run_all_benchmarks.py` | Complete |

**Key Outputs**:
- `runtime_table.csv`: Optimized = 0.0006 s/glycan
- `hardware.json`: Intel Xeon Gold 6346

---

### E5: Split Robustness

| Version | Script | Repeats | Status |
|---------|--------|---------|--------|
| v1 | `scripts/run_all_benchmarks.py` | N/A | Original |
| v2 | `scripts/run_e5_scaffold_split_nulls_v2.py` | 1000 | Extended |

**v2 Input Files**:
- `data/ssv/targeted_sugarbind_v0/ssv_table.csv`
- `data/binding/sugarbind_v0/labels.csv`
- SHA256 recorded in `E5_split_robustness_v2/metadata.json`

**v2 Key Outputs**:
- `scaffold_split_aggregate.csv`: MRR = 0.279 ± 0.160
- `pvalues.json`: p = 0.016 (vs random), p = 0.114 (vs perm)
- `figures/scaffold_split_null_overlay.pdf`: Main manuscript figure

**Key Finding**: Scaffold splits significantly exceed random (p < 0.05) but not permutation baselines after Holm correction, indicating attenuated structural signal for topologically distinct test glycans.

---

### E6: Size Control

| Script | Status |
|--------|--------|
| `scripts/run_all_benchmarks.py` | Complete |

**Key Outputs**:
- `summary.json`: Residualized Recall@5 = 0.601, p < 0.005 vs size-matched null

---

## Script Inventory

| Script | Purpose | Seed |
|--------|---------|------|
| `scripts/run_e1_ensemble_sensitivity_v2.py` | E1 v2 analysis | 42 |
| `scripts/run_e5_scaffold_split_nulls_v2.py` | E5 v2 null baselines | 42 |
| `scripts/build_benchmark_release_summary.py` | Aggregate summary | N/A |

---

## Execution Logs

All logs are stored in `logs/` with format `{experiment}_{timestamp}.log`.

---

## Checksums

File integrity can be verified via SHA256 checksums stored in each experiment's `metadata.json`.

---

*Generated: 2024-12-27*
