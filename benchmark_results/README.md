# Public Glyco Mirror Benchmark Release v1

This directory contains the complete benchmark results for the Public Glyco Mirror resource, evaluating Structural Shape Vector (SSV) features for glycan-agent binding prediction under a positive-unlabeled (PU) ranking framework.

## Overview

All values in this release are computed from real experimental runs. No post-hoc tuning, manual adjustment, or fabrication of results was performed. Each metric can be traced to its source file as documented below.

## Experiments

### E1: Ensemble Sensitivity
**Purpose**: Quantify structural feature stability across the glycan dataset.

**Key finding**: All 125 glycans derive from ES-B tier (GlycoShape/GLYCAM-Web computational pipeline). Because each glycan has a single conformer source, cross-source metrics (ICC, Spearman rho, upstream dominance ratio R) cannot be computed. Bootstrap-based coefficient of variation is reported as a stability proxy (median CV = 0.034 across features).

**Files (v1)**:
- `E1_ensemble_sensitivity/summary.csv` - Per-feature statistics across 125 glycans
- `E1_ensemble_sensitivity/per_glycan_feature.csv` - Detailed per-glycan, per-feature values
- `E1_ensemble_sensitivity/provenance_rules.json` - Tier classification rules
- `E1_ensemble_sensitivity/figures/` - Feature distribution plots

**Files (v2 - extended analysis)**:
- `E1_ensemble_sensitivity_v2/summary.csv` - Updated per-feature statistics with bootstrap CV
- `E1_ensemble_sensitivity_v2/per_glycan_feature.csv` - Extended per-glycan analysis (1000 rows)
- `E1_ensemble_sensitivity_v2/provenance_rules.json` - Documented limitations for single-source data
- `E1_ensemble_sensitivity_v2/metadata.json` - Execution metadata with input hashes
- `E1_ensemble_sensitivity_v2/figures/` - ICC/R distribution plots (showing N/A due to single source)

### E2: PU Ranking Evaluation
**Purpose**: Evaluate prototype-based ranking under positive-unlabeled setting with null baselines.

**Key finding**: Observed MRR = 0.620, Recall@5 = 0.537 significantly exceed both random (MRR = 0.089) and permutation (MRR = 0.433) baselines with p < 0.001 (Holm-corrected).

**Files**:
- `E2_pu_ranking/aggregate_metrics.csv` - Observed and baseline summary statistics
- `E2_pu_ranking/null_distributions.csv` - Full null distribution data (1000 repeats each)
- `E2_pu_ranking/pvalues.json` - Computed p-values and Holm correction results
- `E2_pu_ranking/figures/` - Null distribution histograms

### E4: Runtime Scaling
**Purpose**: Benchmark SSV computation performance across dataset scales.

**Key finding**: Optimized configuration achieves < 1 ms per glycan (0.0006 s/glycan at N=125), meeting practical deployment requirements.

**Files**:
- `E4_runtime_scaling/runtime_table.csv` - Runtime measurements across configurations and scales
- `E4_runtime_scaling/hardware.json` - Hardware specification for reproducibility
- `E4_runtime_scaling/figures/` - Scaling plots

### E5: Split Robustness
**Purpose**: Evaluate generalization under agent-holdout and glycan scaffold splits.

**Key finding**: Performance degrades under scaffold splits (MRR = 0.28 ± 0.16 vs. 0.62 in standard evaluation), indicating generalization challenges for structurally distinct glycans. Scaffold split performance significantly exceeds random baselines (p = 0.016 for MRR) but does not reach significance versus permutation baselines (p = 0.11), indicating that structural signal is attenuated for topologically distinct test glycans.

**Files (v1)**:
- `E5_split_robustness/agent_holdout_metrics.csv` - Leave-one-positive-out results per agent
- `E5_split_robustness/scaffold_split_metrics.csv` - GroupKFold results by topology signature
- `E5_split_robustness/scaffold_rules.json` - Scaffold grouping criteria

**Files (v2 - with null baselines)**:
- `E5_split_robustness_v2/scaffold_split_observed_by_fold.csv` - Per-fold observed metrics
- `E5_split_robustness_v2/scaffold_split_null_by_fold.csv` - Per-fold null distributions (1000 repeats each)
- `E5_split_robustness_v2/scaffold_split_aggregate.csv` - Aggregated observed vs null statistics
- `E5_split_robustness_v2/pvalues.json` - P-values and Holm correction results
- `E5_split_robustness_v2/metadata.json` - Execution metadata with input hashes
- `E5_split_robustness_v2/figures/` - Null distribution overlay and degradation plots

### E6: Size Control
**Purpose**: Guard against trivial size confounds in binding prediction.

**Key finding**: Residualized features (size-corrected) maintain or improve performance (Recall@5 = 0.60 vs. 0.54). Observed metrics significantly exceed size-matched permutation null (p < 0.005), confirming predictive signal beyond size alone.

**Files**:
- `E6_size_control/residualized_vs_full.csv` - Full vs. residualized feature comparison
- `E6_size_control/size_strata_metrics.csv` - Performance by size tercile
- `E6_size_control/size_matched_null.csv` - Size-matched permutation null distribution
- `E6_size_control/summary.json` - Aggregated size-control statistics

## Summary Files

- `benchmark_summary.csv` - Headline metrics in tabular format with source file references
- `benchmark_summary.json` - Machine-readable version with full provenance

## Manuscript Mapping

| Manuscript Element | Source File |
|-------------------|-------------|
| Table 3 (PU Results) | `E2_pu_ranking/aggregate_metrics.csv` |
| Table 4 (Size Control) | `E6_size_control/residualized_vs_full.csv`, `E6_size_control/summary.json` |
| Figure 2 (E1 Distributions) | `E1_ensemble_sensitivity/figures/feature_distributions.pdf` |
| Figure 3 (E2 Null Distributions) | `E2_pu_ranking/figures/combined_null_distributions.pdf` |
| Figure 4 (E4 Runtime) | `E4_runtime_scaling/figures/runtime_scaling.pdf` |
| Section E5 Results | `E5_split_robustness/scaffold_split_metrics.csv` |

## Reproducibility

All experiments were executed with:
- Random seed: 42
- Python 3.10+
- Dependencies: numpy, pandas, scipy, scikit-learn, biopython, matplotlib

### How to Reproduce

```bash
# E1_v2: Ensemble sensitivity analysis
python scripts/run_e1_ensemble_sensitivity_v2.py

# E5_v2: Scaffold split null baselines
python scripts/run_e5_scaffold_split_nulls_v2.py

# Regenerate benchmark summary
python scripts/build_benchmark_release_summary.py
```

Execution logs are stored in `logs/` with timestamps.

The full execution pipeline is available in `scripts/run_all_benchmarks.py`.

## Data Integrity

- All numeric values are copied verbatim from computed outputs
- No smoothing, normalization, or rounding was applied during aggregation
- Source files for each metric are explicitly documented
- Empty or placeholder values are marked as NA where applicable

## License

Benchmark results released under CC-BY 4.0. See main repository for full license terms.

---

*Generated from Public Glyco Mirror benchmark pipeline*
