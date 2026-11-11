# Reproducing GlycoAudit

This guide covers analysis from the included data and features, reconstruction
of summary tables from saved results, and checks against the paper's numerical
references. Use the installation commands in the [README](../README.md), then
run the commands below from the repository root.

The documented primary and additional analysis commands were verified with
Python 3.12.14, NumPy 1.26.4, pandas 3.0.6, SciPy 1.17.1, Biopython 1.88, and
HTTPX 0.28.1. The three primary point estimates and bootstrap summaries match
the included references; the additional analysis CSVs match within absolute
numerical tolerance `1e-12`.

## Data and evaluation contract

The quantitative benchmark uses SugarBind positive relations. Identifier
resolution yields 173 glycans; the strict accession, branch, and sequence
contract retains 44. The evaluated population contains 159 SugarBind agent
identifiers, 697 relations, 645 agent-folds, and 6,996 candidate-manifest rows.
Agent counts refer to source identifiers, not independent biological entities.

The annotation-degree control has two distinct populations:

| Role | Agents | Positive relations |
| --- | ---: | ---: |
| Scoring reference pool, all strictly joined relations | 227 | 765 |
| Held-out-positive evaluation population | 159 | 697 |

For each target agent, every target-agent relation is removed from the scoring
pool. The resulting point estimates are MRR 0.49134502083600373, Recall@5
0.4559748427672956, and known-positive AP 0.4894782696522135.

The preserved CFG experiment catalogue contains 6,035 metadata entries.
Historical RFU measurements, the analyzed subset, and positivity thresholds
were not recoverable. CFG is metadata only in this release; quantitative CFG
ranking, confirmed-nonbinding classification, and cross-source transfer are
not supported by these inputs. See [data/README.md](../data/README.md),
[CFG_DATA_STATUS.json](../data/CFG_DATA_STATUS.json), and
[RELEASE_MANIFEST.json](../data/RELEASE_MANIFEST.json) for provenance and counts.

The data and existing scientific implementation are frozen at
[`68d1d53168a738b73af684e5e0d821da6e186d60`](https://github.com/leochenminrui/GlycoAudit/tree/68d1d53168a738b73af684e5e0d821da6e186d60).
The scientific calculations and data remain unchanged. The artifact validator
also accepts an untracked local `.venv` and generated caches; declared release
files and Git-tracked files are always inspected. The root
`CHECKSUMS.sha256` covers the current source package; `data/CHECKSUMS.sha256`
covers the preserved data files. Keep the exact checkout revision with any
new results.

## Check the supplied artifact

```bash
python scripts/validate_data_release.py data
python scripts/validate_results.py
```

The data validator checks schemas, counts, structural joins, fold/candidate
consistency, and the data checksums. The result validator compares the supplied
reference results with [REFERENCE_RESULTS.json](../REFERENCE_RESULTS.json),
using absolute tolerance `1e-12` for the checked numerical values. These checks
validate the included artifact; they do not rerun the ranking analysis.

To verify every file listed in the current source checksum manifest:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

## Recompute the primary ranking

The primary inputs are the evidence table, one-row-per-glycan feature tables,
released agent-positive folds, and candidate manifest. Keep the two manifest
arguments in every paper-aligned run. WURCS requires `tfidf`; SSV and GCV
require `standardize`.

```bash
python scripts/reproduce_primary_results.py \
  --features data/wurcs_features.csv \
  --labels data/evidence_records.csv \
  --preprocessing tfidf \
  --fold-manifest data/positive_holdout_folds.csv \
  --candidate-manifest data/candidate_manifest.csv \
  --outdir results/reproduced/wurcs

python scripts/reproduce_primary_results.py \
  --features data/ssv_features.csv \
  --labels data/evidence_records.csv \
  --preprocessing standardize \
  --fold-manifest data/positive_holdout_folds.csv \
  --candidate-manifest data/candidate_manifest.csv \
  --outdir results/reproduced/ssv

python scripts/reproduce_primary_results.py \
  --features data/gcv_features.csv \
  --labels data/evidence_records.csv \
  --preprocessing standardize \
  --fold-manifest data/positive_holdout_folds.csv \
  --candidate-manifest data/candidate_manifest.csv \
  --outdir results/reproduced/gcv_4_5
```

Each output directory contains `aggregate_metrics.json`,
`metrics_per_agent.csv`, `metrics_per_agent_fold.csv`, and `protocol.json`.
Defaults use seed 42 and 2,000 agent-bootstrap replicates. Add
`--null-repeats 100` to a run to recompute its random-ranking and
feature-permutation controls; this also creates `null_distributions.csv` and
`null_pvalues.json`. Compare the corresponding point estimates and confidence
limits with `data/results/{wurcs,ssv,gcv_4_5}/aggregate_metrics.json`; the
null-control references are in the same directories. Floating-point output can
differ at the last decimal places across numerical-library versions.

These analyses run on a CPU. Runtime depends on the representation and number
of resampling replicates; null controls and sensitivity analyses take longer
than the primary ranking. No model training or GPU checkpoint is required.

## Rebuild summary tables and additional analyses

| Entry point | Included inputs | Outputs |
| --- | --- | --- |
| `summarize_primary_results.py` | Saved primary/null results, support strata, per-model features, evidence, candidates and folds | Primary, null, support, GCV-cutoff, and coordinate-model summary CSVs |
| `reproduce_additional_analyses.py` | Evidence, identifiers, structures, features, folds and saved per-agent results | Joinability, paired comparisons, prototype perturbation, support, scaffold feasibility, identity and coordinate sensitivity |
| `reproduce_annotation_controls.py` | All joined relations, agent metadata, candidates and folds | Annotation-degree baseline, tie sensitivity, entity-exclusion and degree-stratified controls, count funnel and identifier audit |

Write regenerated outputs under the ignored `results/reproduced/` directory
to retain the included references:

```bash
python scripts/summarize_primary_results.py \
  --output-dir results/reproduced/primary_summary
python scripts/reproduce_additional_analyses.py \
  --outdir results/reproduced/additional
python scripts/reproduce_annotation_controls.py \
  --outdir results/reproduced/annotation
```

The summary script rebuilds five tables from saved estimates and recomputes
coordinate-model sensitivity from the released per-model features. It is not
a complete rerun of the original primary or null analyses. The other two
scripts recompute their analyses from the included inputs and references;
consult each emitted `analysis_manifest.json` for generated outputs.

For prototype perturbation, the paper's statistic is the mean over
training-positive deletions of the absolute fold-mean signed held-out-rank
change. It yields 1.7319762510602206 (WURCS), 3.3810432569974553 (SSV), and
3.858001130901894 (GCV 4.5 Å). These are not the mean absolute changes of
individual held-out ranks.

The result validator reads the supplied `data/results/` and `results/`
references, not `results/reproduced/`. Compare a regenerated analysis with its
matching reference directory before interpreting it as a successful rerun.

## Source and data terms

The [LICENSE](../LICENSE) separates MIT-licensed code from CC BY 4.0 benchmark
data. Preserve the attribution and source-specific provenance distributed
with the data. Use the [journal article citation](../README.md#citation) when
using the code, benchmark, or protocol, and record the checkout revision.
