# GlycoAudit

This is the minimal public research package aligned with the manuscript
"GlycoAudit: structural joinability and annotation degree shape glycan-agent
relation ranking" and its reviewer response.

The repository contains the preserved public dataset, the code required to
reproduce the reported analyses, and the machine-readable reference results. It
intentionally contains no manuscript or response LaTeX, PDFs,
review correspondence, tests, figure-generation code, figure source data,
submission packaging, internal audit logs, caches, virtual environments, or
withdrawn historical result code.

## Scientific scope

The quantitative benchmark uses SugarBind positive relations. CFG contributes a
6,035-entry experiment catalogue as provenance metadata only; the historical RFU
files, analyzed subset, and positivity threshold were not recoverable. Missing
relations are unlabeled, not confirmed negatives. Cross-source quantitative
transfer is therefore not estimable from this release.

The strict structure contract retains 44 of 173 resolved glycans. The primary
held-out-positive evaluation contains 159 SugarBind agent identifiers, 697
positive relations, 645 agent-folds, and a fixed 44-glycan candidate universe.

The popularity control uses two explicitly separated scopes:

- scoring reference pool: all 765 strictly joined relations from 227 agents;
- evaluation population: 697 relations from 159 agents across 645 folds.

For each target agent, every target-agent relation is removed from its popularity
scores. This definition reproduces the manuscript values (MRR 0.491, Recall@5
0.456, and known-positive AP 0.489).

Prototype rank sensitivity uses the manuscript statistic: for each
training-positive deletion, signed held-out-rank changes are averaged within the
fold, the absolute value of that fold mean is taken, and the result is averaged
over perturbations. This reproduces 1.73, 3.38, and 3.86 for WURCS, SSV, and GCV.

## Contents

```text
data/                    source snapshots, structures, evidence, mappings,
                         features, folds, and frozen benchmark results
glycoaudit/              ranking and join-audit implementation
scripts/                 reconstruction, feature computation, and analyses
results/                 machine-readable results used by the paper/response
REFERENCE_RESULTS.json   authoritative numerical contract
```

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproduce

Regenerate the frozen primary summaries:

```bash
python scripts/summarize_primary_results.py
```

Regenerate joinability, paired comparisons, prototype perturbation, support,
scaffold-feasibility, agent-identity and coordinate-sensitivity results:

```bash
python scripts/reproduce_additional_analyses.py
```

Regenerate popularity, tie, entity-exclusion and degree-stratified controls:

```bash
python scripts/reproduce_annotation_controls.py
```

Run an individual held-out-positive evaluation with
`scripts/reproduce_primary_results.py`. Use `--preprocessing tfidf` for WURCS
and `--preprocessing standardize` for SSV/GCV. The released fold and candidate
manifests must be supplied.

Validate the preserved data release and the manuscript numerical contract:

```bash
python scripts/validate_data_release.py data
python scripts/validate_results.py
```

## Interpretation

WURCS is the strongest molecular representation evaluated under the common
protocol. The non-molecular popularity control is higher for MRR and
known-positive AP, whereas WURCS is higher for Recall@5. The results demonstrate
that structural joinability, annotation degree, candidate construction and metric
choice jointly shape apparent benchmark performance.

Code is released under the MIT License. Source records and structures retain
their upstream provenance in the released metadata; users are responsible for
complying with the terms of the originating resources.
