# GlycoAudit

GlycoAudit is an auditable benchmark for ranking glycan–agent relations from
incomplete binding evidence. It makes the path from source records to matched
structures, candidate sets, evaluation folds, and ranking results inspectable.
Use it to compare glycan representations under a common protocol and measure
how structural availability and annotation degree affect the comparison.

**Paper:** Minqi Chen, Minrui Chen, Guanjie Zou, and Lunxu Liu.
[GlycoAudit: structural joinability and annotation degree shape glycan–agent relation ranking](https://doi.org/10.1093/bioadv/vbag308).
*Bioinformatics Advances*, 2026, vbag308.

[Quick start](#quick-start) · [Results](#findings) ·
[Reproduction guide](docs/reproduction.md) · [Data](data/README.md) ·
[Citation](#citation)

## Findings

Strict accession, branch, and sequence matching retains **44 of 173 resolved
SugarBind glycans (25.4%)**. The held-out-positive benchmark evaluates **697
relations from 159 agent identifiers across 645 folds**, with a fixed 44-glycan
candidate universe.

| Representation or control | MRR | Recall@5 | Known-positive AP |
| --- | ---: | ---: | ---: |
| WURCS with fold-fitted TF–IDF | 0.381 | 0.688 | 0.373 |
| SSV structural summary | 0.258 | 0.351 | 0.251 |
| GCV structural summary, 4.5 Å | 0.132 | 0.159 | 0.128 |
| Leave-one-agent-out annotation degree | 0.491 | 0.456 | 0.489 |

WURCS performs best among the evaluated molecular representations. Annotation
degree gives higher MRR and known-positive AP, whereas WURCS gives higher
Recall@5. The choice of metric therefore changes the apparent ordering of a
structural representation and a non-molecular control. Exact values are in
[REFERENCE_RESULTS.json](REFERENCE_RESULTS.json).

For each fold, the evaluator builds an agent prototype from training positives,
fits preprocessing without the held-out positives, and ranks the remaining
candidates. The annotation-degree control removes every relation of the target
agent before scoring. Missing relations remain unlabeled: the task is recovery
of held-out known positives.

## Quick start

Use Python 3.12 (the package supports Python 3.10–3.12). The public data,
features, candidate and fold manifests, and reference results are included in
the repository; the commands below need no external data download or GPU.

```bash
git clone https://github.com/leochenminrui/GlycoAudit.git
cd GlycoAudit
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run WURCS ranking from the released feature and evidence tables:

```bash
python scripts/reproduce_primary_results.py \
  --features data/wurcs_features.csv \
  --labels data/evidence_records.csv \
  --preprocessing tfidf \
  --fold-manifest data/positive_holdout_folds.csv \
  --candidate-manifest data/candidate_manifest.csv \
  --outdir results/reproduced/wurcs
```

This writes `aggregate_metrics.json`, per-agent and per-fold CSV tables, and
`protocol.json` to `results/reproduced/wurcs/`. Expected point estimates are
MRR **0.380555**, Recall@5 **0.688260**, and known-positive AP **0.373302**.
The command uses the released folds and 2,000 bootstrap replicates; null-control
reruns and additional analyses are described in the
[reproduction guide](docs/reproduction.md).

## Reproducibility artifact

The repository includes the source snapshots, join decisions and exclusions,
features, fixed candidates and folds, and numerical references supporting the
paper. [The reproduction guide](docs/reproduction.md) maps each analysis to its
inputs, command, outputs, and verification checks.

- [Data and provenance](data/README.md)
- [Data release manifest](data/RELEASE_MANIFEST.json)
- [Numerical reference contract](REFERENCE_RESULTS.json)
- [Paper-aligned scientific source at revision `68d1d53`](https://github.com/leochenminrui/GlycoAudit/tree/68d1d53168a738b73af684e5e0d821da6e186d60)

That revision freezes the scientific implementation and data before this
README update. For work based on a later checkout, record its commit alongside
the paper citation.

## Citation

**If you use GlycoAudit's code, benchmark data, or evaluation protocol, cite the
journal article:**

> Minqi Chen, Minrui Chen, Guanjie Zou, and Lunxu Liu. GlycoAudit: structural
> joinability and annotation degree shape glycan–agent relation ranking.
> *Bioinformatics Advances*, 2026, vbag308.
> https://doi.org/10.1093/bioadv/vbag308

```bibtex
@article{chen2026glycoaudit,
  author = {Chen, Minqi and Chen, Minrui and Zou, Guanjie and Liu, Lunxu},
  title = {{GlycoAudit}: structural joinability and annotation degree shape glycan--agent relation ranking},
  journal = {Bioinformatics Advances},
  year = {2026},
  pages = {vbag308},
  doi = {10.1093/bioadv/vbag308},
  url = {https://doi.org/10.1093/bioadv/vbag308}
}
```

[CITATION.cff](CITATION.cff) supplies the same article as GitHub's preferred
citation. Also record the code revision (`git rev-parse HEAD`) and retain
attribution for upstream resources listed in the data provenance.

## License

Code is available under the **MIT License**; the benchmark dataset in `data/`
is released under **CC BY 4.0**, as specified in [LICENSE](LICENSE). Source
records and structures retain their upstream provenance and applicable source
terms. Cite the GlycoAudit paper when using the benchmark data.
