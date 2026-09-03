# GlycoAudit data

This public dataset was reconstructed from preserved source snapshots. The
record-level files underlying the earlier study package were unavailable, so
the dataset does not claim to be a byte-for-byte recovery of that package.

The quantitative benchmark uses SugarBind evidence only. The complete 6,035-row
CFG public experiment catalogue is included as metadata, but the historical raw
and processed RFU files, analyzed subset, and true positivity thresholds are
unrecoverable. No CFG evidence, reported non-binding label, or cross-source metric
is inferred from the catalogue.

The release contains all evidence records, all structure-join exclusions,
identifier and structure maps, 44 strictly matched GlycoShape structures with 220
level-1 models, SSV/GCV/WURCS features, candidate/scaffold/positive-fold manifests,
record- and glycan-level joinability audits, held-out-positive results, null
controls, sensitivity tables, provenance metadata, and SHA-256 checksums.

Counts at each processing stage are kept separate. The source snapshot contains
1,263 records, 204 SugarBind ligand identifiers, 1,212 source-ligand--agent
relations, and 386 agents. Identifier resolution yields 1,258 records, 173
GlyTouCan accessions, 1,201 canonical glycan--agent relations, and 385 agents.
Strict structure joining yields 801 records,
44 glycans, 765 relations, and 227 agents. The primary benchmark contains 6,996
candidate rows, 697 positives, 159 agents, and 645 agent-folds.

Agent counts are deduplicated within each stage and are not additive across
stages: 83 agents have both joinable and excluded evidence records.

Validate this directory from the repository root:

```bash
python scripts/validate_data_release.py data
```

Machine-readable result files use `known_positive_ap` for average precision over
the held-out known-positive relevance set; this statistic is not claimed to be a
bound under complete labels.

See `CFG_DATA_STATUS.json` and `RELEASE_MANIFEST.json` before interpreting any
counts.
