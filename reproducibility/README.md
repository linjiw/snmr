# Anonymous reproducibility index

This directory is the single source-side index for reproducing the frozen E70 analysis and its
paper values.  It contains no author metadata, machine-specific paths, raw motion, or checkpoints.
The authoritative report is now the three-seed aggregate that backs Table I; the earlier seed-0
snapshot is retained beside it as append-only lineage, not as a paper claim.

## Contents

- [Environment and data contract](environment.md)
- [Exact launch, analysis, and rendering commands](commands.md)
- [Reverified repository hashes](manifest.json)
- [Anonymous three-seed analyzer report (authoritative)](reports/e70_seed0-1-2_analysis.json)
- [Anonymous seed-0 analyzer snapshot (superseded lineage)](reports/e70_seed0_analysis.json)
- [Frozen E70 protocol](../docs/E70_MULTITRAJ_PROTOCOL.md)
- [Frozen 69-pair manifest](../autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json)
- [Negative-results ledger](../docs/EXPERIMENT_LOG.md)
- [Secondary temporal-block preregistration](../docs/E70_SECONDARY_TEMPORAL_BLOCK_ANALYSIS.md)

The raw LAFAN1-derived motions and student/teacher checkpoints are intentionally not redistributed.
With licensed artifacts mounted at `ARTIFACT_ROOT`, the commands reproduce the analyzer JSON and
then the LaTeX values used by Table I.  Both included snapshots were generated from the hash-bound
analyzer after its hierarchical output schema was frozen; neither is a copy of an older
pre-hierarchical archival JSON.  Every behavioral input is SHA-256-bound inside the analyzer output,
while `manifest.json` binds the source-side protocol and implementation.

## Values a reader should reproduce

The three-seed report is the one Table I is rendered from.  Its non-anonymized source JSON has
SHA-256 `05ca3176c0a78eebc6ca49665ce092ddfe0ea423be51e7d61a0192d428ea9b5f`, which is the same digest
recorded in the `analysis_sha256` header line of `paper/e70_results.tex`.  Reproducing that file
reproduces these cluster-bootstrap contrasts over the 69 registered ambiguity pairs and 3 training
seeds:

| Contrast | Difference | 95% CI | Paper |
| --- | --- | --- | --- |
| SNMR - time | +0.1908 | [0.1239, 0.2741] | +0.191 [0.124, 0.274] |
| SNMR - shuffled | +0.1994 | [0.1270, 0.2786] | +0.199 [0.127, 0.279] |

The superseded seed-0 snapshot reports +0.1544 [0.0947, 0.2138] and +0.1871 [0.1365, 0.2376] from a
single training seed.  Those are lineage values only: a single seed cannot produce the paper's
between-seed interval, and no displayed paper quantity is taken from that file.

Run the local audit with:

```bash
python scripts/audit_reproducibility_bundle.py \
  --bundle-root reproducibility --repo-root . \
  --artifact-root "$ARTIFACT_ROOT"
```

Omit `--artifact-root` when checking only the source bundle.  Supplying it additionally replays
every behavioral hash in every anonymous analyzer report under `reports/` (44 across the two
reports: 32 three-seed and 12 seed-0).
