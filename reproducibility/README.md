# Anonymous reproducibility index

This directory is the single source-side index for reproducing the frozen E70 analysis and its
paper values.  It contains no author metadata, machine-specific paths, raw motion, checkpoints, or
final three-seed claim.  The current small report is the labeled seed-0 snapshot; the authoritative
three-seed report remains pending.

## Contents

- [Environment and data contract](environment.md)
- [Exact launch, analysis, and rendering commands](commands.md)
- [Reverified repository hashes](manifest.json)
- [Anonymous seed-0 analyzer snapshot](reports/e70_seed0_analysis.json)
- [Frozen E70 protocol](../docs/E70_MULTITRAJ_PROTOCOL.md)
- [Frozen 69-pair manifest](../autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json)
- [Negative-results ledger](../docs/EXPERIMENT_LOG.md)
- [Secondary temporal-block preregistration](../docs/E70_SECONDARY_TEMPORAL_BLOCK_ANALYSIS.md)

The raw LAFAN1-derived motions and student/teacher checkpoints are intentionally not redistributed.
With licensed artifacts mounted at `ARTIFACT_ROOT`, the commands reproduce the analyzer JSON and
then the LaTeX values used by Table I.  The included seed-0 snapshot was regenerated from the
hash-bound analyzer after its hierarchical output schema was frozen; it is not a copy of the older
pre-hierarchical archival JSON.  Every behavioral input is SHA-256-bound inside the analyzer output,
while `manifest.json` binds the source-side protocol and implementation.

Run the local audit with:

```bash
python scripts/audit_reproducibility_bundle.py \
  --bundle-root reproducibility --repo-root . \
  --artifact-root "$ARTIFACT_ROOT"
```

Omit `--artifact-root` when checking only the source bundle.  Supplying it additionally replays all
behavioral hashes in the anonymous analyzer snapshot.
