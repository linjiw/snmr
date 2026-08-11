# Exact E70 commands

Set `ARTIFACT_ROOT` to the absolute directory containing the `e67/`, `e69/`, and `e70/` artifact
subdirectories.  Activate the repository environment before any CPU command:

```bash
source scripts/activate_snmr.sh
```

## Frozen launch order

The complete queue, including the 26,000-MiB capacity gate, explicit-control gate, all five arms,
evaluation, and final analyzer, is:

```bash
E70_FULL_SEEDS=1 bash scripts/run_e70_multitraj.sh
```

For a clean, explicitly authorized round-0 cell restart, the helper's exact command form is:

```bash
bash scripts/run_e70_cell.sh SEED TAG train
bash scripts/run_e70_cell.sh SEED TAG general
bash scripts/run_e70_cell.sh SEED TAG ambiguity
```

`SEED` is one of `0`, `1`, or `2`.  `TAG` is run in the registered order `explicit`, `snmr`,
`time`, `proprio`, `shuffled`.  The helper maps those tags to C/A/T/B/S and refuses nonempty
training cells.  It does not implement partial-checkpoint continuation.  Do not lower the capacity
gate or reorder cells.

## Primary analyzer

```bash
python scripts/analyze_e67_results.py \
  --students_root "$ARTIFACT_ROOT/e70/students" \
  --teacher_reports \
    "$ARTIFACT_ROOT/e69/teacher_reports/walk1_subject1_eval404.json" \
    "$ARTIFACT_ROOT/e67/teacher_reports/walk1_subject5_eval404.json" \
  --seeds 0 1 2 \
  --protocol "E70 preregistered analysis v1" \
  --out "$ARTIFACT_ROOT/e70/analysis_seed0-1-2.json"
```

The final command must fail if any seed, arm, start grid, evaluation seed, or registered 69-pair
cluster set is missing.  The final JSON's `inputs` array is the behavioral artifact hash manifest.

For an exact replay of the included labeled seed-0 snapshot before the final seed set exists, use
the same command with `--seeds 0` and a separate output path.  With the manifest-bound analyzer and
inputs, the non-anonymized JSON has SHA-256
`9059575b3e7d983e695b93a364da5e06eafb609576d52e7806d170a553ac13f7`; running
`scripts/build_reproducibility_report.py` on it produces the anonymous report in this bundle.

## Secondary dependence check and paper values

```bash
python scripts/analyze_e70_temporal_blocks.py \
  --students-root "$ARTIFACT_ROOT/e70/students" \
  --ambiguity-precheck autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json \
  --seeds 0 1 2 \
  --out "$ARTIFACT_ROOT/e70/secondary_temporal_blocks_seed0-1-2.json"

python scripts/render_e70_paper_values.py \
  --analysis "$ARTIFACT_ROOT/e70/analysis_seed0-1-2.json" \
  --out paper/e70_results.tex

python scripts/render_e70_effect_figure.py \
  --analysis "$ARTIFACT_ROOT/e70/analysis_seed0-1-2.json" \
  --out paper/e70_effect_figure.tex
```

No displayed final E70 quantity is transcribed by hand.  The paper-value and figure generators
reject non-final seed sets unless the figure command is explicitly marked as a non-final preview.
