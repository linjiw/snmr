# Analyzer Amendment

## Status

This amendment was recorded before the development outcome was known. The
defects were identified no later than `2026-07-16T22:10:58Z` (the correction
commit time), while the active policy was approximately 2,000 of 8,000
iterations into training. The training command, reference, seed, budget,
evaluation command, promotion threshold, and frozen `protocol.sh` are
unchanged.

## Launch provenance

- SNMR launch revision: `4085bed6a638d66a18f0da0018bbfe88dfc58458`
- Analyzer correction commit: `aa3e1935e84bcbde66e928c8ce8093250d53628c`
- Dependency-packaging commit: `75fef6e102286086fc9dfa68da220586a95e23d6`
- Frozen protocol SHA-256:
  `a0fb9a3099ab1fdc6d300fb3418645976709854e8ee7f64a7a823d3e174dd971`
- Frozen development analyzer SHA-256:
  `9d4838d9989d06faaf47e8acad2528f74802184721862fb5433e7614093a56f9`

## Validation-only defects

1. The frozen development analyzer requires exact equality between the
   checkpoint and YAML experiment configurations. Holosoma serializes the
   actor output dimension as `[29]` in the checkpoint and as
   `["robot_action_dim"]` in the resolved YAML. These are equivalent for this
   29-DOF policy but the frozen analyzer reports a false configuration
   mismatch.
2. The frozen development analyzer imports `analyze_wbt_horizon.py` and
   `analyze_wbt_latent_pilot.py` from its own directory, but the launch driver
   copied only `analyze_wbt_reference_dev.py`. Consequently the frozen
   analysis step cannot import its helper modules.

Neither defect can affect policy optimization, checkpoint contents, simulator
evaluation, rollout metrics, or the registered `completion_rate >= 0.70`
promotion decision. They affect only post-run validation and packaging.

## Amended analyzer bundle

The corrected, run-local bundle is staged under `amended/`:

| File | Launch SHA-256 | Amended SHA-256 |
| --- | --- | --- |
| `analyze_wbt_reference_dev.py` | `9d4838d9989d06faaf47e8acad2528f74802184721862fb5433e7614093a56f9` | `d36cf6080b86180bfee20ff8948484b04cc73a0c9af9ff354ee3c41998d0d098` |
| `analyze_wbt_horizon.py` | `d3fd6a28cfac722723fce0142dffd383b283d647c4bee920166a6b61f23dbe7b` | `8cc233aa4c681247aa5c00effad589533c2101e1d945e0a36bdf6e92538f1700` |
| `analyze_wbt_latent_pilot.py` | `c56c88fbfba2c3f1f19946c8516ba7b134cf6ffd7cf5f32dee69dff6a0a2bad7` | `c56c88fbfba2c3f1f19946c8516ba7b134cf6ffd7cf5f32dee69dff6a0a2bad7` |

The amended development analyzer changes only the checkpoint/YAML
configuration comparison. The amended horizon helper accepts the two
equivalent actor-output encodings while retaining exact comparison for the
rest of the resolved configuration. The latent helper is unchanged and is
included to make the bundle self-contained.

## Post-run procedure

Allow the frozen driver to finish training and evaluation. If its original
analysis produces partial outputs, preserve them with a `.frozen_failed`
suffix. Then invoke `amended/analyze_wbt_reference_dev.py` with the exact
`--run-dir`, `--checkpoint`, `--report`, `--snmr-reference`, `--gmr-report`,
`--gmr-reference`, and `--output` arguments in `protocol.sh`. Derive
`ANALYSIS_STATUS` from the amended `analysis.json` verdict and write
`COMPLETE` only after that validation succeeds.
