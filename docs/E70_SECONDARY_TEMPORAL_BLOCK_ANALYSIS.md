# E70 secondary temporal-block analysis

**Preregistered:** 2026-08-11, after seed 0 and seed 1 were complete but before the missing
seed-2 proprioception and shuffled-content cells existed.  This document freezes a secondary
dependence check.  It does not amend the E70 primary analyzer, gate, endpoint, or interpretation.

## Question and fixed input

The primary paired bootstrap gives each of the 69 reference-only ambiguity pairs equal weight but
does not group pairs whose 10-second evaluation rollouts begin near one another on the same source
clip.  This analysis asks whether the directions and uncertainty of SNMR-minus-time (A--T) and
SNMR-minus-phase-shuffled (A--S) remain similar when nearby start regions are resampled together.

The only behavioral inputs are the registered per-rollout `*_eval_ambiguity.json` files for the
SNMR, time, and shuffled arms under
`/data/robotixx/snmr-research/e70/students/seed<seed>_<arm>/`.  Pair geometry comes only from the
frozen `autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json` whose registered SHA-256 is
`3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e`.  Every read input is
SHA-256-stamped in the output.

## Frozen block rule

1. The atomic block length is exactly **10.0 seconds**, equal to the frozen ambiguity rollout
   duration.  Blocks are half-open intervals `[10b, 10(b+1))` anchored at clip time zero.  Each of
   a pair's two registered starts is assigned independently by
   `floor(start_seconds / 10.0)` on its loaded clip; boundaries and origin will not move.
2. Treat each ambiguity pair as an edge from its first-clip atomic block to its second-clip atomic
   block.  The temporal resampling units are the connected components of this bipartite graph.
   Therefore any two pairs sharing a 10-second start region on either clip are always resampled
   together.  This deterministically partitions all 69 pairs; no pair is dropped or duplicated.
3. Within each seed and contrast, first average paired rollout completion differences within each
   ambiguity-pair ID, exactly preserving the primary equal-pair point estimand.  The reported point
   estimate is the mean of the 69 pair effects and then the equal mean across training seeds.
4. For each of 10,000 bootstrap replicates using NumPy generator seed **7017**, resample training
   seeds with replacement at the outer level.  For every sampled-seed slot, resample the temporal
   components with replacement, retain every pair in each selected component, take the mean over
   the resulting pair multiset, and finally average the sampled-seed-slot means.  The interval is
   the 2.5th and 97.5th percentiles.

On the frozen geometry, this rule must yield exactly 12 temporal components with sorted pair
counts `[1, 2, 2, 3, 3, 4, 4, 7, 7, 9, 13, 14]`.  Any other partition is an implementation error.

## Execution and interpretation

The final command accepts exactly seeds `0 1 2`.  Any proper subset fails unless `--preview` is
explicit, and preview mode accepts only seed 0 or seeds 0 and 1; it can never read seed 2.  The
seed-0/1 smoke output is named `secondary_seed01_preview.json` and is permanently non-final.

This analysis has **no primary-verdict effect**.  For each contrast it reports the point estimate,
block-bootstrap 95% interval, per-seed point estimates, block count, and whether the interval lies
strictly above zero.  “Directionally consistent” means only that both A--T and A--S point estimates
are positive; a zero-crossing block interval is reported as temporal-dependence sensitivity, not
as a reason to tune E70 or change its registered outcome branch.

## Frozen implementation

- Script: `scripts/analyze_e70_temporal_blocks.py`
- Script SHA-256: `ed9ed8a1374d75056e4df72801bb8d931446e3c06b67c8bcdbeb99eb7fd69d6d`
- Synthetic regression tests: `tests/test_e70_temporal_blocks.py`

The implementation hash is filled once after the synthetic tests pass and before any preview is
run.  Thereafter, any script drift stops the final secondary analysis until the discrepancy is
reported; the rule and implementation are not repaired in response to E70 outcomes.
