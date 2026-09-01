# MorphoRetarget foundation freeze and kinematic-contract iteration

**Started:** 2026-09-01 01:11 America/New_York
**Branch:** `feat/morpho-retarget-foundation`
**Mode:** bounded autoresearch iteration

## Objective

Freeze the existing MorphoRetarget foundation, establish a canonical and hashable human-motion
contract, diagnose the legacy PM01 leave-one-robot-out failure before choosing a new encoder, and
build only the kinematic RobotSpec interfaces needed for fixed-G1 and held-out-morphology tests.

The verifier-infrastructure and kinematic-model tracks may proceed in parallel. They converge only
after their own gates pass: no physics preference labels before verifier qualification, and no
dynamics-conditioned learning before kinematic held-out generalization.

## In scope

1. Commit the existing RobotSpec, verification reports, runners, tests, artifacts, and status docs
   on the dedicated branch with dirty-aware source provenance.
2. Add `HumanMotionSpec` v0.1 with explicit timebase, frames, units, normalized scale descriptors,
   probabilistic contacts, normalized tensor-buffer hashing, JSON round-trip, and resampling tests.
3. Make each available backend worker re-hash motion/asset bytes independently and record SNMR,
   Newton, and (when present) Isaac Lab revisions plus dirty flags.
4. Diagnose the existing PM01 LORO checkpoint with conditioning swaps and per-joint/body error
   localization before changing the encoder.
5. Add a kinematic RobotSpec graph tokenizer with padded batches, tree distances, availability
   masks, and adversarial serialization/permutation tests. Do not train the new multi-robot model in
   this iteration.
6. Specify and, where local assets permit, run the G1 MJCF-to-Isaac random-pose FK parity gate.

## Keep metrics

- Existing repository suite introduces zero new failures.
- `HumanMotionSpec` rejects non-finite/inconsistent inputs and noncanonical frame/unit conventions.
- Motion JSON round-trip preserves all invariant fields and the normalized-buffer SHA-256.
- Resampling emits explicit target timestamps at 50 Hz, uses quaternion SLERP and declared position
  interpolation, and preserves quaternion normalization.
- Identical normalized tensor buffers hash identically across dtype/layout variants; a changed
  scalar changes the digest.
- Robot graph tokenization is independent of asset name and serialization order; inverse-permuted
  outputs/metadata agree within a frozen tolerance.
- Every generated manifest binds source revisions, dirty flags, and independently recomputed
  motion/asset/controller/config hashes where the corresponding environment is available.
- PM01 diagnostics determine whether failure is global, joint-localized, or insensitive to
  conditioning; an inconclusive diagnosis is recorded rather than used to justify architecture.
- G1 cross-asset parity target, when both assets/backends are available: 10,000 registered random
  in-limit poses, key-link position maximum below 1.0 mm and orientation geodesic maximum below
  1e-3 rad.

## Stop rules

- Do not overwrite recorded `runs/`, `exports/`, or prior `autoresearch/` artifacts.
- Do not claim G0 passed when `HumanMotionSpec`, byte-level asset identity, or required cross-asset
  mappings are absent.
- Do not make morphology-wide claims from the single PM01 holdout.
- Do not implement or train dynamics conditioning, a physics critic, preference learning, repair,
  flow matching, sequence-level PPO, or hardware deployment in this iteration.
- Do not let PhysX/MJWarp parity block kinematic unit tests or fixed-G1 integration; do not let
  kinematic implementation generate physics labels before verifier qualification.
- If the Isaac asset or environment is unavailable, freeze a fail-closed parity protocol and report
  the external dependency instead of weakening tolerances.

## Verification commands

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
git diff --check
git status --short
```
