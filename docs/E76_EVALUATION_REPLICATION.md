# E76 — How reproducible is one E70 evaluation?

**Date:** 2026-08-15. **Type:** evaluation-only replication study on frozen artifacts.
**Status:** complete. Nothing under `/data/robotixx/snmr-research/e70/` was written.

## 1. How this was found

While running the E72 control gate — an arm whose motion files are byte-identical to the frozen
E70 motions, so it should have been a pure no-op — the gate **failed**:

```
control seed0: got 0.7470703125  want 0.7646484375  -> MISMATCH
control seed1: got 0.7529296875  want 0.7607421875  -> MISMATCH
control seed2: got 0.7470703125  want 0.7373046875  -> MISMATCH
```

The preregistration required an exact match before any intervention arm could be interpreted, so
the run stopped itself before evaluating a single intervention arm. That is the gate working.

Three checks isolated the cause:

1. **The substituted motions were not to blame.** `sha256` of the control NPZs equals the frozen
   originals exactly.
2. **The start grid was not to blame.** `start_steps` and `motion_ids` were identical across runs;
   188 of 1,024 *rollout outcomes* differed.
3. **The frozen path itself does not reproduce.** Re-running with
   `--motion-dir /data/robotixx/snmr-research/e70/motions` — the frozen directory, frozen
   checkpoint, frozen precheck, evaluation seed 404, `E52_DET=1` — returned `0.7451171875`
   against the recorded `0.7646484375`, with 182/1,024 rollouts differing.

**The E70 evaluation harness is not bit-reproducible.** Two runs in the same session, on the same
GPU, with byte-identical inputs, differ in ~18% of rollout outcomes. The likely mechanism is
ordinary GPU non-determinism (atomics in contact resolution) amplified over 500-step closed-loop
rollouts until near-threshold rollouts flip the completion predicate.

## 2. The measurement

Eight independent repeats of the seed-0 ambiguity evaluation for each of the two arms whose
contrast is the paper's headline. Same start grid, same evaluation seed, only the arm differs.

| arm | n | mean | sd | range | spread |
| --- | ---: | ---: | ---: | --- | ---: |
| SNMR (A) | 8 | 0.756714 | 0.008310 | 0.7402 – 0.7686 | 0.0283 |
| time (T) | 8 | 0.597168 | 0.007899 | 0.5879 – 0.6104 | 0.0225 |

Per clip, the harder clip carries most of the variance:

| arm | walk1_subject1 | walk1_subject5 |
| --- | --- | --- |
| SNMR | 0.9600 ± 0.0056 | 0.5535 ± 0.0159 |
| time | 0.7810 ± 0.0086 | 0.4133 ± 0.0157 |

## 3. What it means for the paper — three findings, all reassuring

**(a) The frozen record is a typical draw, not a lucky one.** The recorded seed-0 values sit at
z = +0.95 (SNMR) and z = +1.42 (time) of their own replication distributions. Both are ordinary
draws. Nothing about the frozen numbers is anomalous.

**(b) The contrast is far more stable than either arm.** The noise is substantially common-mode, so
it partly cancels in the difference:

```
A-T from the 8-repeat replication means (seed 0) = +0.159546
A-T from the frozen record              (seed 0) = +0.156250
```

Agreement to 0.003. Note the frozen record's contrast is slightly **lower** than the replication
mean, so the recorded value is not favourably biased.

**(c) The headline survives with room to spare.** Single-run noise on A−T is
sd = 0.0115; on the three-seed mean, sd = 0.0066. The primary A−T of **+0.191** is about
**14 standard deviations** of single-run evaluation noise.

Propagating it into the registered interval:

| quantity | value |
| --- | --- |
| registered CI `[0.1239, 0.2741]` → implied sd | 0.038325 |
| evaluation-noise sd on the three-seed mean | 0.006620 |
| combined in quadrature | 0.038893 (**1.48% wider**) |
| widened interval | `[0.1145, 0.2670]` |

**The interval widens by under 2% and still excludes zero by a wide margin. No verdict changes.**

## 4. What must be said in the paper

This is an unmodeled variance component. The registered bootstrap resamples ambiguity pairs and
training seeds, but treats each rollout outcome as fixed — it does not include evaluation
replication variance. That is worth one honest sentence, and the honest sentence is favourable:

> Re-running a single arm's evaluation under identical inputs changes about 18% of individual
> rollout outcomes, because closed-loop rollouts amplify GPU non-determinism until near-threshold
> episodes flip. Over eight repeats the per-arm completion sd is 0.008, and the A−T contrast is
> stable to 0.003 because the noise is largely common-mode. Propagating this component widens the
> primary interval by 1.5%, from [0.124, 0.274] to [0.114, 0.267]; it does not affect the verdict.

Reporting it costs nothing and pre-empts a reviewer who tries to replicate and gets a different
third decimal. It is the same pattern as every other result in this project: the instrument was
pointed at its own assumptions and the assumption did not survive intact.

## 5. What it changes downstream

**E72 must replicate.** An intervention effect smaller than about 0.02 in single-run completion is
not distinguishable from evaluation noise. The E72 preregistration's exact-reproduction control
gate is unsatisfiable as written and is amended (see `docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md`,
amendment dated 2026-08-15) to a distributional gate plus per-arm replication.

**E75 is unaffected.** Its outcome was exactly 0.000 in all nine cells — a floor, not a
near-threshold quantity, and not something 18% rollout jitter can move.

**E71, if it runs, is affected in principle** but its coordinate is a continuous normalized
trajectory distance rather than a thresholded completion predicate, so it should be far less
sensitive. Its smoke run should measure this rather than assume it.

## 6. Provenance

- Launcher: `scripts/run_e76_eval_replication.sh` (8 repeats × 2 arms, seed 0).
- Outputs: `/data/robotixx/snmr-research/e76_eval_replication/{snmr,time}/seed0/repeat{1..8}/`.
- Frozen checkpoints reached by symlink; frozen precheck hash-verified by the E72 launcher in the
  same session.
- The three diagnostic runs that found this are under
  `/data/robotixx/snmr-research/e72_latent_sub/students/{control,control_repeat,frozen_path_repeat}/`.
