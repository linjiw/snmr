# E77 — Degraded-command pilot: does the interpretable window exist?

**Date:** 2026-08-15. **Type:** descriptive pilot. **No gate, no preregistration, no replication.**
Seed 0, one run per cell, general start grid, evaluation seed 404, 1,024 rollouts.
Evaluation-only on frozen checkpoints; writes only under
`/data/robotixx/snmr-research/e77_degradation_pilot/`.

**Nothing here may be promoted to a confirmatory result without its own preregistration.**

## 1. The one question this exists to answer

The owner asked whether the learned interface can be shown to degrade more gracefully than the
explicit one, which would support a safety claim. Before designing that experiment, one thing has to
be true:

> Is there a corruption severity at which **both** arms are neither saturated nor floored?

E61-v4 already ran a robustness sweep on this stack and failed its gate at three seeds. Its own note
records why: the only sigma where the effect was positive was one where *"both arms are broken there
(~0.1 completion)"*. A curve measured outside the functional window cannot be interpreted no matter
how well the rest of the design is executed.

## 2. Why the hold axis is the fair one

Cross-arm robustness comparison normally founders on units: Gaussian sigma on a 64-D explicit joint
target and sigma on a standardized latent window are different spaces, and any exchange rate between
them is arguable.

`E52_EVAL_HOLD_Z` avoids the problem entirely. It applies a zero-order hold to **`z_cmd`, the shared
bottleneck** (`train_e52_dagger.py:660-666`). Both arms emit `z_cmd` of identical dimension into an
identical decoder (`distillation.py:63`, `mlp([proprio_dim + z_cmd_dim, ...])`); they differ only in
what feeds `prior_input` (`distillation.py:65-73`). Severity is measured in control ticks. The
comparison is therefore **matched by construction** — same tensor, same units, same downstream network.

E65 declined this comparison, noting the explicit teacher's command "is not directly comparable (the
env recomputes its obs)". That objection applies to holding the *upstream* explicit observation. It
does not apply to holding the bottleneck.

The Gaussian axis is reported too, but it is **not** matched and its cross-arm differences are
therefore contestable.

## 3. Results

### Axis 1 — zero-order hold on `z_cmd` (matched by construction)

| hold | ms | explicit | SNMR | SNMR − explicit |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 0.9258 | 0.6982 | −0.2275 |
| 2 | 40 | 0.8623 | 0.5664 | −0.2959 |
| 5 | 100 | 0.2227 | 0.0039 | −0.2188 |
| 10 | 200 | 0.0000 | 0.0000 | 0.0000 |
| 20 | 400 | 0.0000 | 0.0000 | 0.0000 |

The k=1 cells reproduce the frozen baselines (0.923 / 0.699), so the pilot is measuring what it
claims to measure.

### Axis 2 — Gaussian noise on each arm's own upstream channel (NOT matched)

| sigma | explicit (cmd) | SNMR (z) | SNMR − explicit |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.9229 | 0.6992 | −0.2236 |
| 0.25 | 0.9160 | 0.6973 | −0.2188 |
| 0.5 | 0.8447 | 0.6133 | −0.2314 |
| 1.0 | 0.3926 | 0.3662 | −0.0264 |
| 2.0 | 0.0000 | 0.0020 | +0.0020 |

### Fraction of each arm's own clean performance retained

| severity | explicit | SNMR | SNMR − explicit | both functional? |
| --- | ---: | ---: | ---: | --- |
| hold 40 ms | 0.931 | 0.811 | **−0.120** | yes |
| hold 100 ms | 0.241 | 0.006 | **−0.235** | one floored |
| hold 200/400 ms | 0.000 | 0.000 | 0.000 | no — both floored |
| noise σ 0.1 | 0.997 | 1.001 | +0.005 | yes |
| noise σ 0.25 | 0.989 | 0.999 | +0.009 | yes |
| noise σ 0.5 | 0.912 | 0.878 | −0.034 | yes |
| noise σ 1.0 | 0.424 | 0.524 | **+0.100** | **yes** |
| noise σ 2.0 | 0.000 | 0.003 | +0.003 | no — both floored |

## 4. Reading

**On the fairly matched axis, the learned interface is worse.** Under command staleness the SNMR arm
loses more of its own capability than the explicit arm at every level, and it floors first: at 100 ms
the explicit arm still completes 22% of rollouts while SNMR is at 0.4%. There is no crossing. The
naive "a compressed command is inherently more robust" story is **not supported** — it is contradicted.

**On the unmatched axis there is one interpretable cell with a positive sign.** At σ = 1.0 both arms
are functional (0.393 and 0.366, neither saturated nor floored) and SNMR retains 52.4% of its clean
performance against explicit's 42.4%, a +0.100 relative-retention gap. This is the window E61-v4
never found. Three caveats, all serious:

1. **Absolute performance still favours explicit** (0.393 vs 0.366). SNMR wins only after normalising
   by its own lower baseline, and relative retention mechanically flatters the arm that starts lower
   whenever degradation is sublinear. Whether that normalisation is the right safety metric is
   exactly the question a reviewer would press.
2. **The axis is not matched.** The exchange rate between sigma-on-explicit-cmd and sigma-on-latent is
   arbitrary; the result could be moved by choosing different units.
3. **One seed, one run, no replication.** E76 measured single-run sd at 0.0083, and the σ = 1.0
   absolute gap (−0.026) is only ~3 sd. The relative gap is larger but inherits the same noise.

This is close to what E61-v4 saw — a positive sign appearing only at high sigma — and E61-v4's gate,
set in advance at +5pp, rejected it at three seeds.

## 5. What this rules in and out

**Ruled out for this submission:** a safety claim built on staleness robustness. The matched axis
gives the wrong sign and it is not a near miss.

**Not ruled out, and not tested here:**

- **Source corruption routed through each interface's own encoder.** Corrupt the underlying motion
  and let the explicit arm read corrupted joint targets while the SNMR arm reads a latent re-encoded
  from the same corrupted motion, scoring both against the clean reference. This is the
  deployment-faithful design and the only one where the exchange rate is set by physics rather than
  by choice of units. It requires the SNMR encoder to be runnable on corrupted motion.
- **Physically infeasible references.** Retargeting routinely emits targets that violate joint
  limits, self-collide, or demand impossible root motion. An explicit interface passes them through
  verbatim; a learned prior may project onto its training manifold. Different mechanism, different
  metric (joint-limit violations, falls), untested.
- **Bits.** Performance per bit on the command channel, as a rate-distortion curve.

**A mechanism is missing from the current students and should be stated whenever this is discussed:**
every frozen E70 student was trained with `E52_DET=1` (`run_e70_multitraj.sh:149`; each report
records `deterministic: true`), so the σ = 0.3 reparameterisation noise that E65 credited with 18×
hold robustness was never applied. These are E62-configuration students — the arm that collapsed to
0.027 at 100 ms in E65. The pilot's hold curve is consistent with that.

But note the corollary, which cuts against using it as an SNMR advantage: E65's contrast was arm C
(explicit, CVAE-trained) against E62 (explicit, deterministic) — **both explicit arms**. Train-time
bottleneck noise is a lever available to either interface. It does not favour the latent.

## 6. Provenance

Launcher `scripts/run_e77_degradation_pilot.sh`. Outputs under
`/data/robotixx/snmr-research/e77_degradation_pilot/{explicit,snmr}/<label>/`. Frozen checkpoints
reached by symlink; nothing written under `/data/robotixx/snmr-research/e70/`.

---

# ADDENDUM — 2026-08-15: the one apparent positive dissolves under paired analysis

Section 4 above flagged the σ = 1.0 relative-retention advantage (+0.100) with three caveats. A
paired re-analysis of the same arrays settles it: **it was a baseline-composition artifact, not an
effect.**

Both arms run identical `start_steps` and `motion_ids` (verified). Restricting to the **704 rollouts
both arms complete cleanly at k=1**:

| severity | marginal (E / S) | matched subset (E / S) | subset diff | McNemar E-only : S-only |
| --- | --- | --- | ---: | --- |
| hold 40 ms | 0.931 / 0.811 | 0.940 / 0.722 | −0.219 | 169 : 15 |
| hold 100 ms | 0.241 / 0.006 | 0.295 / 0.004 | −0.291 | 207 : 2 |
| noise σ 0.5 | 0.912 / 0.878 | 0.940 / 0.778 | −0.162 | 133 : 19 |
| noise σ 1.0 | 0.424 / **0.524** | 0.520 / 0.500 | **−0.020** | 107 : 93 |

The +0.100 at σ = 1.0 becomes **−0.020**, and its McNemar counts (107 : 93) are a coin flip. SNMR's
lower clean baseline was excluding the easy rollouts from its own denominator.

**Retention does not cancel the 22.4 pp clean gap; it launders it.** Any cross-arm severity claim
must be reported on the paired matched subset, which is free from the per-rollout `completed` and
`start_steps` arrays every frozen report already writes.

This also strengthens §4's conclusion rather than softening it: on the matched-by-construction axis
the asymmetry against the hypothesis is 11× at 40 ms and 100× at 100 ms.

**Standing caveat on this axis, recorded so it is not forgotten:** `z_cmd = mu_prior(proprio, goal)`,
so holding it also freezes proprioceptive feedback. It simulates "the policy stopped running" rather
than "the command went stale". It is matched by construction and physically unfaithful — the
deployment-faithful version freezes each arm's own *input* while the prior keeps reacting to fresh
proprioception, and that needs a new evaluator. The reversal reported here is strong enough that it
is unlikely to flip, but the axis should be described accurately if it is ever written up.

Full analysis and recommendation: `docs/DEGRADED_COMMAND_RESEARCH_2026-08-15.md`.
