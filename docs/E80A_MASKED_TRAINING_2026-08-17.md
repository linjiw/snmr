# E80-A — robustness is trained, not free: masked training at zero clean cost

**Date:** 2026-08-17. **Status:** seed 0 of a preregistered three-seed design
(`docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md` §3, as amended 2026-08-16). Seeds 1–2 and the
remaining arms are queued. Single-seed results are reported as such and no gate is called yet.

---

## 1. What was run

One arm — **mE**: the E70 explicit command student, retrained under **reference-stream dropout**
(Bernoulli segments, 0.1–0.5 s, target masked-tick fraction 0.3, hazard ramped over the first 300
rounds, `hold` fill) with two validity flag bits `[is_masked, staleness/max_segment]` reaching the
code encoder only. Everything else is the frozen E70 recipe: same teachers, same motions, 1,024
envs, 2,000 rounds, deterministic encoder, best-validation selection, evaluation seed 404 on the
frozen 1,024-rollout general grid and the frozen 69-pair ambiguity grid.

Training realized a masked-tick fraction of 0.285 against a 0.3 target; best validation action loss
0.103 at round 1,949 (the frozen unmasked student reaches 0.043 — it is solving an easier problem).

The comparison below is therefore **one training-recipe change**, evaluated through the same
instrument, against the frozen student of the same seed.

## 2. Result

General grid, seed 0. `R` is floor-relative retention (`docs/COMMAND_INTERFACE_SYNTHESIS_2026-08-16.md`
§II.1), the fraction of the arm's advantage over a goal-blind policy that survives the corruption;
`R < 0` means the arm ends up **worse than having no command channel at all**.

| outage cell | frozen | **masked** | Δ |
| --- | ---: | ---: | ---: |
| clean | 0.929 | 0.925 | **−0.004** |
| f 0.1, 0.1–0.5 s | 0.757 | 0.903 | +0.146 |
| f 0.1, 0.5–1 s | 0.603 | 0.817 | +0.215 |
| f 0.3, 0.1–0.5 s | 0.476 | 0.846 | +0.370 |
| f 0.3, 0.5–1 s | 0.280 | 0.670 | +0.390 |
| f 0.5, 0.1–0.5 s | 0.267 | 0.780 | +0.514 |
| f 0.5, 0.5–1 s | 0.106 | 0.554 | **+0.447** |

> **Correction (2026-08-17, same day).** This table originally carried floor-relative retention
> columns showing `R` flipping from −0.854 to +0.158, and §3 claimed the masked arm "never falls
> below its floor". **That claim is retracted**: it used the *frozen* goal-blind arm (seed-0 clean
> 0.486) as the floor. The goal-blind arm trained in this batch under the same masked recipe (mB)
> reaches **0.660** clean, and against that floor the masked explicit arm's `R` at the severest cell
> is **−0.218**, not +0.158. The E70 frozen goal-blind arm spans **0.249 / 0.556 / 0.487** across its
> three seeds — a 0.31 spread — so a single-seed floor cannot support any claim about crossing it.
> What survives unchanged is the Δ column, which involves no floor at all. See §3a.

Ambiguity grid (69 frozen start pairs):

| cell | frozen | masked | Δ |
| --- | ---: | ---: | ---: |
| clean | 0.979 | 0.971 | −0.009 |
| f 0.3, 0.1–0.5 s | 0.501 | 0.905 | +0.404 |
| f 0.3, 0.5–1 s | 0.297 | 0.686 | +0.389 |
| f 0.5, 0.1–0.5 s | 0.257 | 0.812 | **+0.556** |

## 3. Reading

1. **Zero clean cost.** −0.004 on the general grid and −0.009 on the ambiguity grid, both inside the
   E76 evaluation-noise band and well inside the preregistered co-primary bound of −0.01. The
   masked student is still at teacher parity.
2. **Active harm is reduced, but "eliminated" is not established** — see the correction above and
   §3a. Against the batch's own masked floor (mB = 0.660 clean) the masked explicit arm sits at
   `R` = +0.125 at f 0.3 / 0.5–1 s and −0.218 at f 0.5 / 0.5–1 s, versus the frozen arm's −0.483 and
   −0.854 against its own floor. The direction is a large improvement; the *sign* at the severest
   cell depends on a floor estimate that is not yet reliable.
3. **The effect is the largest in the program.** +0.45 completion at the severest cell, +0.56 on
   the ambiguity grid, from a single training-recipe change — larger than any representation
   contrast this project has measured, and obtained without touching the command representation.
4. **It confirms the E79 conclusion from the other side.** No deployment-side fill rescued a frozen
   policy (best `R` = −0.74 across four fills spanning a 6× range of prediction error); training
   with the dropout present fixes it with the *naive* `hold` fill. Validity-awareness is a training
   property, not an inference-time patch.

## 3a. The floor is the weak link, and that is a finding about the metric

Floor-relative retention needs a denominator, and this instrument's goal-blind arm turns out to be
its noisiest component: across the three frozen E70 seeds it completes **0.487 / 0.556 / 0.249**
(general grid) — a 0.31 spread, far larger than any effect being measured. The masked floor arm mB
lands at 0.660, above all three.

Consequences, adopted immediately:

* **`R` is not reportable from a single seed.** It requires a pooled multi-seed floor, and its
  uncertainty must be propagated rather than ignored. The §II.1 reporting standard in the synthesis
  is amended accordingly.
* **Claims that depend on `R`'s sign near zero are out of reach at this sample size.** Claims that
  depend only on paired completion differences (the Δ column) are unaffected, because both arms are
  evaluated on identical rollouts from identical starts.
* An ablation is running (**mBnf**: goal-blind, masked, `flag_dim=0`) to separate "the flag bits
  raised the floor" from "this arm is simply high-variance". The second is far more likely — the
  flags are uninformative to a policy with no reference — but it is cheap to check rather than
  assume.

This is the second time in two days that the program's own control has disciplined its headline
(E78-F's content reading died to the time-code control; E80-A's floor claim dies to floor variance).
The Δ result stands; the framing around it needed tightening.

## 4. What this does to the rest of the program

- **The headline changes.** The interesting claim is no longer about which representation goes in
  the channel. It is: *a tracker's catastrophic failure under reference dropout is a training
  artifact, not a property of the explicit interface, and one recipe change removes it at no clean
  cost.* That is an engineering result any tracking stack can adopt.
- **The E80 primary got harder, correctly.** mZf must now beat an mE that scores 0.670 / 0.554 at
  the co-primary cells rather than the frozen 0.280 / 0.106. The conjunction gate (beat mE by
  ≥ 0.10 **and** beat the content-free controls mTl/mTf by ≥ 0.05 **and** no clean regression) was
  registered for exactly this situation.
- **E78-F is fully explained.** Its cross-arm ranking measured reliance plus arbitrary
  out-of-distribution response among policies that had never seen a dropout. Once every arm is
  trained under dropout, the comparison finally means what it appears to mean.

## 5. The attribution question, answered: the validity flags are NOT load-bearing

mE differed from the frozen arm in two ways — dropout during training, and two validity flag bits.
**mEnf** (identical masked training, `flag_dim=0`) settles it:

| cell | mE (with flags) | mEnf (no flags) |
| --- | ---: | ---: |
| clean | 0.925 | 0.904 |
| f 0.3, 0.5–1 s | 0.670 | **0.724** |
| f 0.5, 0.1–0.5 s | 0.780 | 0.742 |
| f 0.5, 0.5–1 s | 0.554 | **0.606** |

The flag-free arm is **equal or better** under dropout and marginally worse clean. So the entire
E80-A gain comes from **training with the outages present**, not from telling the policy when it is
blind. The framework's §III.1 claim ("validity is part of the interface") is therefore *not*
supported in this form: on this instrument a policy trained under dropout infers what it needs from
proprioception alone, and the explicit flag channel is decoration. The recipe is simpler than
proposed — which is the better outcome for anyone adopting it.

## 5a. Full arm table (seed 0, general grid)

| arm | what it adds to the masked explicit student | clean | f 0.3 / 0.5–1 s | f 0.5 / 0.5–1 s |
| --- | --- | ---: | ---: | ---: |
| mE | — (validity flags only) | 0.925 | 0.670 | 0.554 |
| mEnf | — (no flags) | 0.904 | **0.724** | **0.606** |
| mTl | live time code | 0.918 | **0.726** | **0.629** |
| mTf | frozen time code | 0.922 | 0.646 | 0.540 |
| mGf | explicit future window `[g_t, g_{t+0.1s}]` | 0.915 | 0.641 | 0.530 |
| mShf | other clip's latent at matched phase | 0.922 | 0.646 | 0.528 |
| mB | nothing (goal-blind floor) | 0.660 | 0.637 | 0.611 |

Two things are visible before the treatment arm (mZf) even lands. **First, every masked arm is
within ~0.1 of every other at the severe cells, against a masked-vs-frozen effect of +0.45** — the
training recipe dominates the representation choice by roughly 4×. **Second, the two best degraded
arms are the ones carrying the least information**: the live clock and the flag-free explicit
student. The window-matched explicit future (mGf) and the shuffled latent (mShf) are at the bottom
with the frozen clock. That is the reliance pattern again, now inside a family of arms that were all
trained under dropout.

## 6. Provenance

`scripts/run_e78_masked_fusion.sh train|sweep 0 mE`; trainer `scripts/train_e78_masked_fusion.py`
(derived from the frozen `train_e52_dagger.py` by asserted replacements, staleness-tested);
outputs under `/data/robotixx/snmr-research/e78_masked_fusion/seed0_mE/`. Frozen comparators are the
E78-F sweep of the same seed, all sanity-gated against the hash-bound E70 reports. Seeds 1–2, mZf,
mB, mTl, mTf, mGf, mShf, mZc, mZg, mS pending.
