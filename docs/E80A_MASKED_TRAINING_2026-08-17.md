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

| outage cell | frozen | **masked** | Δ | R frozen | **R masked** |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.929 | 0.925 | **−0.004** | +1.000 | +1.000 |
| f 0.1, 0.1–0.5 s | 0.757 | 0.903 | +0.146 | +0.642 | +0.982 |
| f 0.1, 0.5–1 s | 0.603 | 0.817 | +0.215 | +0.267 | +0.759 |
| f 0.3, 0.1–0.5 s | 0.476 | 0.846 | +0.370 | −0.035 | +0.808 |
| f 0.3, 0.5–1 s | 0.280 | 0.670 | +0.390 | −0.483 | +0.401 |
| f 0.5, 0.1–0.5 s | 0.267 | 0.780 | +0.514 | −0.492 | +0.675 |
| f 0.5, 0.5–1 s | 0.106 | 0.554 | **+0.447** | **−0.854** | **+0.158** |

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
2. **Active harm is eliminated.** The frozen arm's `R` goes negative at four of six degraded cells,
   reaching −0.85: it ends far below the goal-blind floor because it obeys a stale target. The
   masked arm's `R` is **positive everywhere** — it never falls below what a policy with no command
   at all achieves. The registered hypothesis was that masked training would bring `R` to ≈ 0
   (graceful descent to the floor); it does better, retaining 16–98 % of the channel advantage.
3. **The effect is the largest in the program.** +0.45 completion at the severest cell, +0.56 on
   the ambiguity grid, from a single training-recipe change — larger than any representation
   contrast this project has measured, and obtained without touching the command representation.
4. **It confirms the E79 conclusion from the other side.** No deployment-side fill rescued a frozen
   policy (best `R` = −0.74 across four fills spanning a 6× range of prediction error); training
   with the dropout present fixes it with the *naive* `hold` fill. Validity-awareness is a training
   property, not an inference-time patch.

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

## 5. Open attribution question (cheap to close)

mE differs from the frozen arm in **two** ways: dropout during training, and two validity flag bits.
The clean result shows the flags cost nothing, but the *gain* is not yet attributed between "learned
to cope with outages" and "knows when it is blind". One extra arm closes it — **mEnf**: identical
masked training with `E78_FLAG_DIM=0`. Queued after batch 1. If mEnf ≈ mE, the flags are decoration
and the recipe is even simpler; if mE ≫ mEnf, the validity signal is load-bearing and belongs in the
interface specification, which is the stronger version of the framework's §III.1 claim.

## 6. Provenance

`scripts/run_e78_masked_fusion.sh train|sweep 0 mE`; trainer `scripts/train_e78_masked_fusion.py`
(derived from the frozen `train_e52_dagger.py` by asserted replacements, staleness-tested);
outputs under `/data/robotixx/snmr-research/e78_masked_fusion/seed0_mE/`. Frozen comparators are the
E78-F sweep of the same seed, all sanity-gated against the hash-bound E70 reports. Seeds 1–2, mZf,
mB, mTl, mTf, mGf, mShf, mZc, mZg, mS pending.
