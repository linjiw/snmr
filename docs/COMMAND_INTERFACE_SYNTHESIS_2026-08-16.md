# What twelve experiments say about the retarget→track command interface, and the framework they imply

**Date:** 2026-08-16. **Author:** Fable. **Status:** synthesis + design proposal. Every number below
is from a committed, frozen artifact in this repo; the design in Part III is a proposal with
preregistration drafts in Part IV, not a result.

The program has now run twelve command-interface experiments on the same instrument. Read one at a
time they look like a string of negative results. Read together they are a single, consistent
theory of what a command channel is for — and that theory prescribes an architecture nobody in this
program has built yet, because every design so far treated the channel as a *representation choice*
when the data says it is a *reliability engineering* problem.

---

# Part I — The evidence, organized by what it actually establishes

## I.1 The clean-tracking ranking, and why it is not about representation quality

| arm (E70, two walks, 3 seeds, 1,024 rollouts) | general | ambiguity starts |
| --- | ---: | ---: |
| explicit reference `g` (teacher parity; teacher macro 0.917) | **0.924** | **0.975** |
| SNMR latent window `[z_t, z_{t+0.1s}]` | 0.688 | 0.754 |
| time code (sinusoids of frame index; content-free by construction) | 0.571 | 0.564 |
| shuffled latent (other clip, matched phase) | 0.531 | 0.553 |
| proprioception only (goal-blind floor) | 0.419 | 0.447 |

Four separate experiments establish that the explicit position is unassailable *on clean tracking*:
E52-v4 arm D (explicit + latent fused) returned a three-seed null; E36-era screens found the latent
*hurt* beside the explicit; E70 puts the explicit student at teacher parity; and the conditional
information argument explains why none of this could have gone otherwise — under action-MSE
distillation from a deterministic teacher π*(x, g), I(a*; z_ret | x, g) = 0, so a student has no
gradient reason to read the latent beside the goal. **The clean ranking is closed and understood.**

## I.2 The latent's one positive result is narrow, real, and about *disambiguation*

E70's registered content result: on 69 start pairs whose present states are similar but whose
futures diverge, SNMR − time code = **+0.191** [+0.124, +0.274] (three-seed aggregate; secondary
temporal-block analysis reproduces it; E72 phase-shift rejects a static clip label). This is the
only place a learned latent beat a content-free control anywhere in the program.

E63 supplies the boundary: on a **single** cyclic walk the clock *beats* the latent (0.754 vs
0.656). So the latent's advantage is not "richer motion information" in general — it is exactly the
information a clock cannot carry: **which of several futures is being commanded**. When there is
only one future, phase is sufficient and cleaner.

## I.3 The channel is genuinely load-bearing, and genuinely lossy

Destruction (E70/E75): zeroing, batch-shuffling, or resampling each observed marginal of `z_cmd`
collapses completion to **0.000** in every seed × mode cell — on both arms. Capacity (E64): `z_cmd`
has effective rank 14.1/64, and held-out linear R² of the 64-d goal from `z_cmd` is 0.475 — *less*
than proprioception alone predicts it (0.618). The bottleneck does not transmit the goal; it
transmits a low-rank control code that the decoder cannot run without.

## I.4 Robustness: four experiments, and only the fourth one had the controls

| experiment | axis | verdict |
| --- | --- | --- |
| E65 | zero-order hold on `z_cmd`, CVAE vs deterministic | 18× hold robustness attributed to train-time noise — **but both cells were explicit arms**, and `E52_DET=1` toggles four things at once. A training-recipe result, not an interface result. |
| E61-v4 | Gaussian noise, redundancy gate | failed at three seeds; the only positive sigma had both arms broken. |
| E77 | hold on `z_cmd` (shared bottleneck) | SNMR **worse** at every level, 100× at 100 ms; the single positive cell dissolved from +0.100 to −0.020 on the paired matched subset. Marginal retention laundered a clean gap. |
| **E78-F** | hold on the **upstream reference**, proprioception live, all five arms × 3 seeds | see below |

E78-F (2026-08-16), completion by arm at f = 0.5 with 0.5–1 s outages: explicit **0.108**, SNMR
0.473, time code **0.533**, shuffled 0.483, proprio floor 0.434. Two facts kill the obvious reading
and replace it with a better one:

1. **The content-free clock is the most robust arm.** Any explanation appealing to what `z_ret`
   encodes is excluded by E70's own registered null control.
2. **The explicit arm falls far below the goal-blind floor** (0.108 vs 0.434 measured on the same
   cell, and confirmed within-arm: blanking the reference beats holding it — 0.683 vs 0.603 at
   f = 0.1 / 0.5–1 s on the same frozen student and the same rollouts).

So: *a stale command is worse than no command*, and cross-arm robustness rankings mostly measure
**how much each arm depended on the channel being corrupted**, plus an **active-harm** term when the
stale value conflicts with live proprioception.

## I.5 Methodological findings the program owns

- **Marginal retention launders a clean gap** (E77 addendum): report paired matched subsets.
- **Completion is not a safety metric** (paper Limitations); E78-F sharpens it: *the controller that
  ignores the command wins the robustness comparison.*
- **Evaluation is not bit-reproducible** (E76: 0.83 % of rollouts flip; per-arm sd 0.0083 — measured
  on a high-completion arm). E78-F extends it: at mid completion the replay spread is ~2× that
  (three replays of one null arm: 0.026 general / 0.018 ambiguity → sd ≈ 0.016), so sanity
  tolerances must be completion-dependent.
- **Attribution needs isolation** (E65's 1-of-4 toggle; E77's axis choice) — severity axes are only
  matched if the thing being frozen is the same *kind* of signal for both arms.

## I.6 Upstream (retargeting) facts that constrain any design

- SNMR's decoder **cannot emit an out-of-limit joint target** (tanh + limit rescale, `model.py:318`)
  — an architectural safety guarantee with zero sim-to-real gap.
- Contact is *not* incidentally in `z_ret` (F1 0.088; z-linear AUROC 0.51–0.64): it must be put in
  deliberately.
- Retarget byproducts predict where a tracker fails: E1 pilot, +0.054 held-out R² over kinematics on
  clean labels (below the +0.10 gate; +0.117 on partly circular labels), licensing E1-proper.
- Cross-embodiment is the latent's structural monopoly: a 29-DoF G1 command stream cannot drive
  a Booster T1, but a shared latent can (E54, registration restored 2026-08-15).

---

# Part II — The theory these findings share

**A command channel's value is the information it supplies that proprioception and phase cannot,
and its risk is the harm it does when that information goes wrong.**

Write the tracker's information sources as (i) proprioception `x`, always live; (ii) phase/clock,
always available and never fails; (iii) the reference `g`, accurate but externally supplied and
therefore *droppable*; (iv) a learned motion prior `z`, which is a lossy recoding of the same
motion but is *generative* — it can produce reference-like targets rather than merely index them.

Every result in Part I falls out of this:

| observation | explanation |
| --- | --- |
| explicit wins clean by a wide margin | `g` is the only source of the exact target; the teacher used it, and distillation transfers it exactly |
| latent adds nothing beside explicit | I(a*; z \| x, g) = 0 — no marginal information |
| clock beats latent on one clip (E63) | with one future, phase carries everything the task needs, more cleanly than a lossy recoding |
| latent beats clock on ambiguity starts (E70, +0.191) | two futures ⇒ the task needs *which one*; phase cannot say |
| destruction collapses both arms to 0.000 | whatever the channel carries, the decoder is not autonomous |
| clock is most robust under dropout (E78-F) | it is the only channel that *cannot* go stale |
| explicit falls below the goal-blind floor | a stale `g` is not missing information, it is *wrong* information, and the tightly-coupled student obeys it |
| window-fed arms decay *to* their floor | they were partly running on proprio+phase already |

The design consequence is immediate and, in hindsight, obvious: **the failure mode is not the
representation, it is the absence of any notion of validity.** Today's interface hands the policy a
64-d vector with no statement of whether it is true. Every experiment that swapped what goes in that
vector left the real defect untouched.

## II.1 The measurement framework: floor-relative retention

The theory implies its own metric. If a channel's value is the advantage it buys over a goal-blind
policy, then its robustness is *the fraction of that advantage which survives corruption*:

$$R \;=\; \frac{C_{\text{degraded}} - C_{\text{floor}}}{C_{\text{clean}} - C_{\text{floor}}}$$

where `C_floor` is a goal-blind arm measured **under the same corruption** (for which the corruption
is a structural no-op, so it also absorbs any survivorship artifact). `R = 1` no loss; `R = 0` fell
exactly to the floor; **`R < 0` means the arm ended up worse than having no command at all — its own
channel actively harmed it.** Measured on E78-F (three seeds pooled, `scripts/analyze_e78_dropout.py --floor`):

| arm | clean | f 0.1 / 0.1–0.5 s | f 0.1 / 0.5–1 s | f 0.3 / 0.1–0.5 s | f 0.3 / 0.5–1 s | f 0.5 / 0.1–0.5 s | f 0.5 / 0.5–1 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit | 0.924 | +0.656 | +0.325 | +0.101 | **−0.314** | **−0.337** | **−0.644** |
| SNMR | 0.688 | +0.780 | +0.648 | +0.460 | +0.310 | +0.349 | +0.148 |
| time code | 0.571 | +0.925 | +0.837 | +0.854 | +0.767 | +0.899 | +0.651 |
| shuffled | 0.531 | +0.948 | +0.881 | +0.756 | +0.509 | +0.703 | +0.442 |

Three things this makes visible that raw completion does not:

1. **`R` orders the arms inversely to clean performance** — the exact signature of the reliance
   theory. The clock keeps 65–93 % of its (small) advantage; the explicit reference loses all of its
   (large) one and then some.
2. **The sign is diagnostic.** Only the explicit arm goes negative, and only past ~0.3 masked
   fraction with ≥ 0.5 s outages. That is a *different failure mode* from degradation, and it is
   invisible in every metric this literature reports.
3. **It is a fair cross-arm comparison** in a way completion, retention ratios, and raw differences
   are not — which is exactly the gap AnyBody conceded in print ("not strictly equivalent... due to
   different observation spaces") and then stopped at.

A within-cell dose–response confirms the causal direction rather than assuming it. Regressing each
rollout's completion on *its own* realized masked fraction, and using the goal-blind arm to calibrate
the survivorship bias (short failing rollouts mechanically accumulate a lower masked fraction):
bias-corrected slopes per +10 pp masked are **explicit −0.109, SNMR −0.059, clock −0.004**. The
harm is dose-dependent in the arm that relies on the channel and absent in the one that does not.

**Reporting standard adopted for this program:** no cross-arm robustness contrast is reported
without (i) each arm's clean value, (ii) a goal-blind floor measured under the same corruption, and
(iii) `R` with its sign.

---

# Part III — The framework: a validity-aware command ladder (VACL)

The proposal is not a new representation. It is a contract with four parts, each earned by a
specific finding above, and each independently testable.

```
        ┌─ validity flags v_t  (is the reference live? how stale?)          ← §III.1
 x_t ──▶│
        ├─ command g̃_t  =  ladder(g_live, ĝ_extrap, ĝ_prior, ∅)            ← §III.2
        │        rung 0: true reference          (live)
        │        rung 1: dead-reckoned reference (short outage, free)
        │        rung 2: prior-rolled reference  (long outage, learned)     ← the latent's real job
        │        rung 3: no reference            (fall to the phase floor)
        ├─ phase φ_t  (never fails)                                          ← §III.3
        └─ z_t  (disambiguation + feasibility, when it adds information)     ← §III.4
```

## III.1 Validity is part of the interface, not an implementation detail

Two bits — `is_masked`, `staleness/horizon` — travel with the command. Earned by I.4: a policy that
cannot tell a fresh reference from a 0.8 s-old one has no way to discount it, and the measured
consequence is falling *below* the goal-blind floor. Implemented (`ReferenceDropoutMasker`), and
already carried by every E78 arm including the reference arm, so no arm's advantage can be "knowing
when it is blind".

## III.2 The ladder — and the measurement that reshaped it

The original proposal was "never hold, always synthesize". Two days of measurement say the truth is
more interesting, and more useful.

### III.2.1 In reference space, the fill's error is completely predictable

Reference-prediction error (rad RMSE against the true future reference, E70 walks, 400 random
starts per horizon; strictly causal predictors):

| outage | hold | constant velocity | cycle continuation | worst per-joint CV excursion |
| --- | ---: | ---: | ---: | ---: |
| 0.10 s | 0.083 / 0.128 | **0.050 / 0.088** | 0.138 / 0.223 | 1.3 rad |
| 0.20 s | 0.146 / 0.217 | 0.149 / 0.255 | **0.143** / 0.229 | 3.2 rad |
| 0.50 s | 0.230 / 0.308 | 0.466 / 0.743 | **0.149 / 0.241** | 8.8 rad |
| 1.00 s | 0.223 / 0.230 | 0.925 / 1.388 | **0.155 / 0.250** | 18.3 rad |
| 1.50 s | 0.200 / 0.292 | 1.347 / 2.081 | **0.164 / 0.247** | 27.7 rad |

(two numbers = `walk1_subject1` / `walk1_subject5`.) Three clean facts:

1. **First-order extrapolation is right only below ~0.2 s** and then diverges without bound — at a
   1 s outage it commands joint targets 18 rad outside the range of motion. Its validity horizon is
   a property of the gait, not a tuning choice.
2. **Holding is bounded but never good**: its error saturates near the range of motion (~0.2–0.3 rad)
   because a held pose is at least a real pose.
3. **A motion model is flat in horizon.** Cycle continuation — replaying the channel's own most
   recent matching cycle, model-free and strictly causal — holds 0.15–0.25 rad from 0.1 s to 1.5 s.
   *This is the property that matters:* it converts an unbounded-horizon problem into a bounded one.
   Matching the cycle in the SNMR latent space gives the same answer as matching it in joint space
   on these walks (0.153 vs 0.149 at 0.5 s) — consistent with everything else the program knows
   about cyclic single-motion walking, and the reason the interesting test is *aperiodic* motion.

### III.2.2 In policy space, reference accuracy does not transfer — and that is the finding

The same fills, fed to the **frozen, unmasked** explicit student (seed 0; clean 0.929; goal-blind
floor 0.419), completion:

| outage cell | hold | zero | constant velocity | cycle |
| --- | ---: | ---: | ---: | ---: |
| f 0.1, 0.5–1 s | 0.603 | **0.683** | 0.444 | 0.663 |
| f 0.3, 0.5–1 s | 0.280 | **0.386** | 0.116 | 0.346 |
| f 0.5, 0.5–1 s | 0.106 | **0.155** | 0.022 | 0.140 |
| f 0.1, 0.1–0.5 s | **0.757** | 0.740 | 0.685 | 0.723 |
| f 0.3, 0.1–0.5 s | 0.476 | **0.481** | 0.363 | 0.416 |
| f 0.5, 0.1–0.5 s | **0.267** | 0.265 | 0.124 | 0.207 |

Read the two tables together:

- The **ranking flips**. Blanking the reference has one of the *worst* prediction errors and the
  *best* completion at long outages; cycle continuation has by far the best prediction error and
  comes second; constant velocity is worst in both. Correlation between reference accuracy and
  completion across fills is weak and sign-unstable.
- **Why:** for a policy trained only on live references, every fill is out of distribution, and what
  decides the outcome is not "how close is this to the truth" but "what does this policy do when
  handed it". A plausible-but-out-of-phase pose is obeyed hard; a blanked goal apparently pushes the
  policy toward a neutral posture it survives better.
- **Consequence for the framework — the central design lesson of this synthesis:
  validity-awareness cannot be bolted on at deployment. The fill and the policy must be trained
  together.** A frozen policy can *rank* fills; it cannot establish what a fill is worth. This is
  the same lesson as E65 (train-time noise bought hold robustness) and as the masking design in
  E78, arrived at from the opposite direction.
- It also explains E78-F's cross-arm result without any appeal to representation: with frozen
  policies, a corrupted channel measures *reliance plus arbitrary OOD response*, and the arm whose
  response happens to be benign wins.

### III.2.3 The ladder, restated

Rung selection is **horizon-matched** and **learned**:

| rung | fill | validity horizon (measured) | who can build it |
| --- | --- | --- | --- |
| 0 | true reference | live | anyone |
| 1 | dead reckoning from `q̇_ref` | ≤ ~0.2 s | anyone, free |
| 2 | motion-model continuation | flat to ≥ 1.5 s | needs a model of the motion; model-free cycle matching suffices on periodic gait, a learned prior is required off it |
| 3 | no reference, phase + proprio floor | unbounded | anyone |

The staleness flag is what lets a *trained* policy move between rungs; the E78 masking recipe is
what teaches it to. Rung 2 is where a learned motion prior has a monopoly — a clock cannot emit 29
joint targets, and an explicit-only stack has nothing to roll forward beyond first-order velocity —
but that monopoly only pays on motion a cycle-matcher cannot handle, which is precisely the
aperiodic/interaction regime where E57-B already showed SNMR references beat GMR (fight1 +13 pp).

## III.3 Phase is a first-class channel, not a null control

E63 and E78-F both say the clock is a strong, unfailing signal that this program has only ever used
as a control arm. In VACL it is a permanent input: it costs 32 dimensions, cannot go stale, and
supplies exactly the "keep walking in the same rhythm" competence that the ladder's lower rungs need.

## III.4 The latent earns its place by supplying what the others cannot

Three roles, each with evidence and each separately gated: (a) **disambiguation** when several
futures share a present (E70 +0.191); (b) **regeneration** of the reference under outage (III.2);
(c) **feasibility/difficulty** signals for the curriculum (E1) and, with SNMR v2 heads, contact
intent — none of which the reference format carries. Cross-embodiment (E54) is where (a) and (b)
become *structural* rather than merely useful.

## III.5 Training: masking is what makes any of it real

E65's mechanism, applied to the right channel: a policy trained only on live references never
learns to use validity flags or lower rungs. Bernoulli-segment reference masking during training
(implemented) is what converts the ladder from a runtime hack into a learned policy — and E78-F
predicts the effect will be large *for the explicit arm*, which is the arm that matters.

---

# Part IV — Program (each item has a gate; cheapest first)

| id | question | design | gate | status |
| --- | --- | --- | --- | --- |
| **E79-a** | is a stale reference worse than a blanked one, within one arm? | frozen explicit, fill `hold` vs `zero`, full severity grid | `zero` − `hold` > 0 at ≥ 2 severities | **done (seed 0): yes at long outages (+0.08, +0.11, +0.05), ≈0 at short ones** |
| **E79-b** | does free dead reckoning recover the collapse? | fill = constant velocity | ≥ +0.10 at f ≥ 0.3 | **done: NO — worse than hold everywhere (−0.08…−0.16); its validity horizon is ~0.2 s and outages here are longer** |
| **E79-c** | does a motion-model fill beat both? | fill = causal cycle continuation (model-free) | > hold at long outages | **done (seed 0): beats hold at 0.5–1 s (+0.03…+0.07), loses at 0.1–0.5 s; still below `zero`** |
| **E79-d** | does reference accuracy predict completion? | compare the two tables in §III.2 | — | **done: NO. Ranking flips; the frozen policy's OOD response dominates.** ⇒ fills must be co-trained |
| **E80** | does masked co-training + a learned rung selector dominate, at no clean cost? | E78 arms retrained with masking; fills as observations, staleness flags live; primary conjunction as amended | mZf/ladder − mE-hold ≥ +0.10 **and** − mTl/mTf ≥ +0.05 **and** clean ≥ −0.01 | **the GPU night** |
| **E81** | is rung 2 worth a *learned* prior (vs model-free cycle matching)? | aperiodic clips (dance/fight/interaction); cycle-match vs SNMR-decoded rollout, reference-space error first (CPU), then policy | learned − cycle ≥ 0.05 rad at ≥ 0.5 s, then policy-level | cheap first half; the natural home for SNMR |
| **E82** | does the ladder transfer across embodiments? | E54 T1 teacher → shared-latent student; outage on the T1 reference | any rung-2 advantage replicates on T1 | after E80 |
| **E1-proper / E2** | retarget features predict and pre-empt failures | pool hook labels; warm-started sampler | incremental R² ≥ +0.10 held-out clips; ≥ 20 % sample reduction | independent |

**What E79 already settled, for the cost of a few GPU-minutes.** The whole "fix it at deployment"
branch is closed: no fill — free, model-free, or otherwise — restores a frozen policy under long
outages, and the best of them (`zero`) is best for a reason that has nothing to do with accuracy.
That is a genuinely useful negative: it means the E80 training run is not an optimisation of E79 but
the *only* way the ladder can work, and it tells us what E80 must contain — the fill has to be
present during training so the policy can learn what a stale-but-plausible target means, and the
staleness flag has to be the thing that lets it switch.

E81 is the reframed home for the latent, and it is now sharply posed: cycle matching already
achieves flat-in-horizon reference prediction *on periodic gait*, so a learned prior must earn its
place **off** the cycle — exactly the aperiodic/interaction regime where E57-B found SNMR references
beat GMR by 13 pp. Its first half is CPU-only reference-space error, so it can falsify itself before
any policy is trained.

---

# Part V — What would falsify this framework

- **E79-b null and E79-c null:** if neither dead reckoning nor prior rollout beats holding, then the
  harm is not about the *content* of the fill but about the policy's coupling to any fixed target,
  and the answer is architectural (a policy trained to ignore stale commands), not interface-level.
- **Masked training closes the gap by itself:** if a masked-trained explicit arm already reaches the
  floor gracefully with a naive hold, the ladder's upper rungs are unnecessary complexity and the
  finding is "train with dropouts, that's all".
- **Rung 2 never beats rung 1:** if the learned prior cannot outperform a linear extrapolation at
  any outage length, SNMR has no role in the deployed stack, and the program's remaining home is
  cross-embodiment (E54) and the curriculum (E1/E2) — both of which stand on their own.

Each of these is a publishable finding. That is the point: the framework is arranged so that the
experiment which kills it also produces the result that replaces it.

---

## Appendix — where every number lives

E52-v4/E36 (additive null) `docs/BENCHMARK_QUESTION_2026-08-12.md`; E63/E64/E65/E61-v4
`docs/EXPERIMENT_LOG.md`; E70 (0.924/0.688/0.571/0.531/0.419, A−T +0.191)
`/data/robotixx/snmr-research/e70/analysis_seed0-1-2.json` and `paper/e70_results.tex`; E72 phase
sensitivity `docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md`; E75 destruction
`/data/robotixx/snmr-research/e75_snmr_destruction`; E76 replication
`docs/E76_EVALUATION_REPLICATION.md`; E77 `docs/E77_DEGRADATION_PILOT.md` +
`docs/DEGRADED_COMMAND_RESEARCH_2026-08-15.md`; E78-F
`docs/E78F_FROZEN_DROPOUT_BASELINE_2026-08-16.md` and
`/data/robotixx/snmr-research/e78_masked_fusion/analysis_frozen_3seed_*`; E1 pilot
`/data/robotixx/snmr-research/e1_retarget_difficulty/`; E54 `docs/E54_T1_PORT_STATUS.md` +
`snmr/integration/holosoma_t1_wbt.py`.
