# Final ICRA Review and Research Completion Plan

**Date:** 2026-08-07
**Authoritative manuscript:** `paper/main.tex` at `bc6df49`
**Review scope:** manuscript logic, current experiment artifacts, statistical support,
method implementation, reproducibility, live E66/E54 outcomes, and ICRA 2027 submission
planning.

## 1. Executive verdict

### Scientific status

The project has produced a strong measurement apparatus and several unusually honest
negative results, but the original research goal is not yet complete.

The original goal was to make the retargeter's shared latent load-bearing for control and
to show that retargeting knowledge improves tracking through a reusable command interface.
The evidence currently establishes:

1. A learned 64-d command code can replace direct goal access at the action decoder on one
   cyclic G1 clip and match one evaluated explicit-command teacher checkpoint.
2. The frozen SNMR latent is causally used when it is the only goal source, but it performs
   below an absolute time-index encoding on that clip.
3. The frozen SNMR latent has no detected additive or reference-corruption benefit beside
   an explicit goal.
4. SNMR references are at least competitive with GMR references on the tested tracking
   tasks, with one hard-clip result favoring SNMR but only at one training seed.
5. A shared retargeter absorbs a second interaction-rich teacher at a small home-domain
   cost, but this result is currently one training seed per arm.

What is not established:

1. Motion content beyond time or trajectory index.
2. Stable distillation on a nontrivial aperiodic or multi-motion task.
3. A cross-embodiment control interface.
4. A tracking improvement caused by the retargeting latent.
5. Teacher-level parity as an algorithmic claim across teacher training seeds.

### Paper status

**Current internal verdict: borderline ICRA, roughly a 3/6 in its present form.**

The draft is credible, visually clean, and much stronger than a typical early paper. Its
main weakness is not integrity. It is a mismatch between a broad retarget-to-track thesis
and evidence concentrated on one deterministic cyclic trajectory. The newest E66 result
also invalidates the planned aperiodic comparison and is not yet reflected in the paper.

The paper can become submission-ready by either of two routes:

1. **Positive completion:** demonstrate that the SNMR latent beats a time-only control on
   a multi-trajectory task where time and proprioception are genuinely ambiguous, then add
   a scoped T1 portability result if time permits.
2. **Definitive negative completion:** run the same valid multi-trajectory instrument,
   show that the latent does not beat time, and make the paper a focused measurement and
   falsification paper rather than implying that the retargeting latent is already a
   useful control interface.

Both outcomes are scientifically complete. The current single-clip result is not.

## 2. What is already strong

These parts should be preserved.

1. **The exclusivity idea is useful.** Moving the goal behind a learned code and ablating
   the code's inputs is a clear way to measure an interface.
2. **The post-DEFECT-2 factorial is clean.** E52-v4 has three training seeds and matched
   1,024-rollout evaluations:

   | Prior input | Completion |
   | --- | ---: |
   | Explicit goal | 0.955 +/- 0.008 |
   | Explicit goal + frozen SNMR latent | 0.956 +/- 0.003 |
   | Time-index code | 0.754 +/- 0.010 |
   | Frozen SNMR latent | 0.656 +/- 0.044 |
   | No goal | 0.001 +/- 0.001 |

3. **The project repeatedly rejected attractive stories.** The valid E61-v4 robustness
   sweep, E63 time control, deterministic E62 arm, and MeanFlow mode-coverage discriminator
   all made the final interpretation more credible.
4. **The retargeter audit is nuanced.** "Instance-aligned, not linearly semantic" and
   "aligned, not invariant" are defensible formulations when kept tied to the probes used.
5. **The full-root two-teacher redo fixes the earlier leakage concern.** E55-R now uses
   predicted root pose and group-held-out sibling splits.
6. **The artifact trail is unusually rich.** Protocol snapshots, hashes, invalid-run
   markers, and negative-result records are valuable and should become an anonymous
   artifact rather than remain scattered across the worktree.

## 3. Research-goal completion audit

| Goal component | Required evidence | Current evidence | Status |
| --- | --- | --- | --- |
| Learned code is the decoder's only goal channel | Structural input audit plus clean causal code ablation | Structural slices are fixed; the cited 0.93 -> 0.00 reference-blanking result diagnoses the old leak, not the clean model | Partial |
| Learned code matches explicit command | Multiple teacher and student training seeds, equivalence margin | Three student seeds vs one teacher seed on one clip | Partial |
| Retargeting latent carries control-useful motion content | Beats matched proprio + time control where time is ambiguous | Loses to time on one deterministic clip | Not established |
| Retargeting latent improves tracking | Additive, robustness, generalization, or sample-efficiency gain | Clean additive and corruption tests are null | Not established |
| Interface works beyond one motion | Stable explicit positive control on multiple motions | E53 teacher below gate; E66 explicit student collapses | Not established |
| Cross-embodiment interface | Same command representation demonstrated on at least two robots | T1 port smokes; teacher run failed before training | Not established |
| SNMR references are trackable | Paired multi-seed comparison and assay sensitivity | Strongest completed downstream result; completion noninferiority still underpowered | Mostly established |
| Interaction-rich data can be absorbed | Clean splits, specialists, shared model, uncertainty | E55-R ordering is strong but one training seed per arm | Established descriptively |

## 4. Submission-blocking findings

### P0.1 The clock is a time-index code, not a content upper bound

The E63 control uses 16 log-spaced sinusoid pairs over absolute frame index, followed by a
fixed 32-to-128 projection. On a single deterministic trajectory this code can identify
time, and time identifies the entire target trajectory. This is true for a cyclic clip and
for an aperiodic clip.

Therefore:

1. E63 correctly shows that the SNMR latent does not beat a trivial time-index baseline on
   `walk1_subject5`.
2. E63 does **not** show that the SNMR latent contains "at most timing."
3. E63 does **not** upper-bound the information in the SNMR latent. It only bounds what the
   present controller extracts from it on this task.
4. A single aperiodic clip does not solve the confound. Absolute time can still memorize an
   aperiodic trajectory.

Required manuscript wording:

> On this single deterministic clip, the SNMR latent provides no more usable tracking
> performance than an absolute time-index code. The experiment cannot identify content
> beyond time because time uniquely identifies the trajectory.

Do not use:

> The latent contains at most timing.

The decisive experiment must contain multiple trajectories that share the same time code
but require different actions.

### P0.2 E66 is invalid as a latent comparison and reveals trainer instability

E66 completed after the current manuscript was built.

Artifacts:

- `runs/e66_aperiodic/teacher_eval.json`
- `runs/e66_aperiodic/explicit/c_prior_explicit_eval.json`
- `runs/e66_aperiodic/clock/a_prior_snmr_eval.json`
- `runs/e66_aperiodic/zret/a_prior_snmr_eval.json`

Results:

| Arm | Completion | Mean survival | Joint RMSE |
| --- | ---: | ---: | ---: |
| Explicit PPO teacher | 0.514 | 5.90 s | not reported |
| Explicit-goal student | 0.000 | 0.33 s | 0.318 rad |
| Time-index student | 0.000 | 0.58 s | 0.212 rad |
| SNMR-latent student | 0.000 | 0.36 s | 0.296 rad |

The explicit positive control failed, so clock versus latent is uninterpretable.

The training logs show actual divergence, not merely a hard task:

- Explicit arm action loss initially falls below 0.1, then ends at 10.05.
- Explicit arm KL reaches roughly `1.0e11`.
- Explicit arm smoothness reaches roughly `7.0e14`.
- The clock and SNMR arms also deteriorate late in training.

Method-level causes to address:

1. The teacher gate of 0.5 is too weak. Half of the teacher rollouts fail.
2. Teacher forcing reaches zero after 200 of 2,000 rounds regardless of student survival.
3. The implementation discards prior-round data. It is closer to iterative on-policy
   distillation than classical aggregated-dataset DAgger.
4. Only the final student is saved and evaluated, so a good intermediate model can be lost.
5. The smoothness target is stored under older network parameters and then compared with a
   repeatedly updated network. It also sums over all 64 dimensions without normalization.
6. No finite-loss or latent-norm gate aborts a diverging run.

E66 should be logged as:

> Inconclusive latent comparison; explicit distillation control failed. The run identifies
> an online-distillation stability defect on the harder task.

It must not be presented as evidence that either the clock or the latent fails on
aperiodic motion.

### P0.3 "Within seed noise of its teacher" is unsupported

The paper has three student training seeds and one teacher training seed. Student seed
variation does not estimate teacher seed variation.

Replace every instance of:

> within seed noise of its teacher

with:

> matches the evaluated teacher checkpoint at the same 1,024-rollout protocol

until at least three independently trained teacher checkpoints are available.

For an algorithmic parity claim, pre-specify:

- completion equivalence margin: +/-5 percentage points;
- joint-RMSE relative margin: +/-5%;
- three teacher and three student training seeds;
- identical phase starts and evaluation randomization;
- paired bootstrap or a hierarchical bootstrap over training seeds and rollout windows.

### P0.4 The latency result is not yet causally attributed to noise

E65 compares:

- a CVAE arm with posterior, KL, episodic latent noise, and smoothness;
- a deterministic encoder without all four.

The 100 ms result is large and useful:

- CVAE: 0.485 completion;
- deterministic: 0.027 completion;
- absolute difference: +45.8 points.

But the current comparison cannot isolate train-time noise. The paper currently says
"train-time latent noise buys" the gain while also admitting that the isolating arm is
only registered.

Required action:

Run E65b: deterministic encoder plus the same episodic latent noise, with the posterior and
KL still absent. Prefer three training seeds.

Interpretation:

- If matched-noise deterministic follows the CVAE curve, attribute robustness to noise.
- If it follows the noise-free encoder, posterior/KL/smoothness or an interaction matters.
- If it is intermediate, report the decomposition rather than a single-cause claim.

Until then say:

> The noise-trained CVAE variant is more robust to command hold than the noise-free
> deterministic variant.

Report the absolute point difference before the `18x` ratio; the ratio is unstable near a
floor.

### P0.5 The cited exclusivity ablation diagnoses the old leak

`paper/main.tex:241-242` cites a drop from 0.93 to 0.00 after blanking
reference-derived dimensions as causal verification of the current exclusivity contract.
That experiment showed that the **old leak-affected decoder** used the leaked reference. It
does not verify the clean v4 model.

For the clean model:

1. Keep the structural assertion and unit test that decoder input contains exactly the
   90-d proprioceptive slice plus `z_cmd`.
2. Add a clean causal ablation on arm C:
   - zero `z_cmd`;
   - shuffle `z_cmd` across environments;
   - randomize `z_cmd` with matched marginal scale.
3. Confirm collapse over all three student seeds.

Then state separately:

- structural exclusivity: the decoder has no reference-derived input dimensions;
- causal use: destroying `z_cmd` destroys tracking.

### P0.6 The paper carries too many partially connected stories

The current 225-word abstract contains:

- learned command interface;
- time-index null;
- latent audit;
- two-teacher scaling;
- simulator defect;
- negative-results release.

The third contribution bullet combines a retargeter, two-teacher scaling, a defect report,
protocols, and a ledger. This reads as several papers sharing a repository.

The ICRA paper needs one primary thesis:

> Exclusive command interfaces make retarget-to-track information measurable, and a
> time-only null exposes what single-trajectory experiments cannot establish.

The SNMR audit supports that thesis. The two-teacher and defect results should be shortened
unless the new multi-trajectory result directly uses them.

## 5. Major claim corrections

| Current location | Problem | Required correction |
| --- | --- | --- |
| `main.tex:32-33` | "none has measured" is universal | Use "has rarely been measured directly" or a scoped knowledge claim |
| `main.tex:35-36,167-168` | Goal is said to be invisible to the actor, but the prior is part of the policy | Say "invisible to the action decoder" |
| `main.tex:36-37,170-182,387-388` | Teacher parity uses one teacher seed | Use checkpoint-level matching language |
| `main.tex:48-50` | "most published demonstrations" is uncited; 79% implies a linear recovery scale | Remove "most" or cite it; report 0.754 vs 0.954 directly |
| `main.tex:125-127,401-405` | Clock is called an upper bound on latent content | Call it a required null baseline for this task |
| `main.tex:183-188,245-248` | Effective rank plus z-only linear R2 does not prove "not a copy" | Report these as linear diagnostics; add incremental/partial probes |
| `main.tex:186-187,347-360` | Noise attribution precedes E65b | Use variant-level comparison until isolated |
| `main.tex:241-242` | Causal ablation refers to leak-affected run | Replace with clean structural and causal evidence |
| `main.tex:286-303` | "pure DAgger" overstates the implementation | Implement replay/aggregation or say "DAgger-style online distillation" |
| `main.tex:398-402` | "at most timing" is not identified | Say "no advantage over time was measured" |
| `main.tex:445-454` | Aperiodic single clip is treated as resolving time confound; literature-budget claim is uncited | Replace with multi-trajectory ambiguity requirement; cite or remove budget norm |
| `main.tex:526-527` | "unconditionable" and "per-frame unsamplable" are universal | Scope to available conditioning and tested MeanFlow sampler |
| `main.tex:549-551` | "statistically indistinguishable" risks equating no detection with equivalence | Say "no difference was detected"; retain failed completion noninferiority |
| `main.tex:565-566` | "never track worse" conflicts with dance completion 0.32 vs 0.33 | Say "no material degradation was observed; dance was -1 point and fight +13 points at one seed" |
| `main.tex:596-597` | Shared defect does not guarantee source comparison is unbiased | Say the paired result is suggestive and is partly checked by post-repair hard clips |
| `main.tex:623-624` | Aperiodicity is said to defeat time | Require multiple trajectories or branching ambiguity |
| `main.tex:647-650` | Conclusion is broader than the task | Add "on one cyclic G1 trajectory" and avoid information-content claims |

## 6. Experiment-by-experiment audit

### E52-v4 command factorial

**Verdict:** paper-grade, with scoped claims.

Strengths:

- post-leak repair;
- three training seeds;
- 1,024 rollout evaluations;
- no-goal and additive controls;
- consistent completion, survival, RMSE, and imitation-loss ordering.

Remaining work:

- three teacher training seeds for parity;
- clean `z_cmd` destruction test;
- report per-seed RMSE rather than only the aggregate rounded value;
- do not interpret single-clip latent performance as semantic content.

### E60 goal-conditioning factorial

**Verdict:** useful one-seed mechanism result.

It supports the claim that goal-conditioning, not prior-path action mixing, is the main
design change. State `n=1` beside the table or sentence.

### E62 deterministic encoder

**Verdict:** strong ablation at nominal rate, one training seed.

It establishes no detected advantage for the full CVAE at 20 ms on this task. It does not
show the deterministic model is better. Pair it with E65b before making a final architecture
recommendation.

### E63 time-index control

**Verdict:** central and paper-grade as a null, but currently overinterpreted.

Rename it consistently to **absolute time-index control**. "Phase clock" suggests a single
periodic phase variable, while the implementation is a multi-frequency positional encoding.

### E65 hold robustness

**Verdict:** strong descriptive deployment result, incomplete attribution.

Keep only if E65b lands before the writing freeze. Otherwise move the curve to supplementary
material and reduce it to one caveated sentence.

### E66 aperiodic triad

**Verdict:** invalid comparison, useful trainer failure.

Do not rerun the same protocol unchanged. First stabilize the explicit positive control and
replace the single-clip design with the multi-trajectory design in Section 7.

### E54 T1

**Verdict:** infrastructure mostly ready; no teacher result.

At review time:

- `runs/e54_t1/teacher_ckpt.txt` is empty;
- training failed with CUDA out-of-memory before environment construction;
- evaluation then attempted checkpoint `"."` and failed;
- `TEACHER_DONE` was still written;
- the GPU was occupied by an external `actor_rollout_generate_sequences` process using
  about 13.35 GiB.

The launcher uses `set -uo pipefail`, not `set -euo pipefail`, so failed commands do not stop
the protocol. Fix this before any rerun and write success markers only after validating a
nonempty checkpoint and evaluation JSON.

The T1 configs currently live as uncommitted edits in the Holosoma clone. Export them as a
committed patch or vendor the configuration files into the anonymous artifact.

### E55-R two-teacher scaling

**Verdict:** strong descriptive result, limited uncertainty.

The ordering is convincing:

- LAFAN specialist: 4.1 cm locomotion, 58.6/52.6 cm interaction;
- Omni specialist: 25.9 cm locomotion, 11.5/10.8 cm interaction;
- shared model: 5.9 cm locomotion, 12.2/11.9 cm interaction.

Required disclosure:

- one training seed per arm;
- number of held-out clips and sibling groups in each cell;
- bootstrap interval across held-out groups, if available.

Do not call object modes "unconditionable." The tested object-pose channel explained little;
other variables or stronger conditioning mechanisms remain possible.

### E56-C MeanFlow

**Verdict:** useful negative discriminator, not a universal impossibility result.

The valid claim is:

> This per-frame MeanFlow model matched spread magnitude but did not improve nearest-sibling
> coverage in 7 of 8 held-out groups.

The invalid stronger claim is:

> Per-frame generation cannot sample the modes.

### E57 reference trackability

**Verdict:** good supporting result.

- Original matrix: 3 training seeds per source and two evaluation seeds; joint RMSE
  noninferiority established, completion noninferiority not established.
- Positive control: the assay detects 0.05 rad i.i.d. corruption.
- Hard clips: dance is -1 completion point with better survival; fight is +13 points and
  +1.1 s survival for SNMR, one seed per source.

Keep the original matrix as the statistical result. Present the hard clips as descriptive
stress tests, not proof that SNMR is universally equal or better.

### Defect reports

**Verdict:** valuable validity evidence, but currently over-weighted in the narrative.

The body-indexing defect should remain because it changes the interpretation of all earlier
tracking results. The observation-slicing defect should remain because it validates the
exclusivity contract repair.

Compress both into an "Experimental validity and defect audit" subsection unless the
framework defect is intentionally submitted as a separate systems contribution.

## 7. Decisive next experiment: multi-trajectory ambiguous-time test

This is the single highest-value next experiment.

### 7.1 Research question

> Does the frozen SNMR latent provide usable motion information beyond proprioception and
> absolute time when the same time code can correspond to different desired motions?

### 7.2 Why this design is necessary

On one deterministic clip:

```text
time index -> unique target state/action
```

The time control can therefore memorize the task. The confound disappears only when:

```text
(time index, current proprioception) -> multiple plausible future commands
```

### 7.3 Teacher design

Do not wait for one eight-clip PPO teacher.

Use a **specialist-teacher ensemble**:

1. Train or reuse one strong PPO specialist per clip.
2. Route teacher labels by the environment's motion ID during distillation.
3. Never expose motion ID to the student.
4. Require each specialist to pass:
   - completion >=0.80;
   - mean survival >=9.0 s;
   - stable joint RMSE;
   - 1,024-rollout evaluation.

This removes the failed E53 unified-teacher optimization problem from the representation
test. The student still has to be one policy across all clips.

Start with two or three clips, not eight. Select clips for ambiguity and teacher quality,
not breadth. A reasonable first pool is two locomotion/dynamic clips with reliable
specialists and distinct trajectories. Add push only after its teacher quality improves.

### 7.4 Student arms

Use the same network, optimizer, budget, and evaluation protocol for all arms:

| Arm | Prior input beyond proprioception | Purpose |
| --- | --- | --- |
| C | Explicit robot goal | Positive control |
| A | Frozen SNMR latent window | Main hypothesis |
| T | Shared absolute time-index code, reset identically for every clip | Null |
| B | Nothing | Proprioception control |
| S | Shuffled SNMR latent from another clip at the same normalized time | Causal/content control |

The time arm must receive no clip ID, filename feature, motion index, or per-clip
normalization statistic.

Use one global SNMR-latent normalization over the training pool. Per-clip standardization
can leak clip identity or remove meaningful between-clip structure.

### 7.5 Ambiguity-window evaluation

Macro-average over clips, but make the primary analysis a precomputed ambiguity set.

Construct paired windows satisfying:

1. same normalized time bin;
2. small current proprioceptive-state distance;
3. large future target-trajectory distance over the next 0.5-1.0 s.

This directly tests windows where time plus current state is insufficient.

Report:

- completion and survival over all windows;
- completion and action error on ambiguity windows;
- per-clip metrics;
- teacher-student action error;
- failure timing around branch points.

### 7.6 Training stabilization required first

Implement these changes before the full run:

1. Save checkpoints every 50 rounds.
2. Evaluate a fixed validation set every 50-100 rounds.
3. Select the best validation checkpoint, not the final checkpoint.
4. Abort on nonfinite losses or a pre-specified latent/KL threshold.
5. Keep a nonzero teacher-mixture floor until the student passes a survival gate.
6. Add a replay buffer or aggregate data across rounds. Otherwise rename the method
   "DAgger-style online distillation."
7. Recompute both sides of the temporal smoothness loss under current model parameters,
   using paired consecutive inputs.
8. Normalize smoothness by latent dimension and valid transition count.
9. Start with the deterministic E62 architecture. Add CVAE noise only after explicit
   multi-trajectory distillation is stable.
10. Add a multi-motion unit test that verifies goal, proprioception, latent, clock, and
    motion-ID isolation.

### 7.7 Pre-specified decision gates

Run three training seeds after a seed-0 smoke test.

Positive-control gate:

- explicit student macro completion >=0.80;
- or within 5 points of the teacher ensemble;
- no divergent validation curve.

Main content gate:

- SNMR latent exceeds time control by >=10 completion points on ambiguity windows;
- paired 95% interval excludes zero;
- shuffled-latent arm loses the gain;
- result holds on at least two clips.

Interpretation:

1. **A > T:** SNMR contains control-usable information beyond time on multi-trajectory
   tracking. This completes the central positive claim.
2. **A ~= T and C passes:** the retargeting latent does not expose useful information beyond
   time under this interface. This completes a definitive negative paper.
3. **C fails:** trainer/task invalid; no representation conclusion.
4. **B ~= T:** time adds little beyond proprioception; re-examine reset-state leakage and
   ambiguity selection.

## 8. Supporting experiment priority

### Priority 1: teacher-seed replication

Train two additional walk1 explicit teachers and evaluate all three at 1,024 rollouts.
This upgrades checkpoint matching into an algorithmic parity statement.

### Priority 2: E65b matched-noise arm

Train deterministic encoders with episodic noise at the same scale and run the hold curve.
Use three seeds if feasible.

### Priority 3: clean command-code ablation

Zero, shuffle, and randomize `z_cmd` on clean arm-C checkpoints. This closes the causal
exclusivity evidence gap cheaply.

### Priority 4: incremental information probes

E64 currently compares linear prediction from `z_cmd` alone with prediction from
proprioception alone. Because `z_cmd` is a function of both proprioception and goal, the
correct question is incremental information.

Fit on held-out trajectories:

1. goal from proprioception;
2. goal from `z_cmd`;
3. goal from proprioception + `z_cmd`;
4. goal from proprioception + shuffled `z_cmd`;
5. nonlinear versions of 1-4.

Report incremental `R2` and avoid information-theoretic language unless a real information
bound is measured.

### Priority 5: T1 portability

After the central multi-trajectory result:

1. fix fail-fast behavior and remove the false success marker;
2. rerun T1 teacher only when the GPU is genuinely available;
3. require a stronger teacher gate than 0.5;
4. first demonstrate an explicit-goal T1 student;
5. then compare SNMR latent versus shared time control;
6. state clearly whether policy parameters are shared or only the interface representation
   is shared.

A separate T1 policy using the same human-side latent demonstrates representation
portability, not one universal tracking policy.

### Priority 6: uncertainty for E55-R and E57-B

If compute remains:

- bootstrap E55-R over held-out groups;
- add training seeds for the fight/dance source comparison.

These should not delay the central experiment.

## 9. Recommended ICRA paper architecture

ICRA 2027 permits eight complete pages including references. The current PDF is seven
pages, all fonts are embedded, the page size is letter, and PDF author metadata is blank.
Use the remaining page for decisive evidence, not another side study.

### Proposed title

Keep the current title if the multi-trajectory result is negative or mixed:

> What Crosses the Boundary? Measuring the Retarget-to-Track Interface for Humanoid Tracking

If the latent wins decisively:

> Beyond Time Index: Measuring Learned Retarget-to-Track Interfaces for Humanoid Tracking

### Proposed section order

#### I. Introduction

Four paragraphs:

1. Standard retargeting-to-tracking interface and why it is unmeasured.
2. Measurement instrument: exclusive learned command code.
3. Key result: single-clip results are confounded by time; multi-trajectory result resolves
   the confound.
4. Three contributions only.

#### II. Related Work

Organize around:

1. retargeting and trackability;
2. latent command policies and distillation;
3. representation auditing and time/phase controls.

Add missing foundational citations:

- DAgger;
- LAFAN1;
- VAE/CVAE if retained;
- CKA/probing methodology;
- MeanFlow if retained;
- the named simulator/framework associated with the defect.

Seventeen references is sparse for the current breadth.

#### III. Measurement Instrument

Include:

- formal interface;
- decoder-level exclusivity;
- SNMR latent source;
- stabilized online-distillation algorithm;
- structural and causal validation.

Call the prior/encoder and action decoder separate components. Do not say the goal is
invisible to the whole actor if the prior sees it.

#### IV. Experimental Protocol

Put all validity-critical details in one place:

- robots, clips, train/test split;
- completion definition;
- teacher/student seeds;
- rollout counts;
- equivalence margins;
- time-index construction;
- multi-trajectory ambiguity set;
- defect repairs and pinned revisions.

#### V. What Crosses the Boundary?

This becomes the central results section:

1. single-clip factorial;
2. time-index null;
3. multi-trajectory ambiguous-time factorial;
4. clean command-code causal ablation.

This section should answer the title directly.

#### VI. What Does the Retargeting Latent Contain?

Compress the audit into one table:

| Property | Probe | Result | Valid conclusion |
| --- | --- | --- | --- |
| Instance alignment | retrieval/CKA | values | aligned across embodiments |
| Semantics | linear category probe | near chance | not linearly accessible |
| Embodiment | linear/MLP probes | 0.28/0.91 | aligned, not invariant |
| Contact | probe and co-training | values | weak unless supervised |
| Control | multi-trajectory result | new | content beyond time or definitive null |

#### VII. Scaling and Downstream Utility

Keep only:

- compact E55-R table with sample counts/uncertainty;
- reference trackability result;
- one sentence on MeanFlow mode coverage if it supports the sequence-level limitation.

Move the detailed interaction-conditioning chain and negative ledger to supplementary
material or the anonymous repository.

#### VIII. Validity, Limitations, and Conclusion

Combine:

- defect audit;
- simulation-only limitation;
- one/multiple robot scope;
- data scale;
- final answer to the research question.

### Page budget

| Material | Target pages |
| --- | ---: |
| Abstract + Introduction + Fig. 1 | 1.4 |
| Related Work | 0.55 |
| Instrument | 0.8 |
| Protocol | 0.6 |
| Central results | 1.5 |
| Latent audit | 0.8 |
| Scaling/trackability | 0.7 |
| Validity/limitations/conclusion | 0.55 |
| References | 1.1 |
| Total | 8.0 |

## 10. Abstract rewrite blueprint

The current abstract is about 225 words and carries too many claims. Target 160-180 words.

Use six sentences:

1. **Problem:** the standard interface is robot joint targets and its information content is
   not directly measured.
2. **Instrument:** an exclusive command code makes input ablations possible.
3. **Positive control:** on one G1 walk clip, the code matches the evaluated explicit teacher
   checkpoint.
4. **Critical null:** an absolute time-index code beats the frozen SNMR latent, showing that
   single-trajectory results cannot establish motion content.
5. **Decisive result:** insert the multi-trajectory result, positive or negative.
6. **Supporting evidence:** one short clause on the audited five-robot retargeter and
   released protocols.

Do not put the simulator defect, two-teacher scaling, latency result, and full audit in the
abstract simultaneously.

## 11. Figure and table plan

### Figure 1

Keep the interface schematic, but revise:

- "goal never reaches actor" -> "goal never reaches action decoder";
- "phase clock" -> "absolute time index";
- "caps claims" -> "required single-trajectory null."

Add one small strip of actual G1/T1 or walk/push frames if space permits. The paper currently
contains only schematics and plots, not a visual example of the robot/task.

### Figure 2

Replace the tiny DAgger loop with the multi-trajectory ambiguity design:

- two motions;
- same time code;
- different desired commands;
- explicit and SNMR inputs distinguish them;
- time-only input does not.

The method loop can be a compact algorithm box or supplementary figure.

### Figure 3

Keep the latency curve only after E65b. Otherwise use this slot for the central
multi-trajectory result.

### Table I

Keep the single-clip factorial, but:

- rename "phase clock" to "time-index control";
- scope the caption to one clip;
- add per-seed or interval detail for RMSE;
- describe the teacher as one checkpoint.

### Table II

Keep only with:

- `n` for clips/groups;
- one-training-seed disclosure;
- interval across held-out groups if computable.

### New central table

Multi-trajectory results:

| Input | Macro completion | Ambiguity-window completion | Survival | Action error |
| --- | ---: | ---: | ---: | ---: |
| Teacher ensemble | | | | |
| Explicit | | | | |
| SNMR latent | | | | |
| Time index | | | | |
| No goal | | | | |
| Shuffled SNMR latent | | | | |

## 12. Reproducibility and artifact work

### Must fix

1. Key paper results are currently untracked run artifacts.
2. The worktree contains deleted historical evaluations and many untracked outputs.
3. T1 configuration changes exist only in a dirty external clone.
4. E54 writes success markers after failure.
5. E66 is complete but not yet recorded in `EXPERIMENT_LOG.md`.

### Anonymous artifact layout

```text
artifact/
  README.md
  environment/
    python-lock.txt
    system.txt
  manifests/
    experiments.json
    sha256.txt
  protocols/
    e52_v4.sh
    multi_trajectory.sh
    e55r.sh
  results/
    paper_tables.json
    paper_figures/
  patches/
    holosoma_body_index.patch
    holosoma_t1_config.patch
  tests/
    commands.md
```

Each paper number should map to:

- experiment ID;
- exact command;
- SNMR revision;
- Holosoma revision;
- input hashes;
- training seed;
- evaluation seed;
- artifact path;
- analysis script.

### Verification performed during this review

- Focused experiment, sampling, latent-export, WBT-integration, repair-export,
  confirmatory-analysis, and sharing-cost tests: **29 passed**.
- A full test-suite attempt reached 36% and was terminated by SIGTERM, so a complete
  green-suite claim was not established during this review.
- Current PDF: 7 pages, letter size, fonts embedded, blank author metadata.
- Current LaTeX build: no overfull boxes in the final `main.log`; remaining warnings are
  underfull boxes, PDF-string tokens, and one unavailable small-caps italic font shape.

## 13. ICRA 2027 compliance and schedule

Official current requirements relevant here:

- paper deadline: **2026-09-15**;
- complete paper limit: **8 pages including references**;
- double-anonymous review;
- IEEE conference format;
- optional video strongly encouraged, maximum 3 minutes;
- first video upload window closes **2026-09-09**;
- supplementary deadline: **2026-09-22**.

Sources:

- ICRA 2027 call for papers:
  `https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/`
- ICRA 2027 author instructions:
  `https://2027.ieee-icra.org/contribute/paper-submission-instructions/`
- ICRA 2027 double-anonymous review:
  `https://2027.ieee-icra.org/contribute/double-anonymous-review/`

Before submission:

1. Build with the official ICRA/IEEE template bundle, not only the locally copied class.
2. Check PDF eXpress/PaperPlaza compliance.
3. Remove author-identifying repository URLs, paths, acknowledgments, and metadata.
4. Create an anonymous immutable artifact snapshot.
5. Review the conference generative-AI disclosure policy if AI-produced prose is inserted
   into the manuscript; authors remain responsible for every claim and citation.

## 14. Deadline-driven work plan

### Aug 7-10: freeze truth and repair the experimental harness

- Log E66 as an invalid comparison/trainer failure.
- Fix E54 fail-fast and success-marker behavior.
- Add periodic checkpoints, validation evaluation, finite-loss gates, and normalized
  smoothness.
- Decide whether to implement true data aggregation or rename the algorithm.
- Add clean `z_cmd` causal ablations.
- Draft the multi-trajectory protocol before running it.

### Aug 11-16: seed-0 decisive smoke

- Prepare two strong specialist teachers.
- Build teacher routing by motion ID.
- Run deterministic explicit, time, SNMR, no-goal, and shuffled-latent arms at seed 0.
- Stop immediately if the explicit arm misses its gate.
- Inspect ambiguity-window behavior before scaling seeds.

### Aug 17-24: three-seed central experiment

- Run three seeds for the valid arms.
- Produce paired intervals and per-clip results.
- Run two additional teacher seeds on walk1 in parallel if compute permits.

### Aug 25-29: attribution and portability

- Run E65b matched-noise isolation.
- Finish clean capacity probes.
- Rerun T1 teacher only if the GPU is free and the central result is already secure.

### Aug 30-Sep 4: rewrite and figure freeze

- Rewrite title/abstract/introduction around the actual multi-trajectory verdict.
- Replace the DAgger figure with the ambiguity experiment.
- Reduce defect and interaction side stories.
- Expand references.
- Generate every table and figure from one paper-results manifest.

### Sep 5-9: internal review and video

- Run cold technical, statistical, robotics, and clarity reviews.
- Produce a <=3 minute video:
  1. standard versus exclusive interface;
  2. explicit/time/SNMR/no-goal rollouts;
  3. multi-trajectory ambiguity example;
  4. optional G1/T1 or two-teacher qualitative result;
  5. concise limitations.
- Upload in the first video window if possible.

### Sep 10-14: submission hardening

- Full test suite.
- Clean-clone reproduction of paper tables.
- Official-template and PDF compliance.
- Double-anonymous audit.
- Citation and claim-to-artifact audit.
- Final proofread by someone not involved in the experiments.

### Sep 15: submit

Do not start new architectural branches after Sep 4 unless they repair a direct
submission blocker.

## 15. Stop rules

### Positive paper is ready when

1. explicit multi-trajectory student passes its gate;
2. SNMR latent beats time on ambiguity windows with three seeds;
3. clean code-destruction ablation confirms causal use;
4. teacher checkpoint matching is correctly scoped or teacher seeds are replicated;
5. E65 attribution is either isolated or removed;
6. all central artifacts reproduce from a clean checkout.

### Negative measurement paper is ready when

1. explicit multi-trajectory student passes its gate;
2. SNMR latent fails to beat time under the valid ambiguity design;
3. the paper states that result directly and removes interface-benefit implications;
4. the audit explains why the negative is plausible;
5. the result is replicated across seeds and clips.

### The paper is not ready when

1. the explicit positive control fails;
2. the central conclusion depends on one deterministic clip;
3. the time control has access to a motion ID or per-clip leak;
4. teacher parity is described as multi-seed without teacher seeds;
5. pending attribution experiments are written as completed causal findings;
6. success markers can be produced after failed training.

## 16. Recommended final contribution set

If the multi-trajectory latent result is positive:

1. **A measured retarget-to-track interface:** exclusive learned command code with clean
   causal validation and teacher-checkpoint parity.
2. **A time-index null and its resolution:** single-trajectory latent claims are confounded,
   while SNMR exceeds time on multi-trajectory ambiguity windows.
3. **An audited reusable retargeter:** five embodiments, two-teacher absorption, and
   downstream reference trackability.

If the result is negative:

1. **A measured retarget-to-track interface:** the first controlled completion curve over
   explicit goal, retargeting latent, time, and no-goal inputs.
2. **A definitive null result:** the retargeting latent does not beat time even when time is
   ambiguous, despite being causally used.
3. **A diagnostic audit and released falsification protocol:** representation probes,
   defect checks, and negative-results ledger explaining which claims the evidence does and
   does not support.

That is a complete research outcome either way. The key is to finish the valid
multi-trajectory test rather than add more single-clip variants.
