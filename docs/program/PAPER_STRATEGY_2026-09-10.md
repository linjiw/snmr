# SNMR PAPER STRATEGY — 2026-09-10 (T−5 to ICRA 2027)

**Grounding.** All anchors are `paper/main.tex` at HEAD (`6435ed4`), 841 lines, 8 letter pages, 0 overfull boxes. `git status` shows `paper/` **unmodified** — commit `6435ed4` landed `docs/E70_SHUFFLE_CONTROL_AUDIT_2026-09-10.md` but did **not** touch the manuscript, so every wrong sentence identified below is still in the build.

**The binding constraint on everything in Part A: the paper is already at the 8-page cap.** Every addition must be funded by a cut. R7 is therefore not "last" — it runs in parallel with R1–R6 from Day 1.

**Second constraint: every displayed quantity is renderer-generated** (`paper/e70_results.tex`, `e70_destruction.tex`, `e70_temporal.tex`, `e72_phase.tex`, `e70_validity.tex`, each hash-stamped). No revision below changes a number. This is the single most important fact for the submit/don't-submit decision: the entire revision list is prose.

---

# (A) PAPER 1 — REVISION LIST, PRIORITY ORDER

## P0 — Correctness. Ship-blocking. Do not submit without these.

### R1. Relabel the shuffled arm as a phase-matched, misaligned-reference control

**Established facts to encode** (verified in code, not inherited from the audit doc):
- `snmr/integration/distillation.py:241-249`: `source = (destination + 1) % num_motions`. A deterministic bijection, so `destination = (donor − 1) mod n`. Donor identity determines destination identity **at every pool size** — this is not a two-clip artifact.
- `scripts/run_e70_multitraj.sh:19` → `CLIPS=(walk1_subject1 walk1_subject5)`, so E70 ran `0→1, 1→0`: a pure swap (involution).
- `git log -- snmr/integration/distillation.py` returns exactly one commit, `061cdec` — the commit that created the E70 pipeline. The reviewed public code *is* the code that produced the frozen students.
- The identity-free null is **T**, pinned by `tests/test_distillation.py:63::test_shared_time_index_resets_without_motion_identity_leak`.
- Measured between-clip mean separation after per-arm standardization: **A = 8.042 SD, S = 8.042 SD, T = exactly 0.000.**

**Anchors requiring edits:** 101–102 (abstract), 183–185 (contribution 3), 434 (table row label), 485 (conditional branch), **505–507 (factually wrong)**, 528–529, **671–675 (factually wrong)**, 729–731, 741–743, and `paper/e70_teaser.tex:182-183` (Fig. 1c caption).

**Do NOT touch line 174 or 260.** The word "shuffle" there refers to the `z_cmd` **batch-shuffle destruction mode**, a different operation. That collision is itself a readability hazard — renaming the S arm to "misaligned reference" resolves it for free.

**Exact replacement — abstract (replaces lines 100–102):**

> Under that construction the frozen latent leads an equally trained clock---an identity-free code, bit-identical across clips at equal local frame indices by construction and by regression test---by $\EATDifference$ [\EATCILow, \EATCIHigh], and leads a phase-matched \emph{misaligned-reference} control by $\EASDifference$ [\EASCILow, \EASCIHigh] in the \EEndpointLabel.  That second control supplies the other walk's real latent at matched normalized time: it withholds the correct future, not clip identity.

**Exact replacement — arm description (insert in §\ref{sec:exp} after line 419, replacing the trailing clause of line 410):**

> S is a phase-matched, \emph{misaligned-reference} control, not an identity-erasing one.  It substitutes the other clip's real latent at matched normalized time through the deterministic map $\mathrm{donor}=(\mathrm{destination}+1)\bmod n$, a pure swap on this two-clip pool, so clip identity remains recoverable in principle from S's command at every pool size.  The assay's identity-free null is T.  Measured on the frozen latents, between-clip mean separation after per-arm standardization is 8.04\,SD for \emph{both} A and S and exactly 0.000 for T: S carried as much linearly decodable clip identity as the winning arm, and scored \EShuffledAmb{}---indistinguishable from identity-free T at \ETimeAmb{}.  Identity was maximally available and bought nothing.

**Exact replacement — §\ref{sec:phase}, lines 505–507:**

> The misaligned-reference control cannot separate this either: it withholds the correct future while leaving clip identity recoverable, so it isolates neither a label reader nor a future reader.

**Exact replacement — Limitations, lines 671–675:**

> The matched time null rules out absolute progress through these clips, not every temporal or phase model.  The misaligned-reference control S withholds the correct future but does not erase clip identity: its donor map is the deterministic bijection $(\mathrm{destination}+1)\bmod n$, a pure swap on this two-clip pool, so identity is recoverable in principle from S's command.  We therefore do not claim S bounds what an identity-only command can achieve; that role belongs to T.  What we can report is that S's code carried between-clip separation equal to A's (8.04\,SD, versus 0.000 for T) and still scored \EShuffledAmb{}: the identity was present and the students did not invert the map.  That bounds \emph{accidental} exploitation; it is not evidence that a directly supervised clip-ID-plus-phase reader would fail.  Such an arm requires a new training arm rather than a re-evaluation and is unrun.

**Table row (434):** `Matched-phase shuffle (S)` → `Misaligned reference (S)`. Same in Fig. 1c caption and lines 528–529, 729–731, 741–743.

**Explicitly state that the primary contrast is unaffected.** Add to §\ref{sec:exp} beside the gate sentence (after line 475):

> The registered primary contrast is A--T.  T's code is identity-free by construction and by regression test, so the primary comparison is unaffected by S's construction; only the A--S secondary contrast is reinterpreted.

**Cost:** ~18 added lines, ~6 deleted. Half a day.

**Also do (15 min, high value):** add a phrase guard. `scripts/audit_e70_final_bundle.py` has no phrase check (verified — no `banned`/`phrase`/`forbidden` token in it). Add a test asserting `paper/main.tex` contains none of `identity-erasing`, `content-free null`, `destroys clip identity`, `erases clip identity`. The three properties are already pinned in `tests/test_distillation.py`; this pins the prose to them.

---

### R2. Disclose the lookahead asymmetry's *direction*

**This item is 80% already done and nobody has noticed.** Lines 255–257 and 694–698 already state that C is current-frame-only while A/T/S share `Z_OFFSETS = (0, 5)` (`scripts/train_e52_dagger.py:78`, routed through the shared `z_window()` at 252–254). What is missing is the one sentence that says which way the bias runs.

**Add after line 257 (§\ref{sec:interface}):**

> A, T and S share the two-sample window $[t,\,t{+}0.1\,\mathrm{s}]$ through one code path; C reads the current frame only.  The asymmetry runs \emph{against} the capability control: C receives strictly less lookahead than every representation arm and still reaches \EExplicitAmb{} against A's \ESnmrAmb{}.  The explicit result we report is therefore a lower bound on what a window-matched explicit arm would reach, and no conclusion in this paper depends on C being a tight ceiling.

**Cost:** 4 lines. 20 minutes. Do it.

---

### R3. Two small factual-precision fixes that a hostile reviewer will find

**(a) `held-out` is overloaded, line 313.** `3.66\,cm held-out MPJPE` means *held-out clips on a trained robot* (`runs/bench_g1_v2.json` `_protocol.clips` = seven LAFAN1 clips, one robot, that robot in training). Line 699 uses "unseen robot" for a different thing. Fix line 313 to `3.66\,cm MPJPE on held-out clips of a trained robot`.

**(b) The `$5.2\times$` at line 699 is a cross-checkpoint comparison.** 29.59 cm comes from `runs/phase2_loro_pm01/final_eval.json` (E06 leave-one-robot-out); 5.72 cm is PM01's row in the *different* E04 all-five shared network. The failing checkpoint no longer exists on this machine. Either cut the figure with Table II (see R7) or write: `Zero-shot decoding to an unseen robot fails (29.6\,cm, against 5.7\,cm for that robot when it is in training---a comparison across two training runs, not a within-run contrast)`.

**Cost:** 3 lines. 15 minutes.

---

## P1 — Over-claim reduction

### R4. Soften the destruction claim to a channel-reliance statement

The phrase *"every bit of competence flows through it"* does not literally appear in `main.tex` (grepped: 0 hits) — it is the advisor's paraphrase of what lines 102–104, 173–179, 259–268 and 744–745 convey together. The over-claim is real and it is a **conflation of two separable claims**:

- **Structural (provable, no experiment needed):** `snmr/integration/distillation.py:63` builds `decoder = mlp([proprio_dim + z_cmd_dim, ...])` and lines 96–97 return `decoder(cat(proprio, z_cmd))`. Decoder `in_features = 154 = 90 + 64`, pinned by `tests/test_distillation.py:72-77`. All goal, latent, clock and motion-ID information reaches the action **only** via `z_cmd`.
- **Behavioral (what the experiment adds):** all three interventions collapse completion to 0.000, so the trained decoder has **no proprioception-only fallback**.

And the critical correction: `destroy_command_code` (`distillation.py:312-326`) takes only the `(envs, dim)` code. **Raw proprioception still reaches the decoder.** "Blinding the policy to its own body" would be wrong.

**Exact replacement — §\ref{sec:interface}, replacing lines 265–268 ("Destruction drives... outside the training distribution."):**

> Two claims must be kept apart.  Structurally, the decoder's input is exactly $[x_t\,(90),\,\zcmd\,(64)]$, so every goal, latent, clock and motion-ID signal reaches the action only through \zcmd{}; this is architecture, regression-tested, and needs no experiment.  Behaviorally, the interventions show the trained decoder has no proprioception-only fallback: destruction drives the latent arm \emph{below} the goal-blind proprioception control (\EProprioGeneral{}), so a destroyed channel is not a stand-in for a controller that never had one.  The interventions remove the encoder's learned summary of the goal and of proprioception; they do not remove proprioception, which still reaches the decoder in raw form.  All three modes place \zcmd{} outside its training manifold, so the result measures reliance on the channel and is an upper bound on dependence---it does not decompose \zcmd{} into goal content and state-coupled content.

**Abstract, lines 102–104:** `Destroying the exclusive command collapses completion to \EDestroyCompletion{}...` → `Removing the exclusive command---by zeroing, batch-shuffling, or resampling its marginals---collapses completion to \EDestroyCompletion{} in all \EDestroySeeds{} seeds and all \EDestroyModes{} modes, on \emph{both} arms: the decoder has no proprioception-only fallback.`

**Contribution bullet 2, line 179:** `separating structural exclusivity from behavioral dependence` is already correct phrasing — keep it, it is the good version of the claim. **Conclusion, line 744:** `Within-policy destruction shows both arms depend on the exclusive channel` — already correct. Keep.

**Cost:** ~6 net lines. Half a day including read-through.

---

### R5. Soften the power-analysis language to pilot-dependent estimates

**Finding you need before editing:** the exact strings *"13 clips cannot resolve the margin"* and *"22+ needed"* are **not in `paper/main.tex`**. They are at `docs/site/index.html:820` and `docs/site/index.html:1028`, and they belong to the **MorphoRetarget A2 qualification gate**, not Paper 1. Two edits, two different artifacts:

**(a) `docs/site/index.html:820` and `:1028` — replacement:**

> At the observed per-clip spread (per-clip ratios span 0.98--1.11), thirteen validation clips are unlikely to resolve a 5\% non-inferiority margin.  A projection from these same thirteen clips puts the required count in the low twenties, but that number is estimated from the pilot it is meant to size and is indicative only.  The clip count must be pre-committed from a power calculation fixed before the qualification run, with the variance re-estimated rather than reused; adding clips until the interval clears is selection on the outcome.

**(b) `paper/main.tex:622-625` — the analogous claim inside Paper 1.** Current: *"so formal noninferiority at a $-5$ pp margin is unresolved."* Replacement:

> so the preregistered $-5$\,pp noninferiority margin is not resolved by this design.  The dominant variance term is between training policies rather than between rollouts (per-seed completion spans 0.85--0.91), so resolution requires more training seeds, not more rollouts; any seed count we could quote is projected from these three seeds and we do not treat it as a computed requirement.

**Leave lines 707–710 alone.** *"a three-seed sign test cannot reach conventional significance under any outcome"* is a combinatorial fact, not a power estimate. It is correct and it is the paper's best sentence.

**Cost:** 4 lines in the paper, ~8 on the site. 1 hour.

---

## P2 — Register

### R6. Replace the adversarial framing throughout

The advisor's objection targets the register, not any single sentence. The replacement register is given: **"Tracking success alone does not identify which information a learned command interface supplies to the controller."** Every instance of field-wide rhetoric in `main.tex`, with drop-in rewrites:

| Line | Current | Replacement |
|---|---|---|
| 127–128 | `A clock can track.  Any evaluation that scores a learned command by tracking success alone inherits that confound.` | `A clock can track.  Tracking success alone therefore does not identify which information a learned command interface supplies to the controller.` |
| 136 | `the \emph{interface} between them is rarely the object of study, and whenever a reference clip determines its own target from a time index...` | `whenever a reference clip determines its own target from a time index---the standard evaluation protocol---tracking success alone does not identify which information a learned command interface supplies, so credit cannot be attributed to any candidate channel.` (drop `is rarely the object of study`) |
| 145–146 | `What has been missing is not another repair but a measurement.` | `These are repairs to the channel.  What they do not provide is an identification result: a statement of which information the controller actually receives and uses.` |
| 172 | `we show the ordering this induces is not merely possible but actual` | `the ordering this induces is not hypothetical---we measure it` |
| 227 | `and all of these engineer the channel rather than measure it` | `These works specify the channel by design; we are not aware of one that reports a controlled measurement of what crosses it.` |
| 239–240 | `None makes the interface itself the object of a controlled causal measurement.` | `We are not aware of a controlled causal measurement of the interface itself, and our instrument imports that methodology from elsewhere:` |

**Title (line 84) — author's call, flagged not mandated.** `A Clock Can Track: Why Motion-Tracking Success Does Not Identify What a Learned Command Carries`. The "Why" reads as a lecture. A same-length alternative in the advisor's register: `A Clock Can Track: Measuring What a Learned Command Interface Supplies to a Humanoid Controller`. Keep `A Clock Can Track` either way — it is a measured result, not an accusation, and it is the paper's identity.

**Cost:** 6 substitutions, ~0 net lines. 2 hours including a full read for missed instances.

---

## P3 — Page budget (runs in parallel from Day 1; funds everything above)

### R7. The negative-results ledger: what to keep, what to move to the reproducibility record

Advisor's rule: keep only negatives that change the reader's understanding. Applied:

**KEEP — these change the reader's understanding:**
- **Single-clip clock null** (§4.1, 376–389). It *is* the paper's premise. It retires a favorable prior claim of the authors' own.
- **§\ref{sec:defect}** (633–646). The alphabetical-concatenation leak that collapsed completion 0.93→0.00 when blanked is a generalizable warning to anyone building an exclusivity contract. This is the most transferable thing in the paper.
- **Additive null + its analytic entailment** (492–499). Short, pre-empts "why not just concatenate," and the $I(a^*;z_{\mathrm{ret}}\mid x,g)=0$ argument is elegant.
- **"Precision we do not claim"** (704–719). The seed-level $t$-interval `[0.003, 0.379]` and the 18% replication flip rate. Disclosing these *yourselves* is the paper's integrity core and the single best defense against the reviewer who finds them.
- **"What the instrument cannot see"** (721–731). Completion is not a safety metric; the 0.437 goal-free floor.
- **Static-code arms' registered one-sidedness** (534–541) — **keep but compress to two sentences.** It is the model instance of "we registered in advance that this licenses nothing."

**CUT to the reproducibility record — these change nothing and cost pages:**
- **§Upstream Retargeter Boundary, Table II and its paragraph (590–618): cut entirely.** One training seed per arm; off-domain degradation of a specialist is expected, not informative; and per the evidence audit this section is the *only* place in the manuscript that touches legacy-retargeter numbers, i.e. the only conflation surface Paper 1 has. Retain 2–3 sentences of retargeter description at 306–315 (with R3's "held-out clips on a trained robot" fix) so the reader knows the latent is nontrivial. **Reclaims ~0.4 page.**
- **"Physics: weakly exposed" (568–574): cut entirely.** A probe negative about contact decodability that bears on no interface claim. **~0.15 page.**
- **Line 193–194** (`These controls also falsified three earlier working hypotheses of ours`): cut. An unnamed claim of virtue. Either name three hypotheses or delete; delete.
- **SNMR-vs-GMR noninferiority paragraph (620–631):** compress to two sentences carrying R5's softened language, or move with Table II.
- **§Reproducibility Contract (648–663):** compress 16 lines → 8. The enumerated inventory is a record, not something a reader needs inline.
- **"Content" probe paragraph (554–559):** compress to two sentences. **Keep "Embodiment: aligned, not invariant" (561–566) in full** — the linear-0.28 / nonlinear-0.91 gap is the methodological premise the entire instrument rests on.

**Net budget: ≈0.55–0.7 page reclaimed against ≈0.25 page added by R1–R6.** Comfortable margin. That margin is what lets you ship on Day 4 instead of fighting overfull boxes on Day 5.

---

## P4 — Optional, only if Day 3 ends clean

### R8. E77 dose-response — my recommendation is **do not promote it for this submission**

The advisor's case is strong on the science: `/data/robotixx/snmr-research/e77_degradation_pilot/` shows a zero-order hold of `z_cmd` for 0.1 s drops the explicit student 0.926 → 0.223 and the SNMR student 0.698 → 0.004. That is direct evidence the channel carries fast state-coupled content, and it reframes the 0.000 knockout as the endpoint of a continuous curve rather than a binary.

**But** `docs/E77_DEGRADATION_PILOT.md` states plainly: *"descriptive pilot. No gate, no preregistration, no replication. Seed 0, one run per cell."* Inserting a single-seed unregistered result into a paper whose entire contribution is preregistration discipline hands a reviewer the exact inconsistency they need. It also costs a renderer, a fragment, a figure, and ~0.3 page — on Day 4.

**If you do it anyway**, the conditions are non-negotiable: a `render_e77_*.py` fragment inside the freeze, explicit in-text labeling as a single-seed descriptive pilot with no registered gate, and it comes out of the §audit probe budget, not out of R7's margin.

---

## FIVE-DAY SCHEDULE

| Day | Work | Exit condition |
|---|---|---|
| **Sep 10 (today)** | R1 + R2 + R3. Add the phrase-guard test. Start R7 cuts (Table II, physics probe, line 193–194). | Paper builds; `pytest` green |
| **Sep 11** | R4 + R6. Finish R7 compressions. | ≤8 pages, 0 overfull |
| **Sep 12** | R5 (paper + site). Full rebuild; rerun `render_e70_*.py` chain, `check_e70_video_code_hashes.py`, `audit_e70_final_bundle.py` (must re-certify `positive_content`), full CPU suite (433 passed baseline). | Bundle re-certified |
| **Sep 13** | Cold read-through by a non-author for seams left by the cuts. Anonymity sweep on the PDF. PaperPlaza metadata: keywords `Simulation and Animation`, `Humanoid Robots`, `Learning and Adaptive Systems`. **Verify the video upload window** — `docs/EXECUTION_PLAN_REPORTS.md:62` records windows Aug 5–Sep 9 and Sep 17–22, so the pre-deadline window may already be closed. Video is optional; nothing breaks, but confirm rather than discover. | Metadata staged |
| **Sep 14** | Upload. | Submitted |
| **Sep 15** | Reserve only. Not the plan. | — |

---

# (B) PAPER 2 — MorphoRetarget / transferable retargeting corrections

**Status: NOT submittable. `autoresearch/iterate-260901-1222/a2_prereg.json` has `execution_gate.current_state = "blocked_awaiting_a1_qualification"`, `a1_qualification_report = null`, and every held-out-variant cell reads "not run." There is no held-out-robot result of any kind.** This outline is a target, and every result slot below is an empty bracket. Do not fill one with an imagined number.

**Working title:** `[TITLE — hold until the held-out-robot result exists; the title must name the transfer, not the architecture]`

### Abstract (skeleton)
> Retargeting corrections that improve robot execution are today learned per robot: [BASELINE COST — ReActor ≈6 GPU-h bilevel RL per embodiment; G-DReaM ≈5 GPU-h retraining; OmniTrack requires that robot's own privileged policy]. We ask whether such corrections transfer. We train a correction model conditioned on robot specification rather than robot identity on N−1 humanoids, freeze it, and apply it to a held-out robot with **zero new parameters and zero gradient steps**. On [HELD-OUT ROBOT], corrections reduce [PRIMARY KINEMATIC ERROR: not measured] and change downstream tracker [SAMPLE EFFICIENCY: not measured] / [FINAL SUCCESS RATE: not measured], while human-side semantic preservation stays within [INTENT BAND: not measured] on metrics held out of the training objective. Correction quality scales with the number of training robots: [SCALING SLOPE: not measured].

### §1 — The application problem
Human motion must become a reference trajectory a specific robot can actually execute. Optimizer-based retargeters produce geometrically faithful references that are dynamically wrong for that robot; the resulting artifacts cap what any tracker can achieve (`gmr`, arXiv:2510.02252). Today, fixing this for a new robot is a per-robot engineering cost.

### §2 — The missing learning signal
**This is the section that earns the paper.** Neural retargeters are trained by copying an optimizer's output. That objective teaches **geometric imitation** — reproduce what the IK solver would have produced — and contains no information about **which corrections improve execution**. A network trained this way inherits the teacher's dynamic infeasibility exactly, and it inherits it *perfectly*, because perfect imitation is the objective.

Anchor this with the repo's own frozen finding, which is on disk today and costs zero GPU: on the 13 validation clips, the **GMR teacher fails its own student's fidelity gate** — 35 motion-fidelity violations across 11/13 clips, max per-anchor amplitude ratio 1.598, against a student's 6 violations across 4/13 and max 1.510 (recomputed from `autoresearch/iterate-260901-1222/a1_screen_seed0_20k_root_prior_trajectory_v1/evaluation_endpoint/report.json`; all ten cells reproduce exactly). The supervision target is not a ceiling.

Then cite the literature stating the gap rather than declaring it yourself: NMR (arXiv:2603.22201) writes in its own limitations that *"CEPR is morphology-specific; extending to other platforms requires regenerating the data pipeline, and developing morphology-conditioned architectures remains a direction for future work."*

### §3 — The transfer hypothesis
If corrections are a function of **robot specification** (link lengths, joint axes, joint limits, kinematic topology, end-effector designation) rather than **robot identity**, then a correction model trained on several robots should correct a robot it has never seen, with no new parameters.

State the architectural argument, not a benchmark argument, because it is the one thing the competition structurally cannot say:
- **AdaMorph** (arXiv:2601.07284) learns a per-robot prompt bank *and* a per-robot output MLP head → an unseen robot has neither; its "zero-shot" means unseen *motions*, all 12 robots in training.
- **G-DReaM** (arXiv:2505.20857) encodes real morphology but "adapts" by ~30k-step / ~5 GPU-h retraining with the new skeleton added to the training set.
- **ReActor** (arXiv:2605.06593) re-runs ~6 h of bilevel RL per embodiment and learns nothing reusable.
- **X-Morph** (arXiv:2606.30290) already holds spec conditioning + physics-aware correction + closed-loop tracking, trains per source–target pair, and explicitly defers the controlled sample-efficiency comparison. **Cite it prominently in related work; a reviewer who finds it first reads this paper as taking the easy half of someone else's future work.**

**Falsifiable claim:** an unseen robot requires zero new parameters. Verify this is literally true of the implementation before writing the sentence.

### §4 — Architecture
`[PARAM COUNT — currently 1,583,754, an architectural fact with frozen provenance across five preregistration files, never a performance fact]`. Spec encoder over the URDF-derived graph → shared correction decoder over a canonicalized action space → no per-robot head, no per-robot embedding table.

**State the current scope honestly and in the architecture section, not buried in limitations:** the conditioning is **kinematic only**. `snmr/robot_tokens.py:249-252` zeroes `dynamics_available` whenever `feature_set != "full"`; `snmr/morpho_integration.py:142-148` is a fail-closed guard that *raises* if a batch carries nonzero dynamics availability. Mass, inertia, torque limits, armature, damping, kp, kd cannot reach the trained network. Also state that semantic manifests are currently manual and G1-only (`snmr/robot_spec.py:179-184`: *"The bounded manual annotation allowed for a new robot"*), and what a new robot actually costs: seven link-role assignments, seven hand-measured metric anchor points in link frames, a mapping-version string, and any asset-specific merged-body workaround.

### §5 — Protocol
- **Leave-one-robot-out as the primary protocol, not an ablation.** Train on N−1, freeze, apply to the held-out robot with no gradient step. Report the **full LORO table** over every robot (following H-Zero, arXiv:2512.00971) — with a small robot set, one favorable held-out choice reads as cherry-picking.
- **Compute-matched ladder on the held-out robot.** (a) optimizer-only floor; (b) ReActor bilevel at its stated ~6 GPU-h; (c) G-DReaM-style fine-tune at its stated ~5 GPU-h; (d) per-robot supervised fit as the fit-it-directly upper bound. **The claim that must survive is ≥ (b) and (c) at strictly lower compute on the new robot. Amortization is the result, not accuracy.**
- **Downstream endpoint measured with an independent tracker**, BeyondMimic-style, quoting GMR's rationale that it "does not depend on reward tuning and is developed independently from the retargeting methods" — this pre-empts the single most likely attack, that gains came from reward engineering.
- **Intent metrics held OUT of the training objective**, following GenTrack's explicit discipline. Map "semantic loss" onto `snmr/semantic_metrics.py` per-anchor amplitude / energy / jitter ratios, and **watch the low side of the amplitude band**: `< 0.50` is the advisor's stated kill criterion (repair degenerating into conservative standing). Note that every violation observed so far is high-side amplification, so `snmr/morpho_a1_evaluation.py:157-158` must first be split into `model_amplitude_collapse` (< min) and `model_amplitude_amplification` (> max) — the current property labels a 1.510 amplification as "collapse" and that error has already propagated into a governing status document.
- **Vocabulary discipline, fixed now:** "zero-shot **retargeting** (verified kinematically)" vs "zero-shot **control** (no protocol exists)". If T1 is the held-out robot and a T1-trained tracker is the verifier, the retargeter is zero-shot and the controller is not — a legitimate design that must appear *in the same sentence as the result*.

### §6 — Results (every slot empty)
- **R1 Held-out-robot correction quality.** `[LORO TABLE: per-robot primary kinematic error, correction vs no-correction vs each compute-matched baseline — NOT RUN]`
- **R2 Downstream tracker benefit on the held-out robot.** `[SAMPLE-EFFICIENCY CURVE: episode length / success rate vs environment steps — NOT RUN]`; `[FINAL SR / MPJPE — NOT RUN]`
- **R3 Intent preservation.** `[RETRIEVAL R@k, FID, per-anchor amplitude/energy/jitter ratios, explicitly held out of the objective — NOT RUN]`
- **R4 Scaling in number of training robots.** `[SLOPE of held-out correction quality and downstream sample efficiency vs N — NOT RUN]`. **This is the one result no per-robot method can produce, and it is what upgrades the paper from a conjunction to a phenomenon. If the slope is flat, the transfer claim is not real — learn that before writing anything else.**
- **R5 Negative control: shuffled / mismatched spec at test time.** `[DEGRADATION UNDER MISMATCHED URDF — NOT RUN]`. If held-out performance does not degrade, the conditioning is inert and the gains are just a better average retargeter — which is exactly what a hostile reviewer will allege. **Run this before committing to the framing.**

### §7 — Limitations
Kinematic conditioning only (dynamics structurally excluded, with the fail-closed guard as evidence of the boundary rather than an excuse). Manual semantic manifest per robot, `[CALIBRATION COST: currently sourced nowhere — measure it or drop the estimate]`. `[NUMBER OF TRAINING ROBOTS]` is small. No hardware.

---

### MAIN-FIGURE SPECIFICATION (Figure 1)

**One human motion. One held-out robot. Three columns, left to right, sharing a time axis.**

**(a) Source and proposal.** Top strip: the human motion, `[CLIP ID]`, 5–7 poses. Below it, the **initial proposal** on the held-out robot `[ROBOT ID — held out of training, spec read from URDF only]`, same timestamps, rendered in muted grey. Caption states plainly: *this robot contributed no training data and the model has no parameters specific to it.*

**(b) The corrections.** The paper's centerpiece, and it must show the **correction**, not the corrected pose. Two stacked panels:
- A per-joint correction heatmap, joints × time, signed Δq, diverging colormap centered at zero, with the `[MAX |Δq|: not measured]` and `[FRACTION OF FRAMES TOUCHED: not measured]` annotated. A reader must be able to see that corrections are **sparse and localized**, not a global rewrite.
- The same motion overlaid in 3D at 3 key frames: proposal in grey, corrected in color, with arrows on the joints/links that moved. `[Δ FK displacement at the seven semantic anchors: not measured]`.
- An inset bar: human-side semantic ratios (amplitude / energy / jitter) for proposal vs corrected, with the two-sided `[0.50, 1.50]` band drawn. **This is the panel that proves the correction did not simply shrink the motion into conservative standing** — the reader must be able to check the kill criterion by eye.

**(c) The execution difference.** Rollouts of a **fixed, independently trained tracker** on the uncorrected reference and on the corrected reference, same seed, same start grid. Filmstrip of both, with the failure frame marked on the uncorrected row. Beside it: `[COMPLETION: uncorrected __ vs corrected __ — NOT MEASURED]`, `[SURVIVAL s: __ vs __ — NOT MEASURED]`, `[TRACKING ERROR: __ vs __ — NOT MEASURED]`, with paired CIs clustered by clip.

**Caption must carry three sentences:** the robot was held out; the tracker is independent of the correction model and was not retuned; the semantic ratios are held out of the training objective.

### Before this paper exists (ordered, gating)
1. Write the **tracker qualification floor** — a number, a reference set, a horizon, a rollout count. The only mention in the repository defines no value, and a completed checkpoint (`model_39999.pt`, 2026-09-02 11:20) is already sitting unqualified. **Every day of delay raises the risk of an outcome-conditioned floor, which this project's own governance treats as disqualifying.** Zero GPU.
2. Commit the working tree. All 51 frozen iteration-2 artifacts record `dirty: true` and none records `false`, so no provenance hash in this partition is verifiable from a clean checkout. Minutes of work, unblocks an entire partition.
3. Fix the inverted `model_amplitude_collapse` gate label.
4. R5 (mismatched-spec negative control) — before any framing is committed.
5. R4 (scaling in N) — before any writing is committed.

---

# (C) SUBMIT PAPER 1 ON 2026-09-15?

## **Yes. Submit. Treat September 14 as the deadline.**

**The decisive fact: not one item on the revision list requires an experiment.** The leakage audit's own conclusion is *"No re-runs are required by this audit."* Every displayed quantity is emitted by a hash-stamped renderer from frozen artifacts, so prose edits cannot silently corrupt a number, and the registered gates (`\EExplicitGate{} = PASS`, `\EPositiveGate{} = PASS`) cannot flip from editing. The manuscript is 8 pages, builds with 0 overfull boxes, passes 433 tests, has a bundle auditor, and has a completed anonymity sweep. The submission machinery exists and has been dry-run.

**The primary claim survives the audit intact.** A−T = +0.191 [0.124, 0.274], against a null that is verified identity-free by construction and by regression test. All five control-matching concerns were audited against the code and none threatens it: per-arm standardization is a capacity-neutral matching step (and the 163× raw energy gap means *not* doing it would have been the defect), the observation normalizers are literally the same frozen object across arms, evaluation resets are exact-state and seed-404-matched with a realized-grid assertion, no clip identity reaches the actor, and in E70 the privileged critic observations never reach the student at all because `E52_DET=1` disables the posterior.

**The one real error is confined to a secondary contrast and to wording — and fixing it makes the control harder, not weaker.** S was mislabeled. Correctly labeled, it is a *phase-matched misaligned-reference* arm carrying between-clip identity separation of 8.042 SD — **identical to the winning arm** — that scored 0.5524, indistinguishable from the identity-free clock's 0.5622. The honest version of the sentence is stronger than the dishonest one: identity was maximally available and bought nothing.

### Risk of submitting

1. **The corrected S language invites "then what bounds an identity-only command?"** Honest answer: T bounds it, and a directly supervised clip-ID+phase arm is unrun and needs a new *training* arm — `scripts/eval_e71_command_swap.py` cannot supply it (frozen-student evaluator, arms hard-restricted at line 1163, six checkpoint SHAs pinned). This is a rebuttable weakness because the registered primary contrast is A−T, not A−S. **Rebuttal-ready, not fatal.**
2. **The seed-level $t$-interval lower bound is 0.003, below the paper's own registered 0.10 threshold.** A hostile reviewer can lead with this. Mitigation: it is disclosed by the authors in "Precision we do not claim," not discovered by the reviewer. Disclosure is the entire defense and it is already in the manuscript. Do not remove it under deadline pressure.
3. **R7's cuts could leave seams.** Removing Table II mid-section on Day 2 and not cold-reading until Day 4 is how a rushed paper looks rushed. This is why Day 4 is a read-through day and Day 5 is reserve.
4. **Operational:** verify the PaperPlaza video upload window. `docs/EXECUTION_PLAN_REPORTS.md:62` records Aug 5–Sep 9 and Sep 17–22 — the pre-deadline window may be closed as of today. Video is optional so nothing breaks, but find out rather than discover.

### Risk of not submitting

1. **Six months of shelf time** (IROS 2027 ≈ March; RA-L is rolling but slower to visibility) on a contribution that is *definitional, not empirical*. The instrument — exclusivity contract, matched-present/divergent-future screen, identity-free clock null — is reproducible in a weekend by anyone who reads the idea. The novelty audit shows the surrounding field published four relevant systems in 2026 alone (X-Morph, GenTrack, NMR, H-Zero). A better-resourced group adding one control arm to their existing system takes this result.
2. **This project's demonstrated failure mode is that a held paper acquires experiments instead of getting submitted.** The revision list is finite and closed today. In four weeks it will include E71, the clip-ID+phase arm, and E77 promotion, and the paper will be 12 pages.
3. **Holding Paper 1 does not help Paper 2.** Paper 2's blocker is a held-out-robot result that does not exist and cannot exist in five days. The GPU is contended. Shipping Paper 1 on Sep 14 is what frees the team to write the tracker qualification floor and run the mismatched-spec control — the two zero-GPU items that actually gate Paper 2.

### The hard stop condition on that yes

**If, at end of day Sep 12, the rebuilt PDF is not ≤8 pages with 0 overfull boxes and `audit_e70_final_bundle.py` does not re-certify, stop editing content and ship what builds.** Only two edits are non-negotiable: **R1** (the shuffle relabel — shipping a factually wrong control description is the one thing that would justify withdrawal later) and **R2** (the four-line lookahead direction sentence). R3–R7 are improvements. A correct 8-page paper submitted on time beats a better 8-page paper that misses a deadline the CFP says will not be extended.

### And one thing not to do

**Do not merge any MorphoRetarget material into Paper 1.** `paper/main.tex` currently contains zero occurrences of "MorphoRetarget" or "RobotSpec" — it is the cleanest artifact in the repository on the conflation axis, and R3 + R7 close its last exposure (the legacy `671 fps` / `3.66 cm` figures and the overloaded word "held-out"). The conflation problem lives entirely in `docs/site/benchmark.html`, which has zero occurrences of "legacy," zero of "MorphoRetarget," and presents legacy numbers under the heading *"No per-robot IK configs, no per-robot tuning"* — verbatim Paper 2's value proposition applied to five in-training robots. That page is a separate, higher-priority fix than anything in this strategy, and it is 20 minutes of work.