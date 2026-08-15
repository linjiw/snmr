# Overall judgment

I read the full seven-page draft, checked the current project record, and compared the positioning against the closest 2026 humanoid-interface and representation-diagnostics work.

My honest judgment is that **this is already a scientifically serious paper**. The strongest asset is not merely the positive (+0.191) result. It is that the experimental instrument repeatedly ruled against your preferred hypotheses:

* the latent looked useful on one walk, but the clock performed better;
* latent-plus-explicit looked promising, but the three-seed result became null;
* the first “exclusive” implementation contained an observation-layout leak, and fixing it changed the result;
* only after those failures did the paired ambiguity assay return a positive result.

That sequence gives the work unusual credibility. The acceptance risk is elsewhere: **a skeptical reviewer can currently summarize the result as “the latent tells the student which of two memorized walks—or which specialist teacher—it should imitate.”** Your draft acknowledges much of this limitation, but it does not yet experimentally separate trajectory identity from time-varying trajectory content. 

So I would not broaden the paper or chase a larger capability score. I would make the causal interpretation much tighter and recast the paper as a **general evidence standard for learned humanoid command interfaces**.

My preferred one-sentence thesis is:

> **A learned command interface is credible only when it survives a ladder of tests: exclusive routing, causal necessity, target-specific counterfactual swaps, shortcut-matched nulls, and held-out generalization.**

That is broader and more memorable than “SNMR beats a clock on two walks,” while remaining fully grounded in the experiments you have built.

---

# What the paper has actually discovered

## 1. The single-trajectory failure is the conceptual centerpiece

The draft reports that on one deterministic walk, the frozen retargeting latent reaches (0.656\pm0.044), while absolute time reaches (0.754\pm0.010). That is not merely a negative baseline. It reveals an **identifiability failure in ordinary motion-tracking evaluation**: for a deterministic trajectory (g_{0:T}), time itself is an index into the entire target sequence. A sufficiently flexible controller receiving ((x_t,t)) can learn the same lookup that another controller receives through an apparently meaningful motion code. 

This should be elevated from “Diagnostic 1” into a proposition near the beginning:

> **Single-clip non-identifiability.** On a deterministic reference trajectory, successful tracking cannot distinguish motion-specific command content from progress through the trajectory.

The deeper significance is that this is not only an SNMR issue. It applies to skill latents, demonstration-conditioned controllers, scripted VLA evaluation episodes, and learned motion tokens whenever the target is uniquely determined by episode progress. That broader benchmark-design lesson is what makes the paper interesting beyond your particular retargeter.

## 2. SNMR carries meaningful but incomplete operational command content

On the ambiguity starts, the strongest null is time at (0.562), the latent reaches (0.754), and the explicit ceiling reaches (0.973). The raw content-over-clock margin is therefore (0.192). Another useful way to communicate the magnitude is to normalize it by the available gap between the strongest null and the explicit ceiling:

[
\mathrm{Normalized\ Content\ Recovery}
======================================

\frac{0.754-0.562}{0.973-0.562}
\approx 0.47.
]

Descriptively, the SNMR source recovers about **47% of the ambiguity-completion gap** remaining above the clock null. Using survival time gives about **54% of the available survival gap**. These are not information-theoretic fractions and should not be presented as an additive causal decomposition, because the arms are separately trained. But they are much more intuitive effect sizes than a standalone (+0.191). 

This naturally gives you two benchmark quantities:

[
\mathrm{CNM}(U)
===============

Y_U-\max_{N\in\mathcal N}Y_N,
]

where (\mathcal N) is the matched null battery, and

[
\mathrm{NCR}(U)
===============

\frac{Y_U-Y_{N^\star}}
{Y_C-Y_{N^\star}},
]

where (C) is the explicit ceiling and (N^\star) is the strongest null. CNM says whether content clears the null; NCR says how much of the available control gap it closes.

## 3. Representation usefulness is relational, not intrinsic

The same latent is:

* null beside the explicit target;
* useful when made exclusive;
* not strongly semantic under a linear category probe;
* strongly instance-aligned;
* weak on contact physics unless contact is directly supervised.

That is a coherent finding rather than an awkward collection of results:

> **The latent is not generally “useful” or “useless.” Its behavioral value depends on what competing command channels remain available and what the downstream learner can extract from it.**

This is reinforced by UniTracker’s ablation: when its actor directly receives the strong explicit reference, the influence of its latent largely vanishes. So your additive null is not an embarrassment and should not be “repaired” by another fusion architecture. It is evidence that a dominant explicit route makes latent attribution impossible. ([arXiv][1])

Your probe results suggest a narrower interpretation: SNMR appears to preserve **trajectory instance and pose evolution**, rather than high-level motion category or rich contact semantics. The control assay then shows that this instance-level structure can become actionable when the decoder has no stronger route to the target. 

That is already an interesting scientific statement:

> A motion representation does not need to be linearly semantic or strongly physics-aware to support control; instance-aligned trajectory state may be enough to disambiguate behavior.

## 4. The measurement-substrate defect is part of the methodology

The observation-ordering leak is not merely an implementation anecdote. It proves that there are at least three distinct claims:

1. the architecture appears to have an exclusive channel;
2. the tensor layout actually enforces that exclusivity;
3. the trained controller behaviorally depends on that channel.

The name-derived layout tests establish the second; destruction establishes the third. That distinction deserves to become part of the general evidence ladder. 

---

# The novelty is real—but it must be claimed in the right place

The current humanoid literature already treats command-space design as important. HANDOFF builds a compact explicit 10-D planner-facing interface; ULTRA distills motor behavior into a compact latent and deploys on a real G1; UniTracker and BFM use CVAE-based latent interfaces and online distillation; AnyBody uses a deterministic encoder-decoder latent and treats its frozen decoder as a motor prior. Therefore neither “learned latent interface,” “online distillation,” nor “frozen decoder” is by itself your novelty. ([arXiv][2])

Your novelty is the combination of:

* a deterministic-clip identifiability diagnosis;
* exclusive goal routing;
* matched-present/divergent-future evaluation;
* matched shortcut controls;
* fail-closed capability gates;
* behavioral interventions rather than latent visualization alone.

Causal representation diagnostics are also beginning to appear in adjacent robotics areas—for example, VLA-Trace combines representation tracing, knockout interventions, and rollout-level behavior. Therefore I would avoid a broad claim such as “the first controlled measurement of a learned robot representation.” The defensible claim is narrower and stronger:

> **To our knowledge, this is the first counterfactual assay of control-usable information specifically at the humanoid retarget-to-track command interface.**

([arXiv][3])

---

# The central weakness a reviewer will attack

The draft itself states the problem accurately: the shuffled control destroys both clip identity and future correctness, so the positive result does not distinguish a static trajectory identifier from finer time-varying future information. 

At present:

* (T) knows time but not which walk;
* (A) can easily encode which walk;
* (S) receives the wrong walk’s latent;
* the teacher is a routed pair of specialists;
* the two evaluation clips are also student-training trajectories.

A reviewer can therefore say:

> “The reported (A-T) difference may only show that the SNMR latent supplies one bit of clip or teacher identity. Given that both trajectories are memorized, this does not establish meaningful motion representation.”

That criticism would not invalidate your current carefully worded conclusion—“clip-disambiguating trajectory information beyond absolute time” is accurate—but it would cap the significance.

There are two additional interpretation limits:

First, the paired states are similar, not literally identical. Proprioception can therefore carry some side information. The proprio-only arm reaching (0.437), rather than approximately zero, shows that start state and closed-loop dynamics still provide useful structure.

Second, (A), (T), and (S) are independently trained encoder-decoder students. Your operational estimand is honest about that: it measures the policies produced by a fixed training algorithm, and the draft explicitly says that only within-policy command destruction is a causal intervention on a fixed controller. 

So “causal interface instrument” is slightly too broad today. “Exclusive interventional interface assay” is safer until you add target-specific interventions on the trained (A) controller.

---

# The missing evidence ladder

This table captures what the paper currently establishes and what remains vulnerable:

| Evidence rung                | Scientific question                                                                  | Current status                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Structural exclusivity       | Can the goal enter the decoder outside (z_{\mathrm{cmd}})?                           | **Pass:** name-derived layout contract and repaired leak             |
| Behavioral necessity         | Does the controller depend on the exclusive channel?                                 | **Pass**, with destruction; the PDF needs the newest all-seed result |
| Target specificity           | Does changing one valid command to another select a different intended future?       | **Missing on the trained SNMR controller**                           |
| Beyond absolute time         | Does the source outperform an equally trained clock channel?                         | **Pass on two known walks**                                          |
| Beyond static identity/phase | Is the useful content more than clip identity plus progress?                         | **Not separated**                                                    |
| Held-out generality          | Does the conclusion survive unseen motions, other representations, or another robot? | **Not established**                                                  |

This evidence ladder itself can become a central conceptual contribution. It also makes every limitation constructive: the case study passes some rungs and fails or leaves open others.

---

# The highest-value experiment: same-state counterfactual command swaps

Before training more students, I would run an **evaluation-only, on-manifold command-swap experiment on the frozen (A) policy**.

For every ambiguity pair (i), you already have two matched states and two valid source commands:

[
(x_i^a,u_i^a),\qquad (x_i^b,u_i^b).
]

Evaluate the same trained controller in all four cells:

[
(x_i^a,u_i^a),\quad
(x_i^a,u_i^b),\quad
(x_i^b,u_i^a),\quad
(x_i^b,u_i^b).
]

The important cells are the crossed ones. Starting from the state associated with walk (a), feed the valid latent command from walk (b), and vice versa. Nothing in the policy weights changes. The command remains on-manifold because it comes from a real trajectory.

This separates **what the command asks for** from **which state the robot began in**.

The primary metric should not be only tube completion. Measure whether the rollout becomes closer to the commanded future than to the future originally associated with the start state:

[
M_i =
d(\hat{\tau}*i,\tau*{\mathrm{other}})
-------------------------------------

d(\hat{\tau}*i,\tau*{\mathrm{commanded}}).
]

Positive (M_i) means the command, rather than the initial-state identity, selected the future. Report this over the first 0.5–1.0 seconds and over the full rollout.

A second, sharper metric can operate at the first action:

[
A_i =
\lVert a_{\mathrm{student}}-a^\star_{\mathrm{other}}\rVert
----------------------------------------------------------

\lVert a_{\mathrm{student}}-a^\star_{\mathrm{commanded}}\rVert.
]

If swapping the source command shifts the action toward the corresponding explicit-teacher action, you have a direct causal bridge:

[
\text{valid command}
\rightarrow
\text{different action}
\rightarrow
\text{different future behavior}.
]

This is qualitatively stronger than destroying the channel. Destruction establishes **necessity**; a valid command swap establishes **target specificity**.

The explicit controller should be run through the same four-cell matrix as a feasibility gate. Only pairs where the explicit controller can follow both commands from both starts should enter the confirmatory analysis. That prevents a failed cross-start rollout from being misread as a representation failure.

This experiment would also produce the best video and main figure in the paper: the same initial humanoid state, one green command leading toward future (a), one orange command leading toward future (b).

---

# Then decompose identity, phase, and local motion

The next battery should operate largely within the same frozen (A) controller:

### Correct clip, wrong phase

Feed a latent from the correct trajectory but shifted by (\delta):

[
z^a_{t+\delta},
\qquad
\delta\in{-0.5,-0.25,+0.25,+0.5}\text{ s}.
]

If performance degrades smoothly with phase displacement, the controller uses time-varying trajectory state rather than only a static clip label.

### Static clip code

Repeat the clip mean, first-frame latent, or another constant per-clip code throughout the rollout.

If this retains most of the (A) performance, the current positive result is largely trajectory identity. That is still a useful finding, but the paper must say so.

### Current-only versus two-sample latent

The deployed SNMR source uses ([z_t,z_{t+0.1}]). Evaluate or retrain:

[
z_t,\qquad
[z_t,z_t],\qquad
[z_t,z_{t+0.1}],\qquad
z_{t+0.1}-z_t.
]

This answers whether the command relies on current pose identity, a local velocity cue, or both. The draft is careful that the explicit arm is current-frame only and that an explicit future window remains future work; this decomposition would prevent reviewers from reading “future content” into a result that may be driven by current trajectory identity. 

### Clip-ID-only and time-plus-ID students

These are important diagnostic controls, but they require careful interpretation.

* **ID only** asks whether static side identity plus proprioception is enough.
* **Time + ID** is a memorization oracle: because there are only two deterministic known trajectories, clip identity plus time uniquely indexes the full target.

Therefore, if time+ID matches (A), the latent behaves operationally like a learned trajectory lookup key. If (A) outperforms time+ID under the frozen trainer, the correct claim is not that (A) contains more raw information—time+ID is already sufficient—but that SNMR provides a more **accessible inductive representation** for the bounded learner.

This distinction between raw sufficiency and algorithm-relative usability fits the operational estimand already present in the draft.

---

# The stronger full approach: a calibrated frozen-decoder assay

The most methodologically complete version of this research would use a common action decoder.

## Step 1: Train the explicit interface and freeze its decoder

Train the capable explicit student (C), then freeze:

[
a = D_C(x,z).
]

## Step 2: Train only source-specific adapters

For every upstream source (U), train:

[
z=E_U(x,u),\qquad
a=D_C(x,z),
]

with the same frozen (D_C).

The candidate sources would include:

* explicit reference;
* PCA-(k) or other known-content compressions;
* SNMR;
* time;
* ID;
* time+ID;
* shuffled or phase-shifted SNMR;
* a second external latent.

Now the downstream action map and command geometry are held fixed. Differences are much harder to attribute to independently learned decoder competence.

AnyBody already demonstrates the practical viability of treating a frozen decoder as a motor prior, so the frozen-decoder architecture itself should not be claimed as novel. Your novelty would be using a fixed decoder to build a **calibrated measurement instrument**. ([arXiv][2])

## Step 3: Validate the instrument with known-content channels

Your E73 idea is exactly the right direction: train adapters from nested PCA-(k) compressions of the explicit reference and ask whether assay performance rises with (k). The command code in the current draft has effective rank 14 despite being 64-dimensional, so a ladder around (k\in{4,8,14,16,32,64}) could connect representation geometry to behavioral sufficiency. 

I would supplement PCA with an interpretable nested ladder:

[
\text{no goal}
\rightarrow
\text{time}
\rightarrow
\text{root/velocity}
\rightarrow
\text{joint position}
\rightarrow
\text{position+velocity}
\rightarrow
\text{full explicit}.
]

PCA tests dimensional capacity; the physical ladder tells the reader what information becomes actionable.

Your public roadmap already contains the right three ingredients—noise sensitivity, known-content calibration, and a second external subject. I would place target-specific command swaps before the noise titration, because a smooth noise curve validates sensitivity but does not establish that the channel controls the intended future. E73 is the stronger instrument-validation experiment, and E74 is the stronger generality experiment. ([Linji (Joey) Wang][4])

---

# What should happen before this submission

The practical priority order should be:

| Priority | Change                                                                             |                                 Scientific value |      Expected burden |
| -------- | ---------------------------------------------------------------------------------- | -----------------------------------------------: | -------------------: |
| 1        | Put already-completed temporal-block and all-seed destruction results into the PDF |      Closes an existing paper/site inconsistency |             Very low |
| 2        | Same-policy, same-state valid-command swaps                                        |           Establishes target-specific causal use | Low; evaluation only |
| 3        | Static-code and phase-shift interventions                                          | Separates identity from dynamic trajectory state | Low; evaluation only |
| 4        | ID-only and time+ID controls                                                       |    Exposes the simplest memorization explanation |             Moderate |
| 5        | Rewrite around identifiability and the evidence ladder                             |                   Makes the contribution legible |                  Low |
| 6        | Known-content calibration or fixed-decoder replication                             |             Validates the assay as an instrument |             Moderate |
| 7        | Multiple clip pairs, held-out motions, or external latent                          |               Converts case study into benchmark |                 High |

There is currently a free strength missing from the PDF. The draft says command destruction is shown only for the seed-0 explicit student, and the limitations section says temporal-block or nonoverlapping analysis is still needed.  

The current project page, however, reports:

* zero, shuffle, and marginal-random destruction collapsing all three explicit seeds to zero;
* a 12-block temporal bootstrap with 10,000 replicates.

Those results should be incorporated before anything more expensive is run. ([Linji (Joey) Wang][4])

The sim2sim ONNX/DDS/CPU-MuJoCo result is useful engineering evidence, but it should stay as one sentence, a video, or supplementary validation. It demonstrates that the interface survives the production runtime hop; it should not displace the causal controls or be framed as sim-to-real. ([Linji (Joey) Wang][4])

---

# The story should change

## What the introduction currently promises

The introduction motivates learned interfaces by saying explicit robot targets do not expose contact reasoning, uncertainty, or cross-embodiment structure. But the results later show:

* contact is weak without direct supervision;
* zero-shot decoding to an unseen robot fails;
* no additive value beside explicit targets;
* no hardware or sim-to-real result.

That creates an unnecessary expectation mismatch.   

The paper is not actually demonstrating that a learned latent solves the limitations of explicit references. It is demonstrating that **we currently lack a valid way to tell what a learned command latent contributes**.

That should be the opening problem.

## A better narrative arc

### Act I: the attribution failure

A learned latent appears useful, but a clock performs better. Therefore ordinary tracking performance cannot establish command content.

### Act II: the measurement contract

Make the command exclusive, create matched states with divergent futures, install matched nulls, and gate feasibility before interpreting representation differences.

### Act III: the surviving result

SNMR beats clock and wrong-trajectory controls, but not the explicit ceiling. It carries partial, clip-disambiguating trajectory state.

### Act IV: what the result does not mean

The probes and intervention battery determine whether that content is static identity, phase, local motion, semantics, or physics. Held-out motion remains a higher evidence rung.

This arc turns every negative result into part of the logic rather than a collection of caveats.

---

# Paper structure I recommend

1. **Introduction: Why tracking success cannot identify command content**
2. **An Evidence Ladder for Learned Command Interfaces**
3. **The Exclusive Counterfactual Interface Assay**
4. **Instrument Validation and Matched Nulls**
5. **Case Study: What the SNMR Interface Carries**
6. **Validity Audits, Scope, and Reproducibility**
7. **Conclusion**

The current paper is carrying two partially developed papers:

* an SNMR retargeter paper;
* an interface-measurement paper.

The interface-measurement paper is substantially more distinctive. Table II’s two-teacher absorption experiment uses one seed per arm, and the SNMR-versus-IK tracking comparison does not resolve its formal noninferiority margin. Those results are respectable evidence that SNMR is a nontrivial source, but they are not strong enough to serve as a second headline contribution. Move most of that section to the supplement. 

Similarly:

* move the online-distillation implementation figure to the supplement;
* retain only a compact description of the trainer;
* compress the defect section into a “validity audit” box;
* reduce the probe section to one small table or figure;
* use the recovered space for the counterfactual command-swap result.

---

# Figure strategy

The current Fig. 1 is clean, but it shows the architecture and the bar chart without visually showing the paper’s most important construction: **the present is similar while the futures diverge**. 

The new main figure should contain:

1. two humanoid poses overlaid at the matched present;
2. their two colored future trajectories at (+0.5) and (+1.0) seconds;
3. the exclusive interface schematic;
4. the same-state command-swap rollout;
5. the CNM forest plot.

A reader should understand the entire scientific question without reading the caption:

> Same robot state. Same progress. Different valid command. Which future does the controller follow?

That is much more visually persuasive than another aggregate bar plot.

---

# Safer and stronger claim language

### Safe with the current evidence

> Under a fixed training protocol and a two-walk matched-state assay, the frozen SNMR source produces more successful control than absolute within-clip time or a matched-phase wrong-trajectory source.

> The result identifies clip-disambiguating, control-usable trajectory information.

> The finding is conditional on two known walks, one robot, one representation, and one controller recipe.

### Not yet supported

> The latent understands motion semantics.

> The latent encodes future intent beyond trajectory identity.

> The representation generalizes across motions or embodiments in control.

> The (A-T) contrast is a causal intervention on one fixed controller.

> The assay is the first causal representation analysis in robotics.

### Supported after the proposed command-swap and phase tests

> Valid on-manifold changes to the SNMR command causally select different target futures from the same initial state.

> The controller uses time-varying trajectory state rather than only a static clip identifier.

---

# Preferred title and thesis

I would retain the good question in the current title but make the subtitle concrete:

## **What Crosses the Boundary? A Counterfactual Assay for Humanoid Command Interfaces**

A slightly more memorable alternative is:

## **Beyond the Clock: Measuring Control-Usable Information in Humanoid Motion Interfaces**

The first is better if you want continuity with the existing project; the second is better if you want the clock-confound finding to be what reviewers remember.

The first two sentences of the paper should be approximately:

> Learned command latents are increasingly used in humanoid control, but tracking success does not identify what the controller actually receives from them. On a deterministic reference clip, absolute time indexes every target, so motion-specific content and progress through the clip are observationally confounded.

That is a much stronger opening than beginning with the possible limitations of explicit joint targets.

---

# A current-data-safe abstract rewrite

> Humanoid tracking systems increasingly use learned command latents, yet tracking success does not reveal what the controller actually receives from them. On a deterministic reference clip, absolute time indexes every target, making motion-specific content and progress observationally indistinguishable. We introduce an interventional assay that makes a 64-dimensional code the decoder’s exclusive goal route and evaluates matched present states with divergent future references under explicit, latent, clock, phase-shuffled, and no-goal sources. Applied to a frozen multi-embodiment retargeting latent on a simulated Unitree G1, the assay first rejects the single-clip interpretation: time achieves (0.754) completion, above the latent’s (0.656). In a preregistered two-walk test, the latent reaches (0.754) ambiguity completion, compared with (0.562) for time and (0.552) for matched-phase shuffled content; paired effects are (+0.191,[0.124,0.274]) and (+0.199,[0.127,0.279]). Command destruction establishes that the exclusive route is necessary. Together with representation probes, the evidence supports a narrower conclusion than semantic understanding: the frozen latent supplies control-usable, clip-disambiguating trajectory state when exclusive, while remaining redundant beside explicit targets. The contribution is a reproducible contract for auditing learned humanoid motion interfaces under shortcut-matched controls.

This version leads with the general scientific problem, states the negative result as a contribution, gives only the decisive numbers, and ends with the exact scope of the finding. The same-policy command-swap result should replace the weaker “destruction establishes necessity” sentence once completed. The numerical statements are all supported by the frozen draft results. 

---

# A stronger future method naturally follows from these findings

Once the measurement paper is stable, the natural method contribution is **ambiguity-aware latent training**.

Use the same reference-only selector to mine pairs whose present states are close but whose (H)-step futures diverge. Add a future-contrastive objective to SNMR:

[
\mathcal L_{\mathrm{amb}}
=========================

\max\left(
0,,
\gamma+
d(z_i,z_i^+)
------------

d(z_i,z_j^-)
\right),
]

where (z_i^+) is a nearby state from the same future branch and (z_j^-) is a matched-present, divergent-future negative. A future-prediction or contact-transition head could provide a complementary structured target.

The crucial evaluation would not be another t-SNE plot. It would be:

* matched or controlled MPJPE;
* higher CNM/NCR in the unchanged interface assay;
* stronger command-swap branch adherence;
* held-out ambiguity sets to prevent the loss from merely encoding clip identity.

That would create a clean “measure, diagnose, improve” program. But placing that method into the current paper before closing target specificity would add surface area before securing the foundation.

---

# Final recommendation

Do **not** turn this into an SNMR capability paper and do not spend the remaining submission effort trying to beat the explicit ceiling. The explicit channel is supposed to be the ceiling, and the additive null is already scientifically coherent.

The most valuable next experiment is the **same-state, on-manifold command swap on the frozen SNMR controller**. It is inexpensive, produces the strongest possible figure and video, and changes the epistemic status of the result from:

> “The channel is necessary and the latent-trained policy outperforms a clock-trained policy”

to:

> “Changing a valid latent command, while holding the controller and robot state fixed, causally selects the corresponding target future.”

That single change, followed by the identity/phase decomposition and the already-completed robustness updates, would make the current submission substantially harder to dismiss.

[1]: https://arxiv.org/html/2507.07356v2 "https://arxiv.org/html/2507.07356v2"
[2]: https://arxiv.org/html/2606.29209v1 "https://arxiv.org/html/2606.29209v1"
[3]: https://arxiv.org/html/2605.30117v1 "https://arxiv.org/html/2605.30117v1"
[4]: https://linjiw.github.io/snmr/site/ "https://linjiw.github.io/snmr/site/"
