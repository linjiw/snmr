# Act Through the Latent: Closing the Retarget-to-Track Interface with a Goal-Conditioned Command Prior

**Status:** working draft v0.1 (2026-07-28). Sections I–V drafted; §VI holds slots for E53/E54.
Every number cites its experiment ID in `EXPERIMENT_LOG.md`. Integrity flags from
`PAPER_ICRA_SKELETON.md` are honored inline (FLAG-1: within-regime gap framing; FLAG-2:
exact percentages, never "halved").

---

## Abstract (v0.2, 185 words)

The dominant humanoid motion pipeline is two-stage: retarget human motion to the robot,
then train a tracking policy on the result. The interface between the stages is a stack of
robot-space joint targets — a representation chosen for convenience, not for control. We
ask what a *learned* interface carries instead. We first distill a per-frame IK retargeter
into one multi-embodiment network with a shared latent, then audit that latent with
falsification-first probes: it is content-rich (75% cross-embodiment clip retrieval at 1.8%
chance; CKA 0.91), embodiment-aligned but not invariant (0.28 linear probe vs 0.91
nonlinear attacker), physically impoverished, and — frozen — inert as a control input, a
null we replicate and that UniTracker's ablation independently confirms. Making the
interface load-bearing requires co-training: a goal-conditioned residual-to-prior CVAE,
DAgger-distilled from an RL teacher, whose 64-d latent is the tracking policy's *only*
motion command, closes the interface gap from −33 points (frozen latent, 0.65) to −2.8 points
(0.952±0.002 vs a 0.98 teacher; three seeds, all same-regime). The same latent absorbs a
second, interaction-rich teacher: adding OmniRetarget's terrain and object clips — three
human skeletons, two teachers, one network — cuts interaction-clip error 3–5× at a
measured +1.05 cm cost on locomotion. We release a preregistered negative-results ledger
and a measurement-substrate defect whose repair cut joint tracking error by 46%.

---

## I. Introduction

Nearly every humanoid that has learned to move from human motion has done so in two
stages: a retargeter converts human motion into robot-space joint trajectories, and a
reinforcement-learning tracking policy learns to follow them. The split is a good
engineering decision — retargeters and trackers are developed, evaluated, and reused
independently, and the recent generation of each is strong: interaction-preserving
optimization on the retargeting side [OmniRetarget], physics-in-the-loop retargeting
[ReActor], and unified tracking controllers that follow thousands of clips with one
network [BeyondMimic, GMT]. But the *interface* between the stages has escaped scrutiny.
What actually crosses the boundary is a per-frame stack of joint targets — a format,
not a representation.

That format is lossy in three specific ways. It discards everything the retargeter knew
that is not a joint angle: its contact reasoning, its uncertainty, and the cross-embodiment
structure it learned when one network serves several robots. It is evaluated by kinematic
metrics (pose error, foot skate, penetration) that are blind to *trackability* — the GMR
study [GMR] showed retargeting artifacts silently cap what downstream RL can learn, which
is evidence that the interface transmits harm as readily as signal. And it is
*dimensionally unshareable*: a 29-DoF joint-target stream cannot command a 23-DoF robot,
so every multi-robot system pays the interface cost once per embodiment.

The obvious repair fails, and the failure is informative. Handing the tracking policy the
retargeter's latent *alongside* its usual reference does nothing: in our experiments the
policy ignores the latent (concatenation is null-to-harmful, −10 points at worst,
E36–E39), and UniTracker's ablation reports the same mechanism independently — "when the
actor receives the reference motion directly, the influence of the latent variable z
vanishes." A frozen latent as the *sole* command is feasible but costs 16 points against
its contemporaneous explicit-command baseline (0.72 vs 0.88, E37/E39). Training a latent
by RL from scratch collapses [UniTracker, BFM]. The latent, as produced by retargeting
alone, is not a control interface.

**Our key insight is that the retargeting-to-tracking boundary should not be a
representation the two stages agree on, but one the tracking policy learns to act
through: a latent becomes useful for control exactly when it is the policy's only channel
to the goal, and is inert whenever an explicit reference is available beside it.** The
sections that follow are the constructive and destructive halves of that claim. We build
SNMR, a skeleton-agnostic retargeting network distilled from a classical IK teacher whose
shared latent serves five humanoids, and we audit that latent with falsification-first
probes before asking it to do anything (§IV). We then make the interface load-bearing
(§V): a conditional VAE whose goal-conditioned prior compresses the reference into a 64-d
command latent, trained by DAgger from an explicit-command RL teacher, with the reference
visible to the prior but never to the decoder. On a simulated Unitree G1 the latent-only
command reaches 0.952±0.002 ten-second completion against the 0.98 teacher across three
seeds — the interface gap shrinks from sixteen points to under three.

Two findings we did not go looking for became contributions. First, a world-body indexing
defect in a widely used MuJoCo-Warp training framework silently zeroed the primary
body-position tracking reward in every run — ours and, we believe, others'; its repair cut
joint tracking RMSE by 46% with no change to the learning algorithm and brought our
tracker into an externally calibrated band (§VII). Second, rolling tracking policies out
on their own references and recording the simulated states yields physically repaired
supervision — 2–3× less foot skate at a measured fidelity price — quantifying the
data-side alternative to physics-in-the-loop retargeting (§VI-Q4).

**Contributions.**
- **A learned retarget-to-track interface.** A goal-conditioned, residual-to-prior CVAE
  command prior, DAgger-distilled from an RL tracking teacher, whose 64-d latent is the
  policy's only motion command — 0.952±0.002 completion against a 0.98 explicit-command
  teacher (3 seeds), with the reference never visible to the decoder.
- **A falsification-first audit of a shared motion latent.** Preregistered probe families
  establishing what the latent carries — content-rich, embodiment-aligned but not
  invariant, physically impoverished, control-inert while frozen — with the negative
  verdicts stated as measured results, one independently replicated by UniTracker.
- **One multi-embodiment retargeter distilled from per-robot IK.** A 1.5M-parameter
  skeleton-agnostic network amortizing an IK teacher across five humanoids (23–30 DoF),
  joint limits by construction, 671 fps on CPU, with the sharing cost (3.66 vs
  2.9–6.0 cm) and the zero-shot-transfer failure (5.2×) measured rather than assumed.
- **A measurement-substrate defect, found, fixed, and released**, plus the preregistered
  negative-results ledger — closed research lines with the evidence that closed them, so
  the field can stop re-deriving them.

## II. Related Work

**Retargeting as preprocessing.** Classical retargeting solves per-frame IK with hand-tuned
correspondences [GMR]; interaction-mesh methods add hard contact constraints
[OmniRetarget]; ReActor moves retargeting inside physics via bilevel RL, eliminating
artifacts by construction. All three families emit the same interface — robot joint
trajectories — and are evaluated by kinematic quality plus, recently, downstream RL
success [GMR, OmniRetarget]. None asks whether the *representation* handed to the tracker
should itself be learned. Neural feed-forward retargeters [IKMR, NMR, AdaMorph, GBC]
amortize the optimization and, in AdaMorph's case, span many embodiments through a
morphology-agnostic latent — but their latents are internal machinery, not command
interfaces, and none is analyzed for what it carries.

**Tracking and its command spaces.** Modern trackers follow references specified as joint
targets with body-pose terms [BeyondMimic, GMT, ExBody2]; HOVER unifies multiple command
*modes* by distillation; UniTracker compresses the reference into a CVAE latent and is the
closest precedent for our Stage-2 — single-embodiment, with no retargeter and no
representation analysis. Latent skill spaces for physics characters [PULSE, MaskedMimic,
ControlVAE] establish the recipe elements we verify at robot scale: learned conditional
priors, residual-to-prior encoders, distillation-not-RL. The Disney lineage [VMP,
BFMTrack] treats latents as tracking commands for characters; BFMTrack's latent-sequence
optimization is test-time-only. To our knowledge no published system co-trains a
*multi-embodiment retargeting* latent under the *control* objective; the closest
single-loop work [Dexplore] is dexterous-hand-specific.

**Interface critiques.** BeyondMimic names a "planning-control gap"; the GMR study
quantifies how retargeting quality caps tracking. Both stop at the boundary we cross:
neither changes what the boundary *is*.

## III. The Two-Stage Interface

*(Formalization: source motion m_t → retargeter R → reference g_t → tracker π. The
standard interface is g_t = joint targets; ours replaces the actor-visible g_t with
z_cmd = μ_prior(proprio, goal), decoder never sees goal. The exclusivity contract and the
three hypotheses H-interface / H-transfer / H-parity, each mapped to an experiment
section. SNMR itself compressed to one paragraph + one table: GAT encoder (per-node
features + adjacency, global pool = skeleton-agnostic), z=128, AdaLN embodiment-conditioned
decoder, tanh limits; 77 LAFAN1 clips × 5 robots = 2.48M paired frames from a GMR teacher;
held-out MPJPE 3.66 cm CI [3.46, 3.86] (G1 specialist), 2.9–6.0 cm shared; 671 fps CPU.
Foot skate 0.29 vs teacher 0.05 m/s stated as the known inherited gap.)*

## IV. Auditing the Latent

*(Four claim-headed paragraphs, severity-ordered, one table. (1) Content: 75% cross-
embodiment retrieval @1.8% chance, CKA 0.91 [E2/E23]. (2) Embodiment: aligned, not
invariant — linear probe 0.28 vs MLP attacker 0.91, the Elazar-Goldberg gap [E1/E3];
scale explains ~1% of leak [E11]. (3) Physics: contact not decodably present under pure
distillation (z-linear F1 0.023) but co-trained contact BCE injects it durably (F1 0.227
at 100k, 1.8× a deployable-mask baseline) at a real fidelity price (+1.5 cm at weight 0.5
→ registered default 0.25) [E38, E48, E48-100k]. (4) Control: frozen-latent inertness,
null-to-harmful concatenation, the E49 config-only null — closing with UniTracker's
independent replication. Framing sentence: "the audit is the specification for §V.")*

## V. Making the Latent Load-Bearing

*(Method: goal-conditioned prior ρ(z|proprio, goal), residual-to-prior posterior with tied
σ=0.3 giving closed-form KL, decoder(proprio, z) only, DAgger from the explicit teacher,
episodic noise, PULSE smoothness regularizer, deterministic μ_prior deployment. Then the
v1→v3 arc AS A FINDING: (v1) collect on the posterior path → deployment collapses to
0.001 while the posterior path scores 0.84 — train/deploy path consistency is not
optional; (v2) prior-path collection alone recovers only 20×, and mixing prior-z into the
action loss damages the posterior (0.84→0.42); (v3) conditioning the prior on the goal is
the load-bearing fix — 0.935/0.955. Design lesson sentence: "the prior needs the goal;
the posterior needs only a small privileged residual; the decoder needs neither.")*

## VI. Experiments

**Q1 — Does the learned interface work?** E52 three-seed table: teacher 0.98 / D
0.952±0.002 / C 0.944±0.008 / frozen-latent 0.65 (E52-L1R, re-run post-fix so all rows
share one regime); RMSE 0.127–0.128 vs teacher 0.122. The frozen-vs-co-trained gap
*widened* under the repaired reward (30pp): a working body-position signal lifts explicit
and co-trained interfaces alike, but a frozen non-causal latent cannot exploit it —
co-training is what converts the representation into an interface.

**Q2 — Does the retargeting latent add value beyond the explicit goal?** Single-clip:
null (paired D−C = +2.1/+0.5/−0.2pp; D's seed-variance 4× tighter, suggestive only).
*(E53 multi-clip slot: teacher gates failed at 8k/16k [0.378/0.465, dynamic-tail-limited
at 1024 envs]; 2048-env retry queued. E54 cross-embodiment slot: the case where explicit
commands are dimensionally unshareable.)*

**Q2b — Does the interface scale with better retargeting data?** Yes, measurably: adding
a second, interaction-rich teacher (OmniRetarget; 1,938 clips across 52/53-joint human
skeletons with synthetic orientations) cuts held-out interaction-clip error 28.9→5.5 cm
(object) and 25.1→8.2 cm (terrain) while locomotion pays +1.05 cm (4.36 vs 3.31) — after
a sampling-diet fix whose before/after itself attributes 61% of the naive run's
interference to data balance rather than representational conflict. The terrain plateau
(~8.2 cm, vs 5.5 for object) is the measured signature of a hidden conditioning variable
(terrain scale), tested directly in the variant-code experiment (E56-D, running).

**Q3 — Are learned references as trackable as IK references?** B1-confirmatory: 86.7% =
86.7% point-equal completion (3 seeds × 2 eval seeds × 100 rollouts each source), RMSE
noninferior (+0.33%, CI [−3.1, +4.9]); completion CI half-width ±6pp exceeds the −5pp
preregistered margin — a power statement, not a quality gap. *(E57-A positive-control and
E57-B harder-clips slots reserved; equality without a sensitivity check is not evidence.)*

**Q4 — What does physics-repaired supervision buy and cost?** E50-A/E51: rollouts of a
calibrated tracker show 2–3× lower stance-foot speed than the references they track
(3–6× on pre-fix policies) with zero penetration (a simulator property, not a claim), at
6.2–6.7 cm heading-local reconstruction distance — the quantified data-side alternative
to bilevel physics retargeting.

## VII. A Defect in the Measurement Substrate

*(DEFECT-1 self-contained: mechanism (world-body off-by-one; 33-wide state tensors vs
32-entry name list; pelvis slot read the world body), evidence (reward term identically
0.0 for 8,000 iterations; body-position error flat at 6.78 m), consequence (every
MuJoCo-Warp run — ours, and plausibly other groups' — trained as an orientation+velocity
tracker), repair (+monkeypatch released), effect (joint RMSE 0.263→0.142 rad, −46%;
completion 0.90→0.94; with a joint-space reward term added, 0.122 rad / 0.98,
matching the mjlab external calibration band of 97.9%/3.0 cm), scoping (paired
comparisons under the defect survive; absolute capability claims did not).)*

## VIII. Limitations

*(Stage-headed, mechanistic: **Interface scope** — single-clip additive value is null;
multi-clip/cross-embodiment in progress. **Embodiment generalization** — zero-shot to an
unseen robot fails at 5.2×; embodiment augmentation is the identified path.
**Physics of the latent** — contact enters only by co-training at a fidelity price;
repaired references cost 6+ cm at current tracker quality. **Simulation only** — no
hardware; MuJoCo-Warp externally calibrated but sim-to-real unvalidated.
**Data scale** — 4.6 h/robot is small; two-teacher expansion (OmniRetarget) underway,
with the terrain subset providing verified same-human/multiple-robot multimodality.
Closing agenda sentence.)*

## IX. Conclusion

The boundary between retargeting and tracking has been a file format. Treated as a
representation — audited before trusted, co-trained before deployed, and made the
policy's only channel to the goal — it carries nearly everything the explicit interface
carried, in a form one network can share across robots. The measurement lesson runs
deeper than the architecture: two of this paper's strongest results are a reward channel
that was silently dead and a set of preregistered nulls; we commend both practices to the
field.
