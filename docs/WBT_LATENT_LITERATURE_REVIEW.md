# Literature Review: SNMR Latent as an Observation for GMR-Reference WBT

**Written 2026-07-15, while the frozen `runs/wbt_latent_pilot` screen was training.** This document
reviews the published evidence bearing on the three integrations in
`docs/WBT_LATENT_INTEGRATION_STUDY.md` (S1 current-latent, S2 latent tangent preview, S3
privileged latent critic). All claims below were verified against arXiv full texts on 2026-07-15
unless marked otherwise. It does not change the frozen protocol; it records what the literature
predicts before results are unblinded.

## 1. Where our configuration sits in the design space

Published systems use motion latents in exactly two ways:

1. **Latent as the action space** of a high-level policy over a frozen decoder: ASE
   (arXiv:2205.01906), CALM (2305.02195), ControlVAE (2210.06063), PhysicsVAE (TOG 2022), PULSE
   (2310.04582), NCP (2308.07200), MoConVQ (2310.10198), MotionPyramid (2606.20705).
2. **Latent as the sole carrier of the reference** into the tracking policy, replacing raw target
   poses in the observation: VMP (SCA 2024), MaskedMimic decoder (2409.14393), UniTracker
   (2507.07356), LeVERB (2506.13751), FB-CPR/Meta Motivo (2504.11054), BFM-Zero (2511.04131).

**No published work adds a frozen, precomputed motion latent as a purely auxiliary observation
next to an intact explicit reference with unchanged rewards** — our exact S1/S2 configuration
appears unpublished. That is an opportunity (novel datapoint either way) and a warning (no
positive precedent).

The closest structural relatives:

- **GMT** (2506.14770) concatenates a **learned 128-d conv encoding of a ~2 s future reference
  window** with the raw next-frame target — same "explicit command + 128-d latent summary" shape
  as S2, except their encoder is trained end-to-end with the policy, not frozen.
- **UniTracker** (2507.07356) conditions the deployed tracking policy on a CVAE latent whose prior
  sees 5 future reference frames. Directly relevant negative result: adding the explicit reference
  to the actor **alongside** the latent made tracking worse (SR 88.20 with both vs 91.82
  latent-only); with the explicit reference present, "the influence of the latent variable z
  vanishes."
- **VMP** (Disney, SCA 2024) trains an RL tracking policy conditioned on a frozen kinematic-VAE
  latent trajectory while keeping explicit imitation rewards — frozen latent + explicit reward,
  but the latent *replaces* the reference in the observation.
- **FLD** (2402.13820) feeds the latent to the policy *and* decodes it into the reward-side
  reference; also shows latent reconstruction error predicts trackability (a deployable OOD/
  feasibility signal — relevant to our Gate-1b mask work).

## 2. Evidence on future-reference preview (bears on S2)

The two quantitative preview-horizon ablations that exist as of mid-2026:

| Study | Sweep | Result |
|---|---|---|
| GMT (2506.14770, Table 2, E-mpkpe mm, AMASS/LAFAN1) | next-frame only → +0.5 s → +1 s → +2 s window | 46.02 → 43.64 → 43.15 → 42.07 (monotone, diminishing); window **without** the current frame degrades to 49.52 |
| UniTracker (2507.07356) | 1/5/10/20 future frames in the prior | SR 90.62 / **91.82** / 91.25 / 90.31 — optimum at 5 frames (~0.1–0.2 s), longer slightly hurts |

Qualitative support: TWIST (2505.02833) — teacher sees 2 s of future frames; single-frame pure-RL
policies show anticipation failures (foot sliding); the future-aware teacher must be distilled in
via RL+BC. KungfuBot2/VMS (2509.16638) adopts a ~20-frame future window in teacher and student
without ablation. By contrast, the entire H2O/OmniH2O/HOVER/PHC lineage uses **only** a t+1
reference and reaches 94%+ success — near-term reference alone suffices for much of tracking.

Implications for S2: the chosen offsets (+0.2 s, +0.5 s) bracket UniTracker's optimum and sit in
GMT's productive range; keeping the full GMR command (never dropping the current frame) matches
GMT's central caveat. Expected effect size at our scale: single-digit relative error improvements,
concentrated on anticipation-heavy segments.

## 3. Evidence on asymmetric/privileged critics (bears on S3)

- Theory: the benefit of an asymmetric critic is eliminating actor-state **aliasing**
  (Lambrechts et al. 2501.19116); a privileged signal helps iff it improves value prediction
  beyond what the actor's observation supports (Informed Asymmetric AC, 2509.26000). A centralized
  /state-informed critic is "not strictly beneficial" (Lyu et al. 2102.04402); critic-only
  privileged info is the weakest way to use privileged sensing (Scaffolder, 2405.14853).
- Practice in humanoid tracking: privileged-info-through-the-**teacher** (distillation) is the
  dominant pattern (OmniH2O, HOVER, ExBody2, GMT, CLONE, AMO). True critic-only reference
  privileges exist in ASAP (2502.01143: critic gets global reference positions + root lin vel,
  actor is phase-only), PBHC/KungfuBot (2506.12851: critic gets reference body positions, root
  lin vel, DR params; 630-d critic vs 380-d actor), and BeyondMimic (2508.08241: critic gets
  per-body Cartesian tracking errors). None ablate critic-only **future** preview — S3 is a novel
  datapoint on a well-established pattern.
- On our single deterministic clip the actor's observation (explicit GMR command ≈ phase) is
  near-Markov, so the aliasing term an asymmetric critic removes is ~0. PPO value fits are poor in
  practice (Ilyas et al. 1811.02553) and V is strongly time-to-go dependent (Pardo et al.
  1712.00378), so a small early-training speedup in value fitting is the most S3 can plausibly
  deliver. Holosoma's actor and critic are separate MLPs (no shared trunk), so there is no
  interference channel by which critic inputs contaminate the actor.

## 4. Evidence on frozen pretrained representations as auxiliary RL inputs (bears on S1)

- Negative-leaning: frozen out-of-domain features are matched by learn-from-scratch with
  augmentation (2212.05749); "not more sample efficient" in model-based RL (2411.10175);
  task-dependent at best (2407.17238); representation utility tracks task alignment, not
  pretraining pedigree (2312.12444). RL-based evaluations of representations are high-variance
  (2304.04591) — expect seed noise of the same order as the effect.
- Positive existence proof: frozen features *can* beat ground-truth state (PVR, 2203.03580); wider
  learned input expansions can help (OFENet, 2003.01629) — but those gains came from features
  trained with control-relevant objectives.
- Redundant/correlated extra channels: policies can latch onto spurious channels (1912.02975);
  observation normalization is among the highest-impact PPO choices (2006.05990). **Checked in our
  stack: holosoma PPO has `empirical_normalization: true`** (running mean/std over the full
  concatenated observation), so the z-scale mismatch (per-dim z std ~0.02 vs joint command std
  ~0.57 in `walk1_subject5_mj_z.npz`) is normalized away; this documented pitfall is already
  handled.
- Single-clip determinism is the great neutralizer: on one clip, z_t is (approximately) a
  deterministic function of clip phase, which the explicit GMR command already pins down. All
  three arms are therefore predicted closer to null than multi-clip literature suggests. A tie is
  the literature-consistent outcome, not an anomaly. The latent's differentiated value —
  morphology-invariant motion semantics — can only show up across clips/embodiments, which this
  screen deliberately does not test.

## 5. Retargeting quality is first-order (context for the whole program)

- "Retargeting Matters" (2510.02252): identical BeyondMimic policies, only the retargeter varies —
  success on hard LAFAN1 clips ranges 0% (PHC-style) to ~100% (GMR/Unitree); mean global body
  error 247.8 → 77.2 mm across retargeters. GMR's uniform root-translation scaling is called
  "crucial to avoid foot sliding."
- OmniRetarget (2509.26633): contact preservation at retargeting time is directly proportional to
  downstream RL success (82.2% vs 3.9–71.3% on object tasks).
- Feasibility filtering/curation pays consistently: H2O +4.6 pp success from filtering alone
  (2403.04436); ExBody2 policy-in-the-loop filtering improves real E-mpjpe 0.1361 → 0.1074
  (2412.13196); PBHC's physics-rejected clips cap at 54% episode-length ratio.

This is direct external validation of the project's premise that reference quality (foot skate,
penetration, velocity spikes) bounds WBT tracking, and of the Gate-1/1b contact focus.

## 6. Pre-registered predictions for the frozen screen

Recorded before unblinding `runs/wbt_latent_pilot/analysis.json`:

- **S1 (current z to actor+critic):** neutral to mildly negative vs the 88%-completion GMR
  baseline. No published mechanism predicts a robust win from a redundant frozen latent on a
  single deterministic clip; UniTracker's explicit-ref+latent interference is the nearest
  precedent. If S1 improves, suspect the latent acting as a denoised phase code.
- **S2 (z + tangent preview to actor+critic):** best chance of a real improvement, most likely as
  faster early learning and better anticipation segments rather than a large asymptotic gain;
  GMT-scale (~5–10% relative) joint-RMSE improvement is the optimistic case.
- **S3 (preview to critic only):** approximately null at 8k iterations (dense reward, near-Markov
  actor observation); at most an early-training speedup. A null S3 replicates asymmetric-AC
  theory, not a failed infrastructure.
- Cross-arm reading: S2 > S1 ≈ baseline ⇒ the preview, not the latent, carries the value. S1 ≈ S2
  > baseline ⇒ the latent is a useful (phase-like) code even without preview. Any arm ≪ baseline
  ⇒ check optimization confounds before concluding the information hurt (normalization is already
  handled; input width 154→282/410 remains a real difference).

## 7. What the literature recommends if the screen is null

In priority order, each a minor change to the existing stack:

1. **Latent-only command arm** (drop the explicit joint command from the actor, keep rewards and
   termination on GMR robot-space): the UniTracker/VMP-supported configuration, and the C2
   design's actual target. Tests whether z can *carry* the command, which the auxiliary test
   cannot show. Infrastructure exists (`snmr_latent` / `snmr_latent_tangent_preview` terms).
2. **Explicit-preview control arm** (GMR command at t, t+0.2 s, t+0.5 s in robot space, no
   latent): isolates "preview helps" from "latent preview helps" — the GMT ablation says most of
   S2's expected gain may be available without SNMR at all. This is the key attribution control if
   S2 promotes.
3. **Multi-clip / held-out-motion screen**: the literature (UniTracker, VMP, PVR studies) locates
   frozen-latent value in generalization and partial observability, which a single-clip screen
   cannot detect by construction.
4. **Make the latent load-bearing rather than concatenated** (KL-aligned prior / residual
   conditioning à la MaskedMimic/UniTracker, or FLD-style latent→reference decoding): every
   published success makes the latent structurally necessary; concatenation with unchanged rewards
   gives PPO no pressure to use it.
5. **Latent-space feasibility signals**: probe whether SNMR latent reconstruction error predicts
   per-clip WBT trackability (FLD's fallback trick), and whether contact states are linearly
   decodable from z (ties into Gate-1b mask precision).

## 8. Budget sanity check against the field

Per-clip PPO budgets in deployed-humanoid papers: BeyondMimic ~1.5k–4k iterations at 4096 envs
per clip (hard clips up to 10k+); PBHC stabilizes ~20k steps at 4096 envs; CLONE teacher 8192
envs ~24 h on an A800. Our 8,000 iterations × 1,024 envs sits within the published single-clip
range, consistent with the E35 finding that walk1 is trainable (88%) at this budget while dance2
is not (10%).
