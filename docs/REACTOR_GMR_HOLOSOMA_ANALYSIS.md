# ReActor vs GMR vs holosoma vs SNMR: Mechanics, Threat Model, and Research Directions

**Date:** 2026-07-23. **Status:** research memo (literature + code analysis; no new experiments).
**Sources:** ReActor paper (arXiv:2605.06593, Disney Research, May 2026); code dives into the
pinned GMR and holosoma clones (file:line citations verified 2026-07-23); web-verified
literature pass (arXiv IDs resolved via API; items not on arXiv flagged).
**Companions:** `RESEARCH_PROPOSAL_RETARGET_TO_TRACKING.md` (live plan H1–H4),
`FORWARD_RESEARCH_MEMO.md` (prior lit pass), `PROGRAM_CONSOLIDATION.md`.

---

## 1. What ReActor is (precise mechanics)

ReActor (Müller et al., Disney, TOG 2026) frames retargeting itself as RL inside physics,
co-optimized with a small retargeting parameterization — a **bilevel, single-loop (TTSA-style)**
scheme:

- **Parameterization (upper level).** User gives sparse semantic rigid-body pairs in nominal
  pose (T-pose). System extracts a global scale `s = h_target/h_source` and per-pair nominal
  transforms `(x_nom, R_nom)`, then adds *learnable* local offsets `p_pos, p_ori` per pair and
  a per-motion vertical offset `p_z`, each constrained to a norm ball (δ = 0.5). The
  parameterized reference `g_t` is a closed-form map of the source motion `m_t` (Eqs. 10–13).
- **Upper-level gradient trick.** Instead of implicit-function-theorem differentiation through
  RL, they assume the optimal rollout shifts with the reference as `∂s*/∂g = αI`, collapsing
  the total derivative to `d̃_p L = (1−α)·∂ℓ/∂g · d_p g` — the Jacobian of the *kinematic
  mapping only*. Projected gradient step on `p` every PPO iteration.
- **Lower level.** PPO tracking policy at 50 Hz: PD setpoints + **residual root wrench** with a
  continuous deadband (d = 0.1) and an L1 penalty scaled by a phase variable ψ. ψ ∈ [0,1] also
  replaces RSI: robot starts at source root state + noisy nominal joints, reference paused,
  policy *learns* the initialization; segments with ψ<1 are excluded from upper-level batches.
  Adaptive per-clip failure-rate sampling. 4096 envs, 20k iters, ~6 h on one RTX 5090
  (filtered AMASS).
- **Output = the simulated rollout.** The retargeted dataset is the *policy's state sequence*,
  so ground/self-penetration are zero and foot slide ≈ contact-solver noise **by construction**.
- **Bonus property:** the trained policy is itself a real-time (88 Hz) retargeter that
  generalizes to unseen motions (100STYLE pseudo-GT eval: pos err 0.19 cm training on AMASS vs
  0.12 cm training on 100STYLE itself).

**Headline numbers (G1, PHC-filtered AMASS):** foot slide 0.17 cm/s vs GMR 1.25 / OmniRetarget
2.00; self-pen 0 vs 0.07/0.12 (time fraction); downstream DeepMimic-style tracking success
**97.45%** vs OmniRetarget 95.51 / GMR 89.93. Compute parity: ~6.5 h GPU vs OmniRetarget ~7 h
CPU vs GMR ~5 h CPU for the dataset.

**Caveats they admit:** external-force penalty weight is the sensitive knob (realism vs
coverage); retargeting physically impossible motions stays ill-posed; parameters are
time-invariant; correspondences still manual. Note also the generated data is "feasible modulo
residual wrench" — downstream policies trained *without* the wrench still succeed, so the
laundering works, but it is laundering, not strict feasibility.

## 2. What our two upstream codebases actually do (verified against code)

### GMR (our teacher)
Per-frame **mink QP differential IK** (LM-damped Gauss-Newton, daqp backend, damping 5e-1) over
a contact-disabled MuJoCo model (`motion_retarget.py:18-19,173-219`). Two sequential
hand-weighted match tables (13 body pairs for G1; pelvis/feet pos-weight 100), per-body
*position-only* scaling in the root-local frame (0.8–0.9 × height ratio,
`scale_human_data():243-266`), hand-tuned per-pair frame offsets. Joint limits are hard QP
constraints; **no contact handling** (global z-shift only, `smplx_to_robot_dataset.py:118-127`),
**no temporal terms** beyond warm-starting, hard motions (crawl/lie/stairs) blacklisted rather
than solved. CPU 35–70 fps. Every artifact ReActor measures against GMR is structural: the
solver has no mechanism that could remove skate/penetration, only reweight it.

### holosoma retargeting (OmniRetarget-family)
Interaction-mesh (Laplacian) objective + **hard-constraint SQP QP** via cvxpy/CLARABEL
(`interaction_mesh_retargeter.py:65-85,703-734`): foot-sticking XY, foot-lock Z, ground/object
non-penetration (`mj_geomDistance`), self-collision clearance, joint limits, trust region.
Contact-*aware* but purely kinematic — no mass/torque/momentum reasoning. Per-frame
optimization; offline; not feed-forward.

### holosoma WBT (our tracking stage)
BeyondMimic-style goal-conditioned PPO, fully **decoupled** from retargeting: obs = current
reference frame only (ref joint pos/vel + ref base ori in body frame; asymmetric critic gets
privileged body poses); 50 Hz PD (action_scale 0.25 × effort/kp); exp-kernel tracking rewards
(root pos/ori σ 0.3/0.4 w 0.5; per-body relative pos/ori w 1.0; body lin/ang vel; action-rate
−0.1; soft-limit −10; undesired-contact −0.1); RSI + failure-binned adaptive timestep sampling;
friction/COM/push DR; MLP 512/256/128; 4096 envs. Confirmed absent: residual root forces (RFI
is disabled torque-noise DR, not a root wrench), any reference adaptation during training, any
latent machinery. Motion npz contract: fps, joint_pos/vel, body_{pos,quat wxyz,lin_vel,ang_vel}_w,
joint/body names (`config_types/command.py:68-131`).

**Structural takeaway:** GMR and holosoma jointly instantiate exactly the decoupled
pipeline ReActor's bilevel coupling replaces. Neither has any element of ReActor's two
innovations (physics-generated references; reference/policy co-adaptation).

## 3. Where the field is (verified lit pass, 2026-07)

- **Feed-forward "train once, retarget in real time" is now crowded:** IKMR (2509.15443, shared
  topological latent, >5000 fps, physics-refined), NMR "Make Tracking Easy" (2603.22201 — RL
  experts repair references, distilled into a CNN-Transformer; our closest analog on the
  distillation axis, single robot), GBC (2508.09960, differentiable-IK net, any-robot),
  Human2Humanoid CycleGAN (2606.03476), ULTRA (2603.03279). Real-time feed-forward alone is
  **no longer a contribution**.
- **AdaMorph (2601.07284) is the most dangerous competitor**, not ReActor: single
  embodiment-aware transformer, "morphology-agnostic latent intent space," AdaLN prompts,
  **12 humanoids, claims zero-shot to unseen robots** — the claim our LORO currently fails at
  5.2×. Our moat against it is latent *analysis* + latent-as-command, not retargeting quality.
- **Bilevel ancestry:** Zhao et al. CoRL 2024 (2410.01968) already alternated policy/reference
  optimization with a latent motion model; KungfuBot's bilevel adapts tolerance, not geometry;
  PhyGile (2603.19305) is the nearest concurrent (generation-side). ReActor is plausibly first
  to do single-loop joint optimization of a *parameterized retargeting map* + RL tracker.
- **Latent-as-command SOTA moved:** VMP (our N10 template) is now explicitly criticized by
  BFMTrack (2606.25056, Disney 2026): latent-*sequence optimization* over a behavioral
  foundation model, real humanoid, no reward engineering. BeyondMimic's guided diffusion is the
  other frontier. PULSE/UniTracker remain the verified precedents for *co-trained* latent
  commands (already load-bearing in our H2). **Nobody in the retargeting set (ReActor, NMR,
  AdaMorph, IKMR) analyzes or controls through their latent — ReActor has no latent at all.**

## 4. Threat model: what ReActor kills vs leaves open for SNMR

**Killed (do not compete here):**
1. *Single-robot physical retargeting quality.* Foot-skate/penetration as a metric battle is
   over; physics-in-the-loop wins by construction. Our Gate-1 line (soft penalties, deployable
   masks, projection, flow retrofits) already reached the same verdict internally — E29's
   oracle-vs-deployable gap, E47's Dirac result, and the Gate-1 postmortem directive
   ("physics-repaired supervision or retargeting/tracking co-optimization") are the same
   conclusion ReActor proves constructively.
2. *Real-time feed-forward retargeting as standalone novelty* (ReActor's own policy does it at
   88 Hz on unseen motions; plus the §3 crowd).

**Left open (our defensible ground, in order of moat depth):**
1. **The latent as an analyzable, control-bearing object.** No competitor probes what their
   retargeter learned (embodiment vs content factorization, contact decodability, retrieval,
   CKA) or acts *through* the retargeting latent (H2/E49). ReActor's knowledge is locked in
   policy weights + 6 scalars/body — nothing to interpret, nothing to reuse for control.
2. **Multi-embodiment shared structure at distillation cost.** ReActor is one policy per robot
   (~6 h GPU each, 4096-env sim); GMR-teacher distillation costs CPU hours and one network
   serves 5 robots. AdaMorph contests this claim, so it must be paired with (1).
3. **Zero-shot embodiment generalization** — open in principle; AdaMorph claims it; our LORO
   fails it. Embodiment augmentation (E12 variants) is the path if we re-enter.
4. **Cross-robot transfer through a shared latent (C7)** — no one else can even pose the
   question (requires the shared space).

## 5. What we should learn from ReActor (concrete mechanisms)

1. **Physics-repaired supervision is buildable from what we already have.** ReActor's deepest
   point: *the simulated rollout of a competent tracking policy IS the physically-consistent
   version of the reference.* We already train such policies locally (B1: SNMR-ref 86% / GMR
   88% completion on MuJoCo-Warp WBT). Rolling out B1-grade policies on teacher references and
   re-exporting the *simulated states* as training pairs gives zero-penetration, low-skate
   supervision without implementing any bilevel machinery. This is the cheapest attack on the
   closed Gate-1 endpoint (0.08 m/s) — it routes around the deployable-contact-mask bottleneck
   entirely, because physics supplies the contact signal.
2. **Residual root wrench (deadband + annealed penalty) is the coverage knob for data
   generation.** Our 5,400-rollout study had *zero* 10-s completions; ReActor's wrench exists
   precisely to make a single policy cover hard clips during *generation* (deployment policies
   train without it). If rollout-repair coverage is the blocker, add a wrench action to the
   data-generation WBT variant only. holosoma has no such action term today
   (`base_task.py:433-435` applies only joint PD); it would be a new action term, not a config flag.
3. **The simplified bilevel gradient legitimizes cheap co-adaptation.** `(1−α)·∂ℓ/∂g·d_p g`
   needs only the kinematic map's Jacobian — for us, `g` is the SNMR decoder output, so the
   "upper level" can be decoder (or adapter) weights updated by backprop through the decoder
   against rollout states, no differentiation through RL. This is exactly the deferred F4 item
   ("ReActor-style bilevel: only if #1 and #3 fail") — its prerequisite question is now
   answered by ReActor: yes, the scheme converges and single-loop is enough (with Zhao CoRL'24
   as independent precedent).
4. **ψ-phase learned initialization** replaces RSI where source/target configurations mismatch —
   directly relevant to multi-embodiment WBT (no per-robot IK at reset) and to any co-training
   arm where the reference is being adapted.
5. **Evaluation devices worth adopting:** (a) downstream tracking success as the *primary*
   retargeting metric (validates making C6/B1 the main-paper claim); (b) pseudo-ground-truth
   generalization eval (train a reference policy on the test set, measure others against it) —
   applicable to E49/H4 generalization claims without ground truth.

## 6. Proposed integration with the live plan (no derailment of H1–H4)

The proposal's Stage 0–4 sequence stands. Additions, in priority order:

- **P0 (blocking, hours):** fix the B1 confirmatory analyzer path bug
  (`analysis.json` = `invalid_confirmatory_artifacts`; per-policy rollout reports exist,
  e.g. SNMR seed0/eval404 = 0.84 completion). The 17 h of GPU work is done; C6 is gated on a
  path-join bug. Re-run analyzer only.
- **P1 (registered next after E49): E50 "physics-repaired teacher" (rollout laundering).**
  Use the best B1 tracking policy per source to roll out all 77 LAFAN1 clips; accept clips by
  completion gate; export simulated states as corrected pairs; fine-tune SNMR (contact-BCE 0.25
  per E48-100k) on repaired pairs. Primary endpoint: deployable foot-skate ≤ 0.08 m/s (the
  failed Gate-1 endpoint) with MPJPE guard ≤ +0.5 cm vs current. Kill criteria: (a) rollout
  coverage < 60% of clips at completion ≥ 0.8 without a wrench (then evaluate P1b before
  proceeding); (b) rollout skate not ≤ 0.5× teacher skate (laundering premise false).
  *This is the only remaining path to C5-positive the program's own postmortem left open, and
  ReActor is the constructive proof it works.*
- **P1b (conditional on P0/P1 coverage):** wrench-augmented data-generation WBT variant
  (new action term in the pinned holosoma clone or monkeypatched like `wbt_latent.py`).
- **P2 (upgrade of H2's story, not a new experiment yet): unified co-training.** If E49/H2
  promote, the flagship framing becomes: *one shared latent that simultaneously (i) decodes the
  reference (retargeting), (ii) commands the tracker (act-through-latent), and (iii) is
  co-adapted against physics rollouts (ReActor-style upper level on the decoder).* That
  combination — representation + control + physics co-adaptation, multi-embodiment — is beyond
  ReActor (no representation), UniTracker (single embodiment, no retargeting), and AdaMorph
  (no control coupling). It is also falsifiable piecewise: each ingredient has its own gate.
- **P3 (interpretability, cheap, riding on E49/H2/E50):** re-run the E16/E38/E48 probe suite
  on latents *after* control co-training and after physics-repaired supervision. Preregistered
  prediction: contact decodability and motion-category structure increase when the latent is
  made load-bearing for control (currently: content-specific but semantically unorganized,
  contact not deployably decodable). This converts the "latent/skill understanding" goal into
  before/after measurements attached to experiments we are running anyway.
- **Positioning note for PAPER_DRAFT:** cite ReActor as the constructive upper bound for
  single-robot physical quality and reframe C5 as diagnosis + negative-space theory; keep the
  headline on C6 (quality transfers to control) + shared-latent analysis + act-through-latent.
  Treat AdaMorph as the baseline to differentiate against on the multi-robot axis.

## 7. Honest limits of the ReActor route for us

- **Compute asymmetry:** ReActor uses 4096 IsaacSim envs on an RTX 5090 for 6 h per robot; we
  have a single A10G and MuJoCo-Warp WBT at ~13–17 h for six 8k-iter policies. Full bilevel
  per-robot training is near our budget ceiling; rollout laundering (P1) reuses policies we
  already train, which is why it is ranked first.
- **RFC is a data-generation device, not free:** references repaired under a wrench can encode
  dynamics the robot cannot realize unassisted; ReActor's own force-penalty sweep (Fig. 8)
  shows the realism/coverage trade is a real dial, and their quadruped "gives up tracking to
  avoid force usage." Any P1b arm must report wrench-usage stats alongside quality metrics.
- **ReActor's generalization eval is pseudo-GT, not GT** — its 88 Hz "neural retargeter" claim
  is measured against a policy trained on the test set. Adopt the device, but don't overrate
  the claim when positioning.
- **Per-motion `p_z` and time-invariant offsets** mean ReActor still hand-waves per-frame
  contact timing; it wins on artifacts because *physics* filters them, not because the
  parameterization is rich. Our latent is strictly more expressive — the open question (H2/E50)
  is whether we can make expressiveness pay under physics, which is exactly the program's
  current bet.
