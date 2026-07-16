# WBT Latent Integration — Design, Implementation, Validation, and Experiment Plan v2

**Drafted 2026-07-15, after the literature review (`docs/WBT_LATENT_LITERATURE_REVIEW.md`) and
while the frozen Phase-0 screen (`runs/wbt_latent_pilot/`) is still training.** Phase-0 results
are not unblinded at drafting time; every Phase ≥ 1 decision below is conditioned on them through
the pre-registered decision tree in §5. Predictions for Phase 0 were pre-registered in the
literature review §6 and in this document's §5.1.

**Status update 2026-07-16 (E39):** Phase 0/1/2 are complete. L1 replicated only its absolute
70% command-feasibility floor (all six cells 72-77%); S3 did not replicate the benefit rule; C3
matched S3 at seed 0, so no latent-specific critic benefit is established. The multi-clip
protocol is now gated on a baseline-only budget calibration; see
`docs/WBT_LATENT_PHASE3_PROTOCOL.md`.

---

## 1. Research question and framing

**Primary question (near-term):** Does the frozen SNMR latent `z` carry information that improves
Holosoma WBT PPO tracking when the high-quality GMR robot-space reference is retained for
rewards, termination, and official evaluation?

**Attribution question (new in v2):** If a latent arm improves tracking, is the improvement caused
by (a) *anticipation* (any future information helps, latent or not), (b) the *latent
representation itself* (morphology-aligned motion code beats raw robot-space preview), or (c)
*optimization side-effects* (input width, normalization)? Phase 0 alone cannot distinguish these;
Phase 1 adds the controls that can.

**Strategic question (long-term, unchanged from C2):** Can `z` serve as the *command interface*
of a shared, embodiment-conditioned tracking policy? The literature (UniTracker, VMP, LeVERB, GMT)
says latent commands work when the latent is load-bearing, and that frozen auxiliary latents next
to explicit references are the weakest configuration. The plan therefore escalates from
"auxiliary" to "load-bearing" as evidence accumulates, rather than betting everything on the
auxiliary test.

### Design principles carried over from the lit review

1. **Never drop the current-frame explicit target from a *tracking* policy without a control**
   (GMT: window-only is worse than next-frame-only; but UniTracker: latent-only beats
   latent+explicit — the field is genuinely split, so we test both configurations).
2. **Preview offsets +0.2 s / +0.5 s are well-placed** (bracket UniTracker's optimum; inside
   GMT's productive range). Keep them fixed across arms for comparability.
3. **A single-clip screen biases every arm toward null** (z is ≈ a function of phase there).
   Interpret Phase-0/1 nulls as "not detectable in this regime," not "latent is useless";
   the latent's differentiated value is testable only in Phase 3's multi-clip setting.
4. **Critic-only arms are theory-predicted null** on near-Markov single-clip observations; they
   stay in the matrix because they are free at deployment and the field has never ablated
   critic-only *future* preview.
5. **Observation normalization is already handled** (holosoma `empirical_normalization: true`);
   do not add per-term scaling that would double-normalize.

---

## 2. Experiment matrix

All arms use: G1 `g1-29dof-wbt`, MuJoCo/Warp, 1,024 envs, PPO 8,000 iterations, save interval
2,000, GMR reference NPZ (standard fields byte-identical across arms), official terminal-aware
evaluator, 100 phase-stratified 10-s windows, eval seed 404. Baseline: existing GMR seed-0
8,000-iteration checkpoint (88% completion, 8.953 s survival, 0.2543 rad joint RMSE).

| Arm | Actor motion term | Critic motion term | Actor dim | Status |
|---|---|---|---|---|
| B (baseline) | `motion_command` (GMR qpos+qvel @ t) | same | 154 | done (frozen artifact) |
| S1 | GMR@t + `z_t` | same as actor | 282 | Phase 0 (training) |
| S2 | GMR@t + `z_t` + Δz(+0.2s) + Δz(+0.5s) | same as actor | 538 | Phase 0 (queued) |
| S3 | `motion_command` (= B) | S2 term | 154 | Phase 0 (queued) |
| **C1** | GMR@t + GMR joint_pos(+0.2s) + joint_pos(+0.5s) | same as actor | 212 | Phase 1 — **explicit-preview control** |
| **L1** | `z_t` + Δz(+0.2s) + Δz(+0.5s), **no explicit joint command** | GMR@t + S2 term (privileged) | 480 | Phase 1 — **latent-only command** |
| C2 (optional) | GMR@t + 64-d PCA of z_t (+deltas) | same as actor | 346 | Phase 2 — only if S2 promotes and C1 does not explain it |

Arm rationale:

- **C1 is the attribution control for S2.** GMT's ablation predicts explicit preview alone
  captures most anticipation gains. If S2 ⊤ C1 ≈ B, the latent representation matters; if
  S2 ≈ C1 > B, the preview matters and SNMR adds nothing on this axis; if C1 > S2, raw
  robot-space preview beats the frozen latent (negative for the latent, useful for WBT anyway —
  and a free product improvement for the GMR-reference pipeline).
- **L1 is the first load-bearing test of the C2 direction.** The actor must extract the joint
  command from `z` (plus the existing `motion_ref_ori_b` orientation cue and proprioception);
  rewards/termination stay on GMR robot space. UniTracker predicts this configuration is where a
  good latent shines; it is also the honest prerequisite for the shared-policy vision — if a
  single-robot policy cannot track from `z` on one clip, a five-robot policy cannot either.
  L1's critic keeps the explicit command (privileged, deployment-free), which the ASAP/PBHC
  precedent supports.
- C2 (dimensionality control) is deferred: only worth GPU-hours if latent width is the suspected
  confound after Phase 1.

New-arm observation dims (smoke-verified 2026-07-15): holosoma's MotionLoader `joint_pos` is
29-wide DOF-only (root excluded), so the base motion command is 58-d. C1 actor = 154 + 2×29 =
**212** (the explicit preview is therefore root-free — correct GMT-style content, no global
frame leakage); L1 actor = 154 − 58 + 384 = **480**; L1 critic = 286 − 58 + 58 + 384 = **670**.
Both new terms trained 20 smoke iterations with finite rewards at 256 envs.

---

## 3. Implementation plan

All changes live in the SNMR repo; the pinned holosoma clone is never edited (same monkeypatch-free
`func=` override mechanism the Phase-0 arms use).

### 3.1 New observation terms — `snmr/integration/wbt_latent.py`

1. `motion_command_with_explicit_preview(env)` (arm C1):
   gather `motion.joint_pos[min(t+10, end)]` and `motion.joint_pos[min(t+25, end)]` and
   concatenate with `motion_command.command`. Reuses the `_latent_at_offsets` boundary-clip logic
   generalized to arbitrary per-frame arrays — refactor it into
   `_gather_at_offsets(motion_command, array, offsets)` used by both latent and explicit preview.
   Preview uses *positions only* (not velocities): matches GMT's target-frame content and keeps
   the width increase modest.
2. `latent_preview_command(env)` (arm L1 actor): `[z_t, Δz_short, Δz_long]` — this is the existing
   `snmr_latent_tangent_preview`, adopted as-is.
3. `motion_command_with_latent_preview_privileged(env)` (arm L1 critic): explicit GMR command +
   full latent preview — this is the existing `motion_command_with_latent_preview`, reused.

Net new code: one small refactor + one new function (~20 lines). No holosoma changes.

### 3.2 Driver and analyzer extension

- Extend `scripts/run_wbt_latent_pilot.sh` → `scripts/run_wbt_latent_phase1.sh` with arms
  `c1_explicit_preview` and `l1_latent_command`, writing to `runs/wbt_latent_phase1/`. Same
  idempotent/resumable structure (training_map/evaluation_map TSVs, sha256 provenance, COMPLETE
  marker). The Phase-0 baseline report/checkpoint stay the comparison anchor.
- Extend `scripts/analyze_wbt_latent_pilot.py`: parameterize the frozen `ARMS` dict per phase
  (constructor argument or `--arms-json`), keeping the validation logic (config cross-check,
  rollout schema, paired bootstrap) unchanged. Add a **pairwise arm-vs-arm effect table**
  (S2 vs C1 is the attribution readout, not just arm-vs-baseline).
- L1 changes the actor input contract, so add an obs-dim assertion per arm to the config
  validation (read `holosoma_config.yaml`, check the registered actor/critic obs shapes in the
  train log against the expected dims of §2).

### 3.3 Multi-clip infrastructure (Phase 3) — **BUILT 2026-07-15 (while Phase 2 trains)**

- `scripts/export_wbt_gmr_latent_batch.py`: batch GMR-reference + latent export; output directory
  is directly usable as a holosoma `motion_dir`. Verified byte-identical to the pilot NPZ on
  walk1_subject5. Candidate Phase-3 sets exported to `runs/wbt_latent_gmr_multi/`:
  train = {walk1_subject2, walk2_subject1, walk3_subject1, walk3_subject3, run1_subject2,
  sprint1_subject2}, heldout = {walk1_subject5, run2_subject1}. (Set selection is provisional
  until the Phase-3 protocol freeze; walk1_subject5 goes to the held-out side because every
  Phase-0/1/2 policy trained on it — reusing it in Phase-3 training would leak the screen clip.)
- `_ensure_latent_loaded` now supports `motion_dir` (concatenates per-clip latents in
  MultiMotionLoader's sorted-glob order, validates each clip's length against
  `motion_start_idx`/`motion_end_idx`, fails loudly on skipped/misaligned files). Unit-tested
  (glob-order concat + mismatch rejection). `_latent_at_offsets` already respects per-clip
  boundaries via `motion_end_idx[motion_ids]`.
- Per-clip PPO budgets are a known open problem (E35: dance2 misses the 25% floor at 8k). Phase 3
  therefore uses a **walk-family multi-clip set first** (clips the budget calibration says are
  trainable), before touching dance/fight.

### 3.4 Explicitly out of scope for this plan

Online SNMR inference in the RL loop; latent rewards; joint SNMR+PPO training; multi-robot policy
batching; RL-to-retargeter feedback. Each is gated behind Phase-3 evidence (§5.4) — the
literature is unanimous that these are only worth building once the latent demonstrates value in
a cheaper configuration.

---

## 4. Validation plan (before any Phase-1 GPU-hours)

1. **Unit tests** (`tests/test_wbt_latent_integration.py` extension):
   - `_gather_at_offsets` boundary behavior at clip end (offset clamps, no wraparound);
   - C1 term returns `(N, 226−…)` expected width and equals manual gather on a toy loader;
   - L1 actor term contains no explicit joint command (guard against silent concat regressions);
   - multi-motion latent concat aligns with `motion_end_idx` (Phase-3 pre-work).
2. **NPZ integrity** — reuse `_validate_augmented_reference`: standard fields byte-identical to
   the GMR reference, `latent_z` finite `(T,128)`, per-clip in Phase 3.
3. **Smoke runs** (~5 min each, 256 envs, 50 iterations, logger disabled) per new arm: assert
   registered actor/critic obs dims match §2, reward is finite and moving, checkpoint writes.
4. **Config cross-validation** — analyzer asserts per-arm `holosoma_config.yaml` actor/critic
   `func` strings and motion file, as Phase 0 does; extended with obs-dim log checks.
5. **Protocol freeze** — Phase-1 driver + analyzer sha256'd into the run directory before launch;
   endpoints and promotion rules in this document are frozen at Phase-1 launch time. Any
   post-launch edit requires a dated amendment note in this file.

---

## 5. Experiment protocol and decision tree

### 5.1 Phase 0 (running) — frozen screen, S1/S2/S3 vs B

Endpoints, promotion rule, and predictions are already frozen
(`docs/WBT_LATENT_INTEGRATION_STUDY.md`; predictions in `docs/WBT_LATENT_LITERATURE_REVIEW.md`
§6: S1 ≈ neutral/negative, S2 best chance, S3 ≈ null). One clip, one training seed —
a development screen, not a tracking claim.

### 5.2 Phase 1 — attribution and load-bearing test (C1 + L1, ~5 GPU-hours)

Launch regardless of Phase-0 outcome — both arms answer questions Phase 0 cannot:

- **C1 runs unconditionally**: if any preview arm (S2 or C1) helps, WBT benefits; the S2−C1
  contrast is the only way to attribute latent-specific value. C1 is also the cheapest
  potentially-publishable WBT improvement in the plan.
- **L1 runs unconditionally**: it is the C2 go/no-go signal. Even if S1/S2 are null (frozen
  auxiliary latents are the literature's weakest configuration), L1 tests the configuration the
  shared-policy vision actually needs.

**Endpoints (frozen now):** same three primaries as Phase 0 — 10-s completion, mean survival,
joint-position RMSE; same official metric set retained. Same promotion rule vs baseline
(completion floor −5 pp AND ≥5% relative improvement on RMSE or survival, or +5 pp completion).
**L1-specific floor:** L1 is *not* required to beat B — it promotes to Phase-2 replication if
completion ≥ 70% (i.e., tracks from latent alone at all; a policy that tracks from z within
~20 pp of the explicit-command baseline on first attempt is strong evidence for the latent
command direction).

**Pre-registered predictions (2026-07-15, before any Phase-0/1 unblinding):**

- C1 ≥ B, with the same or larger gain than S2 (GMT precedent).
- L1 materially below B on first attempt (UniTracker needed a CVAE prior + DAgger to make
  latent-only work; PPO-from-scratch on a frozen latent command is harder), but above the 70%
  floor for a walk clip. If L1 ≥ 85%, that is a strong positive surprise → fast-track Phase 3.

### 5.3 Phase 2 - COMPLETE (E39)

The registered replication design was:

1. **Seeds:** training seeds {1, 2} added (seed 0 exists), evaluation seeds {404, 405} per
   checkpoint → 3×2 reports per arm and for B (B already has seed-0; train B seeds {1, 2} too —
   without them, seed variance of the baseline is unknown and any per-seed comparison is
   uninterpretable). This is the expensive, non-negotiable part: the aux-representation
   literature's variance warning (Hu et al. 2304.04591) applies directly.
2. **Readout:** per-seed paired deltas (same eval windows), report mean ± range across seeds; an
   arm survives replication if the median seed's effect still meets the promotion threshold and
   no seed violates the completion floor.
3. **Clip transfer spot-check:** promoted arm + B on `fight1` (walk-adjacent budget per E35;
   dance2 excluded until per-clip budgets exist).

Observed result: S3's median training-seed effect misses the promotion threshold, while L1 clears
its absolute completion floor in every cell. The planned fight1 spot-check was not run because
E35 did not establish a valid fight1 budget. C3's seed-0 explicit-preview critic result matches
S3, removing a latent-specific interpretation of the critic signal. Full numbers and caveats are
recorded in E39 and `runs/wbt_latent_phase2/analysis.json`.

### 5.4 Phase 3 - multi-clip command generalization (draft; baseline calibration first)

Gate: at least one latent arm (S1/S2/L1) survived Phase 2, **or** all latent arms are null but
C1 replicated (in which case Phase 3 tests whether the latent adds value *on top of* explicit
preview in the multi-clip regime, where phase-aliasing across clips finally gives z a job).

The detailed protocol is in `docs/WBT_LATENT_PHASE3_PROTOCOL.md`. Its first step is now a
baseline-only 4k/8k/16k checkpoint calibration; the full arm matrix cannot launch until the
multi-clip control reaches its registered floor. This prevents another undertrained-control
verdict.

- One policy per arm trained on a 5–8 clip walk-family `motion_dir` set with per-clip latents;
  held-out-clip evaluation (2 clips never trained on) — the first setting where the literature
  actually predicts latent benefit (clip-identity/semantics disambiguation).
- E39-resolved arms: B-multi, S3-multi, L1-multi, C3-multi. L1 is the only positive
  latent-feasibility question; S3/C3 are a paired secondary attribution question.
- Primary endpoint shifts to **held-out-clip completion/RMSE** (generalization), with trained-clip
  metrics as secondary.

### 5.5 Phase 4 — escalation paths (design later; listed for direction only)

- **Load-bearing latent:** KL-aligned prior / residual conditioning (UniTracker/MaskedMimic
  pattern) instead of concatenation, if auxiliary latents stay null but L1 shows promise.
- **Shared multi-robot policy commanded by z + embodiment code** (C2 proper): requires WBT
  configs for a second robot; gated on Phase-3 L1-multi.
- **Latent feasibility probes (cheap, parallel, CPU-only):** (a) does SNMR latent reconstruction
  error predict per-clip WBT trackability (FLD's trick — testable *now* against E33/E35 reports);
  (b) are contact states linearly decodable from z (ties into Gate-1b mask work). Either result
  feeds the retargeter program independent of the RL arms.

### 5.6 Stop rules

- If, after Phase 2, no latent arm (S1/S2/L1) shows a replicated benefit **and** L1-multi in
  Phase 3 (if reached) is also null → the latent-observation direction is closed for the tracking
  claim; the latent's value case reverts to retargeting amortization + representation analysis,
  and WBT work continues on the C1 explicit-preview line if it replicated.
- Standing constraint: one GPU job at a time on the A10G; Gate-1b mask work has priority if a
  conflict arises (it is the nearer-term paper claim).

---

## 6. Compute budget summary

| Phase | Content | GPU-hours (A10G est.) |
|---|---|---|
| 0 (running) | S1, S2, S3 train+eval | ~7–8 |
| 1 | C1, L1 train+eval + smoke tests | ~5 |
| 2 | 2 extra seeds × (B + up to 2 promoted arms) + eval-seed 405 reruns + fight1 spot-check | ~15–20 |
| 3 | 4 multi-clip arms (~12–16k iters each, larger set) | ~25–35 |

Phases 0–2 fit in ~3 machine-days serialized; Phase 3 is a separate commitment gated on results.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Single-clip determinism nulls every arm | Pre-registered as literature-consistent; Phase 3 is the designed escape, not a post-hoc rescue |
| S2 gain actually from preview, not latent | C1 control is unconditional in Phase 1 |
| Input-width confound (154 vs 538) | C2 dimensionality-control arm held in reserve; empirical normalization already on |
| Baseline seed variance unknown | B seeds {1,2} trained in Phase 2 before any claim |
| dance2-class clips untrainable at 8k iters | Phase 3 restricted to walk-family until per-clip budget calibration exists |
| L1 fails for optimization (not information) reasons | L1 critic keeps explicit command (privileged); if L1 fails, Phase-4 prior-conditioning is the designed follow-up, not a retry of concatenation |
| Analyzer drift across phases | Same frozen validation core, arms parameterized; sha256 provenance per run dir |
