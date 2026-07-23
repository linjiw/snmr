# E50 — Physics-Repaired Teacher (rollout laundering)

**Date registered:** 2026-07-23. **Status:** REGISTERED, not yet run.
**Motivation:** `docs/REACTOR_GMR_HOLOSOMA_ANALYSIS.md` §5.1/§6-P1. ReActor (arXiv:2605.06593)
proves constructively that *the simulated rollout of a competent tracking policy is the
physically-consistent version of its reference* (zero penetration, foot slide at contact-solver
noise, by construction). We already train such policies locally (B1: SNMR-ref 86–91% completion
on walk1, MuJoCo-Warp WBT). E50 tests whether rolling out our policies on teacher references
and re-exporting the *simulated states* yields repaired supervision that (a) has the physical
properties Gate 1 failed to reach with every deployable corrector, and (b) preserves enough
kinematic fidelity to retrain SNMR on.

This is the only path to a C5-positive that the Gate-1 postmortem left open
("physics-repaired supervision or retargeting/tracking co-optimization"), and it routes around
the closed deployable-contact-mask bottleneck entirely: physics supplies the contact signal,
no mask is estimated anywhere.

## 1. Hypotheses (falsifiable)

- **H-A (laundering premise):** a B1-grade policy rollout on the walk1 reference has
  (i) stance-foot speed ≤ 0.5× the GMR teacher's on the same clip, (ii) zero ground
  penetration beyond solver tolerance, while (iii) staying kinematically close to the
  reference (rollout-vs-reference MPJPE ≤ 5 cm over completed segments).
- **H-B (repair transfers through distillation):** fine-tuning SNMR on repaired pairs moves
  *deployable* foot-skate toward the physical level — endpoint: stance-foot speed ≤ 0.08 m/s
  (the failed Gate-1 endpoint) on held-out evaluation, with MPJPE guard ≤ +0.5 cm vs the
  matched unrepaired baseline.
- **H-C (coverage):** the repair mechanism scales beyond walk1 — a multi-clip tracking policy
  (or per-clip policies within budget) yields accepted repaired segments for ≥ 60% of the
  77-clip LAFAN1 set at completion ≥ 0.8 per clip. (This is the ReActor residual-wrench
  question: they needed a root wrench for full-AMASS coverage; walk-class LAFAN1 may not.)

## 2. Stages, gates, and kill criteria

**Stage A — premise check (walk1 only; needs existing artifacts + one GPU-hour).**
Roll out the best B1-confirmatory policies (per source: GMR seed0 = 0.90 completion,
SNMR seed2 = 0.91) on their own walk1 references. Record simulated states per step with an
extended recording callback (see §3). Score with `snmr/metrics.py::compute_metrics` using the
**simulator contact mask** (feet contact forces are ground truth here — no estimated mask):

| readout | gate |
|---|---|
| rollout stance-foot speed vs teacher (same metric, same clip) | ≤ 0.5× teacher → H-A(i) PASS |
| ground penetration (frames > 1 cm) | ≈ 0 → H-A(ii) PASS |
| rollout-vs-reference MPJPE on completed segments | ≤ 5 cm → H-A(iii) PASS |
| completed-segment coverage of the clip | ≥ 80% report only |

*Kill:* if H-A(i) fails (rollout skates as much as the teacher), the laundering premise is
false at our policy quality — STOP, log negative, revisit only after tracking improves.
*Caution flag (not kill):* if H-A(iii) fails but (i)+(ii) pass, physics repair distorts
content at our policy quality; consider tightening the eval-time termination thresholds or
using only low-error segments before proceeding to Stage B with reduced scope.

**Stage B — distillation of repaired pairs (walk1-class pilot; ~1 GPU-day).**
Export repaired pairs (§4) for the clips accepted in Stage A/C. Fine-tune the reference SNMR
checkpoint (large config) on repaired pairs with contact-BCE weight 0.25 (E48-100k registered
default; the simulator contact mask now supplies *true* labels, upgrading E48's supervision).
Matched control: identical fine-tune schedule on the *unrepaired* teacher pairs (isolates the
repair effect from the extra training). Primary endpoint per H-B on held-out val clips.
*Kill:* repaired-arm skate not better than control skate by ≥ 30% at matched MPJPE → repair
does not transfer through distillation; log negative (still publishable against C5).

**Stage C — coverage scale-up (gated on A; ~1–2 GPU-days).**
Train one multi-clip WBT policy per source on a stratified 10-clip LAFAN1 subset (walk/run/
dance mix; MultiMotionLoader + adaptive sampler, 8k–16k iters — calibrate on the GMR arm
first per E35's lesson), then roll out on each clip. Accept clips per H-C.
*Fallback (P1b, separate registration):* if coverage < 60%, evaluate a ReActor-style residual
root wrench as a *data-generation-only* action term before abandoning scale-up. Not part of E50.

## 3. Mechanics — rollout recording (no edits to the pinned clone)

- holosoma's `EvalRecordingCallback` (`agents/callbacks/recording.py`) already records
  per-step `dof_pos`, `root_pos`, `root_quat_xyzw`, `body_pos_w`, torques for one env at the
  50 Hz policy rate, and is instantiated from the `_target_` string
  `holosoma.agents.callbacks.recording.EvalRecordingCallback`
  (`config_types/eval_callback.py:28`).
- SNMR-side extension `snmr/integration/wbt_repair.py`: subclass adding channels needed for
  segmentation and alignment — `dones` (from `actor_state["dones"]`, as consumed by
  `wbt_metrics.py`), motion `time_steps` (from
  `env.command_manager.get_state("motion_command").time_steps`), reference `joint_pos` at the
  commanded frame, and per-foot contact forces (simulator contact mask source). Installed by
  monkeypatching the module attribute before config instantiation (same idempotent-patch
  pattern as `wbt_latent.py`); `_target_` resolution picks up the subclass.
- Eval invocation: `eval_agent.py` with `--recording.config.enabled`, `num_envs=1`
  (deterministic `act_inference`; eval resets start at frame 0), init-pose noise zeroed,
  `max-eval-steps` = clip length, `max-episode-length-s` effectively unbounded. On failure the
  env resets to frame 0; the `dones`/`time_steps` channels segment the recording so repeated
  attempts contribute only their completed segments.

## 4. Mechanics — repaired-pair export

`scripts/export_e50_repaired_pairs.py`:
1. Load recording npz + the original pair npz (`data/pairs/<robot>/<clip>.npz`: human_pos/
   human_quat @30 fps, teacher qpos @30 fps).
2. Segment rollout by `dones`; within each segment, alignment is `time_steps[t]` → reference
   frame (50 Hz policy rate vs 30 fps pairs: resample rollout to the pair timeline with lerp
   on dof/root pos + slerp on root quat, standard wxyz internally — GMR pkl xyzw only at that
   boundary, N/A here since recording emits xyzw explicitly named).
3. Gate segments: drop the first 0.5 s after any reset (settle-in), drop segments whose
   per-frame joint RMSE vs reference exceeds the eval termination thresholds.
4. Write repaired pair npz alongside originals (`data/pairs_e50/<robot>/<clip>.npz`), human
   side byte-identical, `qpos` = simulated, plus `contact_mask` (from recorded foot contact
   forces, threshold 1 N — holosoma's undesired-contact convention) and a `segments` index.
   Provenance: policy checkpoint sha256, holosoma/snmr revisions, source pair sha256.
5. Emit a metrics sidecar (teacher-vs-repaired skate/penetration/MPJPE per clip) — this file
   IS the Stage-A readout.

## 5. Why this is the right next move (and its honest risks)

- **Cost:** reuses trained B1 policies; Stage A is about an hour of GPU. Full bilevel
  (ReActor proper) is ~6 h × 4096 IsaacSim envs *per robot* — outside our single-A10G budget.
- **Risk 1 — laundering ≠ feasibility:** even without a residual wrench our policies were
  trained with push/friction DR; the rollout is feasible *for the sim robot under those
  conditions*, which is exactly what downstream WBT consumes. Acceptable.
- **Risk 2 — B1 joint RMSE is ~0.25 rad:** the rollout may deviate visibly from the
  reference (H-A(iii) may fail). That is a *finding*, not a bug — it bounds how much fidelity
  physics-repair costs at our tracking quality, the same trade ReActor tunes with its
  force-penalty weight.
- **Risk 3 — single-clip policies:** Stage A's policies saw only walk1; coverage claims
  (H-C) need Stage C's multi-clip policy. Preregistered as separate gates so a Stage-A pass
  cannot be over-read.

## 6. Relation to the live program

E50 does not touch the E49/H2 act-through-latent line (different GPU windows; E49 runs
first). If both succeed they compose: repaired references improve the *data*, act-through-
latent improves the *interface* — jointly the P2 "unified co-training" framing of the
analysis memo. If E49 nulls, E50 is unaffected (it acts on supervision, not on the command
path). Results append to `docs/EXPERIMENT_LOG.md` as E50-A/B/C.
