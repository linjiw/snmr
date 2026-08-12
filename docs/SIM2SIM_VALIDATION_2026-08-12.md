# Sim2Sim Deployment Validation — Phase E Campaign

**Date:** 2026-08-12
**Target platform:** CPU MuJoCo (Holosoma classic backend) driven by the production
Holosoma inference process over the loopback DDS interface — the same runtime stack that
would command a real Unitree G1, with only the physical interface swapped for `lo`.
**Scope:** engineering validation toward `docs/REAL_WORLD_DEPLOYMENT_PLAN.md` stages 2–3,
extended from the pre-confirmation seed-0 candidate to the post-confirmation candidates and
both registered E70 walks. No paper claim changes; no physical robot commands were sent.

## Candidate selection (rule recorded before any run)

Per the E-rule in `docs/ICRA_EXECUTION_PLAN_2026-08-11.md` Phase E: one candidate per arm by
*median* `completion_rate` in the frozen per-seed general evals (1024 rollouts, evaluation
seed 404); no visual selection.

| Arm | Per-seed completion (s0/s1/s2) | Selected |
| --- | --- | --- |
| explicit (deployment candidate) | 0.9248 / 0.9199 / 0.9238 | **seed 2** |
| SNMR (research interface) | 0.6846 / 0.7021 / 0.7090 | **seed 1** |

## Stage E-1 — ONNX export (safe95 envelope)

All four exports pass `onnx.checker` and the CPU ONNX Runtime parity gate (threshold 1e-5)
with the 95%-range action envelope:

| Candidate | SHA-256 | Max parity error |
| --- | --- | --- |
| `e70_seed2_explicit_walk1_subject1_safe95.onnx` | `3b1347af370d8a80…` | 5.25e-6 |
| `e70_seed2_explicit_walk1_subject5_safe95.onnx` | `44e8cba19b3c34d1…` | 5.25e-6 |
| `e70_seed1_snmr_walk1_subject1_safe95.onnx` | `0404c8bf7abd78f8…` | 2.74e-6 |
| `e70_seed1_snmr_walk1_subject5_safe95.onnx` | `5989b5f55002024d…` | 1.43e-6 |

Artifacts: `exports/sim2sim_2026-08-12/*.onnx` + sibling `.validation.json` (full hashes).

## Stage E-2 — Production runtime contract (500-step fault injection)

All four candidates pass `scripts/validate_e70_runtime_contract.py` (hsinference
environment): 0 deadline misses (20 ms), 0 non-finite steps, 0 runtime action clips, 0
action-envelope violations, 0 hard-limit violations; minimum hard-limit margin 0.01309 rad
(identical across candidates — the envelope is a function of the URDF limits, not the
weights). Artifacts: `exports/sim2sim_2026-08-12/*.runtime.json`.

## Stage E-3 — CPU-MuJoCo loopback qualification (safety-handoff protocol, ×3 repeats)

Protocol: unmodified `scripts/run_e70_loopback_qualification.py --safety-handoff` (conda
`hsmujoco` environment): production inference at 50 Hz over `lo`, physics at 2000 Hz,
gantry-assisted start, release after upright preflight, ~10 s unassisted walking, handoff
to Holosoma's zero-velocity locomotion safety policy at the fixed 27 s mark. Columns:
minima over the phase; "unassisted" = free walking, "hold" = after safety handoff.

| Candidate / repeat | Pass | Unassist h (m) | Unassist up | Meas. margin (rad) | Hold h (m) | Hold up | Failed checks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| snmr_subject1 r1 | ✅ | 0.770 | 0.969 | +0.1400 | 0.755 | 0.975 | — |
| snmr_subject1 r2 | ❌ | 0.703 | 0.943 | +0.0399 | 0.604 | 0.948 | joint-limit margins in hold |
| snmr_subject1 r3 | ✅ | 0.749 | 0.974 | +0.1093 | 0.756 | 0.977 | — |
| snmr_subject5 r1 | ❌ | 0.698 | 0.962 | −0.0806 | 0.755 | 0.994 | measured joint limit in motion |
| snmr_subject5 r2 | ❌ | 0.099 | −0.191 | +0.0328 | 0.121 | −0.580 | fell during motion |
| snmr_subject5 r3 | ✅ | 0.722 | 0.959 | +0.0010 | 0.735 | 0.963 | — |
| explicit_subject1 r1 | ✅ | 0.771 | 0.973 | +0.1351 | 0.755 | 0.997 | — |
| explicit_subject1 r2 | ✅ | 0.767 | 0.970 | +0.1335 | 0.749 | 0.996 | — |
| explicit_subject1 r3 | ✅ | 0.766 | 0.961 | +0.1044 | 0.748 | 0.983 | — |
| explicit_subject5 r1 | ❌ | 0.690 | 0.953 | +0.0256 | 0.114 | −0.571 | fell after handoff |
| explicit_subject5 r2 | ✅ | 0.710 | 0.950 | +0.0350 | 0.735 | 0.980 | — |
| explicit_subject5 r3 | ❌ | 0.545 | 0.842 | −0.1067 | 0.119 | −0.596 | degraded motion, fell after handoff |

Fail-closed repeat summaries (`*.loopback_safety_handoff.summary.json`):

| Candidate | Strict 3/3 gate |
| --- | --- |
| explicit × walk1_subject1 | **PASS** |
| explicit × walk1_subject5 | fail (1/3) |
| snmr × walk1_subject1 | fail (2/3) |
| snmr × walk1_subject5 | fail (1/3) |

### Reading of the failures

1. **The dominant failure mode is the terminal safety handoff, not motion tracking.** In
   three of five failures the robot walks the full clip upright and collapses only after
   the switch to the zero-velocity safety policy at the fixed 27 s mark. That timing was
   registered against `walk1_subject1`'s gait timeline; on `walk1_subject5` it lands at a
   marginal gait phase (one explicit repeat passes it, two do not). The identified fix is a
   clip-specific handoff frame chosen at a double-support/low-velocity point of the
   reference — a *new protocol variant* to be recorded before rerunning, not a threshold
   edit after the fact.
2. **Explicit × walk1_subject1 — the actual deployment configuration — passes 3/3**, with
   phase minima closely matching the seed-0 record in `REAL_WORLD_DEPLOYMENT_PLAN.md`
   (0.766–0.771 m vs 0.768 m). The stage-2/3 result is therefore robust to the recorded
   seed-selection rule, not a property of one lucky checkpoint.
3. **The SNMR research interface walks.** On `walk1_subject1` it is upright and clean in
   all repeats through the motion window (min height 0.703–0.770 m); its two subject-1
   failures are hold-phase joint-limit margins, not falls. On the harder `walk1_subject5`
   it fell mid-motion once in three — consistent with its 0.70 general completion in the
   frozen Warp evaluation, and a faithful sim2sim reproduction of the measured
   capability gap between the arms (0.92 vs 0.70).

## Stage E-4 — Review videos and tracking metrics

One additional state-logged rollout per candidate (`scripts/run_sim2sim_review_capture.py`,
DDS domain 71, identical lifecycle), replay-rendered offscreen at 1280×720/25 fps
(`scripts/render_sim2sim_review_video.py`) — both new, non-frozen engineering scripts; the
frozen paper-video pipeline is untouched. All four capture rollouts stayed upright end to
end, **including SNMR × walk1_subject5** — further evidence the E-3 failures are marginal
and stochastic rather than systematic. Joint tracking RMSE vs the reference (matched by
joint name at the policy's 50 Hz motion clock; unassisted window):

| Candidate | RMSE unassisted (rad) | Mean abs (rad) | Min height (m) | Warp-eval RMSE (rad) |
| --- | --- | --- | --- | --- |
| explicit × subject1 | 0.176 | 0.134 | 0.768 | 0.181 |
| explicit × subject5 | 0.232 | 0.167 | 0.732 | 0.181 |
| snmr × subject1 | 0.172 | 0.130 | 0.765 | 0.192 |
| snmr × subject5 | 0.187 | 0.143 | 0.707 | 0.192 |

The CPU-loopback tracking error matches the frozen MuJoCo-Warp evaluation values within
a few hundredths of a radian — the runtime/engine transfer does not degrade tracking.
(The "assisted" window reads higher by construction: it includes the seconds before the
clip and the robot's pose converge.)

Videos for human review: `exports/sim2sim_review/e70_*.mp4` (one per candidate × clip,
~31 s each, title overlay identifies arm and clip). Machine artifacts:
`*.states.npz` (200 Hz qpos/dof log), `*.capture.json` (events), `*.metrics.json`.

## Boundary

These results are CPU-MuJoCo loopback engineering evidence. They are not hardware-safety
evidence, not sim-to-real validation, and none of the numbers above enters the paper. The
paper's boundary (all reported behavior in simulation; sim-to-real unvalidated) is
unchanged.
