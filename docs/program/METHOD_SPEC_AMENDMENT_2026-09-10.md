# Method-spec amendment (2026-09-10): withdrawn guarantees, tested contracts, revised filter

Amends `METHOD_SPEC_TRANSFERABLE_CORRECTIONS_2026-09-10.md` §1–§3. The original file is left
as written; this amendment governs where they disagree. Every statement below that is
labelled *tested* has a test in `tests/test_correction_basis.py` or
`tests/test_repair_intent.py` (28 tests, all passing on commit `f4411ae` + this working tree).
Code: `snmr/correction_basis.py`, `snmr/repair_intent.py`.

## 1. Two structural guarantees are withdrawn (counterexamples, tested)

**Withdrawn:** "No time channel → cannot slow down, cannot time-warp."
For the ramp q₀(t) = t/T and an admissible interior residual δ, q₀ + δ = q₀(φ(t)) with
φ(0) = 0, φ(T) = T and a local progression rate that ranges below 0.85 and above 1.15 on the
same frame grid (`test_no_time_channel_still_permits_local_time_warp`). Timestamp
preservation preserves the grid and the duration; it does not preserve local event timing.

**Withdrawn:** "Zero-clamped endpoints, no multiplicative parameter → cannot shrink."
For an interior basis function b, q₀ = 0.2 b and δ = −0.1 b halve the amplitude while staying
inside the 0.35 rad box with zero value, first and second derivative at both ends
(`test_additive_endpoint_clamped_residual_halves_amplitude`). Attenuation needs no
multiplicative parameter.

**Corrected:** "first two and last two control points pinned to zero → C²."
For a clamped cubic B-spline that pins value and first derivative only; the second derivative
at the join is generically nonzero (`test_pinned_control_points_are_C1_not_C2`). C² at a
zero join requires three pinned coefficients per end, implemented as an explicit
constraint nullspace (`EndpointConstraint(order=2)`), which consumes 6 of the K control
points (`test_clamped_C2_constraint_pins_three_coefficients_per_end`). The free
dimensionality is therefore ⌈L/8⌉ + 3 − 6 per channel, not ⌈L/8⌉ − 1.

**Corrected:** "0.16 s knots bandlimit η to ≈ 3 Hz."
Alternating control points on the same knot grid put > 80 % of their energy above 3 Hz at
50 fps (`test_knot_spacing_is_not_a_frequency_cutoff`). Knot spacing is a smoothness scale.
Note also that the pair NPZs are 30 fps and the tracker references are 50 fps: "8 frames" is
0.267 s on one grid and 0.16 s on the other; the spec must name the grid.

Consequence for §1's closing paragraph: **none** of the three named degenerate solutions
(shrink, slow down, delete a transition) is removed by the parameterization. All three must
be caught by the filter, and the filter is therefore part of the method.

## 2. What the parameterization does guarantee (tested)

1. Fixed sample grid and duration: a correction is a function on the interval's own frame
   indices; nothing is resampled (`test_fixed_grid_duration_and_bounds`).
2. Bounded edits: control points inside the per-channel box imply the realized per-frame
   edit is inside the box (convex hull).
3. Endpoint continuity of the declared order, preserved by box projection (coefficient
   clipping or radial scaling, both verified) and by additive zero-padded stitching
   (`test_projection_methods_keep_constraints_and_box`,
   `test_continuity_survives_projection_and_stitching`). Frame-level clipping of the realized
   trajectory breaks C¹ and is not used (`test_frame_level_clipping_breaks_C1`).
4. Joint limits enforced by uniform shrinking, never by clipping; a baseline that already
   violates limits yields `baseline_invalid`, i.e. an abstention
   (`test_joint_limits_enforced_by_scaling_not_clipping`,
   `test_invalid_baseline_yields_abstention_not_a_repair`).
5. Satisfaction of the named, tested intent checks of §3 below.

Nothing here is a semantic or dynamic-feasibility guarantee.

## 3. Revised intent filter (replaces §2 Blocks A–D)

Implemented in `snmr/repair_intent.py`; adversaries in `tests/test_repair_intent.py`.

| Check | What it preserves / bounds | Old filter's blind spot it closes |
| --- | --- | --- |
| `amplitude`, `energy` | per-anchor ratio bands with an absolute floor for static anchors | near-zero denominators were undefined |
| `jitter_not_increased` | one-sided: reductions always allowed | Block B demanded ≥ 95 % of the proposal's jitter, rejecting smoothing (`test_jitter_removal_is_accepted…`) |
| `signed_travel` | net displacement direction (cosine) and magnitude | a mirrored oscillation passed amplitude/energy/jitter (`test_direction_reversal…`) |
| `local_direction_speed` | windowed signed displacement cosine and speed ratio | global statistics are window means |
| `events` | per reference event: presence, ±4-frame timing, magnitude, direction; no added events; events defined on the human source (`event_source`) | ±8-frame peak search did not fix direction or count |
| `contact_sequence` | per-foot stance-run count, order, onset/offset shifts | a deleted foot-strike (`test_removed_foot_strike…`) |
| global cross-correlation lag | **diagnostic only** | zero lag with a local phase shift (`test_local_phase_shift_has_zero_global_lag_but_fails_events`) |

Statuses are explicit: each check is `passed` / `failed` / `not_evaluable`; the interval
decision is `accept_candidate` / `fallback_baseline` / `abstain`, and coverage is reported over
all three with every interval in the denominator (`coverage()`).

Tolerances (`IntentTolerances`) are hash-bound and carry `calibration_status`. The shipped
values are **provisional** and are not to be used as a frozen gate. Calibration on the
development set (`reproducibility/reports/locomotion_dev_set_2026-09-10.json`) must precede
any reported use, keep the historical A1 T5 failures as regression cases, and be frozen with
its own hash. The A1 envelope `[0.50, 1.50]` is not imported as truth.

Any metric used to select candidates is part of the method. Independent evaluation needs its
own measures (closed-loop execution on separate confirmation starts, and where available
annotated contact intervals); the filter's own statistics are not a held-out evaluation.

## 4. Consequences for §3 (acceptance rule)

- Stage 2 as written re-introduces open-loop PD survival as the inner ranking objective. The
  repository record rejects it as a ranker (`docs/MORPHORETARGET_FOUNDATION_2026-08-30.md`
  "Negative result: open-loop PD is not a verifiable-RL reward";
  `docs/MORPHORETARGET_PLAN_2026-08-29.md:181`). It remains a wiring diagnostic. Candidates
  are scored by the qualified closed-loop tracker; a cheap proxy may be *compared* against
  those rankings, not substituted for them.
- Acceptance thresholds must be coherent with headroom: +0.10 completion is impossible above
  a 0.90 baseline. On the E70 explicit verifier, walk1_subject1 sits at 0.98 and
  walk1_subject5 at 0.86–0.87 (seeds 0–2), so a fixed +0.10 rule excludes the former by
  arithmetic. The pilot manifest defines headroom per clip before any candidate is scored.
- Search and confirmation use disjoint start grids and seeds; a per-clip bootstrap over
  within-clip start blocks is conditional on that clip; between-clip claims need distinct
  clips.
- "Never accept per-rollout" stands.

## 5. Consequences for §6 (modules)

- `scripts/eval_wbt_multimotion.py` is not needed: `scripts/train_e52_dagger.py` already
  evaluates multi-motion students with equal-count per-clip start grids, per-motion teacher
  routing and per-clip reporting (`E52_EVAL_ONLY=1`). Deleting the `num_motions != 1` guard in
  `eval_e67_teacher.py` would not have qualified anything.
- `correction_basis.py` and `repair_intent.py` exist; `correction_search.py`,
  `correction_head.py` and the trainers do not, and are not to be built before the pilot in
  `reproducibility/reports/pilot_manifest_g1_locomotion_2026-09-10.json` reports whether valid
  corrections improve closed-loop execution beyond no-op and simple repair.
