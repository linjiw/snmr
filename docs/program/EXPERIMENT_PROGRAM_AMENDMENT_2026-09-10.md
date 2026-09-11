# Experiment-program amendment (2026-09-10): corrections before scaling

Amends `EXPERIMENT_PROGRAM_2026-09-10.md` and the learning plan of
`METHOD_SPEC_TRANSFERABLE_CORRECTIONS_2026-09-10.md` §4–§5. The originals are left as written;
where they disagree with this file, this file governs. Nothing here launches a job.

## 1. The two documents describe different studies; one development study is chosen

The method spec trains on G1 plus variants and reserves T1 for a later real-model transfer.
The experiment program trains on T1/PM01/N1/Toddy with G1 held out. The correction-label
pipeline (search against a qualified tracker) exists for no robot but G1, so the second plan
cannot start. **The next milestone is the G1 development pilot** in
`reproducibility/reports/pilot_manifest_g1_locomotion_2026-09-10.json`: G1 is the development
robot, every inspected clip is development, and the split fields (proposal-training robots,
correction-label robots, tracker-training clips, verifier-qualification clips, final
evaluation clips) are named separately. A strict held-out fold later must exclude the target's
supervision from *every* inherited component; freezing a G1-trained proposal does not make G1
unseen, and "correction transfer to a robot seen during proposal pretraining" is the honest
label if that is what was done.

## 2. The crossed matrix (§4.2) is reinterpreted

For M[i, j] with training reference source i and evaluation source j, the causal comparison
is **M[full, j] − M[baseline, j] at fixed j** (same evaluation references, different
training references). A row mean averages one training source across evaluation sources; a
column mean averages one evaluation source across policies. **Row mean minus column mean is
not a causal measure of motion simplification** and is withdrawn as a reading. Source
simplification is measured directly on the references with the intent statistics (amplitude,
signed travel, events) and reported beside execution.

The 6 × 6 matrix and five robot folds are not prerequisites. The development comparison is
**two arms on a common evaluation panel**: geometry-only references vs full (corrected)
references, with configured GMR and simple repair as strong baselines, matched recipe and
budget.

## 3. Tracker-generalization leakage in the common panel (§4)

The program trains G1 trackers on three Panel-F clips and then includes all of Panel F in the
"common held-out evaluation panel". Those three clips are tracker-seen. The panel is split
into *tracker-seen* (reported as familiar-motion execution) and *tracker-unseen* (the only
part that can support a tracker-generalization claim). Retargeter-unseen and tracker-unseen
are different splits and are reported as such (the development-set manifest already labels
walk1_subject5 as tracker-seen / retargeter-unseen).

## 4. Horizon

A 500-step evaluation at 50 Hz is a 10-second window from each start, not full-clip success
for a 261-second clip. Endpoint 1 is renamed "10-s completion from a start grid"; a
full-clip endpoint needs its own definition (a single start at frame 0 with the full length,
or chained windows) and is not claimed by the current protocol.

## 5. GMR as a fixed common benchmark

Configured GMR-G1 is the field-standard reference source and is kept as the fixed comparator
for every arm. It is not automatically neutral: its references carry their own foot-skate,
jitter and IK conventions (the CPU ledger shows joint-space jitter that cancels at the end
effector). Every reference-source preference is therefore reported together with
source-intent outcomes on the same references, and "policy X prefers source Y" is interpreted
only with those in hand.

## 6. Statistics (§5)

- The three-seed σ_seed ≈ 0.076 from E70 is a *pilot* estimate for that task. It is not a
  universal variance for the downstream tracker-learning task; the A−T and A−S contrasts share
  arm A and are not independent replications. The pilot's own seeds set K.
- K = 3 is a small pilot. Report policy-seed heterogeneity and motion heterogeneity
  separately; run a sensitivity analysis on the seed-level and clip-level intervals; and never
  read CI overlap as equivalence.
- Per-clip acceptance in the correction pilot: paired bootstrap over within-clip start
  blocks, explicitly conditional on that clip. A clip-cluster bootstrap cannot estimate
  between-clip variance from one clip; across-motion claims aggregate distinct clips.
- Search and confirmation are disjoint start grids and seeds (manifest §search_vs_confirmation).

## 7. Conditioning ablations (§3 negative control, spec §4 shuffled-token control)

Keep the target's topology, joint masks, output-axis interpretation, joint limits, FK and the
analytic root prior **correct** in every arm, and ablate only the *learned numeric
conditioning* (zeroed or permuted kinematic/topology features under the fixed target
binding). Swapping a whole `RobotSpec` changes output width and axis interpretation and
confounds the test. A fixed deterministic shuffled spec can become a learnable surrogate
identifier; use per-batch permutation or feature zeroing with the binding intact. Lack of
degradation does not prove absence of transfer; information can arrive through another path
(the binding itself, the human encoder).

## 8. Scaling in N is secondary

Where the question is diversity, keep total supervision and compute controlled, vary subset
composition at fixed N, and keep the same target. A flat curve at N ≤ 4 can reflect
saturation, redundancy or low power and does not refute a positive held-out gain; the direct
held-out improvement is primary. H-Zero already studies training-set scaling, so a slope is
not a contribution on its own.

## 9. Learning only after the pilot, with labelled outcomes (spec §4–§5)

Distillation starts only after the pilot reports valid corrections that improve closed-loop
execution beyond no-op and simple repair on confirmation starts. When it starts:

- keep explicit robot references as the downstream interface; latent-only control is not
  an added requirement;
- label every interval: `accepted`, `known_invalid`, `search_not_found`,
  `verifier_unqualified`; a zero fallback is operationally conservative and is **not** a
  ground-truth negative label; calibrate any gating head against acceptance under the named
  search protocol and report coverage;
- establish learnability on unseen development clips *within* the source embodiment before
  any transfer fold; same-topology variants are a diagnostic, not a held-out real-humanoid
  result;
- target-specific search on the held-out robot runs only after the frozen zero-shot outputs
  and decisions are recorded; it is an adaptive upper-bound diagnostic with separate cost and
  its outputs never enter training or thresholds;
- report the four separations explicitly: search fails (no accepted solution under this
  operator/budget) · search succeeds but in-domain prediction fails (distillation/coverage) ·
  in-domain succeeds but held-out fails (generalization) · held-out references improve but
  downstream learning does not (reference executability improved; tracker-learning claim
  unsupported).

## 10. Stage 2 proxy

Open-loop PD survival remains a wiring diagnostic (repo record 2026-08-30). It is not the
inner candidate-ranking objective. A failed proxy-ranked search cannot establish that useful
corrections do not exist; a failed search of any kind is `search_not_found`, not
uncorrectable.
