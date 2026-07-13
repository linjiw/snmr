# SNMR Research Audit and Decision-Gated Next-Step Plan

- **Audit date:** 2026-07-13
- **Scope:** `NEURAL_RETARGETING_DESIGN.md`, implementation, experiment records, available run
  artifacts, and literature available through the audit date.
- **Audit constraint:** The initial findings were produced read-only. Later implementation and
  execution updates are marked explicitly.
- **Gate 0 implementation checkpoint:** `3d18005` (`research: add Gate 0 provenance and
  diagnostics`).

## Executive verdict

SNMR has a defensible core result: one compact model amortizes a GMR teacher across five *trained*
humanoid embodiments, respects configured joint limits, and learns representations that are strongly
aligned for paired motion. The most interesting current results are mixed rather than uniformly
positive: the latent is aligned but not invariant, the shared model currently loses fidelity to a
specialist, and unseen-robot decoding fails.

The project does **not yet support** claims of contact-consistent output, improved downstream
tracking, positive multi-robot transfer, zero-shot new-robot generalization, or embodiment
invariance. The exact `12x` throughput claim also needs a controlled benchmark.

The immediate decision is:

1. Repair experiment provenance and measurement before launching another contact sweep.
2. Run the baseline SNMR-vs-GMR WBT study now, without waiting for a contact breakthrough.
3. Factor contact classification, stance velocity, teacher-velocity matching, and constrained
   projection into separate hypotheses.
4. If normalized soft objectives and a real constrained projection both fail, stop tuning contact
   weights and move to physics-repaired supervision or retargeting/tracking co-optimization.

The supplied six findings are directionally correct. Finding 1 needs one important correction:
available code history and checkpoint contents indicate that E10a already included an EDGE-style
contact head, self-consistency loss, and BCE, despite the experiment log saying otherwise. E10b is
therefore a test of a different *combined* objective, not the first clean EDGE-head test.

## Prioritized findings

### P0. Contact is the main kinematic blocker, but E10 does not isolate the stated hypothesis

**Observed result.** The completed shared-model sweep made contact-gated foot speed worse:

| Run | Contact weight | SNMR skate | Teacher skate |
|---|---:|---:|---:|
| E10a | 0.05 | 0.574 m/s | 0.0517 m/s |
| E10a | 0.20 | 0.497 m/s | 0.0517 m/s |
| E10a | 1.00 | 0.541 m/s | 0.0517 m/s |

Sources: `runs/e10_contact_w*/benchmark.json`,
`docs/EXPERIMENT_LOG.md:153-168`, and `runs/ablations/SUMMARY.md:3-10`.

This establishes that the **implemented, bundled E10 objective failed**. It does not establish that
contact supervision as a class of methods is ineffective.

There are four experiment-validity problems:

1. **The E10 provenance narrative conflicts with the artifacts.**
   `docs/EXPERIMENT_LOG.md:153-158` calls E10a "NO EDGE head." However,
   `scripts/train_phase2.py:170-175,206-208,267-275` enables a contact head whenever
   `contact_weight > 0` and applies self-consistency plus BCE. Git history shows this wiring predates
   E10a, and `runs/e10_contact_w0.2/ckpt.pt` contains `decoder.contact_head.*` parameters. The exact
   run SHA and objective must be reconstructed before E10 is used as evidence.

2. **The movement penalty is badly scaled relative to BCE and other losses.**
   `foot_contact_loss` and `contact_self_consistency_loss` use squared frame displacement, not
   velocity in m/s (`snmr/losses.py:79-103,147-179`). At 30 Hz, even 0.3 m/s is only 0.01 m per
   frame, or about `1e-4 m^2` before averaging. The mean also includes masked zeros from every
   non-contact frame, foot, and coordinate instead of dividing by the number of contact samples.
   In a combined objective, BCE is normally orders of magnitude larger. A single
   `contact_weight` therefore does not define how much optimization pressure reaches foot motion.

3. **The bundled objective prevents attribution.**
   E10a appears to combine contact classification and predicted-contact self-consistency.
   E10b additionally combines teacher-mask velocity, self-consistency, and BCE
   (`scripts/train_phase1.py:221-246`). A better or worse result cannot identify which component
   caused it.

4. **E25 does not specifically target stance.**
   `foot_velocity_distill_loss` matches 3D teacher displacement over every swing and stance frame
   (`snmr/losses.py:106-130`). Swing motion can dominate its samples and gradient. It should be
   compared with contact-weighted and phase-balanced variants, not treated as a definitive stance
   intervention.

**Current artifact state.** At audit time no training process was active. E10b `w=0.5` has only a
5k log point at 19.3 cm MPJPE; `w=2.0` is empty. E25 `w=2.0` has logged through 65k and has a
checkpoint, but no final evaluation; `w=10.0` is absent. These are incomplete experiments, not
running or negative results.

**Decision.** Do not resume the current jobs until objective scales, gradient contributions, and the
E10 manifest are known. Then run the factorized contact matrix in Gate 1.

### P0. The tracking-benefit thesis is still untested

The design correctly calls WBT decisive (`NEURAL_RETARGETING_DESIGN.md:276-281`). Existing
evidence is:

- WBT files pass schema-level checks.
- Open-loop PD replay gives 0.87 s vs 0.82 s survival and 0.212 vs 0.210 rad DOF error.
- The differences are within window variation and support only "not detectably worse under this
  proxy" (`docs/EXPERIMENT_LOG.md:70-78`).

No matched closed-loop policy training result exists. In addition,
`runs/wbt_validation/WBT_COMMANDS.md:1-19` still documents IsaacSim even though the E20 scout says
the local MuJoCo/Warp path is feasible (`docs/EXPERIMENT_LOG.md:80-93`). Three exported clips and
one training seed would be a pilot, not evidence of benefit.

GMR evaluates 21 trajectories with separate policies and 100 sim, 4096 sim-DR, and 100 sim2sim
evaluation rollouts per policy. SNMR need not duplicate every count initially, but its confirmatory
study needs multiple clips and training seeds under identical settings.

**Decision.** Start a MuJoCo/Warp smoke test immediately, then run the paired baseline WBT study in
parallel with contact work. Make non-inferiority the first hypothesis; claim superiority only if a
predefined paired confidence interval supports it.

### P1. Zero-shot target-embodiment generalization fails, and the conditioning is too weak to expect it

LORO PM01 is 29.59 cm versus 5.72 cm in training, a 5.2x degradation
(`docs/EXPERIMENT_LOG.md:42-46`). This is a clear negative result, but one holdout is not enough to
characterize the failure distribution.

The model's embodiment code is a node-wise MLP followed by global max pooling
(`snmr/model.py:194-208`). Its static input contains only rest offset, joint axis, has-DOF, and
end-effector flag (`snmr/data.py:165-181`). It omits joint-limit values, mass, inertia, actuator
limits, contact geometry, and semantic body identity. The decoder does receive graph adjacency and
uses limits at its output, but its global conditioning code is a lossy bag of local features.

Synthetic scaled G1 variants are useful for local morphology interpolation. They do not by
themselves test generalization to a new topology, DOF layout, actuator envelope, or contact geometry.

`--zr_decode_prob` addresses a different problem: the decoder's exposure to robot-source latents
(`scripts/train_phase2.py:246-262`). It can address C7 robot-to-robot transfer, but it cannot teach
the decoder how to produce an unseen target robot.

**Decision.** Separate unseen-target decoding from robot-source latent transfer. Treat few-shot
robot adapters as the realistic near-term target, and retain zero-shot as a longer-term test.

### P1. Shared training has a likely fidelity cost, but the exact 1.4 cm estimate is not fully controlled

The available comparison is specialist 3.75 cm versus shared G1 5.12 cm
(`docs/EXPERIMENT_LOG.md:48-53`; `NEURAL_RETARGETING_DESIGN.md:420-427`). This is strong evidence
against claiming positive transfer.

It is not completely "confound-free":

- The shared run gives each robot about 40k effective sampled exposures (`100k * 2/5`), while the
  specialist receives 50k.
- Learning-rate trajectories are indexed by global optimizer step, not per-robot exposure.
- Four-window and 16-window evaluations must not be mixed
  (`docs/EXPERIMENT_LOG.md:3-6`).

Raw capacity is also not the obvious sole cause. At the matched 50k specialist schedule, the 0.41M
small model and 1.6M base model are nearly tied at 4.78 and 4.71 cm
(`runs/ablations/SUMMARY.md:5-8`). Gradient interference, conditioning limits, unequal task
difficulty, or exposure can explain the shared gap.

**Decision.** Measure per-robot gradient norms and cosine conflicts first. Then compare width,
lightweight per-robot adapters, and gradient balancing as separate hypotheses under exactly matched
robot exposures and evaluation windows.

### P1. The latent is aligned and useful, not embodiment-invariant

The strongest current description is:

- Linear CKA: 0.910.
- Exact paired-window human-to-robot retrieval R@1: 0.749.
- Linear embodiment probe: 0.278 at 0.167 chance.
- Post-hoc MLP attacker: 0.909.

Source: `docs/EXPERIMENT_LOG.md:29-36`.

High CKA and paired retrieval show that same-motion information is aligned across encoders. The MLP
attacker shows that embodiment identity remains highly recoverable. These results are compatible,
not contradictory.

The analysis needs stronger controls:

- Retrieval assumes gallery item `i` is the exact synchronized counterpart of query `i`
  (`scripts/analyze_latent.py:163-175`). Timing and clip identity can make this easy.
- CKA on aligned windows can be high because all embodiments share motion content.
- t-SNE/UMAP overlap is exploratory, not proof of invariance.
- The reported multiclass "proxy-A-distance" is a custom normalization, while the standard
  quantity is defined for binary domain discrimination
  (`scripts/analyze_latent.py:145-150`).

**Decision.** Publish "aligned, not invariant." Add controls before adding a domain adversary, and
report any invariance intervention as a Pareto tradeoff against MPJPE, retrieval, and transfer.

### P1. The temporal ablation does not reject temporal modeling

Removing the current mixer improves MPJPE from 4.71 to 3.95 cm and skate from 0.386 to 0.358 m/s,
while worsening DOF jerk from 614 to 720 (`runs/ablations/SUMMARY.md:5-6`). That weakens the claim
that the current module suppresses harmful jitter.

The implementation uses `nn.TransformerEncoder` without positional or relative-time information
(`snmr/model.py:164-190`). Self-attention without recurrence, convolution, or positional encoding
cannot explicitly represent sequence order. It can still mix frames by content and may smooth
similar poses, which is consistent with lower jerk and worse pose fidelity.

The Transformer paper explicitly adds positional encoding because attention alone has no order
signal. Therefore E08 tests "content-only global attention" versus framewise encoding, not a
properly order-aware temporal model.

**Decision.** Keep framewise as the fidelity default. Compare it with a position-aware transformer
and a temporal convolutional network before making a broad temporal-modeling claim.

### P1. The negative foot-lock result does not test a hard constrained solver

`snmr/footlock.py:1-17` describes damped Gauss-Newton, but the implementation computes the gradient
of a scalar squared error, normalizes that gradient, and takes a scaled gradient step
(`snmr/footlock.py:75-86`). It does not form a foot Jacobian or solve a damped least-squares system.

It also:

- detects contact from decoded motion (`snmr/footlock.py:58-68`), while E24 shows decoded contact
  under-fires at 0.33 versus teacher 0.74;
- holds root pose fixed, which can make single- and double-support targets infeasible;
- edits frames independently;
- does not jointly enforce limits, smoothness, or bounded deviation from the network output.

Its 0.160 to 0.153 m/s result rules out this local gradient heuristic. It does not rule out
windowed constrained IK, SQP/QP projection, or latent-space refinement.

### P2. The current contact metrics contain a circular teacher comparison

`detect_contact` declares contact only when teacher speed is below 0.3 m/s
(`snmr/metrics.py:70-86`). The default slide event is also speed above 0.3 m/s
(`snmr/metrics.py:89-98,124-133`). Teacher slide fraction is therefore zero by construction, as
seen in `runs/ablations/SUMMARY.md:11`.

Mean candidate speed under a teacher mask remains useful. However:

- teacher slide fraction should not be presented as an independent result;
- contact depends partly on the same velocity being evaluated;
- height is relative to each clip's own minimum, which can hide global floating or ground offset.

Use multiple stance definitions: source/human contact transfer, height-only contact with hysteresis,
teacher mask, and simulator contact/normal force where available. Report sensitivity across
thresholds and keep absolute ground-height metrics.

### P2. The scale-leak causal conclusion is too strong

The scale probe divides position and velocity features by height only at evaluation time
(`scripts/probe_scale_leak.py:52-70`). The encoder was trained on raw features, so this creates an
out-of-distribution input. The 0.943 to 0.933 attacker result shows that this intervention does not
erase identity; it does not prove that scale causally explains only one percent of the leak.

Better tests are a model trained with normalized inputs, matched-scale robot subsets, regression or
residualization against explicit morphology descriptors, and conditional attackers.

### P2. Throughput and several records are not publication-ready

Reported throughput ranges from about 377 to 11,939 FPS, and the no-temporal model is paradoxically
reported at 687 FPS while base is 10,554 FPS (`runs/ablations/SUMMARY.md:5-10`). The benchmark has
no warm-up or repeated timing distribution and synchronizes CUDA only after inference
(`scripts/benchmark.py:66-99`). System contention also differed across runs.

The paper draft still marks C5 as pending (`docs/PAPER_DRAFT.md:35-43`), calls output
"IK-teacher fidelity" despite the contact gap, and points to a stale WBT command file. These are
record-maintenance issues, not model failures, but they can invalidate claims if left unresolved.

## Claim ledger

| Proposed claim | Current verdict | Defensible wording now |
|---|---|---|
| One model retargets to five humanoids | Supported for five trained robots | One shared model amortizes a GMR teacher across five trained embodiments |
| IK-teacher fidelity | Partially supported | Low held-out MPJPE and valid limits, with a large unresolved contact-velocity gap |
| Contact-consistent decoding | Not supported | Current contact objectives worsen or fail to improve skate |
| Benefits motion tracking | Not tested | Open-loop proxy is not detectably worse; closed-loop WBT pending |
| Shared latent | Supported | Same-motion representations are strongly aligned |
| Embodiment-invariant latent | Refuted by current attacker | Aligned but embodiment identity remains nonlinearly decodable |
| Positive transfer from joint training | Not supported | Shared training likely incurs a fidelity cost at current setup |
| Zero-shot new-robot decoding | Failed | One LORO target degrades 5.2x; broader LORO needed |
| Robot-to-robot transfer | Failed in current training | Decoder exposure to robot-source latents is an untested repair |
| Temporal transformer suppresses jitter | Unsupported for current module | Current content-only attention trades fidelity for lower jerk |
| 12x throughput | Measurement incomplete | Neural inference is plausibly faster, exact factor pending controlled timing |
| First representation analysis | Novelty search incomplete | Quantitative analysis of alignment and identity leakage; use "to our knowledge" only after a systematic search |

## Literature synthesis

| Work | Relevant evidence | Implication for SNMR |
|---|---|---|
| [GMR](https://arxiv.org/abs/2510.02252) | Retargeting artifacts reduce tracking robustness under matched BeyondMimic settings; evaluates 21 trajectories with substantial repeated rollout evaluation | WBT is the decisive endpoint, not an open-loop proxy |
| [NMR](https://arxiv.org/abs/2603.22201) | Uses RL expert rollouts to repair targets before neural fine-tuning, then evaluates tracking under matched settings | A neural kinematic loss alone is not the paper's route to physical output |
| [ReActor](https://arxiv.org/abs/2605.06593) | Bilevel RL jointly adapts references and trackers; reports lower slide and better full-dataset tracking | If projection fails, physics-aware retargeting/tracking feedback is the strongest next direction |
| [ULTRA](https://arxiv.org/abs/2603.03279) | Treats retargeting as simulation-constrained RL with kinematic, dynamic, and contact rewards | Contact-rich robot references benefit from simulator-derived constraints |
| [X-Morph](https://arxiv.org/abs/2606.30290) | Uses neural retargeting, a physics-aware correction stage, then tracking and distillation | Retargeting output should be viewed as an intermediate reference, not necessarily the final executable motion |
| [Contact-Aware Retargeting](https://arxiv.org/abs/2109.07431) | Neural output does not precisely satisfy contact; 30-step encoder-space optimization refines it | A trained decoder plus constrained refinement is a well-supported hybrid |
| [UnderPressure](https://arxiv.org/abs/2208.04598) | Height/velocity contact heuristics degrade under noise and skate; learned force/contact labels plus optimization-based IK clean up motion | Avoid circular contact labels and test a real optimization pass |
| [EDGE](https://arxiv.org/abs/2211.10658) | Self-predicted contact consistency improves PFC from 3.08 to 1.54 in dance diffusion | Motivation for a factorized head/self-consistency test, not proof that SNMR's bundled loss should work |
| [ReConForM](https://arxiv.org/abs/2502.21207) | Adaptively selects sparse contact constraints and diagnoses conflicting objectives with gradient cosine | Measure conflicts before selecting fixed weights |
| [AdaMorph](https://arxiv.org/abs/2601.07284) | Trains on 12 robots using learned robot prompts, AdaLN, and embodiment-specific output adapters; "zero-shot" concerns unseen motions | Supports adapters and conditioning, not zero-shot unseen-robot claims |
| [Unified Latent Space for Cross-Embodiment Control](https://arxiv.org/abs/2601.15419) | Uses part-wise latents, contrastive alignment, and lightweight robot-specific encoder/decoder layers; new robots require about 15 minutes of adaptation | Few-shot adapters and part-wise latents are realistic next targets |
| [Human2Humanoid](https://arxiv.org/abs/2606.03476) | Uses T-pose-normalized end-effector objectives and explicit contact/height terms, evaluated on G1 | Supports morphology-normalized objectives, not multi-robot zero-shot evidence |
| [DANN](https://arxiv.org/abs/1505.07818) and [Elazar-Goldberg](https://arxiv.org/abs/1808.06640) | Gradient reversal can align domains, but a training adversary at chance does not prove information removal | Keep a held-out post-hoc attacker and measure utility loss |
| [PCGrad](https://arxiv.org/abs/2001.06782) and [GradNorm](https://arxiv.org/abs/1711.02257) | Negative gradient cosine and unequal gradient magnitude can make shared training worse; methods address different failure modes | Diagnose interference and imbalance before scaling capacity |
| [Transformer](https://arxiv.org/abs/1706.03762) | Positional information is explicitly required when recurrence and convolution are absent | E08 cannot be generalized to all temporal models |

The common lesson is not "replace neural retargeting with optimization." It is that high-throughput
neural prediction and exact physical/contact satisfaction solve different parts of the problem.
The literature increasingly uses a hybrid: amortized prediction followed by constrained correction,
physics-derived supervision, or joint retargeting/control optimization.

## Research questions and hypotheses

The next work should answer six distinct questions:

1. **Contact:** Is the skate gap caused by ineffective scaling/labeling, or does it require hard or
   physics-aware correction?
2. **Tracking:** Is SNMR non-inferior or superior to GMR as policy-training data?
3. **Sharing:** Is the generalist gap caused by capacity, gradient interference, exposure, or weak
   embodiment conditioning?
4. **Generalization:** Can SNMR decode robot-source latents, and separately, can it adapt to unseen
   targets?
5. **Representation:** Can identity leakage be reduced without sacrificing motion fidelity or
   transfer?
6. **Time:** Does an order-aware temporal model improve the fidelity/contact/jerk frontier?

Each gate below resolves one question and has an explicit stopping rule.

## Gate 0: Reproducibility and measurement repair

Complete this before more full training.

### Required records

For every retained checkpoint, record:

- exact git SHA and dirty-state indicator;
- complete command and parsed configuration;
- data split and data hashes;
- seed, effective samples per robot, optimizer steps, and LR schedule;
- checkpoint step and completion state;
- evaluator version, window count, FPS, and contact definition.

First reconcile E10a: determine whether each arm used BCE, self-consistency, teacher-mask velocity,
or a combination. Rename the experiment arms by their actual objective rather than by "contact."

### Required diagnostics

For each objective component, log:

- raw scalar value;
- weighted scalar value;
- gradient norm on the decoder output head and shared trunk;
- cosine similarity with distillation and with other active terms;
- contact prevalence and per-class contact precision, recall, and F1;
- per-robot values for shared runs.

For contact, define velocity as `(p[t] - p[t-1]) / dt` in m/s and normalize masked losses by the
number of active foot-coordinate samples. Keep BCE, teacher-stance velocity, self-consistency,
penetration, and teacher-velocity matching under separate weights.

### Metric suite

Report all of:

- MPJPE and DOF error on one fixed window protocol;
- stance XY speed under source, teacher, and height-only/hysteretic masks;
- contact precision/recall/F1;
- absolute foot height, floating, and penetration;
- FS-MANN as a secondary metric;
- DOF and body jerk;
- per-foot, per-clip, mean, median, and p95 values;
- bootstrap confidence intervals across clips.

For throughput, use warm-up, pre- and post-timing CUDA synchronization, at least 30 repeats, fixed
batch/window sizes, an idle device, and median plus p10/p90. Compare teacher and SNMR on declared
hardware; do not reduce the result to one cross-hardware ratio.

**Exit criterion:** E10 provenance is resolved and a baseline produces repeatable metrics within a
predefined tolerance across repeated evaluator runs.

### Gate 0 implementation status (2026-07-13)

**Status:** complete for provenance, objective diagnostics, and quality metrics. Throughput remains
blocked as a publication result because this environment has no available CUDA device and repeated
CPU timing was not stable enough.

#### E10 objective reconstruction

The completed E10a checkpoints resolve the earlier experiment-log contradiction:

| Renamed arm | Stored config | Actual contact objective | Contact head | Checkpoint SHA-256 |
|---|---|---|---:|---|
| `P2-edge-displacement+bce-w0.05` | Phase 2, 60k, `contact_weight=0.05` | predicted-contact 3D frame-displacement self-consistency **plus penetration**, and teacher-mask BCE | Yes | `35b165531386bdc6afef2f4d8922876b211dc5f5e2a00c464ca1d4af37dd98a3` |
| `P2-edge-displacement+bce-w0.2` | Phase 2, 60k, `contact_weight=0.2` | same | Yes | `73be5c3319dd3f0041acb2ea280e8d285171b6445a35e7a7ca0d24e67385a168` |
| `P2-edge-displacement+bce-w1.0` | Phase 2, 60k, `contact_weight=1.0` | same | Yes | `3c0b8f61b465671b613159ccf3ffb03195e011ae0e6a127140924ff1ecc8afd5` |

Evidence is the stored trainer configs, `decoder.contact_head.*` parameters in every checkpoint, and
the Phase 2 implementation present before all three run timestamps. These arms did **not** use
teacher-mask kinematic velocity. Their labels should not say "no EDGE head" or generic "contact."

The latest committed revision before their inferred start times was
`940966c4c535151622fe41cbb9dc8af2c30d05af`. That SHA was not stored in the checkpoints, and the
historical dirty-worktree state is irrecoverable, so this is a timestamp-derived candidate rather
than exact provenance. The incomplete Phase 1 `runs/e10_edge_w0.5` arm has only a 5k log point and
no checkpoint; its code path bundled teacher-mask displacement, EDGE displacement/penetration, and
BCE under one weight. It remains non-evidence.

#### Implemented contracts

- `snmr/experiment.py` writes an atomic `manifest.json` with Git/dirty-diff identity, source hashes,
  exact command/config, train/validation and robot-asset hashes, optimizer/schedule, planned and
  observed per-robot exposures, checkpoint hashes, completion state, and resume checks. Checkpoints
  now retain Python, NumPy, Torch, and CUDA RNG states.
- `snmr/diagnostics.py` measures raw/weighted terms, shared-trunk and output-head gradient norms,
  every pairwise gradient cosine, and contact prevalence/precision/recall/F1 without mutating
  `.grad`. Phase 2 additionally records per-robot terms and cross-robot gradient cosines.
- `snmr/losses.py` keeps every legacy loss intact and adds independently weighted BCE,
  predicted-contact consistency, teacher-stance velocity, penetration, and teacher-velocity terms.
  New movement losses compute velocity in m/s and divide by active foot-coordinate mass.
- Both trainers expose the factorized weights while preserving the historical meanings of
  `--contact_weight`, `--edge_contact`, and `--foot_vel_weight`.
- `snmr/metrics.py` supports explicit masks and height-only hysteresis. The benchmark reports
  source-contact, source-height, teacher-height, and legacy teacher masks; per-foot/per-clip values;
  contact-head classification; absolute height/floating/penetration; clip mean/median/p95; and
  deterministic clip-bootstrap intervals.
- Benchmark timing now uses 10 warm-ups, 30 repeats, `perf_counter`, pre/post CUDA synchronization,
  and median/p10/p90. The JSON records evaluator/checkpoint hashes and the full protocol. It does not
  invent teacher timing or a cross-hardware ratio.

Legacy objective behavior was preserved deliberately: existing flags still reproduce historical
loss formulas and old checkpoints still load. New experiments must use the factorized flags.

#### Validation and baseline reread

Validation passed in two parts because variant tests write generated MJCFs into the sibling
`holosoma` asset tree:

```bash
/home/ec2-user/work/retarget/.venv-snmr/bin/python -m pytest -q \
  --ignore=tests/test_variants.py
# 80 passed

/home/ec2-user/work/retarget/.venv-snmr/bin/python -m pytest tests/test_variants.py -q
# 11 passed
```

One-step Phase 1, Phase 2, and all-factorized-objective smokes completed with manifests,
diagnostics, checkpoints, and final evaluation. These smokes validate plumbing only. On the tiny
eight-frame factorized smoke at weight `0.1`, shared-trunk gradient norms were `3.80` for distill,
`0.118` for BCE, `6.24` for predicted-contact velocity, `6.26` for teacher-stance velocity, and
`4.03` for phase-balanced teacher-velocity matching. This confirms why weights cannot be selected
from raw scalar magnitudes or inherited from E10.

The retained G1 base checkpoint (`fd61ffd...e631a`) was then evaluated twice with the current
42-window, 192-frame protocol. All quality scalars, per-clip values, and bootstrap intervals matched
exactly between runs:

| Metric/mask | SNMR | GMR teacher |
|---|---:|---:|
| MPJPE | 0.0471 m | reference |
| Source-contact stance speed | 0.4421 m/s | 0.1226 m/s |
| Source-height stance speed | 0.7430 m/s | 0.5579 m/s |
| Teacher-height/hysteretic stance speed | 0.3206 m/s | 0.0604 m/s |
| Legacy full-clip teacher-mask stance speed | 0.2871 m/s | 0.0335 m/s |
| Historical window-local legacy metric | 0.3857 m/s | 0.0517 m/s |

Across seven clips, source-contact stance speed has a 95% bootstrap interval of
`[0.3728, 0.5072]` m/s. Teacher-height stance speed has a much wider
`[0.1117, 0.5559]` m/s interval, exposing strong clip heterogeneity. The teacher's non-circular
height-mask slide fraction is nonzero (`0.0468`), unlike the legacy circular comparison.

The two current CPU timing medians were 754 and 657 frames/s (12.9% relative difference), despite
identical quality payloads. Therefore Gate 0 passes for quality measurement, but no throughput
claim should be updated until the same timing protocol runs on an idle declared CUDA device.

**Next action:** run short C1-C4 diagnostic jobs on G1 and choose weights from measured gradient
ratios before any full schedule. Run the WBT smoke/pilot in parallel; neither task depends on a new
bundled contact sweep.

## Gate 1: Resolve contact with a factorized G1 study

Use one robot first. Hold architecture, data, schedule, seed, and evaluator fixed. Do not combine
this gate with embodiment augmentation or `zr_decode_prob`.

### Training and inference matrix

| ID | Contact head BCE | Predicted-contact consistency | Teacher-stance velocity | Teacher-velocity match | Post-process |
|---|---:|---:|---:|---:|---:|
| C0 | No | No | No | No | No |
| C1 | Yes | No | No | No | No |
| C2 | Yes | Yes | No | No | No |
| C3 | No | No | Yes, normalized XY m/s | No | No |
| C4 | No | No | No | Yes, phase-balanced | No |
| C5 | Yes | Yes | Yes | Optional only after C2-C4 attribution | No |
| C6 | N/A | N/A | N/A | N/A | Real constrained projection on C0 and the best soft arm |

C1 is a negative control: classification alone should not improve kinematics. C2 isolates the EDGE
mechanism. C3 tests supervised stance directly. C4 tests E25's hypothesis while balancing stance
and swing contributions. C5 is run only if component results justify it.

### Gate 1 execution plan

**Status:** next active experiment. The sequence is diagnostic calibration, a seed-0 screen, exact
benchmarking, then replication. A 50k run must not start before its diagnostic arm passes.

#### Frozen comparison protocol

| Item | Frozen value |
|---|---|
| Trainer/target | `scripts/train_phase1.py`, `unitree_g1` only |
| Data | `/home/ec2-user/work/retarget/data/pairs/unitree_g1`, existing 70/7 clip split |
| Architecture | temporal SNMR, latent 128, encoder 256, decoder 256 |
| Optimization | 50,000 steps, window 64, AdamW, LR `3e-4` to `1e-5` cosine |
| Evaluation during training | every 5,000 steps; checkpoint every 5,000 steps |
| Contact labels | teacher-foot height hysteresis, `--contact_mask teacher_height` |
| Screen seed | 0 |
| Confirmatory seeds | 0, 1, 2 for C0 and each promoted arm |
| Final evaluator | 192 frames, six windows per clip, seven held-out clips, bootstrap seed 0 |

This reproduces the matched E07 G1 specialist configuration. The old E07 checkpoint is context, not
the Gate 1 control: C0 must be retrained under the current manifest and evaluator code. Temporal
mixing remains enabled by intentionally omitting `--no_temporal`; changing it would create a
different experiment. All legacy objective weights remain zero: keep `--contact_weight 0` and
`--foot_vel_weight 0`, and do not pass `--edge_contact`.

Use a clean, fixed source revision for every arm. If unrelated work keeps the main checkout dirty,
run from a clean worktree rather than deleting or hiding that work. Every output directory is
immutable: never resume a directory with changed arguments.

#### Stage 1: short diagnostic calibration

The provisional weights below come from the Gate 0 all-objective smoke. They are starting values,
not a performance sweep:

| Arm | Provisional factorized weights |
|---|---|
| C0 | none |
| C1 | BCE `1.0` |
| C2 | BCE `1.0`, predicted-contact XY velocity `0.03` |
| C3 | teacher-height stance XY velocity `0.03` |
| C4 | phase-balanced teacher 3D velocity `0.05` |

Run five thousand calibration-only steps, with ten gradient observations per arm:

```bash
set -euo pipefail
cd /home/ec2-user/work/retarget/snmr

PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
PAIRS=/home/ec2-user/work/retarget/data/pairs/unitree_g1
COMMON=(
  --robot unitree_g1
  --pairs_dir "$PAIRS"
  --window 64
  --lr 0.0003
  --min_lr 0.00001
  --latent_dim 128
  --enc_hidden 256
  --dec_hidden 256
  --contact_mask teacher_height
  --contact_weight 0
  --foot_vel_weight 0
  --device cuda
)
DIAG=(
  --steps 5000
  --eval_every 1000
  --ckpt_every 5000
  --diag_every 500
  --seed 0
)

test ! -e runs/gate1_g1/diagnostics

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${DIAG[@]}" \
  --out runs/gate1_g1/diagnostics/c0_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${DIAG[@]}" \
  --contact_bce_weight 1.0 \
  --out runs/gate1_g1/diagnostics/c1_bce_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${DIAG[@]}" \
  --contact_bce_weight 1.0 \
  --edge_velocity_weight 0.03 \
  --out runs/gate1_g1/diagnostics/c2_edge_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${DIAG[@]}" \
  --stance_velocity_weight 0.03 \
  --out runs/gate1_g1/diagnostics/c3_stance_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${DIAG[@]}" \
  --teacher_velocity_weight 0.05 \
  --phase_balanced_velocity \
  --out runs/gate1_g1/diagnostics/c4_teacher_velocity_seed0
```

For each active added term, define the shared-trunk gradient ratio at diagnostic event `i` as

`r_i = ||grad(w_k L_k)||_shared_trunk / ||grad(L_distill)||_shared_trunk`.

Use the final five events, not the most favorable event. A movement term passes when median `r` is
in `[0.1, 1.0]` and p90 `r <= 2.0`. BCE passes when median `r` is in `[0.03, 0.5]` and p90
`r <= 1.0`. C0 passes when its distillation gradients and losses are finite and nonzero. Every arm
also requires finite total loss, nonempty contact support in at least four of the final five events,
and a completed manifest. Record gradient cosines, but do not tune a weight to make its cosine look
favorable.

The following command emits the ratios and distillation cosines used for that decision:

```bash
for D in runs/gate1_g1/diagnostics/*; do
  echo "$D"
  jq -r '
    . as $row
    | $row.loss_terms.distill.gradient_norm.shared_trunk as $base
    | $row.loss_terms
    | to_entries[]
    | select(.key == "contact_bce"
          or .key == "edge_velocity"
          or .key == "teacher_stance_velocity"
          or .key == "teacher_velocity")
    | [
        $row.step,
        .key,
        (.value.gradient_norm.shared_trunk / $base),
        $row.gradient_cosine.shared_trunk[("distill|" + .key)]
      ]
    | @tsv
  ' "$D/diagnostics.jsonl"
done
```

If a ratio fails, perform at most one deterministic recalibration:

`w_new = w_old * target_ratio / median(r)`,

using target `0.3` for movement and `0.1` for BCE. Cap a single change to `[w_old/4, 4*w_old]`,
start a fresh `*-r1` directory, and rerun the same 5k diagnostic. This is scale calibration, not a
quality-based weight search. Drop an arm that still fails. Before full training, add the accepted
weights and diagnostic statistics to this document so the full commands are preregistered.

#### Stage 2: seed-0 full screen

If all provisional weights pass, run these commands unchanged. If Stage 1 changes a weight, replace
only that accepted value below and commit the decision before launching.

```bash
set -euo pipefail
cd /home/ec2-user/work/retarget/snmr

PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python
PAIRS=/home/ec2-user/work/retarget/data/pairs/unitree_g1
COMMON=(
  --robot unitree_g1
  --pairs_dir "$PAIRS"
  --window 64
  --lr 0.0003
  --min_lr 0.00001
  --latent_dim 128
  --enc_hidden 256
  --dec_hidden 256
  --contact_mask teacher_height
  --contact_weight 0
  --foot_vel_weight 0
  --device cuda
)
FULL=(
  --steps 50000
  --eval_every 5000
  --ckpt_every 5000
  --diag_every 5000
  --seed 0
)

test ! -e runs/gate1_g1/screen

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${FULL[@]}" \
  --out runs/gate1_g1/screen/c0_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${FULL[@]}" \
  --contact_bce_weight 1.0 \
  --out runs/gate1_g1/screen/c1_bce_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${FULL[@]}" \
  --contact_bce_weight 1.0 \
  --edge_velocity_weight 0.03 \
  --out runs/gate1_g1/screen/c2_edge_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${FULL[@]}" \
  --stance_velocity_weight 0.03 \
  --out runs/gate1_g1/screen/c3_stance_seed0

"$PY" scripts/train_phase1.py "${COMMON[@]}" "${FULL[@]}" \
  --teacher_velocity_weight 0.05 \
  --phase_balanced_velocity \
  --out runs/gate1_g1/screen/c4_teacher_velocity_seed0
```

Stop an individual job only for a nonfinite loss/gradient, a failed checkpoint/manifest, or two
consecutive diagnostics with an added-term gradient ratio above `3.0`. Do not stop or extend an arm
because an intermediate quality number looks favorable.

#### Stage 3: fixed evaluation and promotion

Evaluate every completed seed-0 arm with the same command:

```bash
set -euo pipefail
cd /home/ec2-user/work/retarget/snmr
PY=/home/ec2-user/work/retarget/.venv-snmr/bin/python

for ARM in \
  c0_seed0 \
  c1_bce_seed0 \
  c2_edge_seed0 \
  c3_stance_seed0 \
  c4_teacher_velocity_seed0
do
  "$PY" scripts/benchmark.py \
    --ckpt "runs/gate1_g1/screen/$ARM/ckpt.pt" \
    --robots unitree_g1 \
    --pairs_root /home/ec2-user/work/retarget/data/pairs \
    --window 192 \
    --windows_per_clip 6 \
    --timing_warmup 10 \
    --timing_repeats 30 \
    --bootstrap_samples 2000 \
    --bootstrap_seed 0 \
    --out "runs/gate1_g1/screen/$ARM/benchmark" \
    --device cuda
done
```

The primary contact endpoint is held-out `teacher_height_stance_speed_ms`; it is non-circular
because contact uses height hysteresis, not the velocity being scored. `source_contact` and
`source_height` are required robustness endpoints. For each arm, retain the aggregate, all seven
per-clip values, both feet, MPJPE, DOF error, contact F1 when available, floating, penetration,
joint-limit proximity, and DOF/body jerk.

C1 is retained as the classification-only negative control whether or not it improves motion. A
soft kinematic arm is promoted to replication only when all of these seed-0 rules pass:

1. Teacher-height stance speed falls by at least 25% relative to the fresh C0.
2. At least five of seven clips improve on that endpoint.
3. MPJPE is no more than `0.005 m` above C0.
4. Source-contact stance speed is no more than 10% worse than C0.
5. DOF jerk is at most `1.2x` C0, limit violations remain zero, penetration mean rises by at most
   `0.002 m`, and penetration fraction rises by at most `0.02`.

Rank passing arms by teacher-height stance speed, then MPJPE, then source-contact stance speed.
Promote at most two. Do not fill the quota with a failing arm.

#### Stage 4: seed replication and decision

Retrain C0 and each promoted arm from scratch with seeds 1 and 2. Change only `--seed` and the output
directory from the accepted Stage 2 command; use
`runs/gate1_g1/replication/<arm>_seed<seed>`. Evaluate with the exact Stage 3 protocol. This yields
three matched training seeds for every final comparison. Evaluation bootstrap samples quantify clip
uncertainty and do not replace training seeds.

The contact mechanism passes Gate 1 only if the promoted arm:

- reaches teacher-height stance speed `<= 0.08 m/s` in at least two of three seeds;
- has a three-seed mean MPJPE no more than `0.005 m` above matched C0;
- retains the seed-0 physical guards above on the three-seed aggregate; and
- improves teacher-height stance speed in the same direction for at least two of three seeds.

The design's `<= 4.0 cm` MPJPE remains an absolute product target, but it is reported separately
from the causal contact decision. Conflating it with contact would make Gate 1 impossible to
interpret when a freshly trained C0 itself exceeds 4.0 cm.

#### Parallel task: WBT backend smoke

Gate 2A uses existing exports and is independent of Gate 1 weights. Run it in a separate holosoma
environment or GPU slot; do not alter a Gate 1 command to accommodate it. First create/activate the
documented MJWarp environment and verify imports:

```bash
cd /home/ec2-user/work/retarget/holosoma
bash scripts/setup_mujoco_via_uv.sh
source scripts/source_mujoco_uv_setup.sh
python -c "import torch, warp, mujoco_warp; print(torch.cuda.is_available())"
```

Then run a plumbing-only, paired 100-iteration smoke on `walk1_subject5`, changing only the motion
source and run name:

```bash
set -euo pipefail
cd /home/ec2-user/work/retarget/holosoma
source scripts/source_mujoco_uv_setup.sh

for SRC in snmr gmr; do
  python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    simulator:mjwarp \
    logger:disabled \
    --training.seed=0 \
    --training.num_envs=64 \
    --training.name="gate2a_${SRC}_walk_seed0" \
    --algo.config.num_learning_iterations=100 \
    --algo.config.save_interval=100 \
    --randomization.ignore_unsupported=True \
    "--command.setup_terms.motion_command.params.motion_config.motion_file=/home/ec2-user/work/retarget/snmr/runs/wbt_validation/${SRC}/walk1_subject5_mj.npz"
done
```

Gate 2A passes only when both sources reach iteration 100 with finite policy/value losses, nonzero
rewards and rollout buffers, meaningful episode lengths/terminations, and identical resolved
configuration except motion path and run name. This smoke is not tracking evidence. Freeze the
working resolved command before the three-clip, three-seed Gate 2B pilot, and update the stale
historical WBT command record only after this command succeeds.

### Constrained projection specification

C6 must be materially different from the existing foot-lock:

1. Infer stance from a non-circular source/teacher mask with onset/offset hysteresis.
2. Build one anchor per contiguous stance interval, including double support.
3. Optimize a full temporal window over leg joints and bounded root translation/yaw correction.
4. Minimize stance-anchor residual, deviation from SNMR output, velocity/acceleration, and optional
   end-effector residual.
5. Enforce joint limits and bounded root correction.
6. At each Gauss-Newton iteration, form the FK Jacobian and solve
   `(J^T W J + lambda I) delta = -J^T W r`, with line search or trust region. A normalized scalar
   gradient step is not Gauss-Newton.
7. Re-evaluate contact, penetration, jerk, and MPJPE after projection.

A windowed SQP or differentiable L-BFGS formulation is also acceptable if its constraints and
stopping conditions are explicit. Latent-space optimization is worth testing only after the
output-space constrained baseline is correct.

### Success and stopping rules

The exact soft-objective criteria are preregistered in the execution plan above. C6 uses the same
held-out evaluator and physical guards. Its primary target remains teacher-height stance speed
`<= 0.08 m/s`, with MPJPE no more than `+0.5 cm` versus its matched unprojected input and the
absolute `<= 4.0 cm` design target reported separately.

Decisions:

- **Soft arm passes:** use the simplest passing objective and validate on all five trained robots.
- **Soft arms fail, projection passes:** adopt prediction plus projection as the product/paper path.
- **Both fail:** stop penalty sweeps. Generate physics-repaired targets or couple retargeting with
  tracking, following the NMR/ReActor/ULTRA direction.

## Gate 2: Matched WBT validation

Do not wait for Gate 1 to finish before starting the baseline smoke test.

### Stage A: backend smoke

- Update the experimental command from the E20 MuJoCo/Warp scout, without editing the historical
  command record until the command succeeds.
- Train one short policy for one SNMR and one GMR reference.
- Verify learning progresses, buffers are nonzero, episode termination is meaningful, and both
  sources use identical configuration.

### Stage B: pilot

- Use the three existing clips across walk, fight, and dance.
- Train each source with three paired training seeds.
- Fix environment count, updates, wall-clock cap, network, reward, randomization, initialization,
  and evaluation seeds.
- Save learning curves, not just final points.

The pilot may detect catastrophic regressions. It cannot support a broad tracking-benefit claim.

### Stage C: confirmatory study

- Expand to 15-21 clips spanning locomotion, turning, jumping, fighting, and dance.
- Use at least five paired training seeds per clip and source if compute permits.
- Keep GMR and SNMR as the preregistered primary comparison. Add `SNMR+contact` or
  `SNMR+projection` as a third arm only after Gate 1 chooses it.
- Evaluate success/completion, survival, root position/orientation error, joint/body tracking error,
  contact violations, torque/energy, and sim2sim robustness.
- Aggregate with paired clip-level effects and a hierarchical bootstrap over clips and seeds.
  Evaluation rollouts quantify policy stochasticity; they do not replace independent training seeds.

Predefine practical non-inferiority margins before looking at results. A reasonable starting point is
no more than a 5 percentage-point loss in completion and no more than a 10% increase in joint
tracking error, but these margins should be tied to deployment requirements.

Claim rules:

- CI crosses the non-inferiority margin: **inconclusive**.
- Non-inferior but not superior: **SNMR is a faster reference generator without detectable tracking
  degradation**.
- Paired CI excludes zero in the favorable direction on primary endpoints: **tracking benefit**.
- Inferior: prioritize physics/contact correction and do not hide the result behind the PD proxy.

## Gate 3: Explain and reduce the shared-model fidelity cost

### Diagnostic first

On the same human window, compute each robot's loss gradient on shared parameters. Report:

- pairwise gradient cosine matrix;
- fraction and magnitude of negative-cosine pairs;
- per-robot gradient norm;
- per-robot training rate and loss scale;
- exposure counts.

### Controlled matrix

| ID | Trunk | Robot-specific parameters | Optimizer treatment | Question |
|---|---|---|---|---|
| S0 | Specialist | Full model | Standard | Exact per-robot control |
| S1 | Shared base | Existing conditioning | Standard | Reproduce the gap with matched exposure |
| S2 | Shared wider | Existing conditioning | Standard | Is raw capacity limiting? |
| S3 | Shared base | Small AdaLN/LoRA/output adapters | Standard | Is conditioning specialization limiting? |
| S4 | Shared base | Existing or adapters | PCGrad if conflicts are negative | Is gradient direction limiting? |
| S5 | Shared base | Existing or adapters | GradNorm if magnitudes/rates are imbalanced | Is task balance limiting? |

Do not run S4 merely because it exists; PCGrad addresses negative directional conflict. Do not run
S5 unless gradient magnitudes or training rates are imbalanced.

All arms need identical per-robot sample counts, equivalent LR progress per exposure, fixed
evaluation windows, and at least three seeds for the final comparison. Report both total and
robot-specific parameter counts.

**Target:** reduce the shared-specialist gap to `<= 0.5 cm` per robot without more than doubling
total model parameters. If adapters win, frame the system as a shared backbone with lightweight
embodiment specialization rather than strict one-model zero-shot decoding.

## Gate 4: Separate robot-source transfer from unseen-target adaptation

Define two tasks:

- **G-source:** encode motion from robot A and decode it on a target robot already seen in training.
- **G-target:** decode human or robot motion on a target embodiment excluded from training.

### G-source experiment

Run `zr_decode_prob` as its own intervention with `p` in a small preregistered set such as
`{0, 0.1, 0.3, 0.5}`. Evaluate human-to-robot and robot-to-robot paths together so improvement in
one cannot silently destroy the other. Include same-source reconstruction and cross-source pairs.

### G-target experiment

1. Repeat LORO on at least three structurally different robots, not only PM01.
2. Test scaled variants as interpolation and reserve genuinely different topologies for
   extrapolation.
3. Enrich morphology features with normalized offsets, limit midpoint/range, body mass/inertia,
   actuator effort/velocity bounds, collision/contact geometry, and side/body-part semantics.
4. Replace the MLP-max embodiment encoder with topology-aware message passing or learned part
   tokens.
5. Measure adaptation curves for 0, 1, 5, and 10 clips or a fixed number of target frames.
6. Compare a frozen shared backbone plus lightweight target adapters against full fine-tuning.

A useful 2x2 diagnostic is:

| | No synthetic morphology variants | Synthetic variants |
|---|---|---|
| Human-source only | Current target-generalization baseline | Effect of target diversity |
| Human + robot-source decoding | Effect of `zr_decode_prob` | Interaction, run only after main effects are known |

This factorial separates two mechanisms. Combining every intervention in one expensive run would
not.

**Near-term success:** a lightweight adapter reaches within 1.25x of in-domain error using a fixed,
small adaptation budget.

**Zero-shot success:** at least three held-out robots remain within 2x of their in-domain controls.
Until then, describe this as few-shot morphology adaptation.

## Gate 5: Strengthen the representation claim

### Measurement controls

Add:

- motion-permutation and time-shift controls for CKA and retrieval;
- retrieval across clips/subjects and within motion category, not only exact synchronized windows;
- paired same-motion distance versus hard negatives matched for phase/category;
- CKA after residualizing motion identity or pose content;
- attacker results across several architectures, regularization values, and splits;
- explicit morphology-regression probes for scale, limb ratios, and topology descriptors.

t-SNE and UMAP remain visualizations. They should not be called proof.

### Interventions

Only after the controls:

1. Use paired same-motion contrastive learning with hard negatives.
2. Test part-wise latent tokens for legs, arms, and trunk, following the unified-latent precedent.
3. If identity leakage remains harmful, add a gradient-reversal or conditional domain adversary.
4. Always train a new held-out post-hoc attacker after training.

Report a Pareto frontier:

- attacker accuracy;
- exact and cross-clip retrieval;
- human-to-robot MPJPE/contact;
- robot-to-robot transfer;
- unseen-target adaptation.

Chance attacker accuracy is not the optimization target if it removes motion information or harms
decoding.

## Gate 6: Test temporal modeling correctly

Compare under matched parameter and training budgets:

| Model | Order signal | Deployment mode |
|---|---|---|
| Framewise encoder | None needed | Causal |
| Transformer | Relative or absolute positional encoding | Bidirectional |
| Transformer | Relative position plus causal mask | Causal |
| Dilated TCN | Convolutional order | Causal and bidirectional variants |

Measure MPJPE, stance speed, jerk, boundary discontinuity, latency, and controlled throughput.
Evaluate long clips with overlapping windows so boundary artifacts are visible.

Keep the framewise model as default unless an order-aware model improves the Pareto frontier. A
temporal model may be justified for jerk or contact timing even if its raw MPJPE is slightly worse,
but that trade must be explicit.

## Recommended execution order

| Order | Work | Why now | Stop/go decision |
|---:|---|---|---|
| 1 | Gate 0 provenance, loss/gradient instrumentation, metric repair | Current contact evidence is not attributable | No more full runs until complete |
| 2 | Gate 2A/2B WBT smoke and pilot | Tests the central downstream thesis with existing checkpoints | Decide whether contact is blocking tracking |
| 3 | Gate 1 factorized G1 contact study | Resolves the largest kinematic gap cheaply | Soft loss, projection, or physics supervision |
| 4 | Gate 2C confirmatory WBT | Establishes non-inferiority or benefit | Determines publishable tracking claim |
| 5 | Gate 3 sharing diagnosis and adapters | Current positive-transfer claim is invalid | Capacity vs interference vs conditioning |
| 6 | Gate 4 source/target generalization | Prevents conflating two failures | Few-shot adapter or longer-term zero shot |
| 7 | Gate 5 latent intervention | Invariance is useful only if utility survives | Select Pareto point |
| 8 | Gate 6 proper temporal ablation | Important but not the present blocker | Framewise or order-aware temporal default |

WBT smoke/pilot and contact diagnostics can run in parallel. Do not combine contact, morphology
augmentation, `zr_decode_prob`, domain confusion, and capacity scaling into one retrain to save GPU
time; that would save compute while losing the research result.

## Publication framing

### Minimum defensible story now

> SNMR amortizes an optimization teacher across five trained humanoid embodiments and learns a
> quantitatively aligned cross-embodiment representation. Controlled negative results expose three
> remaining limits: contact fidelity, shared-versus-specialist accuracy, and unseen-robot decoding.

This story is honest and technically useful, but it still needs a WBT result for a robotics paper
whose motivation is downstream control.

### Stronger paper after Gates 1 and 2

The strongest realistic story is:

> A shared neural retargeter provides high-throughput initialization, while a lightweight
> contact/physics correction stage closes exact constraints. The resulting references are
> non-inferior or better than GMR for matched whole-body tracking.

If WBT is non-inferior but contact remains worse, publish speed and representation as the benefit and
state contact as a measured limitation. If WBT is inferior, the next contribution should be
physics-repaired supervision or joint retargeting/control, not another claim rewrite.

### Claims to avoid until gates pass

- "contact-consistent";
- "improves tracking";
- "embodiment-invariant";
- "positive transfer";
- "zero-shot to new robots";
- "temporal transformer suppresses jitter";
- exact `12x` speedup;
- t-SNE/UMAP as proof;
- teacher slide fraction as an independent baseline.

## No-go rules

1. Do not run another single bundled `contact_weight` sweep.
2. Do not call an incomplete or inactive run "running."
3. Do not compare four-window and 16-window values.
4. Do not infer causality from evaluation-time feature normalization.
5. Do not scale the shared model before checking gradient conflict and exposure.
6. Do not use one LORO robot to support a morphology-wide conclusion.
7. Do not let evaluation rollouts substitute for policy-training seeds.
8. Do not treat a training adversary at chance as proof of invariance.
9. Do not claim hard projection from the current normalized-gradient foot-lock.
10. Do not update the paper claim table until the artifact manifest and evaluator protocol identify
    the exact evidence for each number.
11. Do not start a full Gate 1 run before its short diagnostic passes the gradient-ratio contract.
12. Do not change morphology data, `zr_decode_prob`, capacity, temporal architecture, masks, or
    schedule inside the C0-C4 comparison.
13. Do not add penetration or another objective to C1-C4 after seeing seed-0 results; C5 requires a
    separate preregistered rationale.

## References

1. Araujo et al., [Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking](https://arxiv.org/abs/2510.02252), 2025.
2. Zhao et al., [Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control](https://arxiv.org/abs/2603.22201), 2026.
3. Muller et al., [ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting](https://arxiv.org/abs/2605.06593), 2026.
4. He et al., [ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation](https://arxiv.org/abs/2603.03279), 2026.
5. Sharma et al., [X-Morph: Human Motion Priors for Scalable Robot Learning Across Morphologies](https://arxiv.org/abs/2606.30290), 2026.
6. Zhang et al., [AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers](https://arxiv.org/abs/2601.07284), 2026.
7. Yan and Lee, [Learning a Unified Latent Space for Cross-Embodiment Robot Control](https://arxiv.org/abs/2601.15419), 2026.
8. Huang et al., [Human2Humanoid: Physics-Aware Cross-Morphology Motion Retargeting for Humanoid Robots](https://arxiv.org/abs/2606.03476), 2026.
9. Villegas et al., [Contact-Aware Retargeting of Skinned Motion](https://arxiv.org/abs/2109.07431), 2021.
10. Mourot et al., [UnderPressure: Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup](https://arxiv.org/abs/2208.04598), 2022.
11. Tseng et al., [EDGE: Editable Dance Generation From Music](https://arxiv.org/abs/2211.10658), 2022.
12. Cheynel et al., [ReConForM: Real-time Contact-aware Motion Retargeting for More Diverse Character Morphologies](https://arxiv.org/abs/2502.21207), 2025.
13. Ganin et al., [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818), 2016.
14. Elazar and Goldberg, [Adversarial Removal of Demographic Attributes from Text Data](https://arxiv.org/abs/1808.06640), 2018.
15. Yu et al., [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782), 2020.
16. Chen et al., [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257), 2018.
17. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
