# Can the learned interface beat the explicit one? — research memo

**Date:** 2026-08-15. **Question, from the owner:** SNMR cannot surpass the explicit command on clean
tracking. Can we change the method so it does, or show it degrades more gracefully when the tracking
target is polluted — a safety claim — so the work benefits real humanoid deployment?

**Answer:** the safety-via-robustness claim is **contradicted by our own data**, not merely unproven.
Two of the three deployment framings are dead. One is real but is a claim about the *retargeter*,
not the command interface. The recommendation is to close the line with evidence, bank the honest
negative, and spend the remaining days on the manuscript.

This memo records what was checked, by whom, and what is verified versus disputed.

---

## 1. Beating explicit on clean tracking is structurally closed

The explicit student sits at teacher parity (0.923 against a 0.917 teacher macro). The latent is a
lossy compression of the same underlying motion the explicit reference states exactly. On clean,
in-distribution tracking of a clip the teacher already knows, there is no information for the latent
to add — and E52-v4 arm D (explicit + latent fused) already returned a three-seed null.

No experiment was proposed that credibly overturns this, and none should be run. This is settled in
`docs/BENCHMARK_QUESTION_2026-08-12.md` and nothing found here reopens it.

## 2. The robustness/safety claim is contradicted

### 2.1 What was measured

E77 (`docs/E77_DEGRADATION_PILOT.md`) swept both arms over two severity axes on the frozen students.
The key design point: `E52_EVAL_HOLD_Z` acts on `z_cmd`, the **shared** bottleneck — identical
tensor, identical dimension, identical decoder — so severity in control ticks is matched by
construction, sidestepping the cross-space unit problem.

### 2.2 The apparent positive dissolved under paired analysis

The pilot's marginal numbers showed one encouraging cell: at σ = 1.0, SNMR retained 52.4% of its
clean performance against explicit's 42.4%, a **+0.100** advantage, with both arms functional.

Restricting to the **704 rollouts both arms complete cleanly from identical starts** — the same
runs, same 1,024 rollouts, only the normalisation changes:

| severity | marginal retention (E / S) | matched-subset (E / S) | subset diff | McNemar (E-only : S-only) |
| --- | --- | --- | ---: | --- |
| hold 40 ms | 0.931 / 0.811 | 0.940 / 0.722 | **−0.219** | 169 : 15 |
| hold 100 ms | 0.241 / 0.006 | 0.295 / 0.004 | **−0.291** | 207 : 2 |
| noise σ 0.5 | 0.912 / 0.878 | 0.940 / 0.778 | **−0.162** | 133 : 19 |
| noise σ 1.0 | 0.424 / **0.524** | 0.520 / 0.500 | **−0.020** | 107 : 93 |

**The +0.100 becomes −0.020.** It was never an effect: SNMR's lower clean baseline excludes the easy
rollouts from its own denominator. Retention does not cancel the 22.4 pp clean gap — it launders it.

On the axis whose matching cannot be attacked, the asymmetry is 11× at 40 ms and **100× at 100 ms**,
against the hypothesis.

### 2.3 Why no better-designed version rescues it

- **The window and the matching trade off against each other.** The hold axis is matched by
  construction but has no window: explicit falls 0.862 → 0.223 between 40 and 100 ms while SNMR
  falls 0.566 → 0.004, and both read exactly 0.000 by 200 ms. The Gaussian axis has a window but no
  defensible matching — `E52_EVAL_NOISE_CMD` is a **structural no-op** on the SNMR arm
  (`distillation.py:65-76`: `cmd` never enters that prior), and the channels are 64-D against 256-D.
  The one axis with a window is the one with no valid matching. That is the whole program in a
  sentence.
- **The discrimination floor is training-seed variance, not evaluation noise.** Per-seed A−T is
  [0.154, 0.278, 0.140]: sd 0.0757, three-seed SEM **0.044**. A new cross-arm effect needs ~10 pp,
  not the ~2 pp that E76's 0.0083 evaluation floor suggests. This project has set 5 pp gates twice
  (E61b, E61-v4) and failed both for exactly this reason.
- **A physical objection to the matched axis.** `z_cmd = mu_prior(proprio, goal)`, so holding it also
  freezes proprioceptive feedback. It simulates *"the policy stopped running"*, not *"the command
  went stale"* — opposite failure modes with opposite correct responses. There is no wire on a real
  robot that freezes an internal activation. The deployment-faithful version freezes each arm's own
  **input** while the prior keeps reacting to fresh proprioception; that is a different experiment
  and needs a new evaluator.

### 2.4 Two prior results, both corrected

- **E65's 18× hold robustness does not support a latent advantage.** Both cells ran on
  `c_prior_explicit` — *both explicit arms*. It is a training-recipe result about the shared
  bottleneck and contains no interface comparison. Its checkpoints are on a decommissioned host and
  cannot be re-analysed.
- **`E52_DET=1` toggles four things at once**, not one: posterior, KL, reparameterization noise, and
  the paired temporal-smoothness loss on `z_cmd` (`train_e52_dagger.py:445-476`). The smoothness term
  is the better a-priori explanation of zero-order-hold tolerance. E65's attribution of its 18× to
  "σ = 0.3 reparameterization noise" is a 1-of-4 attribution that was never isolated. Every frozen
  E70 student is `deterministic: True`, so whatever that mechanism is, it is absent from the students
  the paper ships.

## 3. Bandwidth is dead

The G1 exposes a 1 Gbps link; the worst-case uncompressed command payload is ~0.04% of it, and
contemporary stacks run the policy onboard with no link in the command path. At face value SNMR
spends ~4× the command bits for 22.4 pp worse completion. Bits-per-frame matching across 64 explicit
and 128 latent dimensions forces the exchange rate, and that choice *is* the result — the textbook
"rigged by severity units" attack. Quantizing the shared `z_cmd` instead removes the matching problem
but inherits §2's verdict.

Keep at most one honest sentence: the channel is over-provisioned by four orders of magnitude, so
report milliseconds, not bits.

## 4. What is actually real

### 4.1 Verified: the learned decoder cannot emit an out-of-limit joint target

`snmr/model.py:318-323`:

```python
node_angle = torch.tanh(self.angle_head(h).squeeze(-1))   # (T, N) in [-1, 1]
dof_pos = lo + (hi - lo) * 0.5 * (gathered + 1.0)          # scaled to limits
```

Joint-limit satisfaction is an **architectural guarantee**, not a learned behaviour. A limit-riding
target on hardware means a saturated actuator with no authority left for balance, so this is a real
safety property — and it is a property of the offline signal, with **zero sim-to-real gap**.

### 4.2 DISPUTED and must not be quoted yet: how often the IK reference violates limits

The complementary half of that claim — that the IK reference the field tracks *does* pin joints at
hard limits — is **not established**. A research agent reported 9.83% of frames on
`walk1_subject1`. My own measurement, loading the scene XML and matching all 29 joints by name
through `mj_name2id`, gives **96.27% of frames** and 7.38% of samples, with `left_knee_joint` at
64% and `left_hip_roll_joint` at 62%.

A 10× disagreement, and 96% of frames is not physically plausible for a walk. Likely causes: a joint
offset convention (holosoma recently added per-robot joint offset calibration), a different limit
source, or a sign convention. **Neither number may be used until this is resolved.** It is a
half-day CPU task and it gates the strongest deployment claim available.

### 4.3 Plausible and cheap: the retargeter cannot close a 50 Hz loop, the encoder can

Published GMR runs per-frame IK at 22–29 ms/frame; the in-repo encoder benchmark records ~1.5
ms/frame. If that holds on matched hardware, it decides whether the retargeter can live in the
control loop at all — a real plumbing decision. The current comparison is GMR-on-CPU against
encoder-on-GPU and is **not creditable as stated**; it needs one CPU day of matched-hardware timing.
Note it is a claim about the retargeter, not the command interface.

### 4.4 Free, and possibly the most valuable: completion is not a safety metric

The E70 instrument terminates on a z-axis tracking tube. **A robot that safely ignores a garbage
command scores identically to one that faceplants.** Every completion-based robustness result in
this program, positive or negative, is silent about safety. `mean_survival_s` is the same metric with
lower variance — a statistical improvement, not a semantic one.

This is worth stating in Limitations as an argument, for free, and it is a genuine methodological
contribution: *completion-rate robustness studies on humanoid tracking benchmarks are not safety
studies.*

### 4.5 Free: the normalisation reversal is itself a methods finding

Same runs, same rollouts, opposite cross-arm conclusion depending only on whether retention is
computed marginally or on the paired matched subset. No paper in the sweep reports an evaluation
noise floor, and several published robustness gaps would not clear ours. That is a worked example a
reviewer cannot wave off.

## 5. Novelty check

A research agent reports that **UniAct (arXiv:2512.24321, Dec 2025)** already published "compressed
command channel degrades more gracefully" with a 19.2 pp effect. **I have not independently verified
this** and it should be checked before any framing decision rests on it — but if correct, shipping a
weaker, underpowered version of a known result is worse than shipping nothing on that axis.

## 6. Recommendation

**Run no new experiment for this submission. Write the paper.**

The evidence is complete and frozen: the clock confound, A−T +0.191 [0.124, 0.274], all-seed
destruction on *both* arms, E72's phase sensitivity rejecting clip identity, E76's reproducibility
measurement, and a certified video. What is not built is what reviewers actually read — Figure 1
still shows the system rather than the experiment, and the first result arrives in abstract sentence
three.

A degradation result would not strengthen an identification claim; it would announce a second,
weaker claim inside it and invite "why is the robustness study n = 3, two walk clips, on the wrong
severity axis?"

**Also recommended: kill P1/E71.** Its own protocol concedes it supplies zero likelihood ratio
between clip identity and content — which was its purpose, and which E72 has since addressed
behaviourally. Its remaining value was an ICRA-legible video, and B4 certified one on 2026-08-15. It
costs roughly twelve days, GPU exclusivity, and a repo commit freeze that directly blocks the
manuscript pass (`prepare_e71_freeze.py:437-438` binds tracked working-tree status;
`eval_e71_command_swap.py:929-935` aborts on any difference). If it is kept anyway, run it in a
separate git worktree so the freeze cannot reach the manuscript tree.

### What to do now, in order

1. The single `main.tex` pass — new Figure 1, finding-first abstract, title, the P0 corrections, and
   the E72/E75/E76 integration.
2. Free reanalysis on `mean_survival_s`: E75's completion is exactly 0.000 in all nine cells, but
   survival separates `marginal_random` (0.832/0.780/0.844 s) from `zero` and `shuffle` (~0.54–0.62).
   Zero GPU, strictly a writing input. Report `zero` and `shuffle` as indistinguishable; only
   `marginal_random` separates.
3. One scope sentence in Limitations staking the degradation axis **without** claiming it — a
   sentence that promises robustness invites a reviewer to demand it in this paper.

### The follow-on paper

Its identity should be the **matched-severity protocol**, not "latents degrade more gracefully".
AnyBody conceded in print that cross-command-space robustness comparisons are "not strictly
equivalent... due to different observation spaces" and then stopped; that methods slot is empty, and
this project now owns four verified facts about why the comparison is hard (§2.3, §2.4, §4.5). The
primary endpoint should be survival or a real safety event, never completion.

Two half-day checks set it up and neither touches this submission — both strictly **after** the text
lock:

- resolve the joint-limit measurement in §4.2;
- the GMR reproduction gate: does GMR at a repaired sha reproduce the frozen `qpos`? The
  deployment-faithful experiment (corrupt the human source, route it through each interface's own
  encoder, score against the clean tube with reference-free safety events) is entirely gated on it.

## 7. Provenance

Six parallel research agents plus three adversarial lenses over the codebase, the experiment log, and
external literature. The matched-subset reversal in §2.2 was produced by the skeptic agent from the
E77 pilot arrays and **independently reproduced by me** before being recorded here. The tanh
guarantee in §4.1 was read directly from source. The disputed measurement in §4.2 is flagged
precisely because it did not survive that treatment.
