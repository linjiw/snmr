# ICRA 2027 Readiness Audit

**Audit date:** 2026-08-09  
**Target deadline:** 2026-09-15, 23:59 PST  
**Current manuscript:** `paper/main.tex`, six US-letter pages after the E70-centered rewrite,
anonymous, with all fonts embedded

Official ICRA 2027 instructions currently require a double-anonymous first submission, at least
three PaperPlaza keywords, and no more than eight total pages including references.  Reviewers
need not inspect external links or artifacts; an optional video is the only separate supporting
medium.  Sources:

- https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/
- https://2027.ieee-icra.org/announcements/call-for-technical-papers/

## Verdict

The paper is not submission-ready today, but its central seed-0 result is now complete and positive.
E69 removed the nuisance-model feasibility blocker, and E70 passed every preregistered seed-0
content gate: SNMR-minus-time +0.1544, 95% CI [0.0932, 0.2147], and
SNMR-minus-shuffled +0.1871, [0.1368, 0.2358].  The remaining P0 is the already launched seed-1/2
confirmation followed by replacing seed-0 values with the frozen three-seed aggregation.

The current draft's strongest contribution is the measurement instrument: an exclusive learned
command channel with matched goal-source controls.  Its weakest feature is breadth.  It combines
the interface assay, representation probes, multi-teacher retargeter scaling, simulator-defect
forensics, and several future research programs.  Without E70, reviewers can reasonably conclude
that the central evidence still comes from one memorisable trajectory.  With E70, the paper should
be narrowed around one thesis:

> An exclusive retarget-to-track command interface makes control-usable information measurable;
> matched time, proprioception, shuffle, and channel-destruction controls reveal what crosses it.

The closest new positioning threat is ULTRA (arXiv:2603.03279, 2026), which also distills a
tracker into a compact multimodal latent controller.  Its dense-reference path passes a residual
full-body goal directly to the decoder, so it demonstrates capability but does not isolate what
the latent carries.  RoboGhost (arXiv:2510.14952, 2025) similarly motivates replacing the
retarget-then-track boundary with latent guidance.  Both should be cited; the paper's defensible
novelty is the exclusive measurement contract and controlled assay, not the existence of a latent
humanoid command.

## Submission-critical experiment matrix

| Priority | Evidence | Ready state | Submission condition |
| --- | --- | --- | --- |
| P0 | Two passing specialists | complete | report as checkpoint-specific assay calibration |
| P0 | Reference-only ambiguous pair | complete | 69 frozen windows; preserve loader order |
| P0 | Explicit positive-control student | seed 0 passes | confirm seeds 1-2 before final aggregate |
| P0 | A versus time and shuffled controls | seed 0 passes | confirmation queue for seeds 1-2 is live |
| P0 | Exact outcome-conditioned claims | integrated | retain the scoped positive language from E70 |
| P1 | Clean command-channel intervention | seed 0 complete | zero, shuffle, and marginal-random all collapse to zero |
| P1 | Paired uncertainty | implemented | cluster over frame pair, retain training-seed identity |
| P1 | Anonymous reproducibility bundle | partial | consolidate commands, pins, hashes, and small reports |
| P2 | T1 cross-embodiment result | deferred | include only if independently valid after E70 |

E70 has three legitimate outcomes.  A passing A-over-time gate supports control-usable
trajectory-disambiguating information under this interface.  A null with a passing explicit
control supports a scoped interface-extraction null.  A failed explicit control invalidates the
student assay.  None licenses an information-theoretic claim about everything contained in the
frozen retargeting latent.

## Manuscript surgery after E70

The outcome-conditioned rewrite now compiles in six pages, leaving two pages under the
initial-submission cap.  The broad interaction-multimodality narrative and latency figure were cut;
the defect story was reduced to a validity audit; and E70 is the central table and teaser result.
Use the remaining space for confirmation-seed uncertainty, a paired-effect figure, reproducibility,
and clearer method detail rather than reopening unrelated research programs.

A 2026-08-09 executable-method audit also removed a stale CVAE description from the paper.  The
registered E70 runs set `E52_DET=1`: a deterministic encoder/decoder trained only by teacher-action
MSE, with teacher-mixture collection and four-round FIFO replay.  The paper figure and method now
state that recipe exactly; posterior, KL, sampling noise, and smoothness are explicitly labeled as
earlier calibration variants rather than E70 ingredients.

Make room by cutting before compressing typography:

1. Reduce “A Defect in the Measurement Substrate” to one validity-audit paragraph and one
   before/after number.  The negative-results ledger belongs in the artifact, not the main story.
2. Reduce “Scaling the Retargeter Behind the Interface” to the minimum evidence that the upstream
   latent is genuinely cross-embodiment and imperfect.  The interaction-rich two-teacher study is
   a separate paper-sized story unless it directly explains E70.
3. Merge the clock explanation, ambiguity motivation, and limitations discussion; the draft now
   explains the same single-trajectory confound in several places.
4. Shorten the conclusion to the measured result and its boundary.  It currently restates several
   deferred programs.
5. Keep the first figure only if panel (c) is replaced with the decisive two-walk result.  A figure
   centered on the older single-clip table will visually contradict the final thesis.

Recommended final structure:

1. Introduction and three contributions.
2. Related work.
3. Exclusive interface and structural/causal validity.
4. Assay design: single-clip diagnostic plus preregistered two-trajectory test.
5. Results: C/A/T/B/S, paired intervals, per-clip direction, command destruction.
6. What the upstream latent exposes, limited to probes that explain the result.
7. Limitations and concise conclusion.

## Claim edits that are required regardless of outcome

- Say “matches the evaluated teacher checkpoint,” never teacher-algorithm parity; there is one
  teacher training seed.
- Say the noise-trained CVAE *variant* is more robust than the deterministic variant; the current
  factorial does not isolate noise as the cause.
- Separate structural exclusivity from causal use.  The old reference-blanking test diagnosed a
  leak; E70's clean `z_cmd` destruction is the relevant causal evidence.
- Treat time as a matched null, not an information upper bound.
- Do not market failed E66/E67/E68 endpoints as latent evidence.  One compact validity ledger is
  enough.
- Scope any E70 positive to two LAFAN1 walks, the Unitree G1 simulator, the frozen SNMR endpoint,
  and this controller class.

## Reproducibility and compliance checklist

- [x] Anonymous author line and no visible affiliations.
- [x] Six-page US-letter build leaves two pages of headroom under the current eight-page rule.
- [x] Core environment, GMR/Holosoma revisions, data count, artifact paths, and hashes recorded.
- [x] Full repository test baseline recorded: 204 passed, 4 skipped.
- [x] E70 seed-0 positive control and four representation arms complete.
- [ ] Seeds 1 and 2 complete if seed 0 is valid.
- [x] Seed-0 paired analysis and machine-readable report frozen with input hashes.
- [x] Three-seed paper-value generator rejects partial/nonfrozen analyses and stamps the analyzer
  SHA-256 into generated LaTeX macros.
- [x] Manuscript claims and Figure 1 updated from the frozen seed-0 E70 outcome.
- [x] Local PDF rebuilt: six letter pages, all fonts embedded; PaperPlaza check remains.
- [x] Recommended PaperPlaza keywords selected: `Motion Retargeting`, `Human and Humanoid Motion
  Analysis and Synthesis`, and `Machine Learning for Robot Control` (verify the live taxonomy when
  entering metadata).
- [x] Optional video protocol and policy-independent exemplar manifest frozen.
- [x] Video composer binds its labels to the raw-capture index and visibly distinguishes completed
  from terminated rollouts; the local size gate is a stricter 19 MB.
- [ ] Raw simulation rendering and final composition complete; both wait for the live three-seed
  queue so capture cannot perturb confirmation.
- [ ] Author metadata entered in PaperPlaza while remaining absent from the review PDF.

## Time-critical path

The scientific critical path is now the live seed-1/2 confirmation queue, the frozen three-seed
analyzer, and a final aggregate manuscript pass.  After that, spend effort on artifact packaging,
PaperPlaza compliance, keywords, and an optional paired-rollout video rather than a new simulator
or embodiment.  Isaac Lab, SONIC, and additional datasets are not on the ICRA critical path.
