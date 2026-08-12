# ICRA 2027 Submission Execution Plan (SNMR / E70)

**Authored:** 2026-08-11, from a full repo + artifact audit (Fable).
**Deadline:** 2026-09-15 23:59 PST (ICRA 2027, double-anonymous, ≤8 pages incl. references).
**Video upload windows:** Aug 5–Sep 9 and Sep 17–22.
**Intended executor:** an autonomous agent working in `/home/robotixx/snmr`, reporting to the human owner.

This plan is self-contained. It assumes no memory of prior conversations. Read
`docs/RESEARCH_GOAL_2026-08-10.md`, `docs/ICRA_READINESS_AUDIT_2026-08-09.md`, and
`docs/E70_MULTITRAJ_PROTOCOL.md` before executing anything.

---

## 0. Current state (audited 2026-08-11)

- **E70** (frozen two-walk five-arm assay, seeds 0/1/2): seed 0 complete and **passing every
  preregistered content gate** (A−T +0.1544 [0.0932, 0.2147]; A−S +0.1871 [0.1368, 0.2358];
  positive per clip; explicit control 0.925 → 0.000 under zero/shuffle/matched-marginal command
  destruction). Seed 1 complete, all five arms. Seed 2: explicit/SNMR/time complete;
  **seed2_proprio and seed2_shuffled missing** (proprio was interrupted at round 1550, quarantined,
  must restart from round 0).
- **Blocker:** the frozen 26,000-MiB GPU launch gate is unmet (~22.7 GiB free) because unrelated
  same-owner projects (traversal-critic PPO / cosmos-framework) hold ~9.4 GiB. Two detached
  supervisors are alive and will auto-launch when the gate opens:
  `scripts/run_e70_recovery_supervisor.sh` (log `/tmp/snmr-e70-recovery-supervisor.log`) and
  `scripts/run_e70_postprocess_supervisor.sh`.
- **Remaining GPU work:** ~2–4 hours (two cells + evals + analyzer; seed-1 cells completed
  ~36 min apart).
- **Paper:** `paper/main.tex`, 6 US-letter pages, anonymous, fonts embedded, no stubs/TODOs,
  fail-closed conditional macro system with three outcome branches. Currently renders the
  labeled seed-0 fallback because `paper/e70_results.tex` (three-seed macros) does not exist yet.
- **Repo risk:** all E67–E70 docs, ~20 scripts, tests, and the paper rewrite are **uncommitted**
  (last commit 04dc60d, 2026-08-07).
- Artifacts live under `/data/robotixx/snmr-research/` (e70 students, analyses, video manifest,
  deployment exports). Frozen hash sets: 5 confirmation + 24 deployment + 12 paper-video.

---

## 0b. Update — 2026-08-11 evening (status deltas, cross-project findings, codex review)

**Status deltas since the plan was authored:**

- **A1 is DONE.** Commits `061cdec` (E67–E70 pipeline/docs/tests), `f3d6a44` (paper related-work
  + bibliography upgrade), `cf1f76e` (site E70 update + chart-bug fix) are pushed to
  `origin/main`; working tree clean. The `.venv/`, artifacts, and `.codex/` stay ignored.
- **Literature review and positioning are DONE** (`docs/LIT_REVIEW_AND_POSITIONING_2026-08-11.md`):
  no scoop exists as of 2026-08-11; 16 verified citations added to the paper (2026 landscape +
  measurement-methodology lineage); five venue upgrades applied. **A4 scope is now reduced to:**
  abstract trim toward ≤180 words, claim-language sweep, and the three branch builds. A4(b)
  (orthogonality sentence) is discharged.
- Paper rebuilt and verified after the citation upgrade: 6 pages, 0 overfull, 0 undefined
  references, fonts embedded, outcome-branch machinery untouched.

**H1 sharpened — the GPU tenants are now identified and are not equivalent:**

1. The **traversal-critic-research frozen 3-seed × 3-arm PPO matrix** (plus its scorer/sequencers).
   Its own frozen protocol and its 2026-08-11 execution report require outcome-blind, untouched
   completion — **interrupting or rescheduling any part of it now carries scientific cost to that
   project. It is off-limits**, exactly as tenant processes always were under Constraint 1.
   Note: its orchestration snapshot shows multi-day *scheduled waits* (one lane ~7 days, the
   policy-eval sequencer ~26 days), so natural free-memory windows may occur; the SNMR recovery
   supervisor will catch any window that crosses 26,000 MiB without anyone acting.
2. A **cosmos-framework job holding ~7,350 MiB** that is *not* part of any frozen protocol.
   This is the clean lever: pausing it at a checkpoint frees ≈30 GiB total — above the 26,000-MiB
   gate — for the ~2–4 h SNMR needs. **Recommended H1 decision: pause cosmos, never the matrix.**
   The decision and the pause itself remain human-only.

**Codex traversal-critic report — reviewed and verified.** The report
(`traversal-critic-research/docs/reviews/claude_fable_plan_execution_report_2026-08-11.md`) was
independently checked on 2026-08-11: all four SHA-256 hashes match on disk; every headline number
(0.2681 / 0.6661 / 0.6827 / 0.0402 / 0.5646; ratios 0.4749 / 1.2093) reproduces from
`temporal_shortcut_results.json`; `semantic_alignment_passed=false` confirmed over all 5 audited
strata; the E1 comparator protocol is genuinely prospective with correct guardrails. Verdict:
**accepted in full** — the E5 interface-mismatch finding is decisive, the reframing of Pearson
0.5646 as historical mismatched-interface evidence is correct, and leaving the frozen matrix
untouched is the right call. Directed next steps for that workstream are in Phase D below.

---

## 1. Global constraints (binding on the executing agent)

1. **Never weaken the 26,000-MiB launch gate. Never kill, pause, or nice any GPU tenant process.**
   Freeing the GPU is a human scheduling decision (Step H1). If the gate is unmet, do only
   Phase A work.
2. **No E70 tuning of any kind.** No threshold, clip, seed, checkpoint rule, endpoint, training
   budget, or interpretation may change after observing results. Accept whichever registered
   outcome branch the frozen analyzer produces (positive / explicit-pass-content-null /
   invalid-assay).
3. **All displayed E70 numbers must be machine-generated** from the final analyzer JSON via
   `paper/e70_results.tex`. Never transcribe a number by hand into the manuscript.
4. **No real robot commands, no hardware requests, no sim-to-real claims.** The safe95 ONNX +
   CPU loopback result stays out of the paper.
5. **Stop and preserve evidence** if: any frozen hash changes; a partial/unstable capture exists;
   the analyzer output lacks a registered seed or the 69 clusters; or a step would send commands
   outside loopback simulation. Report to the human before continuing.
6. **Phase A work must be provably incapable of influencing E70 outcomes** (no writes under
   `/data/robotixx/snmr-research/e70/students/`, no edits to frozen protocol/analyzer/launcher
   files, no GPU use).
7. Phase B steps execute **in the listed order**; do not reorder or parallelize across steps.

## 2. Reporting protocol (how the agent reports back)

After every completed step, append a dated entry to `docs/EXECUTION_PLAN_REPORTS.md` (create it
if absent) with exactly these fields:

```
## <step id> — <step name> — <UTC timestamp>
- Status: DONE | BLOCKED | STOPPED (stop-rule triggered)
- What was done: <2–4 sentences, concrete>
- Verification evidence: <commands run + observed outputs/hashes/paths>
- Validation judgment: <does it satisfy the step's validation criteria — yes/no + why>
- Deviations: <NONE, or exact description + justification>
- Next step: <id>
```

Never mark a step DONE without pasting verification evidence. Never continue past a STOPPED
entry without explicit human approval recorded in the same file.

---

## Phase H — Human decision (not for the agent to execute)

### H1. GPU scheduling decision

- **Decision:** whether to pause one traversal-critic/cosmos GPU job at a checkpoint boundary
  for one evening (~3.4 GiB must free to cross the gate) or keep waiting.
- **Agent's role:** monitor only. Report gate status (free MiB vs 26,000) in each report entry
  while blocked. Do not act on tenant processes under any circumstances.

---

## Phase A — GPU-blocked work (zero influence on E70; do all of these now, in order)

### A1. Commit and back up the repository — **DONE 2026-08-11 (see §0b)**

- **Objective:** eliminate the single-disk risk covering all E67–E70 work and the paper rewrite.
- **Design:** commit source, docs, scripts, tests, and paper to git; large generated artifacts
  stay excluded via the existing `.gitignore`. One commit for docs+scripts+tests, one for the
  paper rewrite, is acceptable granularity. Push to a **private** remote if one is configured or
  the human provides one; otherwise create a local bundle backup.
- **Implementation:**
  1. `git status --porcelain` — enumerate everything; confirm no `/data` paths or large binaries
     would be added (check `git check-ignore` on suspicious paths; anything >5 MB needs a reason).
  2. Stage and commit untracked docs/, scripts/, tests/, snmr/, and modified tracked files.
  3. `git bundle create /data/robotixx/snmr-research/backups/snmr-<date>.bundle --all` regardless
     of remote availability.
- **Verification:** `git status` clean except intentionally ignored artifacts; `git log --stat`
  shows the new commits; bundle file exists and `git bundle verify` passes.
- **Validation:** a fresh clone from the bundle/remote builds the paper PDF
  (`paper/` build script or tectonic) and passes the repo test suite (`pytest`, expected baseline:
  204 passed / 4 skipped, plus the newer E67–E70 tests).
- **Report:** commit SHAs, bundle path + verify output, test counts.

### A2. Temporal-block secondary analysis (preregister BEFORE seed-2 data exists)

- **Objective:** discharge the paper's promised robustness check against temporal dependence of
  the 69 overlapping ambiguity windows (Limitations §IX promises it; reviewers will demand it).
- **Design:** a *secondary* analyzer that does not modify the primary. Preregister exactly one
  rule before any new data lands, e.g.: partition the 69 pairs into non-overlapping temporal
  blocks per clip (fixed block length chosen from window geometry, stated in the doc); recompute
  A−T and A−S with a cluster/block bootstrap at the block level; same seed-level hierarchy as the
  primary. The rule, block definition, and pass/consistency language are frozen in a new doc
  `docs/E70_SECONDARY_TEMPORAL_BLOCK_ANALYSIS.md` **before** the script ever sees seed-2 data.
  It reads only existing per-rollout eval JSONs under `/data/robotixx/snmr-research/e70/students/`
  (read-only) — CPU only.
- **Implementation:** write `scripts/analyze_e70_temporal_blocks.py` + unit tests
  (`tests/test_e70_temporal_blocks.py`) using synthetic fixtures; then run it once on seed-0
  (and seed-1) data as a smoke, labeling output `secondary_seed01_preview.json` (clearly marked
  non-final). Record the script's SHA-256 in the preregistration doc.
- **Verification:** tests pass; script refuses to run if given a partial seed set without an
  explicit `--preview` flag; output JSON includes input-file hashes; preregistration doc committed
  with the script hash before seed-2 completion.
- **Validation:** the frozen rule is unambiguous enough that two independent readers would
  implement the same blocks; the preview direction is consistent with the primary seed-0 result
  (if it is not, that is a *finding to report*, not a reason to change the rule).
- **Report:** doc path, script hash, preview summary numbers labeled non-final.

### A3. Paired-effect figure pipeline (generated, never hand-drawn)

- **Objective:** use the 2-page headroom for uncertainty visualization: a forest-style figure of
  per-seed and per-clip A−T and A−S paired effects with 95% intervals, plus the aggregate.
- **Design:** a script that consumes the analyzer JSON schema (validate against
  `/data/robotixx/snmr-research/e70/analysis_seed0.json`) and emits either a PDF/PGF figure or
  TikZ coordinates written into a generated `.tex` fragment, wired with the same fail-closed
  outcome-branch conditionals as `e70_results.tex`. It must refuse non-final analyses unless
  `--preview` is passed.
- **Implementation:** `scripts/render_e70_effect_figure.py` + test; integrate an `\ifethreeseed`
  guarded `\input` in `main.tex` so the figure appears only when final macros exist; build a
  preview from seed-0 to check layout, but do not leave preview output in the submitted tree.
- **Verification:** paper compiles in both states (fragment absent → current 6-page build
  unchanged; fragment present from seed-0 preview → figure renders, page count ≤8, no overfull
  boxes).
- **Validation:** figure is legible in grayscale, labels match Table I arm names, and every
  plotted number traces to the analyzer JSON (spot-check three values).
- **Report:** script path + hash, both build results, page counts.

### A4. Manuscript polish pass (language only — no claim changes)

- **Objective:** tighten the abstract and harden claim language before final numbers land.
- **Design:** (a) trim abstract from ~211 rendered words toward ≤180 by compressing the
  single-clip diagnostic middle; (b) sharpen one sentence in Related Work stating explicitly that
  capability benchmarking (ULTRA/HOVER-class) is orthogonal to representation measurement;
  (c) verify scope phrases everywhere: "evaluated teacher checkpoint" (never teacher-algorithm
  parity), variant-level CVAE wording, time as matched null not upper bound, E70 scope = two
  LAFAN1 walks / Unitree G1 / frozen SNMR endpoint / this controller class.
- **Implementation:** edit `paper/main.tex`; rebuild; `grep` sweep for banned phrases
  (`within seed noise`, `pure DAgger`, `at most timing`, `statistically indistinguishable`,
  `never track worse`, `phase clock`) — must remain 0 hits.
- **Verification:** PDF rebuilds at ≤6 pages (pre-figure), fonts embedded and subset, 0 banned
  phrases, abstract word count reported, all three outcome branches still compile
  (build each branch as the existing branch-build scripts do).
- **Validation:** read the abstract and contributions aloud-check: every claim is scoped, the
  instrument framing ("exclusivity makes the interface measurable") is the visible thesis.
- **Report:** word count before/after, page count, branch-build results.

### A5. Anonymous reproducibility bundle consolidation

- **Objective:** close the audit's partial P1 item: one place holding commands, environment pins,
  data counts, artifact hashes, and small frozen reports, anonymized.
- **Design:** a `reproducibility/` (or existing artifact dir) index: exact launch commands per
  E70 cell, env pins (Python 3.10.12 / PyTorch 2.13.0+cu130 / MuJoCo 3.11.0, GMR `bb1bbe40`,
  Holosoma `20699ffa`), the 69-pair manifest, analyzer + protocol hashes, and the negative-results
  ledger pointer. No author-identifying strings.
- **Implementation:** assemble from existing docs; add a small script or checklist that greps the
  bundle for names/emails/hostnames.
- **Verification:** anonymity grep clean; every referenced hash re-verified against disk; index
  links resolve.
- **Validation:** a third party with the bundle + repo could re-run the analyzer on the frozen
  eval JSONs and reproduce Table I numbers.
- **Report:** bundle path, hash-verification summary.

### A6. PaperPlaza dry run (human-in-the-loop for credentials)

- **Objective:** remove submission-day risk.
- **Implementation:** verify the live keyword taxonomy against the selected three
  (`Motion Retargeting`, `Human and Humanoid Motion Analysis and Synthesis`,
  `Machine Learning for Robot Control`); run the PaperPlaza PDF compliance check on the current
  build; stage author metadata (kept out of the PDF). The agent prepares everything and reports;
  the human performs any login-gated action.
- **Verification:** compliance-check output recorded; metadata checklist complete.
- **Report:** any taxonomy mismatch or compliance warning.

---

## Phase B — On gate opening (frozen order; each step gates the next)

### B1. Seed-2 completion under supervisors (monitor only)

- **Objective:** seed2_proprio restarts from round 0 under the unchanged launcher, then
  seed2_shuffled, both driven automatically by the recovery supervisor.
- **Design:** the agent does not launch anything manually; it watches
  `/tmp/snmr-e70-recovery-supervisor.log` and the output root.
- **Verification:** both cell directories appear under
  `/data/robotixx/snmr-research/e70/students/` with `*_student.pt`, `*_eval.json`,
  `*_eval_ambiguity.json`, complete `train.log` (no exit-143); quarantined round-1550 state
  remains untouched and unused.
- **Validation:** cells trained the registered round count from round 0; frozen confirmation
  hashes still match.
- **Report:** cell timestamps, eval file paths, hash-check result.

### B2. Frozen three-seed analyzer

- **Objective:** produce the single authoritative aggregate.
- **Verification:** final JSON contains exactly seeds 0/1/2, 69 paired clusters, all five arms,
  per-seed effects, per-clip A−T, hierarchical intervals, input hashes, and the frozen
  content-gate verdict.
- **Validation:** **inspect the analyzer output before any prose edit.** Determine which of the
  three registered branches it selects. Then (and only then) run the preregistered A2 secondary
  temporal-block analysis on the final data and record whether it is directionally consistent.
- **Report:** the verdict branch, headline intervals, secondary-analysis consistency statement.
  If any registered element is missing → STOPPED per stop rules.

### B3. Final paper generation

- **Objective:** replace the seed-0 fallback with generated three-seed macros.
- **Implementation:** run the frozen paper-value generator (`scripts/render_e70_paper_values.py`)
  to emit `paper/e70_results.tex` (it must stamp the analyzer SHA-256 and reject partial input);
  rebuild; the A3 effect figure now activates.
- **Verification:** every displayed E70 number is macro-driven (grep the tex for hard-coded
  values in result-bearing passages); page count ≤8 incl. references; fonts embedded; anonymous;
  the active outcome branch's language matches B2's verdict exactly; abstract/teaser/table/
  limitations agree.
- **Validation:** cross-read Table I, Fig. 1c, the effect figure, and the abstract against the
  analyzer JSON directly — three independent spot checks per surface.
- **Report:** analyzer hash embedded in macros, page count, branch, spot-check table.

### B4. Video pipeline (presentation-only, after B3)

- **Objective:** the optional ICRA video, fully provenance-bound.
- **Implementation (frozen order):** render exactly the six manifest captures
  (`autoresearch/iterate-260809-0351/e70_video_manifest.json`) — no substitutions for
  visually-awkward rollouts; index; compose the registered 70-s storyboard (1920×1080, 30 fps,
  H.264/yuv420p, ≤19,000,000 bytes); generate contact sheet; **a human watches the full MP4**;
  record the review via `scripts/record_e70_visual_review.py`; run
  `scripts/audit_e70_final_bundle.py`.
- **Verification:** all binding hashes (starts, checkpoint, teacher, motion, evaluator, camera,
  report, raw video) recorded; terminated rollouts labeled as such; aggregate cards populated
  only from the final analyzer; result card fail-closes on the explicit gate; size/format gates
  pass; bundle auditor passes.
- **Validation:** the human visual review checklist (framing, label accuracy, reset leakage,
  clipping, misleading synchronization) is complete — `POSTPROCESS_COMPLETE` alone is not
  acceptance.
- **Report:** MP4 + contact-sheet hashes, auditor output, review record path.

### B5. Submission package

- **Objective:** submit.
- **Implementation:** final PDF compliance check; upload paper + optional video within the
  window (target the Sep 9 window if possible); enter author metadata in PaperPlaza (human);
  archive the complete bundle (paper source + PDF, analyzer JSON, macros, video, hashes,
  reports) under `/data/robotixx/snmr-research/` and commit the source-side artifacts.
- **Verification:** PaperPlaza confirmation; archived bundle passes `audit_e70_final_bundle.py`
  re-run; final git tag created.
- **Report:** submission confirmation ID (human-provided), tag name, archive path.

---

## Phase C — Post-submission (DO NOT START before B5 completes)

Listed only so the agent does not misinterpret them as current work:

1. Preregistered held-out multi-trajectory generalization of the ambiguity assay (the named next
   study in `docs/RESEARCH_GOAL_2026-08-10.md`).
2. Teacher-seed replication (currently one teacher training seed backs the "evaluated teacher
   checkpoint" wording).
3. E65b deterministic+matched-noise arm (registered, never run).
4. Deployment robustness matrix preregistration (latency/dropout, encoder bias, gains, friction,
   mass/CoM, pushes) with the same safety handoff; hardware stages remain separately approved.
5. E54/T1 Holosoma config restore.

---

## Phase D — Codex workstream guidance (traversal-critic-research)

Review basis: the verified 2026-08-11 execution report (§0b). These steps direct that project's
agent; they are ordered so all CPU-only integrity work lands before any GPU spend, and so that
project's GPU use never collides with SNMR's E70 window. The same reporting protocol as §2
applies, written to that repo's `docs/reviews/` directory.

### D1. Canonical E5 amendment (CPU, do first)

- **Objective:** freeze the record of the temporal-interface mismatch before any critic code
  changes, exactly as the report's action 2 proposes.
- **Design:** one dated amendment doc: the four-frame v5 SFT reality vs the 4–48-frame historical
  validation vs the 5–32-frame policy-scorer interface; C1 wording flagged as requiring revision;
  the historical 0.5646 preserved verbatim as "historical mismatched-interface evidence," never
  overwritten. Bind the three evidence hashes from `autoresearch/run-260811-1753/`.
- **Verification:** amendment committed; no critic source file modified before its commit
  timestamp; hashes restated and re-checked in the doc.
- **Validation:** a reader of the amendment alone can state what the 0.5646 does and does not
  estimate.

### D2. Preregister the expanded shortcut battery BEFORE running it (CPU)

- **Objective:** convert the post-hoc 8×8 ridge finding into registered evidence (report
  action 6) with the same discipline SNMR's E70 used.
- **Design:** freeze in one doc, before execution: readout set (duration-only, terminal-frame-only,
  duration+terminal, onset-aligned prefix, endpoint-masked, shuffled-label floor),
  scene-clustered interval method, train-only lambda selection, and the exact comparison language
  against any critic. State explicitly that cross-model-class ratios (e.g., "120.9%") are
  descriptive, not variance-explained, and that the corrected critic generation must be
  re-benchmarked against this same frozen battery.
- **Verification:** doc + script SHA-256 recorded before the first registered run; the current
  8×8 row remains labeled post-hoc descriptive everywhere it appears.
- **Validation:** the battery becomes the paper's "readout challenge" table — main text, not
  appendix.

### D3. Shared temporal materializer + tensor-equality tests (CPU)

- Report action 3, unchanged: one materializer returning selected frames plus original source
  indices and source fps; every route calls the processor with `do_sample_frames=False` and
  explicit metadata; tensor-equality tests cover the five E5 strata. **Verification:** tests
  green; byte-identical tensors across routes on all five strata.

### D4. E1 factorial re-evaluation of the eight exports (GPU — scheduled, small)

- Report action 4 under the frozen 2×2 protocol. **Scheduling rule:** inference-only, so run it
  opportunistically, but never start a chunk that would hold GPU memory during (a) a frozen-matrix
  lane run or (b) an open SNMR E70 window (the E70 supervisor needs 26,000 MiB free for ~2–4 h
  and has first claim on the first available window — it is submission-deadline-bound;
  E1 re-evaluation is not).

### D5. Corrected critic generation (GPU — after the matrix completes and E70's cells finish)

- Report action 5: separately named generation through the shared materializer; one-batch tensor
  audit sealed before training; never mixed into v5 tables; benchmarked against the D2 frozen
  battery. This is the largest GPU consumer in either project — it goes last.

### D6. Sequencing and venue

- E2 / E6 only after D1 and the matrix-critical path (report action 7). Venue: ICLR primary,
  as the report concludes; an ICRA branch-A submission only if the frozen policy result closes
  early *and* the corrected temporal story is complete — and it must not contend with SNMR's
  E70/paper window for GPU or attention before 2026-09-15.

**Cross-project GPU rule (binding on both agents):** the frozen traversal-critic matrix is
untouchable; the SNMR E70 supervisor has first claim on the first ≥26,000-MiB window; new
discretionary GPU work in either project must check both conditions before starting.

---

## Phase E — Sim2sim deployment validation (user-directed 2026-08-12; MuJoCo is the deploy target)

Authorized by the owner on 2026-08-12: "our final target is to actually deploy our model in real
G1 robot ... make sure our motion tracking works in sim2sim ... aim mujoco as the target deploy
platform." This is engineering validation toward `docs/REAL_WORLD_DEPLOYMENT_PLAN.md` stages 2–4.
It changes no paper claim (the paper boundary — simulation only, no sim-to-real claims — is
unchanged), uses CPU only, and touches no file in either frozen hash manifest. The full
preregistered robustness matrix stays in Phase C; Phase E extends the already-passed stage-2/3
gates from one seed-0 candidate and one motion to the post-confirmation candidates and both
registered walks.

**E-rule (recorded before any run): deployment-candidate selection.** One candidate per arm,
chosen by *median* `completion_rate` in the frozen per-seed general evals
(`students/seed*_{explicit,snmr}/*_eval.json`, evaluation_seed 404, 1024 rollouts); ties break to
the lower seed index. Applied: explicit → **seed 2** (0.9238 of 0.9248/0.9199/0.9238);
SNMR → **seed 1** (0.7021 of 0.6846/0.7021/0.7090). No visual selection at any stage.

- **E-1 Export.** `scripts/export_e70_policy_onnx.py --safety-limit-fraction 0.95` for
  {explicit seed2, snmr seed1} × {walk1_subject1, walk1_subject5} →
  `exports/sim2sim_2026-08-12/`. Gate: `onnx.checker` + ONNX Runtime parity ≤1e-5.
- **E-2 Runtime contract.** `scripts/validate_e70_runtime_contract.py` per export; gate: zero
  non-finite/deadline/hard-limit/envelope violations on the 500-step production contract.
- **E-3 Loopback qualification.** `scripts/run_e70_loopback_qualification.py --safety-handoff`
  ×3 identical repeats per (candidate × motion), serially (shared DDS domain), then
  `scripts/summarize_e70_loopback_repeats.py` fail-closed summaries. Gate: all repeats pass all
  phase safety checks. Expectation note recorded in advance: the SNMR arm is the research
  interface at ~0.70 general completion; a loopback fall for SNMR is a *reportable measurement*,
  not a campaign failure — deployment eligibility is only ever claimed for the explicit arm.
- **E-4 Review videos + tracking metric.** New non-frozen capture script (separate from the
  frozen paper-video pipeline, which stays untouched under the B4 STOPPED entry) reruns the same
  loopback lifecycle with offscreen rendering → mp4 per candidate × motion in
  `exports/sim2sim_review/`, and logs joint-position tracking RMSE against the embedded reference
  (engineering metric, not preregistered, never enters the paper).
- **E-5 Report.** `docs/SIM2SIM_VALIDATION_2026-08-12.md` with hashes for every ONNX, report, and
  summary; update the stage table in `docs/REAL_WORLD_DEPLOYMENT_PLAN.md`; report per §2.

---

## Positioning notes for any writing step (context, not tasks)

- **Thesis:** the contribution is a *measurement instrument* — the exclusivity contract plus
  matched controls (time, proprioception, phase-matched shuffle) plus causal command destruction
  plus preregistration. Not a capability claim. Do not widen it.
- **Differentiation:** ULTRA (arXiv:2603.03279) and RoboGhost (arXiv:2510.14952) demonstrate
  latent humanoid command channels but do not isolate what the latent carries; both are cited.
- **Best "so what" evidence:** the single-clip diagnostic where the time-index (0.754) beats the
  latent (0.656) and retires a favorable prior claim — keep prominent.
- **Expected attacks:** breadth (answer: scoping + Phase C study named in limitations);
  no external baselines (answer: orthogonality sentence, A4b); window dependence (answer: A2).
- **Outcome honesty:** per-seed descriptive A−T (+0.156/+0.275/+0.145) suggests the aggregate
  passes, but seed-2 A−S does not exist yet. If the final A−S interval includes zero, the
  registered answer is the explicit-pass/content-null branch — accept it without flinching; the
  paper builds and remains publishable.
