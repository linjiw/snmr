# E70 Simulation Video Protocol

**Recorded:** 2026-08-09, while the frozen seed-1/2 confirmation queue was running.  This is a
presentation protocol, not a new behavioral endpoint.  It does not change E70's models, starts,
statistics, or interpretation gates.

## Purpose and submission contract

The video should make the paper's controlled comparison visually legible: the same robot, motion
pair, start state, camera, horizon, and playback rate under explicit, SNMR, time, and destroyed-code
interfaces.  Aggregate results remain the evidence; individual rollouts are illustrations.

The current official ICRA 2027 contributed-paper limits are one `mpeg`, `mp4`, or `mpg` file,
maximum 180 seconds and 20 MB, minimum height 480 px, minimum 20 fps, progressive scan.  The two
upload windows are August 5--September 9 and September 17--22, 2026.  Source:

- https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/

Our stricter production target is anonymous, 16:9, 1920x1080, 30 fps, H.264/yuv420p, progressive,
at most 120 seconds and 19 MB.  Use no author names, institution marks, copyrighted music, or
unlicensed motion footage.

## Frozen exemplar rule

`scripts/prepare_e70_video.py` selects a window using only the GMR reference precheck: nearest to
the median selected-window future distance, then median current-state distance, then frame index.
It does not inspect completion, survival, tracking error, or video appearance.  A frozen ambiguity
report is read only to translate pair/side identity into the exact global simulator start.

The resulting manifest is
`autoresearch/iterate-260809-0351/e70_video_manifest.json`:

- pair `walk1_subject1,walk1_subject5`, pair index 1;
- `walk1_subject1` start 33 (reference time 0.667 s);
- `walk1_subject5` start 13,168 (local reference time 2.033 s);
- fixed training seed 0, selected by index rather than outcome;
- six captures: A/SNMR and T/time on both clips, then clean C/explicit and
  marginal-random-destroyed C on the first clip.

Reproduce the selection with:

```bash
source scripts/activate_snmr.sh
python scripts/prepare_e70_video.py \
  --precheck autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json \
  --ambiguity-report /data/robotixx/snmr-research/e70/students/seed0_explicit/c_prior_explicit_eval_ambiguity.json \
  --student-root /data/robotixx/snmr-research/e70/students \
  --seed 0 \
  --out autoresearch/iterate-260809-0351/e70_video_manifest.json
```

## Capture and edit plan

Capture starts only after the three-seed E70 queue and final analyzer finish, so rendering cannot
perturb or delay the confirmatory run.  `scripts/run_e70_video.sh` enforces both completion files
before it launches. Use MuJoCo-Warp's headless EGL renderer at 1920x1080 with
a frozen pelvis-tracking Cartesian camera: position offset `[2.0, 2.0, 1.0]` m, target offset
`[0.0, 0.0, 0.3]` m, smoothing 0.95, and 45-degree vertical FOV.  Disable Holosoma's
velocity-command overlay because it does
not describe the WBT command; add method, clip, start, elapsed time, and termination state in post.
Keep identical camera parameters, real-time playback, and crop across methods.  Retain every raw
capture and its command log under `exports/e70_video/raw/`; the directory is generated and ignored
by git.

Capture uses the dedicated `scripts/eval_e70_video.py`, leaving the hash-frozen confirmation
trainer and distillation utilities byte-identical while seeds are pending.  Five clean captures
simulate and render one environment.  The marginal-random intervention renders env 0 but retains
1,023 deterministic companion starts (1,024 total), because the registered intervention estimates
each command dimension's contemporaneous batch marginal; a one-sample standard deviation is
undefined and would produce a falsely broken illustration.  The capture report records both the
single illustrated rollout and the intervention-pool size.
The executable paper/video path is separately frozen in
`autoresearch/iterate-260809-0351/e70_video_code_hashes.json`; both the capture launcher and
post-processing supervisor fail closed if any listed evaluator, indexer, compositor, manuscript,
or selection-manifest file drifts before rendering.

Implemented 70-second storyboard (the composer durations are the source of truth):

| Time | Content | Scientific role |
| ---: | --- | --- |
| 0--6 s | Question and exclusive-interface contract | Define what is measured |
| 6--13 s | Reference-only two-walk ambiguity assay | Explain the controlled start selection |
| 13--33 s | A/SNMR versus T/time, both clips, synchronized panels | Show the matched comparison |
| 33--43 s | Clean explicit versus matched-marginal code destruction | Show causal channel use |
| 43--62 s | Three-seed C/A/T/B/S completion and paired intervals | Put examples behind aggregate evidence |
| 62--70 s | Deployment boundary and next gates | Separate simulation evidence from hardware claims |

If the frozen exemplar is visually uninformative, do not replace it with a better-looking rollout.
Use a short grid of multiple fixed-index windows or rely on aggregate animation instead.  Any
failed rollout remains visible through its first termination; do not loop a reset as if it were a
continuous success.

Render and compose with:

```bash
bash scripts/run_e70_video.sh
source scripts/activate_snmr.sh
python scripts/compose_e70_video.py \
  --manifest autoresearch/iterate-260809-0351/e70_video_manifest.json \
  --analysis /data/robotixx/snmr-research/e70/analysis_seed0-1-2.json \
  --capture-index exports/e70_video/raw_capture_index.json \
  --raw-dir exports/e70_video/raw \
  --out exports/e70_video/snmr_e70_icra.mp4
```

The capture launcher validates exact starts, student and teacher checkpoint hashes, motion hashes,
evaluator-code provenance, report identity, intervention-pool size, camera settings,
executed-step/frame-count agreement, and raw-media hashes into
`exports/e70_video/raw_capture_index.json`.  The composer requires that index, labels every panel
as completed or terminated with its simulated survival time, fills all aggregate cards from the
analyzer JSON, writes a contact sheet and validation JSON, and
fails unless the result is H.264/yuv420p, progressive, at least 480 px and 20 fps, at most 180 s,
and at most 20 MiB.  A synthetic 70-s 1080p/30 smoke passed this complete media path on 2026-08-09;
it is tooling validation only and is not retained as scientific footage.

## Acceptance checks

- Raw start, checkpoint SHA-256, camera config, frame count, and terminal step agree with the
  manifest and capture report.
- Every side-by-side panel is synchronized in simulation time and uses the same speed.
- Labels distinguish `simulation`, `seed-0 illustration`, and `three-seed aggregate`.
- The final three-seed numbers are read from the frozen analyzer, never typed from memory.
- `ffprobe` confirms H.264, yuv420p, progressive, at least 480 px, at least 20 fps, no more than
  180 seconds, and no more than 20 MB.
- A frame-contact sheet is visually checked for clipping, resets, mislabeled arms, and unreadable
  text before upload.
- The full 70-second MP4 is also watched end to end.  Record the reviewer, UTC timestamp, exact
  MP4/contact-sheet hashes, and passing booleans for framing/camera tracking, labels/outcomes, reset
  leakage, clipping/readability, and misleading synchronization in
  `exports/e70_video/snmr_e70_icra_visual_review.json` using
  `scripts/record_e70_visual_review.py`; do not transcribe the hashes manually.
- Only then run `scripts/audit_e70_final_bundle.py`.  It recomputes the analyzer-derived paper
  macros, cross-binds analysis, capture index, raw captures, final video, contact sheet, code
  freeze, and visual review, replays every indexed raw-video/report/checkpoint hash, requires
  explicit full-video confirmation, checks PDF pages/media/fonts, and atomically writes the final
  bundle.

The final certification command is:

```bash
source scripts/activate_snmr.sh
python scripts/record_e70_visual_review.py \
  --video exports/e70_video/snmr_e70_icra.mp4 \
  --contact-sheet exports/e70_video/snmr_e70_icra_contact_sheet.png \
  --reviewer '<reviewer identity>' \
  --confirm-full-video-watched \
  --pass-framing-and-camera-tracking \
  --pass-labels-and-outcome-status \
  --pass-no-reset-leakage \
  --pass-no-clipping-or-unreadable-text \
  --pass-no-misleading-synchronization \
  --out exports/e70_video/snmr_e70_icra_visual_review.json

python scripts/audit_e70_final_bundle.py \
  --repo-root . \
  --analysis /data/robotixx/snmr-research/e70/analysis_seed0-1-2.json \
  --paper-values paper/e70_results.tex \
  --paper /data/robotixx/snmr-research/paper-build-final/main.pdf \
  --video exports/e70_video/snmr_e70_icra.mp4 \
  --video-validation exports/e70_video/snmr_e70_icra_validation.json \
  --capture-index exports/e70_video/raw_capture_index.json \
  --manifest autoresearch/iterate-260809-0351/e70_video_manifest.json \
  --contact-sheet exports/e70_video/snmr_e70_icra_contact_sheet.png \
  --visual-review exports/e70_video/snmr_e70_icra_visual_review.json \
  --code-manifest autoresearch/iterate-260809-0351/e70_video_code_hashes.json \
  --out /data/robotixx/snmr-research/e70/final_paper_video_bundle.json
```
