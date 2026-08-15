# E72 — Latent-Substitution Motion Generator (source intervention on `latent_z`)

Status: **generator built and CPU-verified; no evaluation has been run.** The GPU was occupied by
the B4 video pipeline while this was produced, so this document records only what was built,
verified on CPU, and pre-registered. Every number below that is not a hash comes from a CPU check.

Date: 2026-08-14.
Generator: `scripts/build_latent_substitution_motions.py`
Verifier: `scripts/verify_latent_substitution_motions.py`
Tests: `tests/test_latent_substitution_motions.py` (24 tests, all passing, CPU-only)
Output root: `/data/robotixx/snmr-research/e72_latent_sub/` (NEW root; nothing under
`/data/robotixx/snmr-research/e70/` was written, moved, or deleted)

---

## 1. Mechanism — why this needs zero edits to any frozen file

The student's latent input is read from the motion NPZ, not from any student-side artifact:

* `snmr/integration/wbt_latent.py:53-62` (`_load_latent_npz`) opens the WBT motion NPZ and takes
  the `latent_z` field verbatim; `_load_multi_motion_latent` (`:65-98`) concatenates the per-clip
  latents in the loader's own sorted-glob order and asserts each clip's latent length equals that
  clip's frame count, so a length mismatch fails loudly rather than misaligning.
* `_gather_at_offsets` (`:124-136`) indexes that array at `time_steps + offset`, clamped at the
  clip end.
* `scripts/train_e52_dagger.py:236` obtains the latents through `_ensure_latent_loaded`, and in
  **eval-only** mode (`:265-270`) overwrites `z_mean`/`z_std` with the values stored in the student
  **checkpoint**. Standardization therefore cannot be contaminated by whatever latents the
  evaluation environment happens to load.

Consequence: an intervention on the latent can be delivered **at the source** — by writing a new
motion directory whose NPZs are identical to the frozen originals in every field except `latent_z`
(identical physics, identical reference target, identical clip and file names) and pointing the
frozen eval-only path at that directory with
`--command.setup-terms.motion-command.params.motion-config.motion-dir`. No frozen file is edited,
the frozen student checkpoint is reused as-is, and the frozen ambiguity precheck still validates
because the clip names it checks are derived from the file names, which are preserved exactly
(`scripts/train_e52_dagger.py:584-600`: `pair_report["clips"] != clip_names` is fatal;
`clip_names` comes from the teacher manifest matched against the sorted motion file stems at
`:186-217`, `stem.startswith(clip)`).

## 2. Frame rate — derived, not assumed

Both frozen motions declare `fps = 50` (int64, shape `(1,)`) inside the NPZ itself. The generator
reads that field, requires both clips to agree, and aborts if the value is not the expected 50.
The consumer agrees: `MotionLoader` exposes `motion.fps`, which the ambiguity path uses to convert
the precheck's `time_seconds_*` into frames (`train_e52_dagger.py:604-608`), and
`wbt_latent.PREVIEW_OFFSETS` / `Z_OFFSETS` are documented in 50 Hz steps
(`Z_OFFSETS = (0, 5)` = "current + 0.1 s at 50 Hz").

Note the precheck records `source_fps: 30.0` — that is the LAFAN1 *source* rate of the human clip,
not the retargeted WBT motion rate. The motion files used by the loader are 50 Hz. Nothing in this
protocol depends on 30 Hz.

Frame offsets are obtained by **truncating the magnitude** of the nominal offset, never rounding
up, so the realized misalignment is never larger than its label:

| arm | nominal | frames at 50 Hz | realized seconds |
|---|---|---|---|
| `shift_m0250` | −0.25 s | **−12** (⌊0.25·50⌋ = 12, since 12.5 is not integral) | −0.24 s |
| `shift_p0250` | +0.25 s | **+12** | +0.24 s |
| `shift_p0500` | +0.50 s | **+25** (exact) | +0.50 s |

## 3. Arms

Each arm is one motion directory containing **both** clips under their **original filenames**
(`walk1_subject1_mj_z.npz`, `walk1_subject5_mj_z.npz`), so loader order, clip names, teacher
routing, and the ambiguity precheck are all unchanged.

| arm | `latent_z` substitution |
|---|---|
| `control` | copied through unchanged — byte-identical to the source |
| `shift_m0250` | `out[t] = z[clip(t − 12, 0, T−1)]` (latent lags the physics by 0.24 s) |
| `shift_p0250` | `out[t] = z[clip(t + 12, 0, T−1)]` (latent leads by 0.24 s) |
| `shift_p0500` | `out[t] = z[clip(t + 25, 0, T−1)]` (latent leads by 0.50 s) |
| `first_frame` | `z[clip_start]` broadcast across the clip (static code) |
| `clip_mean` | `mean_t z[t]` (accumulated in float64, cast once to float32) broadcast across the clip |

Every other array — `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
`body_lin_vel_w`, `body_ang_vel_w`, `joint_names`, `body_names` — is copied through unchanged and
was verified bit-identical (dtype, shape, raw bytes) after writing. `latent_z` stays `float32`
with shape `(13066, 128)` in every arm.

### Boundary safety (by construction)

Each clip is its own NPZ, so a clip is exactly one file's frame range `[0, T)`. Shifts are
computed as `index = clip(arange(T) + delta, 0, T−1)`, which

* never reads across a clip boundary (there is no other clip in the file),
* never wraps (no negative index, no modular arithmetic),
* is well defined at both edges: the first `|delta|` frames of a negative shift hold the clip's
  **first** latent, and the last `delta` frames of a positive shift hold the clip's **last**
  latent.

This matters because the frozen `wbt_latent._gather_at_offsets` applies an **upper** clamp only
(`torch.minimum(current + offset, end)`) and no lower clamp; a negative index would silently wrap
in the gather. The generator makes that impossible: the written arrays already contain the
edge-held values, and the runtime offsets `Z_OFFSETS = (0, 5)` are non-negative.

## 4. Inputs (verified before reading)

| file | SHA-256 |
|---|---|
| `/data/robotixx/snmr-research/e70/motions/walk1_subject1_mj_z.npz` (→ `e69/motions/…`) | `b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa` |
| `/data/robotixx/snmr-research/e70/motions/walk1_subject5_mj_z.npz` (→ `e67/motions/…`) | `d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51` |

Both: 13066 frames, `latent_z (13066, 128) float32`, `fps = 50`. These are the same hashes
`scripts/run_e70_multitraj.sh` gates on.

Source `latent_z` content hashes (SHA-256 of the raw array bytes):
`walk1_subject1` `0b378a9f98159393fdbff7a123b616274ee413835c863d578ac54c98612567b3`,
`walk1_subject5` `6de39a4ee9e3c6997e047bf38eb208298dec8a4786a32ff378fb45471348ad4d`.

## 5. Outputs

`/data/robotixx/snmr-research/e72_latent_sub/motions/<arm>/{walk1_subject1_mj_z.npz,
walk1_subject5_mj_z.npz}` (693 MB total), plus the machine-readable
`/data/robotixx/snmr-research/e72_latent_sub/latent_substitution_manifest.json`.

NumPy writes NPZ members with a fixed 1980-01-01 zip timestamp, so the whole-file hashes below are
reproducible, not run-dependent. In particular the `control` files are byte-identical **whole
files**, hashing to the frozen source hashes.

| arm | frame offset | file SHA-256 `walk1_subject1_mj_z.npz` | file SHA-256 `walk1_subject5_mj_z.npz` |
|---|---|---|---|
| `control` | 0 | `b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa` | `d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51` |
| `shift_m0250` | −12 | `762c03952be563566e451e20cdedeb3a6f5b1e0bdb3659ceb1cc329f5ad40032` | `b5bdf547e8ac1fca20756502cb037fcd13f4204542d854ba0809f5d8d0064519` |
| `shift_p0250` | +12 | `38886fc9be20958132a8929a0c4cb0fa112eefa57b19aadb6cfdd624fef24f54` | `c9503fd68304a4f99afc463a662031a1f4cc785b8ebfc1308cb8b0151d0549dd` |
| `shift_p0500` | +25 | `5a3df6075bf1a5da2ad29bdd7a08c330a7f37a6bdaadc978e3707ce1142d92ea` | `4baa5d0c662148a267af0799e3b1df0e72eb9588742f2f5a8b30ab261033db85` |
| `first_frame` | — | `6fcb52b7d98fc85af7650593dc52022153f6b26b3f189dc4a2c7662d997474db` | `f890f8c158bb58e35c928403a4fd3d18490cc7378d5f9ae08bc6ae41d47053f7` |
| `clip_mean` | — | `3a33f1e11391198a35be9f15de9a81c2b1985068540cc720f50967fffc27878e` | `0cbb4fc2fb1973dee9ceae4a94415db980056baf075704e2e3c2c3a23e4904e7` |

Per-arm L2 (Frobenius) distance between substituted and original `latent_z`, from the standalone
verifier (`ALL CLIPS` = √(sum of squared per-clip distances)):

| arm | walk1_subject1 | walk1_subject5 | all clips | frames changed / clip |
|---|---|---|---|---|
| `control` | **0.000000** | **0.000000** | **0.000000** | 0 / 13066 |
| `shift_m0250` | 28.864072 | 32.338412 | 43.346367 | 13065 / 13066 |
| `shift_p0250` | 28.867220 | 32.338363 | 43.348427 | 13065 / 13066 |
| `shift_p0500` | 39.279159 | 42.374077 | 57.779016 | 13065 / 13066 |
| `first_frame` | 66.131251 | 60.929243 | 89.920603 | 13065 / 13066 |
| `clip_mean` | 39.508631 | 29.944294 | 49.574112 | 13066 / 13066 |

Verifier result: `OK: 6 arms verified; every non-latent array bit-identical to source`.

## 6. Registered mandatory control (run this arm FIRST)

The `control` arm is the δ = 0 arm. Its motion files are byte-identical to the frozen E70 motions,
so an eval-only ambiguity run over it is the *same computation* as the frozen E70 SNMR ambiguity
evaluation, merely reading from a different path.

**Registration: the `control` arm must reproduce the frozen SNMR ambiguity completion**

> **0.7542317708333334** (seed-macro over seeds 0/1/2, `reproducibility/reports/e70_seed0-1-2_analysis.json` → `arms.snmr.ambiguity_completion`)

with the per-seed values it is composed of:

| seed | frozen `a_prior_snmr_eval_ambiguity.json` completion | rollouts | eval seed |
|---|---|---|---|
| 0 | 0.7646484375 | 1024 | 404 |
| 1 | 0.7607421875 | 1024 | 404 |
| 2 | 0.7373046875 | 1024 | 404 |

**No other arm may be interpreted until the control reproduces these numbers.** A deviation is not
a result: it means the evaluation is not deterministic across the intervening software/hardware
state, in which case every δ-arm difference is uninterpretable and the whole comparison must be
re-run within a single session with the control included.

## 7. Registered interpretation asymmetry (pre-committed, before any result is seen)

> **Static-code RETENTION is evidence for clip identity. Static-code COLLAPSE is NOT evidence for
> time-varying content, because a constant z is outside the student's training input distribution,
> so collapse is confounded with distribution shift. Only the delta arms, which feed a real latent
> trajectory that is merely misaligned, are two-sided.**

Practical reading of each arm under that rule:

* `first_frame`, `clip_mean` — if performance is **retained**, a single static code suffices to
  select the clip, i.e. the latent is carrying clip identity rather than per-frame content. That
  conclusion is licensed. If performance **collapses**, nothing is licensed: a constant
  `z`-window is off-distribution for a student trained only on real latent trajectories
  (`Z_OFFSETS = (0, 5)` always saw `z_t ≠ z_{t+5}`), so collapse is confounded with distribution
  shift and must not be reported as evidence that the latent carries time-varying content.
* `shift_m0250`, `shift_p0250`, `shift_p0500` — two-sided. The input remains a genuine latent
  trajectory with the same marginal statistics and the same local temporal structure; only its
  alignment to the physics is wrong. Degradation that grows with |δ| is evidence that the latent
  is read for time-aligned content; insensitivity across δ up to 0.5 s is evidence that only the
  slowly-varying/identity component is used.
* `control` — mandatory determinism check, not a scientific arm (see §6).

## 8. Command line for the later GPU run (DO NOT RUN NOW — GPU is occupied)

Derived from `scripts/run_e70_multitraj.sh` (`student_command`, `eval_mode = ambiguity`, arm
`a_prior_snmr`, tag `snmr`, `run_seed = 404`), with three deliberate deviations:

1. `motion-dir` points at the arm's substituted motion directory instead of `$E70_ROOT/motions`;
2. `E52_OUT` and `--logger.base-dir` point under the new E72 root — the eval writes
   `<arm>_eval_ambiguity.json` into `E52_OUT`, and **nothing may be written under
   `/data/robotixx/snmr-research/e70/`**, so the frozen student checkpoint is *copied* out first;
3. the GPU capacity gate from the E70 script (`≥ 26000 MiB free`) must be applied by hand, since
   `run_e70_multitraj.sh` itself must not be invoked.

```bash
source /home/robotixx/snmr/scripts/activate_snmr.sh      # sets SNMR_HOLOSOMA_ROOT, WBT_PYTHON

ARM=control            # then shift_m0250 shift_p0250 shift_p0500 first_frame clip_mean
SEED=0                 # repeat for 1 and 2 to recover the seed-macro
E72_ROOT=/data/robotixx/snmr-research/e72_latent_sub
OUT="$E72_ROOT/students/seed${SEED}_snmr_${ARM}"
mkdir -p "$OUT" "$E72_ROOT/student_holosoma_logs"

# frozen student, copied (never evaluated in place: E52_OUT is written to)
cp /data/robotixx/snmr-research/e70/students/seed${SEED}_snmr/a_prior_snmr_student.pt "$OUT/"

cd "$SNMR_HOLOSOMA_ROOT"
env \
  E52_ARM=a_prior_snmr \
  E52_TEACHER_MANIFEST=/data/robotixx/snmr-research/e70/teacher_manifest.json \
  E52_OUT="$OUT" \
  E52_ROUNDS=2000 E52_DET=1 E52_PHASE_ONLY=0 E52_SHUFFLE_LATENT=0 \
  E52_REPLAY_ROUNDS=4 E52_TEACHER_FLOOR=0.1 E52_TEACHER_ANNEAL_ROUNDS=200 \
  E52_CKPT_EVERY=50 E52_VAL_EVERY=50 E52_VAL_SAMPLES=4096 E52_BEST_AFTER=50 \
  E52_EVAL_ONLY=1 \
  E52_EVAL_STARTS_JSON=/home/robotixx/snmr/autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json \
  PYTHONPATH=/home/robotixx/snmr \
  nice -n 10 "$WBT_PYTHON" /home/robotixx/snmr/scripts/train_e52_dagger.py \
    exp:g1-29dof-wbt simulator:mjwarp logger:disabled \
    --training.num-envs 1024 \
    --training.seed 404 \
    --training.headless True \
    --training.name "e72_seed${SEED}_snmr_${ARM}" \
    --logger.base-dir "$E72_ROOT/student_holosoma_logs" \
    --randomization.ignore-unsupported True \
    --simulator.config.sim.max-episode-length-s 100000.0 \
    --command.setup-terms.motion-command.params.motion-config.motion-file "" \
    --command.setup-terms.motion-command.params.motion-config.motion-dir \
      "$E72_ROOT/motions/${ARM}"
```

Result file: `$OUT/a_prior_snmr_eval_ambiguity.json`, field `completion_rate` (compare against
§6). `E52_ROUNDS`/`E52_VAL_*`/`E52_TEACHER_*` are inert under `E52_EVAL_ONLY=1` (`rounds = 0`) and
are kept only for byte-parity with the frozen recipe.

## 9. Reproducing / re-verifying the CPU half

```bash
cd /home/robotixx/snmr
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
    tests/test_latent_substitution_motions.py -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.build_latent_substitution_motions \
    --out-root /data/robotixx/snmr-research/e72_latent_sub
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.verify_latent_substitution_motions \
    --out-root /data/robotixx/snmr-research/e72_latent_sub
```

The generator refuses to write inside `/data/robotixx/snmr-research/e70` (explicit guard), refuses
to proceed on a source SHA-256 mismatch, refuses a frame-rate disagreement between clips, and
refuses any rate other than 50 Hz unless `--expected-fps` is passed deliberately.
