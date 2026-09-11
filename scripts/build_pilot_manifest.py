#!/usr/bin/env python
"""Freeze the single executable G1 locomotion pilot manifest (2026-09-10).

One development study, not the multi-robot program.  It names every split field separately,
pins the verifier and its qualification evidence by hash, separates search from confirmation
starts/seeds, defines headroom per clip BEFORE any candidate is scored, and states the exact
closed-loop command, its measured cost basis, and the approval it needs.  It launches nothing.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snmr.experiment import git_state, utc_now  # noqa: E402

E70 = pathlib.Path("/data/robotixx/snmr-research/e70")


def sha(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = ROOT / "reproducibility/reports/pilot_manifest_g1_locomotion_2026-09-10.json"
    if out.exists() and "--overwrite" not in sys.argv:
        raise FileExistsError(out)
    student = E70 / "students/seed0_explicit/c_prior_explicit_student.pt"
    eval_json = E70 / "students/seed0_explicit/c_prior_explicit_eval.json"
    ev = json.load(open(eval_json))
    seeds = {}
    for s in (0, 1, 2):
        d = json.load(open(E70 / f"students/seed{s}_explicit/c_prior_explicit_eval.json"))
        seeds[f"seed{s}"] = {
            "completion": d["completion_rate"], "survival_s": d["mean_survival_s"],
            "per_clip": {k: {"completion": v["completion_rate"], "survival_s": v["mean_survival_s"], "rollouts": v["num_rollouts"]} for k, v in d["per_clip"].items()},
            "checkpoint_sha256": sha(E70 / f"students/seed{s}_explicit/c_prior_explicit_student.pt"),
        }
    dev = json.load(open(ROOT / "reproducibility/reports/locomotion_dev_set_2026-09-10.json"))
    manifest = {
        "created_at": utc_now(),
        "git": git_state(ROOT),
        "generator": {"path": __file__, "sha256": sha(pathlib.Path(__file__))},
        "study": "G1 locomotion DEVELOPMENT pilot: do valid, source-faithful bounded corrections improve closed-loop execution beyond no-op and simple repair?",
        "not_this_study": [
            "no held-out robot claim (G1 is the development robot; every learned component here has seen G1)",
            "no learnability or transfer claim (no correction head is trained)",
            "no downstream tracker-learning claim (no tracker is retrained on corrected references)",
        ],
        "split_fields": {
            "development_robot": "unitree_g1",
            "proposal_training_robots": ["unitree_g1 (GMR teacher references; the proposal here IS the configured GMR-G1 reference, no learned proposal is used)"],
            "correction_label_robots": ["unitree_g1"],
            "downstream_tracker_training_clips": ["walk1_subject1", "walk1_subject5"],
            "verifier_qualification_clips": ["walk1_subject1", "walk1_subject5"],
            "candidate_search_clips_development": dev["valid_walk_clips"],
            "final_evaluation_clips": "NONE declared. Every clip in this manifest is development. A later held-out evaluation must draw clips not listed here and not used to calibrate the intent filter.",
            "tracker_seen_vs_retargeter_seen": {
                "walk1_subject1": "tracker-seen, retargeter-seen (train partition)",
                "walk1_subject5": "tracker-seen, retargeter-unseen (subject5 held out in the frozen split)",
                "walk1_subject2": "tracker-UNSEEN, retargeter-seen",
                "walk2_subject4": "tracker-UNSEEN, retargeter-seen",
            },
        },
        "verifier": {
            "name": "E70 seed-0 explicit-command DAgger student (arm c_prior_explicit), two-walk policy",
            "checkpoint": {"path": str(student), "sha256": sha(student)},
            "evaluator": "scripts/train_e52_dagger.py with E52_EVAL_ONLY=1 E52_DET=1 (multi-motion general grid: equal-count per-clip linspace starts, per-motion teacher routing, per-clip report)",
            "teacher_manifest": {"path": str(E70 / "teacher_manifest.json"), "sha256": sha(E70 / "teacher_manifest.json")},
            "reference_motions": {
                "walk1_subject1": {"path": "/data/robotixx/snmr-research/e69/motions/walk1_subject1_mj_z.npz", "sha256": sha(pathlib.Path("/data/robotixx/snmr-research/e69/motions/walk1_subject1_mj_z.npz"))},
                "walk1_subject5": {"path": "/data/robotixx/snmr-research/e67/motions/walk1_subject5_mj_z.npz", "sha256": sha(pathlib.Path("/data/robotixx/snmr-research/e67/motions/walk1_subject5_mj_z.npz"))},
            },
            "qualification_evidence": {
                "protocol": "1,024 rollouts (512 per clip), 500-step (10 s at 50 Hz) horizon, seed 404, exact-state resets on a deterministic per-clip linspace grid",
                "floor_applied": "E67 precedent: >= 0.80 completion and >= 9.0 s survival on the uncorrected reference, per clip",
                "seed_results": seeds,
                "verdict": "qualified for own-clip execution verification on walk1_subject1 and walk1_subject5 only; NOT qualified on any unseen motion (never evaluated there); walk1_subject5 completion 0.86-0.87 passes the floor with ~0.13 headroom; walk1_subject1 at 0.98 has ~0.02 headroom",
                "evaluation_wall_time_observed": "general_eval.log timestamps span 2026-08-09 02:02:39 to 02:02:49 (about 10 s per 1,024-rollout evaluation on the shared RTX 5090, warp kernels cached); not a benchmark",
            },
            "disclosed_confound": "the verifier was trained on the uncorrected GMR references of both clips, so it is biased toward the no-op candidate; the bias runs against the hypothesis",
        },
        "candidate_bank_small": [
            {"id": "noop", "definition": "the GMR-G1 reference unchanged"},
            {"id": "smooth_s1", "definition": "Gaussian low-pass of dof_pos, sigma=1 frame at 30 fps, root unchanged"},
            {"id": "smooth_s2", "definition": "Gaussian low-pass of dof_pos, sigma=2 frames"},
            {"id": "footlock_dls", "definition": "snmr.footlock.foot_lock_masked with the deployable source_contact mask, eval_footlock defaults (iters 12, lr 0.5, damping 1e-2, blend 2, merge_gap 6, extend 2)"},
            {"id": "c6_projection", "definition": "snmr.projection.windowed_contact_projection with the deployable source_contact mask, frozen Gate-1b config (bounds 0.04 m / 0.12 rad / 0.35 rad)"},
            {"id": "spline_bounded", "definition": "c6_projection's dof delta least-squares projected onto the C2 clamped cubic B-spline family (8-frame knots, 3 pinned coefficients per end) and box-clipped; a bounded-correction variant, not a searched one"},
        ],
        "scoring_order": [
            "Stage 0 (CPU): joint limits, penetration, intent filter (snmr.repair_intent, PROVISIONAL tolerances), kinematic guards (stance foot speed under source_contact and teacher-height masks, MPJPE-to-proposal, jerk ratio). Rejections are recorded with reasons; rejected candidates are never scored closed-loop.",
            "Stage 1 (GPU, approval required): closed-loop rollouts of every surviving candidate with the verifier, matched starts and budgets; open-loop PD survival is NOT a ranking objective (repo record: MORPHORETARGET_FOUNDATION_2026-08-30). If a cheap proxy is run at all, its ranking is compared to the closed-loop ranking and reported, never substituted.",
        ],
        "search_vs_confirmation": {
            "search_starts": "per clip, the even-indexed positions of a 256-point linspace over [first, last] (128 starts), evaluation seed 404",
            "confirmation_starts": "per clip, the odd-indexed positions of the same 256-point linspace (128 starts, disjoint from search), evaluation seed 405",
            "rule": "the candidate chosen on search starts is reported on confirmation starts only; search feedback never enters the confirmatory interval",
            "uncertainty": "per-clip: paired bootstrap over 8 contiguous within-clip start blocks of 16 starts (10,000 replicates), explicitly conditional on that clip; across-motion claims aggregate distinct clips and are not made from one clip",
            "never_accept_per_rollout": True,
        },
        "headroom_and_thresholds": {
            "baseline_completion_per_clip_seed0": {k: v["completion_rate"] for k, v in ev["per_clip"].items()},
            "headroom_per_clip_seed0": {k: 1.0 - v["completion_rate"] for k, v in ev["per_clip"].items()},
            "practical_threshold": "improvement >= 0.25 x headroom on confirmation starts AND paired-bootstrap 95% lower bound > 0 AND survival not lower by more than 0.05 s; a fixed +0.10 is not used because it is unattainable on walk1_subject1 (headroom 0.018)",
            "eligibility": "clips with baseline completion >= 0.95 are reported as no-headroom, kept in the denominator, not scored for gain",
        },
        "denominator_rule": "every candidate x clip x window is in the ledger with one of: accepted / rejected(reason) / abstained(baseline invalid) / not_scored(no headroom) / search_not_found; search-not-found is never reported as uncorrectable",
        "closed_loop_execution": {
            "status": "NOT RUN in this sprint",
            "approval_required": "yes: GPU is shared (other tenant holds ~22.6 of 32.6 GB); the evaluation must be launched behind scripts/resume_when_gpu_exclusive.sh or with the owner's explicit go-ahead",
            "dependencies": [
                "corrected references must be exported to the holosoma motion NPZ schema at 50 fps: scripts/export_wbt_gmr_batch.py path (mujoco_replay + resample_qpos) applied to the corrected 30-fps qpos, then scripts/attach_latent_to_wbt.py to carry latent_z (unused by the explicit arm but required by the loader)",
                "one motion-dir per candidate holding exactly the two clips in lexical order so the E70 teacher manifest routes correctly",
                "search/confirmation start grids need an evaluator flag for even/odd linspace subsets and a second evaluation seed; both are ~20-line changes to scripts/train_e52_dagger.py's eval path and must be committed and hashed before the run",
            ],
            "exact_command_template": (
                "cd $SNMR_HOLOSOMA_ROOT && env E52_ARM=c_prior_explicit E52_TEACHER_MANIFEST=/data/robotixx/snmr-research/e70/teacher_manifest.json "
                "E52_OUT=<out_dir> E52_ROUNDS=2000 E52_DET=1 E52_PHASE_ONLY=0 E52_SHUFFLE_LATENT=0 E52_EVAL_ONLY=1 "
                "PYTHONPATH=$SNMR_ROOT $WBT_PYTHON $SNMR_ROOT/scripts/train_e52_dagger.py exp:g1-29dof-wbt simulator:mjwarp logger:disabled "
                "--training.num-envs 1024 --training.seed <404 search | 405 confirmation> --training.headless True "
                "--randomization.ignore-unsupported True --simulator.config.sim.max-episode-length-s 100000.0 "
                "--command.setup-terms.motion-command.params.motion-config.motion-file '' "
                "--command.setup-terms.motion-command.params.motion-config.motion-dir <candidate_motion_dir>"
            ),
            "sample_counts": "per candidate per evaluation: 1,024 rollouts x 500 steps = 512,000 env steps; 6 candidates x 2 (search, confirmation) = 12 evaluations = ~6.1 M env steps",
            "cost_basis": "about 10 s wall per evaluation observed for E70 (log span; includes env build with cached kernels); budget 12 x 2 min = under 30 GPU-minutes, exclusive access not required if ~9 GB free suffices for 1,024 mjwarp envs (E70 ran at 1,024 envs on this GPU)",
        },
    }
    out.write_text(json.dumps(manifest, indent=2))
    print("wrote", out)
    print("verifier sha", manifest["verifier"]["checkpoint"]["sha256"][:16], "| headroom", manifest["headroom_and_thresholds"]["headroom_per_clip_seed0"])


if __name__ == "__main__":
    main()
