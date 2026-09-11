#!/usr/bin/env python
"""Build the G1 locomotion DEVELOPMENT set manifest from the oracle label-source audits.

Rule (frozen here, applied mechanically; every clip is listed with its reasons):

A clip is a *valid oracle-scored locomotion development clip* iff, for BOTH feet,
  (a) the per-foot clip minimum lies within ``enter_height`` (0.03 m) of the stance plateau
      (``plateau_minus_min_m <= 0.030``), so the oracle's ground reference is a stance plateau
      and not a single tilted/penetrating frame;
  (b) the oracle stance prevalence lies in a walking band [0.30, 0.80];
  (c) the oracle finds at least 50 stance runs (gait-like alternation over the clip).

Everything inspected here is DEVELOPMENT data.  The manifest also records which clips the E70
two-walk tracker was trained on (tracker-seen), because tracker-seen and retargeter-seen are
different splits and the pilot manifest must keep them apart.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snmr.experiment import git_state, sha256_file, utc_now  # noqa: E402

RULE = {
    "plateau_minus_min_max_m": 0.030,
    "prevalence_band": [0.30, 0.80],
    "min_stance_runs": 50,
    "applies_to": "both feet",
}
TRACKER_SEEN_E70 = ("walk1_subject1", "walk1_subject5")
TRACKER_SEEN_E67_SPECIALIST_FAILED = ("walk3_subject1",)


def judge(clip: dict) -> tuple[bool, list[str]]:
    reasons = []
    for foot in clip["per_foot"]:
        name = foot["body"].split("_")[0]
        if foot["plateau_minus_min_m"] > RULE["plateau_minus_min_max_m"]:
            reasons.append(
                f"{name}: oracle ground reference unsound (clip minimum {foot['plateau_minus_min_m']:.3f} m below the stance plateau, > {RULE['plateau_minus_min_max_m']:.3f}; tilt at that frame {foot['tilt_deg_at_min_frame']:.1f} deg)"
            )
        lo, hi = RULE["prevalence_band"]
        if not (lo <= foot["oracle_prevalence"] <= hi):
            reasons.append(f"{name}: oracle stance prevalence {foot['oracle_prevalence']:.3f} outside [{lo}, {hi}]")
        if foot["oracle_runs"] < RULE["min_stance_runs"]:
            reasons.append(f"{name}: only {foot['oracle_runs']} oracle stance runs (< {RULE['min_stance_runs']})")
    return (not reasons), reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs="+", default=[
        str(ROOT / "runs/gate1b/oracle_label_audit_walkfamily_2026-09-10.json"),
        str(ROOT / "runs/gate1b/oracle_label_audit_2026-09-10.json"),
    ])
    ap.add_argument("--out", default=str(ROOT / "reproducibility/reports/locomotion_dev_set_2026-09-10.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(outp)

    clips: dict[str, dict] = {}
    for path in args.audits:
        data = json.load(open(path))
        for clip in data["clips"]:
            clips.setdefault(clip["clip"], {**clip, "audit_source": {"path": path, "sha256": sha256_file(path)}})

    families = {"walk": "walk", "run": "run", "sprint": "run"}
    rows = []
    for name in sorted(clips):
        clip = clips[name]
        family = families.get(name.rstrip("0123456789_subject").rstrip("0123456789"), "other")
        for key, fam in families.items():
            if name.startswith(key):
                family = fam
        valid, reasons = judge(clip)
        rows.append({
            "clip": name,
            "family": family,
            "subject": name.split("_")[-1],
            "role": "development",
            "oracle_valid": valid,
            "rejection_reasons": reasons,
            "oracle_prevalence_per_foot": [f["oracle_prevalence"] for f in clip["per_foot"]],
            "plateau_minus_min_m_per_foot": [f["plateau_minus_min_m"] for f in clip["per_foot"]],
            "oracle_runs_per_foot": [f["oracle_runs"] for f in clip["per_foot"]],
            "source_contact_prevalence_per_foot": [f["human_toe"]["source_contact_prevalence"] for f in clip["per_foot"]],
            "tracker_seen": (
                "E70 two-walk explicit student (seed0-2) training clip" if name in TRACKER_SEEN_E70
                else "E67 specialist trained, failed its gate" if name in TRACKER_SEEN_E67_SPECIALIST_FAILED
                else None
            ),
            "frames": clip["frames"], "fps": clip["fps"], "input_sha256": clip["input"]["sha256"],
        })
    valid_names = [r["clip"] for r in rows if r["oracle_valid"] and r["family"] == "walk"]
    out = {
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "generator": {"path": __file__, "sha256": sha256_file(__file__)},
        "status": "DEVELOPMENT SET. Every clip here has been inspected; none may be reported as held-out evaluation.",
        "rule": RULE,
        "oracle_definition": "teacher-height hysteresis on the ankle body origin, per-foot clip-minimum ground, enter 0.03 / exit 0.05 m (a stance proxy, not force-verified)",
        "valid_walk_clips": valid_names,
        "n_valid_walk": len(valid_names),
        "n_walk_family_inspected": sum(r["family"] == "walk" for r in rows),
        "n_run_family_inspected": sum(r["family"] == "run" for r in rows),
        "clips": rows,
        "caveats": [
            "Validity is validity of the ORACLE's ground reference on the clip, not evidence that the deployable mask or any corrector works there.",
            "run/sprint clips fail the walking prevalence band by construction (flight phases); a run-family development set needs its own band and is not declared here.",
            "walk1_subject1 and walk1_subject5 are tracker-seen for the E70 verifier; walk1_subject2 and walk2_subject4 are tracker-unseen but retargeter-seen (train partition of the frozen split).",
        ],
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{'VALID  ' if r['oracle_valid'] else 'reject '} {r['clip']:<16} {r['family']:<5} prev {[round(p,3) for p in r['oracle_prevalence_per_foot']]} plateau-min {[round(p,3) for p in r['plateau_minus_min_m_per_foot']]} runs {r['oracle_runs_per_foot']} tracker_seen={bool(r['tracker_seen'])}")
        for reason in r["rejection_reasons"]:
            print("      -", reason)
    print("valid walk dev clips:", valid_names)
    print("wrote", outp)


if __name__ == "__main__":
    main()
