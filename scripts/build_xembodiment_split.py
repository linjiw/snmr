"""Emit the frozen SNMR cross-embodiment split manifest (design deliverable, not a run)."""
import hashlib, json, pathlib, re, sys

PAIRS = pathlib.Path("/data/robotixx/pairs")
ROBOTS = ["unitree_g1", "booster_t1_29dof", "engineai_pm01", "fourier_n1", "stanford_toddy"]

# Coarse action families. Near-neighbour LAFAN1 sequence labels are merged so that
# holding out a family cannot leave a near-duplicate category in training.
FAMILY = {
    "walk": "locomotion", "run": "locomotion", "sprint": "locomotion",
    "obstacles": "obstacle",
    "dance": "dance",
    "fallAndGetUp": "fall_recover", "ground": "fall_recover",
    "push": "perturb", "pushAndFall": "perturb", "pushAndStumble": "perturb",
    "fight": "combat", "fightAndSports": "combat",
    "aiming": "aiming",
    "jumps": "jump",
    "multipleActions": "QUARANTINE",   # mixes categories; assignable to no family
}
TRAIN_FAMILIES = ["locomotion", "obstacle", "dance", "fall_recover"]
HELDOUT_FAMILIES = ["aiming", "jump", "combat", "perturb"]
HELDOUT_SUBJECT = "subject5"

clip_re = re.compile(r"([A-Za-z]+)(\d*)_(subject[1-5])$")

stems = sorted(p.stem for p in (PAIRS / "unitree_g1").glob("*.npz"))
for r in ROBOTS:
    other = sorted(p.stem for p in (PAIRS / r).glob("*.npz"))
    assert other == stems, f"{r} clip set differs from unitree_g1"

train, panel_F, panel_U, quarantine = [], [], [], []
for s in stems:
    m = clip_re.fullmatch(s)
    assert m, s
    seq, _, subj = m.groups()
    fam = FAMILY[seq]
    if fam == "QUARANTINE":
        quarantine.append(s)
    elif fam in HELDOUT_FAMILIES:
        panel_U.append(s)
    elif subj == HELDOUT_SUBJECT:
        panel_F.append(s)
    else:
        train.append(s)

assert len(train) + len(panel_F) + len(panel_U) + len(quarantine) == 77

# Leakage assertions that the current repo split (subject5 only) fails.
def fams(c):
    return {FAMILY[clip_re.fullmatch(x).group(1)] for x in c}
def subs(c):
    return {clip_re.fullmatch(x).group(3) for x in c}
assert fams(train).isdisjoint(fams(panel_U)), "family leakage: train vs panel U"
assert set(train).isdisjoint(panel_F) and set(train).isdisjoint(panel_U)
assert HELDOUT_SUBJECT not in subs(train), "subject leakage: subject5 in train"

manifest = {
    "schema_version": "snmr.xembodiment-split.v1",
    "frozen_utc": "2026-09-10",
    "source_corpus": str(PAIRS),
    "robots": ROBOTS,
    "family_map": FAMILY,
    "train_families": TRAIN_FAMILIES,
    "heldout_families": HELDOUT_FAMILIES,
    "heldout_subject": HELDOUT_SUBJECT,
    "motion_partition": {
        "train": {"n": len(train), "clips": train},
        "panel_F_familiar_family_unseen_clip": {"n": len(panel_F), "clips": panel_F},
        "panel_U_unseen_family": {
            "n": len(panel_U), "clips": panel_U,
            "subject_unseen_stratum": sorted(c for c in panel_U if HELDOUT_SUBJECT in c),
            "subject_seen_stratum": sorted(c for c in panel_U if HELDOUT_SUBJECT not in c),
        },
        "quarantine_unassignable_family": {"n": len(quarantine), "clips": quarantine},
    },
    "leakage_checks_passed": [
        "train and panel_U share no action family",
        "train contains no subject5 clip",
        "train, panel_F, panel_U are pairwise clip-disjoint",
        "multipleActions quarantined from both train and evaluation",
    ],
}
payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
out = pathlib.Path("/home/robotixx/snmr/reproducibility/reports/xembodiment_split_manifest_2026-09-10.json")
out.write_text(json.dumps(manifest, indent=2) + "\n")
print("train", len(train), "| panel_F", len(panel_F), "| panel_U", len(panel_U),
      "(s5 stratum", len(manifest['motion_partition']['panel_U_unseen_family']['subject_unseen_stratum']),
      ") | quarantine", len(quarantine))
print("panel_F:", panel_F)
print("panel_U:", panel_U)
print("sha256:", manifest["manifest_sha256"])
print("wrote", out)
