# Dataset: LAFAN1 → 5-robot teacher pairs

The training dataset (`data/pairs/<robot>/<clip>.npz`) is **regenerable, LAFAN1-derived, and never
committed or redistributed** (Ubisoft LAFAN1 license). This file records exactly how it was made
so the paper numbers are reproducible.

## Provenance

- Source motion: LAFAN1 `lafan1.zip` (77 BVH clips) from
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset (extract to `data/lafan1_bvh/`).
- Teacher: GMR @ `bb1bbe40774794fceb2a7c579a3464a28e68c844` (IK retargeting, default configs).
- Generated 2026-07-09 on 8-core EC2: 77 clips × 5 robots = **2,483,360 teacher frames
  (~23 h of paired motion, 1.6 GB)** in 4.3 h at ~159 fps.

## Regeneration

```bash
bash scripts/fetch_externals.sh          # pinned GMR + holosoma clones
# download + extract lafan1.zip to ../data/lafan1_bvh, then:
python scripts/make_pairs_lafan1.py \
    --robots unitree_g1 booster_t1_29dof fourier_n1 engineai_pm01 stanford_toddy
```

Each NPZ holds: `human_pos (T,24,3)`, `human_quat (T,24,4) wxyz`, `human_names`,
`qpos (T,7+D)` `[root_pos, root_quat wxyz, dof]`, `fps=30`, `robot`, `human_height`.

## Split

Held-out clips (val, fixed across all experiments — defined in `scripts/train_phase1.py::VAL_CLIPS`):
`walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2,
sprint1_subject4, aiming2_subject3`. The remaining 70 clips are train.

## Spot checksums (reproducibility claim)

SHA256 of three representative files from the 2026-07-09 generation — regenerated data will differ
byte-wise only if GMR/config versions differ (IK is deterministic given inputs):

```
79565402a381d54122c28b9f1f88bc43a46dfdca9f5aeee0ff0afbaee6c8c845  pairs/unitree_g1/walk1_subject1.npz
af6bf59c573258350055b723546109b54d8102bf16a09e23a0dcc73e3bcaa695  pairs/stanford_toddy/dance1_subject1.npz
d9109c658e2eadb7722e8a97031c1ffb65530e0f632f6430f1c9e39160d6a625  pairs/booster_t1_29dof/sprint1_subject4.npz
```

See `git log` tag `data-v1` for the generation-time state of all scripts.
