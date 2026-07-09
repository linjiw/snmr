# Third-party dependencies (external sibling clones, pinned)

SNMR does not vendor or modify these projects; it reads their asset files and (for data
generation) imports GMR as an installed package. `scripts/fetch_externals.sh` clones both at the
pinned SHAs into the expected sibling layout (override locations with `SNMR_GMR_ROOT` /
`SNMR_HOLOSOMA_ROOT`, see `snmr/paths.py`).

| Project | URL | Pinned SHA | License | How we use it |
|---|---|---|---|---|
| GMR (General Motion Retargeting) | https://github.com/YanjieZe/GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` | MIT | Teacher retargeter (`pip install -e`, unmodified); LAFAN1 BVH loader; robot MJCF assets |
| holosoma | https://github.com/amazon-far/holosoma | `38009aad61851d59277fa4ebaf4f54c44ec483f7` | Apache-2.0 (see its NOTICE) | G1 MJCF with hardware joint limits; WBT training-NPZ format reference + sample fixture |

## Robot models read at runtime (per-robot licenses in `GMR/assets/<robot>/`)

| Robot | File | License |
|---|---|---|
| Unitree G1 (29 dof) | `holosoma/.../models/g1/g1_29dof.xml` | BSD-3-Clause |
| Booster T1 (29 dof) | `GMR/assets/booster_t1_29dof/t1_mocap.xml` | Apache-2.0 |
| Fourier N1 | `GMR/assets/fourier_n1/n1_mocap.xml` | LGPL-3.0 (assets not vendored partly for this reason) |
| EngineAI PM01 | `GMR/assets/engineai_pm01/pm_v2.xml` | BSD-3-Clause |
| Stanford ToddlerBot | `GMR/assets/stanford_toddy/toddy_mocap.xml` | MIT |
| Unitree H1 / H1-2 (tests only) | `GMR/assets/unitree_h1*/...` | BSD-3-Clause |

## Motion data

LAFAN1 (Ubisoft La Forge Animation Dataset) is downloaded by the user from the official
repository and is subject to its own (non-commercial research) license terms. The generated
`data/pairs/` dataset is LAFAN1-derived and is **not** redistributed; regenerate it with
`scripts/make_pairs_lafan1.py` (see `docs/DATA.md`).
