"""Single source of truth for external asset locations.

SNMR reads robot models and fixtures from two unmodified sibling clones (GMR, holosoma) and a
regenerable data directory. All scripts and tests resolve those locations through this module:

  * ``SNMR_GMR_ROOT``      — path to a GMR clone       (default: ``<repo>/../GMR``)
  * ``SNMR_HOLOSOMA_ROOT`` — path to a holosoma clone  (default: ``<repo>/../holosoma``)
  * ``SNMR_DATA_ROOT``     — path to the data dir      (default: ``<repo>/../data``)

Pinned SHAs and licenses are recorded in ``THIRD_PARTY.md``; ``scripts/fetch_externals.sh``
creates the default layout.
"""

from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve(env_var: str, default: pathlib.Path, hint: str) -> pathlib.Path:
    path = pathlib.Path(os.environ.get(env_var, default))
    if not path.exists():
        raise FileNotFoundError(
            f"{hint} not found at {path}. Clone it with scripts/fetch_externals.sh "
            f"or set ${env_var}."
        )
    return path


def gmr_root() -> pathlib.Path:
    return _resolve("SNMR_GMR_ROOT", REPO_ROOT.parent / "GMR", "GMR clone")


def holosoma_root() -> pathlib.Path:
    return _resolve("SNMR_HOLOSOMA_ROOT", REPO_ROOT.parent / "holosoma", "holosoma clone")


def data_root() -> pathlib.Path:
    return _resolve("SNMR_DATA_ROOT", REPO_ROOT.parent / "data", "data directory")


# --------------------------------------------------------------------------------------
# canonical robot model paths (see THIRD_PARTY.md for licenses)
# --------------------------------------------------------------------------------------
def g1_mjcf() -> pathlib.Path:
    """The G1 model with true hardware joint limits (holosoma copy — NOT GMR's mocap variant,
    whose narrowed hip-pitch cannot represent the training data; see design doc §0.1)."""
    return (
        holosoma_root()
        / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    )


def robot_mjcf(robot: str) -> pathlib.Path:
    table = {
        "unitree_g1": g1_mjcf(),
        "booster_t1_29dof": gmr_root() / "assets/booster_t1_29dof/t1_mocap.xml",
        "fourier_n1": gmr_root() / "assets/fourier_n1/n1_mocap.xml",
        "engineai_pm01": gmr_root() / "assets/engineai_pm01/pm_v2.xml",
        "stanford_toddy": gmr_root() / "assets/stanford_toddy/toddy_mocap.xml",
        "unitree_h1": gmr_root() / "assets/unitree_h1/h1.xml",
        "unitree_h1_2": gmr_root() / "assets/unitree_h1_2/h1_2_handless.xml",
    }
    if robot not in table:
        raise KeyError(f"unknown robot '{robot}'; known: {sorted(table)}")
    return table[robot]


def holosoma_sample_npz() -> pathlib.Path:
    """The holosoma WBT sample motion; prefers the vendored test fixture when present."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "sub3_largebox_003_mj.npz"
    if fixture.exists():
        return fixture
    return (
        holosoma_root()
        / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
    )


def pairs_dir(robot: str) -> pathlib.Path:
    return data_root() / "pairs" / robot
