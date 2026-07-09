import pathlib
import sys

import pytest

# Make the package importable without installation.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Path to the real Unitree G1 MJCF.
#
# We use the holosoma_retargeting copy rather than GMR's ``g1_mocap_29dof.xml`` because it carries the
# true hardware joint ranges (e.g. hip-pitch [-2.53, 2.88]) that match the training NPZ this repo
# ships. GMR's mocap model narrows hip-pitch to [-1.57, 1.57], which is < the range spanned by ~38% of
# the NPZ frames — using it would make the tanh joint-limit head structurally unable to reproduce the
# data (an adversarial review caught this). The GMR ``g1_mocap_29dof_with_hands.xml`` also has the
# correct ranges; we keep GMR paths available for the multi-robot FK test.
REPO = ROOT.parent
G1_MJCF = (
    REPO
    / "holosoma"
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
    / "g1"
    / "g1_29dof.xml"
)
G1_MJCF_GMR = REPO / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
G1_TRAIN_NPZ = (
    REPO
    / "holosoma"
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "sub3_largebox_003_mj.npz"
)


@pytest.fixture(scope="session")
def g1_mjcf() -> str:
    if not G1_MJCF.exists():
        pytest.skip(f"G1 MJCF not found at {G1_MJCF}")
    return str(G1_MJCF)


@pytest.fixture(scope="session")
def g1_mjcf_gmr() -> str:
    if not G1_MJCF_GMR.exists():
        pytest.skip(f"GMR G1 MJCF not found at {G1_MJCF_GMR}")
    return str(G1_MJCF_GMR)


@pytest.fixture(scope="session")
def g1_train_npz() -> str:
    if not G1_TRAIN_NPZ.exists():
        pytest.skip(f"G1 training NPZ not found at {G1_TRAIN_NPZ}")
    return str(G1_TRAIN_NPZ)
