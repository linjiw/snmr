import pathlib
import sys

import pytest

# Make the package importable without installation.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snmr import paths  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def g1_mjcf() -> str:
    """The G1 model with true hardware joint limits (holosoma copy — NOT GMR's mocap variant,
    whose narrowed hip-pitch cannot represent the training data; see design doc §0.1)."""
    try:
        return str(paths.g1_mjcf())
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def g1_mjcf_gmr() -> str:
    try:
        p = paths.gmr_root() / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not p.exists():
        pytest.skip(f"GMR G1 MJCF not found at {p}")
    return str(p)


@pytest.fixture(scope="session")
def g1_train_npz() -> str:
    """Holosoma WBT sample motion; vendored fixture preferred (works on a bare clone)."""
    fixture = FIXTURES / "sub3_largebox_003_mj.npz"
    if fixture.exists():
        return str(fixture)
    try:
        p = paths.holosoma_sample_npz()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not p.exists():
        pytest.skip(f"G1 training NPZ not found at {p}")
    return str(p)


@pytest.fixture(scope="session")
def g1_pair_npz() -> str:
    """One (LAFAN1 human, GMR-teacher) pair; vendored fixture preferred."""
    fixture = FIXTURES / "pair_g1_aiming1_subject1.npz"
    if fixture.exists():
        return str(fixture)
    try:
        candidates = sorted(paths.pairs_dir("unitree_g1").glob("*.npz"))
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not candidates:
        pytest.skip("no pair NPZs found (run scripts/make_pairs_lafan1.py)")
    return str(candidates[0])
