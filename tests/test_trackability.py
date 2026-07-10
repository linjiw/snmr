"""Trackability-proxy correctness: the open-loop PD replay must be deterministic, physically
sane, and — crucially — VALID: corrupting the reference must shorten survival. If noise did not
hurt survival the proxy would not be measuring data trackability at all."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from trackability_proxy import (  # noqa: E402
    HOLOSOMA_G1_DAMPING,
    HOLOSOMA_G1_EFFORT,
    HOLOSOMA_G1_STIFFNESS,
    _match_gain,
    replay,
    replay_clip,
)

WBT_DIR = ROOT / "runs" / "wbt_validation"


@pytest.fixture(scope="session")
def wbt_clip() -> str:
    """One real exported WBT NPZ (GMR-teacher walk clip); skip on a bare clone."""
    p = WBT_DIR / "gmr" / "walk1_subject5_mj.npz"
    if not p.exists():
        pytest.skip(f"wbt_validation NPZs absent (run scripts/prepare_wbt_validation.py): {p}")
    return str(p)


def test_gain_tables_cover_all_hinges_unambiguously(g1_mjcf):
    import mujoco

    m = mujoco.MjModel.from_xml_path(g1_mjcf)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(names) == 29
    for n in names:
        for table in (HOLOSOMA_G1_STIFFNESS, HOLOSOMA_G1_DAMPING, HOLOSOMA_G1_EFFORT):
            assert _match_gain(n, table) > 0  # raises if 0 or 2+ substring hits


def test_replay_is_deterministic_and_well_formed(g1_mjcf, wbt_clip):
    r1 = replay(wbt_clip, g1_mjcf, seconds_max=3.0)
    r2 = replay(wbt_clip, g1_mjcf, seconds_max=3.0)
    assert r1 == r2  # bit-identical: no randomness anywhere
    assert 0.0 <= r1["survival_time_s"] <= r1["seconds_evaluated"] <= 3.0
    if r1["diverged"]:
        assert r1["diverge_reason"]
    # while-alive errors exist and are sane whenever it survived at least one tick
    if r1["survival_time_s"] > 0:
        assert 0.0 <= r1["mean_dof_err_rad"] < 3.0
        assert 0.0 <= r1["mean_root_height_err_m"] < 1.0


def test_corrupted_reference_survives_less(g1_mjcf, wbt_clip, tmp_path):
    """VALIDITY: 0.3 rad iid noise on the reference dofs every frame must topple the PD replay
    sooner than the clean clip. Margin calibrated by measurement on this exact setup
    (2026-07-10, mujoco 3.x CPU): clean mean survival 0.98 s vs corrupted 0.67 s over the same
    3 deterministic windows — a 0.31 s gap; we assert > 0.1 s so the test has 3x slack without
    being satisfiable by noise-level jitter."""
    src = np.load(wbt_clip, allow_pickle=True)
    corrupt = {k: src[k] for k in src.files}
    corrupt["joint_pos"] = corrupt["joint_pos"].copy()
    rng = np.random.default_rng(0)  # fixed seed: the test itself stays deterministic
    corrupt["joint_pos"][:, 7:] += rng.normal(0.0, 0.3, corrupt["joint_pos"][:, 7:].shape)
    corrupt_path = tmp_path / "corrupted.npz"
    np.savez_compressed(corrupt_path, **corrupt)

    clean = replay_clip(wbt_clip, g1_mjcf, seconds_max=10.0, control_hz=50.0, num_starts=3)
    noisy = replay_clip(str(corrupt_path), g1_mjcf, seconds_max=10.0, control_hz=50.0,
                        num_starts=3)

    assert noisy["survival_time_s"] < clean["survival_time_s"] - 0.1, (
        f"corrupted reference should survive less: clean {clean['survival_time_s']:.2f}s "
        f"vs corrupted {noisy['survival_time_s']:.2f}s"
    )
    # and while alive it tracks worse (reference is 0.3 rad away on average)
    assert noisy["mean_dof_err_rad"] > clean["mean_dof_err_rad"]


def test_root_gets_no_applied_force(g1_mjcf, wbt_clip):
    """The proxy must not cheat: with gravity + contacts OFF and zero initial velocity, a replay
    whose reference equals the frozen initial pose leaves the whole state exactly where it
    started — any applied force leaking into the free-joint dofs would move the root. (Contacts
    must be disabled: the holosoma G1 MJCF has permanent knee-ankle self-penetration whose
    internal forces would nudge the root ~1 cm and mask a leak.)"""
    import mujoco

    src = np.load(wbt_clip, allow_pickle=True)
    frozen = {k: src[k] for k in src.files}
    T = 60
    jp = np.repeat(frozen["joint_pos"][:1], T, axis=0)
    jp[:, 2] += 0.5  # hover well above the divergence floor / ground contact
    frozen["joint_pos"] = jp
    frozen["joint_vel"] = np.zeros((T, src["joint_vel"].shape[1]))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "frozen.npz"
        np.savez_compressed(p, **frozen)
        m = mujoco.MjModel.from_xml_path(g1_mjcf)
        xml = pathlib.Path(td) / "nogravity.xml"
        # same model, gravity off; absolutize meshdir so the copy resolves the original assets
        text = pathlib.Path(g1_mjcf).read_text()
        assert "<option" not in text  # holosoma G1 XML has no option block to clash with
        assets = str(pathlib.Path(g1_mjcf).parent / "assets")
        text = text.replace('meshdir="assets/"', f'meshdir="{assets}/"')
        xml.write_text(text.replace(
            '<mujoco model="g1_29dof">',
            '<mujoco model="g1_29dof">'
            '<option gravity="0 0 0"><flag contact="disable"/></option>'))
        r = replay(str(p), str(xml), seconds_max=1.0)

    assert not r["diverged"], r["diverge_reason"]
    assert r["mean_root_height_err_m"] < 1e-6
    assert r["mean_dof_err_rad"] < 1e-6
