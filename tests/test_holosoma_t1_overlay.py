"""CPU tests for the E54 T1 WBT overlay (snmr/integration/holosoma_t1_wbt.py)."""

import re

import pytest

from snmr.integration.holosoma_t1_wbt import (
    G1_TO_T1_TRACKED,
    T1_ALLOWED_CONTACTS,
    t1_body_names,
    undesired_contacts_regex,
)


def test_mapping_covers_the_fourteen_g1_tracked_bodies_and_is_injective():
    assert len(G1_TO_T1_TRACKED) == 14
    assert len(set(G1_TO_T1_TRACKED.values())) == 14
    g1 = ["pelvis", "left_hip_roll_link", "torso_link", "right_wrist_yaw_link"]
    assert t1_body_names(g1) == ["Waist", "Hip_Roll_Left", "Trunk", "right_hand_link"]
    with pytest.raises(KeyError):
        t1_body_names(["not_a_body"])


def test_undesired_contact_regex_allows_only_feet_and_hands():
    rx = re.compile(undesired_contacts_regex(T1_ALLOWED_CONTACTS))
    for allowed in T1_ALLOWED_CONTACTS:
        assert rx.match(allowed) is None
    for body in ("Trunk", "Waist", "AL2", "Shank_Left", "H1"):
        assert rx.match(body) is not None


def test_overlay_builds_against_holosoma_when_available():
    pytest.importorskip("holosoma")
    from snmr.integration.holosoma_t1_wbt import build_t1_29dof_wbt

    cfg = build_t1_29dof_wbt()
    mc = cfg.command.setup_terms["motion_command"].params["motion_config"]
    assert mc.body_name_ref == ["Trunk"] and len(mc.body_names_to_track) == 14
    assert cfg.robot.init_state.pos[2] == pytest.approx(0.68)
