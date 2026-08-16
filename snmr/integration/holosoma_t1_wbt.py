"""E54 — Booster T1 29-DoF whole-body-tracking experiment preset as an snmr-side overlay.

The T1 WBT registration described in ``docs/E54_T1_PORT_STATUS.md`` was created inside the
holosoma clone at rev ``9fb2b57`` and was lost when the pinned clone moved to ``20699ff``
(the files were never committed anywhere).  The pinned clone is bound in the reproducibility
index and must stay clean, so the preset is rebuilt *here*, from holosoma's own G1 WBT presets
plus its T1 locomotion robot/action presets, by ``dataclasses.replace`` — the same body-name
mapping and hyper-parameters as the 2026-08-06 port:

* robot ``t1_29dof_waist_wrist``; ``action_scale=0.25``,
  ``action_scales_by_effort_limit_over_p_gain=True``; self-collisions on; init z = 0.68
  (T1 locomotion default; G1 uses 0.76);
* action ``t1_29dof_joint_pos``;
* command: G1 term-identical MotionCommand with T1 tracked bodies, reference body ``Trunk``;
* termination: ``BadTrackingZOnly`` with T1 body names; end-effectors = feet + hands;
* reward: G1 terms/weights/sigmas, ``UndesiredContacts`` allow-list swapped to T1 feet/hands
  (and their contact points);
* observation / curriculum / randomization: G1 presets verbatim (all name-agnostic; robot-only
  DR, object DR omitted).

Use :func:`annotated_experiment_config` in place of ``holosoma.train_agent.AnnotatedExperimentConfig``
to expose ``exp:t1-29dof-wbt`` as a tyro subcommand alongside the stock presets.
"""

from __future__ import annotations

from dataclasses import replace

# 14 tracked bodies, G1 -> T1 (docs/E54_T1_PORT_STATUS.md, "Body-name mapping")
G1_TO_T1_TRACKED = {
    "pelvis": "Waist",
    "left_hip_roll_link": "Hip_Roll_Left",
    "left_knee_link": "Shank_Left",
    "left_ankle_roll_link": "left_foot_link",
    "right_hip_roll_link": "Hip_Roll_Right",
    "right_knee_link": "Shank_Right",
    "right_ankle_roll_link": "right_foot_link",
    "torso_link": "Trunk",
    "left_shoulder_roll_link": "AL2",
    "left_elbow_link": "AL3",
    "left_wrist_yaw_link": "left_hand_link",
    "right_shoulder_roll_link": "AR2",
    "right_elbow_link": "AR3",
    "right_wrist_yaw_link": "right_hand_link",
}
T1_REF_BODY = "Trunk"
T1_END_EFFECTORS = ["left_foot_link", "right_foot_link", "left_hand_link", "right_hand_link"]
T1_ALLOWED_CONTACTS = T1_END_EFFECTORS + ["left_foot_contact_point", "right_foot_contact_point"]
T1_INIT_Z = 0.68


def undesired_contacts_regex(allowed: list[str]) -> str:
    """Match every body except the allowed ones (holosoma's negative-lookahead convention)."""
    return "^" + "".join(f"(?!{name}$)" for name in allowed) + ".+$"


def t1_body_names(g1_names: list[str]) -> list[str]:
    missing = [n for n in g1_names if n not in G1_TO_T1_TRACKED]
    if missing:
        raise KeyError(f"no T1 mapping for G1 bodies {missing}")
    return [G1_TO_T1_TRACKED[n] for n in g1_names]


def build_t1_29dof_wbt(motion_file: str | None = None):
    """Construct the ``t1_29dof_wbt`` ExperimentConfig from holosoma's presets (import-time safe)."""
    from holosoma.config_values import action, command, robot  # noqa: WPS433 (holosoma-only)
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt

    g1 = g1_29dof_wbt

    # --- command: T1 tracked bodies + reference body --------------------------------------
    g1_cmd = command.g1_29dof_wbt_command
    g1_motion_cfg = g1_cmd.setup_terms["motion_command"].params["motion_config"]
    t1_motion_cfg = replace(
        g1_motion_cfg,
        body_names_to_track=t1_body_names(list(g1_motion_cfg.body_names_to_track)),
        body_name_ref=[T1_REF_BODY],
        **({"motion_file": motion_file} if motion_file else {}),
    )
    t1_setup = dict(g1_cmd.setup_terms)
    t1_setup["motion_command"] = replace(
        g1_cmd.setup_terms["motion_command"],
        params={**g1_cmd.setup_terms["motion_command"].params, "motion_config": t1_motion_cfg},
    )
    t1_command = replace(g1_cmd, setup_terms=t1_setup)

    # --- termination: BadTrackingZOnly with T1 names ------------------------------------
    g1_term = g1.termination
    bad = g1_term.terms["bad_tracking"]
    t1_bad = replace(
        bad,
        params={
            **bad.params,
            "body_names_to_track": t1_body_names(list(bad.params["body_names_to_track"])),
            "bad_motion_body_pos_body_names": list(T1_END_EFFECTORS),
        },
    )
    t1_termination = replace(g1_term, terms={**g1_term.terms, "bad_tracking": t1_bad})

    # --- reward: swap the undesired-contact allow-list -----------------------------------
    g1_rew = g1.reward
    uc = g1_rew.terms["undesired_contacts"]
    t1_uc = replace(
        uc, params={**uc.params, "undesired_contacts_body_names": undesired_contacts_regex(T1_ALLOWED_CONTACTS)}
    )
    t1_reward = replace(g1_rew, terms={**g1_rew.terms, "undesired_contacts": t1_uc})

    # --- robot / action ------------------------------------------------------------------
    t1_robot_base = robot.t1_29dof_waist_wrist
    t1_robot = replace(
        t1_robot_base,
        control=replace(
            t1_robot_base.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(t1_robot_base.asset, enable_self_collisions=True),
        init_state=replace(t1_robot_base.init_state, pos=[0.0, 0.0, T1_INIT_Z]),
    )

    return replace(
        g1,
        training=replace(g1.training, name="t1_29dof_wbt_manager"),
        robot=t1_robot,
        action=action.t1_29dof_joint_pos,
        command=t1_command,
        termination=t1_termination,
        reward=t1_reward,
    )


def annotated_experiment_config():
    """``AnnotatedExperimentConfig`` with ``exp:t1-29dof-wbt`` added to holosoma's DEFAULTS."""
    import tyro
    from typing_extensions import Annotated

    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_values.experiment import DEFAULTS

    defaults = dict(DEFAULTS)
    if "t1_29dof_wbt" not in defaults:
        defaults["t1_29dof_wbt"] = build_t1_29dof_wbt()
    return Annotated[
        ExperimentConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"exp:{k.replace('_', '-')}": v for k, v in defaults.items()}
            )
        ),
    ]
