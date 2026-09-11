"""Freeze the disclosed manual annotation for all five robots: semantic manifests + 19-role joint map."""
import hashlib, json, pathlib
from snmr.robot_spec import RobotSpec, SemanticManifest
from snmr.robot_tokens import RobotGraphTokenizer

A = "/data/robotixx/snmr-externals/GMR/assets"
SEM = {
 "unitree_g1": (f"{A}/unitree_g1/g1_mocap_29dof.xml", dict(root_link="pelvis", torso_link="torso_link", head_link="head_link", left_hand_link="left_rubber_hand", right_hand_link="right_rubber_hand", left_foot_link="left_ankle_roll_link", right_foot_link="right_ankle_roll_link")),
 "booster_t1_29dof": (f"{A}/booster_t1_29dof/t1_mocap.xml", dict(root_link="Trunk", torso_link="Waist", head_link="H2", left_hand_link="left_hand_link", right_hand_link="right_hand_link", left_foot_link="left_foot_link", right_foot_link="right_foot_link")),
 "engineai_pm01": (f"{A}/engineai_pm01/pm_v2.xml", dict(root_link="LINK_BASE", torso_link="LINK_TORSO_YAW", head_link="LINK_HEAD_YAW", left_hand_link="LINK_ELBOW_END_L", right_hand_link="LINK_ELBOW_END_R", left_foot_link="LINK_ANKLE_ROLL_L", right_foot_link="LINK_ANKLE_ROLL_R")),
 "fourier_n1": (f"{A}/fourier_n1/n1_mocap.xml", dict(root_link="base_link", torso_link="torso_link", head_link="camera_link", left_hand_link="left_end_effector_link", right_hand_link="right_end_effector_link", left_foot_link="left_foot_pitch_link", right_foot_link="right_foot_pitch_link")),
 "stanford_toddy": (f"{A}/stanford_toddy/toddy_mocap.xml", dict(root_link="torso", torso_link="waist_link", head_link="head", left_hand_link="hand", right_hand_link="hand_2", left_foot_link="ank_roll_link", right_foot_link="ank_roll_link_2")),
}
specs, rec = [], {}
for k, (p, sm) in SEM.items():
    s = RobotSpec.from_mjcf(p, SemanticManifest(**sm))
    specs.append(s)
    rec[k] = {"mjcf": p, "mjcf_sha256": hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(),
              "semantic_manifest": sm, "n_links": len(s.links), "n_joints": len(s.joints)}
b = RobotGraphTokenizer("kinematic")(specs)
doc = {
  "schema_version": "snmr.xembodiment-annotation.v1",
  "frozen_utc": "2026-09-10",
  "disclosure": ("These semantic manifests and the 19-role joint correspondence are MANUAL "
                 "annotation. They are inputs to the human-side ruler and to the strong "
                 "nearest-embodiment transfer baseline. They are NOT model parameters and "
                 "carry no motion-derived or outcome-derived information."),
  "robots": rec,
  "batched_token_shape": list(b.node_features.shape),
  "joints_per_robot": b.joint_mask.sum(1).tolist(),
  "common_joint_roles": ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ank_pitch","L_ank_roll",
      "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ank_pitch","R_ank_roll","waist_yaw",
      "L_sho_pitch","L_sho_roll","L_elbow","R_sho_pitch","R_sho_roll","R_elbow"],
  "role_map_source": "reproducibility/reports/xembodiment_morphology_distance_2026-09-10.json",
  "uncorresponded_joints_note": ("Joints outside the 19 common roles (G1 waist roll/pitch + 6 wrist, "
      "T1 4 wrist/hand-roll + 2 elbow-yaw, PM01 2 elbow-yaw + head yaw, N1 2 wrist-yaw, Toddy waist roll) "
      "are held at their nominal pose by the nearest-transfer baseline and are reported as an "
      "uncovered-DoF fraction per robot."),
}
payload = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
doc["annotation_sha256"] = hashlib.sha256(payload).hexdigest()
out = pathlib.Path("/home/robotixx/snmr/reproducibility/reports/xembodiment_robot_annotation_2026-09-10.json")
out.write_text(json.dumps(doc, indent=2) + "\n")
for k, v in rec.items():
    print(f"{k:20s} links={v['n_links']:3d} joints={v['n_joints']:3d}")
print("uncovered DoF (outside 19 common roles):",
      {k: v["n_joints"] - 19 for k, v in rec.items()})
print("sha256:", doc["annotation_sha256"], "\nwrote", out)
