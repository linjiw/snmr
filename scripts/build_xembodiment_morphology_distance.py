"""Robot geometry descriptors, common joint-role map, and scale-free morphology distance."""
import json, itertools, pathlib
import numpy as np, mujoco

A = "/data/robotixx/snmr-externals/GMR/assets"
XML = {
 "unitree_g1": f"{A}/unitree_g1/g1_mocap_29dof.xml",
 "booster_t1_29dof": f"{A}/booster_t1_29dof/t1_mocap.xml",
 "engineai_pm01": f"{A}/engineai_pm01/pm_v2.xml",
 "fourier_n1": f"{A}/fourier_n1/n1_mocap.xml",
 "stanford_toddy": f"{A}/stanford_toddy/toddy_mocap.xml",
}
# Manual semantic joint-role correspondence (the disclosed annotation, 19 common roles).
ROLE = {
"unitree_g1": dict(zip(
 ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ank_pitch","L_ank_roll",
  "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ank_pitch","R_ank_roll","waist_yaw",
  "L_sho_pitch","L_sho_roll","L_elbow","R_sho_pitch","R_sho_roll","R_elbow"],
 ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
  "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
  "right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
  "waist_yaw_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_elbow_joint",
  "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_elbow_joint"])),
"booster_t1_29dof": dict(zip(
 ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ank_pitch","L_ank_roll",
  "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ank_pitch","R_ank_roll","waist_yaw",
  "L_sho_pitch","L_sho_roll","L_elbow","R_sho_pitch","R_sho_roll","R_elbow"],
 ["Left_Hip_Pitch","Left_Hip_Roll","Left_Hip_Yaw","Left_Knee_Pitch","Left_Ankle_Pitch",
  "Left_Ankle_Roll","Right_Hip_Pitch","Right_Hip_Roll","Right_Hip_Yaw","Right_Knee_Pitch",
  "Right_Ankle_Pitch","Right_Ankle_Roll","Waist","Left_Shoulder_Pitch","Left_Shoulder_Roll",
  "Left_Elbow_Pitch","Right_Shoulder_Pitch","Right_Shoulder_Roll","Right_Elbow_Pitch"])),
"engineai_pm01": dict(zip(
 ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ank_pitch","L_ank_roll",
  "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ank_pitch","R_ank_roll","waist_yaw",
  "L_sho_pitch","L_sho_roll","L_elbow","R_sho_pitch","R_sho_roll","R_elbow"],
 ["J00_HIP_PITCH_L","J01_HIP_ROLL_L","J02_HIP_YAW_L","J03_KNEE_PITCH_L","J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L","J06_HIP_PITCH_R","J07_HIP_ROLL_R","J08_HIP_YAW_R","J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R","J11_ANKLE_ROLL_R","J12_WAIST_YAW","J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L","J16_ELBOW_PITCH_L","J18_SHOULDER_PITCH_R","J19_SHOULDER_ROLL_R",
  "J21_ELBOW_PITCH_R"])),
"fourier_n1": dict(zip(
 ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ank_pitch","L_ank_roll",
  "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ank_pitch","R_ank_roll","waist_yaw",
  "L_sho_pitch","L_sho_roll","L_elbow","R_sho_pitch","R_sho_roll","R_elbow"],
 ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_pitch_joint",
  "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
  "right_hip_yaw_joint","right_knee_pitch_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
  "waist_yaw_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_elbow_pitch_joint",
  "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_elbow_pitch_joint"])),
"stanford_toddy": dict(zip(
 ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ank_pitch","L_ank_roll",
  "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ank_pitch","R_ank_roll","waist_yaw",
  "L_sho_pitch","L_sho_roll","L_elbow","R_sho_pitch","R_sho_roll","R_elbow"],
 ["left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ank_pitch","left_ank_roll",
  "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ank_pitch","right_ank_roll",
  "waist_yaw","left_sho_pitch","left_sho_roll","left_elbow_roll","right_sho_pitch",
  "right_sho_roll","right_elbow_roll"])),
}
GEOM = {  # (root, L ankle, L foot-tip-ish, shoulder, hand, torso/head)
 "unitree_g1":("pelvis","left_ankle_roll_link","left_knee_link","left_shoulder_pitch_link","left_rubber_hand","head_link"),
 "booster_t1_29dof":("Trunk","left_foot_link","Shank_Left","AL1","left_hand_link","H2"),
 "engineai_pm01":("LINK_BASE","LINK_ANKLE_ROLL_L","LINK_KNEE_PITCH_L","LINK_SHOULDER_PITCH_L","LINK_ELBOW_END_L","LINK_HEAD_YAW"),
 "fourier_n1":("base_link","left_foot_pitch_link","left_shank_pitch_link","left_upper_arm_pitch_link","left_end_effector_link","torso_link"),
 "stanford_toddy":("torso","ank_roll_link","left_calf_link","sho_pitch_link","hand","head"),
}
ROLES = list(ROLE["unitree_g1"].keys())
rows = {}
for r, path in XML.items():
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    bp = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
    root, ank, knee, sho, hand, head = GEOM[r]
    leg = float(np.linalg.norm(bp(root) - bp(ank)))
    thigh = float(np.linalg.norm(bp(root) - bp(knee)))
    shank = float(np.linalg.norm(bp(knee) - bp(ank)))
    arm = float(np.linalg.norm(bp(sho) - bp(hand)))
    torso = float(np.linalg.norm(bp(root) - bp(sho)))  # root-to-shoulder: defined for all five
    shoulder_w = float(abs(bp(sho)[1] - bp(root)[1])) * 2.0
    rng = {}
    for role, jn in ROLE[r].items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert jid >= 0, (r, role, jn)
        lo, hi = m.jnt_range[jid]
        rng[role] = float(hi - lo)
    rows[r] = dict(leg=leg, thigh=thigh, shank=shank, arm=arm, torso=torso,
                   shoulder_w=shoulder_w, mass=float(m.body_mass.sum()), dof=int(m.nu),
                   rom=rng,
                   # scale-free shape descriptor
                   shape=[arm/leg, torso/leg, thigh/leg, shank/leg, shoulder_w/leg])

# scale-free morphology distance: shape ratios + joint ROM (rad, already scale-free)
def desc(r):
    return np.array(rows[r]["shape"] + [rows[r]["rom"][k] for k in ROLES], float)
D = np.vstack([desc(r) for r in XML])
scale = D.std(0, ddof=0); scale[scale < 1e-8] = 1.0
Z = D / scale
names = list(XML)
dist = {a: {b: float(np.sqrt(((Z[i]-Z[j])**2).mean())) for j, b in enumerate(names)}
        for i, a in enumerate(names)}
print(f"{'robot':20s} {'leg':>6s} {'arm/leg':>8s} {'torso/leg':>10s} {'mass':>7s} {'dof':>4s}")
for r in names:
    v = rows[r]
    print(f"{r:20s} {v['leg']:6.3f} {v['arm']/v['leg']:8.3f} {v['torso']/v['leg']:10.3f} {v['mass']:7.1f} {v['dof']:4d}")
print("\nscale-free morphology distance (rows = held out, min over the other four = nearest):")
print(f"{'':20s}" + "".join(f"{n[:9]:>11s}" for n in names) + f"{'NEAREST':>18s}{'legratio':>10s}")
for a in names:
    o = {b: dist[a][b] for b in names if b != a}
    nn = min(o, key=o.get)
    print(f"{a:20s}" + "".join(f"{dist[a][b]:11.3f}" for b in names)
          + f"{nn[:17]:>18s}{rows[a]['leg']/rows[nn]['leg']:10.3f}")
out = pathlib.Path("/home/robotixx/snmr/reproducibility/reports/xembodiment_morphology_distance_2026-09-10.json")
out.write_text(json.dumps({"roles": ROLES, "geometry": rows, "distance": dist,
    "nearest": {a: min((b for b in names if b != a), key=lambda b: dist[a][b]) for a in names}},
    indent=2, sort_keys=True) + "\n")
print("\nwrote", out)
