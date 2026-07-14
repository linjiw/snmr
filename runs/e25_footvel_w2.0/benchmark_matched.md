# SNMR benchmark — `runs/e25_footvel_w2.0/ckpt.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (42 windows, 19563 frames/s median inference; p10/p90 19221/19720)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0301 | — |
| dof err (rad) | 0.0600 | — |
| stance speed, source-contact mask (m/s) | 0.2887 | 0.1226 |
| stance speed, source-height mask (m/s) | 0.6724 | 0.5579 |
| stance speed, teacher-height mask (m/s) | 0.4174 | 0.1334 |
| stance speed, legacy teacher mask (m/s) | 0.3032 | 0.0741 |
| slide frac, teacher-height mask | 0.4602 | 0.1034 |
| floating frac, teacher-height mask | 0.0221 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.2251 | 0.0517 |
| FS-MANN (cm/f) | 0.2219 | 0.1418 |
| foot height mean (m) | 0.1064 | 0.1059 |
| stance floating mean (m) | 0.0073 | 0.0137 |
| foot floating frac | 0.0196 | 0.0000 |
| pen. mean (m) | 0.0002 | 0.0000 |
| pen. frac | 0.0071 | 0.0017 |
| dof jerk (rad/s³) | 682.1 | 620.9 |
| body jerk (m/s³) | 364.4 | 199.2 |
| joint jumps | 0.0163 | 0.0182 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.3681 | 0.6052 |
