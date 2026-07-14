# SNMR benchmark — `/home/ec2-user/work/retarget/snmr/runs/gate1_g1/screen/c1_bce_seed0/ckpt.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (42 windows, 19041 frames/s median inference; p10/p90 18897/19057)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0575 | — |
| dof err (rad) | 0.1152 | — |
| stance speed, source-contact mask (m/s) | 0.4874 | 0.1226 |
| stance speed, source-height mask (m/s) | 0.7677 | 0.5579 |
| stance speed, teacher-height mask (m/s) | 0.7540 | 0.1334 |
| stance speed, legacy teacher mask (m/s) | 0.7423 | 0.0741 |
| slide frac, teacher-height mask | 0.7428 | 0.1034 |
| floating frac, teacher-height mask | 0.0622 | 0.0000 |
| contact-head F1 vs teacher-height | 0.1765 | — |
| legacy foot skate (m/s) | 0.4308 | 0.0517 |
| FS-MANN (cm/f) | 0.2336 | 0.1418 |
| foot height mean (m) | 0.0927 | 0.1059 |
| stance floating mean (m) | 0.0038 | 0.0137 |
| foot floating frac | 0.0211 | 0.0000 |
| pen. mean (m) | 0.0007 | 0.0000 |
| pen. frac | 0.0345 | 0.0017 |
| dof jerk (rad/s³) | 641.4 | 620.9 |
| body jerk (m/s³) | 481.6 | 199.2 |
| joint jumps | 0.0158 | 0.0182 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.3146 | 0.6052 |
