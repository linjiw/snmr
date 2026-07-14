# SNMR benchmark — `/home/ec2-user/work/retarget/snmr/runs/gate1_g1/replication/c0_seed1/ckpt.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (42 windows, 20033 frames/s median inference; p10/p90 19681/20086)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0478 | — |
| dof err (rad) | 0.0869 | — |
| stance speed, source-contact mask (m/s) | 0.4669 | 0.1226 |
| stance speed, source-height mask (m/s) | 0.7579 | 0.5579 |
| stance speed, teacher-height mask (m/s) | 0.7327 | 0.1334 |
| stance speed, legacy teacher mask (m/s) | 0.6135 | 0.0741 |
| slide frac, teacher-height mask | 0.7126 | 0.1034 |
| floating frac, teacher-height mask | 0.0034 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.4100 | 0.0517 |
| FS-MANN (cm/f) | 0.2455 | 0.1418 |
| foot height mean (m) | 0.0972 | 0.1059 |
| stance floating mean (m) | 0.0071 | 0.0137 |
| foot floating frac | 0.0551 | 0.0000 |
| pen. mean (m) | 0.0006 | 0.0000 |
| pen. frac | 0.0291 | 0.0017 |
| dof jerk (rad/s³) | 620.6 | 620.9 |
| body jerk (m/s³) | 429.1 | 199.2 |
| joint jumps | 0.0145 | 0.0182 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.3703 | 0.6052 |
