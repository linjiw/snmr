# SNMR benchmark — `runs/ablations/no_temporal/ckpt.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (42 windows, 687 frames/s inference)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0395 | — |
| dof err (rad) | 0.0777 | — |
| foot skate (m/s) | 0.3579 | 0.0517 |
| slide frac | 0.4301 | 0.0000 |
| FS-MANN (cm/f) | 0.2153 | 0.1418 |
| pen. mean (m) | 0.0001 | 0.0000 |
| pen. frac | 0.0089 | 0.0017 |
| dof jerk (rad/s³) | 719.7 | 620.9 |
| body jerk (m/s³) | 451.7 | 199.2 |
| joint jumps | 0.0153 | 0.0182 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.2748 | 0.6052 |
