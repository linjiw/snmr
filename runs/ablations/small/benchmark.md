# SNMR benchmark — `/home/ec2-user/work/retarget/snmr/runs/ablations/small/ckpt.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (42 windows, 11939 frames/s inference)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0478 | — |
| dof err (rad) | 0.0937 | — |
| foot skate (m/s) | 0.4121 | 0.0517 |
| slide frac | 0.4853 | 0.0000 |
| FS-MANN (cm/f) | 0.1912 | 0.1418 |
| pen. mean (m) | 0.0005 | 0.0000 |
| pen. frac | 0.0247 | 0.0017 |
| dof jerk (rad/s³) | 658.1 | 620.9 |
| body jerk (m/s³) | 476.0 | 199.2 |
| joint jumps | 0.0138 | 0.0182 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.2779 | 0.6052 |
