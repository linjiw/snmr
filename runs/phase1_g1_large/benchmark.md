# SNMR benchmark — `runs/phase1_g1_large/ckpt_100k_final.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (28 windows, 631 frames/s inference)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0362 | — |
| dof err (rad) | 0.0642 | — |
| foot skate (m/s) | 0.2545 | 0.0472 |
| slide frac | 0.3189 | 0.0000 |
| FS-MANN (cm/f) | 0.2152 | 0.1323 |
| pen. mean (m) | 0.0002 | 0.0000 |
| pen. frac | 0.0112 | 0.0026 |
| dof jerk (rad/s³) | 568.3 | 527.3 |
| body jerk (m/s³) | 342.1 | 177.4 |
| joint jumps | 0.0050 | 0.0067 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.2872 | 0.5668 |
