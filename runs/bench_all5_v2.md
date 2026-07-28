# SNMR benchmark — `runs/phase2_all5/ckpt_100k_final.pt`

SNMR scored against the GMR teacher on held-out clips (walk1_subject5, dance2_subject4, fight1_subject3, run2_subject1, jumps1_subject2, sprint1_subject4, aiming2_subject3). Teacher rows = the optimization baseline's own metric values (no MPJPE: it is the reference).

## unitree_g1  (42 windows, 312 frames/s median inference; p10/p90 218/509)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0604 | — |
| dof err (rad) | 0.1181 | — |
| stance speed, source-contact mask (m/s) | 0.4970 | 0.1226 |
| stance speed, source-height mask (m/s) | 0.7696 | 0.5579 |
| stance speed, teacher-height mask (m/s) | 0.8154 | 0.1334 |
| stance speed, legacy teacher mask (m/s) | 0.7174 | 0.0741 |
| slide frac, teacher-height mask | 0.7484 | 0.1034 |
| floating frac, teacher-height mask | 0.0178 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.4404 | 0.0517 |
| FS-MANN (cm/f) | 0.2003 | 0.1418 |
| foot height mean (m) | 0.0955 | 0.1059 |
| stance floating mean (m) | 0.0086 | 0.0137 |
| foot floating frac | 0.0593 | 0.0000 |
| pen. mean (m) | 0.0016 | 0.0000 |
| pen. frac | 0.0580 | 0.0017 |
| dof jerk (rad/s³) | 526.4 | 620.9 |
| body jerk (m/s³) | 445.5 | 199.2 |
| joint jumps | 0.0140 | 0.0182 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.1637 | 0.6052 |

## booster_t1_29dof  (42 windows, 1151 frames/s median inference; p10/p90 1001/1263)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0511 | — |
| dof err (rad) | 0.1188 | — |
| stance speed, source-contact mask (m/s) | 0.4191 | 0.0303 |
| stance speed, source-height mask (m/s) | 0.6120 | 0.3455 |
| stance speed, teacher-height mask (m/s) | 0.5381 | 0.2010 |
| stance speed, legacy teacher mask (m/s) | 0.4752 | 0.0369 |
| slide frac, teacher-height mask | 0.5642 | 0.1534 |
| floating frac, teacher-height mask | 0.0713 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.4227 | 0.0379 |
| FS-MANN (cm/f) | 0.2344 | 0.3157 |
| foot height mean (m) | 0.0334 | 0.0417 |
| stance floating mean (m) | 0.0055 | 0.0101 |
| foot floating frac | 0.0487 | 0.0000 |
| pen. mean (m) | 0.0042 | 0.0000 |
| pen. frac | 0.1854 | 0.0000 |
| dof jerk (rad/s³) | 552.7 | 627.2 |
| body jerk (m/s³) | 346.9 | 255.6 |
| joint jumps | 0.0133 | 0.0123 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.0025 | 0.4399 |

## fourier_n1  (42 windows, 1314 frames/s median inference; p10/p90 1248/1366)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0486 | — |
| dof err (rad) | 0.1383 | — |
| stance speed, source-contact mask (m/s) | 0.4399 | 0.1063 |
| stance speed, source-height mask (m/s) | 0.6771 | 0.4917 |
| stance speed, teacher-height mask (m/s) | 0.5502 | 0.1669 |
| stance speed, legacy teacher mask (m/s) | 0.4224 | 0.0483 |
| slide frac, teacher-height mask | 0.5444 | 0.1273 |
| floating frac, teacher-height mask | 0.0311 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.3899 | 0.0462 |
| FS-MANN (cm/f) | 0.2212 | 0.2474 |
| foot height mean (m) | 0.0918 | 0.1027 |
| stance floating mean (m) | 0.0062 | 0.0110 |
| foot floating frac | 0.0487 | 0.0000 |
| pen. mean (m) | 0.0005 | 0.0000 |
| pen. frac | 0.0334 | 0.0000 |
| dof jerk (rad/s³) | 633.8 | 784.8 |
| body jerk (m/s³) | 335.5 | 300.0 |
| joint jumps | 0.0182 | 0.0160 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.0836 | 0.5232 |

## engineai_pm01  (42 windows, 1213 frames/s median inference; p10/p90 1105/1320)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0502 | — |
| dof err (rad) | 0.1221 | — |
| stance speed, source-contact mask (m/s) | 0.4781 | 0.1084 |
| stance speed, source-height mask (m/s) | 0.7323 | 0.5219 |
| stance speed, teacher-height mask (m/s) | 0.5559 | 0.2009 |
| stance speed, legacy teacher mask (m/s) | 0.4267 | 0.0383 |
| slide frac, teacher-height mask | 0.5576 | 0.1406 |
| floating frac, teacher-height mask | 0.0474 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.4277 | 0.0441 |
| FS-MANN (cm/f) | 0.2188 | 0.3052 |
| foot height mean (m) | 0.0972 | 0.1057 |
| stance floating mean (m) | 0.0070 | 0.0092 |
| foot floating frac | 0.0660 | 0.0000 |
| pen. mean (m) | 0.0006 | 0.0000 |
| pen. frac | 0.0314 | 0.0000 |
| dof jerk (rad/s³) | 582.8 | 661.7 |
| body jerk (m/s³) | 365.1 | 229.3 |
| joint jumps | 0.0136 | 0.0111 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.0040 | 0.4599 |

## stanford_toddy  (42 windows, 1167 frames/s median inference; p10/p90 976/1237)
| metric | SNMR | teacher (GMR) |
|---|---|---|
| MPJPE (m) | 0.0292 | — |
| dof err (rad) | 0.1594 | — |
| stance speed, source-contact mask (m/s) | 0.2254 | 0.0462 |
| stance speed, source-height mask (m/s) | 0.3220 | 0.2140 |
| stance speed, teacher-height mask (m/s) | 0.3103 | 0.1906 |
| stance speed, legacy teacher mask (m/s) | 0.2176 | 0.0437 |
| slide frac, teacher-height mask | 0.3604 | 0.1848 |
| floating frac, teacher-height mask | 0.0054 | 0.0000 |
| contact-head F1 vs teacher-height | — | — |
| legacy foot skate (m/s) | 0.2211 | 0.0473 |
| FS-MANN (cm/f) | 0.2868 | 0.3954 |
| foot height mean (m) | 0.0378 | 0.0452 |
| stance floating mean (m) | 0.0028 | 0.0078 |
| foot floating frac | 0.0164 | 0.0000 |
| pen. mean (m) | 0.0011 | 0.0000 |
| pen. frac | 0.0461 | 0.0000 |
| dof jerk (rad/s³) | 598.4 | 483.3 |
| body jerk (m/s³) | 181.5 | 125.0 |
| joint jumps | 0.0175 | 0.0106 |
| limit viol. | 0.0000 | 0.0000 |
| limit prox. | 0.2266 | 0.5167 |
