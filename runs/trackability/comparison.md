# Trackability proxy — open-loop PD replay stability

Per clip: mean over 3 evenly spaced start windows of 10 s each. Survival = time until root z<0.35 m, root xy off-reference by >0.5 m, or tilt >60 deg. Errors are means while alive. delta = snmr - gmr (survival: positive favors snmr; errors: negative favors it).

| clip | surv gmr (s) | surv snmr (s) | dsurv | dof err gmr (rad) | dof err snmr (rad) | ddof | z err gmr (m) | z err snmr (m) | dz |
|---|---|---|---|---|---|---|---|---|---|
| dance2_subject4 | 0.74 | 0.91 | 0.17 | 0.208 | 0.235 | 0.028 | 0.043 | 0.053 | 0.009 |
| fight1_subject3 | 0.73 | 0.86 | 0.13 | 0.182 | 0.188 | 0.006 | 0.047 | 0.060 | 0.012 |
| walk1_subject5 | 0.98 | 0.85 | -0.13 | 0.242 | 0.214 | -0.028 | 0.070 | 0.082 | 0.012 |
| **mean** | 0.82 | 0.87 | 0.06 | 0.210 | 0.212 | 0.002 | 0.054 | 0.065 | 0.011 |
