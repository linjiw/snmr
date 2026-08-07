# E52 observation-ordering leak probe (2026-07-31, paper review)

**Finding (DEFECT-2):** `scripts/train_e52_dagger.py` splits `actor_obs` assuming the
58-d `motion_command` term is FIRST (`full[:, :58]`), but holosoma's ObservationManager
concatenates observation terms **alphabetically**
(`managers/observation/manager.py:141`): `actions[0:29]`, `base_ang_vel[29:32]`,
`dof_pos[32:61]`, `dof_vel[61:90]`, **`motion_command[90:148]`**,
`motion_ref_ori_b[148:154]`.

Verified at pinned holosoma rev `9fb2b57` by runtime probe:
`max|actor_obs[:, 90:148] − motion_command.command| = 0.0` while
`max|actor_obs[:, 0:58] − motion_command.command| = 6.84`.

**Consequences for E52 v1–v3 and the 3-seed replication:**
- "proprio" = `full[:, 58:]` **contains the full explicit reference**
  (`motion_command` at proprio[32:90], `motion_ref_ori_b` at proprio[90:96]) →
  the decoder saw the goal directly; the C1 exclusivity contract ("z is the actor's
  only motion command") did not hold in any E52 run.
- The prior's "cmd" input (`full[:, :58]`) was actually `actions + base_ang_vel +
  dof_pos[:26]` — proprioceptive, not the reference. All four arms' priors saw the
  true goal only through "proprio". The v2→v3 "goal-conditioned prior" attribution
  is therefore confounded with dropping the 50% prior-z action-loss mix.

**Causal probes** (seed-1 arm C student, 256 rollouts, eval seed 404, this dir):

| eval variant | completion | survival (s) |
|---|---|---|
| prior path (paper condition) | 0.930 | 9.37 |
| blank leaked cmd only (proprio[32:90] = 0 at decoder) | 0.109 | 3.48 |
| blank cmd + ref_ori (proprio[32:96] = 0) | 0.000 | 0.70 |
| z = 0 (keep leaked proprio) | 0.000 | 0.93 |
| z ~ N(0,1) | 0.000 | 0.93 |

Both channels are load-bearing: the policy uses the leaked explicit reference AND
z_cmd. The 0.952 headline number does not measure a z-only command interface.

Note: zero_z/rand_z are off-manifold inputs for the decoder, so their collapse alone
would not prove z carries goal information; the decisive row is blank_cmd_only —
removing the leaked reference destroys the policy even with the intact prior z.

L1/L1R/E49 frozen-latent baselines are unaffected (they replace the
`motion_command` obs term at config level; no manual index split).

**Fix:** split by the alphabetical layout (cmd = `full[:, 90:148]`; decoder proprio
must exclude both `[90:148]` and `motion_ref_ori_b [148:154]`), or set the obs term
funcs at config level like the L1 runs did. Then re-run E52 v3 + seeds (and E58/E59/E60,
whose designs inherit the same splitter).

Artifacts: `eval_patch.diff` (the probe-only eval patch applied to a copy of the
trainer; training code untouched), five eval jsons.
