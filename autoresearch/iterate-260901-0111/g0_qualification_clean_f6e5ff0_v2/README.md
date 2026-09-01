# G1 G0 qualification evidence

This write-once bundle separates the successful cross-asset FK subgate from the
still-failing controller/standing-state contract. It was generated from detached clean
worktrees for the code that produced the reports:

- SNMR: `f6e5ff04df8eaa82c7680888174fb9b69252b9c8` (`dirty=false`)
- Newton: `7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a` (`dirty=false`)
- Isaac Lab: `3c6e67bb5c7ada942a6d1884ab69338f57596f77` (`dirty=false`)

The Holosoma checkout is recorded as
`20699ffa20f494b9563aa68601940c53397bf088` with `dirty=true` because it contains
generated/untracked files. The workers independently captured and hashed every Holosoma
source, recipe, checkpoint, MJCF, URDF, and USD bundle they consumed; this bundle does not
pretend the untracked USD was recoverable from Git alone.

## FK subgate: pass

The MuJoCo worker sampled 10,000 independent 29-joint configurations uniformly within
the frozen limits. The live Isaac Lab worker loaded the exact G1 USD bundle and read
poses from `Articulation.data.body_link_{pos,quat}_w`; it did not substitute offline
URDF FK.

- maximum position error: `3.932182638741046e-06 m < 1.0e-03 m`
- maximum orientation geodesic error: `3.7592295534926906e-06 rad < 1.0e-03 rad`
- evaluated poses: `10,000 x 7` key links
- result: `g0_evaluated=true`, `g0_pass=true` for the FK subgate

The supervisor also observed clean Isaac framework shutdown, stage closure, simulation
context release, and deletion of the private materialized USD bundle before promoting
the fail-closed worker journal to a pass report.

## Controller and standing-state subgate: fail

The paired source/checkpoint audit records 14 hard failures and 8 missing live checks.
Principal failures include incompatible action scaling, different reset/randomization
contracts, MJCF hip-roll motor clipping at 88 Nm versus 139 Nm in the saved recipe,
0.1 Nm MJCF friction versus zero in the recipe, nominal-state initialization before
`mj_resetData`, and the absence of one identical frozen checkpoint across backends.

Missing evidence includes live nominal/controller snapshots, a same-motion/same-seed
paired reset, training-time USD/Holosoma identity, and explicit MJCF velocity-limit
enforcement. Therefore overall program G0 remains false. No rollout ranking,
preference-label, or tracker-qualification claim follows from the FK pass.

## Artifact hashes

- `g1_mujoco_reference.npz`: `5c4f2448055e71a25377c8b4dafa72b20523bd957651cdc3fec0692d7e014d16`
- `g1_mujoco_reference.json`: `24e1fd8dec4bbf878acffc2f767994c7e26c500e7436c02599caf4d32c4b0416`
- `g1_physx_report.json`: `3b0ebd1981438dc11fed1bffa2c5f2ecbde4469e04c20eba56c6724a38bbd7eb`
- `g0_controller_parity.json`: `70964d80b106d971a0ba7349ca3983d3d2ebf6a84e3e0a18840eef96b6a33706`

The exact absolute commands, environments, input hashes, and source revisions are
embedded in the machine-readable reports.
