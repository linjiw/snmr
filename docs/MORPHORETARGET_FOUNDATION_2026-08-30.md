# MorphoRetarget foundation: RobotSpec and verifiable-physics pilot

**Date:** 2026-08-30
**Status:** contract implemented; dynamics-twin signal validated; open-loop rollout reward rejected

## Decision

The advisor guidance improves SNMR's direction in one decisive way: the next claim should not be
"one shared latent serves several trained robots." SNMR already does that, and its complete
leave-one-robot-out result shows that the current 8D static embedding does not generalize reliably.
The next falsifiable claim is instead:

> A variable-DoF retargeter reads a new humanoid's physical specification rather than its identity,
> changes its output under controlled dynamics interventions, and survives independent physics
> verification.

This pass implements the prerequisite contract and verifier evidence. It deliberately does not wire
the new 37D features into old SNMR checkpoints: doing so before establishing decoder permutation,
held-out-robot, and dynamics-label tests would conflate representation and supervision failures.

## What was added

### RobotSpec v0.1

`snmr/robot_spec.py` resolves the existing G1 MJCF into a self-contained specification containing
frames, semantic anchors, kinematic tree, link geometry/collision proxies, mass/COM/inertia,
joint axes/limits, actuator limits/gains, control rate, latency, and simulation rate. The contract:

- separates kinematic, dynamics, and full hashes;
- emits dimensionless, identity-free model features;
- never exposes asset names, paths, hashes, or trainable robot IDs to the model;
- records missing dynamics fields with availability masks instead of invented defaults;
- creates coherent dynamics twins while asserting unchanged kinematics.

MJCF is the first source adapter because that is the common denominator of the current five-robot
SNMR corpus. URDF/USD ingestion must target the same dataclasses after parity tests; it is not an
excuse to change the downstream schema.

### Verification report v0.1

`snmr/verification.py` defines backend identity, candidate/motion/spec/controller hashes, overflow,
metrics, and typed failure intervals. This lets separate SNMR, Isaac Lab, and Newton environments
exchange JSON rather than importing one another's simulator packages. Reports fail closed on
invalid hashes, non-finite metrics, out-of-clip intervals, and overlapping same-type failures.

### Dynamics-twin and Newton runners

`scripts/experiment_dynamics_twins.py` keeps one G1 motion, geometry, PD gains, timing, and initial
state fixed while scaling only effort limits. `scripts/verify_newton_pd.py` replays an endpoint in
Newton's MuJoCo-Warp solver with explicit SNMR `wxyz` to Newton `xyzw` conversion, internal MJWarp
contacts, zero-order-hold targets, direct torque control, overflow auditing, and dirty-aware source
revision provenance.

## Result: the causal signal works

| Torque scale | Fixed-reference max requested torque / limit | MuJoCo CPU survival | Newton/MJWarp survival |
|---:|---:|---:|---:|
| 0.50 | 0.641 | 0.18 s | 0.22 s |
| 0.75 | 0.427 | 0.18 s | not run |
| 1.00 | 0.321 | 0.18 s | not run |
| 1.25 | 0.256 | 0.18 s | 0.22 s |

All six RobotSpec/intervention gates pass: identical kinematic hashes, distinct dynamics hashes,
changed identity-free features, monotonic reference demand, exact inverse scaling, and localized
failure output. At `0.50x`, a sustained low-torque-margin interval is localized to frames 37--41.

This establishes that a future model can be tested for actually using dynamics. It does not show
that the current retargeter does so.

## Negative result: open-loop PD is not a verifiable-RL reward

The fixed reference has no static torque saturation at any scale, while both rollouts diverge almost
immediately. More importantly, rollout saturation does not improve monotonically as torque limits
increase. MuJoCo CPU and Newton/MJWarp agree on the failure categories but not on a useful ranking.

Therefore:

- keep open-loop PD as a smoke diagnostic;
- do not use it to label candidate preferences;
- do not run sequence-level PPO against it;
- use a qualified frozen closed-loop tracker for L2 ranking;
- use best-of-N selection, pairwise preference learning, and failure-localized repair before trying
  an RL repair policy.

This changes "verifiable RL" into a more testable loop:

```text
deterministic generator -> 4-8 local proposals -> L1 screen
  -> frozen tracker in PhysX and MJWarp -> structured failure intervals
  -> low-dimensional CEM/MPPI repair -> successful tuple replay/self-distillation
```

RL becomes optional for choosing repair operators after a supervised repair corpus exists; it is not
the first motion generator or torque controller.

## Newton and Isaac Lab boundary

Current upstream documentation supports using Newton for accelerated rollout and cross-solver
falsification, but not for the originally suggested end-to-end differentiable humanoid contact loss.
Isaac Lab's Newton integration is beta and focused on MuJoCo Warp; its own transition guidance says
that identical USD assets still need solver-specific tuning and exact observation/action/timing
contracts. Newton's solver table marks `SolverMuJoCo` non-differentiable.

Accordingly, the strong design is:

- **Isaac Lab/PhysX:** primary frozen-tracker verifier;
- **Newton/MJWarp:** batched second solver and disagreement detector;
- **SNMR/refeas:** controller-independent L0/L1 checks and failure localization;
- **no simulator gradients in the critical path.**

Official capability references:

- [Isaac Lab Newton integration](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)
- [Newton supported features in Isaac Lab](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/supported-features.html)
- [Solver transition requirements](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/solver-transitioning.html)
- [PhysX/Newton policy-transfer contract](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html)
- [Newton solver capability table](https://github.com/newton-physics/newton/blob/main/docs/solvers/index.rst)

## Next gate: paired PhysX/MJWarp frozen-tracker verification

Do not launch a large training run yet. The next bounded branch should:

1. pair one G1 MJCF with the Isaac Lab URDF/USD asset;
2. prove key-link FK, joint names/order/axes/limits, quaternion frames, standing state, control and
   simulation timing, gains, and effort limits match;
3. export one frozen tracker checkpoint to both backends with identical observation/action logic;
4. replay the same 20 stratified references and dynamics twins;
5. report pass agreement, failure-time delta, rank correlation, saturation, overflow, and a solver
   disagreement taxonomy;
6. accept a physics preference label only when L1 and both strong backends agree, otherwise retain
   it as disagreement data.

Only after this gate should the new features enter the graph encoder and the held-out T1 experiment
begin. The first learned experiment should compare kinematic RobotSpec, full RobotSpec, and full
RobotSpec plus a physics critic; GMR remains the kinematic teacher/oracle, not a source of dynamics
labels.

## Reproduction

```bash
.venv/bin/python scripts/experiment_dynamics_twins.py \
  --out autoresearch/iterate-260830-0026/dynamics_twins_mujoco.json

/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json autoresearch/iterate-260830-0026/dynamics_twins_mujoco.json \
  --scale 0.5 --out autoresearch/iterate-260830-0026/newton_torque_0.5.json

/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json autoresearch/iterate-260830-0026/dynamics_twins_mujoco.json \
  --scale 1.25 --out autoresearch/iterate-260830-0026/newton_torque_1.25.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

The experimental artifacts are under `autoresearch/iterate-260830-0026/`; no existing run summary
or generated experiment directory was overwritten.
