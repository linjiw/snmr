# RobotSpec and cross-solver verification pilot

**Started:** 2026-08-30 00:26 America/New_York
**Mode:** bounded autoresearch iteration
**Scientific objective:** implement and falsify the minimum contract needed before claiming
RobotSpec-conditioned, physics-verified retargeting or verifiable RL.

## Scope

This iteration may add a versioned RobotSpec, dynamics-twin interventions, backend-neutral failure
reports, and one matched MuJoCo CPU/Newton-MJWarp diagnostic. It must not change existing SNMR
checkpoints, edit recorded run directories, claim unseen-robot learning, or treat open-loop PD as a
strong tracker.

## Keep metrics

Keep the foundation only if:

1. torque twins preserve their kinematic hash and change their dynamics/spec hashes;
2. identity-free model features respond to torque without exposing robot name/path/hash;
3. a fixed reference's normalized torque demand changes monotonically under the intervention;
4. threshold failures are localized to repair-addressable frame intervals;
5. Newton/MJWarp consumes the exact same motion/spec/controller contract without overflow;
6. focused tests and the full repository suite introduce zero failures.

## Stop rules

- Do not use Newton/MJWarp contact gradients: the deployed solver does not expose supported
  differentiability.
- Do not call the open-loop PD result an RL reward if its dynamics-twin response is non-monotonic.
- Do not run PhysX until MJCF to URDF/USD FK, joint order, actuator, frame, and timing parity is
  mechanically checked.

## Verification commands

```text
.venv/bin/python scripts/experiment_dynamics_twins.py \
  --out autoresearch/iterate-260830-0026/dynamics_twins_mujoco.json
/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json autoresearch/iterate-260830-0026/dynamics_twins_mujoco.json \
  --scale 0.5 --out autoresearch/iterate-260830-0026/newton_torque_0.5.json
/home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
  --pilot-json autoresearch/iterate-260830-0026/dynamics_twins_mujoco.json \
  --scale 1.25 --out autoresearch/iterate-260830-0026/newton_torque_1.25.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
git diff --check
```
