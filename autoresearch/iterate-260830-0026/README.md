# RobotSpec and cross-solver verification pilot

This iteration converts the advisor's MorphoRetarget proposal into the smallest falsifiable
foundation inside SNMR. It implements a versioned, identity-free RobotSpec and a backend-neutral
verification report, then holds motion, kinematics, controller gains, timing, and initial state fixed
while changing only motor torque.

## Result

The contract experiment passes all six registered gates. For torque scales
`0.50/0.75/1.00/1.25`, fixed-reference maximum requested torque ratios are
`0.641/0.427/0.321/0.256`, exactly the expected inverse intervention. The `0.50x` twin produces a
localized low-margin interval at frames 37--41.

The rollout experiment rejects a tempting shortcut. MuJoCo CPU and Newton/MJWarp agree that this
open-loop PD replay fails (0.18 s versus 0.22 s survival, with the same rollout saturation and
divergence failure types), and neither backend produces a monotonic rollout response to torque.
Therefore this proxy is not suitable as a candidate ranker or RL reward. It remains a diagnostic;
the next dynamics-learning gate requires a frozen closed-loop tracker.

## Artifacts

- `dynamics_twins_mujoco.json`: base RobotSpec, four twins, reports, and intervention gates.
- `newton_torque_0.5.json`, `newton_torque_1.25.json`: matched Newton/MJWarp endpoint reports.
- `cross_solver_torque_0.5.json`, `cross_solver_torque_1.25.json`: machine-readable report comparisons.
- `scope.md`: frozen scope, keep metrics, stop rules, and reproduction commands.
- `results.tsv`: keep/kill ledger.

This does not establish learned dynamics adaptation, strong trackability, PhysX parity, or
held-out-robot generalization.
