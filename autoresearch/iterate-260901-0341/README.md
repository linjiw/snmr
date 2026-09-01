# Fixed-G1 amortization audit and preregistration

The machine-readable preregistration is [`fixed_g1_amortization_prereg.json`](fixed_g1_amortization_prereg.json). No training was run for this audit.

## Recommendation

The existing G1 corpus is sufficient for the smallest fixed-target amortization test: 77 paired clips, 496,672 source frames at 30 Hz (4.60 hours), resampled to 827,714 frames on the canonical 50 Hz timeline. Use the fixed 70/7 split already recorded by `scripts/train_phase1.py`.

Start with one deliberately selected overfit window: adapt the full `walk1_subject1` source and pair, then take canonical indices `[0,128)`, covering timestamps 0.00 through 2.54 seconds. Train the 1.584M-parameter fixed-target integration from scratch for 1,000 steps. The pass gate is a 75% loss reduction plus less than 5 cm teacher-FK MPJPE, less than 0.10 rad joint MAE, less than 5 cm local-root MAE, less than 0.15 rad root-angle MAE, zero limit violations, finite gradients through every model block, and inverse-permutation agreement within `1e-5`.

Before any 1,000-step result, the temporal encoder setting was amended from `temporal_positional=false` to `true`. This adds an explicit order signal, addresses the known content-only transformer weakness, and does not change the registered 1,583,754 parameter count. The machine-readable preregistration preserves both the original and effective values.

If that passes, run one seed-0, 20,000-step screen on all 70 training clips with 64-frame canonical windows. Treat it only as a promotion screen. The real fixed-G1 qualification is a fresh 50,000-step endpoint evaluated on all seven validation clips with full-sequence 128/64 overlap stitching and paired clip bootstrap.

## Frozen root convention

For this G1-only compatibility gate, use the canonical-train fit `s_xy = 0.8749322702593619`. The root target is expressed in the human-yaw frame:

```text
p_local = R_yaw(human)^T (p_robot - [s_xy*h_x, s_xy*h_y, 0])
q_local = conjugate(q_yaw(human)) * q_robot
```

Local Z is absolute world height. Reconstruct world root pose with the inverse transform. The model must predict this root; copying the teacher root invalidates root, FK, and contact metrics. The fitted scale is explicitly forbidden for held-out T1 because it uses G1 target trajectories. A later LORO experiment needs an identity-free graph-conditioned root displacement/velocity contract.

## Data and checkpoints

- Pairs: `/data/robotixx/pairs/unitree_g1`
- Raw BVH: `/data/robotixx/lafan1_bvh`
- Train split SHA-256: `70ffd1b4aab94f1d7ae39d4d1f5332c1b60b6d15ec205626c14924897baeeea6`
- Validation split SHA-256: `c7a00005934ab3893c862e41b6cd56767ba77fb79518c3cd6f9aed985c4d582c`
- G1 MJCF SHA-256: `8c586e4747da85804180fe44d8692e0fd8231356728b6327e256dca498087a78`
- Overfit pair SHA-256: `79565402a381d54122c28b9f1f88bc43a46dfdca9f5aeee0ff0afbaee6c8c845`
- Overfit BVH SHA-256: `4c9d591f323ffa660d9de6ef20c5d7077e66b2102a009feedf01afa22a6fbc73`

The usable 100k all-five checkpoint includes Booster T1 and was trained at 30 Hz, so it is only an integration smoke artifact and must not initialize a held-out-T1 claim. The 2-step G1 checkpoint is not trained. The historical 100k G1 specialist checkpoint is missing; only its benchmark record and expected hash remain. The scientifically clean run is therefore from scratch.

## Remaining blockers

The canonical human and teacher contracts, fixed-target encoder/graph/root module, bounded overfit runner, explicit joint-name-to-child-link mapping, root localization/reconstruction, gradient audit, and write-once report/checkpoint manifests now exist. A real-data two-step smoke reproduced the registered RobotSpec/token hashes, passed all four gradient-block checks, and achieved serialization agreement within `4.77e-7`; it is explicitly ineligible for the scientific gate. The 1,000-step run has not yet been executed. Full-dataset training and the full-sequence stitcher remain unwired.

The final human-side semantic gate is additionally blocked on a real G1 `SemanticCorrespondence`. The MJCF has no head link, while the current semantic evaluator requires a unique head link. Add a hash-bound virtual semantic head frame; do not assign an anatomically false body just to satisfy the schema. Freeze this before the 20k/final RobotSpec binding, because changing semantic roles changes token and checkpoint hashes.

Contacts do not require a learned head. Derive human contacts from `HumanMotionSpec` and robot contacts from predicted sole FK under the registered protocol. This keeps GMR as a training teacher, not the evaluation ruler.
