# Fixed-G1 amortization audit, preregistration, and result

The machine-readable preregistration is [`fixed_g1_amortization_prereg.json`](fixed_g1_amortization_prereg.json). It is preserved as the pre-result record; its internal status still describes the moment before the registered training run. Compact decisions are append-only in [`results.tsv`](results.tsv).

## Registered single-window result: PASS

The seed-0, 1,000-step run from scratch on canonical `walk1_subject1` frames `[0,128)` passed every frozen gate. This is a fixed-G1 wiring and representational-capacity result only.

| Measure | Registered result | Threshold |
| --- | ---: | ---: |
| Last-20 versus first-20 loss reduction | 0.9845259903 | >=0.75 |
| Final teacher-FK MPJPE | 0.0202687457 m | <0.05 m |
| Final joint MAE | 0.0196268205 rad | <0.10 rad |
| Final local-root position MAE | 0.0018408921 m | <0.05 m |
| Final root-orientation geodesic | 0.0934716910 rad | <0.15 rad |
| Joint-limit violations | 0 | 0 |
| Inverse-permutation maximum absolute error | 4.7683716e-7 | <=1e-5 |

All parameters in the human encoder, robot graph encoder, root head, and shared joint head had finite nonzero gradients on all 1,000 steps. Total loss moved from `1.9795267582` before training to `0.0130084809` after training. Exact bindings include:

- aligned 128-frame window SHA-256: `ab89dad9d4f5e1ee764daf610db450a2d5913198cbab0936b8657bdf59f8804e`;
- final state-dict SHA-256: `943ac36b79f43c2fba1367de6270e95bf79ac9039ccf142b3ba5a3e9943912db`;
- serialized checkpoint SHA-256: `fafcf42b67d2fb828e7b5e1e158efc9865af98e4d95bbf0c591edb09e73143e0`;
- config bytes SHA-256: `4bbc07ce7d95220885aadf071be037c8142d3fe8be946398909dd782e5a0c5b7`;
- provenance-rerun report bytes SHA-256: `c1b079bb7d6cce77f83f893ca97ae8edf0d448b6bfb3afeea97637f1730a42ef`;
- provenance-rerun artifact manifest's canonical `manifest_sha256`: `cb31a20bcb626469f9d318419d10088ede6291a85410b3e0ad4a5fc2c534d530`.

The initial write-once report did not discover external Newton or Isaac Lab repository paths. Those simulators were unused, so the omission does not invalidate the CPU/MuJoCo-only model metrics. The write-once rerun is the preferred provenance reference: it records clean SNMR `5679fc784cee09c76b597ba4e7b6f615289217db`, Newton `7bb6d02d8eeab2cffc3adfa453ddd63799a2ac6a` with `dirty=true`, and clean Isaac Lab `3c6e67bb5c7ada942a6d1884ab69338f57596f77`. The dirty, unused Newton checkout prevents calling the entire external environment clean or release-archival, but its state is disclosed and it does not enter the model computation. The two runs have exactly identical canonical contracts, configs, gates, gradient audits, before/after metrics, full loss histories, state-dict hashes, serialization results, config bytes, and checkpoint bytes. Their report bytes differ because the rerun records the repaired provenance. Use [`g1_overfit_seed0_provenance_rerun/report.json`](g1_overfit_seed0_provenance_rerun/report.json) as the preferred report; keep [`g1_overfit_seed0/report.json`](g1_overfit_seed0/report.json) as the immutable initial record.

The two `checkpoint.pt` binaries remain local generated artifacts and are ignored by Git; each run's tracked `artifact_manifest.json` binds its exact size and SHA-256, and the command above deterministically regenerates the same bytes from the recorded clean SNMR revision.

This result does not establish full fixed-G1 validation, teacher-independent semantic non-inferiority, contact quality, held-out-T1 generalization, or physics feasibility.

## Recommendation

The existing G1 corpus is sufficient for the smallest fixed-target amortization test: 77 paired clips, 496,672 source frames at 30 Hz (4.60 hours), resampled to 827,714 frames on the canonical 50 Hz timeline. Use the fixed 70/7 split already recorded by `scripts/train_phase1.py`.

Start with one deliberately selected overfit window: adapt the full `walk1_subject1` source and pair, then take canonical indices `[0,128)`, covering timestamps 0.00 through 2.54 seconds. Train the 1.584M-parameter fixed-target integration from scratch for 1,000 steps. The pass gate is a 75% loss reduction plus less than 5 cm teacher-FK MPJPE, less than 0.10 rad joint MAE, less than 5 cm local-root MAE, less than 0.15 rad root-angle MAE, zero limit violations, finite gradients through every model block, and inverse-permutation agreement within `1e-5`.

Before any 1,000-step result, the temporal encoder setting was amended from `temporal_positional=false` to `true`. This adds an explicit order signal, addresses the known content-only transformer weakness, and does not change the registered 1,583,754 parameter count. The machine-readable preregistration preserves both the original and effective values.

Because the single-window gate passed, the next learning experiment is one seed-0, 20,000-step screen on all 70 training clips with 64-frame canonical windows. Treat it only as a promotion screen. The real fixed-G1 qualification is a fresh 50,000-step endpoint evaluated on all seven validation clips with full-sequence 128/64 overlap stitching and paired clip bootstrap.

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

The canonical human and teacher contracts, fixed-target encoder/graph/root module, bounded overfit runner, explicit joint-name-to-child-link mapping, root localization/reconstruction, gradient audit, and write-once report/checkpoint manifests now exist. The 1,000-step single-window gate passes. Full-dataset training, validation, and the full-sequence stitcher remain unwired.

The final human-side semantic gate is additionally blocked on a real G1 `SemanticCorrespondence`. The MJCF has no head link, while the current semantic evaluator requires a unique head link. Add a hash-bound virtual semantic head frame; do not assign an anatomically false body just to satisfy the schema. Freeze this before the 20k/final RobotSpec binding, because changing semantic roles changes token and checkpoint hashes.

Contacts do not require a learned head. Derive human contacts from `HumanMotionSpec` and robot contacts from predicted sole FK under the registered protocol. This keeps GMR as a training teacher, not the evaluation ruler.

## Materialized morphology boundary

Commit `554022268b27e4b72953a60038a1394c45be358d` adds a captured-byte G1 MJCF materializer. A physical parent owns each outgoing child anchor; geometry/local length uses scale `s`, mass uses `s^3`, and inertia uses `s^5`. The generator preserves the root spawn and `qpos0`, hash-binds the semantic manifest and realized bilateral body-scale groups, and requires a backend-consumer bundle re-hash before an archival manifest is accepted. The real G1 `+/-15%` tests pass for seeds 0--7, with 50 compiled bodies and 29 hinge joints each; the focused suite is 7/7 passing.

The boundary is strict: these are same-topology G1-only MJCF assets using per-semantic-path-depth draws, not yet physically interpretable human segment-family variants across G1/H1/H1-2. Nonzero joint-limit shifts are disabled. Collision/contact behavior, standing-state parity, randomized-pose FK parity, URDF/USD counterparts, teacher retargeting, and a usable training corpus remain absent. Do not count the materializer tests as P4 data or embodiment evidence.

Overall G0 still fails on the frozen controller/standing audit. No dynamics model, physics critic, preference label, or repair training is licensed by either the single-window PASS or the materializer.

Final repository verification used the recorded non-oversubscribed CPU protocol:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Result: `638 passed, 5 skipped, 27 warnings in 177.40 s`. The warnings are the existing differentiable-zero tensor-cast warning and TorchScript deprecation warnings; no test failed.
