# PM01 conditioning diagnosis

**Date:** 2026-09-01
**Scope:** read-only diagnosis of the legacy PM01 leave-one-robot-out (LORO) failure before selecting a new morphology encoder.

## Bottom line

The exact PM01-LORO failure mechanism remains **inconclusive** because the E06 LORO checkpoint and predictions are no longer present. The surviving E06 scalar reports cannot localize error by root, body, or joint.

One hypothesis is nevertheless ruled out: the trained model is **not conditioning-insensitive**. In a topology-safe intervention on a frozen all-five reproduction, replacing only the pooled 32D embodiment code while retaining the target robot's raw 8D node features, adjacency, joint limits, and FK graph made the target output substantially worse. The correct code ranked first for all five trained robots. Wrong-code MPJPE was 1.95x to 15.91x the correct-code result.

The leading, but unproven, explanation is therefore **coverage/extrapolation with identity-like or non-smooth conditioning geometry**, rather than wholesale conditioning neglect. Learned-code distance predicts swap damage only moderately, and the G1/Toddy endpoint pair is a counterexample: they are mutually nearest in learned code distance but mutually worst swaps. A semantic-mapping/localized failure remains unresolved until the historical E06 weights are restored or an exact LORO rerun produces per-body and per-joint outputs.

Machine-readable results are beside this report in `pm01_conditioning_diagnosis.json`.

## Artifact inventory and provenance limitation

| Artifact | Status | SHA-256 / provenance |
|---|---|---|
| Historical E04 all-five checkpoint | Missing locally. Historical path: `/home/ec2-user/work/retarget/snmr/runs/phase2_all5/ckpt_100k_final.pt` | `95bf78a49e40836b06903f4557796b2ead4f1fa02c0cc892e00398553b6f195f`, recorded in `runs/phase2_all5/sharing_gradient_diagnosis_eval.json` |
| Historical E04 final eval | Present: `runs/phase2_all5/final_eval.json` | `dc84960ad336822e9ca5382aa93f6849d9e63464cb38d71dbe8a9e420fcf8c62` |
| Historical E04 log | Present: `runs/phase2_all5/log.jsonl` | `89cb3d5353a2ea90c54c31b862fab47fe8e471e2a00a1075aa21e7cf3b762c65` |
| Historical E06 PM01-LORO checkpoint | **Missing** from the repository, `/data/robotixx`, all accessible `/home` trees, and backups. The trainer's expected name is `runs/phase2_loro_pm01/ckpt.pt`; no checkpoint hash was recorded. | unavailable |
| Historical E06 final eval | Present: `runs/phase2_loro_pm01/final_eval.json` | `d6b3b05ca2bee2323dca29e7739d8af4defad575a180d9c4551b43ca2e662fae` |
| Historical E06 log | Present: `runs/phase2_loro_pm01/log.jsonl` | `25ca588cf86b7467d68a10325f468a20b5684cf0002500122e5d555938a3c8d3` |
| Frozen all-five seed-0 reproduction used for the intervention | Present: `/data/robotixx/snmr-research/e67/phase2_all5_seed0/ckpt.pt` | `de0cf88d8c8fb3ad6dd459e3ec274a1f1e527d5fd7de5eed447140eb3b9fffbc` |
| Reproduction manifest | Present: `/data/robotixx/snmr-research/e67/phase2_all5_seed0/manifest.json` | file hash `38ec492a0f63aca1686f75780f92a0ff59a80e17b164b808e297fa67bc0a18f4`; dataset fingerprint `d5c077105b54eed878509c9d1b85d548d8ef9d517a703a20066c38a5d9a71dd1` |
| Reproduction final eval | Present: `/data/robotixx/snmr-research/e67/phase2_all5_seed0/final_eval.json` | `c97a60b4db17338ec9a4a8baf8455b6ccdfb0baa6744eb3911e1e136b5b42453` |

The reproduction completed 100,000 steps with five robots and approximately 40,000 exposures per robot. Its recorded source Git state was SHA `04dc60d5f218f2ede60b1c6f6356f30e5581fbef`, dirty, with tracked-diff SHA-256 `7a0e20aba1178ec5b26823506b87ee79dabfb377769eeb526b97837479dc2779`. It used Python 3.10.12, PyTorch 2.13.0+cu130, and an RTX 5090. Its dense final PM01 result was 5.813 cm versus the historical E04 5.723 cm, close enough for this causal conditioning probe but not a bit-identical replacement for E04.

## Probe 1: topology-safe conditioning swap

### Why the raw 8D rows were not swapped

The static node tensors have incompatible shapes and serialization semantics:

| Robot | Static tensor | Bodies | DoF |
|---|---:|---:|---:|
| Unitree G1 | 50 x 8 | 50 | 29 |
| Booster T1 | 32 x 8 | 32 | 27 |
| Fourier N1 | 29 x 8 | 29 | 23 |
| EngineAI PM01 | 29 x 8 | 29 | 24 |
| Stanford Toddy | 33 x 8 | 33 | 22 |

Even N1 and PM01, which both have 29 bodies, use different body order and joint semantics. A rowwise raw-feature swap would attach donor offsets and axes to unrelated target bodies and would confound conditioning with a broken serialization contract. The nearest valid intervention was therefore:

- keep the target's 8D per-node static features;
- keep the target adjacency, topology, DoF gathering, limits, and FK model;
- replace only `model.embodiment_encoder(target.static)`, the pooled 32D code, with the code of another trained robot;
- keep the same human latent and target decoder path.

### Registered evaluation subset

- Checkpoint: E67 all-five reproduction above.
- Clips: `walk1_subject5`, `dance2_subject4`, `fight1_subject3`, `run2_subject1`, `jumps1_subject2`, `sprint1_subject4`, `aiming2_subject3`.
- Four deterministic starts per clip from `numpy.linspace(0, frames - 64, 4, dtype=int)`.
- Window length: 64 frames.
- Total: 28 matched windows per target and donor-code condition, 1,792 frames per condition.
- Metric: whole-body target-FK MPJPE against the target robot's GMR pair. All own/best/worst comparisons below are within this one protocol; they are not mixed with the historical 16-window final evaluator.

### Results

"Best wrong" and "worst wrong" refer to donor codes while the target graph stays fixed.

| Target | Own code MPJPE | Own rank | Best wrong code | Best wrong MPJPE | Ratio | Worst wrong code | Worst wrong MPJPE | Ratio |
|---|---:|---:|---|---:|---:|---|---:|---:|
| Unitree G1 | 5.139 cm | 1/5 | Fourier N1 | 22.310 cm | 4.342x | Toddy | 40.382 cm | 7.858x |
| Booster T1 | 4.830 cm | 1/5 | Fourier N1 | 9.417 cm | 1.950x | Toddy | 28.223 cm | 5.843x |
| Fourier N1 | 4.130 cm | 1/5 | Booster T1 | 8.067 cm | 1.953x | Toddy | 31.224 cm | 7.560x |
| EngineAI PM01 | 4.607 cm | 1/5 | Fourier N1 | 9.435 cm | 2.048x | Toddy | 33.241 cm | 7.215x |
| Stanford Toddy | 2.407 cm | 1/5 | Booster T1 | 30.705 cm | 12.755x | Unitree G1 | 38.304 cm | 15.912x |

The best-wrong swap changed target-FK positions relative to the own-code output by 21.06 cm (G1), 7.78 cm (T1), 6.57 cm (N1), 7.83 cm (PM01), and 30.89 cm (Toddy). The conditioning path is therefore not merely numerically active; it changes physically decoded poses at a scale far above the baseline errors.

Across all 20 off-diagonal target/donor pairs, learned-code L2 distance versus swap MPJPE had Spearman rho = **0.5796267685**, p = **0.0073935113**. The same rank association held for learned-code distance versus FK change. This is only moderate morphology directionality:

- T1, N1, and PM01 generally show the expected direction; PM01's nearest code is N1 (L2 1.2565), and N1 is also PM01's least harmful wrong code.
- G1's nearest code is Toddy (L2 1.7514), yet Toddy is G1's worst swap.
- Toddy's nearest code is G1 (L2 1.7514), yet G1 is Toddy's worst swap.

This pattern is consistent with a code that is strongly used but partly categorical or scale-entangled, not a reliably smooth morphology coordinate system.

## Historical PM01 trajectory

The historical logs use their original periodic evaluator. They show persistent zero-shot failure rather than a late-training collapse.

| Step | All-five PM01 MPJPE | All-five DoF MAE | LORO PM01 MPJPE | LORO DoF MAE |
|---:|---:|---:|---:|---:|
| 4,000 | 16.70 cm | 0.290 | 29.13 cm | 0.497 |
| 8,000 | 13.85 cm | 0.265 | 24.47 cm | 0.483 |
| 12,000 | 10.48 cm | 0.223 | 30.24 cm | 0.488 |
| 16,000 | 9.90 cm | 0.201 | 28.37 cm | 0.471 |
| 20,000 | 8.70 cm | 0.202 | 32.03 cm | 0.446 |
| 24,000 | 9.99 cm | 0.184 | 30.28 cm | 0.468 |
| 28,000 | 8.23 cm | 0.170 | 33.86 cm | 0.489 |
| 32,000 | 6.75 cm | 0.155 | 31.25 cm | 0.505 |
| 36,000 | 6.99 cm | 0.152 | 31.01 cm | 0.492 |
| 40,000 | 6.64 cm | 0.144 | 28.53 cm | 0.487 |
| 44,000 | 6.30 cm | 0.143 | 30.79 cm | 0.513 |
| 48,000 | 6.25 cm | 0.150 | 31.26 cm | 0.489 |
| 52,000 | 6.19 cm | 0.138 | 31.07 cm | 0.501 |
| 56,000 | 6.74 cm | 0.143 | 29.90 cm | 0.509 |
| 60,000 | 5.44 cm | 0.127 | 30.62 cm | 0.512 |
| 64,000 | 5.27 cm | 0.120 | 30.45 cm | 0.488 |
| 68,000 | 4.95 cm | 0.121 | 29.73 cm | 0.475 |
| 72,000 | 4.82 cm | 0.119 | 30.27 cm | 0.486 |
| 76,000 | 4.75 cm | 0.119 | 30.70 cm | 0.507 |
| 80,000 | 4.76 cm | 0.113 | 29.53 cm | 0.497 |
| 84,000 | 4.72 cm | 0.114 | 30.54 cm | 0.492 |
| 88,000 | 4.41 cm | 0.111 | 29.70 cm | 0.506 |
| 92,000 | 4.69 cm | 0.110 | 29.70 cm | 0.500 |
| 96,000 | 4.56 cm | 0.109 | 29.29 cm | 0.500 |
| 100,000 | 4.50 cm | 0.109 | 29.42 cm | 0.501 |

The denser historical final evaluation reports PM01 at 29.589 cm / 0.512 rad under LORO versus 5.723 cm / 0.135 rad in the all-five model, a 5.170x MPJPE ratio. The other four trained robots differ by at most about 0.04 cm between E04 and E06, so the failure is specific to the excluded embodiment rather than a globally failed run.

## All-five PM01 anatomy — explicitly **not LORO**

The following localization uses the 28-window all-five reproduction. It is only a reference anatomy for a successfully supervised PM01 and must not be cited as LORO localization.

- Root-position L2 error: 2.042 cm.
- Root-orientation geodesic error: 0.0751 rad.
- Base, torso, and proximal-hip body errors: approximately 2.0-2.5 cm.
- Distal arm body errors: approximately 3-6 cm.
- Ankle/foot body errors: approximately 8.0-9.4 cm.
- Highest joint-angle MAEs are shoulder roll/pitch/yaw and elbows, 0.128-0.173 rad.
- Head-yaw MAE is 0.0169 rad.

This supervised pattern is compatible with ordinary distal FK error accumulation. It neither proves nor rules out a localized semantic mapping failure in E06.

## Reproduction details

Artifact discovery and hashing were executed from `/home/robotixx/snmr` with:

```bash
find /data/robotixx /home/robotixx -type f \
  \( -name 'ckpt_100k_final.pt' -o -path '*/phase2_loro_pm01/*.pt' \
     -o -path '*/phase2_all5/*.pt' -o -iname '*loro*pm01*.pt' \) \
  -printf '%p\t%s bytes\n' 2>/dev/null | sort

sha256sum \
  runs/phase2_all5/final_eval.json runs/phase2_all5/log.jsonl \
  runs/phase2_loro_pm01/final_eval.json runs/phase2_loro_pm01/log.jsonl \
  /data/robotixx/snmr-research/e67/phase2_all5_seed0/ckpt.pt \
  /data/robotixx/snmr-research/e67/phase2_all5_seed0/final_eval.json \
  /data/robotixx/snmr-research/e67/phase2_all5_seed0/manifest.json
```

The intervention was an inline, non-persisted Python program launched as:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'PY'
# Loaded scripts.benchmark.load_model and scripts.train_phase2.RobotContext/VAL_CLIPS.
# Cached human latents once for the 7 clips x 4 linspace starts x 64 frames.
# For each target, decoded with target static/adjacency/graph and each robot's pooled code.
# Reconstructed world root with snmr.data.local_root_to_world.
# Computed target FK with RobotKinematics.forward_kinematics and accumulated MPJPE/FK deltas.
PY
```

The complete algorithm and all persisted numeric endpoints are specified in the adjacent JSON. No model, pair data, run directory, or existing report was modified during diagnosis.

## Classification and required follow-up

- `classification`: `inconclusive_exact_loro`
- `ruled_out`: `conditioning_insensitive`
- `leading_but_unproven`: `coverage/extrapolation_identity_like`
- `semantic_mapping_localized`: unresolved

The smallest decisive follow-up is to restore E06 `ckpt.pt` from the original EC2 artifact store. If it cannot be restored, rerun the exact historical LORO configuration under a new output root and frozen data/source hashes, then compute root, per-body, and per-DoF errors plus this same code-swap test. Until that result exists, the diagnosis should guide P3 priorities but should not be presented as causal proof that morphology augmentation alone will fix PM01.
