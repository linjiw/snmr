# Environment and data contract

## Software pins

| Component | Frozen version or revision |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| MuJoCo | 3.11.0 |
| GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` |
| Holosoma | `20699ffa20f494b9563aa68601940c53397bf088` |
| Tectonic | 0.16.9 |

GMR is used as the unmodified retargeting teacher and data converter.  Holosoma supplies the G1
model and the MuJoCo/MJWarp tracking runtime.  Their licenses and asset boundaries are recorded in
[the third-party index](../THIRD_PARTY.md).

## Data counts and boundaries

- Retargeter corpus: 77 LAFAN1 clips x 5 humanoids = 385 paired files and 2,483,360 frames.
- E70 scope: 2 LAFAN1 walks, 1 simulated Unitree G1, 5 student arms, 3 registered training seeds,
  and 1,024 evaluation rollouts per arm and seed.
- Primary ambiguity grid: 69 reference-only present-matched, future-divergent pair identities;
  each pair occurs on both trajectory sides.
- Internal quaternion convention: `wxyz`; I/O conversions are explicit.
- Raw LAFAN1-derived files are not redistributed.  The analyzer needs only the frozen per-rollout
  evaluation JSONs and the two specialist reports after training is complete.

The paper claim is an interface measurement in this frozen simulation assay.  It is not a hardware,
sim-to-real, teacher-algorithm-parity, or broad motion-generalization claim.
