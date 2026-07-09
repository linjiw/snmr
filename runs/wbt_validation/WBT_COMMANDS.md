# WBT tracking validation — run on an IsaacSim-capable machine

SNMR-retargeted and GMR-teacher clip sets are in `snmr/` and `gmr/`, both in holosoma WBT
format (schema-validated against the shipped sample — see manifest.json). The ONLY difference
the tracking policy sees is the retargeting source, so this isolates retargeting → tracking
quality (the GMR-paper thesis).

## Per-clip policies (GMR-paper protocol: one policy per trajectory, identical config)
```bash
cd holosoma
source scripts/source_isaacsim_setup.sh

for SRC in snmr gmr; do
  for CLIP in <clip names>; do
    python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt logger:wandb \
      --command.setup_terms.motion_command.params.motion_config.motion_file=<PATH>/$SRC/${CLIP}_mj.npz \
      --run_name ${SRC}_${CLIP}
  done
done
```

## Metrics to compare (SNMR vs GMR, paired by clip)
- success rate (episode reaches end without early-termination), sim and sim-dr
- E_g-mpbpe / E_mpbpe / E_mpjpe tracking errors (GMR-paper conventions)
- sim2sim MuJoCo eval via holosoma_inference (deploy stack)

## Local pre-check (no IsaacSim): confirm every NPZ loads in MotionLoader
```bash
python - <<'PY'
from holosoma.managers.command.terms.wbt import MotionLoader  # adjust import if needed
import glob
for f in glob.glob('<PATH>/{snmr,gmr}/*_mj.npz'):
    MotionLoader(f)  # should not raise
PY
```

Clips exported: ['dance2_subject4', 'fight1_subject3', 'walk1_subject5']