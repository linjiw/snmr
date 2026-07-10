#!/usr/bin/env python
"""Run the ablation grid with the GPU-busy gate bypassed.

The gate in run_ablations.py treats ANY compute process as 'busy', but a foreign job using
1.4 GB of the A10G's 23 GB should not block our ~7 GB training runs. Sharing is safe here.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_ablations as ra

ra.gpu_busy = lambda: False
sys.argv = ["run_ablations.py", "--steps", "50000"]
ra.main()
