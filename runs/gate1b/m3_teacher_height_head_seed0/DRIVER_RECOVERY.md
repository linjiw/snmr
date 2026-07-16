# M3 Driver Recovery

- Training completed normally at step 50,000 under launch revision `921978f`; the manifest and
  checkpoint were complete before the driver failed.
- While training was active, later provenance commits modified `scripts/run_gate1b_m3.sh` in
  place. Bash resumed reading the changed file at the old byte offset and interpreted a fragment
  of `--train_contact_head_only` as the command `head_only`.
- The failed fragment overwrote the raw `train_stdout.log`; the structured manifest,
  `log.jsonl`, diagnostics, and checkpoint were preserved.
- The frozen manifest assertion, mask audit, 42-window projection, and staged analyzer were then
  run unchanged. The copied analyzer required `PYTHONPATH` to import the repository package; the
  source analyzer and driver now include that packaging fix.
- `analysis.json` passes every provenance and tensor-integrity check and returns
  `fail_close_mask_iteration`. The recovery did not alter scientific source paths, checkpoint
  tensors, masks, solver settings, endpoints, or guards.
