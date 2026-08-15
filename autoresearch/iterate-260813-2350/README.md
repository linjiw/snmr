# E71 executable-assay iteration

Started 2026-08-13T23:50 America/New_York to build a preregisterable, same-state, source-valid
command-swap assay (workstream P1 of `docs/PAPER_STRENGTHENING_DESIGN_2026-08-13.md`) on the six
frozen E70 students.  The registered scope, non-goals, and stop rules are in `scope.md`; the frozen
preflight protocol is `docs/E71_COMMAND_SWAP_PROTOCOL.md`.

**Deliverables produced by this iteration (CPU only, no simulator run):** the reset/grid layer
(`snmr/integration/counterfactual_eval.py`), the nominal evaluator
(`scripts/eval_e71_command_swap.py`), the analyzer (`scripts/analyze_e71_command_swap.py`), the
bundle auditor (`scripts/audit_e71_bundle.py`), the write-once freeze generator
(`scripts/prepare_e71_freeze.py`), the gated launcher (`scripts/run_e71_command_swap.sh`), and four
CPU regression suites (65 tests, green).

**State on 2026-08-14:** `DESIGN_DRAFT`.  No E71 artifact root, DRAFT manifest, report, gate,
analysis, or certificate exists.  The evaluator has never been simulator-smoke-tested.

**Two blockers, both recorded in `docs/PLAN_2026-08-14.md`:**

1. `/data/robotixx/snmr-research/e70/POSTPROCESS_COMPLETE` is absent — the launcher refuses even the
   four-environment smoke without it, and only the B4 video pipeline writes it.  (The GPU-memory
   half of this gate is satisfied; the driver-unavailable note in `scope.md` is retracted.)
2. A blocking defect in `counterfactual_eval.py:561-564`: the root state is initialized from motion
   body index 0 (pelvis) while the frozen E70 reset uses `ref_body_index` (`torso_link`).  No E71
   audit can detect it, because all four cells receive the identical wrong initialization.  It must
   be fixed before any DRAFT manifest, which hash-binds the file.
