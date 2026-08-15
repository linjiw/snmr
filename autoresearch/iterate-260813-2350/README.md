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

**State on 2026-08-15:** `DESIGN_DRAFT`.  No E71 artifact root, DRAFT manifest, report, gate,
analysis, or certificate exists.  The evaluator has never been simulator-smoke-tested — that remains
the dominant risk on this workstream.

**Gate status:**

1. `/data/robotixx/snmr-research/e70/POSTPROCESS_COMPLETE` — **satisfied** as of
   `2026-08-15T03:48:51Z`, when the B4 video pipeline completed.  The GPU-memory half of the gate is
   also satisfied; the driver-unavailable note in `scope.md` is retracted.
2. ~~A blocking defect in `counterfactual_eval.py:561-564`.~~ **RETRACTED 2026-08-15 — no such defect
   exists, and the fix it prescribed would have introduced one.**  The frozen E70 reset
   (`wbt_bodyfix.py:118-122`) reads the `root_*_w` family, which resolves to `wbt.py:877-890` and
   indexes a literal `0`; `wbt.py:862` is `ref_pos_w`, a different property the reset never calls.
   `MotionLoader.body_pos_w` also reorders into simulator body order (excluding `world`), so slot 0
   is the pelvis — the free-joint body `robot_root_states` describes.  The E71 code already matched
   the frozen convention.  The literals have since been named, documented, guarded by a fail-closed
   `assert_frozen_root_convention()`, and covered by a byte-equality test against the frozen reset.
   Full retraction: `docs/PLAN_2026-08-14.md` Track D.
