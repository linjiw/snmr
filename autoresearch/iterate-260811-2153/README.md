# ICRA 2027 execution-plan loop

Started 2026-08-11T21:51:47Z to execute
`docs/ICRA_EXECUTION_PLAN_2026-08-11.md` in its frozen order.  The active metric is completion of
each plan step against its explicit verification and validation criteria, with dated evidence in
`docs/EXECUTION_PLAN_REPORTS.md`.  E70 scientific outputs remain frozen: Phase A is CPU-only and
must not write under `/data/robotixx/snmr-research/e70/students/` or modify the registered
protocol, launcher, or primary analyzer.

The first unbounded-thread pytest attempt was interrupted after 201 passing tests because PyTorch
oversubscribed the heavily loaded host with 59 threads.  The semantically identical CPU-only run
with CUDA hidden, external plugin autoload disabled, and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
completed in 176.60 seconds: 279 passed and 4 skipped.  This execution-only adjustment is kept for
subsequent validation commands; it changes no test or scientific artifact.

The configured GitHub remote is public, so the plan's private-remote condition is not satisfied
and no push is allowed.  A local Git bundle under `/data/robotixx/snmr-research/backups/` is the
authoritative A1 off-worktree backup.

The A5 replay found that the preserved August 9 seed-0 JSON predates the hierarchical multi-seed
schema now frozen in the analyzer.  The original artifact remains untouched; the anonymous bundle
contains a newly replayed, deterministic seed-0 snapshot whose non-anonymous source hash reproduced
exactly twice.  This is source-version provenance, not a change to a result, gate, or E70 input.
