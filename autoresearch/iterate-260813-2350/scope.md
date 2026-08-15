# E71 executable-assay iteration

**Started:** 2026-08-13 23:50 America/New_York  
**Mode:** bounded autoresearch iteration  
**Scientific objective:** close the target-specific evidence rung with a preregisterable,
same-state, source-valid command swap on the six frozen E70 students, while explicitly treating
the crossed state-command combination as outside observed joint support.

## Scope

This iteration may add E71 evaluator, launcher, auditor, protocol, and tests.  It must not edit
frozen E70 artifacts, change E70 results, train a new controller, or launch discretionary GPU
work while the existing B4/GPU gates are closed.

## Keep metric

Keep the iteration only if all of the following hold:

1. the evaluator implements the frozen 69-pair four-cell state/command grid;
2. state and command cursors, complete Markov/policy state, semantic observations, warm-up
   callback order, zero MJWarp overflow, checkpoint inputs, and report layout fail closed;
3. the analyzer and bundle auditor enforce the prewritten explicit/SNMR decision branches;
4. DRAFT, independent smoke certificate, and PREREGISTERED child form an immutable, audited
   transition that cannot be skipped;
5. focused tests and the full repository test suite introduce zero failures;
6. no frozen/generated E70 output is modified.

## Verification

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest <E71 tests> -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python -m py_compile <E71 Python files>
git diff --check
```

The live Holosoma smoke test remains mandatory before preregistration, but is an external gate,
not a reason to weaken or reinterpret CPU verification.  At iteration start,
`POSTPROCESS_COMPLETE` is absent and `nvidia-smi` cannot communicate with the NVIDIA driver.
**[RETRACTED 2026-08-14: the driver claim is stale — 30,827 MiB free. Only `POSTPROCESS_COMPLETE`
still blocks. See `docs/PLAN_2026-08-14.md`.]**
