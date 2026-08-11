# SNMR paper-completion autoresearch run

- Started: 2026-08-08 03:38 EDT
- Scope: finish the valid multi-trajectory retarget-to-track experiment and revise the paper only from verified evidence.
- Primary metric: explicit multi-motion student macro completion >= 0.80 (or within 5 percentage points of its specialist-teacher ensemble), followed by SNMR-minus-time completion on preregistered ambiguity starts.
- Validity gate: at least 20 ambiguity windows spanning every selected clip; no motion ID, filename, or per-clip normalization reaches a student.
- Scientific decision: SNMR must exceed time by >= 10 completion points on ambiguity starts with a paired interval excluding zero and lose the gain under same-time shuffled latents. Otherwise, with a passing explicit control, report the scoped interface null.
- Safety/compute: deterministic student first; stop on a failed explicit control, nonfinite loss, or preregistered divergence threshold. Do not overwrite recorded runs.

All iterations are recorded in `results.tsv`; kept changes must pass targeted tests before a GPU run.
