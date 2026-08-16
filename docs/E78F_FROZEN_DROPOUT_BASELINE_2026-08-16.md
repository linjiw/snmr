# E78-F — the frozen E70 students under reference-stream dropout (descriptive baseline)

**Date:** 2026-08-16 (run overnight 2026-08-15 → 16). **Type:** descriptive baseline for E78; **not
preregistered, no gate, no claim promoted.** Evaluation-only on the frozen E70 students of all
three training seeds; nothing written under `/data/robotixx/snmr-research/e70/`. Outputs under
`/data/robotixx/snmr-research/e78_masked_fusion/frozen_seed{0,1,2}_{explicit,snmr}/` and
`analysis_frozen_3seed_{general,ambiguity}.json`.

This is the "what dropout does to today's students" curve the E78 launcher was built to produce
first, and which the advisor guidance called *the motivating baseline*. It came out much stronger
than a baseline; this document records it carefully so nobody over- or under-reads it.

---

## 1. What was run

- **Instrument:** the derived E78 trainer in eval-only mode, `flag_dim=0`, loading each frozen E70
  student (`c_prior_explicit`, `a_prior_snmr`; seeds 0, 1, 2) unchanged. **Sanity gate passed for all
  six students** (`SANITY.json`): clean general and ambiguity completion reproduce the seed-exact
  hash-bound E70 reports within 0.004 (tolerance 0.02 = 2.5 E76 sd), start grids identical.
- **Perturbation:** reference-stream dropout, scope `all`, `hold` mode: with per-tick hazard set from
  a target masked fraction *f* and segment length U[lo, hi] ticks at 50 Hz, the reference stream is
  held at its last valid value; the explicit student holds `g` (58-d command + 6-d ref orientation),
  the SNMR student holds its 256-d `[z_t, z_{t+0.1s}]` window; **proprioception is live in both**;
  each arm's own encoder keeps running on live proprio. Same seeded schedule (seed 404) for every arm
  → paired by rollout. Realized masked fractions: 0.098–0.100 / 0.290–0.301 / 0.496–0.513.
- **Grids:** the frozen 1,024-rollout general start grid, and the frozen 69-pair ambiguity grid.
- **Analysis:** `scripts/analyze_e78_dropout.py`: paired all-rollout completion difference with a
  cluster bootstrap (start step / frame pair; seeds as separate clusters), the E77-addendum matched
  subset (rollouts both arms complete cleanly) with McNemar counts, survival difference.

Contrast with E77: E77 held **`z_cmd`** — the shared bottleneck *downstream* of proprioception —
which freezes the policy's own feedback ("the policy stopped running"). E78-F holds the **upstream
reference input** while the encoder keeps reading proprio ("the command went stale"). The E77 memo
(§2.3) predicted these are opposite failure modes and asked for this evaluator.

## 2. Results — SNMR (frozen, replacement arm) minus explicit (frozen), three seeds pooled

General grid, n = 3,072 paired rollouts (matched subset n = 2,073):

| severity | S | E | paired diff [95% CI] | matched-subset S / E | McNemar S-only : E-only | survival diff |
| --- | ---: | ---: | --- | --- | --- | ---: |
| clean | 0.688 | 0.924 | **−0.237** [−0.253, −0.221] | 1.000 / 1.000 | — | −1.30 s |
| f 0.1, seg 5–25 (0.1–0.5 s) | 0.632 | 0.754 | −0.122 [−0.139, −0.105] | 0.823 / 0.875 | 122 : 230 | −0.80 s |
| f 0.1, seg 25–50 (0.5–1 s) | 0.604 | 0.594 | +0.010 [−0.006, +0.027] | 0.786 / 0.715 | 293 : 147 | −0.14 s |
| f 0.3, seg 5–25 | 0.553 | 0.480 | +0.073 [+0.057, +0.088] | 0.735 / 0.626 | 342 : 117 | +0.41 s |
| **f 0.3, seg 25–50** | 0.522 | 0.280 | **+0.242** [+0.225, +0.260] | 0.711 / 0.383 | 732 : 52 | +1.39 s |
| f 0.5, seg 5–25 | 0.517 | 0.253 | +0.264 [+0.246, +0.281] | 0.698 / 0.352 | 757 : 39 | +1.74 s |
| **f 0.5, seg 25–50** | 0.473 | 0.108 | **+0.365** [+0.347, +0.383] | 0.649 / 0.153 | 1050 : 22 | +2.57 s |

Ambiguity grid (69 pairs, n = 3,072 paired rollouts, matched n = 2,292):

| severity | S | E | paired diff [95% CI] | matched S / E | McNemar |
| --- | ---: | ---: | --- | --- | --- |
| clean | 0.754 | 0.975 | −0.221 [−0.242, −0.201] | 1.000 / 1.000 | — |
| f 0.3, seg 5–25 | 0.559 | 0.493 | +0.066 [+0.049, +0.084] | 0.700 / 0.603 | 354 : 131 |
| f 0.3, seg 25–50 | 0.512 | 0.302 | **+0.210** [+0.192, +0.226] | 0.651 / 0.379 | 678 : 53 |
| f 0.5, seg 5–25 | 0.513 | 0.255 | +0.258 [+0.241, +0.276] | 0.653 / 0.332 | 760 : 26 |

Per training seed (general grid, paired diff): 

| cell | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| clean | −0.251 | −0.232 | −0.227 |
| f 0.1 / 5–25 | −0.138 | −0.104 | −0.125 |
| f 0.3 / 5–25 | +0.077 | +0.078 | +0.062 |
| f 0.3 / 25–50 | +0.235 | +0.250 | +0.240 |
| f 0.5 / 5–25 | +0.257 | +0.267 | +0.269 |
| f 0.5 / 25–50 | +0.359 | +0.382 | +0.354 |

## 3. Reading

1. **The explicit student collapses under a stale reference; the SNMR student barely moves.**
   Explicit: 0.924 → 0.754 → 0.480 → 0.108. SNMR: 0.688 → 0.632 → 0.553 → 0.473. The curves cross
   between f = 0.1 and f = 0.3; from f = 0.3 with 0.5–1 s outages onward the *replacement* arm with a
   24 pp clean deficit completes more rollouts **in absolute terms**, on the paired matched subset,
   in all three training seeds, on both grids.
2. **This is not the E77 artefact.** E77's positive lived only in marginal retention and vanished on
   the matched subset (107 : 93). Here the matched subset *widens* the gap (0.711 vs 0.383) and the
   discordant counts are 14 : 1 to 48 : 1. Seed-to-seed spread is ≤ 0.03 against effects of
   0.24–0.37. Evaluation noise (E76 sd 0.008) is irrelevant at this size.
3. **It reverses the reading of E77 rather than contradicting it:** *the latent arm tolerates a stale
   reference but not a stale code.* Freezing `z_cmd` freezes the SNMR encoder's proprioceptive
   feedback (E77: SNMR worse, 100× at 100 ms); freezing only the reference leaves that encoder
   reacting to live proprio (E78-F: SNMR far better). The E77 memo's physical objection was correct
   and load-bearing.
4. **Mechanism candidates (untested):** (a) *lookahead* — the held SNMR window still contains a
   +0.1 s sample the held `g` lacks; plausible for 0.1–0.5 s outages, implausible as the whole story
   for 0.5–1 s outages, and it does not explain the *shape* (SNMR flat while explicit collapses);
   (b) *coupling* — the explicit student sits at teacher parity by tracking `g` tightly, so a stale
   `g` conflicting with live proprio drives it off the tube, while the lossy 64-d code learned from
   the SNMR window makes the encoder lean on proprio and phase and be tolerant of misalignment
   (its clean deficit is the flip side of the same property); (c) an *interaction* with the E70
   window/projection path independent of content. E70's own controls (time code, shuffled latent)
   are the natural discriminators for (c) and can be run frozen the same way (they exist for all
   three seeds).

## 4. What it does and does not license

- **Does not** license a paper claim by itself: unregistered, replacement arm with −0.24 clean, and
  the lookahead confound is unresolved. It is a baseline, recorded as such.
- **Does** change E78's expectations and order:
  * the primary cell f = 0.3 / seg 5–25 shows only +0.07 even for the frozen replacement arm — the
    effect lives at 0.5–1 s outages. Since **no masked arm has been trained yet**, the E78 prereg is
    amended now, transparently: f = 0.3 / seg 25–50 becomes a **co-primary** cell alongside
    f = 0.3 / seg 5–25 (both must be reported; the +0.10 gate applies to each), with the note that
    the addition was made after seeing the *frozen, unmasked* baseline and before any E78 training;
  * `cfut` (unmasked explicit future window `[g_t, g_{t+0.1s}]` through the A-arm path) moves to the
    **first training job**: evaluated frozen-style under the same sweep it decides "lookahead vs
    representation" for E78-F, and it is the paper's C-future arm — one training answers both;
  * the frozen time-code and shuffled-latent students of all three seeds are added to the frozen
    sweep (evaluation only) as the content controls for E78-F.
- **Paper (2026-09-15):** default is to keep this **out** of the submission — it is a robustness
  result and the memo's argument (a second, weaker claim inside an identification paper invites
  "why n = 3, two walks, no fusion arm") still holds; the analysis is ready if the owner decides
  otherwise. UniAct's compression claim must be read before any positioning sentence.
- **Deployment reading, hedged:** for mocap dropouts / teleop stalls of ≥ 0.5 s, today's frozen
  SNMR command student keeps 47–55 % of rollouts on the tube where the explicit student keeps
  11–28 %; the price is 24 pp clean. E78's job is to show that masked-training *fusion* removes
  the price without losing the robustness.

## 5. Provenance

Launcher `scripts/run_e78_masked_fusion.sh frozen {0,1,2} {explicit,snmr}` (gate lowered to
12–14 GB free because the cells use ~1–2 GB; each cell < 1 min). Sanity: `SANITY.json` in each
frozen directory, all `pass: true`. Analysis JSONs listed above; per-seed numbers reproducible with
`analyze_e78_dropout.py --treatment <seed>_snmr --reference <seed>_explicit`.
