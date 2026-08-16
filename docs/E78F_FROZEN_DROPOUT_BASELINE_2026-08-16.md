# E78-F — the frozen E70 students under reference-stream dropout

**Date:** 2026-08-16. **Type:** descriptive baseline for E78; **not preregistered, no gate, no claim
promoted.** Evaluation-only on the frozen E70 students, **all five arms × three training seeds**
(15 students, 11 evaluation cells each). Nothing written under `/data/robotixx/snmr-research/e70/`;
outputs under `/data/robotixx/snmr-research/e78_masked_fusion/frozen_seed{0,1,2}_{explicit,snmr,time,shuffled,proprio}/`.

**Headline, after controls: there is no latent-content effect here.** The first cut looked like a
large robustness advantage for the SNMR command student. The content controls dissolve it — a
*content-free clock* is the most robust arm of all — and the goal-blind floor arm shows what is
really happening: **a stale explicit reference is far worse than no reference at all**, while every
window-fed arm merely decays toward its own goal-free floor. That is a methods finding about
degraded-command studies, and it is the fifth favored story this project's controls have killed.

---

## 1. What was run

- **Instrument:** the derived E78 trainer in eval-only mode, `flag_dim=0`, loading each frozen E70
  student unchanged. **Sanity gate passed for all 15 students** (`SANITY.json`): clean general and
  ambiguity completion reproduce the seed-exact hash-bound E70 reports, start grids identical.
- **Perturbation:** reference-stream dropout, scope `all`, `hold` mode — per-tick hazard set from a
  target masked fraction *f*, segment length U[lo, hi] ticks at 50 Hz; the arm's upstream reference
  is held at its last valid value while **proprioception stays live** and the encoder keeps running.
  Identical seeded schedule (404) for every arm → paired by rollout. Realized masked fractions
  0.098–0.100 / 0.290–0.301 / 0.496–0.513.
- **Arms:** explicit (reference), SNMR latent window, E70 time code (content-free), matched-phase
  shuffled latent (wrong clip), and proprio-only. **Dropout is a structural no-op for the proprio
  arm** — its encoder reads no reference — so its curve is the flat line the others are read against.
- **Grids:** frozen 1,024-rollout general grid, frozen 69-pair ambiguity grid. Analysis:
  `scripts/analyze_e78_dropout.py` (paired cluster bootstrap; E77 matched subset; McNemar; survival).

## 2. Results — completion by arm, three seeds pooled (n = 3,072 paired rollouts per cell)

**General grid**

| severity | explicit | SNMR | time code | shuffled | proprio (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | **0.924** | 0.688 | 0.571 | 0.531 | 0.419 |
| f 0.1, seg 5–25 (0.1–0.5 s) | 0.754 | 0.632 | 0.563 | 0.528 | 0.422 |
| f 0.1, seg 25–50 (0.5–1 s) | 0.594 | 0.604 | 0.557 | 0.529 | 0.430 |
| f 0.3, seg 5–25 | 0.480 | 0.553 | 0.559 | 0.514 | 0.429 |
| f 0.3, seg 25–50 | 0.280 | 0.522 | **0.555** | 0.496 | 0.439 |
| f 0.5, seg 5–25 | 0.253 | 0.517 | **0.560** | 0.502 | 0.424 |
| f 0.5, seg 25–50 | **0.108** | 0.473 | **0.533** | 0.483 | 0.434 |

**Ambiguity grid** (69 pairs)

| severity | explicit | SNMR | time code | shuffled | proprio |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.975 | 0.754 | 0.564 | 0.553 | 0.447 |
| f 0.3, seg 5–25 | 0.493 | 0.559 | 0.556 | 0.528 | 0.438 |
| f 0.3, seg 25–50 | 0.302 | 0.512 | 0.556 | 0.516 | 0.438 |
| f 0.5, seg 5–25 | 0.255 | 0.513 | 0.549 | 0.530 | 0.438 |

Paired SNMR − explicit differences (general grid, three seeds pooled, cluster bootstrap):
+0.242 [+0.225, +0.260] at f 0.3 / 25–50 and +0.365 [+0.347, +0.383] at f 0.5 / 25–50, matched-subset
McNemar 732 : 52 and 1050 : 22, per-seed spread ≤ 0.03. **The same contrasts for the content-free
time code are larger: +0.275 and +0.424.**

## 3. Reading

1. **The apparent SNMR robustness is not about the latent.** The time-code arm carries a fixed
   sinusoidal function of frame index — no motion content, no clip identity, by construction (it is
   E70's registered null) — and it is the *flattest and most robust* arm at every severe cell,
   beating SNMR. The shuffled-latent arm (wrong clip's latent) matches SNMR too. Any explanation
   that appeals to what `z_ret` encodes is excluded by its own controls.
2. **What actually happens: a stale reference is worse than no reference.** The explicit student
   falls to 0.108 — **far below the 0.434 goal-blind floor** measured on the same cell. Holding a
   reference that no longer matches the robot's state actively drives the policy off the tracking
   tube; deleting the reference entirely (proprio arm) leaves 0.43. Meanwhile every window-fed arm
   decays *toward* its own floor and stops there. Nothing here is "graceful degradation"; it is
   "how much did this arm depend on the channel we corrupted, and does the stale value actively
   mislead it".
3. **Therefore the ranking under channel corruption mostly measures channel reliance.** The arm at
   teacher parity is the arm that tracks its reference most tightly, so it is the arm a stale
   reference hurts most. An arm that half-ignores its command channel (clean 0.53–0.69 on this
   two-walk tube, whose goal-free floor is 0.42–0.45) has little left to lose. This generalises the
   sentence already in the paper's Limitations — *"a controller that safely ignores a corrupted
   command scores identically to one that falls"* — into a measured, sharper form: **the controller
   that ignores the command wins the robustness comparison, and a content-free clock wins it
   outright.**
4. **It also completes E77's picture.** E77 held `z_cmd` (downstream, freezing the encoder's own
   proprioceptive feedback) and found SNMR *worse*; E78-F holds the upstream reference and finds
   SNMR *better*. Both are consistent with (2)–(3): what changes is which arm's live feedback is
   being cut, not which representation is better. The E77 memo's physical objection to the `z_cmd`
   axis was correct and load-bearing.
5. **Lookahead is not needed as an explanation** and is not supported: the time code has no
   lookahead beyond a deterministic clock, yet it wins. `cfut` (explicit future window) remains
   worth running for the paper's A/C window question, but it is no longer the decisive control for
   *this* observation — the time code already decided it.

## 4. Consequences

### 4.1 A flaw in the E78 preregistration, found before any masked arm was trained

The registered E78 primary is `mZf − mE ≥ +0.10` under dropout. E78-F shows that gate is **nearly
guaranteed for trivial reasons**: any arm that leans less on the reference clears it, including a
clock. The primary is therefore amended, transparently and before any E78 training run, to a
**conjunction** — all must hold:

1. `mZf − mE ≥ +0.10` at each co-primary cell (f = 0.3, seg 5–25 and seg 25–50), CI excluding zero,
   three seeds, both clips; **and**
2. `mZf − mTl ≥ +0.05` and `mZf − mTf ≥ +0.05` at those cells with CIs excluding zero — the fused
   arm must beat the *content-free* controls under identical masking, not merely beat mE; **and**
3. clean regression `mZf − mE ≥ −0.01` (unchanged co-primary).

Every arm's curve must additionally be **reported against the goal-blind floor** (mB, trained under
the same masking) — an arm below its own floor is being actively misled, not merely uninformed, and
an arm at its floor is not using its command.

### 4.2 Reporting rules added

- Never report a cross-arm robustness contrast without the goal-blind floor on the same axes.
- Never report an arm's dropout curve without its clean value on the same plot (E77's lesson) *and*
  the floor (E78-F's lesson).
- The follow-on paper's identity is now concrete and is a methods paper: **what degraded-command
  robustness comparisons actually measure** — channel reliance plus an active-harm term — with four
  worked instances from this program (E65's 1-of-4 attribution, E77's marginal-vs-paired reversal,
  E78-F's content-free winner, and the stale-worse-than-absent floor crossing).

### 4.3 Not for the 2026-09-15 submission

Unregistered, and it is a robustness result inside an identification paper. The paper's existing
Limitations sentence already carries the argument; upgrading it with these data would require
importing the whole E78-F apparatus. Recommend leaving the text as it stands; the analysis is ready
if the owner decides otherwise.

## 5. Provenance

`scripts/run_e78_masked_fusion.sh frozen {0,1,2} {explicit,snmr,time,shuffled,proprio}`; each cell
< 1 min at ~1–2 GB GPU. Sanity gate `scripts/check_e78_frozen_sanity.py` — **tolerance is measured,
not assumed**: two null-arm cells failed a flat 0.02 gate while their own general grids reproduced
to ≤ 0.007, so replay-to-replay noise was measured directly (three fresh replays of one frozen
null arm: spread 0.026 general / 0.018 ambiguity → sd ≈ 0.016), and the gate became two-tier
(0.02 for clean completion ≥ 0.85, 0.05 below). E76's 0.0083 sd was measured on the high-completion
explicit arm and does not transfer to mid-completion arms. Pooled analyses:
`analysis_frozen_3seed_{snmr,time,shuffled,proprio}_{general,ambiguity}.json`.
