# Flow Retargeting v3: End-to-End Track — Protocol

Registered 2026-07-17, BEFORE any W-gate results (the W0 training job and W1 probe launched
concurrently with this registration; their gates and thresholds are fixed here). Successor to
v2 (`docs/FLOW_RETARGETING_V2_PROTOCOL.md`, closed at entry: E46). Same side-project rules:
main-line priority, stop rules at gates, no post-hoc tuning.

**Status: CLOSED AT ENTRY (E47, same day). W0b FAIL (encoder-joint descent 0.219 vs ≤0.166 —
substrate line dead in all three variants). W1a FAIL (3/21 windows manufacturable vs 60% bar;
oracle-mask support collapses at 64-frame windows). W1b FAIL (absolute diversity 0.0019 rad vs
0.01 floor; anchor jitter beats config jitter 0.74 vs 0.29 diversity/edit ratio but the
corrected-solution set is locally near-unique). W2/W3 never armed. Multimodality must enter at
the teacher level → main-line Track A5. See EXPERIMENT_LOG E47.**

## 0. What E43–E46 force

The inference-time correction line on the FROZEN model is closed (guidance structurally inert,
descent insufficient, decoder-only substrate/conditional retrofits fail). E46's verdict names
the two levers that remain, and v3 is their minimal test:

1. **Encoder-joint substrate (W0):** V0 froze the encoder — noise then teaches the decoder to
   tolerate a fixed latent geometry; it cannot reshape the geometry itself. One confound left
   before "noise-regularized substrate" is fully dead: train encoder+decoder jointly under
   latent noise. This is the cheapest remaining substrate claim (1 GPU-day equivalent; ~10 min
   at our scale).
2. **Multi-solution supervision (W1):** the Dirac conditional came from one-teacher data. The
   projection passes all guards under the oracle mask, so we can manufacture K guard-clean
   solutions per window by jittering the projection budget — IF jitter actually yields distinct
   solutions rather than one fixed point. That premise is measurable for free and must be
   probed BEFORE any dataset build or flow training.

Additionally, both deep-research passes flag the **output-space vs latent-space** question:
guidance/costs in qpos-space are exact (no frozen-decoder pass-through), and our E45
orthogonality finding is specific to the latent. If W1's premise holds, the v3 flow head moves
to OUTPUT space (per-frame qpos residuals relative to the deterministic decode), where the
physics cost gradient is exact and the multi-solution variance actually lives.

## 1. Gates

Shared eval: unchanged frozen protocol (VAL_CLIPS × 6 × 64-frame windows, Gate-1b guards,
paired raw baseline, matched-compute controls).

### W0 — encoder-joint noise substrate (GPU, running at registration)

`train_v0_noise_decoder.py --train_encoder --sigma 0.1` (50/50 clean/noisy mixture unchanged).
- **W0a fidelity guard:** clean MPJPE ≤ baseline + 0.2 cm (same as V0a).
- **W0b descent gate:** z-descent (identical config to v1/V0b) reaches teacher-height stance
  speed ≤ 0.166 m/s (25% better than the v1 0.221). PASS revives the substrate line for W2;
  FAIL closes "noise-regularized substrate" permanently (decoder-only AND joint both dead).

### W1 — multi-solution premise probe (CPU, running at registration)

`probe_multisolution_diversity.py`: K=4 jittered oracle-mask projections per eval window.
- **W1a manufacturability:** ≥ 60% of windows have ≥ 3/4 guard-accepted variants.
- **W1b diversity:** among accepted variants, mean pairwise dof diversity ≥ 20% of the mean
  projection edit scale AND absolute diversity ≥ 0.01 rad. (Diversity below edit scale means
  the jitters collapse to one fixed point; diversity ≥ noise floor means a flow would have
  genuine conditional variance to model. The 0.01 rad floor keeps a degenerate tiny-edit
  regime from passing on ratio alone.)
- PASS unlocks W2's dataset build. FAIL means single-window projection is a contraction to a
  unique solution; manufactured diversity would need *different correctors* (not jitter), and
  W2 is skipped — the generative-retargeter line closes for lack of data, leaving physics-
  repaired supervision of the deterministic SNMR (main-line A5) as the only survivor.

### W2 — output-space conditional flow on manufactured solutions (GPU, only if W1 passes)

Build: for each of the 70 training clips × tiled 64-frame windows, K=4 projected variants
under the *teacher-height oracle mask* (legitimate at TRAINING time — teacher qpos exists for
training clips; the deployment product is the generative model, not the mask). Train the
existing `LatentFlowNet` re-pointed at output space: state = per-frame qpos residual
`Δq = q_variant − q_raw_decode` (root local pose + dof, ~36-d), condition = z_h (frozen
encoder is fine here — it only conditions, never targets). CFM objective unchanged.
- **W2a fidelity:** unguided samples decode within +0.5 cm MPJPE of raw (F1 convention).
- **W2b diversity transfer:** flow endpoint diversity ≥ 50% of the dataset's W1b diversity
  (the flow must actually model the variance, not mode-collapse onto one variant).
- **W2c the payoff gate:** best-of-N=8 sampling (select by EXACT output-space physics cost,
  deployable soft stance signal, no oracle at eval) meets both frozen speed endpoints with
  all guards on the eval windows. This tests the core claim "a one-to-many generative head +
  cheap selection beats every corrector" — sampling replaces steering, which sidesteps the
  entire E45 guidance pathology.

### W3 — WBT hand-off (unchanged; after W2c PASS and B1 resolution)

Best-of-N flow references as the third arm in the calibrated tracking comparison.

## 2. Bounded exploratory arm (no gate): SSL probe

IF deep-research round 2 verifies that contact/foot-velocity pretext heads or masked-motion
objectives measurably structure motion latents, one bounded arm (≤ 1 GPU-day) may retrain the
G1 specialist with the verified auxiliary objective and re-run the E38 contact probe (held-out
F1 vs the 0.127 source-height baseline). This is a *representation* experiment feeding the
main paper's latent-analysis section, not a v3 gate; it proceeds only on verified findings.

## 3. Non-goals

Unchanged from v2 (no reflow, no sampler stochasticity as diversity, no text conditioning),
plus: no latent-space flow retraining (output space or nothing — the latent path was closed
three times), and no per-step guidance in W2 (best-of-N selection only; if selection passes,
OC-Flow-style refinement is a later optimization, not a gate).

## 4. Budget

W0 ≈ 10 GPU-min (running). W1 probe ≈ CPU-hours (running). W2 build ≈ CPU-day (projection is
CPU-bound; parallelizable over 8 cores) + 20k-step flow train ≈ 15 GPU-min + eval. W3 queues
behind main-line WBT. Worst case well under v1's cost; every W2 artifact (multi-solution
windows) doubles as a prototype of main-line physics-repaired supervision.
