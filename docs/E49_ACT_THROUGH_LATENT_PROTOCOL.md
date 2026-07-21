# E49 — Act-Through-Latent WBT Co-Training (v-next #1)

**Registered 2026-07-20, before any run.** Follow-up to the Track-B latent-command line (E36–E39)
and the #1 recommendation of `docs/FORWARD_RESEARCH_MEMO.md`. Companion prior art:
`docs/WBT_LATENT_LITERATURE_REVIEW.md` §7.4 ("make the latent load-bearing rather than
concatenated"). This document is the design + stop rules; results append to `docs/EXPERIMENT_LOG.md`.

## 1. The question

E39 established that a **frozen** SNMR latent, *concatenated* beside the explicit robot-space
reference, does not improve WBT tracking (S3 non-replication; C3 explains it). The L1 arm removed
the explicit reference from the actor (latent-only command) and reached **72–77% completion**
across seeds — feasibility, but **8–18 pp below** the explicit-command baseline. Two verified
external results locate the missing lever:

- **UniTracker (arXiv:2507.07356, CONFIRMED):** a command latent **co-trained with the controller**
  improves tracking (SR 91.82 vs 88.03), and the gain widens under noise. Crucially: "when the
  actor receives the reference motion directly, the influence of the latent variable z vanishes" —
  so the latent must be the actor's *only* command (which L1 already does).
- **PULSE / MaskedMimic (CONFIRMED):** downstream policies benefit because they **act through** a
  control-trained decoder/latent, not from concatenation; the load-bearing piece is the residual
  prior, not the VAE wrapper.

E39's frozen latent and L1's plain readout share one defect: the map from motion to command is
**not trained under the control objective**, and SNMR's exported z is **offline / non-causal**
(full-clip bidirectional encoding, no positional encoding — `docs/20260716-1931-sol.md` P1).

**E49 asks:** does making the SNMR-latent command **load-bearing and co-trained** — a causal
encoder over a *window* of the SNMR latent trajectory, trained inside the PPO loss and acted
through by the actor — close L1's gap to the explicit baseline?

## 2. What is and isn't "co-trained" — the precise mechanism (verified against the pinned clone)

Holosoma architecture audit (2026-07-20, revision `9fb2b57`). **Two things could be "co-trained,"
and they are very different:**

1. **The readout from z → action.** In *every* WBT arm the actor MLP reads the whole `actor_obs`
   (including any latent command) and is trained by PPO (`BaseModule.forward = self.module(...)`,
   `modules.py:405`; update re-runs the actor on stored obs, `ppo.py:556`). So L1 **already**
   co-trains a linear-through-MLP readout of the latent command. `type="MLPEncoder"`
   (`PPOActorEncoder`, `ppo_modules.py:128`) only inserts a **deeper bottleneck** over a named obs
   slice before the trunk (`encoder`, `modules.py:389`; params in `self.actor` → in
   `actor_optimizer`, `ppo.py:278`). This is a **config-only** change (no source edit, no
   monkeypatch) but it does **not** train anything upstream of the NPZ.
2. **The SNMR encoder that produces z.** This is *frozen* in all config-only arms — z is baked
   into the WBT NPZ offline (`export_wbt_with_latent.py`), detached from PPO by construction
   (rollout under `torch.inference_mode()` `ppo.py:392`; `_compute_term` returns `obs.clone()`
   `manager.py:172`; `RolloutStorage.add` rejects grad-carrying tensors `data_utils.py:65`).
   Co-training *this* — the UniTracker/PULSE mechanism — requires the `algo._target_` custom-PPO
   subclass route (scout-confirmed feasible without editing the clone; a multi-day build). **That
   is Stage-3 (§6), NOT this experiment.**

**Consequence — E49 is a narrower, config-only question than "full act-through-latent":** does a
*learned temporal readout* (MLPEncoder) over a **raw latent window** `[z_t, z_{t+0.1s},
z_{t+0.2s}, z_{t+0.5s}]` beat L1's *hand-designed* 3-offset tangent features `[z_t, dz+0.2s,
dz+0.5s]` fed to a plain MLP? This is a GMT-style "learned preview encoding vs fixed preview
features" ablation on the SNMR latent. It is cheap (one config-only arm) and a clean prerequisite
signal for Stage-3, but it does not itself realize the full UniTracker co-trained-encoder design.
Honest framing kept in §7.

## 3. Design (single clip, staged; matched to the E36–E39 protocol)

Clip `walk1_subject5`, robot G1-29dof-wbt, train seed 0, eval seed 404, 1024 envs, 8k iters,
100 phase-stratified 10-s rollouts. Reward + termination stay on the **explicit robot-space GMR
reference** (unchanged) — only the **actor observation** changes. Augmented NPZ
`runs/wbt_latent_gmr/walk1_subject5_mj_z.npz` already carries `latent_z` (validated in Phase 1/2).

**Arms** (all keep the explicit reference OUT of the actor — UniTracker condition — and ON the
reward/termination, unchanged; three factors: {tangent features vs raw window} × {plain MLP vs
MLPEncoder bottleneck}):

| Arm | Actor latent command | Actor arch | Isolates |
|---|---|---|---|
| `baseline` | explicit GMR joint command | MLP | the 88% / 8.95 s reference (reuse E35/B1 GMR policy — do NOT retrain) |
| `l1_frozen` (control) | tangent `[z_t, dz+0.2s, dz+0.5s]` (384-d) | MLP | reuse frozen Phase-1 L1 (72–77%) — the existing feature+plain-readout point |
| `e49_window_mlp` (attribution) | raw window `[z_t, z+0.1s, z+0.2s, z+0.5s]` (512-d) | MLP | raw window vs tangent features, readout held plain |
| `e49_window_enc` (**treatment**) | same raw window (512-d) | **MLPEncoder** (enc 256→64) | learned readout vs plain, window held fixed |

`e49_window_mlp` vs `l1_frozen` isolates the *raw-window-vs-tangent-feature* effect — a clean,
zero-config-risk GMT-style ablation, and **the primary E49 science arm**.

**Caveat on `e49_window_enc`:** the WBT `actor_obs` is a single bundled group (proprio + command
concatenated), and a config-only `MLPEncoder` with `encoder_input_name=actor_obs` would bottleneck
*the entire* observation, not just the latent — so it does **not** cleanly isolate "learned readout
of the latent." Cleanly isolating that needs a *separate* latent obs group (the risky tyro
add-a-group op). Therefore `e49_window_enc` is run primarily as a **Stage-3 wiring-feasibility
probe** (does the MLPEncoder path construct + train at all via config), not as a clean science arm;
its completion number is secondary and interpreted only loosely. The clean encoder-isolation
experiment belongs to Stage-3's custom-PPO build with a dedicated latent group.

**Encoder input.** A new actor-obs group carrying `snmr_latent_window` (raw latent at
`WINDOW_OFFSETS=(0,5,10,25)` 50 Hz steps ≈ current, +0.1 s, +0.2 s, +0.5 s; UniTracker's optimum
was ~5 future frames). Implemented and unit-tested in `snmr/integration/wbt_latent.py`
(`snmr_latent_window`, `tests/test_wbt_latent_integration.py::test_latent_window_gathers_absolute_latents_clipped`).
For `e49_window_enc`: actor `type=MLPEncoder`, `encoder_input_name=<latent group>`,
`encoder_hidden_dims=[256]`, `encoder_output_dim=64`, `module_input_name=(proprio groups)`.

**Config-mechanics risk (de-risk in Stage 0):** the existing L1 arm only *overrode a term's func*;
E49 additionally needs (a) a **new obs group** whose term uses `snmr_latent_window`, and (b) the
actor `input_dim` to name both proprio groups and the latent group with `type=MLPEncoder`. Whether
tyro can add a group + retype the actor purely via `--observation.groups.*` / `--algo.config.*`
overrides is UNPROVEN and is the explicit target of the Stage-0 construction smoke. If it cannot,
the fallback is a tiny SNMR-side config-builder module referenced by `--config` / an `algo._target_`
wrapper — still no edit to the pinned clone.

## 4. GPU plan (careful, staged — per PI instruction)

**Risk-ordered:** `e49_window_mlp` uses the *exact* L1 mechanism (override
`motion-command.func` → `snmr_latent_window`, plain MLP) — **zero config risk, proven**. It is the
primary config-only arm. `e49_window_enc` needs a new obs group + actor retype, which tyro may not
support via CLI — it is **gated on the Stage-0 construction smoke** and dropped (deferred to
Stage-3) if the wiring can't be expressed without editing the clone.

0. **Stage 0 — construction smoke, `e49_window_enc` only (GPU, ~2 min): `num_envs=64`,
   `num-learning-iterations=20`, `save-interval=20`.** Sole purpose: prove the MLPEncoder +
   new-obs-group wiring *constructs and steps* (actor prints an encoder submodule; loss finite;
   checkpoint saves). Monitor `nvidia-smi`. If tyro cannot add the group / retype the actor, **drop
   `e49_window_enc`** from E49 and defer the bottleneck/encoder question to Stage-3 (custom-PPO
   route); proceed with `window_mlp` only. (`window_mlp` needs no such smoke — it is the L1 path.)
2. **Stage 1 — batch/memory calibration (GPU): `num_envs=256` for 200 iters,** record peak GPU MiB
   and steps/s. Only scale `num_envs` to 1024 if peak < ~18 GB (A10G is 23 GB; leave headroom).
   If 1024 OOMs, step down (768/512) — the completion metric is env-count-robust at these scales
   per E33.
3. **Stage 2 — full arm (GPU, ~2.2 h): 8k iters at the calibrated `num_envs`, seed 0,** then the
   standard 100-rollout eval. One GPU job at a time (`nice -n 15`, check `nvidia-smi` + `pgrep`).
   Do NOT launch while E48 100k or any WBT job is running.

## 5. Promotion / stop rules (preregistered)

Primary readout = 10-s completion at 8k, eval seed 404, of **`e49_window_mlp`** (the clean arm).
`e49_window_enc` is secondary (wiring probe; interpreted loosely). Let `best_e49 =
max(e49_window_mlp, e49_window_enc if it ran cleanly)`; the promote decision is driven by
`e49_window_mlp` and only reinforced by `window_enc`.

- **PROMOTE** (to multi-seed {0,1,2} × {404,405}, then multi-clip / Stage-3 encoder co-training)
  iff `best_e49` **≥ l1_frozen + 5 pp AND ≥ 80%**. Report which factor carried it: window
  (window_mlp > l1_frozen) and/or learned readout (window_enc > window_mlp). Secondary: joint-RMSE
  and survival not worse than L1.
- **NULL / STOP config-only line** iff both E49 arms are within ±5 pp of `l1_frozen`: neither a
  raw window nor a learned readout of the *offline* SNMR latent helps — the binding limit is the
  offline/non-causal latent itself. This is an *informative* null that *strengthens* Stage-3 (only
  co-training the encoder under control can help), and closes the cheap config-only line.
- **NEGATIVE** iff `best_e49` < `l1_frozen` − 5 pp: the richer command destabilized tracking;
  report and stop.
- **No hyperparameter fishing.** Encoder width/offsets fixed above; one architecture, one seed for
  each development arm; promote *then* replicate. Any change to encoder config after seeing the
  result reopens the experiment as a new registration.

## 6. Registered Stage-3 (only if E49 promotes or returns an informative null)

The full `docs/FORWARD_RESEARCH_MEMO.md` #1 — a residual-to-prior **CVAE command latent** co-trained
via DAgger distillation of the 86% teacher (UniTracker/PULSE pattern), with the SNMR latent as the
prior mean — requires the `algo._target_` custom-PPO subclass route (scout-confirmed feasible
without editing the clone: add encoder/decoder params to the optimizer, rebuild the loss graph
through them). It is deferred behind E49 because it is a multi-day build and E49 first tests whether
the *co-training mechanism* moves the single-clip number at all with a config-only change. If E49 is
a clean null (co-trained readout of offline z ≈ frozen z), Stage-3's value proposition (a *causal,
control-trained* encoder over raw reference initialized from SNMR) is strengthened, not weakened.

## 7. Scope / honesty

Single-clip walk is, by construction, near-null for latent-command *value* (the explicit command ≈
clip phase; `docs/WBT_LATENT_LITERATURE_REVIEW.md` §4). E49 therefore tests **mechanism**
(does co-training a load-bearing latent readout beat a frozen one), not SNMR-latent
*semantic* value — the latter needs the multi-clip / cross-embodiment setting that single-clip
cannot detect. A promote here is a green light to that larger test, not a paper claim on its own.
