#!/usr/bin/env python
"""Derive scripts/train_e78_masked_fusion.py from the frozen scripts/train_e52_dagger.py.

The E52/E70 harness is hash-frozen; E78 needs reference-channel masking, structured
fusion, and dropout flag bits.  Rather than fork by hand, this script applies a fixed
list of asserted textual replacements so the derived trainer's difference from the
frozen one is fully enumerated here.  Re-run after any intentional change:

    python scripts/derive_e78_trainer.py [--check]
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "train_e52_dagger.py"
DST = ROOT / "scripts" / "train_e78_masked_fusion.py"

REPLACEMENTS: list[tuple[str, str]] = []


def rep(old: str, new: str) -> None:
    REPLACEMENTS.append((old, new))


rep('#!/usr/bin/env python\n"""', '''#!/usr/bin/env python
"""E78 masked-fusion student trainer (derived from the frozen E52/E70 harness).

DERIVED FILE — DO NOT EDIT BY HAND.  Regenerate with ``scripts/derive_e78_trainer.py``.
``scripts/train_e52_dagger.py`` and ``snmr/integration/distillation.py`` are bound in
the E70 confirmation hash manifest and are NOT modified.  This trainer adds, all
switchable by environment variables:

* reference-channel dropout masking during training and evaluation
  (``E78_MASK_*`` / ``E78_EVAL_MASK_*``; see ``snmr/integration/fusion.py``);
* structured explicit/latent fusion (``E78_FUSION`` in {concat, film, gated});
* dropout flag bits appended to the code encoder input (never to the decoder).

With every E78 knob at its default (no masking, ``concat``) the student differs from the
frozen recipe only by two zero-valued flag inputs.  Original harness docstring follows.

''')

rep("from snmr.integration import wbt_bodyfix, wbt_latent\n",
    "from snmr.integration import wbt_bodyfix, wbt_latent\n"
    "from snmr.integration.fusion import (  # noqa: E402\n"
    "    FLAG_DIM,\n    FusionCommandStudent,\n    ReferenceDropoutMasker,\n"
    "    dropout_hazard,\n    ramp,\n)\n")

rep('''    if phase_only and shuffle_latent:
        raise ValueError("time-index and shuffled-latent controls are mutually exclusive")''',
'''    # --- E78 knobs ------------------------------------------------------------------
    fusion = os.environ.get("E78_FUSION", "concat")           # concat | film | gated
    mask_frac = float(os.environ.get("E78_MASK_FRAC", "0"))   # target masked-tick fraction
    mask_seg_min = int(os.environ.get("E78_MASK_SEG_MIN", "5"))    # 0.1 s at 50 Hz
    mask_seg_max = int(os.environ.get("E78_MASK_SEG_MAX", "25"))   # 0.5 s at 50 Hz
    mask_scope = os.environ.get("E78_MASK_SCOPE", "all")      # all | explicit | snmr
    mask_mode = os.environ.get("E78_MASK_MODE", "hold")       # hold | zero
    mask_ramp_rounds = int(os.environ.get("E78_MASK_RAMP_ROUNDS", "300"))
    mask_seed = int(os.environ.get("E78_MASK_SEED", "78"))
    eval_mask_frac = float(os.environ.get("E78_EVAL_MASK_FRAC", "0"))
    eval_mask_seg_min = int(os.environ.get("E78_EVAL_MASK_SEG_MIN", str(mask_seg_min)))
    eval_mask_seg_max = int(os.environ.get("E78_EVAL_MASK_SEG_MAX", str(mask_seg_max)))
    eval_mask_scope = os.environ.get("E78_EVAL_MASK_SCOPE", mask_scope)
    eval_mask_mode = os.environ.get("E78_EVAL_MASK_MODE", mask_mode)
    eval_mask_seed = int(os.environ.get("E78_EVAL_MASK_SEED", "404"))  # paired across arms
    flag_dim = int(os.environ.get("E78_FLAG_DIM", str(FLAG_DIM)))  # 0 = frozen-E70-compatible
    if mask_scope not in {"all", "explicit", "snmr"} or eval_mask_scope not in {"all", "explicit", "snmr"}:
        raise ValueError("mask scope must be all, explicit, or snmr")
    if phase_only and shuffle_latent:
        raise ValueError("time-index and shuffled-latent controls are mutually exclusive")''')

rep('''    student = CommandStudent(
        proprio_dim,
        critic_dim,
        num_act,
        ARM_GOALS[arm],
        MOTION_CMD_DIM,
        z_window_dim=Z_SNMR_DIM * len(Z_OFFSETS),
        z_cmd_dim=Z_CMD_DIM,
    ).to(device)''',
'''    student = FusionCommandStudent(
        proprio_dim,
        critic_dim,
        num_act,
        ARM_GOALS[arm],
        MOTION_CMD_DIM,
        z_window_dim=Z_SNMR_DIM * len(Z_OFFSETS),
        z_cmd_dim=Z_CMD_DIM,
        fusion=fusion,
        flag_dim=flag_dim,
    ).to(device)
    mask_hazard = dropout_hazard(mask_frac, 0.5 * (mask_seg_min + mask_seg_max))
    train_masker = ReferenceDropoutMasker(
        env.num_envs, 0.0, mask_seg_min, mask_seg_max,
        device=device, mode=mask_mode, seed=mask_seed,
    )

    def apply_mask(masker, scope, cmd, zwin):
        """Route the reference-stream dropout to the arm-visible signals in ``scope``."""
        signals = {}
        if scope in ("all", "explicit"):
            signals["cmd"] = cmd
        if scope in ("all", "snmr"):
            signals["zwin"] = zwin
        shown, flags = masker.step(**signals)
        return shown.get("cmd", cmd), shown.get("zwin", zwin), flags[:, :flag_dim]''')

rep('''        "cmd": torch.zeros(n_envs, MOTION_CMD_DIM, device=device),
        "priv": torch.zeros(n_envs, critic_dim, device=device),
    }
    prev_valid''',
'''        "cmd": torch.zeros(n_envs, MOTION_CMD_DIM, device=device),
        "priv": torch.zeros(n_envs, critic_dim, device=device),
        "flags": torch.zeros(n_envs, flag_dim, device=device),
    }
    prev_valid''')

rep('''                "teacher_manifest": teacher_manifest_path or None,
            },''',
'''                "teacher_manifest": teacher_manifest_path or None,
                "e78": {
                    "fusion": fusion, "mask_frac": mask_frac,
                    "mask_seg_ticks": [mask_seg_min, mask_seg_max],
                    "mask_scope": mask_scope, "mask_mode": mask_mode,
                    "mask_ramp_rounds": mask_ramp_rounds, "mask_seed": mask_seed,
                    "flag_dim": flag_dim,
                },
            },''')

rep('''                mu_prior = student.mu_prior(
                    proprio,
                    validation_data["zwin"][start:stop],
                    validation_data["cmd"][start:stop],
                )''',
'''                mu_prior = student.mu_prior(
                    proprio,
                    validation_data["zwin"][start:stop],
                    validation_data["cmd"][start:stop],
                    validation_data["flags"][start:stop],
                )''')

rep('''        fields = (
            "proprio", "zwin", "cmd", "priv", "a_teacher",
            "prev_proprio", "prev_zwin", "prev_cmd", "prev_priv", "prev_valid",
        )''',
'''        fields = (
            "proprio", "zwin", "cmd", "priv", "a_teacher", "flags",
            "prev_proprio", "prev_zwin", "prev_cmd", "prev_priv", "prev_valid", "prev_flags",
        )
        train_masker.set_hazard(mask_hazard * ramp(rnd, mask_ramp_rounds))''')

rep('''                full, proprio, cmd, priv = split_obs(obs_dict)
                zwin = z_window()
                a_teacher = routed_teacher_action(obs_dict["actor_obs"])
                mu_p = student.mu_prior(proprio, zwin, cmd)''',
'''                full, proprio, cmd, priv = split_obs(obs_dict)
                zwin = z_window()
                a_teacher = routed_teacher_action(obs_dict["actor_obs"])
                # Teacher labels use the clean observation; the student sees the masked one.
                cmd, zwin, flags = apply_mask(train_masker, mask_scope, cmd, zwin)
                mu_p = student.mu_prior(proprio, zwin, cmd, flags)''')

rep('''                for k, v in (("proprio", proprio), ("zwin", zwin), ("cmd", cmd),
                             ("priv", priv), ("a_teacher", a_teacher),
                             ("prev_proprio", previous["proprio"]),
                             ("prev_zwin", previous["zwin"]),
                             ("prev_cmd", previous["cmd"]),
                             ("prev_priv", previous["priv"]),
                             ("prev_valid", prev_valid)):
                    buf[k].append(v.clone())
                previous = {
                    "proprio": proprio.clone(),
                    "zwin": zwin.clone(),
                    "cmd": cmd.clone(),
                    "priv": priv.clone(),
                }''',
'''                for k, v in (("proprio", proprio), ("zwin", zwin), ("cmd", cmd),
                             ("priv", priv), ("a_teacher", a_teacher), ("flags", flags),
                             ("prev_proprio", previous["proprio"]),
                             ("prev_zwin", previous["zwin"]),
                             ("prev_cmd", previous["cmd"]),
                             ("prev_priv", previous["priv"]),
                             ("prev_valid", prev_valid),
                             ("prev_flags", previous["flags"])):
                    buf[k].append(v.clone())
                previous = {
                    "proprio": proprio.clone(),
                    "zwin": zwin.clone(),
                    "cmd": cmd.clone(),
                    "priv": priv.clone(),
                    "flags": flags.clone(),
                }''')

rep('''                if len(done_idx):
                    eps[done_idx] = torch.randn(len(done_idx), Z_CMD_DIM, device=device)
                    prev_valid[done_idx] = 0.0''',
'''                if len(done_idx):
                    eps[done_idx] = torch.randn(len(done_idx), Z_CMD_DIM, device=device)
                    prev_valid[done_idx] = 0.0
                    train_masker.reset(done_idx)''')

rep('''                for key in ("proprio", "zwin", "cmd", "a_teacher")''',
    '''                for key in ("proprio", "zwin", "cmd", "a_teacher", "flags")''')

rep('''                proprio, zwin, cmd, priv = (data["proprio"][idx], data["zwin"][idx],
                                            data["cmd"][idx], data["priv"][idx])
                mu_p = student.mu_prior(proprio, zwin, cmd)''',
'''                proprio, zwin, cmd, priv = (data["proprio"][idx], data["zwin"][idx],
                                            data["cmd"][idx], data["priv"][idx])
                flags = data["flags"][idx]
                mu_p = student.mu_prior(proprio, zwin, cmd, flags)''')

rep('''                    mu_res = student.mu_residual(proprio, zwin, cmd, priv)''',
    '''                    mu_res = student.mu_residual(proprio, zwin, cmd, priv, flags)''')

rep('''                    prev_mu_p = student.mu_prior(
                        data["prev_proprio"][idx],
                        data["prev_zwin"][idx],
                        data["prev_cmd"][idx],
                    )
                    prev_mu_q = prev_mu_p + student.mu_residual(
                        data["prev_proprio"][idx],
                        data["prev_zwin"][idx],
                        data["prev_cmd"][idx],
                        data["prev_priv"][idx],
                    )''',
'''                    prev_mu_p = student.mu_prior(
                        data["prev_proprio"][idx],
                        data["prev_zwin"][idx],
                        data["prev_cmd"][idx],
                        data["prev_flags"][idx],
                    )
                    prev_mu_q = prev_mu_p + student.mu_residual(
                        data["prev_proprio"][idx],
                        data["prev_zwin"][idx],
                        data["prev_cmd"][idx],
                        data["prev_priv"][idx],
                        data["prev_flags"][idx],
                    )''')

rep('''                "best_round": best_round,
            }
            print(json.dumps(rec), flush=True)''',
'''                "best_round": best_round,
                "mask_hazard": train_masker.hazard,
                "masked_fraction_cum": train_masker.masked_fraction,
                "gate": student.gate,
            }
            print(json.dumps(rec), flush=True)''')

rep('''    hold_z = int(os.environ.get("E52_EVAL_HOLD_Z", "0"))  # E65: refresh z_cmd only every
    # k control ticks (zero-order hold) — latency/dropout robustness readout. k=1 = off.''',
'''    hold_z = int(os.environ.get("E52_EVAL_HOLD_Z", "0"))  # E65: refresh z_cmd only every
    # k control ticks (zero-order hold) — latency/dropout robustness readout. k=1 = off.
    eval_masker = ReferenceDropoutMasker(
        n_envs,
        dropout_hazard(eval_mask_frac, 0.5 * (eval_mask_seg_min + eval_mask_seg_max)),
        eval_mask_seg_min, eval_mask_seg_max,
        device=device, mode=eval_mask_mode, seed=eval_mask_seed,
    )
    masked_ticks_by_env = torch.zeros(n_envs, device=device)''')

rep('''            if hold_z > 1 and step % hold_z != 0:
                z_cmd = held_z  # zero-order hold between refreshes
            else:
                z_cmd = student.mu_prior(proprio, zwin, cmd)
                if eval_z == "posterior":
                    z_cmd = z_cmd + student.mu_residual(proprio, zwin, cmd, priv)
                held_z = z_cmd''',
'''            cmd, zwin, flags = apply_mask(eval_masker, eval_mask_scope, cmd, zwin)
            masked_ticks_by_env += eval_masker.last_masked.float() * active.float()
            if hold_z > 1 and step % hold_z != 0:
                z_cmd = held_z  # zero-order hold between refreshes
            else:
                z_cmd = student.mu_prior(proprio, zwin, cmd, flags)
                if eval_z == "posterior":
                    z_cmd = z_cmd + student.mu_residual(proprio, zwin, cmd, priv, flags)
                held_z = z_cmd''')

rep('''        "rounds": rounds, "beta_kl": beta_kl, "prior_mix": prior_mix,
        "deterministic": deterministic,
    }''',
'''        "rounds": rounds, "beta_kl": beta_kl, "prior_mix": prior_mix,
        "deterministic": deterministic,
        "e78": {
            "fusion": fusion, "gate": student.gate,
            "train_mask": {"frac": mask_frac, "seg_ticks": [mask_seg_min, mask_seg_max],
                           "scope": mask_scope, "mode": mask_mode,
                           "ramp_rounds": mask_ramp_rounds, "seed": mask_seed},
            "eval_mask": {"frac": eval_mask_frac,
                          "seg_ticks": [eval_mask_seg_min, eval_mask_seg_max],
                          "scope": eval_mask_scope, "mode": eval_mask_mode,
                          "seed": eval_mask_seed,
                          "realized_masked_fraction": eval_masker.masked_fraction},
            "masked_fraction_by_rollout": (
                masked_ticks_by_env / survival.clamp_min(1)
            ).cpu().tolist(),
        },
    }''')

rep('''    if destroy_zcmd != "none":
        suffix_parts.append(f"destroy_{destroy_zcmd}")''',
'''    if destroy_zcmd != "none":
        suffix_parts.append(f"destroy_{destroy_zcmd}")
    if eval_mask_frac > 0:
        suffix_parts.append(
            f"mask{eval_mask_scope}_{eval_mask_mode}_f{eval_mask_frac:g}"
            f"_s{eval_mask_seg_min}-{eval_mask_seg_max}"
        )''')


def derive() -> str:
    text = SRC.read_text()
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"expected exactly one match, found {count}: {old[:70]!r}")
        text = text.replace(old, new)
    return text


def main() -> None:
    derived = derive()
    if "--check" in sys.argv:
        if not DST.exists() or DST.read_text() != derived:
            raise SystemExit(f"{DST} is stale; re-run {pathlib.Path(__file__).name}")
        print("up to date")
        return
    DST.write_text(derived)
    print(f"wrote {DST} ({len(REPLACEMENTS)} replacements)")


if __name__ == "__main__":
    main()
