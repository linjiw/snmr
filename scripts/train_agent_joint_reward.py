#!/usr/bin/env python
"""E51: holosoma train_agent with joint-space tracking rewards added (no clone edits).

tyro CLI overrides cannot ADD dict keys, so this wrapper parses the experiment config
exactly like holosoma's main() and then rebuilds the reward manager config with two extra
terms from ``snmr.integration.wbt_rewards`` before calling train(). Weights/sigmas come from
env vars so the driver stays declarative:

  E51_JOINT_POS_WEIGHT (default 1.0)   E51_JOINT_POS_SIGMA (default 0.5)
  E51_JOINT_VEL_WEIGHT (default 0.0)   E51_JOINT_VEL_SIGMA (default 5.0)

Run inside .venv-wbt with PYTHONPATH including the snmr repo root.
"""

import dataclasses
import os

import tyro

from snmr.integration import wbt_bodyfix

wbt_bodyfix.patch()

from holosoma.config_types.reward import RewardTermCfg  # noqa: E402
from holosoma.train_agent import AnnotatedExperimentConfig, train  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)

    pos_weight = float(os.environ.get("E51_JOINT_POS_WEIGHT", "1.0"))
    vel_weight = float(os.environ.get("E51_JOINT_VEL_WEIGHT", "0.0"))
    terms = dict(tyro_cfg.reward.terms)
    if pos_weight != 0.0:
        terms["motion_joint_pos_error_exp"] = RewardTermCfg(
            func="snmr.integration.wbt_rewards:motion_joint_pos_error_exp",
            params={"sigma": float(os.environ.get("E51_JOINT_POS_SIGMA", "0.5"))},
            weight=pos_weight,
        )
    if vel_weight != 0.0:
        terms["motion_joint_vel_error_exp"] = RewardTermCfg(
            func="snmr.integration.wbt_rewards:motion_joint_vel_error_exp",
            params={"sigma": float(os.environ.get("E51_JOINT_VEL_SIGMA", "5.0"))},
            weight=vel_weight,
        )
    reward = dataclasses.replace(tyro_cfg.reward, terms=terms)
    tyro_cfg = dataclasses.replace(tyro_cfg, reward=reward)
    print(
        f"E51 reward injection: joint_pos w={pos_weight} "
        f"sigma={terms.get('motion_joint_pos_error_exp') and terms['motion_joint_pos_error_exp'].params['sigma']}, "
        f"joint_vel w={vel_weight}"
    )
    train(tyro_cfg)


if __name__ == "__main__":
    main()
