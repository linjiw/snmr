from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.render_e70_destruction_values import (
    DESTROY_MODES,
    SEEDS,
    SNMR,
    collect_reports,
    latex_macros,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_e70_destruction_values.py"


def _report(destroy: str, completion: float, survival: float) -> dict:
    return {
        "arm": "c_prior_explicit",
        "eval_z": "prior",
        "destroy_zcmd": destroy,
        "noise_cmd": 0.0,
        "noise_zret": 0.0,
        "noise_proprio": 0.0,
        "num_rollouts": 1024,
        "evaluation_seed": 404,
        "completion_rate": completion,
        "mean_survival_s": survival,
    }


def _students(tmp_path: Path) -> Path:
    root = tmp_path / "students"
    baselines = {0: 0.9248046875, 1: 0.919921875, 2: 0.923828125}
    survivals = {"zero": 0.85, "shuffle": 0.67, "marginal_random": 0.95}
    for seed in SEEDS:
        directory = root / f"seed{seed}_explicit"
        directory.mkdir(parents=True)
        (directory / "c_prior_explicit_eval.json").write_text(
            json.dumps(_report("none", baselines[seed], 9.5 + seed / 100.0))
        )
        for mode in DESTROY_MODES:
            (directory / f"c_prior_explicit_eval_destroy_{mode}.json").write_text(
                json.dumps(_report(mode, 0.0, survivals[mode] + seed / 1000.0))
            )
    return root


def test_macros_bind_all_seed_destruction_and_stamp_every_input(tmp_path: Path) -> None:
    reports, digests = collect_reports(_students(tmp_path))
    assert len(reports) == 12 and len(digests) == 12
    rendered = latex_macros(reports, input_digests=digests)
    assert r"\newcommand{\EDestroySeeds}{3}" in rendered
    assert r"\newcommand{\EDestroyModes}{3}" in rendered
    assert r"\newcommand{\EDestroyBaselineCompletion}{0.923}" in rendered
    assert r"\newcommand{\EDestroyCompletion}{0.000}" in rendered
    assert r"\newcommand{\EDestroyMaxSurvival}{0.952}" in rendered
    for name, digest in digests:
        assert f"% input_sha256[{name}]={digest}" in rendered
    assert "% inputs_sha256=" in rendered
    assert "% trace[c_prior_explicit] seed0/marginal_random: completion_rate=0" in rendered


def test_missing_input_fails_closed(tmp_path: Path) -> None:
    root = _students(tmp_path)
    (root / "seed2_explicit" / "c_prior_explicit_eval_destroy_shuffle.json").unlink()
    with pytest.raises(ValueError, match="missing its input"):
        collect_reports(root)


def test_mutated_completion_fails_closed(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "seed1_explicit" / "c_prior_explicit_eval_destroy_zero.json"
    mutated = json.loads(path.read_text())
    mutated["completion_rate"] = 0.01
    path.write_text(json.dumps(mutated))
    reports, digests = collect_reports(root)
    with pytest.raises(ValueError, match="exactly zero"):
        latex_macros(reports, input_digests=digests)


def test_mutated_input_changes_the_stamped_hash(tmp_path: Path) -> None:
    root = _students(tmp_path)
    before = dict(collect_reports(root)[1])
    path = root / "seed0_explicit" / "c_prior_explicit_eval.json"
    path.write_text(path.read_text() + "\n")
    after = dict(collect_reports(root)[1])
    key = "seed0_explicit/c_prior_explicit_eval.json"
    assert before[key] != after[key]


def test_wrong_destroy_mode_label_fails_closed(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "seed0_explicit" / "c_prior_explicit_eval_destroy_shuffle.json"
    mutated = json.loads(path.read_text())
    mutated["destroy_zcmd"] = "zero"
    path.write_text(json.dumps(mutated))
    reports, digests = collect_reports(root)
    with pytest.raises(ValueError, match="wrong destroy_zcmd"):
        latex_macros(reports, input_digests=digests)


def test_wrong_arm_or_evaluation_grid_fails_closed(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "seed1_explicit" / "c_prior_explicit_eval.json"
    mutated = json.loads(path.read_text())
    mutated["evaluation_seed"] = 405
    path.write_text(json.dumps(mutated))
    reports, digests = collect_reports(root)
    with pytest.raises(ValueError, match="frozen evaluation grid"):
        latex_macros(reports, input_digests=digests)


def test_noisy_evaluation_is_rejected(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "seed2_explicit" / "c_prior_explicit_eval_destroy_marginal_random.json"
    mutated = json.loads(path.read_text())
    mutated["noise_cmd"] = 0.05
    path.write_text(json.dumps(mutated))
    reports, digests = collect_reports(root)
    with pytest.raises(ValueError, match="noise-free"):
        latex_macros(reports, input_digests=digests)


def test_incomplete_seed_grid_fails_closed(tmp_path: Path) -> None:
    reports, digests = collect_reports(_students(tmp_path))
    reports.pop("seed2/shuffle")
    with pytest.raises(ValueError, match="missing the seed2/shuffle report"):
        latex_macros(reports, input_digests=digests)


def test_unstamped_inputs_fail_closed(tmp_path: Path) -> None:
    reports, _ = collect_reports(_students(tmp_path))
    with pytest.raises(ValueError, match="hash-stamped inputs"):
        latex_macros(reports, input_digests=[])


def test_intact_baseline_must_survive(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "seed0_explicit" / "c_prior_explicit_eval.json"
    mutated = json.loads(path.read_text())
    mutated["mean_survival_s"] = 2.0
    path.write_text(json.dumps(mutated))
    reports, digests = collect_reports(root)
    with pytest.raises(ValueError, match="must survive most of the episode"):
        latex_macros(reports, input_digests=digests)


def test_direct_entrypoint_help_uses_repository_scripts_package(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--students-root" in completed.stdout


# --- E75: the same battery on the SNMR arm ---------------------------------------------------
# The SNMR arm's intact baselines are frozen E70 artifacts while its destruction reports are E75
# outputs under a separate root, because nothing may be written under the frozen E70 tree.


def _snmr_report(destroy: str, completion: float, survival: float) -> dict:
    report = _report(destroy, completion, survival)
    report["arm"] = "a_prior_snmr"
    return report


def _snmr_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Return (baseline_root, destroy_root) with the SNMR reports split across two trees."""
    baseline_root = tmp_path / "e70_students"
    destroy_root = tmp_path / "e75_students"
    baselines = {0: 0.684570, 1: 0.702148, 2: 0.708984}
    survivals = {"zero": 0.62, "shuffle": 0.59, "marginal_random": 0.84}
    for seed in SEEDS:
        base_dir = baseline_root / f"seed{seed}_snmr"
        base_dir.mkdir(parents=True)
        (base_dir / "a_prior_snmr_eval.json").write_text(
            json.dumps(_snmr_report("none", baselines[seed], 8.27 + seed / 100.0))
        )
        destroy_dir = destroy_root / f"seed{seed}_snmr"
        destroy_dir.mkdir(parents=True)
        for mode in DESTROY_MODES:
            (destroy_dir / f"a_prior_snmr_eval_destroy_{mode}.json").write_text(
                json.dumps(_snmr_report(mode, 0.0, survivals[mode] + seed / 1000.0))
            )
    return baseline_root, destroy_root


def _explicit(tmp_path: Path):
    """Explicit-arm reports and digests, so two-arm renders have a valid first arm."""
    return collect_reports(_students(tmp_path))


def test_snmr_arm_reads_baselines_and_destruction_from_separate_roots(tmp_path: Path) -> None:
    baseline_root, destroy_root = _snmr_roots(tmp_path)
    reports, digests = collect_reports(baseline_root, SNMR, destroy_root=destroy_root)
    _ex_reports, _ex_digests = _explicit(tmp_path)
    assert len(digests) == len(SEEDS) * (1 + len(DESTROY_MODES))
    rendered = latex_macros(
        _ex_reports, input_digests=_ex_digests,
        snmr_reports=reports, snmr_digests=digests,
    )
    assert r"\newcommand{\EDestroySnmrBaselineCompletion}{0.699}" in rendered
    assert r"\newcommand{\EDestroySnmrCompletion}{0.000}" in rendered
    assert r"\newcommand{\EDestroySnmrMaxSurvival}{0.842}" in rendered
    assert r"\newcommand{\EDestroyArms}{2}" in rendered
    # The explicit macros must survive unchanged alongside the new arm.
    assert r"\newcommand{\EDestroyBaselineCompletion}{0.923}" in rendered
    assert r"\newcommand{\EDestroyCompletion}{0.000}" in rendered


def test_snmr_traces_are_labelled_by_arm(tmp_path: Path) -> None:
    baseline_root, destroy_root = _snmr_roots(tmp_path)
    reports, digests = collect_reports(baseline_root, SNMR, destroy_root=destroy_root)
    _ex_reports, _ex_digests = _explicit(tmp_path)
    rendered = latex_macros(
        _ex_reports, input_digests=_ex_digests,
        snmr_reports=reports, snmr_digests=digests,
    )
    assert "% trace[a_prior_snmr] seed0/zero: completion_rate=0" in rendered
    assert "% trace[c_prior_explicit] seed0/zero: completion_rate=0" in rendered


def test_explicit_only_render_omits_the_snmr_macros(tmp_path: Path) -> None:
    reports, digests = collect_reports(_students(tmp_path))
    rendered = latex_macros(reports, input_digests=digests)
    assert "EDestroySnmr" not in rendered
    assert r"\newcommand{\EDestroyArms}{1}" in rendered


def test_snmr_report_mislabelled_as_explicit_fails_closed(tmp_path: Path) -> None:
    baseline_root, destroy_root = _snmr_roots(tmp_path)
    stray = destroy_root / "seed1_snmr" / "a_prior_snmr_eval_destroy_zero.json"
    payload = json.loads(stray.read_text())
    payload["arm"] = "c_prior_explicit"
    stray.write_text(json.dumps(payload))
    reports, digests = collect_reports(baseline_root, SNMR, destroy_root=destroy_root)
    _ex_reports, _ex_digests = _explicit(tmp_path)
    with pytest.raises(ValueError, match="a_prior_snmr"):
        latex_macros(
            _ex_reports, input_digests=_ex_digests,
            snmr_reports=reports, snmr_digests=digests,
        )


def test_snmr_battery_without_digests_fails_closed(tmp_path: Path) -> None:
    baseline_root, destroy_root = _snmr_roots(tmp_path)
    reports, _ = collect_reports(baseline_root, SNMR, destroy_root=destroy_root)
    _ex_reports, _ex_digests = _explicit(tmp_path)
    with pytest.raises(ValueError, match="hash-stamped"):
        latex_macros(
            _ex_reports, input_digests=_ex_digests,
            snmr_reports=reports, snmr_digests=[],
        )


def test_a_surviving_snmr_rollout_fails_closed(tmp_path: Path) -> None:
    """A non-zero destroyed cell must never be rendered as a collapse."""
    baseline_root, destroy_root = _snmr_roots(tmp_path)
    cell = destroy_root / "seed2_snmr" / "a_prior_snmr_eval_destroy_shuffle.json"
    payload = json.loads(cell.read_text())
    payload["completion_rate"] = 0.12
    cell.write_text(json.dumps(payload))
    reports, digests = collect_reports(baseline_root, SNMR, destroy_root=destroy_root)
    _ex_reports, _ex_digests = _explicit(tmp_path)
    with pytest.raises(ValueError, match="exactly zero"):
        latex_macros(
            _ex_reports, input_digests=_ex_digests,
            snmr_reports=reports, snmr_digests=digests,
        )
