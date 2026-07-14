import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_gate1_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("analyze_gate1_diagnostics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate1)


def _write_run(tmp_path, name, terms, *, revision="abc123", completed=True):
    run = tmp_path / name
    run.mkdir()
    rows = []
    for index in range(10):
        loss_terms = {
            "distill": {
                "raw": 1.0,
                "weight": 1.0,
                "weighted": 1.0,
                "gradient_norm": {"shared_trunk": 2.0},
            }
        }
        cosines = {}
        for term, (weight, ratio) in terms.items():
            loss_terms[term] = {
                "raw": 1.0,
                "weight": weight,
                "weighted": weight,
                "gradient_norm": {"shared_trunk": 2.0 * ratio},
            }
            cosines[f"distill|{term}"] = 0.25
        rows.append({
            "step": (index + 1) * 500,
            "loss_terms": loss_terms,
            "gradient_cosine": {"shared_trunk": cosines},
            "contact_labels": {"samples": 20, "prevalence": 0.2},
        })
    (run / "diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    manifest = {
        "status": "completed" if completed else "running",
        "git": {"sha": revision, "dirty": False},
        "training": {"planned_optimizer_steps": 5000},
        "progress": {
            "step": 5000 if completed else 4500,
            "completion_state": "completed" if completed else "running",
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_gate1_diagnostic_contract_passes_control_and_factorized_terms(tmp_path):
    _write_run(tmp_path, "c0_seed0", {})
    _write_run(tmp_path, "c1_bce_seed0", {"contact_bce": (1.0, 0.1)})
    _write_run(tmp_path, "c2_edge_seed0", {
        "contact_bce": (1.0, 0.1),
        "edge_velocity": (0.03, 0.3),
    })
    _write_run(tmp_path, "c3_stance_seed0", {
        "teacher_stance_velocity": (0.03, 0.3),
    })
    _write_run(tmp_path, "c4_teacher_velocity_seed0", {
        "teacher_velocity": (0.05, 0.3),
    })

    report = gate1.analyze_root(tmp_path)

    assert report["passed"]
    assert report["revision"] == "abc123"
    edge = next(arm for arm in report["arms"] if arm["arm"] == "c2_edge_seed0")[
        "terms"
    ]["edge_velocity"]
    assert edge["median_ratio"] == 0.3
    assert edge["p90_ratio"] == 0.3
    assert edge["suggested_recalibrated_weight"] is None


def test_gate1_diagnostic_contract_suggests_one_capped_recalibration(tmp_path):
    run = _write_run(tmp_path, "c3_stance_seed0", {
        "teacher_stance_velocity": (0.03, 0.01),
    })

    result = gate1.analyze_run(run)

    assert not result["passed"]
    term = result["terms"]["teacher_stance_velocity"]
    assert term["suggested_recalibrated_weight"] == 0.12
    assert "median ratio" in term["errors"][0]


def test_gate1_diagnostic_contract_rejects_incomplete_and_mixed_revision_runs(tmp_path):
    _write_run(tmp_path, "c0_seed0", {}, completed=False)
    _write_run(tmp_path, "c1_bce_seed0", {"contact_bce": (1.0, 0.1)}, revision="different")

    report = gate1.analyze_root(tmp_path)

    assert not report["passed"]
    assert report["revision"] is None
    assert report["cross_arm_errors"]
    assert "manifest status" in report["arms"][0]["errors"][0]


def test_gate1_diagnostic_contract_rejects_missing_expected_objective(tmp_path):
    run = _write_run(tmp_path, "c1_bce_seed0", {})

    result = gate1.analyze_run(run)

    assert not result["passed"]
    assert "active factorized terms" in result["errors"][0]
