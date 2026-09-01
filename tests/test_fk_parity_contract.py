"""Unit tests for the simulator-independent G0 FK parity contract."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from snmr.fk_parity import (
    DIRECTORY_BUNDLE_HASH_SCHEMA_VERSION,
    URDF_BUNDLE_HASH_SCHEMA_VERSION,
    capture_directory_bundle,
    capture_urdf_bundle,
    compare_fk_poses,
    evaluate_fixed_link_frames,
    float64_buffer_sha256,
    joint_sample_payload_sha256,
    quaternion_geodesic_xyzw,
    resolve_fixed_link_frames,
    sample_uniform_joint_positions,
)


def _load_physx_worker_module():
    script = Path(__file__).resolve().parents[1] / "scripts/g0_fk_parity_physx.py"
    spec = importlib.util.spec_from_file_location("snmr_test_g0_fk_parity_physx", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uniform_sampling_is_deterministic_bounded_and_float64() -> None:
    lower = np.array([-1.0, 0.25, -0.2], dtype=np.float64)
    upper = np.array([2.0, 0.5, 0.8], dtype=np.float64)
    first = sample_uniform_joint_positions(lower, upper, num_samples=128, seed=17)
    second = sample_uniform_joint_positions(lower, upper, num_samples=128, seed=17)
    different = sample_uniform_joint_positions(lower, upper, num_samples=128, seed=18)

    assert first.dtype == np.float64
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.all(first >= lower[None])
    assert np.all(first <= upper[None])


@pytest.mark.parametrize(
    "lower,upper",
    [
        ([-1.0], [-1.0]),
        ([0.0, 1.0], [1.0]),
        ([float("nan")], [1.0]),
        ([-1.0], [float("inf")]),
    ],
)
def test_uniform_sampling_rejects_invalid_limits(lower: list[float], upper: list[float]) -> None:
    with pytest.raises(ValueError):
        sample_uniform_joint_positions(
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )


def test_sample_hash_binds_exact_buffer_and_joint_meaning() -> None:
    names = ("hip", "knee")
    lower = np.ascontiguousarray([-1.0, 0.0], dtype=np.float64)
    upper = np.ascontiguousarray([1.0, 2.0], dtype=np.float64)
    samples = sample_uniform_joint_positions(lower, upper, num_samples=4, seed=3)

    expected_raw = hashlib.sha256(memoryview(samples).cast("B")).hexdigest()
    assert float64_buffer_sha256(samples) == expected_raw
    original_payload = joint_sample_payload_sha256(names, lower, upper, samples)
    changed = samples.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    assert float64_buffer_sha256(changed) != expected_raw
    assert joint_sample_payload_sha256(names, lower, upper, changed) != original_payload
    assert joint_sample_payload_sha256(tuple(reversed(names)), lower, upper, samples) != original_payload

    with pytest.raises(TypeError):
        float64_buffer_sha256(samples.astype(np.float32))
    with pytest.raises(ValueError, match="C contiguous"):
        float64_buffer_sha256(samples[:, ::-1])


def _rotation_x_xyzw(angle: float) -> np.ndarray:
    return np.array([np.sin(angle / 2.0), 0.0, 0.0, np.cos(angle / 2.0)])


def test_quaternion_geodesic_is_sign_invariant() -> None:
    identity = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert quaternion_geodesic_xyzw(identity, -identity)[0] == pytest.approx(0.0)
    angle = 0.25
    observed = quaternion_geodesic_xyzw(identity, _rotation_x_xyzw(angle)[None])[0]
    assert observed == pytest.approx(angle, abs=1.0e-12)


def test_compare_fk_poses_uses_strict_maximum_gate_and_reports_worst_pair() -> None:
    reference_position = np.zeros((3, 2, 3), dtype=np.float64)
    candidate_position = reference_position.copy()
    reference_quaternion = np.zeros((3, 2, 4), dtype=np.float64)
    reference_quaternion[..., 3] = 1.0
    candidate_quaternion = reference_quaternion.copy()
    candidate_position[2, 1, 0] = 9.0e-4
    candidate_quaternion[1, 0] = _rotation_x_xyzw(9.0e-4)

    passing = compare_fk_poses(
        reference_position,
        reference_quaternion,
        candidate_position,
        candidate_quaternion,
    )
    assert passing.g0_pass
    assert (passing.worst_position_sample, passing.worst_position_link) == (2, 1)
    assert (passing.worst_orientation_sample, passing.worst_orientation_link) == (1, 0)
    manifest = passing.to_manifest(("root", "foot"))
    assert manifest["worst_position_link_name"] == "foot"
    assert manifest["threshold_semantics"] == "strict_less_than"

    candidate_position[2, 1, 0] = 1.0e-3
    equal_to_threshold = compare_fk_poses(
        reference_position,
        reference_quaternion,
        candidate_position,
        candidate_quaternion,
    )
    assert not equal_to_threshold.position_pass
    assert not equal_to_threshold.g0_pass


def test_compare_fk_poses_rejects_zero_quaternion_and_nonfinite_position() -> None:
    position = np.zeros((1, 1, 3), dtype=np.float64)
    identity = np.array([[[0.0, 0.0, 0.0, 1.0]]])
    with pytest.raises(ValueError, match="zero-norm"):
        compare_fk_poses(position, identity, position, np.zeros_like(identity))
    nonfinite = position.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compare_fk_poses(position, identity, nonfinite, identity)


def test_fixed_link_resolution_and_evaluation_for_collapsed_runtime_body() -> None:
    urdf = b"""
    <robot name="synthetic">
      <link name="base"/><link name="mount"/><link name="tip"/>
      <joint name="base_mount" type="fixed">
        <parent link="base"/><child link="mount"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
      </joint>
      <joint name="mount_tip" type="fixed">
        <parent link="mount"/><child link="tip"/>
        <origin xyz="1 0 0" rpy="0 0 0"/>
      </joint>
    </robot>
    """
    frames = resolve_fixed_link_frames(urdf, (("endpoint", "tip"),), ("base",))
    assert frames[0].runtime_body == "base"
    assert frames[0].local_position == pytest.approx((1.0, 1.0, 0.0), abs=1.0e-12)
    body_position = np.array([[[5.0, 0.0, 0.0]]])
    body_quaternion = np.array([[[0.0, 0.0, 0.0, 1.0]]])
    key_position, key_quaternion = evaluate_fixed_link_frames(
        body_position, body_quaternion, ("base",), frames
    )
    assert key_position[0, 0] == pytest.approx((6.0, 1.0, 0.0), abs=1.0e-12)
    assert quaternion_geodesic_xyzw(
        key_quaternion[0, 0][None], _rotation_x_xyzw(0.0)[None]
    )[0] == pytest.approx(np.pi / 2.0)


def test_fixed_link_resolution_refuses_to_cross_moving_joint() -> None:
    urdf = b"""
    <robot name="synthetic">
      <link name="base"/><link name="tip"/>
      <joint name="moving" type="revolute">
        <parent link="base"/><child link="tip"/><origin xyz="0 0 1" rpy="0 0 0"/>
      </joint>
    </robot>
    """
    with pytest.raises(ValueError, match="non-fixed"):
        resolve_fixed_link_frames(urdf, (("endpoint", "tip"),), ("base",))


def test_directory_bundle_v02_hash_binds_selected_entrypoint(tmp_path: Path) -> None:
    bundle_root = tmp_path / "usd"
    bundle_root.mkdir()
    (bundle_root / "first.usd").write_bytes(b"first")
    (bundle_root / "second.usd").write_bytes(b"second")
    (bundle_root / "payload.bin").write_bytes(b"payload")

    first, first_members = capture_directory_bundle(bundle_root, entrypoint="first.usd")
    second, second_members = capture_directory_bundle(bundle_root, entrypoint="second.usd")

    assert first["hash_schema_version"] == DIRECTORY_BUNDLE_HASH_SCHEMA_VERSION
    assert first["hash_schema_version"] == "snmr.directory-bundle.v0.2"
    assert first["hash_algorithm"] == "sha256"
    assert first["entrypoint_bound"] is True
    assert first_members == second_members
    assert first["sha256"] != second["sha256"]

    # The pre-hardening v0.1 algorithm hashed the same member set without binding
    # which member was the entrypoint.  Both choices therefore had one legacy hash.
    legacy = hashlib.sha256(b"snmr.directory-bundle.v0.1\0")
    for relative, data in first_members:
        encoded = relative.encode("utf-8")
        legacy.update(len(encoded).to_bytes(8, "big") + encoded)
        legacy.update(len(data).to_bytes(8, "big") + data)
    assert first["sha256"] != legacy.hexdigest()
    assert second["sha256"] != legacy.hexdigest()


def test_urdf_bundle_v02_manifest_declares_entrypoint_bound_hash(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "link.stl").write_bytes(b"mesh")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="synthetic"><link name="base"><visual><geometry>'
        '<mesh filename="meshes/link.stl"/></geometry></visual></link></robot>',
        encoding="utf-8",
    )

    manifest, captured_urdf = capture_urdf_bundle(urdf)

    assert captured_urdf == urdf.read_bytes()
    assert manifest["hash_schema_version"] == URDF_BUNDLE_HASH_SCHEMA_VERSION
    assert manifest["hash_schema_version"] == "snmr.urdf-bundle.v0.2"
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["entrypoint_bound"] is True
    assert manifest["entrypoint"] == "robot.urdf"
    assert [item["relative_path"] for item in manifest["files"]] == [
        "meshes/link.stl",
        "robot.urdf",
    ]


def test_physx_cleanup_releases_callbacks_and_singleton_without_timeline_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_physx_worker_module()
    calls: list[str] = []

    class FakeSimulationContext:
        def stop(self) -> None:
            raise AssertionError("headless cleanup must not synchronously stop the timeline")

        def clear(self) -> None:
            raise AssertionError("stage close is owned by SimulationApp, not sim.clear")

        def clear_all_callbacks(self) -> None:
            calls.append("clear_all_callbacks")

        def clear_instance(self) -> None:
            calls.append("clear_instance")

    monkeypatch.setattr(worker, "_progress", lambda *_args, **_kwargs: None)
    worker._cleanup_simulation_context(FakeSimulationContext(), 0.0)

    assert calls == ["clear_all_callbacks", "clear_instance"]
