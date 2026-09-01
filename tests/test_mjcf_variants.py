import json
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from snmr.mjcf_variants import (
    GeneratorProvenance,
    build_semantic_limb_variant_plan,
    materialize_mjcf_variant,
    require_archival_ready_variant_manifest,
    write_materialized_mjcf_variant,
    zero_perturbation_plan,
)
from snmr.provenance import MjcfBundleSnapshot
from snmr.robot_spec import RobotSpec, SemanticManifest


def _semantics() -> SemanticManifest:
    return SemanticManifest(
        root_link="pelvis",
        left_hand_link="left_hand",
        right_hand_link="right_hand",
        left_foot_link="left_foot",
        right_foot_link="right_foot",
        allowed_contact_links=("left_foot", "right_foot"),
    )


def _tiny_mjcf(path: Path) -> Path:
    path.write_text(
        """<mujoco model="tiny">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.2 0.2 0.2"/>
      <geom name="pelvis_geom" type="box" pos="0 0 0" size="0.1 0.1 0.1"/>
      <body name="left_leg" pos="0 0.1 -0.2">
        <inertial pos="0 0 -0.1" mass="1" diaginertia="0.02 0.02 0.01"/>
        <joint name="left_hip" axis="0 1 0" range="-1 1" ref="0.2"/>
        <geom name="left_leg_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03"/>
        <body name="left_foot" pos="0 0 -0.25">
          <inertial pos="0.03 0 0" mass="0.5" diaginertia="0.01 0.008 0.004"/>
          <joint name="left_ankle" axis="1 0 0" range="-0.5 0.5" ref="0.1"/>
          <geom name="left_foot_geom" type="box" pos="0.04 0 -0.01" size="0.08 0.04 0.02"/>
        </body>
      </body>
      <body name="right_leg" pos="0 -0.1 -0.2">
        <inertial pos="0 0 -0.1" mass="1" diaginertia="0.02 0.02 0.01"/>
        <joint name="right_hip" axis="0 1 0" range="-1 1" ref="0.2"/>
        <geom name="right_leg_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03"/>
        <body name="right_foot" pos="0 0 -0.25">
          <inertial pos="0.03 0 0" mass="0.5" diaginertia="0.01 0.008 0.004"/>
          <joint name="right_ankle" axis="1 0 0" range="-0.5 0.5" ref="0.1"/>
          <geom name="right_foot_geom" type="box" pos="0.04 0 -0.01" size="0.08 0.04 0.02"/>
        </body>
      </body>
      <body name="left_arm" pos="0 0.2 0.2">
        <inertial pos="0 0.08 0" mass="0.4" diaginertia="0.006 0.002 0.006"/>
        <joint name="left_shoulder" axis="1 0 0" range="-1.2 1.2" ref="-0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0.16 0" size="0.02"/>
        <body name="left_hand" pos="0 0.2 0">
          <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
          <joint name="left_wrist" axis="0 1 0" range="-0.6 0.6"/>
          <geom type="sphere" size="0.03"/>
        </body>
      </body>
      <body name="right_arm" pos="0 -0.2 0.2">
        <inertial pos="0 -0.08 0" mass="0.4" diaginertia="0.006 0.002 0.006"/>
        <joint name="right_shoulder" axis="1 0 0" range="-1.2 1.2" ref="-0.1"/>
        <geom type="capsule" fromto="0 0 0 0 -0.16 0" size="0.02"/>
        <body name="right_hand" pos="0 -0.2 0">
          <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
          <joint name="right_wrist" axis="0 1 0" range="-0.6 0.6"/>
          <geom type="sphere" size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    )
    return path


def test_zero_perturbation_is_exact_bundle_copy_and_write_once(tmp_path: Path) -> None:
    source = _tiny_mjcf(tmp_path / "tiny.xml")
    snapshot = MjcfBundleSnapshot.capture(source)
    plan = zero_perturbation_plan(snapshot, _semantics())
    variant = materialize_mjcf_variant(snapshot, plan, _semantics())
    assert variant.output_bundle_sha256 == snapshot.sha256
    assert variant.members == tuple(
        (name, item.data) for name, item in snapshot.members
    )
    assert variant.validation.qpos0_max_abs_error == 0.0
    assert variant.validation.root_spawn_max_abs_error == 0.0
    assert variant.validation.passed
    assert variant.manifest()["archival_ready"] is False
    with pytest.raises(ValueError, match="not archival-ready"):
        require_archival_ready_variant_manifest(
            variant.manifest(), observed_bundle_sha256=variant.output_bundle_sha256
        )

    provenance = GeneratorProvenance(
        command=("/usr/bin/python3", "scripts/materialize_mjcf_variants.py", "--zero"),
        python_version="3.10.0",
        mujoco_version="3.3.7",
        revisions=(
            ("snmr_commit", "1" * 40),
            ("snmr_dirty", False),
            ("snmr_repo_status", "available"),
            ("newton_commit", "2" * 40),
            ("newton_dirty", False),
            ("newton_repo_status", "available"),
            ("isaac_lab_commit", "3" * 40),
            ("isaac_lab_dirty", False),
            ("isaac_lab_repo_status", "available"),
        ),
    )
    variant = replace(variant, generator_provenance=provenance)
    assert variant.manifest()["archival_ready"] is True
    require_archival_ready_variant_manifest(
        variant.manifest(), observed_bundle_sha256=variant.output_bundle_sha256
    )
    require_archival_ready_variant_manifest(
        json.loads(json.dumps(variant.manifest(), allow_nan=False)),
        observed_bundle_sha256=variant.output_bundle_sha256,
    )
    with pytest.raises(ValueError, match="backend-rehashed"):
        require_archival_ready_variant_manifest(
            variant.manifest(), observed_bundle_sha256="0" * 64
        )
    assert (
        variant.manifest()["generator_provenance"]["command"][0] == "/usr/bin/python3"
    )

    entrypoint = write_materialized_mjcf_variant(variant, tmp_path / "published")
    replay = MjcfBundleSnapshot.capture(entrypoint)
    assert replay.sha256 == snapshot.sha256
    assert (entrypoint.parent / "variant_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        write_materialized_mjcf_variant(variant, tmp_path / "published")


def test_coherent_plan_is_bilateral_and_materialized_physics_obeys_rules(
    tmp_path: Path,
) -> None:
    source = _tiny_mjcf(tmp_path / "tiny.xml")
    snapshot = MjcfBundleSnapshot.capture(source)
    plan = build_semantic_limb_variant_plan(
        snapshot,
        _semantics(),
        seed=7,
        link_length_fraction=0.15,
        joint_limit_shift_rad=0.0,
    )
    scales = dict(plan.body_scales)
    shifts = dict(plan.joint_limit_center_shifts_rad)
    assert scales["pelvis"] == 1.0
    assert scales["left_leg"] == scales["right_leg"]
    assert scales["left_foot"] == scales["right_foot"]
    assert scales["left_arm"] == scales["right_arm"]
    assert scales["left_hand"] == scales["right_hand"]
    assert set(shifts.values()) == {0.0}
    assert len(plan.semantic_manifest_sha256) == 64
    assert plan.semantic_manifest == _semantics()
    assert {name for name, _ in plan.body_scale_groups} == {
        "feet.divergent_path_depth_0",
        "feet.divergent_path_depth_1",
        "hands.divergent_path_depth_0",
        "hands.divergent_path_depth_1",
    }

    variant = materialize_mjcf_variant(snapshot, plan, _semantics())
    assert variant.output_bundle_sha256 != snapshot.sha256
    assert variant.validation.passed
    assert variant.validation.qpos0_max_abs_error == 0.0
    assert variant.validation.root_spawn_max_abs_error == 0.0
    assert variant.validation.mass_rule_max_abs_error < 1.0e-12
    assert variant.validation.inertia_rule_max_abs_error < 1.0e-12
    entrypoint = write_materialized_mjcf_variant(variant, tmp_path / "variant")
    spec = RobotSpec.from_mjcf(entrypoint, _semantics())
    original = RobotSpec.from_mjcf(source, _semantics())
    assert spec.total_mass != pytest.approx(original.total_mass)
    assert [joint.nominal_position for joint in spec.joints] == pytest.approx(
        [joint.nominal_position for joint in original.joints], abs=0.0
    )


def test_adjacent_scales_assign_child_anchor_to_physical_parent(
    tmp_path: Path,
) -> None:
    source = _tiny_mjcf(tmp_path / "tiny.xml")
    snapshot = MjcfBundleSnapshot.capture(source)
    zero = zero_perturbation_plan(snapshot, _semantics())
    scales = dict(zero.body_scales)
    scales["left_leg"] = scales["right_leg"] = 1.10
    scales["left_foot"] = scales["right_foot"] = 0.90
    plan = replace(
        zero,
        body_scale_groups=(
            ("feet.divergent_path_depth_0", ("left_leg", "right_leg")),
            ("feet.divergent_path_depth_1", ("left_foot", "right_foot")),
        ),
        body_scales=tuple(sorted(scales.items())),
        max_link_length_fraction=0.10,
        seed=0,
    )
    variant = materialize_mjcf_variant(snapshot, plan, _semantics())
    assert variant.validation.passed

    source_root = ET.parse(source).getroot()
    output_root = ET.fromstring(dict(variant.members)[variant.entrypoint])

    def body(root: ET.Element, name: str) -> ET.Element:
        return next(item for item in root.iter("body") if item.get("name") == name)

    def vector(element: ET.Element, attribute: str) -> list[float]:
        return [float(value) for value in str(element.get(attribute)).split()]

    # The pelvis/root is fixed, so its outgoing hip anchor remains fixed even though
    # the leg itself has a 1.10 scale.  The ankle anchor is owned by that 1.10 leg,
    # while the foot's own geometry follows its independent 0.90 scale.
    output_leg = body(output_root, "left_leg")
    assert vector(output_leg, "pos") == pytest.approx(
        vector(body(source_root, "left_leg"), "pos"), abs=1.0e-15
    )
    source_foot = body(source_root, "left_foot")
    output_foot = body(output_root, "left_foot")
    assert vector(output_foot, "pos") == pytest.approx(
        [1.10 * value for value in vector(source_foot, "pos")], abs=1.0e-15
    )
    source_geom = next(child for child in source_foot if child.tag == "geom")
    output_geom = next(child for child in output_foot if child.tag == "geom")
    assert vector(output_geom, "size") == pytest.approx(
        [0.90 * value for value in vector(source_geom, "size")], abs=1.0e-15
    )


def test_plan_is_bound_and_root_scale_cannot_leak_spawn_height(tmp_path: Path) -> None:
    source = _tiny_mjcf(tmp_path / "tiny.xml")
    snapshot = MjcfBundleSnapshot.capture(source)
    plan = zero_perturbation_plan(snapshot, _semantics())
    scales = dict(plan.body_scales)
    scales["pelvis"] = 1.01
    bad_root = replace(
        plan,
        body_scales=tuple(sorted(scales.items())),
        max_link_length_fraction=0.02,
    )
    with pytest.raises(ValueError, match="free-root body scale"):
        materialize_mjcf_variant(snapshot, bad_root, _semantics())
    with pytest.raises(ValueError, match="different source"):
        materialize_mjcf_variant(
            snapshot,
            replace(plan, source_bundle_sha256="0" * 64),
            _semantics(),
        )

    with pytest.raises(ValueError, match="mirrored physical"):
        build_semantic_limb_variant_plan(
            snapshot,
            _semantics(),
            seed=0,
            joint_limit_shift_rad=math.radians(1.0),
        )

    shifts = dict(plan.joint_limit_center_shifts_rad)
    shifts["left_hip"] = 1.0e-13
    with pytest.raises(ValueError, match="exactly zero"):
        materialize_mjcf_variant(
            snapshot,
            replace(
                plan,
                joint_limit_center_shifts_rad=tuple(sorted(shifts.items())),
            ),
            _semantics(),
        )


def test_semantic_path_draws_ignore_xml_order_and_opaque_names(tmp_path: Path) -> None:
    source = _tiny_mjcf(tmp_path / "tiny.xml")
    source_snapshot = MjcfBundleSnapshot.capture(source)
    source_plan = build_semantic_limb_variant_plan(
        source_snapshot,
        _semantics(),
        seed=91,
        link_length_fraction=0.15,
    )

    mapping = {
        "pelvis": "node_0",
        "left_leg": "node_1",
        "left_foot": "node_2",
        "right_leg": "node_3",
        "right_foot": "node_4",
        "left_arm": "node_5",
        "left_hand": "node_6",
        "right_arm": "node_7",
        "right_hand": "node_8",
    }
    tree = ET.parse(source)
    root = tree.getroot()
    pelvis = next(
        element for element in root.iter("body") if element.get("name") == "pelvis"
    )
    body_children = [child for child in pelvis if child.tag == "body"]
    for child in body_children:
        pelvis.remove(child)
    pelvis.extend(reversed(body_children))
    for body in root.iter("body"):
        name = body.get("name")
        if name in mapping:
            body.set("name", mapping[name])
    renamed = tmp_path / "renamed.xml"
    tree.write(renamed, encoding="unicode")
    renamed_semantics = SemanticManifest(
        root_link=mapping["pelvis"],
        left_hand_link=mapping["left_hand"],
        right_hand_link=mapping["right_hand"],
        left_foot_link=mapping["left_foot"],
        right_foot_link=mapping["right_foot"],
        allowed_contact_links=(mapping["left_foot"], mapping["right_foot"]),
    )
    renamed_plan = build_semantic_limb_variant_plan(
        MjcfBundleSnapshot.capture(renamed),
        renamed_semantics,
        seed=91,
        link_length_fraction=0.15,
    )
    source_scales = dict(source_plan.body_scales)
    renamed_scales = dict(renamed_plan.body_scales)
    for original, opaque in mapping.items():
        assert source_scales[original] == renamed_scales[opaque]


def test_qualified_dialect_rejects_defaults_instead_of_guessing_inherited_physics(
    tmp_path: Path,
) -> None:
    source = _tiny_mjcf(tmp_path / "tiny.xml")
    data = source.read_text().replace(
        '<compiler angle="radian"/>',
        '<compiler angle="radian"/><default><geom density="500"/></default>',
    )
    source.write_text(data)
    snapshot = MjcfBundleSnapshot.capture(source)
    plan = zero_perturbation_plan(snapshot, _semantics())
    # A zero plan is allowed to preserve arbitrary valid bytes exactly.  A physical
    # intervention must pass through the qualified mutator and therefore rejects defaults.
    scales = dict(plan.body_scales)
    scales["left_leg"] = scales["right_leg"] = 1.01
    nonzero = replace(
        plan,
        body_scale_groups=(("feet.divergent_path_depth_0", ("left_leg", "right_leg")),),
        body_scales=tuple(sorted(scales.items())),
        max_link_length_fraction=0.02,
    )
    with pytest.raises(ValueError, match="default"):
        materialize_mjcf_variant(snapshot, nonzero, _semantics())


def test_real_g1_bundle_round_trip_and_registered_15_percent_seeds(
    g1_mjcf: str, tmp_path: Path
) -> None:
    semantics = SemanticManifest(
        root_link="pelvis",
        torso_link="torso_link",
        left_hand_link="left_rubber_hand_link",
        right_hand_link="right_rubber_hand_link",
        left_foot_link="left_ankle_roll_link",
        right_foot_link="right_ankle_roll_link",
        allowed_contact_links=("left_ankle_roll_link", "right_ankle_roll_link"),
        symmetry_pairs=(
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ("left_rubber_hand_link", "right_rubber_hand_link"),
        ),
    )
    snapshot = MjcfBundleSnapshot.capture(g1_mjcf)
    zero = materialize_mjcf_variant(
        snapshot, zero_perturbation_plan(snapshot, semantics), semantics
    )
    assert zero.output_bundle_sha256 == snapshot.sha256

    variants = []
    for seed in range(8):
        plan = build_semantic_limb_variant_plan(
            snapshot,
            semantics,
            seed=seed,
            link_length_fraction=0.15,
            joint_limit_shift_rad=0.0,
        )
        variant = materialize_mjcf_variant(snapshot, plan, semantics)
        assert variant.validation.passed, seed
        assert variant.validation.body_count == 50
        assert variant.validation.hinge_joint_count == 29
        variants.append(variant)

    variant = variants[0]
    entrypoint = write_materialized_mjcf_variant(variant, tmp_path / "g1_variant")
    assert MjcfBundleSnapshot.capture(entrypoint).sha256 == variant.output_bundle_sha256
