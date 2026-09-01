from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from snmr.robot_spec import (
    CollisionProxy,
    ControlSpec,
    FrameConvention,
    JointSpec,
    LinkSpec,
    RobotSpec,
    RuntimeSpec,
    SemanticManifest,
)
from snmr.robot_variants import (
    CoherentVariantConfig,
    generate_virtual_coherent_variant,
)


def _link(name: str, parent: str | None, xyz: tuple[float, float, float]) -> LinkSpec:
    return LinkSpec(
        name=name,
        parent=parent,
        local_position=xyz,
        local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        mass=2.0,
        center_of_mass=(0.0, 0.0, -0.1),
        inertia_diagonal=(0.02, 0.03, 0.04),
        inertia_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_proxies=(CollisionProxy(
            geometry_type="box",
            local_position=(0.0, 0.0, -0.1),
            local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            size=(0.05, 0.04, 0.2),
            friction=(1.0, 0.005, 0.0001),
            contact_type=1,
            contact_affinity=1,
        ),),
    )


def _joint(name: str, child: str) -> JointSpec:
    return JointSpec(
        name=name,
        parent_link="pelvis",
        child_link=child,
        joint_type="revolute",
        axis=(0.0, 1.0, 0.0),
        lower_limit=-0.8,
        upper_limit=1.2,
        velocity_limit=4.0,
        torque_limit=20.0,
        armature=0.01,
        damping=0.1,
        friction_loss=0.0,
        nominal_position=0.0,
        kp=30.0,
        kd=1.0,
        local_position=(0.0, 0.0, -0.02),
    )


def _spec() -> RobotSpec:
    spec = RobotSpec(
        schema_version="snmr.robot.v0.1",
        asset_sha256="a" * 64,
        frames=FrameConvention(),
        semantics=SemanticManifest(
            root_link="pelvis",
            left_foot_link="left_foot",
            right_foot_link="right_foot",
            allowed_contact_links=("left_foot", "right_foot"),
            symmetry_pairs=(("left_foot", "right_foot"),),
        ),
        control=ControlSpec(control_dt=0.02, latency_seconds=0.0),
        runtime=RuntimeSpec(simulation_dt=0.002),
        total_mass=6.0,
        standing_height=0.5,
        arm_span=None,
        links=(
            _link("pelvis", None, (0.0, 0.0, 0.0)),
            _link("left_foot", "pelvis", (0.0, 0.1, -0.5)),
            _link("right_foot", "pelvis", (0.0, -0.1, -0.5)),
        ),
        joints=(
            _joint("left_joint", "left_foot"),
            _joint("right_joint", "right_foot"),
        ),
    )
    spec.validate()
    return spec


def test_variant_is_deterministic_hash_bound_and_same_topology():
    source = _spec()
    first, first_manifest = generate_virtual_coherent_variant(
        source, parent_family="fixture", seed=7
    )
    second, second_manifest = generate_virtual_coherent_variant(
        source, parent_family="fixture", seed=7
    )
    assert first == second
    assert first_manifest == second_manifest
    assert first_manifest.output_spec_hash == first.spec_hash
    assert first_manifest.output_kinematic_hash == first.kinematic_hash
    assert first_manifest.virtual_asset_sha256 == first.asset_sha256
    assert first_manifest.virtual_asset_sha256 == hashlib.sha256(
        first_manifest.virtual_asset_json.encode("utf-8")
    ).hexdigest()
    assert first_manifest.backend_compatible is False
    with pytest.raises(RuntimeError, match="tokenizer-only"):
        first_manifest.require_backend_compatible()
    assert first.runtime.source_format == "snmr.virtual-robot-spec-json.v0.1"
    assert first_manifest.topology_changed is False
    assert all(value for _, value in first_manifest.consistency_checks)
    assert len(first_manifest.to_dict()["manifest_sha256"]) == 64


def test_variant_respects_registered_ranges_and_bilateral_symmetry():
    source = _spec()
    config = CoherentVariantConfig(
        link_length_fraction=0.15,
        joint_limit_shift_rad=math.radians(10.0),
    )
    variant, manifest = generate_virtual_coherent_variant(
        source, parent_family="fixture", seed=11, config=config
    )
    scales = dict(manifest.link_scales)
    shifts = dict(manifest.joint_limit_center_shifts_rad)
    assert scales["left_foot"] == scales["right_foot"]
    assert all(0.85 <= scale <= 1.15 for scale in scales.values())
    assert all(abs(shift) <= math.radians(10.0) for shift in shifts.values())
    assert shifts["left_joint"] == shifts["right_joint"]
    source_joints = {joint.name: joint for joint in source.joints}
    for joint in variant.joints:
        original = source_joints[joint.name]
        assert joint.upper_limit - joint.lower_limit == pytest.approx(
            original.upper_limit - original.lower_limit
        )
        assert joint.lower_limit <= joint.nominal_position <= joint.upper_limit


def test_geometry_mass_and_inertia_follow_constant_density_rule():
    source = _spec()
    variant, manifest = generate_virtual_coherent_variant(
        source, parent_family="fixture", seed=13
    )
    scales = dict(manifest.link_scales)
    source_links = {link.name: link for link in source.links}
    for link in variant.links:
        original = source_links[link.name]
        scale = scales[link.name]
        assert link.local_position == pytest.approx(
            np.asarray(original.local_position) * scale
        )
        assert link.center_of_mass == pytest.approx(
            np.asarray(original.center_of_mass) * scale
        )
        assert link.mass == pytest.approx(original.mass * scale ** 3)
        assert link.inertia_diagonal == pytest.approx(
            np.asarray(original.inertia_diagonal) * scale ** 5
        )
        assert link.collision_proxies[0].size == pytest.approx(
            np.asarray(original.collision_proxies[0].size) * scale
        )
    assert variant.total_mass == pytest.approx(sum(link.mass for link in variant.links))
    assert variant.standing_height > 0.0


def test_variant_is_invariant_to_input_serialization_with_names_held_fixed():
    source = _spec()
    serialized = replace(
        source,
        links=(source.links[2], source.links[0], source.links[1]),
        joints=tuple(reversed(source.joints)),
    )
    serialized.validate()
    first, _ = generate_virtual_coherent_variant(source, parent_family="fixture", seed=17)
    second, _ = generate_virtual_coherent_variant(serialized, parent_family="fixture", seed=17)
    assert first == second
    assert first.spec_hash == second.spec_hash


def test_free_root_spawn_pose_is_not_a_variant_or_seed_cue():
    source = _spec()
    root = next(link for link in source.links if link.name == "pelvis")
    translated = replace(
        source,
        links=tuple(
            replace(link, local_position=(0.4, -0.2, 0.9))
            if link.name == "pelvis"
            else link
            for link in source.links
        ),
    )
    translated.validate()
    variant, manifest = generate_virtual_coherent_variant(
        translated, parent_family="fixture", seed=19
    )
    variant_root = next(link for link in variant.links if link.name == "pelvis")
    assert variant_root.local_position == (0.4, -0.2, 0.9)
    assert dict(manifest.link_scales)["pelvis"] == 1.0
    assert root.local_position == (0.0, 0.0, 0.0)


def test_variant_manifest_is_deeply_immutable_and_rehashable():
    _, manifest = generate_virtual_coherent_variant(
        _spec(), parent_family="fixture", seed=23
    )
    with pytest.raises((AttributeError, TypeError)):
        manifest.config.link_length_fraction = 0.9  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.consistency_checks[0] = ("tampered", False)  # type: ignore[index]
    assert manifest.to_dict() == manifest.to_dict()


def test_nonzero_nominal_is_rejected_until_nominal_fk_is_qualified():
    source = _spec()
    bent_joint = replace(source.joints[0], nominal_position=0.2)
    bent = replace(source, joints=(bent_joint, source.joints[1]))
    bent.validate()
    with pytest.raises(ValueError, match="requires zero nominal joint positions"):
        generate_virtual_coherent_variant(bent, parent_family="fixture", seed=0)


@pytest.mark.parametrize(
    "config,match",
    [
        (CoherentVariantConfig(link_length_fraction=1.0), "link_length_fraction"),
        (CoherentVariantConfig(joint_limit_shift_rad=math.pi), "joint_limit_shift_rad"),
    ],
)
def test_variant_config_rejects_unregistered_ranges(config, match):
    with pytest.raises(ValueError, match=match):
        generate_virtual_coherent_variant(
            _spec(), parent_family="fixture", seed=0, config=config
        )
