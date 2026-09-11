"""Counterexamples to the withdrawn spline guarantees, and the narrower contracts that hold."""

from __future__ import annotations

import numpy as np
import pytest

from snmr.correction_basis import (
    BoundedSplineCorrection,
    ChannelBounds,
    EndpointConstraint,
    basis_at,
    clamped_cubic_basis,
    clip_realized_frames,
    continuity_report,
    energy_fraction_above,
    enforce_joint_limits,
    local_time_map_from_ramp,
    pinned_endpoint_coefficients,
    stitch_additive,
)

L, H = 100, 8  # a 2 s interval at 50 Hz with the spec's 8-frame knots
BOUND = ChannelBounds({"q": 0.35})


def _random_correction(seed: int, channels=("q",), bounds=BOUND, length=L) -> BoundedSplineCorrection:
    rng = np.random.default_rng(seed)
    constraint = EndpointConstraint(length, H, 2)
    theta = rng.normal(size=(constraint.free_dim, len(channels)))
    return BoundedSplineCorrection(length, H, channels, bounds, 2, theta).projected()


# --- basis sanity -------------------------------------------------------------------------


def test_basis_partition_of_unity_and_shape():
    b = clamped_cubic_basis(L, H)
    assert b.shape == (L, 16)
    assert np.allclose(b.sum(axis=1), 1.0)
    assert (b >= 0).all()


def test_analytic_derivative_matches_finite_difference():
    t = np.linspace(5.0, 90.0, 300)
    eps = 1e-4
    d1 = basis_at(L, H, t, deriv=1)
    fd = (basis_at(L, H, t + eps) - basis_at(L, H, t - eps)) / (2 * eps)
    assert np.allclose(d1, fd, atol=1e-5)
    d2 = basis_at(L, H, t, deriv=2)
    fd2 = (basis_at(L, H, t + eps, 1) - basis_at(L, H, t - eps, 1)) / (2 * eps)
    assert np.allclose(d2, fd2, atol=1e-4)


# --- counterexample 1: additive endpoint-clamped residuals CAN attenuate motion -----------


def test_additive_endpoint_clamped_residual_halves_amplitude():
    """q0 = 0.2 b(t), delta = -0.1 b(t): inside the box, zero value/1st/2nd derivative at both
    ends, and the amplitude is exactly halved.  The old "cannot shrink" claim is false."""
    b = clamped_cubic_basis(L, H)
    k = b.shape[1] // 2  # an interior basis function
    q0 = 0.2 * b[:, k]
    coefficients = np.zeros((b.shape[1], 1))
    coefficients[k, 0] = -0.1
    delta = BoundedSplineCorrection.from_coefficients(L, H, ("q",), BOUND, coefficients)
    assert np.abs(delta.coefficients).max() <= 0.35
    assert np.abs(delta.endpoint_derivatives(order=2)).max() == 0.0
    corrected = q0 + delta.evaluate()[:, 0]
    assert corrected.max() == pytest.approx(0.5 * q0.max())
    assert np.allclose(corrected, 0.5 * q0)


# --- counterexample 2: no time channel does NOT prevent local timing changes ---------------


def test_no_time_channel_still_permits_local_time_warp():
    """ramp + admissible residual == ramp(phi(t)) with a non-trivial interior time map."""
    b = clamped_cubic_basis(L, H)
    k = b.shape[1] // 2
    coefficients = np.zeros((b.shape[1], 1))
    coefficients[k, 0] = 0.045
    residual = BoundedSplineCorrection.from_coefficients(L, H, ("q",), BOUND, coefficients)
    delta = residual.evaluate()[:, 0]
    phi, rate = local_time_map_from_ramp(delta, L)
    ramp = np.arange(L) / (L - 1)
    assert np.allclose(ramp + delta, phi / (L - 1))  # q0(t)+delta(t) == q0(phi(t))
    assert phi[0] == 0.0 and phi[-1] == pytest.approx(L - 1)  # fixed endpoints, same duration
    assert (rate > 0).all()  # a valid (monotone) time map
    assert rate.min() < 0.85 and rate.max() > 1.15  # local progression rate changes by >15%
    assert not np.allclose(rate, 1.0)


# --- counterexample 3: pinning two control points per end is C1, not C2 --------------------


def test_pinned_control_points_are_C1_not_C2():
    rng = np.random.default_rng(1)
    b = clamped_cubic_basis(L, H)
    c = pinned_endpoint_coefficients(L, H, rng.normal(size=b.shape[1] - 4))
    ends = np.array([0.0, L - 1.0])
    assert np.allclose(basis_at(L, H, ends, 0) @ c, 0.0)
    assert np.allclose(basis_at(L, H, ends, 1) @ c, 0.0)
    assert not np.allclose(basis_at(L, H, ends, 2) @ c, 0.0)
    with pytest.raises(ValueError):  # the C2 constructor refuses it
        BoundedSplineCorrection.from_coefficients(L, H, ("q",), ChannelBounds({"q": 10.0}), c[:, None], continuity_order=2)
    piece = BoundedSplineCorrection.from_coefficients(L, H, ("q",), ChannelBounds({"q": 10.0}), c[:, None], continuity_order=1)
    report = continuity_report([(0, piece)], order=2)
    join = report["joins"][0]
    assert join["C0"]["pass"] and join["C1"]["pass"] and not join["C2"]["pass"]


def test_clamped_C2_constraint_pins_three_coefficients_per_end():
    constraint = EndpointConstraint(L, H, 2)
    assert constraint.free_dim == clamped_cubic_basis(L, H).shape[1] - 6
    corr = _random_correction(2)
    c = corr.coefficients[:, 0]
    assert np.allclose(c[:3], 0.0) and np.allclose(c[-3:], 0.0)


# --- counterexample 4: uniform knot spacing is a smoothness scale, not a cutoff ------------


def test_knot_spacing_is_not_a_frequency_cutoff():
    """The spec claimed 0.16 s knots 'bandlimit eta to ~3 Hz'.  Alternating control points on
    the same knot grid put most of their energy above 3 Hz at 50 fps."""
    length = 400
    b = clamped_cubic_basis(length, H)
    coefficients = np.zeros((b.shape[1], 1))
    coefficients[3:-3, 0] = [(-1) ** i for i in range(b.shape[1] - 6)]
    corr = BoundedSplineCorrection.from_coefficients(length, H, ("q",), ChannelBounds({"q": 10.0}), coefficients)
    assert energy_fraction_above(corr.evaluate()[:, 0], fps=50.0, f_cut_hz=3.0) > 0.5
    # And a single basis function is not bandlimited either (small but nonzero tail).
    assert energy_fraction_above(b[:, b.shape[1] // 2], fps=50.0, f_cut_hz=3.0) > 0.0


# --- the contracts that DO hold ------------------------------------------------------------


def test_fixed_grid_duration_and_bounds():
    corr = _random_correction(3)
    delta = corr.evaluate()
    assert delta.shape == (L, 1)  # no resampling, no duration change
    assert np.abs(corr.coefficients).max() <= 0.35 + 1e-12
    assert np.abs(delta).max() <= 0.35 + 1e-12  # convex-hull property


def test_projection_methods_keep_constraints_and_box():
    rng = np.random.default_rng(4)
    constraint = EndpointConstraint(L, H, 2)
    raw = BoundedSplineCorrection(L, H, ("q",), BOUND, 2, 5.0 * rng.normal(size=(constraint.free_dim, 1)))
    assert raw.coefficient_violation()[0] > 1.0
    for method in ("clip", "scale"):
        proj = raw.projected(method)
        assert proj.coefficient_violation()[0] <= 1.0 + 1e-12
        assert np.abs(constraint.matrix @ proj.coefficients).max() <= 1e-9
        assert np.abs(proj.endpoint_derivatives()).max() <= 1e-9


def test_continuity_survives_projection_and_stitching():
    a = _random_correction(5)
    b = _random_correction(6, length=60)
    base = np.zeros((200, 1))
    stitched = stitch_additive(base, [(10, a.evaluate()), (80, b.evaluate())])  # overlapping
    assert stitched.shape == base.shape
    report = continuity_report([(10, a), (80, b)], order=2)
    assert report["pass"], report
    assert len(report["joins"]) == 4


def test_frame_level_clipping_breaks_C1():
    """Clipping the realized trajectory (rather than the coefficients) introduces kinks."""
    corr = _random_correction(7)
    delta = 3.0 * corr.evaluate()
    clipped = clip_realized_frames(delta, np.array([0.35]))
    active = (np.abs(delta) > 0.35).sum()
    assert active > 0
    kink = np.abs(np.diff(clipped, 2, axis=0)).max()
    smooth = np.abs(np.diff(corr.evaluate(), 2, axis=0)).max()
    assert kink > 3.0 * smooth


def test_joint_limits_enforced_by_scaling_not_clipping():
    corr = _random_correction(8)
    q0 = np.full((L, 1), 0.3)
    result = enforce_joint_limits(q0, corr, lower=np.array([-0.5]), upper=np.array([0.45]))
    assert result.status == "scaled" and 0.0 < result.scale < 1.0
    q = q0 + result.correction.evaluate()
    assert (q <= 0.45 + 1e-6).all() and (q >= -0.5 - 1e-6).all()
    assert result.corrected_violations == 0
    assert np.abs(result.correction.endpoint_derivatives()).max() <= 1e-9
    fine = enforce_joint_limits(q0, corr.scaled(0.01), lower=np.array([-0.5]), upper=np.array([0.45]))
    assert fine.status == "unchanged" and fine.scale == 1.0


def test_invalid_baseline_yields_abstention_not_a_repair():
    corr = _random_correction(9)
    q0 = np.full((L, 1), 0.9)  # already outside the limits
    result = enforce_joint_limits(q0, corr, lower=np.array([-0.5]), upper=np.array([0.45]))
    assert result.status == "baseline_invalid"
    assert result.correction is None and result.baseline_violations == L


def test_rejects_wrong_theta_shape_and_nonfinite():
    constraint = EndpointConstraint(L, H, 2)
    with pytest.raises(ValueError):
        BoundedSplineCorrection(L, H, ("q",), BOUND, 2, np.zeros((constraint.free_dim + 1, 1)))
    bad = np.zeros((constraint.free_dim, 1))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        BoundedSplineCorrection(L, H, ("q",), BOUND, 2, bad)
    with pytest.raises(KeyError):
        BoundedSplineCorrection(L, H, ("missing",), BOUND, 2, np.zeros((constraint.free_dim, 1))).projected()
