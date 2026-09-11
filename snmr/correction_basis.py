"""Bounded clamped cubic B-spline corrections on a fixed frame grid.

What this module guarantees (each item has a test in ``tests/test_correction_basis.py``):

* **Fixed sample grid and duration.**  A correction is a function on the interval's own frame
  indices ``0..L-1``; it never resamples, drops, or adds frames.
* **Bounded edits.**  Coefficients are kept inside a per-channel box.  Because a B-spline lies
  in the convex hull of its control points, the realized per-frame edit is inside the same box.
* **Endpoint continuity of the declared order.**  Zero value, first and second derivative at
  both interval ends are enforced through an explicit linear-constraint nullspace
  (:class:`EndpointConstraint`).  Pinning the first two / last two control points of a clamped
  cubic spline enforces only value and first derivative — see
  ``test_pinned_control_points_are_C1_not_C2`` — so the old "C² by construction" wording is
  withdrawn and replaced by a tested constraint.
* **Continuity survives box projection and stitching.**  Projection is radial in the free
  parameters (so it stays inside the constraint nullspace exactly) and stitching is additive
  with zero-padded corrections, so the C⁰/C¹/C² report is unchanged by either step.

What this module does **not** guarantee, and what the tests demonstrate by counterexample:

* It does **not** prevent amplitude reduction.  ``delta = -0.5 * q0`` is representable
  whenever ``q0`` is itself a spline in the family (``test_additive_endpoint_clamped_residual_halves_amplitude``).
* It does **not** prevent local timing changes.  A ramp plus an admissible residual equals the
  ramp composed with a non-trivial interior time map with fixed endpoints
  (``test_no_time_channel_still_permits_local_time_warp``).
* Uniform knot spacing is a smoothness *scale*, not a frequency cutoff
  (``test_knot_spacing_is_not_a_frequency_cutoff``).

Whether a correction preserves task intent is therefore a property that has to be **checked**
(:mod:`snmr.repair_intent`), never one that follows from this parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

DEGREE = 3


# --- basis --------------------------------------------------------------------------------


def clamped_knots(length: int, knot_spacing: int, degree: int = DEGREE) -> np.ndarray:
    """Clamped knot vector on ``[0, length-1]`` with interior knots every ``knot_spacing`` frames."""
    if length < 2:
        raise ValueError("an interval needs at least two frames")
    if knot_spacing < 1:
        raise ValueError("knot_spacing must be a positive number of frames")
    end = float(length - 1)
    interior = np.arange(knot_spacing, length - 1, knot_spacing, dtype=np.float64)
    return np.concatenate([np.zeros(degree + 1), interior, np.full(degree + 1, end)])


def _bspline_basis(t: np.ndarray, knots: np.ndarray, degree: int, deriv: int = 0) -> np.ndarray:
    """Cox–de Boor basis (and derivatives) evaluated at ``t``; shape ``(len(t), n_basis)``."""
    t = np.asarray(t, dtype=np.float64)
    n_basis = len(knots) - degree - 1
    end = knots[-1]
    # Degree-0 basis: half-open intervals, with the last non-empty interval closed at the end.
    basis = np.zeros((t.size, len(knots) - 1))
    for i in range(len(knots) - 1):
        lo, hi = knots[i], knots[i + 1]
        if hi > lo:
            inside = (t >= lo) & (t < hi)
            if hi == end:
                inside |= t == end
            basis[:, i] = inside
    # Raise the degree.
    for p in range(1, degree + 1):
        nxt = np.zeros((t.size, len(knots) - p - 1))
        for i in range(len(knots) - p - 1):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]
            if left_den > 0:
                nxt[:, i] += (t - knots[i]) / left_den * basis[:, i]
            if right_den > 0:
                nxt[:, i] += (knots[i + p + 1] - t) / right_den * basis[:, i + 1]
        basis = nxt
    if deriv == 0:
        return basis[:, :n_basis]
    # Derivative: B'_{i,p} = p [ B_{i,p-1}/(t_{i+p}-t_i) - B_{i+1,p-1}/(t_{i+p+1}-t_{i+1}) ].
    lower = _bspline_basis(t, knots, degree - 1, deriv - 1)
    out = np.zeros((t.size, n_basis))
    for i in range(n_basis):
        left_den = knots[i + degree] - knots[i]
        right_den = knots[i + degree + 1] - knots[i + 1]
        if left_den > 0:
            out[:, i] += degree * lower[:, i] / left_den
        if right_den > 0 and i + 1 < lower.shape[1]:
            out[:, i] -= degree * lower[:, i + 1] / right_den
    return out


def clamped_cubic_basis(length: int, knot_spacing: int, deriv: int = 0) -> np.ndarray:
    """Basis matrix ``(length, K)`` of the clamped cubic B-spline on the frame grid."""
    knots = clamped_knots(length, knot_spacing)
    return _bspline_basis(np.arange(length, dtype=np.float64), knots, DEGREE, deriv)


def basis_at(length: int, knot_spacing: int, t: np.ndarray, deriv: int = 0) -> np.ndarray:
    """Basis matrix at arbitrary (possibly fractional) frame times ``t``."""
    return _bspline_basis(np.asarray(t, dtype=np.float64), clamped_knots(length, knot_spacing), DEGREE, deriv)


# --- endpoint constraints -----------------------------------------------------------------


@dataclass(frozen=True)
class EndpointConstraint:
    """Zero value / derivatives up to ``order`` at both ends, as a linear map on coefficients."""

    length: int
    knot_spacing: int
    order: int = 2

    def __post_init__(self) -> None:
        if self.order < 0 or self.order > DEGREE - 1:
            raise ValueError("continuity order must be between 0 and degree-1 (2 for cubic)")
        k = clamped_cubic_basis(self.length, self.knot_spacing).shape[1]
        if k <= 2 * (self.order + 1):
            raise ValueError(
                f"interval of {self.length} frames with knot spacing {self.knot_spacing} has "
                f"{k} control points; {2 * (self.order + 1)} are consumed by the constraints"
            )

    @property
    def matrix(self) -> np.ndarray:
        """Constraint rows ``A`` such that admissible coefficient vectors satisfy ``A c = 0``."""
        ends = np.array([0.0, float(self.length - 1)])
        rows = [basis_at(self.length, self.knot_spacing, ends, deriv=d) for d in range(self.order + 1)]
        return np.concatenate(rows, axis=0)

    @property
    def nullspace(self) -> np.ndarray:
        """Orthonormal basis ``N`` (K × F) of ``{c : A c = 0}``; free parameters live in R^F."""
        a = self.matrix
        _, s, vt = np.linalg.svd(a, full_matrices=True)
        rank = int((s > 1e-10 * max(s.max(), 1.0)).sum())
        return vt[rank:].T

    @property
    def free_dim(self) -> int:
        return self.nullspace.shape[1]


def pinned_endpoint_coefficients(length: int, knot_spacing: int, interior: np.ndarray) -> np.ndarray:
    """The spec's original construction: first two / last two control points pinned to zero."""
    k = clamped_cubic_basis(length, knot_spacing).shape[1]
    interior = np.asarray(interior, dtype=np.float64)
    if interior.shape != (k - 4,):
        raise ValueError(f"expected {k - 4} interior coefficients, got {interior.shape}")
    return np.concatenate([[0.0, 0.0], interior, [0.0, 0.0]])


# --- bounded correction -------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelBounds:
    """Per-channel L∞ bounds on the correction, addressed by channel name."""

    bounds: Mapping[str, float]

    def __post_init__(self) -> None:
        for name, value in self.bounds.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"bound for {name!r} must be positive and finite")

    def vector(self, channels: Sequence[str]) -> np.ndarray:
        try:
            return np.array([self.bounds[c] for c in channels], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(f"no bound declared for channel {exc.args[0]!r}") from None


@dataclass(frozen=True)
class BoundedSplineCorrection:
    """A bounded, endpoint-constrained additive correction on one interval.

    ``theta`` has shape ``(F, C)``: ``F`` free spline parameters per channel, ``C`` channels.
    """

    length: int
    knot_spacing: int
    channels: tuple[str, ...]
    bounds: ChannelBounds
    continuity_order: int = 2
    theta: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        constraint = EndpointConstraint(self.length, self.knot_spacing, self.continuity_order)
        theta = (
            np.zeros((constraint.free_dim, len(self.channels)))
            if self.theta is None
            else np.asarray(self.theta, dtype=np.float64)
        )
        if theta.shape != (constraint.free_dim, len(self.channels)):
            raise ValueError(
                f"theta must have shape ({constraint.free_dim}, {len(self.channels)}), got {theta.shape}"
            )
        if not np.isfinite(theta).all():
            raise ValueError("theta contains NaN/Inf")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "channels", tuple(self.channels))

    # -- structure --
    @property
    def constraint(self) -> EndpointConstraint:
        return EndpointConstraint(self.length, self.knot_spacing, self.continuity_order)

    @property
    def coefficients(self) -> np.ndarray:
        """Control points ``(K, C)``; always satisfy the endpoint constraints exactly."""
        return self.constraint.nullspace @ self.theta

    @property
    def dim(self) -> int:
        return int(self.theta.size)

    # -- bound handling --
    def coefficient_violation(self) -> np.ndarray:
        """Per-channel ratio ``max|c| / bound`` (≤ 1 means inside the box)."""
        c = self.coefficients
        return np.abs(c).max(axis=0) / self.bounds.vector(self.channels)

    def projected(self, method: str = "clip") -> "BoundedSplineCorrection":
        """Return a copy whose control points lie inside the per-channel box.

        ``method="clip"`` clips control points coordinate-wise (the Euclidean projection).  For
        clamped ends the continuity constraints pin individual end coefficients to zero, so
        clipping keeps them satisfied; this is verified and, should the residual ever be
        nonzero, the method falls back to ``"scale"``: radial scaling of the free parameters,
        which stays inside the constraint nullspace for any constraint matrix.

        Neither method touches the realized trajectory directly: clipping *frames* of the
        realized correction would introduce kinks (``test_frame_level_clipping_breaks_C1``).
        """
        if method not in ("clip", "scale"):
            raise ValueError("method must be 'clip' or 'scale'")
        bound = self.bounds.vector(self.channels)
        constraint = self.constraint
        if method == "clip":
            c = np.clip(self.coefficients, -bound[None, :], bound[None, :])
            residual = np.abs(constraint.matrix @ c).max() if c.size else 0.0
            if residual <= 1e-9:
                return BoundedSplineCorrection(
                    self.length, self.knot_spacing, self.channels, self.bounds,
                    self.continuity_order, constraint.nullspace.T @ c,
                )
        ratio = self.coefficient_violation()
        scale = np.where(ratio > 1.0, 1.0 / np.maximum(ratio, 1e-12), 1.0)
        return BoundedSplineCorrection(
            self.length, self.knot_spacing, self.channels, self.bounds, self.continuity_order,
            self.theta * scale[None, :],
        )

    def scaled(self, factor: float) -> "BoundedSplineCorrection":
        """Uniformly scale the correction (stays in the nullspace, so continuity is kept)."""
        return BoundedSplineCorrection(
            self.length, self.knot_spacing, self.channels, self.bounds, self.continuity_order,
            self.theta * float(factor),
        )

    def endpoint_derivatives(self, order: int | None = None) -> np.ndarray:
        """Analytic ``(order+1, 2, C)`` array of d^k/dt^k at the first and last frame."""
        order = self.continuity_order if order is None else order
        ends = np.array([0.0, float(self.length - 1)])
        return np.stack(
            [basis_at(self.length, self.knot_spacing, ends, deriv=k) @ self.coefficients for k in range(order + 1)]
        )

    # -- evaluation --
    def evaluate(self, deriv: int = 0) -> np.ndarray:
        """Realized correction ``(length, C)`` on the frame grid (or its ``deriv``-th derivative)."""
        return clamped_cubic_basis(self.length, self.knot_spacing, deriv) @ self.coefficients

    def realized_violation(self) -> np.ndarray:
        return np.abs(self.evaluate()).max(axis=0) / self.bounds.vector(self.channels)

    @classmethod
    def from_coefficients(
        cls,
        length: int,
        knot_spacing: int,
        channels: Sequence[str],
        bounds: ChannelBounds,
        coefficients: np.ndarray,
        continuity_order: int = 2,
        atol: float = 1e-9,
    ) -> "BoundedSplineCorrection":
        """Build from control points that already satisfy the constraints (checked)."""
        constraint = EndpointConstraint(length, knot_spacing, continuity_order)
        c = np.asarray(coefficients, dtype=np.float64)
        residual = np.abs(constraint.matrix @ c).max() if c.size else 0.0
        if residual > atol:
            raise ValueError(
                f"coefficients violate the endpoint constraints (max residual {residual:.3e})"
            )
        theta = constraint.nullspace.T @ c
        return cls(length, knot_spacing, tuple(channels), bounds, continuity_order, theta)


# --- stitching and continuity checks ------------------------------------------------------


def stitch_additive(base: np.ndarray, corrections: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """Add zero-padded interval corrections ``(start, delta)`` to ``base`` ``(T, C)``.

    Overlapping intervals simply sum (each is zero-clamped, so sums stay continuous).  The
    sample grid and duration of ``base`` are unchanged by construction.
    """
    out = np.array(base, dtype=np.float64, copy=True)
    if out.ndim != 2:
        raise ValueError("base must be (T, C)")
    for start, delta in corrections:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.ndim != 2 or delta.shape[1] != out.shape[1]:
            raise ValueError("each delta must be (L, C) with the base channel count")
        if start < 0 or start + delta.shape[0] > out.shape[0]:
            raise ValueError("correction interval falls outside the base trajectory")
        out[start : start + delta.shape[0]] += delta
    return out


def _composite_derivative(
    pieces: Sequence[tuple[int, "BoundedSplineCorrection"]], t: float, deriv: int, side: str
) -> np.ndarray:
    """One-sided k-th derivative of the stitched correction at global time ``t``.

    Each piece contributes its analytic spline derivative when ``t`` lies inside its interval
    (approached from ``side``), and zero otherwise — the correction is zero-padded.
    """
    channels = pieces[0][1].channels
    out = np.zeros(len(channels))
    for start, corr in pieces:
        if corr.channels != channels:
            raise ValueError("all stitched pieces must share the channel tuple")
        local = t - start
        end = corr.length - 1
        inside = (0.0 < local < end) or (local == 0.0 and side == "right") or (local == end and side == "left")
        if inside:
            out += (basis_at(corr.length, corr.knot_spacing, np.array([local]), deriv) @ corr.coefficients)[0]
    return out


def continuity_report(
    pieces: Sequence[tuple[int, "BoundedSplineCorrection"]],
    *,
    order: int = 2,
    abs_tol: float = 1e-9,
) -> dict:
    """Analytic C⁰..C^order check of a stitched correction at every interval boundary.

    Finite differences on the frame grid cannot certify continuity (the second difference
    across a C² join scales with the *third* derivative), so this compares one-sided analytic
    derivatives of the stitched spline at each join.  A join passes at order ``k`` when the
    left and right limits agree to ``abs_tol`` for every channel.
    """
    if not pieces:
        return {"order": order, "joins": [], "pass": True}
    joins = sorted({float(s) for s, _ in pieces} | {float(s + c.length - 1) for s, c in pieces})
    report: dict = {"order": order, "joins": [], "pass": True}
    for t in joins:
        entry: dict = {"time": t}
        for k in range(order + 1):
            left = _composite_derivative(pieces, t, k, "left")
            right = _composite_derivative(pieces, t, k, "right")
            gap = float(np.abs(left - right).max())
            entry[f"C{k}"] = {"jump": gap, "pass": bool(gap <= abs_tol)}
            report["pass"] &= bool(gap <= abs_tol)
        report["joins"].append(entry)
    return report


def discrete_join_differences(delta: np.ndarray, start: int, total_length: int, order: int = 2) -> dict:
    """Magnitudes of finite differences across the joins of a zero-padded frame correction.

    Diagnostic only: the k-th difference across a join of a spline with zero derivatives up to
    order k scales with the (k+1)-th derivative, so it is not a pass/fail continuity test.
    """
    delta = np.asarray(delta, dtype=np.float64)
    padded = np.zeros((total_length, delta.shape[1]))
    padded[start : start + delta.shape[0]] = delta
    out = {}
    for k in range(order + 1):
        diff = padded
        for _ in range(k):
            diff = np.diff(diff, axis=0)
        lo, hi = max(start - 1, 0), min(start + delta.shape[0] - k + 1, diff.shape[0])
        out[f"delta{k}"] = {
            "left_join": float(np.abs(diff[lo : lo + 2]).max()),
            "right_join": float(np.abs(diff[max(hi - 2, 0) : hi]).max()),
            "interior_max": float(np.abs(diff[lo + 2 : hi - 2]).max()) if hi - 2 > lo + 2 else 0.0,
        }
    return out


def clip_realized_frames(delta: np.ndarray, bound: np.ndarray) -> np.ndarray:
    """Frame-wise clipping of a realized correction — introduces kinks; provided for the test."""
    bound = np.asarray(bound, dtype=np.float64)
    return np.clip(delta, -bound[None, :], bound[None, :])


@dataclass(frozen=True)
class LimitEnforcement:
    """Outcome of :func:`enforce_joint_limits`."""

    status: str  # "unchanged" | "scaled" | "baseline_invalid"
    scale: float
    correction: "BoundedSplineCorrection | None"
    baseline_violations: int
    corrected_violations: int


def enforce_joint_limits(
    q0: np.ndarray,
    correction: "BoundedSplineCorrection",
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 40,
) -> LimitEnforcement:
    """Shrink a correction until ``q0 + delta`` respects ``[lower, upper]`` per channel.

    Uniform scaling keeps the correction in its constraint nullspace, so continuity and the
    coefficient box are preserved.  If the *baseline* already violates the limits the result
    is ``baseline_invalid`` (the caller must abstain rather than claim a valid output);
    scaling to zero could never repair it.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if q0.shape != (correction.length, len(correction.channels)):
        raise ValueError("q0 must be (length, channels) matching the correction")

    def violations(q: np.ndarray) -> int:
        return int(((q < lower[None, :] - tol) | (q > upper[None, :] + tol)).sum())

    base_bad = violations(q0)
    if base_bad:
        return LimitEnforcement("baseline_invalid", 0.0, None, base_bad, base_bad)
    delta = correction.evaluate()
    if not violations(q0 + delta):
        return LimitEnforcement("unchanged", 1.0, correction, 0, 0)
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if violations(q0 + mid * delta):
            hi = mid
        else:
            lo = mid
    scaled = correction.scaled(lo)
    return LimitEnforcement("scaled", lo, scaled, 0, violations(q0 + scaled.evaluate()))


def local_time_map_from_ramp(delta: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """For ``q0(t) = t/(L-1)`` and an additive residual ``delta``, the equivalent time map.

    ``q0(t) + delta(t) = q0(phi(t))`` with ``phi(t) = t + (L-1) * delta(t)``.  Returns
    ``(phi, dphi/dt)`` on the frame grid; ``dphi/dt`` is the local progression rate.
    """
    t = np.arange(length, dtype=np.float64)
    phi = t + (length - 1) * np.asarray(delta, dtype=np.float64).reshape(-1)
    rate = np.gradient(phi)
    return phi, rate


def energy_fraction_above(signal: np.ndarray, fps: float, f_cut_hz: float) -> float:
    """Fraction of the signal's spectral energy above ``f_cut_hz`` (mean removed)."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fps)
    total = spectrum.sum()
    return float(spectrum[freqs > f_cut_hz].sum() / total) if total > 0 else 0.0
