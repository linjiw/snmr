"""Pure validation helpers for provenance-bound E70 simulator captures."""

from __future__ import annotations

import re

import torch


_CAPTURE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def validate_capture_name(name: str) -> str:
    if _CAPTURE_NAME.fullmatch(name) is None:
        raise ValueError(f"invalid capture name: {name!r}")
    return name


def capture_start_grid(
    requested: str,
    motion_starts: torch.Tensor,
    motion_ends: torch.Tensor,
    *,
    num_envs: int,
    horizon_steps: int,
) -> torch.Tensor:
    """Put an exact global start in env 0 and deterministically fill companions.

    A single environment is sufficient for clean, time, and shuffled-content
    captures.  The matched-marginal command intervention needs a population from
    which to estimate each command dimension, so it uses the registered 1,024-env
    phase grid as companions while still rendering only env 0.
    """
    if num_envs < 1:
        raise ValueError("capture requires at least one environment")
    try:
        exact = int(requested)
    except ValueError as exc:
        raise ValueError(f"exact capture start is not an integer: {requested!r}") from exc
    if str(exact) != requested.strip():
        raise ValueError(f"exact capture start is not canonical: {requested!r}")
    if motion_starts.shape != motion_ends.shape or motion_starts.ndim != 1:
        raise ValueError("motion boundaries must be equal-length vectors")
    if horizon_steps < 1:
        raise ValueError("capture horizon must be positive")

    ranges: list[tuple[int, int]] = []
    for motion_start, motion_end in zip(motion_starts.tolist(), motion_ends.tolist()):
        first = int(motion_start) + 1
        last = int(motion_end) - horizon_steps - 2
        if last < first:
            raise ValueError("a loaded motion is shorter than the capture horizon")
        ranges.append((first, last))
    if not any(first <= exact <= last for first, last in ranges):
        raise ValueError(
            f"exact capture start {exact} does not fit a {horizon_steps}-step rollout "
            "inside one motion"
        )

    if num_envs == 1:
        return torch.tensor([exact], device=motion_starts.device, dtype=torch.long)

    counts = [num_envs // len(ranges)] * len(ranges)
    for index in range(num_envs % len(ranges)):
        counts[index] += 1
    grids = [
        torch.linspace(first, last, count, device=motion_starts.device).long()
        for count, (first, last) in zip(counts, ranges)
    ]
    result = torch.cat(grids)
    result[0] = exact
    return result


def expected_simulator_envs(destroy_zcmd: str) -> int:
    """Return the frozen simulator batch size for an illustrative capture."""
    return 1024 if destroy_zcmd == "marginal_random" else 1

