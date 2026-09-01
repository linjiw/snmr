"""Fail-closed adapter from SNMR's LAFAN1 pair NPZ to HumanMotionSpec.

This module integrates exactly one existing training path: files written by
``scripts/make_pairs_lafan1.py`` and read by :func:`snmr.human.load_pair_npz`.
It deliberately does not infer dataset identity, clip identity, split membership,
transformations, landmark semantics, contacts, or validity from filenames.

The pair's human tensors are already world-space positions and scalar-first ``wxyz``
orientations.  The adapter preserves those tensors and uses the declared ``Hips`` body as
both root position and root orientation.  Heading-local encoder features are downstream
derived inputs and are not substituted for the canonical world-space motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from .human import LAFAN1_BODY_NAMES, load_pair_npz
from .motion_spec import (
    BilateralSegmentLandmarks,
    BodySegmentScales,
    FootContactProtocol,
    HumanFrameConvention,
    HumanMotionSpec,
    MotionProvenance,
    MotionSource,
)
from .provenance import ArtifactMismatchError, ArtifactSnapshot


LAFAN1_ROOT_BODY = "Hips"
MOTION_ADAPTER_SCHEMA_VERSION = "snmr.motion-adapter.v0.1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class HumanMotionAdaptation:
    """Canonical motion plus the separately verified derived-pair artifact.

    ``motion_spec.source.sha256`` identifies the unprocessed human source clip.  The
    fields here identify the exact robot-specific pair container consumed by this
    adapter.  Keeping both prevents a regenerated pair ZIP or different robot teacher
    from silently changing the human source identity used for split grouping.
    """

    motion_spec: HumanMotionSpec
    pair_artifact_sha256: str
    pair_artifact_size_bytes: int
    scale_landmarks: BilateralSegmentLandmarks
    schema_version: str = MOTION_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.motion_spec, HumanMotionSpec):
            raise TypeError("motion_spec must be HumanMotionSpec")
        if not isinstance(self.scale_landmarks, BilateralSegmentLandmarks):
            raise TypeError("scale_landmarks must be BilateralSegmentLandmarks")
        if self.schema_version != MOTION_ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported motion-adapter schema {self.schema_version!r}")
        if not isinstance(self.pair_artifact_sha256, str) or _SHA256_RE.fullmatch(
            self.pair_artifact_sha256
        ) is None:
            raise ValueError("pair_artifact_sha256 must be a 64-character hexadecimal digest")
        if (
            isinstance(self.pair_artifact_size_bytes, bool)
            or not isinstance(self.pair_artifact_size_bytes, int)
            or self.pair_artifact_size_bytes <= 0
        ):
            raise ValueError("pair_artifact_size_bytes must be a positive integer")
        object.__setattr__(
            self, "pair_artifact_sha256", self.pair_artifact_sha256.lower()
        )

    @property
    def source_group_key(self) -> tuple[str, str, str]:
        """Stable split key from the raw human source, never the pair container."""

        source = self.motion_spec.source
        return (source.dataset, source.sequence_id, source.sha256)

    def to_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema_version": self.schema_version,
            "source": {
                "dataset": self.motion_spec.source.dataset,
                "sequence_id": self.motion_spec.source.sequence_id,
                "sha256": self.motion_spec.source.sha256,
            },
            "source_group_key": list(self.source_group_key),
            "pair_artifact": {
                "sha256": self.pair_artifact_sha256,
                "size_bytes": self.pair_artifact_size_bytes,
            },
            "motion_buffer_sha256": self.motion_spec.buffer_sha256,
            "motion_spec_sha256": self.motion_spec.spec_sha256,
            "scale_landmarks": self.scale_landmarks.to_dict(),
        }
        encoded = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        return manifest


def _declared_body_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("body_names must be an explicit sequence of names")
    names = tuple(values)
    if not names:
        raise ValueError("body_names must not be empty")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise ValueError("body_names must contain non-empty, whitespace-trimmed strings")
    if len(set(names)) != len(names):
        raise ValueError("body_names must be unique")
    return names


def _validity_array(value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("validity_mask must be on CPU at the file-adapter boundary")
        array = value.detach().numpy()
    else:
        array = np.asarray(value)
    if array.dtype.kind != "b":
        raise TypeError("validity_mask must have boolean dtype")
    if array.shape != expected_shape:
        raise ValueError(f"validity_mask must have shape {expected_shape}, got {array.shape}")
    return np.asarray(array, dtype=np.bool_, order="C")


def adapt_lafan1_pair_npz(
    path: str | Path,
    *,
    pair_artifact_sha256: str,
    source: MotionSource,
    provenance: MotionProvenance,
    body_names: Sequence[str],
    root_body_name: str,
    landmark_pairs: BilateralSegmentLandmarks,
    contact_protocol: FootContactProtocol,
    frames: HumanFrameConvention,
    validity_mask: Any,
) -> HumanMotionAdaptation:
    """Bind one current SNMR LAFAN1/GMR pair artifact to HumanMotionSpec.

    Required caller declarations are intentionally verbose.  ``source.sha256`` identifies
    the unprocessed human clip.  ``pair_artifact_sha256`` independently binds the exact
    robot-specific pair NPZ bytes consumed here.  ``provenance.clip_range`` is interpreted
    as a half-open frame slice into the pair.  The validity mask covers that selected slice,
    not the full unsliced file.

    The pair-provided ``human_height`` is the normalization length for bilateral segment
    extraction.  Contacts are derived by the supplied protocol.  No binary legacy contact
    flags or filename-derived split/subject metadata are imported.
    """

    if not isinstance(source, MotionSource):
        raise TypeError("source must be MotionSource")
    if not isinstance(provenance, MotionProvenance):
        raise TypeError("provenance must be MotionProvenance")
    if not isinstance(landmark_pairs, BilateralSegmentLandmarks):
        raise TypeError("landmark_pairs must be BilateralSegmentLandmarks")
    if not isinstance(contact_protocol, FootContactProtocol):
        raise TypeError("contact_protocol must be FootContactProtocol")
    if not isinstance(frames, HumanFrameConvention):
        raise TypeError("frames must be HumanFrameConvention")
    if not isinstance(pair_artifact_sha256, str) or _SHA256_RE.fullmatch(
        pair_artifact_sha256
    ) is None:
        raise ValueError("pair_artifact_sha256 must be a 64-character hexadecimal digest")
    expected_pair_sha256 = pair_artifact_sha256.lower()

    declared_names = _declared_body_names(body_names)
    canonical_names = tuple(LAFAN1_BODY_NAMES)
    if declared_names != canonical_names:
        raise ValueError(
            "this adapter integrates only the current 24-body LAFAN1 training order; "
            "body_names must equal snmr.human.LAFAN1_BODY_NAMES"
        )
    if root_body_name != LAFAN1_ROOT_BODY or declared_names[0] != root_body_name:
        raise ValueError(
            f"root_body_name must be {LAFAN1_ROOT_BODY!r} at index 0 for this adapter"
        )

    snapshot = ArtifactSnapshot.capture(path)
    if not hmac.compare_digest(snapshot.sha256, expected_pair_sha256):
        raise ArtifactMismatchError(
            "pair_artifact_sha256 does not match the exact LAFAN1 pair NPZ bytes "
            f"({expected_pair_sha256} != {snapshot.sha256})"
        )

    # Load the immutable captured bytes rather than reopening the mutable source path.
    with tempfile.TemporaryDirectory(prefix="snmr-human-pair-") as directory:
        consumed_path = snapshot.materialize(Path(directory) / "pair.npz")
        pair = load_pair_npz(str(consumed_path), dtype=torch.float64)

    loaded_names = tuple(pair["human_names"])
    if loaded_names != declared_names:
        raise ValueError(
            "pair human_names do not match the caller-declared canonical LAFAN1 body order"
        )
    human_pos = pair["human_pos"]
    human_quat = pair["human_quat"]
    if not isinstance(human_pos, torch.Tensor) or not isinstance(human_quat, torch.Tensor):
        raise TypeError("load_pair_npz must return tensor human_pos and human_quat")
    if human_pos.device.type != "cpu" or human_quat.device.type != "cpu":
        raise ValueError("pair human tensors must remain on CPU at the file-adapter boundary")
    if human_pos.ndim != 3 or human_pos.shape[1:] != (len(declared_names), 3):
        raise ValueError(
            f"human_pos must have shape (T, {len(declared_names)}, 3), "
            f"got {tuple(human_pos.shape)}"
        )
    if human_quat.shape != human_pos.shape[:2] + (4,):
        raise ValueError(
            f"human_quat must have shape {human_pos.shape[:2] + (4,)}, "
            f"got {tuple(human_quat.shape)}"
        )

    start, stop = provenance.clip_range
    total_frames = int(human_pos.shape[0])
    if stop > total_frames:
        raise ValueError(
            f"provenance clip_range {provenance.clip_range} exceeds pair length {total_frames}"
        )
    selected_frames = stop - start
    validity = _validity_array(validity_mask, (selected_frames, len(declared_names)))

    selected_pos = human_pos[start:stop].detach().numpy()
    selected_quat = human_quat[start:stop].detach().numpy()
    normalization_length_m = float(pair["human_height"])
    if not np.isfinite(normalization_length_m) or normalization_length_m <= 0.0:
        raise ValueError("pair human_height must be finite and positive")
    segment_scales = BodySegmentScales.from_landmarks(
        body_positions=selected_pos,
        body_names=declared_names,
        landmark_pairs=landmark_pairs,
        normalization_length_m=normalization_length_m,
        validity_mask=validity,
    )

    root_index = declared_names.index(root_body_name)
    motion_spec = HumanMotionSpec.from_source(
        source=source,
        provenance=provenance,
        source_fps=pair["fps"],
        body_names=declared_names,
        segment_scales=segment_scales,
        root_position=selected_pos[:, root_index, :],
        root_orientation_wxyz=selected_quat[:, root_index, :],
        body_positions=selected_pos,
        body_orientations_wxyz=selected_quat,
        validity_mask=validity,
        contacts=None,
        contact_protocol=contact_protocol,
        frames=frames,
    )
    return HumanMotionAdaptation(
        motion_spec=motion_spec,
        pair_artifact_sha256=snapshot.sha256,
        pair_artifact_size_bytes=snapshot.size_bytes,
        scale_landmarks=landmark_pairs,
    )


__all__ = [
    "adapt_lafan1_pair_npz",
    "HumanMotionAdaptation",
    "LAFAN1_ROOT_BODY",
    "MOTION_ADAPTER_SCHEMA_VERSION",
]
