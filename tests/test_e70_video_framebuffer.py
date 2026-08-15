"""Regression tests for the E70 capture offscreen-framebuffer clamp.

The 2026-08-15T02:34:59Z B4 capture crashed with ``Image width 1920 > framebuffer width 640``
because the G1 MJCF declares no ``<visual><global offwidth/offheight>`` clause and MuJoCo
defaults the offscreen framebuffer to 640x480.  ``fit_offscreen_framebuffer`` grows it to the
realized recorder config before the first ``env.reset_all()``.
"""

from __future__ import annotations

import types

import pytest

from scripts.e70_video_runtime import fit_offscreen_framebuffer


class _Global:
    def __init__(self, offwidth: int, offheight: int) -> None:
        self.offwidth = offwidth
        self.offheight = offheight


def _env(*, offwidth=640, offheight=480, width=1920, height=1080, recorder=True, model=True):
    simulator = types.SimpleNamespace()
    simulator.video_recorder = (
        types.SimpleNamespace(config=types.SimpleNamespace(width=width, height=height))
        if recorder
        else None
    )
    simulator.root_model = (
        types.SimpleNamespace(vis=types.SimpleNamespace(global_=_Global(offwidth, offheight)))
        if model
        else None
    )
    return types.SimpleNamespace(simulator=simulator)


def test_grows_default_framebuffer_to_the_configured_capture_size():
    env = _env()
    report = fit_offscreen_framebuffer(env)
    visual = env.simulator.root_model.vis.global_
    assert (visual.offwidth, visual.offheight) == (1920, 1080)
    assert report["offwidth_before"] == 640
    assert report["offheight_before"] == 480
    assert report["requested_width"] == 1920
    assert report["requested_height"] == 1080


def test_never_shrinks_an_already_large_framebuffer():
    env = _env(offwidth=3840, offheight=2160)
    fit_offscreen_framebuffer(env)
    visual = env.simulator.root_model.vis.global_
    assert (visual.offwidth, visual.offheight) == (3840, 2160)


def test_takes_dimensions_from_the_recorder_config_not_a_hardcoded_constant():
    """The clamp must track --logger.video.width/height so it cannot silently disagree."""
    env = _env(width=1280, height=720)
    fit_offscreen_framebuffer(env)
    visual = env.simulator.root_model.vis.global_
    assert (visual.offwidth, visual.offheight) == (1280, 720)


def test_grows_each_axis_independently():
    env = _env(offwidth=2560, offheight=480)
    fit_offscreen_framebuffer(env)
    visual = env.simulator.root_model.vis.global_
    assert (visual.offwidth, visual.offheight) == (2560, 1080)


@pytest.mark.parametrize("missing", ["recorder", "model"])
def test_returns_none_when_capture_surface_is_absent(missing):
    env = _env(recorder=missing != "recorder", model=missing != "model")
    assert fit_offscreen_framebuffer(env) is None
