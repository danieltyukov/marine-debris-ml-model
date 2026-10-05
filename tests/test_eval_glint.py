"""The glint angle in scripts/eval_glint.py: zero in the sun's mirror direction."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def glint():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("eval_glint")
    finally:
        sys.path.remove(str(SCRIPTS))


def test_sensor_in_the_mirror_direction_sees_zero_glint_angle(glint):
    # Sun 20 degrees off zenith to the east; sensor 20 degrees off zenith to the west.
    assert glint.glint_angle(20.0, 90.0, 20.0, 270.0) == pytest.approx(0.0, abs=1e-9)


def test_sensor_on_the_sun_side_is_far_from_glint(glint):
    assert glint.glint_angle(20.0, 90.0, 10.0, 90.0) == pytest.approx(30.0)


def test_nadir_view_glint_angle_is_the_sun_zenith(glint):
    assert glint.glint_angle(25.0, 120.0, 0.0, 0.0) == pytest.approx(25.0)
