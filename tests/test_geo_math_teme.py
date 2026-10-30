"""TEME -> ECEF state conversion and the LVLH frame helpers (Issue #19).

The golden vectors were produced once with Skyfield 1.53
(skyfield.sgp4lib.TEME_to_ITRF(jd, r, v * 86400, 0, 0, fr), velocity divided by
86400 because Skyfield works in km/day) from sgp4 2.25 states of three synthetic
TLEs at 2026-10-30 06:00 UTC by tools/gen_teme_golden.py. Skyfield is not a
CI dependency.
"""

from __future__ import annotations

import logging
import math
import warnings

import pytest

sgp4_api = pytest.importorskip("sgp4.api")

from astra_shared.geo_math import (
    FRAME_VELOCITY_UNITS_ERROR,
    OMEGA_EARTH_RAD_S,
    check_inertial_speed_km_s,
    lvlh_reference_velocity,
    teme_state_to_ecef,
    teme_to_ecef,
    teme_vel_to_ecef,
)

GOLDEN = {
    "leo53": {
        "jd": 2461343.5, "fr": 0.27569444444444446,
        "r_teme": (-5718.947642669565, -3777.6970255458705, 1019.1109419642153),
        "v_teme": (1.798407499431905, -4.3427439994989925, -5.95664246788026),
        "r_ecef": (1704.6398620740788, 6638.641419525681, 1019.1109419642153),
        "v_ecef": (-3.7635934527278327, 1.8883598905445518, -5.95664246788026),
    },
    "sso97": {
        "jd": 2461343.5, "fr": 0.29930555555555555,
        "r_teme": (1637.6689030504508, -1046.5040251963935, 6692.548522371264),
        "v_teme": (3.454417503214307, -6.465112656038823, -1.8524752215148648),
        "r_ecef": (-1943.158161106676, -35.587521622641475, 6692.548522371264),
        "v_ecef": (-6.459325978819114, 3.611756145697115, -1.8524752215148648),
    },
    "geo0": {
        "jd": 2461343.5, "fr": 0.6666666666666667,
        "r_teme": (-23952.31250897708, -34699.67943799665, 4.976797601063617),
        "v_teme": (2.530425564720641, -1.7466904408854758, -0.0009148354243030834),
        "r_ecef": (30533.89513555953, -29076.834000249775, 4.976797601063617),
        "v_ecef": (7.255780349556449e-05, 7.432044072393857e-05, -0.0009148354243030832),
    },
}

# Synthetic 550 km / 53 deg TLE for the propagation-based checks.
LEO_TLE = (
    "1 00001U 26001A   26303.25000000  .00000000  00000-0  00000-0 0  9990",
    "2 00001  53.0000  40.0000 0000000   0.0000  30.0000 15.05000000    00",
)


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _rotated(case):
    """The pre-fix result: TEME velocity rotated into ECEF axes (inertial velocity)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return teme_vel_to_ecef(*case["v_teme"], case["jd"] + case["fr"])


def _propagate(sat, jd, fr):
    e, r, v = sat.sgp4(jd, fr)
    assert e == 0
    return r, v


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_teme_state_to_ecef_matches_skyfield(name):
    case = GOLDEN[name]
    r, v = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    assert _norm(_sub(r, case["r_ecef"])) * 1000.0 <= 1.0  # metres
    assert _norm(_sub(v, case["v_ecef"])) * 1000.0 <= 1e-3  # m/s


def test_velocity_is_time_derivative_of_ecef_position():
    """v_ecef equals the central difference of the ECEF positions: this test would have caught #19."""
    sat = sgp4_api.Satrec.twoline2rv(*LEO_TLE, sgp4_api.WGS72)
    jd = sat.jdsatepoch
    h_day = 0.5 / 86400.0
    for minute in range(0, 96, 8):
        fr = sat.jdsatepochF + minute / 1440.0
        r, v = _propagate(sat, jd, fr)
        _, v_ecef = teme_state_to_ecef(r, v, jd + fr)
        r_plus = teme_to_ecef(*_propagate(sat, jd, fr + h_day)[0], jd + fr + h_day)
        r_minus = teme_to_ecef(*_propagate(sat, jd, fr - h_day)[0], jd + fr - h_day)
        fd = tuple((p - m) / 1.0 for p, m in zip(r_plus, r_minus))  # km over 1 s
        assert _norm(_sub(v_ecef, fd)) * 1000.0 <= 0.5, minute


def test_speed_is_not_preserved_for_prograde_leo():
    case = GOLDEN["leo53"]
    _, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    delta_m_s = (_norm(_rotated(case)) - _norm(v_ecef)) * 1000.0
    assert 250.0 <= delta_m_s <= 350.0


@pytest.mark.parametrize("name", ["sso97", "geo0"])
def test_speed_change_matches_closed_form(name):
    """|v_in| - |v_ecef| = (2 v_in . w - |w|^2) / (|v_in| + |v_ecef|), w = omega x r."""
    case = GOLDEN[name]
    r, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    v_in = _rotated(case)
    w = (-OMEGA_EARTH_RAD_S * r[1], OMEGA_EARTH_RAD_S * r[0], 0.0)
    expected = (2.0 * sum(a * b for a, b in zip(v_in, w)) - _norm(w) ** 2) / (_norm(v_in) + _norm(v_ecef))
    assert abs((_norm(v_in) - _norm(v_ecef)) - expected) * 1000.0 <= 1.0
    if name == "sso97":
        assert (_norm(v_in) - _norm(v_ecef)) < 0.0  # retrograde: the sign flips


def test_geo_earth_relative_speed_is_small():
    case = GOLDEN["geo0"]
    _, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    assert _norm(v_ecef) * 1000.0 < 5.0


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_lvlh_reference_velocity_round_trips_to_rotated_velocity(name):
    case = GOLDEN[name]
    r, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    assert _norm(_sub(lvlh_reference_velocity(r, v_ecef), _rotated(case))) <= 1e-9


def test_lvlh_reference_velocity_keeps_zero_sentinel():
    assert lvlh_reference_velocity((42164.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_speed_check_passes_correct_states(name):
    case = GOLDEN[name]
    r, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    check_inertial_speed_km_s(r, lvlh_reference_velocity(r, v_ecef))


@pytest.mark.parametrize("name", ["leo53", "sso97"])
@pytest.mark.parametrize("scale", [1000.0, 1e-3])
def test_speed_check_rejects_wrong_units_for_leo(name, scale):
    case = GOLDEN[name]
    r, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    wrong = tuple(c * scale for c in v_ecef)
    with pytest.raises(ValueError, match=FRAME_VELOCITY_UNITS_ERROR):
        check_inertial_speed_km_s(r, lvlh_reference_velocity(r, wrong))


@pytest.mark.parametrize("scale", [1000.0, 1e-3])
def test_speed_check_cannot_see_geo_unit_mistakes(scale):
    """Documented gap: omega x r alone is ~3.07 km/s at GEO, inside the band."""
    case = GOLDEN["geo0"]
    r, v_ecef = teme_state_to_ecef(case["r_teme"], case["v_teme"], case["jd"] + case["fr"])
    wrong = tuple(c * scale for c in v_ecef)
    check_inertial_speed_km_s(r, lvlh_reference_velocity(r, wrong))


def test_speed_check_rejects_nan():
    with pytest.raises(ValueError, match=FRAME_VELOCITY_UNITS_ERROR):
        check_inertial_speed_km_s((6928.0, 0.0, 0.0), (math.nan, 7.6, 0.0))


def test_speed_check_logs_before_raising(caplog):
    with caplog.at_level(logging.ERROR, logger="astra_shared.geo_math"), pytest.raises(ValueError):
        check_inertial_speed_km_s((6928.0, 0.0, 0.0), (0.0, 7600.0, 0.0))
    assert any(FRAME_VELOCITY_UNITS_ERROR in rec.getMessage() for rec in caplog.records)


def test_teme_vel_to_ecef_is_deprecated():
    with pytest.warns(DeprecationWarning):
        teme_vel_to_ecef(1.0, 2.0, 3.0, 2461343.5)
