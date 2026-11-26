"""A0 shared nadir, off-axis and frame-velocity entries (A0 plan v7, sections 6.1.5 and 6.1.7).

The shared part of G07/G08: Radio's tests/test_a0_rf_geometry.py covers the adapter.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import wgs84_reference as ref

from astra_shared.geo_math import (
    FRAME_VELOCITY_UNITS_ERROR,
    OMEGA_EARTH_RAD_S,
    check_inertial_speed_km_s,
    lvlh_reference_velocity,
    lvlh_reference_velocity_m,
    lvlh_reference_velocity_m_vec,
    nadir_unit_vec,
    off_axis_deg_vec,
)

# ---------------------------------------------------------------- nadir and off-axis (section 6.1.5)


def test_nadir_is_geocentric_not_the_geodetic_normal():
    sat = np.array(ref.forward(45.0, 30.0, 550e3))
    nadir = nadir_unit_vec(sat)
    np.testing.assert_allclose(nadir, -sat / np.linalg.norm(sat), atol=1e-15)
    minus_up = -np.array(ref.enu_axes(45.0, 30.0)[2])
    gap = math.degrees(math.acos(float(np.clip(nadir @ minus_up, -1.0, 1.0))))
    assert gap > 0.1


def test_nadir_shapes_and_errors():
    assert nadir_unit_vec((7e6, 0.0, 0.0)).shape == (3,)
    assert nadir_unit_vec(np.ones((4, 2, 3))).shape == (4, 2, 3)
    with pytest.raises(ValueError, match="non-zero"):
        nadir_unit_vec((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        nadir_unit_vec((float("nan"), 0.0, 1.0))
    with pytest.raises(ValueError, match="final axis"):
        nadir_unit_vec((1.0, 2.0))


@pytest.mark.parametrize(
    ("boresight", "point", "expected"),
    [((1, 0, 0), (5, 0, 0), 0.0), ((1, 0, 0), (0, 5, 0), 90.0), ((1, 0, 0), (-5, 0, 0), 180.0),
     ((1, 0, 0), (1, 1, 0), 45.0)],
)
def test_off_axis_known_angles(boresight, point, expected):
    assert float(off_axis_deg_vec(boresight, (0.0, 0.0, 0.0), point)) == pytest.approx(expected, abs=1e-12)


def test_off_axis_keeps_precision_near_zero_and_180():
    tiny = 1e-9  # rad; acos(dot) would lose this
    point = (math.cos(tiny), math.sin(tiny), 0.0)
    assert float(off_axis_deg_vec((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), point)) == pytest.approx(
        math.degrees(tiny), rel=1e-6
    )
    back = (-math.cos(tiny), math.sin(tiny), 0.0)
    assert float(off_axis_deg_vec((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), back)) == pytest.approx(
        180.0 - math.degrees(tiny), abs=1e-12
    )


def test_off_axis_accepts_a_non_unit_boresight():
    for scale in (1e-300, 1e-6, 1.0, 1e300):
        got = float(off_axis_deg_vec((0.0, 0.0, scale), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0)))
        assert got == pytest.approx(45.0, abs=1e-12)


def test_off_axis_steering_angle_of_the_geodetic_sub_point():
    # Plan section 10.1: at 45 deg and 550 km the geodetic sub-point is 0.177 deg off nadir.
    sat = np.array(ref.forward(45.0, 30.0, 550e3))
    ground = np.array(ref.forward(45.0, 30.0, 0.0))
    angle = float(off_axis_deg_vec(nadir_unit_vec(sat), sat, ground))
    assert angle == pytest.approx(0.1771, abs=5e-4)


def test_off_axis_broadcasts_beams_against_points():
    beams = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])  # (2, 1, 3)
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)
    got = off_axis_deg_vec(beams, np.zeros(3), points)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, [[0.0, 90.0, 90.0], [90.0, 0.0, 90.0]], atol=1e-12)


def test_off_axis_errors():
    with pytest.raises(ValueError, match="boresight_unit"):
        off_axis_deg_vec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="point_m must differ"):
        off_axis_deg_vec((1.0, 0.0, 0.0), (7e6, 0.0, 0.0), (7e6, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        off_axis_deg_vec((1.0, 0.0, 0.0), (7e6, 0.0, 0.0), (float("inf"), 0.0, 0.0))


# ---------------------------------------------------------------- frame reference velocity (section 6.1.7)

LEO_R_M = (6928137.0, 0.0, 0.0)
LEO_VREL_MPS = (0.0, 7600.0 - OMEGA_EARTH_RAD_S * 6928137.0, 0.0)  # Earth-relative, so v_ref is 7.6 km/s


def test_stationary_satellite_gets_east_without_a_speed_check():
    v_ref = lvlh_reference_velocity_m(LEO_R_M, (0.0, 0.0, 0.0))
    assert v_ref == pytest.approx((0.0, OMEGA_EARTH_RAD_S * LEO_R_M[0], 0.0), abs=1e-12)
    # 505 m/s at 550 km is far below the bound-orbit band; the stationary branch skips the check.
    assert 500.0 < v_ref[1] < 510.0
    east = ref.enu_axes(0.0, 0.0)[0]
    assert np.dot(np.array(v_ref) / np.linalg.norm(v_ref), east) == pytest.approx(1.0, abs=1e-12)


def test_rebuild_matches_the_km_helper():
    r_km = tuple(c / 1000.0 for c in LEO_R_M)
    v_km = tuple(c / 1000.0 for c in LEO_VREL_MPS)
    legacy = lvlh_reference_velocity(r_km, v_km)
    metre = lvlh_reference_velocity_m(LEO_R_M, LEO_VREL_MPS)
    assert metre == pytest.approx(tuple(c * 1000.0 for c in legacy), rel=1e-15, abs=1e-9)


def test_wrong_unit_velocity_raises_the_frame_units_error(caplog):
    wrong = tuple(c * 1000.0 for c in LEO_VREL_MPS)  # m/s passed where it is already m/s times 1000
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
        lvlh_reference_velocity_m(LEO_R_M, wrong)
    with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
        lvlh_reference_velocity_m(LEO_R_M, tuple(c / 1000.0 for c in LEO_VREL_MPS))  # km/s by mistake


def test_mixed_vector_applies_the_stationary_branch_to_exact_zero_only():
    r = np.array([LEO_R_M, LEO_R_M])
    v = np.array([LEO_VREL_MPS, (0.0, 0.0, 0.0)])
    got = lvlh_reference_velocity_m_vec(r, v)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got[0], lvlh_reference_velocity_m(LEO_R_M, LEO_VREL_MPS), atol=1e-9)
    np.testing.assert_allclose(got[1], (0.0, OMEGA_EARTH_RAD_S * LEO_R_M[0], 0.0), atol=1e-12)


def test_one_out_of_band_element_fails_the_vector_call():
    r = np.array([LEO_R_M, LEO_R_M])
    v = np.array([LEO_VREL_MPS, (0.0, 1.0, 0.0)])
    with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
        lvlh_reference_velocity_m_vec(r, v)


def test_non_finite_is_rejected_before_the_zero_test_and_zero_position_raises():
    with pytest.raises(ValueError, match="finite"):
        lvlh_reference_velocity_m(LEO_R_M, (float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="r_ecef_m must be non-zero"):
        lvlh_reference_velocity_m((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_legacy_km_helper_keeps_zero_in_zero_out():
    assert lvlh_reference_velocity((6928.137, 0.0, 0.0), (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_inclined_geo_keeps_its_northward_component():
    r = (42164000.0, 0.0, 0.0)
    vrel = (0.0, 0.0, 50.0)  # Earth-relative motion through an ascending node
    v_ref = lvlh_reference_velocity_m(r, vrel)
    assert v_ref[2] == 50.0
    assert v_ref[1] == pytest.approx(OMEGA_EARTH_RAD_S * r[0], abs=1e-9)


# ---------------------------------------------------------------- the speed band edges (section 6.1.7)
# The band, computed here from its definition: from the apogee speed of an orbit whose
# perigee touches the 6371 km sphere, sqrt(mu (2/r - 2/(r + R))), to the escape speed
# sqrt(2 mu / r), with 2 % margins.
MU_KM3_S2 = 398600.4418
R_MEAN_KM = 6371.0
EDGE = 1e-9  # relative step either side of an edge; far above the float64 rounding


def band_km_s(r_km):
    return (
        math.sqrt(MU_KM3_S2 * (2.0 / r_km - 2.0 / (r_km + R_MEAN_KM))) * 0.98,
        math.sqrt(2.0 * MU_KM3_S2 / r_km) * 1.02,
    )


def vrel_for_speed_mps(r_m, speed_km_s):
    """An Earth-relative velocity at (r_m, 0, 0) whose v_ref is (0, 0, speed): omega x r is +y."""
    return (0.0, -OMEGA_EARTH_RAD_S * r_m, speed_km_s * 1000.0)


def edge_cases(r_km):
    lo, hi = band_km_s(r_km)
    return [
        (lo * (1.0 + EDGE), True),
        (lo * (1.0 - EDGE), False),
        (hi * (1.0 - EDGE), True),
        (hi * (1.0 + EDGE), False),
    ]


@pytest.mark.parametrize("r_km", [6928.137, 26571.0, 42164.0])
def test_vector_frame_velocity_band_just_inside_and_outside_each_edge(r_km):
    r_m = r_km * 1000.0
    for speed, accepted in edge_cases(r_km):
        vrel = vrel_for_speed_mps(r_m, speed)
        # The edge row is second, after an in-band row, so a per-row slip shows.
        r = np.array([LEO_R_M, (r_m, 0.0, 0.0)])
        v = np.array([LEO_VREL_MPS, vrel])
        if accepted:
            got = lvlh_reference_velocity_m_vec(r, v)
            assert np.linalg.norm(got[1]) / 1000.0 == pytest.approx(speed, rel=1e-12)
        else:
            with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
                lvlh_reference_velocity_m_vec(r, v)
        # The scalar m/s entry and the km/s check agree with the vector path.
        if accepted:
            lvlh_reference_velocity_m((r_m, 0.0, 0.0), vrel)
            check_inertial_speed_km_s((r_km, 0.0, 0.0), (0.0, 0.0, speed))
        else:
            with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
                lvlh_reference_velocity_m((r_m, 0.0, 0.0), vrel)
            with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
                check_inertial_speed_km_s((r_km, 0.0, 0.0), (0.0, 0.0, speed))


def test_km_s_velocity_given_as_m_s_is_refused_at_a_gps_like_radius():
    """3.87 km/s passed as 3.87 m/s at 26,571 km: the rebuilt speed is 1.94 km/s, below the band."""
    r = np.array([[26571e3, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 3.87]])
    lo, _ = band_km_s(26571.0)
    assert np.linalg.norm(v[0] + (0.0, OMEGA_EARTH_RAD_S * 26571e3, 0.0)) / 1000.0 < lo
    with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
        lvlh_reference_velocity_m_vec(r, v)


def test_a_radius_that_rounds_to_zero_fails_the_speed_check():
    with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
        check_inertial_speed_km_s((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=f"^{FRAME_VELOCITY_UNITS_ERROR}"):
        lvlh_reference_velocity_m_vec(np.array([[5e-324, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))



def test_zero_position_raises_even_against_an_empty_velocity():
    """Sections 6.1.1 and 6.1.7: a supplied zero position is refused before broadcasting."""
    for vrel in (np.empty((0, 3)), np.array([[1.0, 0.0, 0.0]]), (0.0, 0.0, 0.0)):
        with pytest.raises(ValueError, match="r_ecef_m must be non-zero"):
            lvlh_reference_velocity_m_vec((0.0, 0.0, 0.0), vrel)
    # Genuinely empty valid calls stay empty; a zero vrel is valid (the stationary branch).
    assert lvlh_reference_velocity_m_vec(np.empty((0, 3)), (0.0, 0.0, 0.0)).shape == (0, 3)
    assert lvlh_reference_velocity_m_vec(LEO_R_M, np.empty((0, 3))).shape == (0, 3)
