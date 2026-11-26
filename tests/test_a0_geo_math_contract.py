"""Conformance tests for the shared part of the frozen A0 API (A0 plan v7, Section 6.1).

Copied from the stand-in master (astra-shared branch ``a0-standin``,
``a0_standin/tests/test_a0_geo_math_contract.py``) with the stand-in modes
removed, as WP1 requires (plan Section 6.6): this copy tests only the real module.

Expected values come from published controls or from small independent
calculations in this file, never from the module under test.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from astra_shared import geo_math as gm

A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)
E2 = F * (2.0 - F)
OMEGA = 7.2921158553e-5


# --- independent references --------------------------------------------------
def ref_forward(lat_deg, lon_deg, h):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = A / math.sqrt(1.0 - E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + h) * math.cos(lat) * math.cos(lon),
            (n + h) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - E2) + h) * math.sin(lat),
        ]
    )


def ref_inverse(x, y, z):
    """Iterative inverse (fixed point on latitude), independent of Bowring."""
    p = math.hypot(x, y)
    lon = math.degrees(math.atan2(y, x))
    lat = math.atan2(z, p * (1.0 - E2))
    for _ in range(50):
        n = A / math.sqrt(1.0 - E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - E2 * n / (n + h)))
    n = A / math.sqrt(1.0 - E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return math.degrees(lat), lon, h


def ref_enu(lat_deg, lon_deg):
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    return east, north, up


def ref_nadir_hit_lat(sat_lat_deg, h):
    """Latitude where the geocentric nadir ray of a satellite at geodetic lat meets WGS84."""
    x, _, z = ref_forward(sat_lat_deg, 0.0, h)
    psi = math.atan2(z, x)  # geocentric latitude of the ray
    return math.degrees(math.atan(math.tan(psi) / (1.0 - E2)))


# --- names, signatures, constants ---------------------------------------------
SIGNATURES = {
    "geodetic_to_ecef": ["lat_deg", "lon_deg", "alt_m"],
    "ecef_to_geodetic": ["x", "y", "z"],  # today's names, kept for keyword callers
    "geodetic_to_ecef_vec": ["lat_deg", "lon_deg", "alt_m"],
    "ecef_to_geodetic_vec": ["x_m", "y_m", "z_m"],
    "enu_basis": ["lat_deg", "lon_deg"],
    "enu_basis_vec": ["lat_deg", "lon_deg"],
    "ecef_delta_to_enu": ["delta_m", "lat_deg", "lon_deg"],
    "ecef_delta_to_enu_vec": ["delta_m", "lat_deg", "lon_deg"],
    "look_angles_ecef": ["sat_m", "ground_m", "lat_deg", "lon_deg"],
    "look_angles_ecef_vec": ["sat_m", "ground_m", "lat_deg", "lon_deg"],
    "nadir_unit_vec": ["sat_m"],
    "off_axis_deg_vec": ["boresight_unit", "sat_m", "point_m"],
    "intersect_ray_wgs84": ["origin_m", "direction"],
    "intersect_ray_wgs84_vec": ["origin_m", "direction"],
    "lvlh_reference_velocity_m": ["r_ecef_m", "vrel_mps"],
    "lvlh_reference_velocity_m_vec": ["r_ecef_m", "vrel_mps"],
}


@pytest.mark.parametrize("name,params", sorted(SIGNATURES.items()))
def test_signature(name, params):
    assert list(inspect.signature(getattr(gm, name)).parameters) == params


def test_constants():
    assert gm.WGS84_A_M == A
    assert gm.WGS84_F == F
    assert gm.WGS84_B_M == pytest.approx(B, abs=1e-9)
    assert gm.WGS84_E2 == pytest.approx(E2, abs=1e-18)
    assert gm.OMEGA_EARTH_RAD_S == OMEGA


# --- converters ---------------------------------------------------------------
CASES = [
    (lat, lon, h)
    for lat in (-70.0, -45.0, 0.0, 45.0, 89.9)
    for lon in (-179.5, 0.0, 30.0)
    for h in (-1000.0, 0.0, 550e3, 35786e3)
]


@pytest.mark.parametrize("lat,lon,h", CASES)
def test_inverse_against_independent_iteration(lat, lon, h):
    x, y, z = ref_forward(lat, lon, h)
    lat_r, lon_r, h_r = ref_inverse(x, y, z)
    lat_v, lon_v, h_v = (float(a) for a in gm.ecef_to_geodetic_vec(x, y, z))
    lat_s, lon_s, h_s = gm.ecef_to_geodetic(x, y, z)
    for la, lo, hh in ((lat_v, lon_v, h_v), (lat_s, lon_s, h_s)):
        assert abs(la - lat_r) <= 1e-6
        assert abs(lo - lon_r) <= 1e-6
        assert abs(hh - h_r) <= 1e-6
        assert math.dist(ref_forward(la, lo, hh), (x, y, z)) <= 0.5


def test_forward_matches_reference_and_broadcasts():
    lat = np.array([[0.0], [45.0]])
    lon = np.array([0.0, 30.0, 180.0])
    x, y, z = gm.geodetic_to_ecef_vec(lat, lon, 550e3)
    assert x.shape == y.shape == z.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert np.allclose((x[i, j], y[i, j], z[i, j]), ref_forward(lat[i, 0], lon[j], 550e3), atol=1e-3, rtol=0)


def test_polar_axis_geocentre_and_longitude_fold():
    lat, lon, h = gm.ecef_to_geodetic(0.0, 0.0, B + 550e3)
    assert (lat, lon) == (90.0, 0.0) and h == pytest.approx(550e3, abs=1e-6)
    lat, lon, h = (float(a) for a in gm.ecef_to_geodetic_vec(0.0, 0.0, -(B + 10.0)))
    assert (lat, lon) == (-90.0, 0.0) and h == pytest.approx(10.0, abs=1e-6)
    assert gm.ecef_to_geodetic(0.0, 0.0, 0.0) == pytest.approx((90.0, 0.0, -B))
    with pytest.raises(ValueError):
        gm.ecef_to_geodetic_vec(0.0, 0.0, 0.0)
    # Longitude is in (-180, 180]: atan2(-0.0, -x) would give -180.
    assert float(gm.ecef_to_geodetic_vec(-7e6, -0.0, 0.0)[1]) == 180.0
    assert gm.ecef_to_geodetic(-7e6, -0.0, 0.0)[1] == 180.0


def test_scalar_converters_gain_no_new_errors():
    assert math.isnan(gm.ecef_to_geodetic(float("nan"), 1.0, 1.0)[0])
    gm.geodetic_to_ecef(91.0, 0.0, 0.0)  # no ValueError (Section 6.1.3)


@pytest.mark.parametrize("bad", [True, np.bool_(False), "1.0", None, 1j, float("nan"), float("inf")])
def test_vector_entries_reject_non_real_or_non_finite(bad):
    with pytest.raises(ValueError):
        gm.geodetic_to_ecef_vec(bad, 0.0, 0.0)
    if not isinstance(bad, (bool, np.bool_)):
        with pytest.raises(ValueError):
            gm.nadir_unit_vec([bad, 0.0, 7e6])


def test_bool_arrays_and_zero_positions_are_rejected():
    with pytest.raises(ValueError):
        gm.nadir_unit_vec(np.array([True, False, True]))
    with pytest.raises(ValueError):
        gm.nadir_unit_vec(np.zeros(3))
    with pytest.raises(ValueError):
        gm.lvlh_reference_velocity_m((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_shape_errors_and_empty():
    with pytest.raises(ValueError):
        gm.geodetic_to_ecef_vec(np.zeros(2), np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        gm.nadir_unit_vec(np.zeros((2, 2)) + 7e6)
    with pytest.raises(ValueError):
        gm.geodetic_to_ecef_vec([0.0, 95.0], 0.0, 0.0)  # one bad element fails the call
    x, _, _ = gm.geodetic_to_ecef_vec(np.zeros(0), np.zeros(0), 0.0)
    assert x.shape == (0,)
    out = gm.geodetic_to_ecef_vec(0.0, 0.0, 0.0)
    assert all(isinstance(a, np.ndarray) and a.shape == () for a in out)


def test_scalar_entries_reject_arrays_with_value_error():
    with pytest.raises(ValueError):
        gm.enu_basis([0.0, 1.0], 0.0)
    g = ref_forward(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        gm.look_angles_ecef(g + 1e6, g, [0.0, 1.0], 0.0)


def test_inputs_are_not_modified():
    lat = np.array([10.0, 20.0])
    lat.setflags(write=False)
    gm.geodetic_to_ecef_vec(lat, 0.0, 0.0)
    gm.enu_basis_vec(lat, 0.0)


# --- ENU and look angles --------------------------------------------------------
def test_iogp_enu_example():
    origin = (3652755.3058, 319574.6799, 5201547.3536)
    target = (3771793.968, 140253.342, 5124304.349)
    e, n, u = gm.ecef_delta_to_enu(np.subtract(target, origin), 55.0, 5.0)
    assert (e, n, u) == pytest.approx((-189013.869, -128642.040, -4220.171), abs=1e-3)


def test_enu_basis_is_geodetic():
    east, north, up = gm.enu_basis(45.0, 30.0)
    re, rn, ru = ref_enu(45.0, 30.0)
    assert np.allclose(east, re, atol=1e-15) and np.allclose(north, rn, atol=1e-15)
    assert np.allclose(up, ru, atol=1e-15)


def test_off_zenith_elevation_uses_geodetic_up():
    g = ref_forward(45.0, 30.0, 0.0)
    _, rn, ru = ref_enu(45.0, 30.0)
    sat = g + 1e6 * rn + 5e5 * ru
    rng, el, az = gm.look_angles_ecef(sat, g, 45.0, 30.0)
    assert el == pytest.approx(math.degrees(math.atan2(5e5, 1e6)), abs=1e-9)
    assert az == pytest.approx(0.0, abs=1e-9) or az == pytest.approx(360.0, abs=1e-9)
    assert 0.0 <= az < 360.0
    assert rng == pytest.approx(math.hypot(1e6, 5e5), abs=1e-6)


def test_zenith_nadir_and_radial_up_control():
    g = ref_forward(45.0, 30.0, 0.0)
    _, _, ru = ref_enu(45.0, 30.0)
    rng, el, az = gm.look_angles_ecef(g + 550e3 * ru, g, 45.0, 30.0)
    assert rng == pytest.approx(550e3, abs=1e-3) and el == 90.0 and az == 0.0
    _, el_down, az_down = gm.look_angles_ecef(g - 1e5 * ru, g, 45.0, 30.0)
    assert el_down == -90.0 and az_down == 0.0
    radial = g * (1.0 + 550e3 / np.linalg.norm(g))
    _, el_radial, _ = gm.look_angles_ecef(radial, g, 45.0, 30.0)
    assert 89.0 < el_radial < 89.9  # the radial direction is not the zenith


def test_azimuth_range_and_zero_range_error():
    g = ref_forward(0.0, 0.0, 0.0)
    _, _, az = gm.look_angles_ecef(g + np.array([1e5, 1e6, 0.0]), g, 0.0, 0.0)  # up, east
    assert az == pytest.approx(90.0, abs=1e-9)
    _, _, az = gm.look_angles_ecef(g + np.array([1e5, 0.0, -1e6]), g, 0.0, 0.0)  # up, south
    assert az == pytest.approx(180.0, abs=1e-9)
    # A tiny negative azimuth (just west of North) must stay in [0, 360), never 360.0.
    _, _, az = gm.look_angles_ecef(g + np.array([1e5, -1e-12, 1e6]), g, 0.0, 0.0)
    assert 0.0 <= az < 360.0
    with pytest.raises(ValueError):
        gm.look_angles_ecef(g, g, 0.0, 0.0)


# --- nadir, off-axis, intersection ------------------------------------------------
def test_nadir_is_geocentric_off_the_equator():
    sat = ref_forward(45.0, 30.0, 550e3)
    assert np.allclose(gm.nadir_unit_vec(sat), -sat / np.linalg.norm(sat), atol=1e-15)
    # From the satellite, the geodetic sub-point is about 0.177 deg off geocentric nadir.
    sub = ref_forward(45.0, 30.0, 0.0)
    psi = float(gm.off_axis_deg_vec(gm.nadir_unit_vec(sat), sat, sub))
    assert psi == pytest.approx(0.177124, abs=1e-4)


@pytest.mark.parametrize("h,expected", [(550e3, 45.015299598), (1200e3, 45.030513685)])
def test_nadir_point_appendix_b(h, expected):
    sat = ref_forward(45.0, 30.0, h)
    hit, dist = gm.intersect_ray_wgs84(sat, -sat)
    lat, lon, _ = gm.ecef_to_geodetic(*hit)
    assert lat == pytest.approx(expected, abs=1e-8)
    assert lat == pytest.approx(ref_nadir_hit_lat(45.0, h), abs=1e-9)
    assert lon == pytest.approx(30.0, abs=1e-9)
    assert dist == pytest.approx(np.linalg.norm(sat - np.array(hit)), abs=1e-6)
    resid = (hit[0] ** 2 + hit[1] ** 2) / A**2 + hit[2] ** 2 / B**2 - 1.0
    assert abs(resid) <= 1e-12


def test_ray_miss_and_strict_policy():
    o = (A + 550e3, 0.0, 0.0)
    assert gm.intersect_ray_wgs84(o, (0.0, 1.0, 0.0)) is None  # above the limb
    assert gm.intersect_ray_wgs84(o, (1.0, 0.0, 0.0)) is None  # pointing away
    hit, dist = gm.intersect_ray_wgs84(o, (-2.0, 0.0, 0.0))  # non-unit direction
    assert hit == pytest.approx((A, 0.0, 0.0), abs=1e-6) and dist == pytest.approx(550e3, abs=1e-6)
    assert gm.intersect_ray_wgs84((A, 0.0, 0.0), (1.0, 0.0, 0.0)) is None  # surface, outward
    hits, dists, ok = gm.intersect_ray_wgs84_vec(np.array([o, o]), np.array([[-1.0, 0, 0], [0, 1.0, 0]]))
    assert ok.tolist() == [True, False] and np.isnan(dists[1]) and np.all(np.isnan(hits[1]))
    with pytest.raises(ValueError):
        gm.intersect_ray_wgs84(o, (0.0, 0.0, 0.0))


def test_near_tangent_rays_are_classified_strictly():
    # In the equatorial plane the ellipsoid is a circle of radius A. The tangent
    # from (r0, 0, 0) leaves at angle beta0 = asin(A / r0) from -x towards +y.
    r0 = A + 550e3
    beta0 = math.asin(A / r0)
    # 1e-12 rad is outside the measured float64 band (about 1e-13 rad, plan §6.1.6 item 6).
    for delta, expect_hit in ((1e-9, False), (-1e-9, True), (1e-12, False), (-1e-12, True)):
        beta = beta0 + delta
        found = gm.intersect_ray_wgs84((r0, 0.0, 0.0), (-math.cos(beta), math.sin(beta), 0.0))
        assert (found is not None) is expect_hit


def test_inside_origin_returns_its_positive_root():
    hit, dist = gm.intersect_ray_wgs84((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert hit == pytest.approx((A, 0.0, 0.0), abs=1e-6) and dist == pytest.approx(A, abs=1e-6)


def test_off_axis_angles():
    sat = ref_forward(0.0, 0.0, 550e3)
    nadir = gm.nadir_unit_vec(sat)
    assert float(gm.off_axis_deg_vec(nadir, sat, np.zeros(3))) == pytest.approx(0.0, abs=1e-9)
    pt = sat + np.array([0.0, 1.0, 0.0]) * 1e5 + nadir * 1e5
    assert float(gm.off_axis_deg_vec(nadir, sat, pt)) == pytest.approx(45.0, abs=1e-9)
    assert float(gm.off_axis_deg_vec(nadir * 7.0, sat, pt)) == pytest.approx(45.0, abs=1e-9)
    with pytest.raises(ValueError):
        gm.off_axis_deg_vec(nadir, sat, sat)


# --- frame reference velocity ----------------------------------------------------
def test_stationary_rebuild_is_east_without_speed_check():
    r = ref_forward(0.0, 30.0, 550e3)  # fixed satellite at low altitude
    v = gm.lvlh_reference_velocity_m(r, (0.0, 0.0, 0.0))
    assert v == pytest.approx((-OMEGA * r[1], OMEGA * r[0], 0.0), rel=1e-15)


def test_wrong_unit_velocity_is_rejected():
    r = ref_forward(0.0, 0.0, 550e3)
    with pytest.raises(ValueError, match="antenna_frame_velocity_units"):
        gm.lvlh_reference_velocity_m(r, (0.0, 7.6, 0.0))  # km/s passed as m/s


def test_mixed_vector_call_checks_every_moving_row():
    r = np.array([ref_forward(0.0, 30.0, 550e3), ref_forward(0.0, 30.0, 550e3)])
    with pytest.raises(ValueError, match="antenna_frame_velocity_units"):
        gm.lvlh_reference_velocity_m_vec(r, np.array([[0.0, 0.0, 0.0], [0.0, 7.6, 0.0]]))
    good = gm.lvlh_reference_velocity_m_vec(r, np.array([[0.0, 0.0, 0.0], [0.0, 7600.0, 0.0]]))
    assert good.shape == (2, 3)
    assert good[0] == pytest.approx([-OMEGA * r[0, 1], OMEGA * r[0, 0], 0.0], rel=1e-15)


def test_off_axis_is_finite_on_boresight_and_at_180():
    rng = np.random.default_rng(7)
    lat = rng.uniform(-80.0, 80.0, 2000)
    lon = rng.uniform(-180.0, 180.0, 2000)
    sats = np.stack(gm.geodetic_to_ecef_vec(lat, lon, 550e3), axis=-1)
    nadir = gm.nadir_unit_vec(sats)
    hits, _, ok = gm.intersect_ray_wgs84_vec(sats, nadir)
    assert ok.all()
    psi = gm.off_axis_deg_vec(nadir, sats, hits)
    assert np.all(np.isfinite(psi)) and np.max(psi) < 1e-9
    sat = sats[0]
    assert float(gm.off_axis_deg_vec(nadir[0], sat, sat - 1e6 * nadir[0])) == pytest.approx(180.0, abs=1e-9)


def test_propagated_geo_with_small_velocity_is_accepted():
    r = np.array([42164e3, 0.0, 0.0])
    v = gm.lvlh_reference_velocity_m(r, (0.1, -0.2, 0.05))
    assert v == pytest.approx((0.1, OMEGA * 42164e3 - 0.2, 0.05), rel=1e-12)


def test_scalar_entries_take_numpy_3_vectors_and_refuse_other_arrays():
    sat = np.array(ref_forward(0.0, 0.0, 550e3))
    ground = np.array(ref_forward(0.0, 0.0, 0.0))
    r, el, _ = gm.look_angles_ecef(sat, ground, 0.0, 0.0)  # NumPy (3,) is a 3-vector
    assert r == pytest.approx(550e3, abs=1e-6) and el == pytest.approx(90.0, abs=1e-9)
    assert gm.ecef_to_geodetic(x=float(ground[0]), y=0.0, z=0.0)[2] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        gm.look_angles_ecef(sat[None, :], ground, 0.0, 0.0)  # shape (1, 3)
    with pytest.raises(ValueError):
        gm.look_angles_ecef(sat, ground, np.array([0.0]), 0.0)  # an angle array
