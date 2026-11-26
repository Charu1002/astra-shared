"""A0 WGS84 geometry: G01 (conversions), G02 (shapes) and G03 (local frame and look angles).

A0 plan v7: API in section 6.1, tests in section 10.3, tolerances in section 10.2.
Expected values come from tests/wgs84_reference.py, the IOGP 373-07-2 published
example and Appendix B of the plan, never from the functions under test.
"""

from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
import pytest
import wgs84_reference as ref

from astra_shared import geo_math
from astra_shared.geo_math import (
    OMEGA_EARTH_RAD_S,
    WGS84_A_M,
    WGS84_B_M,
    WGS84_E2,
    WGS84_F,
    ecef_delta_to_enu,
    ecef_delta_to_enu_vec,
    ecef_to_geodetic,
    ecef_to_geodetic_vec,
    enu_basis,
    enu_basis_vec,
    geodetic_to_ecef,
    geodetic_to_ecef_vec,
    look_angles_ecef,
    look_angles_ecef_vec,
    lvlh_reference_velocity_m,
    lvlh_reference_velocity_m_vec,
    nadir_unit_vec,
    off_axis_deg_vec,
)

FORWARD_TOL_M = 1e-3
ANGLE_TOL_DEG = 1e-6
HEIGHT_TOL_M = 1e-6
ROUND_TRIP_TOL_M = 0.5

LATS = (-90.0, -70.0, -45.0, 0.0, 45.0, 70.0, 90.0)
LONS = (-180.0, -45.0, 0.0, 180.0)
HEIGHTS = (-1000.0, 0.0, 550000.0, 1050000.0, 1200000.0, 35786000.0, 42000000.0)
GRID = list(itertools.product(LATS, LONS, HEIGHTS))


def circular_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------- G01 constants


def test_g01_canonical_constants():
    assert WGS84_A_M == 6378137.0
    assert WGS84_F == 1 / 298.257223563
    assert WGS84_B_M == WGS84_A_M * (1 - WGS84_F)
    assert WGS84_E2 == WGS84_F * (2 - WGS84_F)
    assert WGS84_B_M == pytest.approx(6356752.314245, abs=1e-6)
    assert OMEGA_EARTH_RAD_S == 7.2921158553e-5


def test_g01_e2_changes_by_about_1_3e_15_from_the_old_literal():
    # Section 6.1.2: no private alias is required; existing private names, if kept, carry the same value.
    assert abs(WGS84_E2 - 6.69437999014e-3) < 2e-15
    assert getattr(geo_math, "_E2", WGS84_E2) == WGS84_E2


# ---------------------------------------------------------------- G01 forward and inverse


@pytest.mark.parametrize(("lat", "lon", "h"), GRID)
def test_g01_forward_matches_independent_forward(lat, lon, h):
    expected = ref.forward(lat, lon, h)
    assert math.dist(geodetic_to_ecef(lat, lon, h), expected) <= FORWARD_TOL_M
    vec = geodetic_to_ecef_vec(lat, lon, h)
    assert math.dist([float(c) for c in vec], expected) <= FORWARD_TOL_M


@pytest.mark.parametrize(("lat", "lon", "h"), GRID)
def test_g01_inverse_matches_independent_inverse_and_round_trips(lat, lon, h):
    xyz = ref.forward(lat, lon, h)
    ref_lat, ref_lon, ref_h = ref.inverse(*xyz)
    for got in (ecef_to_geodetic(*xyz), tuple(float(v) for v in ecef_to_geodetic_vec(*xyz))):
        got_lat, got_lon, got_h = got
        assert abs(got_lat - ref_lat) <= ANGLE_TOL_DEG
        if abs(ref_lat) < 89.999:
            assert circular_deg(got_lon, ref_lon) <= ANGLE_TOL_DEG
        assert abs(got_h - ref_h) <= HEIGHT_TOL_M
        # Separate check: forward conversion of the result reproduces the input ECEF.
        assert math.dist(ref.forward(got_lat, got_lon, got_h), xyz) <= ROUND_TRIP_TOL_M
        assert -180.0 < got_lon <= 180.0


@pytest.mark.parametrize("lat", [89.9999999, -89.9999999, 89.999999999, -89.999999999])
@pytest.mark.parametrize("h", [0.0, 550000.0, 35786000.0])
def test_g01_near_poles(lat, h):
    xyz = ref.forward(lat, 30.0, h)
    ref_lat, _, ref_h = ref.inverse(*xyz)
    got_lat, _, got_h = ecef_to_geodetic(*xyz)
    assert abs(got_lat - ref_lat) <= ANGLE_TOL_DEG
    assert abs(got_h - ref_h) <= HEIGHT_TOL_M


@pytest.mark.parametrize("h", [-1000.0, 0.0, 550000.0, 35786000.0])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_g01_exact_poles(h, sign):
    z = sign * (WGS84_B_M + h)
    expected = (sign * 90.0, 0.0, h)
    assert ecef_to_geodetic(0.0, 0.0, z) == pytest.approx(expected, abs=1e-9)
    lat, lon, alt = ecef_to_geodetic_vec(0.0, 0.0, z)
    assert (float(lat), float(lon), float(alt)) == pytest.approx(expected, abs=1e-9)


def test_g01_exact_pole_height_was_wrong_before():
    # Today's formula p / cos(lat) - N gave -6,399,593.6 m here (plan section 6.1.3).
    assert ecef_to_geodetic(0.0, 0.0, WGS84_B_M + 550000.0)[2] == pytest.approx(550000.0, abs=1e-6)


@pytest.mark.parametrize("offset_m", [1e-3, 1e-6])
def test_g01_ecef_radii_just_off_the_axis(offset_m):
    xyz = (offset_m, 0.0, WGS84_B_M + 550000.0)
    lat, lon, h = ecef_to_geodetic(*xyz)
    ref_lat, _, ref_h = ref.inverse(*xyz)
    assert abs(lat - ref_lat) <= ANGLE_TOL_DEG
    assert abs(h - ref_h) <= HEIGHT_TOL_M
    assert lon == 0.0


def test_g01_geocentre():
    with pytest.raises(ValueError, match="geocentre"):
        ecef_to_geodetic_vec(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="geocentre"):
        ecef_to_geodetic_vec([7e6, 0.0], [0.0, 0.0], [0.0, 0.0])
    # The scalar form gains no error: the polar-axis branch gives (90, 0, -b).
    assert ecef_to_geodetic(0.0, 0.0, 0.0) == (90.0, 0.0, -WGS84_B_M)


def test_g01_longitude_minus_180_folds_to_180():
    assert ecef_to_geodetic(-7e6, -0.0, 0.0)[1] == 180.0
    assert float(ecef_to_geodetic_vec(-7e6, -0.0, 0.0)[1]) == 180.0


def test_g01_geo_latitude_witness_is_inside_the_allowance():
    # Plan section 10.2: at GEO the closed-form latitude is off by about 3.5e-7 deg,
    # inside the 1e-6 deg allowance, yet enough to change a 4-decimal token. The
    # published-token check therefore belongs to the same-row gate (P02), not here.
    xyz = ref.forward(45.0000498, 30.0, 35786000.0)
    lat = ecef_to_geodetic(*xyz)[0]
    assert 1e-7 < abs(lat - 45.0000498) <= ANGLE_TOL_DEG
    assert f"{45.0000498:.4f}" == "45.0000"
    assert f"{lat:.4f}" == "45.0001"


def test_g01_mixed_vector_matches_scalar_elementwise():
    xs = np.array([0.0, 7.0e6, 0.0, -4.0e6, 1e-3])
    ys = np.array([0.0, 1.0e6, 0.0, 3.0e6, 0.0])
    zs = np.array([WGS84_B_M + 550e3, 2.0e6, -(WGS84_B_M + 1.0), -5.0e6, WGS84_B_M])
    lat, lon, h = ecef_to_geodetic_vec(xs, ys, zs)
    for i in range(xs.size):
        s_lat, s_lon, s_h = ecef_to_geodetic(xs[i], ys[i], zs[i])
        assert abs(lat[i] - s_lat) <= ANGLE_TOL_DEG
        assert circular_deg(lon[i], s_lon) <= ANGLE_TOL_DEG
        assert abs(h[i] - s_h) <= FORWARD_TOL_M


LARGE_LON = 3.6e11  # 1e9 turns; reduces exactly to 0


def test_g01_strict_entries_reduce_a_large_longitude_exactly():
    # Section 6.1.3: fmod then a 360 shift, before the trigonometry.
    lat, h = 45.0, 550000.0
    reduced = math.fmod(LARGE_LON, 360.0)
    assert [float(v) for v in geodetic_to_ecef_vec(lat, LARGE_LON, h)] == [
        float(v) for v in geodetic_to_ecef_vec(lat, reduced, h)
    ]
    for got, expected in zip(enu_basis_vec(lat, LARGE_LON), enu_basis_vec(lat, reduced)):
        np.testing.assert_array_equal(got, expected)
    assert enu_basis(lat, LARGE_LON) == enu_basis(lat, reduced)
    delta = (1.0e5, -2.0e5, 3.0e5)
    assert ecef_delta_to_enu(delta, lat, LARGE_LON) == ecef_delta_to_enu(delta, lat, reduced)
    ground = geodetic_to_ecef(lat, reduced, 0.0)
    sat = geodetic_to_ecef(lat + 1.0, reduced + 1.0, h)
    assert look_angles_ecef(sat, ground, lat, LARGE_LON) == look_angles_ecef(sat, ground, lat, reduced)
    # Against the independent forward of the reduced value too.
    assert math.dist([float(v) for v in geodetic_to_ecef_vec(lat, LARGE_LON, h)], ref.forward(lat, 0.0, h)) <= 1e-3


@pytest.mark.parametrize(("lon", "reduced"), [(540.0, 180.0), (-540.0, 180.0), (190.0, -170.0), (-190.0, 170.0)])
def test_g01_reduction_lands_in_minus_180_to_180(lon, reduced):
    got = [float(v) for v in geodetic_to_ecef_vec(10.0, lon, 0.0)]
    assert got == [float(v) for v in geodetic_to_ecef_vec(10.0, reduced, 0.0)]


def test_g01_legacy_scalar_forward_keeps_unreduced_trigonometry():
    # The exception to scalar/vector parity (section 6.1.3): about 0.42 m at 3.6e11 deg.
    gap = math.dist(geodetic_to_ecef(45.0, LARGE_LON, 0.0), [float(v) for v in geodetic_to_ecef_vec(45.0, LARGE_LON, 0.0)])
    assert 0.0 < gap < 1.0


def test_g01_vector_forward_rejects_latitude_out_of_range():
    for bad in (90.0000001, -90.0000001, [0.0, 91.0]):
        with pytest.raises(ValueError, match="lat_deg"):
            geodetic_to_ecef_vec(bad, 0.0, 0.0)


def test_g01_scalar_converters_gain_no_new_errors():
    nan, inf = float("nan"), float("inf")
    assert all(math.isnan(v) for v in ecef_to_geodetic(nan, 0.0, 0.0))
    assert not all(math.isfinite(v) for v in ecef_to_geodetic(inf, 0.0, 0.0))
    assert not all(math.isfinite(v) for v in geodetic_to_ecef(0.0, 0.0, nan))
    assert not all(math.isfinite(v) for v in geodetic_to_ecef(0.0, 0.0, inf))
    with pytest.raises(ValueError):  # from math.sin, as before A0
        geodetic_to_ecef(inf, 0.0, 0.0)
    # Out-of-range latitude is not rejected by the scalar forward.
    geodetic_to_ecef(91.0, 0.0, 0.0)


def test_g01_scalar_signatures_are_unchanged():
    assert list(inspect.signature(ecef_to_geodetic).parameters) == ["x", "y", "z"]
    assert list(inspect.signature(geodetic_to_ecef).parameters) == ["lat_deg", "lon_deg", "alt_m"]


# ---------------------------------------------------------------- G02 shapes and inputs


def test_g02_scalar_input_gives_0d_arrays():
    for out in geodetic_to_ecef_vec(45.0, 30.0, 0.0):
        assert isinstance(out, np.ndarray) and out.shape == () and out.dtype == np.float64
    for out in ecef_to_geodetic_vec(7e6, 0.0, 0.0):
        assert out.shape == ()


def test_g02_zero_d_and_one_d_and_broadcast_shapes():
    assert geodetic_to_ecef_vec(np.array(45.0), 30.0, 0.0)[0].shape == ()
    assert geodetic_to_ecef_vec([0.0, 45.0, 70.0], 30.0, 0.0)[0].shape == (3,)
    x, y, z = geodetic_to_ecef_vec(np.zeros((2, 1)), np.zeros((1, 3)), 0.0)
    assert x.shape == y.shape == z.shape == (2, 3)
    east, north, up = enu_basis_vec(np.zeros((2, 1)), np.zeros((1, 3)))
    assert east.shape == north.shape == up.shape == (2, 3, 3)


def test_g02_empty_inputs_give_empty_outputs():
    x, _, _ = geodetic_to_ecef_vec(np.empty(0), np.empty(0), np.empty(0))
    assert x.shape == (0,)
    lat, _, _ = ecef_to_geodetic_vec(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)))
    assert lat.shape == (0, 2)
    r, el, az = look_angles_ecef_vec(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0))
    assert r.shape == el.shape == az.shape == (0,)


def test_g02_vector_matches_scalar():
    lats = np.array([-70.0, -45.0, 0.0, 45.0, 70.0])
    lons = np.array([-45.0, 0.0, 30.0, 120.0, 179.0])
    hs = np.array([0.0, 550e3, 1200e3, 35786e3, -1000.0])
    xs, ys, zs = geodetic_to_ecef_vec(lats, lons, hs)
    for i in range(lats.size):
        assert math.dist((xs[i], ys[i], zs[i]), geodetic_to_ecef(lats[i], lons[i], hs[i])) <= FORWARD_TOL_M


def test_g02_read_only_and_noncontiguous_inputs_are_not_mutated():
    base = np.linspace(-60.0, 60.0, 10)
    lats = base[::2]  # non-contiguous view
    lons = np.full(5, 30.0)
    lons.setflags(write=False)
    before = (lats.copy(), lons.copy())
    x, _, _ = geodetic_to_ecef_vec(lats, lons, 0.0)
    assert x.shape == (5,)
    np.testing.assert_array_equal(lats, before[0])
    np.testing.assert_array_equal(lons, before[1])
    sat = np.array([[7e6, 0.0, 0.0], [0.0, 7e6, 0.0]])
    sat_copy = sat.copy()
    look_angles_ecef_vec(sat, np.array([6378137.0, 0.0, 0.0]), 0.0, 0.0)
    np.testing.assert_array_equal(sat, sat_copy)


@pytest.mark.parametrize(
    "bad",
    [True, np.bool_(True), np.array([True, False]), "45", None, 1 + 2j, np.array([1, "a"], dtype=object)],
    ids=["bool", "numpy-bool", "bool-array", "string", "none", "complex", "object-array"],
)
def test_g02_non_real_inputs_raise_value_error(bad):
    with pytest.raises(ValueError, match="lat_deg"):
        geodetic_to_ecef_vec(bad, 0.0, 0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), [0.0, float("nan")]])
def test_g02_non_finite_inputs_raise_value_error(bad):
    with pytest.raises(ValueError, match="finite"):
        geodetic_to_ecef_vec(0.0, bad, 0.0)


def test_g02_bad_final_axis_and_incompatible_shapes():
    with pytest.raises(ValueError, match="final axis"):
        look_angles_ecef_vec(np.zeros((2, 2)), np.zeros(3), 0.0, 0.0)
    with pytest.raises(ValueError, match="incompatible"):
        geodetic_to_ecef_vec(np.zeros(2), np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="incompatible"):
        look_angles_ecef_vec(np.full((2, 3), 7e6), np.full((3, 3), 6.4e6), 0.0, 0.0)


def test_g02_mixed_valid_and_invalid_batch_fails_whole_call():
    with pytest.raises(ValueError):
        geodetic_to_ecef_vec([0.0, 45.0, 95.0], 0.0, 0.0)
    with pytest.raises(ValueError):
        ecef_to_geodetic_vec([7e6, 0.0], [0.0, 0.0], [0.0, 0.0])


V3 = (7.0e6, 1.0e6, 2.0e6)
G3 = (6378137.0, 0.0, 0.0)
NAN = float("nan")

# Each entry with the argument position that carries a component (angle) and one that carries a 3-vector.
ENTRY_CALLS = {
    "geodetic_to_ecef_vec": (geodetic_to_ecef_vec, lambda bad: (bad, 0.0, 0.0), "lat_deg"),
    "ecef_to_geodetic_vec": (ecef_to_geodetic_vec, lambda bad: (bad, 0.0, 0.0), "x_m"),
    "enu_basis": (enu_basis, lambda bad: (bad, 0.0), "lat_deg"),
    "enu_basis_vec": (enu_basis_vec, lambda bad: (bad, 0.0), "lat_deg"),
    "ecef_delta_to_enu": (ecef_delta_to_enu, lambda bad: (V3, 0.0, bad), "lon_deg"),
    "ecef_delta_to_enu_vec": (ecef_delta_to_enu_vec, lambda bad: (V3, 0.0, bad), "lon_deg"),
    "look_angles_ecef": (look_angles_ecef, lambda bad: (V3, G3, bad, 0.0), "lat_deg"),
    "look_angles_ecef_vec": (look_angles_ecef_vec, lambda bad: (V3, G3, bad, 0.0), "lat_deg"),
}
VECTOR_CALLS = {
    "ecef_delta_to_enu": (ecef_delta_to_enu, lambda bad: (bad, 0.0, 0.0), "delta_m"),
    "look_angles_ecef_vec": (look_angles_ecef_vec, lambda bad: (bad, G3, 0.0, 0.0), "sat_m"),
    "nadir_unit_vec": (nadir_unit_vec, lambda bad: (bad,), "sat_m"),
    "off_axis_deg_vec": (off_axis_deg_vec, lambda bad: ((1.0, 0.0, 0.0), V3, bad), "point_m"),
    "lvlh_reference_velocity_m": (lvlh_reference_velocity_m, lambda bad: (bad, (0.0, 0.0, 0.0)), "r_ecef_m"),
    "lvlh_reference_velocity_m_vec": (lvlh_reference_velocity_m_vec, lambda bad: (V3, bad), "vrel_mps"),
}


@pytest.mark.parametrize("entry", sorted(ENTRY_CALLS))
@pytest.mark.parametrize("bad", [True, np.bool_(False), NAN, float("inf")], ids=["bool", "numpy-bool", "nan", "inf"])
def test_g02_every_component_entry_rejects_bool_and_non_finite(entry, bad):
    func, args, name = ENTRY_CALLS[entry]
    with pytest.raises(ValueError, match=name):
        func(*args(bad))


@pytest.mark.parametrize("entry", sorted(VECTOR_CALLS))
@pytest.mark.parametrize(
    "bad", [np.array([True, False, True]), (NAN, 0.0, 1.0), (0.0, float("inf"), 1.0)], ids=["bool-array", "nan", "inf"]
)
def test_g02_every_cartesian_entry_rejects_bool_and_non_finite(entry, bad):
    func, args, name = VECTOR_CALLS[entry]
    with pytest.raises(ValueError, match=name):
        func(*args(bad))


LATITUDE_CALLS = {
    "geodetic_to_ecef_vec": (geodetic_to_ecef_vec, lambda lat: (lat, 0.0, 0.0)),
    "enu_basis": (enu_basis, lambda lat: (lat, 0.0)),
    "enu_basis_vec": (enu_basis_vec, lambda lat: (lat, 0.0)),
    "ecef_delta_to_enu": (ecef_delta_to_enu, lambda lat: (V3, lat, 0.0)),
    "ecef_delta_to_enu_vec": (ecef_delta_to_enu_vec, lambda lat: (V3, lat, 0.0)),
    "look_angles_ecef": (look_angles_ecef, lambda lat: (V3, G3, lat, 0.0)),
    "look_angles_ecef_vec": (look_angles_ecef_vec, lambda lat: (V3, G3, lat, 0.0)),
}


@pytest.mark.parametrize("entry", sorted(LATITUDE_CALLS))
@pytest.mark.parametrize("lat", [90.000001, -90.000001])
def test_g02_every_latitude_entry_rejects_out_of_range(entry, lat):
    func, args = LATITUDE_CALLS[entry]
    with pytest.raises(ValueError, match=r"lat_deg must be within \[-90, 90\]"):
        func(*args(lat))


def test_g02_zero_boresight_and_non_finite_ray_origin():
    with pytest.raises(ValueError, match="boresight_unit must be non-zero"):
        off_axis_deg_vec((0.0, 0.0, 0.0), V3, G3)
    from astra_shared.geo_math import intersect_ray_wgs84, intersect_ray_wgs84_vec

    with pytest.raises(ValueError, match="origin_m must be finite"):
        intersect_ray_wgs84((NAN, 0.0, 0.0), (-1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="origin_m must be finite"):
        intersect_ray_wgs84_vec(np.array([[7e6, 0.0, 0.0], [float("inf"), 0.0, 0.0]]), (-1.0, 0.0, 0.0))


def test_g02_scalar_entries_reject_arrays():
    with pytest.raises(ValueError, match="scalar"):
        enu_basis([0.0, 1.0], 0.0)
    with pytest.raises(ValueError, match="3-vector"):
        look_angles_ecef(np.zeros((2, 3)), (6378137.0, 0.0, 0.0), 0.0, 0.0)
    east, north, up = enu_basis(45.0, 30.0)
    assert all(isinstance(v, float) for v in east + north + up)


# ---------------------------------------------------------------- G03 ENU and look angles

IOGP_ORIGIN = (3652755.3058, 319574.6799, 5201547.3536)
IOGP_TARGET = (3771793.968, 140253.342, 5124304.349)
IOGP_ENU = (-189013.869, -128642.040, -4220.171)


def test_g03_iogp_published_enu_control():
    delta = np.subtract(IOGP_TARGET, IOGP_ORIGIN)
    assert ecef_delta_to_enu(delta, 55.0, 5.0) == pytest.approx(IOGP_ENU, abs=1e-3)
    got = ecef_delta_to_enu_vec(delta, 55.0, 5.0)
    assert tuple(float(v) for v in got) == pytest.approx(IOGP_ENU, abs=1e-3)


@pytest.mark.parametrize(("lat", "lon"), [(0.0, 0.0), (45.0, 30.0), (-70.0, -120.0), (90.0, 0.0), (-90.0, 45.0)])
def test_g03_basis_is_orthonormal_right_handed_and_matches_reference(lat, lon):
    east, north, up = (np.array(v) for v in enu_basis(lat, lon))
    for got, expected in zip((east, north, up), ref.enu_axes(lat, lon)):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for v in (east, north, up):
        assert abs(np.linalg.norm(v) - 1.0) <= 1e-12
    assert abs(east @ north) <= 1e-12 and abs(north @ up) <= 1e-12 and abs(east @ up) <= 1e-12
    np.testing.assert_allclose(np.cross(east, north), up, atol=1e-12)


def test_g03_up_is_the_geodetic_normal_not_the_radial_direction():
    up = np.array(enu_basis(45.0, 30.0)[2])
    radial = np.array(ref.forward(45.0, 30.0, 0.0))
    radial /= np.linalg.norm(radial)
    gap_deg = math.degrees(math.acos(float(np.clip(up @ radial, -1.0, 1.0))))
    assert 0.19 < gap_deg < 0.20  # geodetic minus geocentric latitude at 45 deg


def _sat_from_enu(lat, lon, h_ground, enu):
    ground = np.array(ref.forward(lat, lon, h_ground))
    east, north, up = (np.array(v) for v in ref.enu_axes(lat, lon))
    return ground + enu[0] * east + enu[1] * north + enu[2] * up, ground


@pytest.mark.parametrize(
    ("enu", "az"), [((0, 1e6, 1e5), 0.0), ((1e6, 0, 1e5), 90.0), ((0, -1e6, 1e5), 180.0), ((-1e6, 0, 1e5), 270.0)]
)
def test_g03_cardinal_directions(enu, az):
    sat, ground = _sat_from_enu(30.0, 40.0, 0.0, enu)
    rng, el, got_az = look_angles_ecef(sat, ground, 30.0, 40.0)
    assert circular_deg(got_az, az) <= 1e-6
    assert el == pytest.approx(math.degrees(math.atan2(1e5, 1e6)), abs=1e-6)
    assert rng == pytest.approx(math.hypot(1e6, 1e5), abs=1e-3)


@pytest.mark.parametrize("h_ground", [0.0, 1000.0])
@pytest.mark.parametrize(("up", "el"), [(550e3, 90.0), (-550e3, -90.0)])
def test_g03_vertical_line_of_sight(h_ground, up, el):
    sat, ground = _sat_from_enu(45.0, 30.0, h_ground, (0.0, 0.0, up))
    rng, got_el, got_az = look_angles_ecef(sat, ground, 45.0, 30.0)
    assert got_el == el and got_az == 0.0
    assert rng == pytest.approx(abs(up), abs=1e-3)


def test_g03_horizon_is_zero_elevation_without_a_gate():
    sat, ground = _sat_from_enu(45.0, 30.0, 0.0, (1e6, 0.0, 0.0))
    assert look_angles_ecef(sat, ground, 45.0, 30.0)[1] == pytest.approx(0.0, abs=1e-9)
    below, _ = _sat_from_enu(45.0, 30.0, 0.0, (1e6, 0.0, -1e5))
    assert look_angles_ecef(below, ground, 45.0, 30.0)[1] < 0.0


@pytest.mark.parametrize("east", [0.0, -0.0, -1e-300, -4e-10, -1e-6, 1e-12])
def test_g03_azimuth_is_in_zero_to_360(east):
    # At latitude 0, longitude 0 the axes are exactly E = +y, N = +z, U = +x.
    ground = (WGS84_A_M, 0.0, 0.0)
    _, _, az = look_angles_ecef((WGS84_A_M + 1e5, east * 1e6, 1e6), ground, 0.0, 0.0)
    az_vec = float(look_angles_ecef_vec((WGS84_A_M + 1e5, east * 1e6, 1e6), ground, 0.0, 0.0)[2])
    for value in (az, az_vec):
        assert 0.0 <= value < 360.0
        assert math.copysign(1.0, value) == 1.0
    if east == -1e-6:
        assert az == pytest.approx(360.0 - math.degrees(1e-6), abs=1e-9)


def test_g03_zero_range_raises():
    ground = ref.forward(10.0, 20.0, 0.0)
    with pytest.raises(ValueError, match="zero range"):
        look_angles_ecef(ground, ground, 10.0, 20.0)


@pytest.mark.parametrize(
    ("lat", "el", "az", "expected_range"),
    [
        (0.0, 10.0, 0.0, 1812271.614),
        (0.0, 30.0, 0.0, 992322.281),
        (0.0, 90.0, 0.0, 550000.000),
        (45.0, 10.0, 90.0, 1816481.580),
        (45.0, 30.0, 90.0, 993005.710),
        (45.0, 90.0, 90.0, 550000.000),
        (70.0, 10.0, 225.0, 1816718.493),
        (70.0, 30.0, 225.0, 993062.500),
        (70.0, 90.0, 225.0, 550000.000),
    ],
)
def test_g03_appendix_b_slant_ranges(lat, el, az, expected_range):
    sat, _ = ref.satellite_at(lat, 30.0, el, az, 550000.0)
    ground = ref.forward(lat, 30.0, 0.0)
    rng, got_el, got_az = look_angles_ecef(sat, ground, lat, 30.0)
    assert rng == pytest.approx(expected_range, abs=1e-3)  # Appendix B is rounded to the millimetre
    assert got_el == pytest.approx(el, abs=1e-6)
    if el < 90.0:
        assert circular_deg(got_az, az) <= 1e-6
    vec = look_angles_ecef_vec(np.array(sat), np.array(ground), lat, 30.0)
    assert abs(float(vec[0]) - rng) <= 1e-3 and abs(float(vec[1]) - got_el) <= 1e-6


# ---------------------------------------------------------------- the vertical threshold (section 6.1.4)
# At latitude 0, longitude 0 the axes are exactly E = +y, N = +z, U = +x, so a satellite at
# ground + (1, h, 0) m has range sqrt(1 + h^2) = 1.0 in float64 and a horizontal part of h.
VERTICAL_GROUND = (WGS84_A_M, 0.0, 0.0)


@pytest.mark.parametrize("up", [1.0, -1.0])
@pytest.mark.parametrize(
    ("h", "vertical"),
    [
        (0.9e-12, True),  # a little under 1e-12 of the range
        (1e-12, True),  # exactly 1e-12 of the range: "at most" is inclusive
        (1.1e-12, False),  # a little over
        (1e-9, False),
    ],
)
def test_g03_vertical_threshold_on_each_side_and_at_the_boundary(h, vertical, up):
    sat = (WGS84_A_M + up, h, 0.0)
    expected_el = math.degrees(math.atan2(up, h))  # the ordinary angles, from E = h, N = 0
    scalar = look_angles_ecef(sat, VERTICAL_GROUND, 0.0, 0.0)
    vector = tuple(float(v) for v in look_angles_ecef_vec(np.array(sat), np.array(VERTICAL_GROUND), 0.0, 0.0))
    for rng, el, az in (scalar, vector):
        assert rng == 1.0
        if vertical:
            assert el == math.copysign(90.0, up) and az == 0.0
        else:
            assert el == pytest.approx(expected_el, abs=1e-12) and el != math.copysign(90.0, up)
            assert az == 90.0


def test_g03_vertical_threshold_in_one_vector_call():
    hs = np.array([0.9e-12, 1e-12, 1.1e-12])
    sat = np.stack([np.full(3, WGS84_A_M + 1.0), hs, np.zeros(3)], axis=-1)
    _, el, az = look_angles_ecef_vec(sat, VERTICAL_GROUND, 0.0, 0.0)
    np.testing.assert_array_equal(az, [0.0, 0.0, 90.0])
    assert el[0] == 90.0 and el[1] == 90.0 and el[2] < 90.0


def test_g03_near_overhead_pass_keeps_its_angles():
    """A satellite 0.15 deg off the zenith of a 45 deg N ground point is not reported vertical."""
    sat, _ = ref.satellite_at(45.0, 30.0, 89.85, 120.0, 550e3)
    ground = ref.forward(45.0, 30.0, 0.0)
    _, el, az = look_angles_ecef(sat, ground, 45.0, 30.0)
    assert el == pytest.approx(89.85, abs=1e-6)
    assert circular_deg(az, 120.0) <= 1e-6


# ---------------------------------------------------------------- overflow and underflow (section 6.1.4)
@pytest.mark.parametrize("scale", [1e-170, 1e-160, 1e160, 1e200, 1e300])
def test_g03_look_angles_at_extreme_finite_distances(scale):
    # At latitude 0, longitude 0 from the geocentre: E = dy, N = dz, U = dx.
    sat = (scale, scale, 0.0)
    for rng, el, az in (
        look_angles_ecef(sat, (0.0, 0.0, 0.0), 0.0, 0.0),
        tuple(float(v) for v in look_angles_ecef_vec(np.array(sat), np.zeros(3), 0.0, 0.0)),
    ):
        assert rng == pytest.approx(math.sqrt(2.0) * scale, rel=1e-15)
        assert el == pytest.approx(45.0, abs=1e-12)
        assert az == pytest.approx(90.0, abs=1e-12)


def test_g01_scalar_inverse_nan_inputs_behave_as_today():
    nan, inf = float("nan"), float("inf")
    # A NaN in x or y gives NaN in all three outputs.
    for xyz in ((nan, 0.0, 0.0), (0.0, nan, 0.0), (inf, nan, 1.0), (nan, 7e6, nan)):
        assert all(math.isnan(v) for v in ecef_to_geodetic(*xyz)), xyz
    # A NaN in z alone keeps the computable longitude.
    for xyz, lon in (
        ((6378137.0, 0.0, nan), 0.0),
        ((-7e6, 1e6, nan), math.degrees(math.atan2(1e6, -7e6))),  # 171.87 deg
        ((0.0, 0.0, nan), 0.0),
        ((inf, inf, nan), 45.0),
        ((-7e6, -0.0, nan), 180.0),  # with the section 6.1.3 fold of -180 to 180
    ):
        lat, got_lon, h = ecef_to_geodetic(*xyz)
        assert math.isnan(lat) and math.isnan(h) and got_lon == lon, xyz
    assert ecef_to_geodetic(-7e6, 1e6, nan)[1] == pytest.approx(171.869898, abs=1e-6)
    # Infinite but not NaN inputs keep today's result, with the section 6.1.3 polar-axis branch.
    assert ecef_to_geodetic(0.0, 0.0, inf) == (90.0, 0.0, inf)
    assert ecef_to_geodetic(0.0, 0.0, -inf) == (-90.0, 0.0, inf)
    assert ecef_to_geodetic(np.float64(7e6), 0, np.float32(0.0))[1] == 0.0  # NumPy and int scalars still work


def _todays_scalar_inverse(x, y, z):
    """The scalar inverse of shared main before A0 (0ba3c35), kept here as the reference for 'as today'."""
    a, e2 = 6378137.0, 6.69437999014e-3
    b = a * math.sqrt(1 - e2)
    ep2 = (a * a - b * b) / (b * b)
    p = math.sqrt(x * x + y * y)
    lon = 0.0 if p == 0 else math.atan2(y, x)
    theta = math.atan2(z * a, p * b)
    st, ct = math.sin(theta), math.cos(theta)
    lat = math.atan2(z + ep2 * b * st**3, p - e2 * a * ct**3)
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n


def test_g01_scalar_inverse_non_finite_triples_match_today_but_for_the_listed_changes():
    """Section 6.1.3: a non-finite input behaves as today, except for the three listed changes.

    The changes are the polar-axis branch (x = y = 0 with a signed z), the height
    formula and the fold of a -180 longitude to 180.
    """

    def same(u, v):
        return (math.isnan(u) and math.isnan(v)) or u == v

    values = [math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0, 6378137.0, -7e6, 1e6, 1e300]
    unexplained = []
    for xyz in itertools.product(values, repeat=3):
        if all(math.isfinite(v) for v in xyz):
            continue
        got, today = ecef_to_geodetic(*xyz), _todays_scalar_inverse(*xyz)
        if all(same(u, v) for u, v in zip(got, today)):
            continue
        x, y, _ = xyz
        has_nan = any(math.isnan(v) for v in xyz)
        polar = not has_nan and x == 0.0 and y == 0.0
        fold = today[1] == -180.0 and got[1] == 180.0
        lon_ok = same(got[1], today[1]) or fold
        height_only = not has_nan and same(got[0], today[0]) and lon_ok
        nan_z_fold = has_nan and fold and same(got[0], today[0]) and same(got[2], today[2])
        if not (polar or height_only or nan_z_fold):
            unexplained.append((xyz, got, today))
    assert unexplained == []


# ---------------------------------------------------------------- integral longitudes (6.1.1, 6.1.3)
# 360 * 2**45 + 1 is 1 deg, but a float64 cast rounds it to 360 * 2**45 (0 deg) before any
# reduction. 10**20 is -80 deg and does not fit in int64.
BIG_ONE_DEG = 360 * 2**45 + 1
assert float(BIG_ONE_DEG) % 360.0 == 0.0  # the fixture loses the degree under a float cast


def test_g01_large_exact_integer_longitude_keeps_its_degree_in_the_forward_conversion():
    expected = geodetic_to_ecef_vec(0.0, 1.0, 0.0)
    assert expected[1] == pytest.approx(111313.83923667614, abs=1e-6)
    for lon in (BIG_ONE_DEG, np.int64(BIG_ONE_DEG), np.array([BIG_ONE_DEG, 1], dtype=np.int64), [BIG_ONE_DEG, 1]):
        got = geodetic_to_ecef_vec(0.0, lon, 0.0)
        for g, e in zip(got, expected):
            np.testing.assert_allclose(g, np.broadcast_to(e, np.shape(g)), rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(("lon", "reduced"), [(10**20, -80.0), (-(10**20), 80.0), (2**1100 + 1, None)])
def test_g01_python_ints_beyond_int64_are_reduced_exactly(lon, reduced):
    if reduced is None:
        reduced = float((2**1100 + 1) % 360)
        reduced = reduced - 360.0 if reduced > 180.0 else reduced
    for got, want in zip(geodetic_to_ecef_vec(10.0, lon, 0.0), geodetic_to_ecef_vec(10.0, reduced, 0.0)):
        assert float(got) == pytest.approx(float(want), abs=1e-9)
    # A Python sequence may mix such ints with floats.
    got = geodetic_to_ecef_vec([10.0, 10.0], [lon, 2.5], 0.0)
    np.testing.assert_allclose(got[1], [geodetic_to_ecef_vec(10.0, reduced, 0.0)[1], geodetic_to_ecef_vec(10.0, 2.5, 0.0)[1]], atol=1e-9)


def test_g01_explicit_object_arrays_and_non_numbers_are_still_refused():
    for bad in (np.array([10**20], dtype=object), np.array([1.0], dtype=object), [None, 1], ["1"], True):
        with pytest.raises(ValueError):
            geodetic_to_ecef_vec(0.0, bad, 0.0)


def test_g01_integers_inside_the_range_and_floats_are_unchanged():
    assert geodetic_to_ecef_vec(0.0, -180, 0.0) == geodetic_to_ecef_vec(0.0, -180.0, 0.0)
    assert geodetic_to_ecef_vec(0.0, 181, 0.0) == geodetic_to_ecef_vec(0.0, -179.0, 0.0)
    assert geodetic_to_ecef_vec(0.0, 3.6e11, 0.0) == geodetic_to_ecef_vec(0.0, 0.0, 0.0)


def test_g03_large_exact_integer_longitude_in_enu_and_look_angles():
    east_1, north_1, up_1 = enu_basis(10.0, 1.0)
    for got, want in zip(enu_basis(10.0, BIG_ONE_DEG), (east_1, north_1, up_1)):
        assert got == pytest.approx(want, abs=1e-15)
    vec = enu_basis_vec(np.array([10.0]), np.array([BIG_ONE_DEG], dtype=np.int64))
    np.testing.assert_allclose(vec[0][0], east_1, atol=1e-15)
    delta = (1000.0, 2000.0, 3000.0)
    assert ecef_delta_to_enu(delta, 10.0, BIG_ONE_DEG) == pytest.approx(ecef_delta_to_enu(delta, 10.0, 1.0), abs=1e-9)
    np.testing.assert_allclose(
        ecef_delta_to_enu_vec(np.array([delta]), 10.0, np.array([BIG_ONE_DEG], dtype=np.int64))[0],
        ecef_delta_to_enu(delta, 10.0, 1.0)[0],
        atol=1e-9,
    )
    ground = geodetic_to_ecef(10.0, 1.0, 0.0)
    sat = geodetic_to_ecef(10.5, 1.4, 550e3)
    expected = look_angles_ecef(sat, ground, 10.0, 1.0)
    assert look_angles_ecef(sat, ground, 10.0, BIG_ONE_DEG) == pytest.approx(expected, abs=1e-9)
    got = look_angles_ecef_vec(np.array([sat]), np.array([ground]), 10.0, np.array([BIG_ONE_DEG], dtype=np.int64))
    assert tuple(float(v[0]) for v in got) == pytest.approx(expected, abs=1e-9)
    assert look_angles_ecef(sat, geodetic_to_ecef(10.0, -80.0, 0.0), 10.0, 10**20)[0] > 0.0
