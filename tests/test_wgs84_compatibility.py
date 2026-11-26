"""G06: public shared names keep their signatures and units, and the package is the canonical source.

A0 plan v7, sections 6.1.8 and 6.3. The spherical helpers stay for non-RF use
only; A0 removes their RF uses in Radio, not the helpers themselves. The clean
installed-wheel import check runs in CI (.github/workflows/tests.yml).
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import astra_shared
from astra_shared import geo_math

REPO_ROOT = Path(__file__).resolve().parents[1]

SIGNATURES = {
    "distance_3d": ["x1", "y1", "z1", "x2", "y2", "z2"],
    "haversine_km": ["lat1_deg", "lon1_deg", "lat2_deg", "lon2_deg"],
    "compute_elevation": ["obs_lat", "obs_lon", "sat_lat", "sat_lon", "sat_alt_km", "obs_alt_km"],
    "compute_elevation_vec": ["obs_lat", "obs_lon", "sat_lat", "sat_lon", "sat_alt_km", "obs_alt_km"],
    "latlon_to_ecef": ["lat_deg", "lon_deg", "alt_km"],
    "teme_to_ecef": ["x", "y", "z", "jd"],
    "teme_state_to_ecef": ["r_teme", "v_teme", "jd"],
    "lvlh_reference_velocity": ["r_ecef_km", "v_ecef_km_s"],
    "check_inertial_speed_km_s": ["r_ecef_km", "v_ref_km_s"],
}

A0_API = [
    "WGS84_A_M", "WGS84_F", "WGS84_B_M", "WGS84_E2", "OMEGA_EARTH_RAD_S",
    "geodetic_to_ecef", "ecef_to_geodetic", "geodetic_to_ecef_vec", "ecef_to_geodetic_vec",
    "enu_basis", "enu_basis_vec", "ecef_delta_to_enu", "ecef_delta_to_enu_vec",
    "look_angles_ecef", "look_angles_ecef_vec", "nadir_unit_vec", "off_axis_deg_vec",
    "intersect_ray_wgs84", "intersect_ray_wgs84_vec",
    "lvlh_reference_velocity_m", "lvlh_reference_velocity_m_vec",
]


@pytest.mark.parametrize(("name", "params"), sorted(SIGNATURES.items()))
def test_g06_public_signatures_are_unchanged(name, params):
    assert list(inspect.signature(getattr(geo_math, name)).parameters) == params


def test_g06_a0_api_names_exist():
    missing = [name for name in A0_API if not hasattr(geo_math, name)]
    assert missing == []


def test_g06_spherical_helpers_keep_their_6371_km_units():
    assert geo_math.latlon_to_ecef(0.0, 0.0) == pytest.approx((6371.0, 0.0, 0.0))
    assert geo_math.latlon_to_ecef(90.0, 0.0, 10.0)[2] == pytest.approx(6381.0)
    assert geo_math.haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(6371.0 * math.pi / 180.0)
    assert geo_math.compute_elevation(10.0, 20.0, 10.0, 20.0, 550.0) == pytest.approx(90.0)
    got = geo_math.compute_elevation_vec(np.array([10.0]), np.array([20.0]), 10.0, 20.0, 550.0)
    assert got.shape == (1,) and float(got[0]) == pytest.approx(90.0)
    assert geo_math.distance_3d(0, 0, 0, 3, 4, 0) == 5.0
    assert geo_math.FRAME_VELOCITY_UNITS_ERROR == "antenna_frame_velocity_units"


def test_g06_canonical_package_import():
    assert Path(geo_math.__file__).parent.name == "astra_shared"
    assert Path(astra_shared.__file__).parent == Path(geo_math.__file__).parent


def test_g06_root_duplicate_is_gone():
    # Shared #106 removed the root geo_math.py; nothing may import it bare.
    assert not (REPO_ROOT / "geo_math.py").exists()
