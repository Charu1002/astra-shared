"""
geo_math.py

Pure-math geodetic coordinate utilities shared across Astra services.
Uses stdlib math for scalar helpers and numpy for vector helpers.

Functions:
- ecef_to_geodetic()   - ECEF (metres) → WGS84 lat/lon/alt
- geodetic_to_ecef()   - WGS84 lat/lon/alt → ECEF (metres)
- distance_3d()        - Euclidean distance between two ECEF points (metres)
- haversine_km()       - Great-circle distance between two geodetic points (km)
- teme_to_ecef()       - TEME position (km) → ECEF position (km)
- teme_state_to_ecef() - TEME state (km, km/s) → Earth-fixed ECEF state (km, km/s)
- lvlh_reference_velocity()   - Earth-relative ECEF velocity → inertial velocity in ECEF axes
- check_inertial_speed_km_s() - Reject a frame velocity in the wrong unit

The TEME functions need sgp4, which is imported lazily.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# WGS84 constants
_A = 6378137.0          # semi-major axis, metres
_E2 = 6.69437999014e-3  # first eccentricity squared

# Earth rotation rate (rad/s). IERS / Vallado et al., AIAA 2006-6753.
OMEGA_EARTH_RAD_S = 7.2921158553e-5

# Gravitational parameter (km^3/s^2) and mean Earth radius (km) for the
# bound-orbit speed band in check_inertial_speed_km_s().
_MU_KM3_S2 = 398600.4418
_R_EARTH_MEAN_KM = 6371.0

# Marker at the start of every frame-velocity unit error, so logs can be searched for it.
FRAME_VELOCITY_UNITS_ERROR = "antenna_frame_velocity_units"


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic (lat, lon, alt).

    Uses WGS84 reference ellipsoid.

    Args:
        x, y, z: ECEF position in metres

    Returns:
        Tuple of (latitude_deg, longitude_deg, altitude_m)
    """
    b = _A * math.sqrt(1 - _E2)
    ep2 = (_A * _A - b * b) / (b * b)
    p = math.sqrt(x * x + y * y)
    lon = 0.0 if p == 0 else math.atan2(y, x)
    theta = math.atan2(z * _A, p * b)
    st = math.sin(theta)
    ct = math.cos(theta)
    lat = math.atan2(z + ep2 * b * st ** 3, p - _E2 * _A * ct ** 3)
    sin_lat = math.sin(lat)
    n = _A / math.sqrt(1 - _E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float
) -> tuple[float, float, float]:
    """
    Convert WGS84 geodetic coordinates to ECEF.

    Args:
        lat_deg: Geodetic latitude in degrees
        lon_deg: Longitude in degrees
        alt_m: Altitude above ellipsoid in metres

    Returns:
        Tuple of (x, y, z) ECEF position in metres
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = _A / math.sqrt(1 - _E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1 - _E2) + alt_m) * sin_lat
    return x, y, z


def distance_3d(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    """
    Euclidean distance between two ECEF points.

    Args:
        x1, y1, z1: First point in metres
        x2, y2, z2: Second point in metres

    Returns:
        Distance in metres
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def haversine_km(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> float:
    """
    Great-circle distance between two geodetic points (Haversine formula).

    Args:
        lat1_deg, lon1_deg: First point in degrees
        lat2_deg, lon2_deg: Second point in degrees

    Returns:
        Distance in kilometres
    """
    R = 6371.0  # Earth mean radius, km
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlat = math.radians(lat2_deg - lat1_deg)
    dlon = math.radians(lon2_deg - lon1_deg)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def compute_elevation(
    obs_lat: float,
    obs_lon: float,
    sat_lat: float,
    sat_lon: float,
    sat_alt_km: float,
    obs_alt_km: float = 0.0,
) -> float:
    """Compute elevation angle in degrees from observer to satellite nadir.

    Uses spherical Earth (R=6371 km). Returns 0.0 on degenerate geometry.
    """
    obs_lat_rad = math.radians(obs_lat)
    obs_lon_rad = math.radians(obs_lon)
    sat_lat_rad = math.radians(sat_lat)
    sat_lon_rad = math.radians(sat_lon)
    dlon = sat_lon_rad - obs_lon_rad
    d_sigma = math.acos(
        max(
            -1.0,
            min(
                1.0,
        math.sin(obs_lat_rad) * math.sin(sat_lat_rad)
        + math.cos(obs_lat_rad) * math.cos(sat_lat_rad) * math.cos(dlon),
            ),
        )
    )
    R = 6371.0
    R_obs = R + obs_alt_km
    Rs = R + sat_alt_km
    slant_km = math.sqrt(Rs**2 + R_obs**2 - 2 * Rs * R_obs * math.cos(d_sigma))
    if slant_km == 0.0:
        return 0.0
    sin_el = (Rs * math.cos(d_sigma) - R_obs) / slant_km
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


def compute_elevation_vec(
    obs_lat,
    obs_lon,
    sat_lat,
    sat_lon,
    sat_alt_km,
    obs_alt_km=0.0,
) -> np.ndarray:
    """Vectorized spherical-Earth elevation angle in degrees."""
    r_earth_km = 6371.0
    obs_alt_arr = np.asarray(obs_alt_km, dtype=np.float64)
    sat_alt_arr = np.asarray(sat_alt_km, dtype=np.float64)
    r_obs = r_earth_km + obs_alt_arr
    r_sat = r_earth_km + sat_alt_arr

    obs_lat_rad = np.radians(obs_lat)
    obs_lon_rad = np.radians(obs_lon)
    sat_lat_rad = np.radians(sat_lat)
    sat_lon_rad = np.radians(sat_lon)
    dlon = sat_lon_rad - obs_lon_rad

    cos_d_sigma = (
        np.sin(obs_lat_rad) * np.sin(sat_lat_rad)
        + np.cos(obs_lat_rad) * np.cos(sat_lat_rad) * np.cos(dlon)
    )
    d_sigma = np.arccos(np.clip(cos_d_sigma, -1.0, 1.0))
    slant_km = np.sqrt(
        r_sat**2 + r_obs**2 - 2.0 * r_sat * r_obs * np.cos(d_sigma)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_el = (r_sat * np.cos(d_sigma) - r_obs) / slant_km
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def teme_to_ecef(x: float, y: float, z: float, jd: float) -> tuple[float, float, float]:
    """Convert TEME position (km) to ECEF (km) using Earth rotation at Julian date jd."""
    try:
        from sgp4.propagation import gstime
    except ImportError as exc:
        raise RuntimeError("sgp4 package required for teme_to_ecef") from exc
    theta = gstime(jd)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * x + sin_t * y, -sin_t * x + cos_t * y, z


def teme_vel_to_ecef(
    vx: float, vy: float, vz: float, jd: float
) -> tuple[float, float, float]:
    """Rotate a TEME velocity (km/s) into ECEF axes. Deprecated.

    The result is the INERTIAL velocity expressed in ECEF axes, NOT the velocity
    relative to the rotating Earth: the -omega x r term is missing, which puts up
    to ~505 m/s of error into a LEO velocity. Use teme_state_to_ecef() instead.
    Kept only until every caller has moved (Issue #19); it will be removed.
    """
    warnings.warn(
        "teme_vel_to_ecef() omits the Earth-rotation term; use teme_state_to_ecef()",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from sgp4.propagation import gstime
    except ImportError as exc:
        raise RuntimeError("sgp4 package required for teme_vel_to_ecef") from exc
    theta = gstime(jd)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * vx + sin_t * vy, -sin_t * vx + cos_t * vy, vz


def teme_state_to_ecef(
    r_teme: tuple[float, float, float],
    v_teme: tuple[float, float, float],
    jd: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a TEME state (km, km/s) to an Earth-fixed ECEF state (km, km/s).

    Polar motion is ignored (PEF ~ ITRF at this accuracy). The velocity is
    relative to the rotating Earth: v_ecef = R3(gmst) . v_teme - omega x r_ecef,
    as in Vallado et al. (AIAA 2006-6753) and Skyfield's TEME_to_ITRF.

    Args:
        r_teme: TEME position in km
        v_teme: TEME velocity in km/s
        jd: Julian date (UT1 ~ UTC), including the day fraction

    Returns:
        ((x, y, z) in km, (vx, vy, vz) in km/s), both ECEF
    """
    try:
        from sgp4.propagation import gstime
    except ImportError as exc:
        raise RuntimeError("sgp4 package required for teme_state_to_ecef") from exc
    theta = gstime(jd)
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = r_teme
    vx, vy, vz = v_teme
    xe = c * x + s * y
    ye = -s * x + c * y
    vxe = c * vx + s * vy + OMEGA_EARTH_RAD_S * ye
    vye = -s * vx + c * vy - OMEGA_EARTH_RAD_S * xe
    return (xe, ye, z), (vxe, vye, vz)


def lvlh_reference_velocity(
    r_ecef_km: tuple[float, float, float],
    v_ecef_km_s: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Inertial velocity in ECEF axes (v + omega x r) for the LVLH along-track axis.

    r_ecef_km must be the producer-consistent satellite position (true or WGS84
    ECEF, km), not a sphere-placed one, so the result equals the rotated TEME
    velocity exactly. Exact zero is returned unchanged: zero velocity is the
    fixed-satellite sentinel that selects the local-North (NED) frame.
    """
    vx, vy, vz = v_ecef_km_s
    if vx == 0.0 and vy == 0.0 and vz == 0.0:
        return 0.0, 0.0, 0.0
    x, y, _ = r_ecef_km
    return vx - OMEGA_EARTH_RAD_S * y, vy + OMEGA_EARTH_RAD_S * x, vz


def check_inertial_speed_km_s(
    r_ecef_km: tuple[float, float, float],
    v_ref_km_s: tuple[float, float, float],
) -> None:
    """Raise ValueError unless |v_ref| is a bound-orbit speed at radius |r|.

    The band runs from the apogee speed of an orbit whose perigee touches the
    Earth, sqrt(mu (2/r - 2/(r + R))), to the escape speed sqrt(2 mu / r), with
    2 % margins. It rejects a velocity passed in m/s and a km/s velocity divided
    by 1000 again (LEO). It cannot catch GEO/IGSO unit mistakes, whose rebuilt
    speed stays inside the band; per-call-site tests cover those.

    The failure is logged before it is raised, because the engines' job and
    route wrappers put the traceback only in the response.
    """
    x, y, z = r_ecef_km
    r = math.sqrt(x * x + y * y + z * z)
    vx, vy, vz = v_ref_km_s
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    lo = math.sqrt(_MU_KM3_S2 * (2.0 / r - 2.0 / (r + _R_EARTH_MEAN_KM))) * 0.98 if r > 0 else math.nan
    hi = math.sqrt(2.0 * _MU_KM3_S2 / r) * 1.02 if r > 0 else math.nan
    if not (lo <= speed <= hi):
        message = (
            f"{FRAME_VELOCITY_UNITS_ERROR}: inertial speed {speed:.6g} km/s at r = {r:.6g} km is outside "
            f"[{lo:.6g}, {hi:.6g}] km/s; the frame velocity must be Earth-relative ECEF in km/s "
            f"(position {tuple(r_ecef_km)}, rebuilt velocity {tuple(v_ref_km_s)})"
        )
        logger.error(message)
        raise ValueError(message)


def latlon_to_ecef(
    lat_deg: float, lon_deg: float, alt_km: float = 0.0
) -> tuple[float, float, float]:
    """Convert lat/lon/altitude to ECEF Cartesian coordinates (km), spherical Earth model.

    Returns (x, y, z) in km. Uses R=6371 km sphere — appropriate for RF link
    budget geometry where sub-km accuracy is not required.
    """
    R = 6371.0 + alt_km
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    z = R * math.sin(lat_rad)
    return x, y, z
