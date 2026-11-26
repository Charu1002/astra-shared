"""
geo_math.py

Pure-math geodetic coordinate utilities shared across Astra services.
Uses stdlib math for scalar helpers and numpy for vector helpers.

Functions:
- ecef_to_geodetic()   - ECEF (metres) → WGS84 lat/lon/alt
- geodetic_to_ecef()   - WGS84 lat/lon/alt → ECEF (metres)
- geodetic_to_ecef_vec(), ecef_to_geodetic_vec() - the same, on arrays
- enu_basis[_vec](), ecef_delta_to_enu[_vec](), look_angles_ecef[_vec]() - local East-North-Up frame
- nadir_unit_vec(), off_axis_deg_vec() - geocentric nadir and the off-axis angle
- intersect_ray_wgs84[_vec]() - first hit of a ray on the WGS84 ellipsoid
- lvlh_reference_velocity_m[_vec]() - metre form of lvlh_reference_velocity()
- distance_3d()        - Euclidean distance between two ECEF points (metres)
- haversine_km()       - Great-circle distance between two geodetic points (km)
- teme_to_ecef()       - TEME position (km) → ECEF position (km)
- teme_state_to_ecef() - TEME state (km, km/s) → Earth-fixed ECEF state (km, km/s)
- lvlh_reference_velocity()   - Earth-relative ECEF velocity → inertial velocity in ECEF axes
- check_inertial_speed_km_s() - Reject a frame velocity in the wrong unit

The WGS84 functions follow the A0 API of the A0 plan v7, section 6.1: metres,
m/s and degrees at the interface, ValueError for an invalid argument, and the
"_vec" forms take NumPy arrays that broadcast together. haversine_km(),
compute_elevation[_vec]() and latlon_to_ecef() use a 6371 km sphere and are
kept for non-RF use only.

The TEME functions need sgp4, which is imported lazily.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# WGS84 constants (NGA TR8350.2). Radio and the UI import these and do not repeat them.
WGS84_A_M = 6378137.0  # semi-major axis, metres
WGS84_F = 1 / 298.257223563  # flattening
WGS84_B_M = WGS84_A_M * (1 - WGS84_F)  # semi-minor axis, metres
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # first eccentricity squared

# Private names kept for existing code; they are the constants above.
_A = WGS84_A_M
_E2 = WGS84_E2

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

    Uses the WGS84 ellipsoid, Bowring's closed-form latitude and the height
    h = p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin(lat)^2), which stays exact
    at the poles. On the polar axis (x = y = 0) the result is latitude +/-90
    by the sign of z, longitude 0 and height |z| - b; the geocentre gives
    (90.0, 0.0, -b). Longitude is in (-180, 180].

    This scalar form raises no new error: a non-finite input gives a
    non-finite output, as before: a NaN in x or y gives NaN in all three
    outputs, and a NaN in z alone gives NaN latitude and height with the
    longitude of (x, y) (0 when x = y = 0). RF code uses ecef_to_geodetic_vec().

    Args:
        x, y, z: ECEF position in metres

    Returns:
        Tuple of (latitude_deg, longitude_deg, altitude_m)
    """
    if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
        p = math.hypot(x, y)  # no overflow or underflow for finite input
    else:
        p = math.sqrt(x * x + y * y)  # a non-finite triple keeps today's result (section 6.1.3)
    if math.isnan(z):  # as today: NaN latitude and height, the longitude of (x, y)
        lon_deg = 0.0 if p == 0.0 else math.degrees(math.atan2(y, x))
        return math.nan, (180.0 if lon_deg == -180.0 else lon_deg), math.nan
    if p == 0.0:
        lat_deg = 90.0 if z >= 0 else -90.0
        return lat_deg, 0.0, abs(z) - WGS84_B_M
    ep2 = (WGS84_A_M * WGS84_A_M - WGS84_B_M * WGS84_B_M) / (WGS84_B_M * WGS84_B_M)
    theta = math.atan2(z * WGS84_A_M, p * WGS84_B_M)
    st = math.sin(theta)
    ct = math.cos(theta)
    lat = math.atan2(z + ep2 * WGS84_B_M * st**3, p - WGS84_E2 * WGS84_A_M * ct**3)
    sin_lat = math.sin(lat)
    alt = p * math.cos(lat) + z * sin_lat - WGS84_A_M * math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    lon_deg = math.degrees(math.atan2(y, x))
    if lon_deg == -180.0:
        lon_deg = 180.0
    return math.degrees(lat), lon_deg, alt


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


def _inertial_speed_band_km_s(radius_km):
    """The bound-orbit speed band (lo, hi) in km/s at radius_km; NaN where radius_km <= 0.

    The one copy of the band: check_inertial_speed_km_s() and
    lvlh_reference_velocity_m_vec() both use it. Takes a float or an array.
    """
    r = np.asarray(radius_km, dtype=np.float64)
    positive = r > 0.0
    safe = np.where(positive, r, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):  # as the old float arithmetic: inf, no warning
        lo = np.sqrt(_MU_KM3_S2 * (2.0 / safe - 2.0 / (safe + _R_EARTH_MEAN_KM))) * 0.98
        hi = np.sqrt(2.0 * _MU_KM3_S2 / safe) * 1.02
    return np.where(positive, lo, np.nan), np.where(positive, hi, np.nan)


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
    band_lo, band_hi = _inertial_speed_band_km_s(r)
    lo, hi = float(band_lo), float(band_hi)
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

    Returns (x, y, z) in km on the R = 6371 km sphere. Kept for non-RF compatibility only
    (A0 plan v7, section 6.1.8): RF geometry uses the WGS84 entries, geodetic_to_ecef_vec()
    or Radio's rf_geometry.satellite_ecef_m(), in metres.
    """
    R = 6371.0 + alt_km
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = R * math.cos(lat_rad) * math.cos(lon_rad)
    y = R * math.cos(lat_rad) * math.sin(lon_rad)
    z = R * math.sin(lat_rad)
    return x, y, z


# ---------------------------------------------------------------------------
# A0 WGS84 API (A0 plan v7, section 6.1)
#
# Units: metres, m/s and degrees at the interface; radians only inside.
# Inputs: Python numbers, sequences of numbers, or real NumPy arrays. Bools,
# strings, None, complex values and object arrays raise ValueError, as do
# non-finite values. One invalid element fails the whole call. Inputs are not
# modified; outputs are new float64 arrays (Python floats for scalar entries).
# Cartesian arguments have a final axis of 3; the other axes broadcast.
# ---------------------------------------------------------------------------

# A horizontal line-of-sight component at or below this fraction of the range is
# treated as vertical: azimuth 0 and elevation +/-90 (plan section 5.4).
_AZIMUTH_EPSILON = 1e-12


def _real_array(name: str, value) -> np.ndarray:
    """Return value as a new finite float64 array, or raise ValueError naming it."""
    if value is None or isinstance(value, (str, bytes, bool, np.bool_, complex)):
        raise ValueError(f"{name} must be a real number or an array of real numbers, got {type(value).__name__}")
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real number or an array of real numbers") from exc
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must hold real numbers, got dtype {arr.dtype}")
    arr = np.asarray(arr, dtype=np.float64)  # no copy for float64 input; inputs are never written
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _cartesian(name: str, value) -> np.ndarray:
    """A finite float64 array whose final axis has length 3."""
    arr = _real_array(name, value)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have a final axis of length 3, got shape {arr.shape}")
    return arr


def _broadcast_shape(**shapes: tuple) -> tuple:
    try:
        return np.broadcast_shapes(*shapes.values())
    except ValueError as exc:
        described = ", ".join(f"{k} {v}" for k, v in shapes.items())
        raise ValueError(f"incompatible shapes: {described}") from exc


def _reduce_longitude(lon_deg: np.ndarray) -> np.ndarray:
    """Exact reduction of a longitude outside [-180, 180] to (-180, 180] (plan section 6.1.3).

    fmod(lon, 360) is exact, and the shift by 360 is exact (Sterbenz), so a large
    periodic value gives the same result as its reduced value. Values inside
    [-180, 180] are returned unchanged.
    """
    outside = np.abs(lon_deg) > 180.0
    if not np.any(outside):
        return lon_deg
    reduced = np.fmod(lon_deg, 360.0)
    reduced = np.where(reduced > 180.0, reduced - 360.0, reduced)
    reduced = np.where(reduced <= -180.0, reduced + 360.0, reduced)
    return np.where(outside, reduced, lon_deg)


def _reduce_integers(arr: np.ndarray) -> np.ndarray:
    """Integer longitudes outside [-180, 180] reduced modulo 360 in integer arithmetic."""
    outside = (arr > 180) | (arr < -180)
    if not np.any(outside):
        return arr
    return np.where(outside, np.mod(arr, 360), arr)


def _longitude_array(name: str, value) -> np.ndarray:
    """A longitude as a finite float64 array, reducing integral input exactly first.

    The strict entries promise an exact reduction for any finite longitude (plan section
    6.1.3), and accept Python ints and integer arrays (6.1.1). A float64 cast would round a
    large integer before the reduction: 360 * 2**45 + 1 (1 deg) would become 0 deg. So an
    integer outside [-180, 180] is reduced modulo 360 in integers before the cast: a Python
    int of any size, an integer NumPy scalar or array, and a Python sequence of ints and
    floats (whose ints may exceed int64). Floats are unchanged. An explicitly supplied
    object array is still refused, as are bools, strings and the other non-numbers.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        number = int(value)
        return _real_array(name, number % 360 if not -180 <= number <= 180 else number)
    if isinstance(value, np.ndarray):
        return _real_array(name, _reduce_integers(value) if value.dtype.kind in "iu" else value)
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError, OverflowError):
            arr = None
        if arr is not None and arr.dtype.kind in "iu":
            return _real_array(name, _reduce_integers(arr))
        if arr is not None and arr.dtype == object:
            flat = arr.ravel()
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat):
                reduced = [
                    float(v % 360) if isinstance(v, int) and not -180 <= v <= 180 else float(v) for v in flat
                ]
                return _real_array(name, np.array(reduced, dtype=np.float64).reshape(arr.shape))
    return _real_array(name, value)


def _scalar_longitude(name: str, value) -> float:
    """Scalar form of _longitude_array()."""
    arr = _longitude_array(name, value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a scalar for this entry; use the _vec form for arrays")
    return float(arr)


def _check_latitude(name: str, lat_deg: np.ndarray) -> None:
    if np.any(np.abs(lat_deg) > 90.0):
        raise ValueError(f"{name} must be within [-90, 90] degrees")


def _scalar_angle(name: str, value) -> float:
    arr = _real_array(name, value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a scalar for this entry; use the _vec form for arrays")
    return float(arr)


def _scalar_vector(name: str, value) -> np.ndarray:
    arr = _cartesian(name, value)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be one 3-vector for this entry, got shape {arr.shape}; use the _vec form")
    return arr


def _scaled_unit(name: str, vec: np.ndarray) -> np.ndarray:
    """Normalise along the final axis with a scaled norm; a zero vector raises."""
    scale = np.max(np.abs(vec), axis=-1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError(f"{name} must be non-zero")
    scaled = vec / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _out(value) -> np.ndarray:
    """A float64 ndarray; a 0-D result stays a 0-D array, not a NumPy scalar."""
    return np.asarray(value, dtype=np.float64)


def _tuple3(arr: np.ndarray) -> tuple[float, float, float]:
    return float(arr[0]), float(arr[1]), float(arr[2])


def geodetic_to_ecef_vec(lat_deg, lon_deg, alt_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 geodetic latitude, longitude (degrees) and ellipsoidal height (m) to ECEF (m).

    The inputs broadcast to a shape S; returns (x_m, y_m, z_m), three arrays of
    shape S. Raises ValueError for a non-finite value, a latitude outside
    [-90, 90] or incompatible shapes. Longitude may be any finite value; one
    outside [-180, 180] is reduced exactly before the trigonometry.
    """
    lat = _real_array("lat_deg", lat_deg)
    lon = _longitude_array("lon_deg", lon_deg)
    alt = _real_array("alt_m", alt_m)
    shape = _broadcast_shape(lat_deg=lat.shape, lon_deg=lon.shape, alt_m=alt.shape)
    _check_latitude("lat_deg", lat)
    lat, lon, alt = (np.broadcast_to(v, shape) for v in (lat, lon, alt))
    phi = np.radians(lat)
    lam = np.radians(_reduce_longitude(lon))
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    x = (n + alt) * cos_phi * np.cos(lam)
    y = (n + alt) * cos_phi * np.sin(lam)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_phi
    return _out(x), _out(y), _out(z)


def ecef_to_geodetic_vec(x_m, y_m, z_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF (m) to WGS84 geodetic latitude, longitude (degrees) and ellipsoidal height (m).

    Bowring's closed-form latitude with the pole-exact height
    h = p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin(lat)^2). On the polar axis
    the latitude is +/-90 by the sign of z, the longitude 0 and the height
    |z| - b. Longitude is in (-180, 180]. Raises ValueError for a non-finite
    value, the geocentre (0, 0, 0) or incompatible shapes.
    """
    x = _real_array("x_m", x_m)
    y = _real_array("y_m", y_m)
    z = _real_array("z_m", z_m)
    shape = _broadcast_shape(x_m=x.shape, y_m=y.shape, z_m=z.shape)
    x, y, z = (np.broadcast_to(v, shape) for v in (x, y, z))
    p = np.hypot(x, y)
    on_axis = p == 0.0
    if np.any(on_axis & (z == 0.0)):
        raise ValueError("the ECEF position (0, 0, 0) is the geocentre and has no geodetic coordinates")
    ep2 = (WGS84_A_M * WGS84_A_M - WGS84_B_M * WGS84_B_M) / (WGS84_B_M * WGS84_B_M)
    theta = np.arctan2(z * WGS84_A_M, p * WGS84_B_M)
    st = np.sin(theta)
    ct = np.cos(theta)
    phi = np.arctan2(z + ep2 * WGS84_B_M * st**3, p - WGS84_E2 * WGS84_A_M * ct**3)
    sin_phi = np.sin(phi)
    alt = p * np.cos(phi) + z * sin_phi - WGS84_A_M * np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    lat = np.degrees(phi)
    lon = np.degrees(np.arctan2(y, x))
    lon = np.where(lon == -180.0, 180.0, lon)
    lat = np.where(on_axis, np.where(z >= 0.0, 90.0, -90.0), lat)
    lon = np.where(on_axis, 0.0, lon)
    alt = np.where(on_axis, np.abs(z) - WGS84_B_M, alt)
    return _out(lat), _out(lon), _out(alt)


def _enu_axes(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.radians(lat)
    lam = np.radians(_reduce_longitude(lon))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_lam, cos_lam = np.sin(lam), np.cos(lam)
    east = np.stack([-sin_lam, cos_lam, np.zeros_like(lam)], axis=-1)
    north = np.stack([-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi], axis=-1)
    up = np.stack([cos_phi * cos_lam, cos_phi * sin_lam, sin_phi], axis=-1)
    return _out(east), _out(north), _out(up)


def _components(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """a - b per component, without building an (..., 3) temporary."""
    return a[..., 0] - b[..., 0], a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]


def _project_enu(dx, dy, dz, lat: np.ndarray, lon: np.ndarray):
    """East, North and Up components of a displacement, by component arithmetic."""
    phi = np.radians(lat)
    lam = np.radians(_reduce_longitude(lon))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_lam, cos_lam = np.sin(lam), np.cos(lam)
    horizontal_radial = cos_lam * dx + sin_lam * dy
    east = cos_lam * dy - sin_lam * dx
    north = cos_phi * dz - sin_phi * horizontal_radial
    up = cos_phi * horizontal_radial + sin_phi * dz
    return east, north, up


def _full(value, shape: tuple) -> np.ndarray:
    """value broadcast to shape, as a new writable float64 array when it had to grow."""
    arr = np.asarray(value, dtype=np.float64)
    return arr if arr.shape == shape else np.array(np.broadcast_to(arr, shape))


def enu_basis_vec(lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East, North and Up unit axes (EPSG 9836/9837) at geodetic latitude and longitude.

    Up is the geodetic normal, not the radial direction. The inputs broadcast to
    a shape S; returns three arrays of shape S+(3,).
    """
    lat = _real_array("lat_deg", lat_deg)
    lon = _longitude_array("lon_deg", lon_deg)
    shape = _broadcast_shape(lat_deg=lat.shape, lon_deg=lon.shape)
    _check_latitude("lat_deg", lat)
    return _enu_axes(np.broadcast_to(lat, shape), np.broadcast_to(lon, shape))


def enu_basis(lat_deg, lon_deg) -> tuple[tuple[float, float, float], ...]:
    """Scalar form of enu_basis_vec(): (east, north, up), three 3-tuples of floats."""
    lat = _scalar_angle("lat_deg", lat_deg)
    lon = _scalar_longitude("lon_deg", lon_deg)
    east, north, up = enu_basis_vec(lat, lon)
    return _tuple3(east), _tuple3(north), _tuple3(up)


def ecef_delta_to_enu_vec(delta_m, lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ECEF displacements (m) onto the local East, North and Up axes.

    delta_m has shape (..., 3); the angles broadcast with its leading shape to
    S. Returns (east_m, north_m, up_m), three arrays of shape S.
    """
    delta = _cartesian("delta_m", delta_m)
    lat = _real_array("lat_deg", lat_deg)
    lon = _longitude_array("lon_deg", lon_deg)
    shape = _broadcast_shape(delta_m=delta.shape[:-1], lat_deg=lat.shape, lon_deg=lon.shape)
    _check_latitude("lat_deg", lat)
    east, north, up = _project_enu(delta[..., 0], delta[..., 1], delta[..., 2], lat, lon)
    return _full(east, shape), _full(north, shape), _full(up, shape)


def ecef_delta_to_enu(delta_m, lat_deg, lon_deg) -> tuple[float, float, float]:
    """Scalar form of ecef_delta_to_enu_vec(): (east_m, north_m, up_m) as floats."""
    delta = _scalar_vector("delta_m", delta_m)
    lat = _scalar_angle("lat_deg", lat_deg)
    lon = _scalar_longitude("lon_deg", lon_deg)
    east, north, up = ecef_delta_to_enu_vec(delta, lat, lon)
    return float(east), float(north), float(up)


def look_angles_ecef_vec(sat_m, ground_m, lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range (m), elevation and azimuth (degrees) from ground points to satellites.

    sat_m and ground_m have shape (..., 3); lat_deg and lon_deg must be the
    geodetic coordinates of ground_m (not checked). All broadcast to a shape S.
    Elevation is in [-90, 90]; azimuth is clockwise from North in [0, 360). A
    line of sight whose horizontal part is at most 1e-12 of the range gives
    azimuth 0 and elevation +/-90. No elevation mask is applied. A zero range
    raises ValueError.
    """
    sat = _cartesian("sat_m", sat_m)
    ground = _cartesian("ground_m", ground_m)
    lat = _real_array("lat_deg", lat_deg)
    lon = _longitude_array("lon_deg", lon_deg)
    shape = _broadcast_shape(
        sat_m=sat.shape[:-1], ground_m=ground.shape[:-1], lat_deg=lat.shape, lon_deg=lon.shape
    )
    _check_latitude("lat_deg", lat)
    dx, dy, dz = _components(sat, ground)
    # hypot neither overflows nor underflows for finite inputs (beyond ~1e154 m or
    # below ~1e-154 m a sum of squares would).
    range_m = np.hypot(np.hypot(dx, dy), dz)
    if np.any(range_m == 0.0):
        raise ValueError("sat_m and ground_m must not coincide (zero range)")
    east, north, up = _project_enu(dx, dy, dz, lat, lon)
    horizontal = np.hypot(east, north)
    elevation = np.degrees(np.arctan2(up, horizontal))
    azimuth = np.degrees(np.arctan2(east, north))
    azimuth = np.where(azimuth < 0.0, azimuth + 360.0, azimuth)
    azimuth = np.where(azimuth >= 360.0, 0.0, azimuth) + 0.0  # fold 360.0 and -0.0 to 0.0
    vertical = horizontal <= _AZIMUTH_EPSILON * range_m
    if np.any(vertical):
        elevation = np.where(vertical, np.where(up >= 0.0, 90.0, -90.0), elevation)
        azimuth = np.where(vertical, 0.0, azimuth)
    return _full(range_m, shape), _full(elevation, shape), _full(azimuth, shape)


def look_angles_ecef(sat_m, ground_m, lat_deg, lon_deg) -> tuple[float, float, float]:
    """Scalar form of look_angles_ecef_vec(): (range_m, elevation_deg, azimuth_deg) as floats."""
    sat = _scalar_vector("sat_m", sat_m)
    ground = _scalar_vector("ground_m", ground_m)
    lat = _scalar_angle("lat_deg", lat_deg)
    lon = _scalar_longitude("lon_deg", lon_deg)
    range_m, elevation, azimuth = look_angles_ecef_vec(sat, ground, lat, lon)
    return float(range_m), float(elevation), float(azimuth)


def nadir_unit_vec(sat_m) -> np.ndarray:
    """Geocentric nadir -sat_m / |sat_m| (not the negative geodetic normal), shape S+(3,)."""
    sat = _cartesian("sat_m", sat_m)
    return _out(-_scaled_unit("sat_m", sat))


def off_axis_deg_vec(boresight_unit, sat_m, point_m) -> np.ndarray:
    """Angle (degrees, [0, 180]) between a boresight and the satellite-to-point direction.

    boresight_unit is a direction and is normalised here, so any finite
    non-zero vector is accepted. All three arguments have shape (..., 3) and
    broadcast together to S+(3,); the result has shape S. Computed as
    atan2(|b x q|, b . q), which keeps its precision near 0 and 180 degrees.
    For the steering (off-nadir) angle of a target, pass nadir_unit_vec(sat_m).
    """
    bore = _cartesian("boresight_unit", boresight_unit)
    sat = _cartesian("sat_m", sat_m)
    point = _cartesian("point_m", point_m)
    shape = _broadcast_shape(boresight_unit=bore.shape[:-1], sat_m=sat.shape[:-1], point_m=point.shape[:-1])
    b = _scaled_unit("boresight_unit", bore)
    dx, dy, dz = _components(point, sat)
    scale = np.maximum(np.maximum(np.abs(dx), np.abs(dy)), np.abs(dz))
    if np.any(scale == 0.0):
        raise ValueError("point_m must differ from sat_m")
    # The angle is independent of the length of d; dividing by its largest
    # component keeps the products below away from overflow and underflow.
    dx, dy, dz = dx / scale, dy / scale, dz / scale
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    cx = by * dz - bz * dy
    cy = bz * dx - bx * dz
    cz = bx * dy - by * dx
    dot = bx * dx + by * dy + bz * dz
    return _full(np.degrees(np.arctan2(np.sqrt(cx * cx + cy * cy + cz * cz), dot)), shape)


def _ray_roots(origin: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the first hit (NaN for a miss) and the hit flag, for unit directions d."""
    a2 = WGS84_A_M * WGS84_A_M
    b2 = WGS84_B_M * WGS84_B_M
    x, y, z = origin[..., 0], origin[..., 1], origin[..., 2]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    qa = (dx * dx + dy * dy) / a2 + dz * dz / b2
    qb = 2.0 * ((x * dx + y * dy) / a2 + z * dz / b2)
    qc = (x * x + y * y) / a2 + z * z / b2 - 1.0
    disc = qb * qb - 4.0 * qa * qc
    real = disc >= 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        q = -0.5 * (qb + np.copysign(np.sqrt(np.where(real, disc, 0.0)), qb))
        repeated = q == 0.0
        t1 = np.where(repeated, -qb / (2.0 * qa), q / qa)
        t2 = np.where(repeated, t1, qc / q)
    t1 = np.where(real & (t1 > 0.0), t1, np.inf)
    t2 = np.where(real & (t2 > 0.0), t2, np.inf)
    t = np.minimum(t1, t2)
    is_hit = np.isfinite(t)
    return np.where(is_hit, t, np.nan), is_hit


def intersect_ray_wgs84_vec(origin_m, direction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First intersection of rays with the WGS84 ellipsoid.

    origin_m (m) and direction (any finite non-zero length) have shape (..., 3)
    and broadcast to S+(3,). Returns (hit_m, distance_m, is_hit): hit_m of
    shape S+(3,), distance_m of shape S (metres along the ray) and a bool
    array; hit_m and distance_m are NaN where is_hit is false.

    The policy is strict (plan section 6.1.6, Q5): the direction is normalised
    with a scaled norm, the discriminant is computed in float64 with no clamp,
    and the hit is the smallest root strictly greater than zero, from the
    stable form q = -(B + copysign(sqrt(D), B)) / 2, t1 = q / A, t2 = C / q. No
    tolerance band, no snapping and no minimum distance. An origin on or inside
    the ellipsoid is allowed and returns its positive root; RF callers pass
    exterior satellite positions only.
    """
    origin = _cartesian("origin_m", origin_m)
    direction_arr = _cartesian("direction", direction)
    shape = _broadcast_shape(origin_m=origin.shape[:-1], direction=direction_arr.shape[:-1])
    origin = np.broadcast_to(origin, shape + (3,))
    # Normalise (and refuse a zero direction) before broadcasting, so a zero
    # direction raises even when the other operand is empty.
    d = np.broadcast_to(_scaled_unit("direction", direction_arr), shape + (3,))
    distance, is_hit = _ray_roots(origin, d)
    hit = origin + distance[..., np.newaxis] * d
    return _out(hit), _out(distance), np.asarray(is_hit, dtype=bool)


def intersect_ray_wgs84(origin_m, direction):
    """Scalar form of intersect_ray_wgs84_vec(): None for a miss, else (hit_m 3-tuple, distance_m)."""
    origin = _scalar_vector("origin_m", origin_m)
    direction_arr = _scalar_vector("direction", direction)
    hit, distance, is_hit = intersect_ray_wgs84_vec(origin, direction_arr)
    if not bool(is_hit):
        return None
    return _tuple3(hit), float(distance)


def lvlh_reference_velocity_m_vec(r_ecef_m, vrel_mps) -> np.ndarray:
    """Inertial velocity on ECEF axes, v_ref = vrel + omega x r, in m/s; shape S+(3,).

    r_ecef_m (m) and vrel_mps (Earth-relative, m/s) have shape (..., 3) and
    broadcast together. An element whose vrel is exactly zero is a satellite
    standing still over the Earth: it gets omega x r (East) with no speed
    check. Any other element must pass check_inertial_speed_km_s(), converted
    once to km and km/s; its error message starts with
    FRAME_VELOCITY_UNITS_ERROR. Non-finite values and a zero position raise
    ValueError. One failing element fails the call.
    """
    r = _cartesian("r_ecef_m", r_ecef_m)
    v = _cartesian("vrel_mps", vrel_mps)
    shape = _broadcast_shape(r_ecef_m=r.shape[:-1], vrel_mps=v.shape[:-1])
    # Checked before broadcasting, so a zero position raises even when vrel is empty.
    if np.any(np.all(r == 0.0, axis=-1)):
        raise ValueError("r_ecef_m must be non-zero")
    r = np.broadcast_to(r, shape + (3,))
    v = np.broadcast_to(v, shape + (3,))
    omega_cross_r = np.stack(
        [-OMEGA_EARTH_RAD_S * r[..., 1], OMEGA_EARTH_RAD_S * r[..., 0], np.zeros(shape)], axis=-1
    )
    v_ref = v + omega_cross_r
    moving = ~np.all(v == 0.0, axis=-1)
    if np.any(moving):
        r_km = r[moving] / 1000.0
        v_km = v_ref[moving] / 1000.0
        radius = np.sqrt(np.sum(r_km * r_km, axis=-1))
        speed = np.sqrt(np.sum(v_km * v_km, axis=-1))
        lo, hi = _inertial_speed_band_km_s(radius)
        bad = ~((lo <= speed) & (speed <= hi))
        for index in np.flatnonzero(bad):
            # The scalar check builds and logs the same message as every other frame path.
            check_inertial_speed_km_s(_tuple3(r_km[index]), _tuple3(v_km[index]))
    return _out(v_ref)


def lvlh_reference_velocity_m(r_ecef_m, vrel_mps) -> tuple[float, float, float]:
    """Scalar form of lvlh_reference_velocity_m_vec(): a 3-tuple of floats in m/s."""
    r = _scalar_vector("r_ecef_m", r_ecef_m)
    v = _scalar_vector("vrel_mps", vrel_mps)
    return _tuple3(lvlh_reference_velocity_m_vec(r, v))
