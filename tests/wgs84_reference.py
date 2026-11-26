"""Independent WGS84 reference calculations for the A0 geometry tests.

Written separately from astra_shared.geo_math so that production code is never
its own oracle (A0 plan v7, section 5.8). Forward conversion is the textbook
formula; the inverse is a fixed-point iteration, a different algorithm from the
production closed-form Bowring inverse.
"""

from __future__ import annotations

import math

A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)
E2 = F * (2.0 - F)


def forward(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    n = A / math.sqrt(1.0 - E2 * math.sin(phi) ** 2)
    return (
        (n + h_m) * math.cos(phi) * math.cos(lam),
        (n + h_m) * math.cos(phi) * math.sin(lam),
        (n * (1.0 - E2) + h_m) * math.sin(phi),
    )


def inverse(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Iterative geodetic inverse, converged to float64 precision."""
    p = math.hypot(x, y)
    if p == 0.0:
        return (90.0 if z >= 0 else -90.0), 0.0, abs(z) - B
    phi = math.atan2(z, p * (1.0 - E2))
    for _ in range(200):
        n = A / math.sqrt(1.0 - E2 * math.sin(phi) ** 2)
        phi_next = math.atan2(z + E2 * n * math.sin(phi), p)
        if phi_next == phi:
            break
        phi = phi_next
    s = math.sin(phi)
    n = A / math.sqrt(1.0 - E2 * s * s)
    if abs(phi) < math.radians(45.0):
        h = p / math.cos(phi) - n
    else:
        h = z / s - n * (1.0 - E2)
    return math.degrees(phi), math.degrees(math.atan2(y, x)), h


def enu_axes(lat_deg: float, lon_deg: float):
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    east = (-math.sin(lam), math.cos(lam), 0.0)
    north = (-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi))
    up = (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))
    return east, north, up


def dot(u, v) -> float:
    return sum(a * b for a, b in zip(u, v))


def satellite_at(lat_deg: float, lon_deg: float, el_deg: float, az_deg: float, h_sat_m: float):
    """Satellite ECEF seen from a ground point (height 0) at an elevation and azimuth.

    Bisection along the line of sight for the range at which the satellite's
    ellipsoidal height is h_sat_m, using this module's own inverse.
    """
    ground = forward(lat_deg, lon_deg, 0.0)
    east, north, up = enu_axes(lat_deg, lon_deg)
    el = math.radians(el_deg)
    az = math.radians(az_deg)
    los = [math.cos(el) * math.sin(az) * e + math.cos(el) * math.cos(az) * n + math.sin(el) * u
           for e, n, u in zip(east, north, up)]
    lo, hi = 0.0, 5.0e7
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        point = [g + mid * d for g, d in zip(ground, los)]
        if inverse(*point)[2] < h_sat_m:
            lo = mid
        else:
            hi = mid
    rng = 0.5 * (lo + hi)
    return tuple(g + rng * d for g, d in zip(ground, los)), rng
