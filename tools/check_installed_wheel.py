"""Check the installed astra_shared wheel: the A0 API inventory and a few numerical controls.

A0 plan v7, sections 6.1, 8.1 and G06. CI runs this from outside the checkout with
the wheel's own interpreter, so the import is the installed package, not the source:

    cd /tmp && /tmp/wheel-env/bin/python "$GITHUB_WORKSPACE/tools/check_installed_wheel.py" /tmp/wheel-env/

The argument is the prefix the imported package must live under. Any failure exits
non-zero. The expected values are independent of the module: published constants and
closed-form geometry.
"""

from __future__ import annotations

import math
import sys

import astra_shared
import astra_shared.geo_math as g

# Section 6.1: the A0 API, and section 6.1.8: the names that stay.
CONSTANTS = ["WGS84_A_M", "WGS84_F", "WGS84_B_M", "WGS84_E2", "OMEGA_EARTH_RAD_S", "FRAME_VELOCITY_UNITS_ERROR"]
FUNCTIONS = [
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "geodetic_to_ecef_vec",
    "ecef_to_geodetic_vec",
    "enu_basis",
    "enu_basis_vec",
    "ecef_delta_to_enu",
    "ecef_delta_to_enu_vec",
    "look_angles_ecef",
    "look_angles_ecef_vec",
    "nadir_unit_vec",
    "off_axis_deg_vec",
    "intersect_ray_wgs84",
    "intersect_ray_wgs84_vec",
    "lvlh_reference_velocity_m",
    "lvlh_reference_velocity_m_vec",
    "distance_3d",
    "haversine_km",
    "compute_elevation",
    "compute_elevation_vec",
    "latlon_to_ecef",
    "teme_to_ecef",
    "teme_state_to_ecef",
    "lvlh_reference_velocity",
    "check_inertial_speed_km_s",
]

A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)


def close(got, expected, tol, what):
    if not (isinstance(got, float) and math.isfinite(got) and abs(got - expected) <= tol):
        raise SystemExit(f"FAIL {what}: got {got!r}, expected {expected!r} within {tol}")


def main(prefix: str) -> None:
    if not astra_shared.__file__.replace("\\", "/").startswith(prefix.replace("\\", "/")):
        raise SystemExit(f"FAIL import: astra_shared is {astra_shared.__file__}, not under {prefix}")

    missing = [name for name in CONSTANTS + FUNCTIONS if not hasattr(g, name)]
    if missing:
        raise SystemExit(f"FAIL inventory: missing {missing}")
    not_callable = [name for name in FUNCTIONS if not callable(getattr(g, name))]
    if not_callable:
        raise SystemExit(f"FAIL inventory: not callable {not_callable}")
    if g.WGS84_A_M != A or g.WGS84_F != F or abs(g.WGS84_B_M - B) > 1e-9:
        raise SystemExit("FAIL constants: WGS84 a, f or b")

    # Ray: from 8e7 m on +x toward the centre, the first hit is (a, 0, 0) at 8e7 - a.
    result = g.intersect_ray_wgs84((8e7, 0.0, 0.0), (-1.0, 0.0, 0.0))
    if result is None:
        raise SystemExit("FAIL ray: a miss for a ray aimed at the centre")
    hit, distance = result
    close(distance, 8e7 - A, 1e-6, "ray distance")
    close(hit[0], A, 1e-6, "ray hit x")
    _, _, is_hit = g.intersect_ray_wgs84_vec([[8e7, 0.0, 0.0], [8e7, 0.0, 0.0]], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    if list(map(bool, is_hit)) != [True, False]:
        raise SystemExit(f"FAIL ray vector: is_hit {is_hit!r}, expected [True, False]")

    # Look angles: at latitude 0, longitude 0 the axes are E = +y, N = +z, U = +x.
    rng, el, az = g.look_angles_ecef((A + 1e6, 0.0, 1e6), (A, 0.0, 0.0), 0.0, 0.0)
    close(rng, math.sqrt(2.0) * 1e6, 1e-6, "look-angle range")
    close(el, 45.0, 1e-9, "look-angle elevation")
    close(az, 0.0, 1e-9, "look-angle azimuth")
    rng, el, az = g.look_angles_ecef((A, 1e6, 0.0), (A, 0.0, 0.0), 0.0, 0.0)
    close(el, 0.0, 1e-9, "horizon elevation")
    close(az, 90.0, 1e-9, "horizon azimuth (east)")

    # Inverse round trip within the section 6.1.3 budget (1e-6 deg, 1e-6 m; the closed
    # form is about 2e-8 deg off at 550 km), and the exact pole of the A0 height formula.
    lat, lon, h = 45.0, 30.0, 550e3
    x, y, z = g.geodetic_to_ecef(lat, lon, h)
    got = g.ecef_to_geodetic(x, y, z)
    close(got[0], lat, 1e-6, "round-trip latitude")
    close(got[1], lon, 1e-6, "round-trip longitude")
    close(got[2], h, 1e-6, "round-trip height")
    vec = g.ecef_to_geodetic_vec(x, y, z)
    close(float(vec[0]), got[0], 1e-12, "scalar and vector inverse parity")
    close(g.ecef_to_geodetic(0.0, 0.0, B + 550e3)[2], 550e3, 1e-6, "polar-axis height")

    print(f"installed wheel OK: {astra_shared.__file__} ({len(FUNCTIONS)} functions, {len(CONSTANTS)} constants)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_installed_wheel.py <expected install prefix>")
    main(sys.argv[1])
