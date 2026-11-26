"""A0 ray-to-WGS84-ellipsoid intersection: G04 (behaviour) and G05 (Appendix B and accuracy).

A0 plan v7, section 6.1.6 (strict miss and tangency policy, Q5) and Appendix B.
References use exact fractions or 50-digit decimals on the represented inputs,
never the function under test.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest
import wgs84_reference as ref

from astra_shared.geo_math import (
    WGS84_A_M,
    WGS84_B_M,
    ecef_to_geodetic,
    ecef_to_geodetic_vec,
    intersect_ray_wgs84,
    intersect_ray_wgs84_vec,
)

HIT_TOL_M = 1e-3  # section 6.1.6 item 7
RESIDUAL_TOL = 1e-12

getcontext().prec = 50


def residual(point) -> float:
    x, y, z = point
    return (x * x + y * y) / WGS84_A_M**2 + z * z / WGS84_B_M**2 - 1.0


def decimal_first_hit(origin, direction):
    """Nearest positive root of the ray-ellipsoid quadratic, in 50-digit decimals."""
    o = [Decimal(v) for v in origin]
    d = [Decimal(v) for v in direction]
    norm = sum(v * v for v in d).sqrt()
    d = [v / norm for v in d]
    a2 = Decimal(WGS84_A_M) ** 2
    b2 = Decimal(WGS84_B_M) ** 2
    qa = (d[0] ** 2 + d[1] ** 2) / a2 + d[2] ** 2 / b2
    qb = 2 * ((o[0] * d[0] + o[1] * d[1]) / a2 + o[2] * d[2] / b2)
    qc = (o[0] ** 2 + o[1] ** 2) / a2 + o[2] ** 2 / b2 - 1
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return None
    roots = [(-qb - disc.sqrt()) / (2 * qa), (-qb + disc.sqrt()) / (2 * qa)]
    positive = [t for t in roots if t > 0]
    if not positive:
        return None
    t = min(positive)
    return [float(o[i] + t * d[i]) for i in range(3)], float(t)


# ---------------------------------------------------------------- G04


@pytest.mark.parametrize(("h", "expected_lat"), [(550000.0, 45.015299598460), (1200000.0, 45.030513685065)])
def test_g04_geocentric_nadir_hit_appendix_b(h, expected_lat):
    sat = np.array(ref.forward(45.0, 30.0, h))
    hit, distance = intersect_ray_wgs84(sat, -sat)
    lat, lon, alt = ecef_to_geodetic(*hit)
    assert lat == pytest.approx(expected_lat, abs=1e-9)  # surface-hit inverse tolerance, section 10.2
    assert lon == pytest.approx(30.0, abs=1e-9)
    assert abs(alt) <= 1e-6
    assert abs(residual(hit)) <= RESIDUAL_TOL
    assert distance == pytest.approx(math.dist(sat, hit), abs=HIT_TOL_M)


@pytest.mark.parametrize(("lat", "lon"), [(0.0, 0.0), (45.0, 30.0), (-60.0, 120.0), (80.0, -100.0)])
@pytest.mark.parametrize("sat_h", [550e3, 35786e3])
def test_g04_oblique_hit_lands_on_the_aimed_ground_point(lat, lon, sat_h):
    sub_lon = lon + 5.0
    sat = np.array(ref.forward(lat * 0.9, sub_lon, sat_h))
    ground = np.array(ref.forward(lat, lon, 0.0))
    result = intersect_ray_wgs84(sat, ground - sat)
    assert result is not None
    hit, distance = result
    assert math.dist(hit, ground) <= HIT_TOL_M
    assert distance == pytest.approx(math.dist(sat, ground), abs=HIT_TOL_M)
    assert abs(residual(hit)) <= RESIDUAL_TOL


def test_g04_explicit_exact_tangent_input():
    # Section 6.1.6 item 6: the line x = a, z = 0 touches the ellipsoid at (a, 0, 0).
    # Either classification is correct. A returned hit needs a finite positive
    # distance and a residual of at most 1e-12; there is no millimetre check (a
    # literal float64 evaluation lands about 0.58 m short of the tangent point).
    for origin, direction in (
        ((6378137.0, 42000000.0, 0.0), (0.0, -1.0, 0.0)),
        (np.array([[6378137.0, 42000000.0, 0.0]]), np.array([[0.0, -1.0, 0.0]])),
    ):
        hit, distance, is_hit = intersect_ray_wgs84_vec(origin, direction)
        if bool(np.all(is_hit)):
            assert np.all(np.isfinite(distance)) and np.all(distance > 0.0)
            assert abs(residual(np.reshape(hit, (-1, 3))[0])) <= RESIDUAL_TOL
    result = intersect_ray_wgs84((6378137.0, 42000000.0, 0.0), (0.0, -1.0, 0.0))
    if result is not None:
        hit, distance = result
        assert math.isfinite(distance) and distance > 0.0
        assert abs(residual(hit)) <= RESIDUAL_TOL


def test_g04_above_the_limb_misses():
    sat = np.array([4.2164e7, 0.0, 0.0])
    limb = math.asin(WGS84_A_M / 4.2164e7)
    for margin in (1e-3, 1e-5):
        angle = limb + margin
        direction = (-math.cos(angle), math.sin(angle), 0.0)
        assert intersect_ray_wgs84(sat, direction) is None


def test_g04_exterior_outward_ray_misses():
    assert intersect_ray_wgs84((7.0e6, 0.0, 0.0), (1.0, 0.0, 0.0)) is None
    assert intersect_ray_wgs84((7.0e6, 0.0, 0.0), (1.0, 0.1, 0.0)) is None


def test_g04_interior_origin_returns_its_positive_root():
    hit, distance = intersect_ray_wgs84((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert hit == pytest.approx((WGS84_A_M, 0.0, 0.0), abs=1e-6)
    assert distance == pytest.approx(WGS84_A_M, abs=1e-6)
    hit, distance = intersect_ray_wgs84((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    assert hit == pytest.approx((0.0, 0.0, -WGS84_B_M), abs=1e-6)


def test_g04_surface_origin_outward_has_no_positive_hit_and_inward_hits_the_far_side():
    surface = (WGS84_A_M, 0.0, 0.0)
    assert intersect_ray_wgs84(surface, (1.0, 0.0, 0.0)) is None
    hit, distance = intersect_ray_wgs84(surface, (-1.0, 0.0, 0.0))
    assert hit == pytest.approx((-WGS84_A_M, 0.0, 0.0), abs=1e-6)
    assert distance == pytest.approx(2.0 * WGS84_A_M, abs=1e-6)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_g04_direction_scale_does_not_change_the_result(scale):
    sat = np.array(ref.forward(10.0, 20.0, 800e3))
    ground = np.array(ref.forward(12.0, 21.0, 0.0))
    unit = (ground - sat) / np.linalg.norm(ground - sat)
    base_hit, base_distance = intersect_ray_wgs84(sat, unit)
    hit, distance = intersect_ray_wgs84(sat, unit * scale)
    assert math.dist(hit, base_hit) <= 1e-6
    assert distance == pytest.approx(base_distance, abs=1e-6)


@pytest.mark.parametrize(
    "direction", [(0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), (float("inf"), 1.0, 0.0)], ids=["zero", "nan", "inf"]
)
def test_g04_zero_or_non_finite_direction_raises(direction):
    with pytest.raises(ValueError, match="direction"):
        intersect_ray_wgs84((7e6, 0.0, 0.0), direction)
    with pytest.raises(ValueError, match="direction"):
        intersect_ray_wgs84_vec(np.array([[7e6, 0.0, 0.0]]), np.array([direction]))


def test_g04_non_finite_origin_raises():
    with pytest.raises(ValueError, match="origin_m"):
        intersect_ray_wgs84((float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0))


def test_g04_vector_form_marks_misses_with_nan():
    origins = np.array([[8e7, 0.0, 0.0], [7e6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    directions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hit, distance, is_hit = intersect_ray_wgs84_vec(origins, directions)
    assert hit.shape == (3, 3) and distance.shape == (3,) and is_hit.dtype == bool
    assert is_hit.tolist() == [True, False, True]
    assert np.all(np.isnan(hit[1])) and math.isnan(distance[1])
    # Select the hits before an inverse: the NaN rows of a miss would raise there.
    lat, _, _ = ecef_to_geodetic_vec(hit[is_hit, 0], hit[is_hit, 1], hit[is_hit, 2])
    assert lat.shape == (2,)
    with pytest.raises(ValueError, match="finite"):
        ecef_to_geodetic_vec(hit[:, 0], hit[:, 1], hit[:, 2])


def test_g04_vector_one_bad_element_fails_the_call():
    with pytest.raises(ValueError, match="direction"):
        intersect_ray_wgs84_vec(np.array([[8e7, 0.0, 0.0], [8e7, 0.0, 0.0]]), np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


# ---------------------------------------------------------------- G05 Appendix B controls


def _appendix_b_ray(delta: float):
    k = WGS84_A_M * (1.0 + delta) / 80000000.0
    return (80000000.0, 0.0, 0.0), (-math.sqrt(1.0 - k * k), k, 0.0)


def _exact_closest_approach_residual(origin, direction) -> Fraction:
    """r^2 dy^2 / (a^2 (dx^2 + dy^2)) - 1 on the represented inputs, exactly."""
    r = Fraction(origin[0])
    dx, dy = Fraction(direction[0]), Fraction(direction[1])
    return r * r * dy * dy / (Fraction(WGS84_A_M) ** 2 * (dx * dx + dy * dy)) - 1


@pytest.mark.parametrize(("delta", "approx_residual"), [(1e-13, 2e-13), (1e-12, 2e-12)])
def test_g05_appendix_b_physical_misses_are_required_misses(delta, approx_residual):
    origin, direction = _appendix_b_ray(delta)
    exact = _exact_closest_approach_residual(origin, direction)
    assert exact > 0  # the represented ray passes outside the ellipsoid
    assert float(exact) == pytest.approx(approx_residual, rel=0.05)
    assert intersect_ray_wgs84(origin, direction) is None
    assert decimal_first_hit(origin, direction) is None


def test_g05_historical_clamp_is_only_a_comparison_control():
    # Clamping a slightly negative discriminant to zero turned the 1e-13 miss into
    # a limb contact. The strict policy keeps it a miss.
    origin, direction = _appendix_b_ray(1e-13)
    norm = math.hypot(*direction)
    dx, dy = direction[0] / norm, direction[1] / norm
    qa = (dx * dx + dy * dy) / WGS84_A_M**2
    qb = 2.0 * origin[0] * dx / WGS84_A_M**2
    qc = origin[0] ** 2 / WGS84_A_M**2 - 1.0
    disc = qb * qb - 4.0 * qa * qc
    assert disc < 0.0
    clamped_t = -qb / (2.0 * qa)  # what max(disc, 0) would have returned
    assert clamped_t > 0.0
    assert intersect_ray_wgs84(origin, direction) is None


def test_g05_inside_case():
    # The exact tangent case is G04's explicit input (section 6.1.6 item 6); the
    # rounded delta = 0 ray is not a tangent reference. A ray aimed inside the limb hits.
    origin, direction = _appendix_b_ray(-1e-6)
    hit, distance = intersect_ray_wgs84(origin, direction)
    ref_hit, ref_distance = decimal_first_hit(origin, direction)
    assert math.dist(hit, ref_hit) <= HIT_TOL_M
    assert distance == pytest.approx(ref_distance, abs=HIT_TOL_M)


def test_g05_exact_surface_zero_root_is_not_a_hit():
    for surface in ((WGS84_A_M, 0.0, 0.0), (0.0, WGS84_A_M, 0.0), (0.0, 0.0, WGS84_B_M)):
        outward = surface
        assert intersect_ray_wgs84(surface, outward) is None


@pytest.mark.parametrize("height_m", [1.0, 100.0, 10000.0])
def test_g05_represented_near_surface_origins(height_m):
    # Section 6.1.6 item 7: the accuracy budget starts at a geodetic height of 1 m.
    origin = ref.forward(30.0, 40.0, height_m)
    down = -np.array(origin)
    hit, distance = intersect_ray_wgs84(origin, down)
    ref_hit, ref_distance = decimal_first_hit(origin, down)
    assert math.dist(hit, ref_hit) <= HIT_TOL_M
    assert distance == pytest.approx(ref_distance, abs=HIT_TOL_M)
    assert intersect_ray_wgs84(origin, -down) is None


def test_g05_one_metre_control():
    hit, distance = intersect_ray_wgs84((6378138.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert math.dist(hit, (WGS84_A_M, 0.0, 0.0)) <= HIT_TOL_M
    assert distance == pytest.approx(1.0, abs=HIT_TOL_M)


def test_g05_recorded_near_surface_origin_is_outside_the_contract():
    # Section 6.1.6 item 7, near-surface band: about 7.8e-12 m above the surface, C can
    # round to zero, and no root is promised. Recorded only: no exception.
    result = intersect_ray_wgs84((6378137.0, 0.01, 0.0), (-1.0, 0.0, 0.0))
    assert result is None or math.isfinite(result[1])


@pytest.mark.parametrize("component", [-5e-324, -1.4e308], ids=["subnormal", "near-overflow"])
def test_g05_extreme_direction_lengths_match_the_unit_direction(component):
    # Section 6.1.6 item 1: scale by the largest component, then normalise.
    origin = (7.0e6, 7.0e6, 0.0)
    unit_hit, unit_distance = intersect_ray_wgs84(origin, (-1.0, -1.0, 0.0))
    assert unit_distance == pytest.approx(3521357.936612, abs=HIT_TOL_M)
    hit, distance = intersect_ray_wgs84(origin, (component, component, 0.0))
    assert math.dist(hit, unit_hit) <= HIT_TOL_M and abs(distance - unit_distance) <= HIT_TOL_M
    hit_v, distance_v, is_hit = intersect_ray_wgs84_vec(np.array([origin]), np.array([[component, component, 0.0]]))
    assert bool(is_hit[0])
    assert math.dist(hit_v[0], unit_hit) <= HIT_TOL_M and abs(float(distance_v[0]) - unit_distance) <= HIT_TOL_M
    reference = decimal_first_hit(origin, (-1.0, -1.0, 0.0))
    assert math.dist(unit_hit, reference[0]) <= HIT_TOL_M


def test_g05_nearest_positive_root_with_cancellation():
    # From 1e8 m the near root is small against the far one; the stable form keeps it.
    origin = (1.0e8, 0.0, 0.0)
    direction = (-1.0, 1e-3, 0.0)
    hit, distance = intersect_ray_wgs84(origin, direction)
    ref_hit, ref_distance = decimal_first_hit(origin, direction)
    assert math.dist(hit, ref_hit) <= HIT_TOL_M
    assert distance == pytest.approx(ref_distance, abs=HIT_TOL_M)
    assert distance < math.dist(origin, (-WGS84_A_M, 0.0, 0.0))


def test_g05_ordinary_hit_accuracy_over_the_supported_domain():
    """Origins up to 1e8 m, rays at least 1e-6 rad from tangency (section 6.1.6 item 7)."""
    rng = random.Random(20261126)
    worst_position = worst_distance = worst_residual = 0.0
    cases = 0
    while cases < 400:
        radius = rng.choice([6.9e6, 7.6e6, 2.66e7, 4.2164e7, 1.0e8])
        u = [rng.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(sum(c * c for c in u))
        origin = [radius * c / n for c in u]
        target = ref.forward(rng.uniform(-89.0, 89.0), rng.uniform(-180.0, 180.0), 0.0)
        direction = [t - o for t, o in zip(target, origin)]
        # Keep rays clearly away from tangency: the target must see the origin above 1 degree.
        _, el, _ = _elevation(origin, target)
        if el < 1.0:
            continue
        reference = decimal_first_hit(origin, direction)
        result = intersect_ray_wgs84(origin, direction)
        assert result is not None and reference is not None
        hit, distance = result
        worst_position = max(worst_position, math.dist(hit, reference[0]))
        worst_distance = max(worst_distance, abs(distance - reference[1]))
        worst_residual = max(worst_residual, abs(residual(hit)))
        cases += 1
    assert worst_position <= HIT_TOL_M
    assert worst_distance <= HIT_TOL_M
    assert worst_residual <= RESIDUAL_TOL


def _elevation(sat, ground):
    lat, lon, _ = ref.inverse(*ground)
    up = ref.enu_axes(lat, lon)[2]
    los = [s - g for s, g in zip(sat, ground)]
    rng = math.sqrt(ref.dot(los, los))
    return rng, math.degrees(math.asin(ref.dot(los, up) / rng)), None



def test_g04_surface_tangent_has_no_zero_distance_hit():
    """Section 6.1.6 items 4 and 6: from (a, 0, 0) along +y the only root is exactly zero.

    Either classification of the tangent is allowed; a returned hit needs a finite
    distance strictly greater than zero.
    """
    origin, direction = (WGS84_A_M, 0.0, 0.0), (0.0, 1.0, 0.0)
    result = intersect_ray_wgs84(origin, direction)
    if result is not None:
        _, distance = result
        assert math.isfinite(distance) and distance > 0.0
    _, distance, is_hit = intersect_ray_wgs84_vec(np.array([origin, (7e6, 0.0, 0.0)]), np.array([direction, (-1.0, 0.0, 0.0)]))
    assert bool(is_hit[1])
    if bool(is_hit[0]):
        assert math.isfinite(float(distance[0])) and float(distance[0]) > 0.0


def test_g04_a_near_root_below_a_millimetre_is_returned():
    """Section 6.1.6 item 5: no minimum hit distance."""
    origin = (WGS84_A_M + 1e-4, 0.0, 0.0)
    expected = origin[0] - WGS84_A_M  # exact in float64: about 1.00000065e-4 m
    for result in (
        intersect_ray_wgs84(origin, (-1.0, 0.0, 0.0)),
        intersect_ray_wgs84_vec(np.array([origin]), np.array([(-1.0, 0.0, 0.0)])),
    ):
        assert result is not None
        distance = float(np.ravel(result[1])[0])
        assert 0.0 < distance < 1e-3
        assert distance == pytest.approx(expected, abs=1e-9)



def test_g04_interior_ray_with_a_small_positive_first_root():
    """Section 6.1.6 item 5: a hit 0.37 m along a ray from just inside the surface is returned.

    Origin (a - 1e-8, 0, 0) m, direction (-1e-9, 1, 0): the only positive root is about
    0.367934 m. Near the surface from inside, the float64 root carries a cancellation
    error of about 4e-4 m. The 1 mm tolerance here is this fixture's: section 6.1.6 item 7's
    budget is for exterior origins at least 1 m above the surface.
    """
    origin, direction = (WGS84_A_M - 1e-8, 0.0, 0.0), (-1e-9, 1.0, 0.0)
    expected_hit, expected = decimal_first_hit(origin, direction)
    assert expected == pytest.approx(0.367934, abs=1e-6)
    result = intersect_ray_wgs84(origin, direction)
    assert result is not None
    hit, distance = result
    assert distance == pytest.approx(expected, abs=HIT_TOL_M)
    assert hit == pytest.approx(expected_hit, abs=HIT_TOL_M)
    _, vec_distance, vec_is_hit = intersect_ray_wgs84_vec(np.array([origin]), np.array([direction]))
    assert bool(vec_is_hit[0])
    assert float(vec_distance[0]) == pytest.approx(expected, abs=HIT_TOL_M)


def test_g04_zero_direction_raises_even_against_an_empty_origin():
    """Sections 6.1.1 and 6.1.6: a supplied zero direction is refused before broadcasting."""
    for origin in (np.empty((0, 3)), np.array([[7e6, 0.0, 0.0]]), (7e6, 0.0, 0.0)):
        with pytest.raises(ValueError, match="direction must be non-zero"):
            intersect_ray_wgs84_vec(origin, (0.0, 0.0, 0.0))
    # Genuinely empty valid calls stay empty.
    hit, distance, is_hit = intersect_ray_wgs84_vec(np.empty((0, 3)), (-1.0, 0.0, 0.0))
    assert hit.shape == (0, 3) and distance.shape == (0,) and is_hit.shape == (0,)
    hit, distance, is_hit = intersect_ray_wgs84_vec((7e6, 0.0, 0.0), np.empty((0, 3)))
    assert hit.shape == (0, 3) and distance.shape == (0,) and is_hit.shape == (0,)



@pytest.mark.parametrize("plane", ["equatorial", "polar"])
@pytest.mark.parametrize("margin_rad", [1.5e-6, 3e-6, 1e-5, 1e-4])
def test_g05_high_radius_grazing_rays_near_the_tangency_exclusion(plane, margin_rad):
    """Origins at 1e8 m, rays 1.5e-6 to 1e-4 rad inside (hits) and outside (misses) the limb.

    Section 6.1.6 item 7 excludes only rays within 1e-6 rad of tangency; the random G05 set
    can't reach this edge because it keeps targets above 1 deg. In the plane of the origin
    and the axis, the limb is at tan(alpha) = semi-axis / sqrt(R^2 - other semi-axis^2).
    """
    radius = 1.0e8
    if plane == "equatorial":  # origin on +x, rays in the xy plane: a circle of radius a
        origin, side_axis, along, across = (radius, 0.0, 0.0), WGS84_A_M, 0, 1
        alpha = math.atan(WGS84_A_M / math.sqrt(radius**2 - WGS84_A_M**2))
    else:  # origin over the north pole, rays in the xz plane: semi-axes a across, b along
        origin, side_axis, along, across = (0.0, 0.0, radius), WGS84_A_M, 2, 0
        alpha = math.atan(WGS84_A_M / math.sqrt(radius**2 - WGS84_B_M**2))
    del side_axis
    for theta, should_hit in ((alpha - margin_rad, True), (alpha + margin_rad, False)):
        direction = [0.0, 0.0, 0.0]
        direction[along] = -math.cos(theta)
        direction[across] = math.sin(theta)
        result = intersect_ray_wgs84(origin, direction)
        reference = decimal_first_hit(origin, direction)
        vec_hit, vec_distance, vec_is_hit = intersect_ray_wgs84_vec(np.array([origin]), np.array([direction]))
        assert (result is not None) is should_hit and (reference is not None) is should_hit
        assert bool(vec_is_hit[0]) is should_hit
        if should_hit:
            hit, distance = result
            assert math.dist(hit, reference[0]) <= HIT_TOL_M
            assert distance == pytest.approx(reference[1], abs=HIT_TOL_M)
            assert abs(residual(hit)) <= RESIDUAL_TOL
            assert float(vec_distance[0]) == pytest.approx(distance, abs=HIT_TOL_M)
        else:
            assert np.all(np.isnan(vec_hit)) and np.isnan(vec_distance[0])
