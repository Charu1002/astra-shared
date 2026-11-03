#!/usr/bin/env python3

import pytest

from astra_shared.custom_antenna_interpolation import (
    lookup_custom_gain_db,
    lookup_custom_gain_db_array,
)


def _payload():
    return {
        "enabled": True,
        "source_format": "csv",
        "frequency_hz": 12_000_000_000,
        "psi_deg": [0.0, 30.0],
        "phi_deg": [0.0, 90.0],
        "gain_dbi": [[10.0, 6.0], [8.0, 4.0]],
        "validation": {"is_valid": True, "errors": [], "warnings": []},
    }


def test_exact_grid_hit_returns_exact_value():
    assert lookup_custom_gain_db(0.0, 90.0, _payload()) == 6.0


def test_bilinear_interpolation_center_cell():
    assert lookup_custom_gain_db(15.0, 45.0, _payload()) == pytest.approx(7.0, abs=1e-9)


def test_psi_clamp_low_and_high():
    payload = _payload()
    assert lookup_custom_gain_db(-5.0, 0.0, payload) == 10.0
    assert lookup_custom_gain_db(999.0, 0.0, payload) == 8.0


def test_array_lookup_clamps_psi_to_last_table_row():
    payload = _payload()
    payload["psi_deg"] = [float(psi) for psi in range(31)]
    payload["gain_dbi"] = [[-float(psi), -float(psi)] for psi in range(31)]

    for psi in (45.0, 89.9):
        assert lookup_custom_gain_db(psi, 0.0, payload) == pytest.approx(-30.0)

    result = lookup_custom_gain_db_array([45.0, 89.9], [0.0, 0.0], payload)
    assert result.tolist() == pytest.approx([-30.0, -30.0])


def test_phi_wrap_360_behavior():
    payload = _payload()
    assert lookup_custom_gain_db(
        0.0, 10.0, payload, phi_policy="wrap_360"
    ) == pytest.approx(
        lookup_custom_gain_db(0.0, 370.0, payload, phi_policy="wrap_360"),
        abs=1e-9,
    )


def test_phi_wrap_interpolates_across_periodic_seam_for_scalar_and_array():
    payload = _payload()
    payload["phi_deg"] = [0.0, 90.0, 180.0, 270.0]
    payload["gain_dbi"] = [[0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0]]

    assert lookup_custom_gain_db(0.0, 315.0, payload) == pytest.approx(15.0)
    assert lookup_custom_gain_db_array(
        [0.0], [315.0], payload
    ).tolist() == pytest.approx([15.0])


def test_nearest_fallback_with_single_row_grid():
    payload = _payload()
    payload["psi_deg"] = [0.0]
    payload["gain_dbi"] = [[10.0, 6.0]]
    assert lookup_custom_gain_db(20.0, 70.0, payload) in {10.0, 6.0}


def test_invalid_payload_returns_zero():
    payload = _payload()
    payload["validation"]["is_valid"] = False
    assert lookup_custom_gain_db(10.0, 20.0, payload) == 0.0


def test_vectorized_lookup_matches_scalar_pointwise():
    payload = _payload()
    psi = [0.0, 15.0, 30.0]
    phi = [0.0, 45.0, 90.0]
    out = lookup_custom_gain_db_array(psi, phi, payload)
    expected = [lookup_custom_gain_db(p, a, payload) for p, a in zip(psi, phi)]
    assert out.tolist() == pytest.approx(expected, abs=1e-9)


def test_none_payload_runtime_guard_returns_zero():
    assert lookup_custom_gain_db(10.0, 20.0, None) == 0.0
    assert lookup_custom_gain_db_array([10.0], [20.0], None).tolist() == [0.0]
