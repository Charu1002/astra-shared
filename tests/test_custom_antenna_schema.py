#!/usr/bin/env python3

from astra_shared.custom_antenna_schema import (
    default_custom_antenna,
    normalize_custom_antenna,
)


def test_default_schema_is_canonical_and_disabled():
    payload = default_custom_antenna()
    assert payload["schema_version"] == 1
    assert payload["enabled"] is False
    assert payload["source_format"] is None
    assert payload["psi_deg"] == []
    assert payload["phi_deg"] == []
    assert payload["gain_dbi"] == []
    assert payload["validation"]["is_valid"] is False
    assert payload["validation"]["errors"] == ["no_pattern_loaded"]


def test_normalize_minimal_valid_enabled_payload():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": "12000000000",
            "psi_deg": [0, 10],
            "phi_deg": [0, 180],
            "gain_dbi": [[30, 20], [10, 0]],
        }
    )
    assert payload["enabled"] is True
    assert payload["source_format"] == "csv"
    assert payload["frequency_hz"] == 12_000_000_000.0
    assert payload["validation"]["is_valid"] is True
    assert payload["validation"]["errors"] == []


def test_normalize_reorders_phi_to_zero_360_convention_with_gain_columns():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12e9,
            "psi_deg": [0, 10],
            "phi_deg": [-90, 0, 90],
            "gain_dbi": [[30, 20, 10], [3, 2, 1]],
        }
    )
    assert payload["phi_deg"] == [0.0, 90.0, 270.0]
    assert payload["gain_dbi"] == [[20.0, 10.0, 30.0], [2.0, 1.0, 3.0]]
    assert payload["validation"]["is_valid"] is True


def test_normalize_rejects_shape_mismatch_with_deterministic_error_codes():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12e9,
            "psi_deg": [0, 10],
            "phi_deg": [0, 120, 240],
            "gain_dbi": [[30, 20], [10, 0]],
        }
    )
    assert payload["validation"]["is_valid"] is False
    assert "gain_cols_mismatch_phi" in payload["validation"]["errors"]


def test_normalize_handles_bad_types_with_safe_defaults():
    payload = normalize_custom_antenna(
        {
            "enabled": "yes",
            "source_format": "xlsx",
            "frequency_hz": "NaN",
            "psi_deg": [0, "bad"],
            "phi_deg": "wrong",
            "gain_dbi": [[1, 2], [3, "x"]],
        }
    )
    assert payload["enabled"] is True
    assert payload["source_format"] is None
    assert payload["frequency_hz"] is None
    assert payload["psi_deg"] == []
    assert payload["phi_deg"] == []
    assert payload["gain_dbi"] == []
    assert payload["validation"]["is_valid"] is False
    assert "missing_source_format" in payload["validation"]["errors"]


def test_disabled_payload_with_pattern_data_warns_but_is_valid():
    payload = normalize_custom_antenna(
        {
            "enabled": False,
            "psi_deg": [0],
            "phi_deg": [0],
            "gain_dbi": [[0]],
        }
    )
    assert payload["validation"]["is_valid"] is True
    assert payload["validation"]["errors"] == []
    assert "pattern_present_but_disabled" in payload["validation"]["warnings"]


def test_normalize_rejects_excessively_large_gain_table():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12e9,
            "psi_deg": list(range(1001)),
            "phi_deg": [v / 10.0 for v in range(1001)],
            "gain_dbi": [[0.0] * 1001 for _ in range(1001)],
        }
    )
    assert payload["validation"]["is_valid"] is False
    assert "gain_table_too_large" in payload["validation"]["errors"]


def test_normalize_warns_when_boresight_is_not_the_pattern_peak():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "ffd",
            "frequency_hz": 12e9,
            "psi_deg": [0, 30],
            "phi_deg": [0, 180],
            "gain_dbi": [[-3, -4], [0, -1]],
        }
    )
    assert "boresight_not_pattern_peak" in payload["validation"]["warnings"]


def test_normalize_warns_when_phi_coverage_is_limited():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12e9,
            "psi_deg": [0, 5],
            "phi_deg": [0, 90],
            "gain_dbi": [[0, 0], [-5, -10]],
        }
    )
    assert "phi_coverage_limited" in payload["validation"]["warnings"]


def test_normalize_rejects_non_monotonic_psi_grid():
    payload = normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12e9,
            "psi_deg": [30, 0],
            "phi_deg": [0, 180],
            "gain_dbi": [[10, 5], [20, 15]],
        }
    )
    assert payload["validation"]["is_valid"] is False
    assert "psi_grid_not_monotonic" in payload["validation"]["errors"]
