"""Plan §4.5: every code the parser or validator can emit has registry metadata."""

import ast
import json
from pathlib import Path

import pytest

from astra_shared import earth_fixed_validation, param_parsing
from astra_shared.earth_fixed_validation import (
    EARTH_FIXED_VALIDATION_REGISTRY,
    validate_earth_fixed_request,
    validation_error_detail,
)
from astra_shared.param_parsing import RfParamParseError, parse_rf_params

_CLUTTER_OFF = {
    "rf_schema_version": 1,
    "clutter_mode": "disabled",
    "clutter_percentile": 50.0,
}
_V2_CLUTTER_OFF = {"rf_schema_version": 2, "clutter_mode": "disabled"}
_TARGET = {"label": "T", "target_lat": 0.0, "target_lon": 0.0}
_EARTH_FIXED = {**_CLUTTER_OFF, "pointing_mode": "earth_fixed", "targets": [_TARGET]}

# Each fixture drives one parser failure branch.
_PARSER_FIXTURES = [
    {**_CLUTTER_OFF, "modulation": "FOO"},
    {**_CLUTTER_OFF, "digital_link": True},
    {**_CLUTTER_OFF, "mcs_selection": "auto"},
    {**_CLUTTER_OFF, "shannon_attenuation_factor": 0.6},
    {**_CLUTTER_OFF, "overhead_percent": 14},
    {**_V2_CLUTTER_OFF, "mcs_selection": "invalid"},
    {
        **_V2_CLUTTER_OFF,
        "digital_link": True,
        "mcs_selection": "auto",
        "shannon_attenuation_factor": 5,
    },
    {**_V2_CLUTTER_OFF, "digital_link": True, "overhead_percent": 60},
    {**_CLUTTER_OFF, "code_rate": "bad"},
    {**_CLUTTER_OFF, "code_rate": 0.33},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "nope"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_ref_bw_hz": "bad"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_ref_bw_hz": "nan"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_ref_bw_hz": 0},
    {**_V2_CLUTTER_OFF, "bandwidth_mhz": "bad"},
    {**_V2_CLUTTER_OFF, "bandwidth_hz": "bad"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "custom"},
    {
        **_CLUTTER_OFF,
        "compute_pfd": True,
        "pfd_limit_band": "custom",
        "pfd_l0_dbw_m2": "bad",
        "pfd_l25_dbw_m2": -100,
    },
    {
        **_CLUTTER_OFF,
        "compute_pfd": True,
        "pfd_limit_band": "custom",
        "pfd_l0_dbw_m2": "inf",
        "pfd_l25_dbw_m2": -100,
    },
    {
        **_CLUTTER_OFF,
        "compute_pfd": True,
        "pfd_limit_band": "custom",
        "pfd_l0_dbw_m2": -300,
        "pfd_l25_dbw_m2": -100,
    },
    {**_CLUTTER_OFF, "pointing_mode": "sideways"},
    {**_EARTH_FIXED, "assignment_rule": "bogus"},
    {**_EARTH_FIXED, "max_steer_deg": "bad"},
    {**_EARTH_FIXED, "max_steer_deg": 120},
    {**_EARTH_FIXED, "min_elevation_deg": "bad"},
    {**_EARTH_FIXED, "min_elevation_deg": 95},
    {**_EARTH_FIXED, "min_el_deg": "bad"},
    {**_EARTH_FIXED, "min_el_deg": 95},
    {**_EARTH_FIXED, "debug_target_reachability": True},
    {**_EARTH_FIXED, "targets_json": "not json"},
    {**_EARTH_FIXED, "targets": {"not": "a list"}},
    {**_EARTH_FIXED, "targets": ["not an object"]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_id": "  "}]},
    {
        **_EARTH_FIXED,
        "targets": [{**_TARGET, "target_id": "A"}, {**_TARGET, "target_id": "A"}],
    },
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lat": "x"}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lat": 95}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lon": "x"}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lon": 200}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_alt_km": 1.0}]},
    {
        **_EARTH_FIXED,
        "targets": [_TARGET],
        "targets_json": json.dumps([{**_TARGET, "target_lat": 5.0}]),
    },
    {**_CLUTTER_OFF, "pointing_mode": "earth_fixed", "targets": []},
    {**_EARTH_FIXED, "antenna_model": "phased_array"},
]

# Validator branches beyond the parser's own codes: (parsed-RF overrides, kwargs).
_VALIDATOR_FIXTURES = [
    ({}, {"analysis_mode": "footprint"}),
    ({}, {"analysis_mode": "coverage", "entry_point": "ui_cli"}),
    (
        {"assignment_rule": "highest_cn", "atmospheric_mode": "enable"},
        {"entry_point": "scalar_point"},
    ),
    (
        {"assignment_rule": "highest_cn", "clutter_mode": "p618"},
        {
            "entry_point": "rf_log",
            "probe_dynamic_adapters": True,
            "candidate_loss_probe_status": "ready",
        },
    ),
]


def _parser_codes(params):
    codes = []
    try:
        parse_rf_params(params)
    except RfParamParseError as exc:
        codes.extend(exc.codes)
    codes.extend(
        parse_rf_params(params, collect_errors=True).get("rf_param_errors", [])
    )
    return codes


def _emitted_codes():
    codes = set()
    for params in _PARSER_FIXTURES:
        codes.update(_parser_codes(params))
    for overrides, kwargs in _VALIDATOR_FIXTURES:
        rf = {**parse_rf_params(_EARTH_FIXED, collect_errors=True), **overrides}
        codes.update(validate_earth_fixed_request(rf, **kwargs))
    return codes


def _literal_codes(module):
    """Every literal ``code=`` emission; dynamic f-string codes are forbidden."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = set()
    dynamic_f_strings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in {"code", "invalid_code", "out_of_range_code"}:
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, str
            ):
                found.add(keyword.value.value)
            elif isinstance(keyword.value, ast.JoinedStr):
                dynamic_f_strings.append((node.lineno, ast.unparse(keyword.value)))
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)
    assert dynamic_f_strings == [], (
        f"RF validation codes must be registered literals: {dynamic_f_strings}"
    )
    return found


def test_every_code_reachable_from_fixtures_is_registered():
    codes = _emitted_codes()
    assert codes, "fixtures produced no codes"
    missing = sorted(
        code for code in codes if code not in EARTH_FIXED_VALIDATION_REGISTRY
    )
    assert missing == []


def test_fixtures_reach_every_literal_parser_and_validator_code():
    # A new literal code in the parser/validator must get a fixture here (and
    # therefore a registry entry), so the registry cannot silently fall behind.
    literals = _literal_codes(param_parsing) | _literal_codes(earth_fixed_validation)
    unreached = sorted(literals - _emitted_codes())
    assert unreached == []


@pytest.mark.parametrize("code", sorted(EARTH_FIXED_VALIDATION_REGISTRY))
def test_registered_codes_never_use_the_internal_fallback(code):
    detail = validation_error_detail(code)
    assert detail["category"] != "internal"
    assert detail["message"] != "Validation failed."


def test_bad_modulation_is_a_typed_request_error():
    detail = validation_error_detail(
        _parser_codes({**_CLUTTER_OFF, "modulation": "FOO"})[0]
    )
    assert detail == {
        "code": "invalid_modulation",
        "message": EARTH_FIXED_VALIDATION_REGISTRY["invalid_modulation"]["message"],
        "field": "modulation",
        "category": "request",
    }
