"""Plan §4.5: every code the parser or validator can emit has registry metadata."""

import json
import re
from pathlib import Path

import pytest

from astra_shared import earth_fixed_validation, param_parsing
from astra_shared.earth_fixed_validation import (
    EARTH_FIXED_VALIDATION_REGISTRY,
    validate_earth_fixed_request,
    validation_error_detail,
)
from astra_shared.param_parsing import RfParamParseError, parse_rf_params

_CLUTTER_OFF = {"rf_schema_version": 1, "clutter_mode": "disabled", "clutter_percentile": 50.0}
_TARGET = {"label": "T", "target_lat": 0.0, "target_lon": 0.0}
_EARTH_FIXED = {**_CLUTTER_OFF, "pointing_mode": "earth_fixed", "targets": [_TARGET]}

# Each fixture drives one parser failure branch.
_PARSER_FIXTURES = [
    {**_CLUTTER_OFF, "modulation": "FOO"},
    {**_CLUTTER_OFF, "data_rate_bps": "bad"},
    {**_CLUTTER_OFF, "data_rate_bps": -1},
    {**_CLUTTER_OFF, "data_rate_bps": "inf"},
    {**_CLUTTER_OFF, "code_rate": "bad"},
    {**_CLUTTER_OFF, "code_rate": 0.33},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "nope"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_ref_bw_hz": "bad"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_ref_bw_hz": "nan"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_ref_bw_hz": 0},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "custom"},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "custom", "pfd_l0_dbw_m2": "bad", "pfd_l25_dbw_m2": -100},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "custom", "pfd_l0_dbw_m2": "inf", "pfd_l25_dbw_m2": -100},
    {**_CLUTTER_OFF, "compute_pfd": True, "pfd_limit_band": "custom", "pfd_l0_dbw_m2": -300, "pfd_l25_dbw_m2": -100},
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
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_id": "A"}, {**_TARGET, "target_id": "A"}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lat": "x"}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lat": 95}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lon": "x"}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_lon": 200}]},
    {**_EARTH_FIXED, "targets": [{**_TARGET, "target_alt_km": 1.0}]},
    {**_EARTH_FIXED, "targets": [_TARGET], "targets_json": json.dumps([{**_TARGET, "target_lat": 5.0}])},
    {**_CLUTTER_OFF, "pointing_mode": "earth_fixed", "targets": []},
    {**_EARTH_FIXED, "antenna_model": "phased_array"},
]

# Validator branches beyond the parser's own codes: (parsed-RF overrides, kwargs).
_VALIDATOR_FIXTURES = [
    ({}, {"analysis_mode": "footprint"}),
    ({}, {"analysis_mode": "coverage", "entry_point": "ui_cli"}),
    ({"assignment_rule": "highest_cn", "atmospheric_mode": "enable"}, {"entry_point": "scalar_point"}),
    (
        {"assignment_rule": "highest_cn", "clutter_mode": "p618"},
        {"entry_point": "rf_log", "probe_dynamic_adapters": True, "candidate_loss_probe_status": "ready"},
    ),
]


def _parser_codes(params):
    codes = []
    try:
        parse_rf_params(params)
    except RfParamParseError as exc:
        codes.extend(exc.codes)
    codes.extend(parse_rf_params(params, collect_errors=True).get("rf_param_errors", []))
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
    """Every quoted code literal the module raises or appends."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    # Scan emitting code only, not the registry's own message table.
    source = source.split("def dedupe_preserving_order", 1)[-1]
    patterns = (
        r'codes=\["([a-z0-9_]+)"\]',
        r'errors\.append\("([a-z0-9_]+)"\)',
        r'invalid_code="([a-z0-9_]+)"',
        r'out_of_range_code="([a-z0-9_]+)"',
        r'"(highest_cn_[a-z_]+_unavailable)"',
    )
    found = set()
    for pattern in patterns:
        found.update(code for code in re.findall(pattern, source) if code)
    return found


def test_every_code_reachable_from_fixtures_is_registered():
    codes = _emitted_codes()
    assert codes, "fixtures produced no codes"
    missing = sorted(code for code in codes if code not in EARTH_FIXED_VALIDATION_REGISTRY)
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
    detail = validation_error_detail(_parser_codes({**_CLUTTER_OFF, "modulation": "FOO"})[0])
    assert detail == {
        "code": "invalid_modulation",
        "message": EARTH_FIXED_VALIDATION_REGISTRY["invalid_modulation"]["message"],
        "field": "modulation",
        "category": "request",
    }
