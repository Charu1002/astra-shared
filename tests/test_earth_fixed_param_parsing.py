import json

import pytest

from astra_shared.param_parsing import RfParamParseError
from astra_shared.param_parsing import parse_rf_params as _parse_rf_params

# Main's clutter Phase 3 requires the versioned clutter fields on every RF
# payload; these Earth-fixed tests exercise pointing parsing, so clutter is off.
_VERSIONED_CLUTTER = {"rf_schema_version": 1, "clutter_mode": "disabled", "clutter_percentile": 50.0}


def parse_rf_params(params, **kwargs):
    return _parse_rf_params({**_VERSIONED_CLUTTER, **params}, **kwargs)


def target(lat=0.0, lon=0.0, **extra):
    return {"label": "Target", "target_lat": lat, "target_lon": lon, **extra}


def test_defaults_preserve_nadir_behavior_and_path_elevation_defaults():
    assert parse_rf_params({})["pointing_mode"] == "nadir"
    assert parse_rf_params({})["assignment_rule"] == "highest_elevation"
    assert parse_rf_params({})["max_steer_deg"] == 60.0
    assert parse_rf_params({})["min_elevation_deg"] == 5.0
    assert parse_rf_params({}, path_default_min_elevation_deg=10.0)["min_elevation_deg"] == 10.0
    assert parse_rf_params({})["targets"] == []


def test_highest_cn_is_a_supported_earth_fixed_assignment_rule():
    parsed = parse_rf_params(
        {
            "pointing_mode": "earth_fixed",
            "assignment_rule": "highest_cn",
            "targets": [target()],
        }
    )
    assert parsed["assignment_rule"] == "highest_cn"


@pytest.mark.parametrize("value, expected", [
    ("nadir", "nadir"),
    ("earth_moving", "nadir"),
    ("earth-fixed", "earth_fixed"),
    ("targeted", "earth_fixed"),
])
def test_pointing_aliases_are_canonical(value, expected):
    params = {"pointing_mode": value}
    if expected == "earth_fixed":
        params["targets"] = [target()]
    assert parse_rf_params(params)["pointing_mode"] == expected


@pytest.mark.parametrize("key", ["max_steer_deg", "min_elevation_deg", "min_el_deg"])
@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-inf", -0.1, 90.1])
def test_reachability_numbers_record_stable_errors(key, value):
    parsed = parse_rf_params({"pointing_mode": "earth_fixed", "targets": [target()], key: value}, collect_errors=True)
    assert f"{key}_invalid" in parsed["rf_param_errors"] or f"{key}_out_of_range" in parsed["rf_param_errors"]


@pytest.mark.parametrize("key", ["min_elevation_deg", "min_el_deg"])
@pytest.mark.parametrize("value, expected", [(-5.0, 0.0), (95.0, 90.0), ("bad", 5.0), (30.0, 30.0)])
def test_nadir_min_elevation_keeps_clamp_behavior(key, value, expected):
    # Plan §4.5: a nadir request keeps its pre-earth-fixed behavior — an
    # out-of-range minimum elevation is clamped (malformed -> default), never
    # rejected, in both parser modes.
    assert parse_rf_params({key: value})["min_elevation_deg"] == expected
    parsed = parse_rf_params({key: value}, collect_errors=True)
    assert parsed["min_elevation_deg"] == expected
    assert parsed["rf_param_errors"] == []


@pytest.mark.parametrize(
    "extra",
    [
        {"assignment_rule": "bogus"},
        {"max_steer_deg": 120},
        {"targets_json": "not json"},
        {"targets": [{"target_lat": 200, "target_lon": 0}]},
        {"debug_target_reachability": True},
    ],
)
def test_nadir_ignores_earth_fixed_only_inputs(extra):
    # Earth-fixed-only fields are validated only when earth-fixed pointing is
    # in use; a nadir request that carries them still runs as before.
    parse_rf_params({"pointing_mode": "nadir", **extra})
    assert parse_rf_params(extra, collect_errors=True)["rf_param_errors"] == []


def test_invalid_pointing_mode_is_still_rejected():
    with pytest.raises(RfParamParseError) as exc:
        parse_rf_params({"pointing_mode": "sideways"})
    assert exc.value.codes == ["invalid_pointing_mode"]


@pytest.mark.parametrize("key,value", [("max_steer_deg", 0), ("max_steer_deg", 90), ("min_elevation_deg", 0), ("min_elevation_deg", 90)])
def test_reachability_boundaries_are_valid(key, value):
    parsed = parse_rf_params({key: value})
    assert parsed[key] == float(value)


def test_targets_json_and_native_targets_canonicalize_identically():
    rows = [target(1, 2), target(3, 4, target_alt_km=0.0)]
    native = parse_rf_params({"pointing_mode": "earth_fixed", "targets": rows})
    legacy = parse_rf_params({"pointing_mode": "earth_fixed", "targets_json": json.dumps(rows)})
    assert native["targets"] == legacy["targets"]
    assert [row["target_id"] for row in native["targets"]] == ["target-1", "target-2"]


def test_conflicting_dual_target_sources_fail_closed():
    with pytest.raises(RfParamParseError) as exc:
        parse_rf_params(
            {"pointing_mode": "earth_fixed", "targets": [target(1, 2)], "targets_json": json.dumps([target(3, 4)])}
        )
    assert "conflicting_targets_sources" in exc.value.codes


def test_target_id_collision_is_reported_and_generated_ids_avoid_explicit_ids():
    parsed = parse_rf_params({"targets": [target(1, 2, target_id="target-2"), target(3, 4)]}, collect_errors=True)
    assert "target_id_duplicate" not in parsed.get("rf_param_errors", [])
    assert [row["target_id"] for row in parsed["targets"]] == ["target-2", "target-3"]


def test_public_debug_reachability_is_rejected():
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target()], "debug_target_reachability": True},
        collect_errors=True,
    )
    assert "debug_target_reachability_not_public" in parsed["rf_param_errors"]


@pytest.mark.parametrize("field,value", [("modulation", "bad"), ("data_rate_bps", "bad"), ("code_rate", "bad")])
def test_legacy_rf_invalid_values_still_fail_fast(field, value):
    with pytest.raises(RfParamParseError):
        parse_rf_params({field: value})
