import json

import pytest

from astra_shared.param_parsing import RfParamParseError
from astra_shared.param_parsing import parse_rf_params as _parse_rf_params

# Main's clutter Phase 3 requires the versioned clutter fields on every RF
# payload; these Earth-fixed tests exercise pointing parsing, so clutter is off.
_VERSIONED_CLUTTER = {
    "rf_schema_version": 1,
    "clutter_mode": "disabled",
    "clutter_percentile": 50.0,
}


def parse_rf_params(params, **kwargs):
    return _parse_rf_params({**_VERSIONED_CLUTTER, **params}, **kwargs)


def target(lat=0.0, lon=0.0, **extra):
    return {"label": "Target", "target_lat": lat, "target_lon": lon, **extra}


def test_defaults_preserve_nadir_behavior_and_path_elevation_defaults():
    assert parse_rf_params({})["pointing_mode"] == "nadir"
    assert parse_rf_params({})["assignment_rule"] == "highest_elevation"
    assert parse_rf_params({})["max_steer_deg"] == 60.0
    assert parse_rf_params({})["min_elevation_deg"] == 5.0
    assert (
        parse_rf_params({}, path_default_min_elevation_deg=10.0)["min_elevation_deg"]
        == 10.0
    )
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


@pytest.mark.parametrize(
    "value, expected",
    [
        ("nadir", "nadir"),
        ("earth_moving", "nadir"),
        ("earth-fixed", "earth_fixed"),
        ("targeted", "earth_fixed"),
    ],
)
def test_pointing_aliases_are_canonical(value, expected):
    params = {"pointing_mode": value}
    if expected == "earth_fixed":
        params["targets"] = [target()]
    assert parse_rf_params(params)["pointing_mode"] == expected


@pytest.mark.parametrize("key", ["max_steer_deg", "min_elevation_deg", "min_el_deg"])
@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-inf", -0.1, 90.1])
def test_reachability_numbers_record_stable_errors(key, value):
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target()], key: value},
        collect_errors=True,
    )
    assert (
        f"{key}_invalid" in parsed["rf_param_errors"]
        or f"{key}_out_of_range" in parsed["rf_param_errors"]
    )


@pytest.mark.parametrize("key", ["min_elevation_deg", "min_el_deg"])
@pytest.mark.parametrize(
    "value, expected", [(-5.0, 0.0), (95.0, 90.0), ("bad", 5.0), (30.0, 30.0)]
)
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


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_steer_deg", 0),
        ("max_steer_deg", 90),
        ("min_elevation_deg", 0),
        ("min_elevation_deg", 90),
    ],
)
def test_reachability_boundaries_are_valid(key, value):
    parsed = parse_rf_params({key: value})
    assert parsed[key] == float(value)


def test_targets_json_and_native_targets_canonicalize_identically():
    rows = [target(1, 2), target(3, 4, target_alt_km=0.0)]
    native = parse_rf_params({"pointing_mode": "earth_fixed", "targets": rows})
    legacy = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets_json": json.dumps(rows)}
    )
    assert native["targets"] == legacy["targets"]
    assert [row["target_id"] for row in native["targets"]] == ["target-1", "target-2"]


def test_conflicting_dual_target_sources_fail_closed():
    with pytest.raises(RfParamParseError) as exc:
        parse_rf_params(
            {
                "pointing_mode": "earth_fixed",
                "targets": [target(1, 2)],
                "targets_json": json.dumps([target(3, 4)]),
            }
        )
    assert "conflicting_targets_sources" in exc.value.codes


def test_target_id_collision_is_reported_and_generated_ids_avoid_explicit_ids():
    parsed = parse_rf_params(
        {"targets": [target(1, 2, target_id="target-2"), target(3, 4)]},
        collect_errors=True,
    )
    assert "target_id_duplicate" not in parsed.get("rf_param_errors", [])
    assert [row["target_id"] for row in parsed["targets"]] == ["target-2", "target-3"]


def test_public_debug_reachability_is_rejected():
    parsed = parse_rf_params(
        {
            "pointing_mode": "earth_fixed",
            "targets": [target()],
            "debug_target_reachability": True,
        },
        collect_errors=True,
    )
    assert "debug_target_reachability_not_public" in parsed["rf_param_errors"]


@pytest.mark.parametrize("field,value", [("modulation", "bad"), ("code_rate", "bad")])
def test_legacy_rf_invalid_values_still_fail_fast(field, value):
    with pytest.raises(RfParamParseError):
        parse_rf_params({field: value})


@pytest.mark.parametrize("value", [0, -1, "abc", "inf", 10**400])
def test_legacy_data_rate_values_convert_to_link_off(value):
    parsed = parse_rf_params({"rf_schema_version": 1, "data_rate_bps": value})

    assert parsed["digital_link"] is False
    assert "data_rate_bps" not in parsed


# --- Review R8-S1 / R8-S2 / R9-S1: collect mode must keep main's Earth-Fixed semantics ---


def test_collect_mode_keeps_earth_fixed_mode_when_targets_are_missing():
    # The scalar point route builds a fallback target from ground_lat/lon when the
    # parsed request is earth_fixed with no targets; the parser must not turn the
    # request into nadir while collecting the error (R8-S1).
    parsed = parse_rf_params({"pointing_mode": "earth_fixed"}, collect_errors=True)
    assert parsed["pointing_mode"] == "earth_fixed"
    assert parsed["targets"] == []
    assert parsed["rf_param_errors"] == ["earth_fixed_targets_required"]


def test_collect_mode_keeps_valid_target_rows_and_reports_bad_ones():
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target(1, 2, target_id="t1"), 5]},
        collect_errors=True,
    )
    assert parsed["pointing_mode"] == "earth_fixed"
    assert [row["target_id"] for row in parsed["targets"]] == ["t1"]
    assert parsed["rf_param_errors"] == ["target_not_object"]


def test_collect_mode_reports_earth_fixed_and_rf_errors_together():
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target()], "max_steer_deg": 91, "modulation": "FOO"},
        collect_errors=True,
    )
    assert parsed["pointing_mode"] == "earth_fixed"
    assert set(parsed["rf_param_errors"]) == {"invalid_modulation", "max_steer_deg_out_of_range"}


@pytest.mark.parametrize("value", ["nearest", "lowest_range", False, 0])
def test_unimplemented_or_falsy_assignment_rule_is_rejected(value):
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target()], "assignment_rule": value},
        collect_errors=True,
    )
    assert "invalid_assignment_rule" in parsed["rf_param_errors"]


@pytest.mark.parametrize("value", [False, 0])
def test_falsy_pointing_mode_is_rejected(value):
    parsed = parse_rf_params({"pointing_mode": value, "targets": [target()]}, collect_errors=True)
    assert "invalid_pointing_mode" in parsed["rf_param_errors"]


@pytest.mark.parametrize(
    "row, code",
    [
        ({"target_lat": "abc", "target_lon": 1.0}, "target_lat_invalid"),
        ({"target_lat": 0.3, "target_lon": "abc"}, "target_lon_invalid"),
        ({"target_lat": 0.3, "target_lon": 1.0, "target_alt_km": "abc"}, "target_altitude_unsupported"),
        ({"target_lat": 10**400, "target_lon": 1.0}, "target_lat_invalid"),
        ({"target_lat": 0.3, "target_lon": 10**400}, "target_lon_invalid"),
    ],
)
def test_each_target_field_reports_its_own_code(row, code):
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [{"target_id": "t1", **row}]},
        collect_errors=True,
    )
    assert code in parsed["rf_param_errors"]


def test_targets_json_may_be_an_already_decoded_list():
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets_json": [target(1, 2, target_id="t1")]},
        collect_errors=True,
    )
    assert parsed["rf_param_errors"] == []
    assert [row["target_id"] for row in parsed["targets"]] == ["t1"]


def test_empty_string_targets_is_not_a_list():
    parsed = parse_rf_params({"pointing_mode": "earth_fixed", "targets": ""}, collect_errors=True)
    assert "targets_not_list" in parsed["rf_param_errors"]


def test_generated_id_that_collides_with_a_later_explicit_id_is_a_duplicate():
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target(1, 2), target(3, 4, target_id="target-1")]},
        collect_errors=True,
    )
    assert "target_id_duplicate" in parsed["rf_param_errors"]


@pytest.mark.parametrize("key", ["max_steer_deg", "min_elevation_deg"])
def test_overflowing_reachability_value_is_a_typed_error(key):
    parsed = parse_rf_params(
        {"pointing_mode": "earth_fixed", "targets": [target()], key: 10**400},
        collect_errors=True,
    )
    assert f"{key}_invalid" in parsed["rf_param_errors"]


def test_parsed_min_el_deg_matches_the_resolved_minimum_elevation():
    parsed = parse_rf_params({"min_elevation_deg": 12.0}, path_default_min_elevation_deg=10.0)
    assert parsed["min_el_deg"] == parsed["min_elevation_deg"] == 12.0
    assert parse_rf_params({}, path_default_min_elevation_deg=10.0)["min_el_deg"] == 10.0


def test_parse_error_codes_are_deduplicated():
    with pytest.raises(RfParamParseError) as info:
        parse_rf_params(
            {"pointing_mode": "earth_fixed", "targets": [{"target_lat": "x", "target_lon": 1}, {"target_lat": "y", "target_lon": 1}]}
        )
    assert info.value.codes.count("target_lat_invalid") == 1
