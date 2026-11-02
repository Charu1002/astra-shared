import math

import pytest

from astra_shared.clutter_config import ClutterConfigError, _require_version
from astra_shared.param_parsing import parse_rf_params
from astra_shared.rf_schema import (
    RF_SCHEMA_VERSION,
    RfConfigError,
    RfKeyError,
    RfParamParseError,
    complete_saved_rf_keys,
    convert_rf_block,
)


def _rf(**updates):
    block = {
        "rf_schema_version": 2,
        "clutter_mode": "disabled",
    }
    block.update(updates)
    return block


@pytest.mark.parametrize("raw, expected", [(1, 1), ("1", 1), (2, 2), ("2", 2)])
def test_require_version_accepts_both_supported_versions(raw, expected):
    assert _require_version({"rf_schema_version": raw}) == expected


@pytest.mark.parametrize("raw", [None, "", 0, 3, "3", True, 2.0])
def test_require_version_rejects_missing_or_unsupported_versions(raw):
    payload = {} if raw is None else {"rf_schema_version": raw}
    with pytest.raises(ClutterConfigError):
        _require_version(payload)


@pytest.mark.parametrize("value", [1, 1.0, "1", "1000000", " 2.5 "])
def test_convert_v1_positive_finite_non_boolean_data_rate_enables_link(value):
    converted, result = convert_rf_block(
        {
            "rf_schema_version": 1,
            "clutter_mode": "disabled",
            "data_rate_bps": value,
            "modulation": "16QAM",
            "code_rate": 0.75,
        }
    )

    assert converted == {
        "rf_schema_version": 2,
        "clutter_mode": "disabled",
        "digital_link": True,
        "mcs_selection": "fixed",
        "modulation": "16QAM",
        "code_rate": 0.75,
    }
    assert result.source_version == 1
    assert result.target_version == 2
    assert result.converted is True
    assert result.digital_link is True


@pytest.mark.parametrize(
    "value",
    [None, "", 0, -1, True, False, "abc", float("inf"), float("nan"), 10**400],
)
def test_convert_v1_other_data_rates_disable_link(value):
    converted, result = convert_rf_block(
        {
            "rf_schema_version": "1",
            "clutter_mode": "disabled",
            "data_rate_bps": value,
        }
    )
    assert converted["rf_schema_version"] == 2
    assert converted["digital_link"] is False
    assert converted["mcs_selection"] == "fixed"
    assert "data_rate_bps" not in converted
    assert result.digital_link is False


@pytest.mark.parametrize(
    "key",
    [
        "digital_link",
        "mcs_selection",
        "shannon_attenuation_factor",
        "overhead_percent",
    ],
)
def test_convert_v1_rejects_v2_only_keys(key):
    with pytest.raises(RfConfigError, match=rf"^{key} is not a key of rf_schema_version 1$"):
        convert_rf_block(
            {"rf_schema_version": 1, "clutter_mode": "disabled", key: None}
        )


def test_convert_v2_is_unchanged_and_reports_no_conversion():
    source = _rf(digital_link=True, mcs_selection="auto")
    converted, result = convert_rf_block(source)
    assert converted == source
    assert converted is not source
    assert result.source_version == 2
    assert result.target_version == 2
    assert result.converted is False
    assert result.digital_link is True


def test_convert_v2_rejects_data_rate_key_even_when_null():
    with pytest.raises(
        RfConfigError,
        match=r"^data_rate_bps is not a key of rf_schema_version 2; remove it$",
    ):
        convert_rf_block(_rf(data_rate_bps=None))


def test_complete_saved_rf_keys_fills_only_missing_values_and_normalizes_boolean():
    completed = complete_saved_rf_keys(
        _rf(
            digital_link="on",
            mcs_selection="",
            modulation=None,
            code_rate="",
            shannon_attenuation_factor=None,
            overhead_percent=0,
        )
    )
    assert completed["rf_schema_version"] == RF_SCHEMA_VERSION == 2
    assert completed["digital_link"] is True
    assert completed["mcs_selection"] == "fixed"
    assert completed["modulation"] == "QPSK"
    assert completed["code_rate"] == 1
    assert completed["shannon_attenuation_factor"] == 0.6
    assert completed["overhead_percent"] == 0


def test_complete_saved_rf_keys_preserves_supplied_hidden_or_invalid_values():
    completed = complete_saved_rf_keys(
        _rf(
            digital_link=False,
            mcs_selection="INVALID",
            modulation="256QAM",
            code_rate=0.5,
            shannon_attenuation_factor="abc",
            overhead_percent=1.5,
        )
    )
    assert completed["digital_link"] is False
    assert completed["mcs_selection"] == "INVALID"
    assert completed["shannon_attenuation_factor"] == "abc"
    assert completed["overhead_percent"] == 1.5


@pytest.mark.parametrize(
    "payload",
    [
        {"rf_schema_version": 1},
        {"rf_schema_version": 2, "data_rate_bps": None},
    ],
)
def test_complete_saved_rf_keys_requires_clean_version_2(payload):
    with pytest.raises(RfConfigError):
        complete_saved_rf_keys(payload)


def test_parser_returns_all_six_keys_with_defaults_and_accepts_256qam():
    parsed = parse_rf_params(_rf(modulation="256QAM"))
    assert parsed["rf_schema_version"] == 2
    assert parsed["digital_link"] is False
    assert parsed["mcs_selection"] == "fixed"
    assert parsed["modulation"] == "256QAM"
    assert parsed["code_rate"] == 1
    assert parsed["shannon_attenuation_factor"] == 0.6
    assert parsed["overhead_percent"] == 0
    assert "data_rate_bps" not in parsed


@pytest.mark.parametrize("value", [None, ""])
def test_parser_null_or_empty_mode_defaults_to_fixed(value):
    assert parse_rf_params(_rf(mcs_selection=value))["mcs_selection"] == "fixed"


@pytest.mark.parametrize("value", ["AUTO", "Fixed", " auto ", 1, False])
def test_parser_rejects_invalid_mode_even_when_link_is_off(value):
    with pytest.raises(RfConfigError, match=r"^mcs_selection must be fixed or auto$"):
        parse_rf_params(_rf(digital_link=False, mcs_selection=value))


@pytest.mark.parametrize("value", [0, -0.1, 1.1, "abc", True, float("inf")])
def test_parser_validates_alpha_only_when_link_on_auto(value):
    with pytest.raises(
        RfConfigError,
        match=r"^shannon_attenuation_factor must be in \(0, 1\]$",
    ):
        parse_rf_params(
            _rf(
                digital_link=True,
                mcs_selection="auto",
                shannon_attenuation_factor=value,
            )
        )


def test_parser_preserves_finite_unused_alpha_in_fixed_mode():
    parsed = parse_rf_params(
        _rf(
            digital_link=True,
            mcs_selection="fixed",
            shannon_attenuation_factor=1.5,
        )
    )
    assert parsed["shannon_attenuation_factor"] == 1.5


@pytest.mark.parametrize("value", [-1, 50, 51, "abc", True, float("inf")])
def test_parser_validates_overhead_only_when_link_on(value):
    with pytest.raises(RfConfigError, match=r"^overhead_percent must be in \[0, 50\)$"):
        parse_rf_params(_rf(digital_link=True, overhead_percent=value))


def test_parser_preserves_finite_unused_overhead_when_link_off():
    assert parse_rf_params(_rf(digital_link=False, overhead_percent=60))["overhead_percent"] == 60


def test_parser_names_code_rate_for_non_numeric_value():
    with pytest.raises(RfConfigError, match=r"^code_rate must be one of:"):
        parse_rf_params(_rf(code_rate="abc"))


def test_key_specific_rf_error_is_both_config_and_typed_parser_error():
    error = RfKeyError(
        "shannon_attenuation_factor must be in (0, 1]",
        key="shannon_attenuation_factor",
        code="shannon_attenuation_factor_invalid",
    )

    assert isinstance(error, RfConfigError)
    assert isinstance(error, RfParamParseError)
    assert error.codes == ["shannon_attenuation_factor_invalid"]


def test_parser_collects_multiple_automatic_mcs_errors_without_raising():
    parsed = parse_rf_params(
        _rf(
            digital_link=True,
            mcs_selection="auto",
            shannon_attenuation_factor=5,
            data_rate_bps=None,
        ),
        collect_errors=True,
    )

    assert parsed["parser_errors"] == [
        "data_rate_bps_not_v2_key",
        "shannon_attenuation_factor_invalid",
    ]
    assert parsed["digital_link"] is True
    assert parsed["mcs_selection"] == "auto"
    assert parsed["shannon_attenuation_factor"] == 0.6


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (
            {
                "rf_schema_version": 1,
                "clutter_mode": "disabled",
                "mcs_selection": "auto",
            },
            "mcs_selection_not_v1_key",
        ),
        (
            _rf(digital_link=True, overhead_percent=50),
            "overhead_percent_invalid",
        ),
        (
            _rf(bandwidth_mhz=0),
            "bandwidth_mhz_invalid",
        ),
    ],
)
def test_collect_mode_returns_stable_codes_and_never_uses_backstop(
    payload, expected_code
):
    parsed = parse_rf_params(payload, collect_errors=True)

    assert parsed["parser_errors"] == [expected_code]
    assert "rf_config_invalid" not in parsed["parser_errors"]


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"bandwidth_mhz": 12.5}, 12_500_000.0),
        ({"bandwidth_hz": 2_500_000}, 2_500_000.0),
        ({}, 10_000_000.0),
    ],
)
def test_parser_resolves_bandwidth_once(payload, expected):
    assert parse_rf_params(_rf(**payload))["bandwidth_hz"] == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"bandwidth_mhz": 0},
        {"bandwidth_mhz": -1},
        {"bandwidth_mhz": "abc"},
        {"bandwidth_hz": float("inf")},
    ],
)
def test_parser_rejects_supplied_non_positive_or_non_finite_bandwidth(payload):
    with pytest.raises(RfConfigError, match="bandwidth"):
        parse_rf_params(_rf(**payload))


def test_parser_converts_version_1_before_normal_parsing():
    parsed = parse_rf_params(
        {
            "rf_schema_version": 1,
            "clutter_mode": "disabled",
            "data_rate_bps": "1000000",
            "modulation": "QPSK",
            "code_rate": 0.5,
        }
    )
    assert parsed["rf_schema_version"] == 2
    assert parsed["digital_link"] is True
    assert parsed["mcs_selection"] == "fixed"
    assert parsed["modulation"] == "QPSK"
    assert math.isclose(parsed["code_rate"], 0.5)
