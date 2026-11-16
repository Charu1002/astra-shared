"""Regression coverage for RF-only non-finite input rejection (#88)."""

import pytest

from astra_shared.param_parsing import _RF_FLOAT_FIELDS, parse_rf_params
from astra_shared.rf_schema import (
    DEFAULT_OVERHEAD_PERCENT,
    DEFAULT_SHANNON_ATTENUATION_FACTOR,
    RfKeyError,
)


RF = {"rf_schema_version": 2, "clutter_mode": "disabled"}
INVALID = ("nan", "inf", "-inf", 10**400)


@pytest.mark.parametrize("field", _RF_FLOAT_FIELDS)
@pytest.mark.parametrize("value", INVALID)
def test_nonfinite_rf_input_names_its_field(field, value):
    with pytest.raises(RfKeyError) as caught:
        parse_rf_params({**RF, field: value})

    assert caught.value.key == field
    assert caught.value.codes == [f"{field}_invalid"]


@pytest.mark.parametrize("field", _RF_FLOAT_FIELDS)
def test_collect_errors_keeps_nonfinite_field_code(field):
    parsed = parse_rf_params({**RF, field: "nan"}, collect_errors=True)

    assert f"{field}_invalid" in parsed["rf_param_errors"]


@pytest.mark.parametrize("field", ("frequency_ghz", "frequency", "bandwidth_mhz"))
def test_rf_unit_conversion_cannot_overflow(field):
    with pytest.raises(RfKeyError) as caught:
        parse_rf_params({**RF, field: 1e308})

    assert caught.value.key == field


def test_bandwidth_mhz_unit_overflow_keeps_positive_finite_message():
    with pytest.raises(RfKeyError, match="positive finite") as caught:
        parse_rf_params({**RF, "bandwidth_mhz": 1e303})

    assert caught.value.key == "bandwidth_mhz"


@pytest.mark.parametrize(
    ("params", "field", "expected"),
    (
        (
            {"digital_link": False, "shannon_attenuation_factor": "nan"},
            "shannon_attenuation_factor",
            DEFAULT_SHANNON_ATTENUATION_FACTOR,
        ),
        (
            {
                "digital_link": True,
                "mcs_selection": "fixed",
                "shannon_attenuation_factor": 10**400,
            },
            "shannon_attenuation_factor",
            DEFAULT_SHANNON_ATTENUATION_FACTOR,
        ),
        (
            {"digital_link": False, "overhead_percent": "inf"},
            "overhead_percent",
            DEFAULT_OVERHEAD_PERCENT,
        ),
    ),
)
def test_hidden_nonfinite_conditional_value_uses_default(params, field, expected):
    parsed = parse_rf_params({**RF, **params})

    assert parsed[field] == expected


@pytest.mark.parametrize(
    ("field", "params"),
    (
        (
            "shannon_attenuation_factor",
            {
                "digital_link": True,
                "mcs_selection": "auto",
                "shannon_attenuation_factor": "nan",
            },
        ),
        (
            "overhead_percent",
            {"digital_link": True, "overhead_percent": "inf"},
        ),
    ),
)
def test_active_nonfinite_conditional_value_is_rejected(field, params):
    with pytest.raises(RfKeyError) as caught:
        parse_rf_params({**RF, **params})

    assert caught.value.key == field
    assert caught.value.codes == [f"{field}_invalid"]
