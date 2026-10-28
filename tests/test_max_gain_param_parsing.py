import pytest

from astra_shared.param_parsing import parse_rf_params


def _versioned_rf(**overrides):
    params = {
        "rf_schema_version": 1,
        "clutter_mode": "disabled",
        "clutter_percentile": 50.0,
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (-10, -10.0),
        (-20, -10.0),
        (-10.1, -10.0),
        (6.6, 6.6),
        (10, 10.0),
        (60, 60.0),
        (70, 60.0),
    ],
)
@pytest.mark.parametrize("model", ["gaussian", "bessel", "phased_array", "custom"])
def test_max_gain_lower_limit_is_minus_10_dbi(model, raw, expected):
    rf = parse_rf_params(_versioned_rf(antenna_model=model, max_gain_dbi=raw))

    assert rf["max_gain_dbi"] == expected


def test_max_gain_defaults_to_30_dbi():
    assert parse_rf_params(_versioned_rf())["max_gain_dbi"] == 30.0
    assert parse_rf_params(_versioned_rf(max_gain_dbi=""))["max_gain_dbi"] == 30.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(-10, 10.0), (6.6, 10.0), (10, 10.0), (35, 35.0), (70, 60.0)],
)
def test_itu_s672_keeps_10_dbi_minimum(raw, expected):
    rf = parse_rf_params(_versioned_rf(antenna_model="itu_s672", max_gain_dbi=raw))

    assert rf["max_gain_dbi"] == expected
