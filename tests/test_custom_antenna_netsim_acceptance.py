#!/usr/bin/env python3
"""Custom Antenna file acceptance matches NetSim (comparison report D8, D15-D17).

Each test is one file case from the review of comparison report revision 4
(R-1, R-2): which files both products accept or reject, and how an accepted
file is cleaned up (phi wrapping, duplicates, pole rows).
"""

import math

import pytest

from astra_shared.custom_antenna_csv_parser import (
    CsvErrorCode,
    CustomAntennaCsvError,
    parse_custom_antenna_csv_text,
)
from astra_shared.custom_antenna_errors import CustomAntennaErrorCode
from astra_shared.custom_antenna_ffd_parser import (
    CustomAntennaFfdError,
    parse_custom_antenna_ffd_text,
)
from astra_shared.custom_antenna_schema import (
    normalize_custom_antenna,
    pole_row_representative_db,
    wrap_phi_deg,
)

CSV_HEADER = "psi_deg,phi_deg,gain_dbi\n"
FFD_HEADER = "frequency_hz=12000000000\ntheta phi e_theta e_phi\n"


def _csv(rows):
    return CSV_HEADER + "".join(f"{psi},{phi},{gain}\n" for psi, phi, gain in rows)


def _ffd(rows):
    return FFD_HEADER + "".join(f"{t} {p} {e} 0.0\n" for t, p, e in rows)


def _csv_error(text):
    with pytest.raises(CustomAntennaCsvError) as exc:
        parse_custom_antenna_csv_text(text)
    return exc.value


def _ffd_error(text):
    with pytest.raises(CustomAntennaFfdError) as exc:
        parse_custom_antenna_ffd_text(text)
    return exc.value


def _enabled_payload(psi_deg, phi_deg, gain_dbi, enabled=True):
    return normalize_custom_antenna(
        {
            "enabled": enabled,
            "source_format": "csv",
            "frequency_hz": 12e9,
            "psi_deg": psi_deg,
            "phi_deg": phi_deg,
            "gain_dbi": gain_dbi,
        }
    )


# --- wrap_phi_deg ----------------------------------------------------------


@pytest.mark.parametrize(
    ("phi", "expected"),
    [
        (0.0, 0.0),
        (360.0, 0.0),
        (-90.0, 270.0),
        (-180.0, 180.0),
        (359.9999999999, 359.9999999999),
        (-1e-10, 360.0 - 1e-10),
        # '%' alone gives 360.0 here; NetSim's loops give 0.0.
        (-1e-20, 0.0),
    ],
)
def test_wrap_phi_deg_matches_netsim_loops(phi, expected):
    assert wrap_phi_deg(phi) == expected
    assert 0.0 <= wrap_phi_deg(phi) < 360.0


# --- CSV parser ------------------------------------------------------------


def test_csv_phi_0_and_360_is_one_column_and_is_rejected_d15():
    # Two raw phi values, but one azimuth after wrapping.
    text = _csv([(1, 0, 30), (1, 360, 30), (10, 0, 20), (10, 360, 20)])
    assert _csv_error(text).code == CsvErrorCode.INSUFFICIENT_PHI_POINTS


def test_csv_mixed_phi_convention_is_rejected_at_first_mixed_row_d16():
    # Line 5 is the first row where both a negative phi and a phi above 180
    # have been seen.
    text = _csv([(1, -90, 30), (1, 0, 30), (1, 90, 30), (1, 200, 30), (10, -90, 20)])
    err = _csv_error(text)
    assert err.code == CsvErrorCode.MIXED_PHI_CONVENTION
    assert err.code == "ca_validate_mixed_phi_convention"
    assert err.line_no == 5


def test_csv_mixed_phi_convention_order_does_not_matter_d16():
    text = _csv([(1, 200, 30), (1, 0, 30), (1, -90, 30)])
    err = _csv_error(text)
    assert err.code == CsvErrorCode.MIXED_PHI_CONVENTION
    assert err.line_no == 4


def test_csv_signed_phi_up_to_180_is_not_mixed():
    # -180..180 is one convention: 180 is not above 180.
    text = _csv(
        [
            (1, -180, 30),
            (1, -90, 30),
            (1, 0, 30),
            (1, 90, 30),
            (1, 180, 30),
            (10, -180, 20),
            (10, -90, 21),
            (10, 0, 22),
            (10, 90, 23),
            (10, 180, 20),
        ]
    )
    payload = parse_custom_antenna_csv_text(text)
    # -180 and 180 name one azimuth, and they agree, so one column remains.
    assert payload["phi_deg"] == [0.0, 90.0, 180.0, 270.0]
    assert payload["gain_dbi"][1] == [-8.0, -7.0, -10.0, -9.0]


def test_csv_signed_phi_conflict_at_180_is_rejected_d17():
    text = _csv(
        [
            (1, -180, 30),
            (1, 0, 30),
            (1, 180, 30),
            (10, -180, 20),
            (10, 0, 22),
            (10, 180, 20.5),
        ]
    )
    assert _csv_error(text).code == CsvErrorCode.DUPLICATE_GRID_POINT


def test_csv_0_and_360_columns_that_conflict_are_rejected_d17():
    # The review's case: the 360 column is 0.7 dB higher.  Before, Astra
    # loaded it, dropped the 360 data and, because the dropped column held
    # the peak, left every gain 0.7 dB low.
    rows = []
    for psi, base in ((1, 30.0), (10, 20.0)):
        for phi in (0, 90, 180, 270):
            rows.append((psi, phi, base))
        rows.append((psi, 360, base + 0.7))
    assert _csv_error(_csv(rows)).code == CsvErrorCode.DUPLICATE_GRID_POINT


def test_csv_0_and_360_columns_that_agree_keep_the_first_d17():
    rows = []
    for psi, base in ((1, 30.0), (10, 20.0)):
        for phi in (0, 90, 180, 270, 360):
            rows.append((psi, phi, base + phi / 1000.0 if phi != 360 else base))
    payload = parse_custom_antenna_csv_text(_csv(rows))
    assert payload["phi_deg"] == [0.0, 90.0, 180.0, 270.0]
    assert payload["gain_dbi"][0][0] == pytest.approx(-0.27)


def test_csv_exact_repeated_row_is_accepted_d17():
    text = _csv([(1, 0, 30), (1, 90, 30), (10, 90, -1.0), (10, 90, -1.0), (10, 0, 20)])
    payload = parse_custom_antenna_csv_text(text)
    assert payload["psi_deg"] == [1.0, 10.0]
    assert payload["phi_deg"] == [0.0, 90.0]
    assert payload["gain_dbi"] == [[0.0, 0.0], [-10.0, -31.0]]


def test_csv_repeat_within_1e9_db_keeps_the_first_copy_d17():
    text = _csv(
        [(1, 0, 30), (1, 90, 30), (10, 0, 20), (10, 90, 10), (10, 90, 10.0000000005)]
    )
    payload = parse_custom_antenna_csv_text(text)
    assert payload["gain_dbi"][1][1] == -20.0


def test_csv_zero_and_360_alias_across_psi_rows_is_complete_d17():
    # One psi row names the azimuth 0, the other 360.  NetSim accepts this;
    # Astra used to report an incomplete table.
    text = _csv([(1, 0, 30), (1, 180, 30), (10, 360, 20), (10, 180, 15)])
    payload = parse_custom_antenna_csv_text(text)
    assert payload["phi_deg"] == [0.0, 180.0]
    assert payload["gain_dbi"] == [[0.0, 0.0], [-10.0, -15.0]]


@pytest.mark.parametrize("pole_psi", [0, 180])
def test_csv_pole_row_spread_above_0p1_db_is_rejected_d8a(pole_psi):
    other = 10 if pole_psi == 0 else 170
    text = _csv(
        [(pole_psi, 0, 30), (pole_psi, 180, 28.5), (other, 0, 20), (other, 180, 15)]
    )
    err = _csv_error(text)
    assert err.code == CsvErrorCode.POLE_ROW_INCONSISTENT
    assert err.code == "ca_validate_pole_row_inconsistent"


def test_csv_pole_row_spread_within_0p1_db_is_accepted_d8a():
    text = _csv([(0, 0, 30), (0, 180, 29.91), (10, 0, 20), (10, 180, 15)])
    payload = parse_custom_antenna_csv_text(text)
    assert payload["validation"]["errors"] == ["missing_frequency_hz"]


def test_csv_pole_row_is_power_averaged_before_peak_normalization_d8b():
    # Boresight row spreads by 0.08 dB.  NetSim power-averages it and
    # normalizes to the averaged value, so the whole pattern sits higher than
    # with normalization to the raw 30 dBi maximum.
    text = _csv(
        [
            (0, 0, 30),
            (0, 90, 29.92),
            (0, 180, 30),
            (0, 270, 29.92),
            (10, 0, 20),
            (10, 90, 20),
            (10, 180, 20),
            (10, 270, 20),
        ]
    )
    payload = parse_custom_antenna_csv_text(text)
    representative = 30 + 10 * math.log10((2 + 2 * 10 ** (-0.008)) / 4)
    assert representative == pytest.approx(
        pole_row_representative_db([30, 29.92, 30, 29.92])
    )
    assert payload["gain_dbi"][0] == [0.0, 0.0, 0.0, 0.0]
    for value in payload["gain_dbi"][1]:
        assert value == pytest.approx(20 - representative)
    # Up to 0.1 dB higher than normalizing to the raw maximum.
    assert 0 < payload["gain_dbi"][1][0] - (20 - 30) <= 0.1


def test_csv_pole_check_runs_after_duplicate_and_completeness_checks():
    # Pole row spread is 2 dB, but the table is incomplete: NetSim reports the
    # completeness error first.
    text = _csv([(0, 0, 30), (0, 180, 28), (10, 0, 20)])
    assert _csv_error(text).code == CsvErrorCode.INCOMPLETE_TABLE


# --- FFD parser ------------------------------------------------------------


def test_ffd_single_theta_is_rejected_before_duplicate_check_d15():
    # One theta value and a conflicting duplicate: NetSim reports the count.
    text = _ffd([(10, 0, 1.0), (10, 90, 0.9), (10, 90, 0.5)])
    err = _ffd_error(text)
    assert err.code == CustomAntennaErrorCode.CSV_INSUFFICIENT_PSI_POINTS


def test_ffd_single_phi_after_360_wrap_is_rejected_d15():
    text = _ffd([(0, 0, 1.0), (0, 360, 1.0), (10, 0, 0.7), (10, 360, 0.7)])
    err = _ffd_error(text)
    assert err.code == CustomAntennaErrorCode.CSV_INSUFFICIENT_PHI_POINTS


def test_ffd_low_power_duplicate_conflict_is_rejected_in_db_d17():
    # E = 1e-8 and 5e-7 are 34 dB apart, both below 1e-12 power.  The old
    # abs_tol=1e-12 test accepted this and silently dropped the second value.
    text = _ffd(
        [(0, 0, 1.0), (0, 90, 1.0), (10, 0, 0.5), (10, 90, 1e-8), (10, 90, 5e-7)]
    )
    err = _ffd_error(text)
    assert err.code == CustomAntennaErrorCode.FFD_DUPLICATE_GRID_POINT
    assert err.line_no == 7


def test_ffd_duplicate_within_1e9_db_keeps_the_first_copy_d17():
    e = 0.5
    e_close = e * 10 ** (0.4e-9 / 20)  # 0.4e-9 dB in power
    text = _ffd(
        [(0, 0, 1.0), (0, 90, 1.0), (10, 0, e), (10, 0, e_close), (10, 90, 0.25)]
    )
    out = parse_custom_antenna_ffd_text(text)
    assert out["validation"]["is_valid"] is True
    assert out["gain_dbi"][1][0] == pytest.approx(20 * math.log10(e), abs=1e-12)


def test_ffd_phi_just_below_360_is_its_own_column_d17():
    text = _ffd(
        [
            (0, 0, 1.0),
            (0, 180, 1.0),
            (0, 359.9999999999, 1.0),
            (10, 0, 0.7),
            (10, 180, 0.5),
            (10, 359.9999999999, 0.6),
        ]
    )
    out = parse_custom_antenna_ffd_text(text)
    assert out["phi_deg"] == [0.0, 180.0, 359.9999999999]
    assert out["validation"]["is_valid"] is True


@pytest.mark.parametrize("pole_theta", [0, 180])
def test_ffd_pole_row_spread_above_0p1_db_is_rejected_d8a(pole_theta):
    other = 10 if pole_theta == 0 else 170
    # 1.0 vs 0.9 field amplitude is 0.92 dB.
    text = _ffd(
        [
            (pole_theta, 0, 1.0),
            (pole_theta, 180, 0.9),
            (other, 0, 0.7),
            (other, 180, 0.5),
        ]
    )
    err = _ffd_error(text)
    assert err.code == CustomAntennaErrorCode.POLE_ROW_INCONSISTENT


def test_ffd_pole_row_is_power_averaged_before_peak_normalization_d8b():
    e_low = 10 ** (-0.05 / 20)  # 0.05 dB below 1.0
    text = _ffd([(0, 0, 1.0), (0, 180, e_low), (10, 0, 0.5), (10, 180, 0.5)])
    out = parse_custom_antenna_ffd_text(text)
    mean_power = (1.0 + e_low**2) / 2
    assert out["gain_dbi"][0] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert out["gain_dbi"][1][0] == pytest.approx(10 * math.log10(0.25 / mean_power))


# --- Schema (API payloads, and a backstop for FFD) -------------------------


def test_schema_rejects_repeated_psi_values_d15():
    payload = _enabled_payload([10, 10], [0, 180], [[0, -1], [0, -1]])
    assert payload["validation"]["is_valid"] is False
    assert "insufficient_psi_points" in payload["validation"]["errors"]


def test_schema_rejects_single_phi_column_after_merge_d15():
    payload = _enabled_payload([0, 10], [0, 360], [[0, 0], [-5, -5]])
    assert payload["phi_deg"] == [0.0]
    assert payload["validation"]["errors"] == ["insufficient_phi_points"]


def test_schema_reports_conflicting_merged_columns_d17():
    payload = _enabled_payload([0, 10], [0, 180, 360], [[0, 0, 0], [-5, -6, -4.3]])
    assert payload["phi_deg"] == [0.0, 180.0]
    # The first column (phi = 0) is kept.
    assert payload["gain_dbi"] == [[0.0, 0.0], [-5.0, -6.0]]
    assert "phi_grid_duplicate_conflict" in payload["validation"]["errors"]


def test_schema_merges_agreeing_columns_silently_d17():
    payload = _enabled_payload([0, 10], [0, 180, 360], [[0, 0, 0], [-5, -6, -5]])
    assert payload["phi_deg"] == [0.0, 180.0]
    assert payload["validation"]["is_valid"] is True


def test_schema_keeps_nearly_equal_phi_values_as_two_columns_d17():
    payload = _enabled_payload(
        [0, 10], [10, 10.0000000001, 200], [[0, 0, 0], [-5, -7, -6]]
    )
    assert payload["phi_deg"] == [10.0, 10.0000000001, 200.0]
    assert "phi_grid_duplicate_conflict" not in payload["validation"]["errors"]


def test_schema_tiny_negative_phi_wraps_like_netsim_d17():
    payload = _enabled_payload([0, 10], [-1e-10, 90], [[0, 0], [-5, -6]])
    assert payload["phi_deg"] == [90.0, 360.0 - 1e-10]
    payload = _enabled_payload([0, 10], [-1e-20, 90], [[0, 0], [-5, -6]])
    assert payload["phi_deg"] == [0.0, 90.0]


def test_schema_disabled_payload_stays_valid_with_any_grid():
    for psi, phi, gain in (
        ([10, 10], [0, 180], [[0, -1], [0, -1]]),
        ([0, 10], [0, 360], [[0, 0], [-5, -5]]),
        ([0, 10], [0, 180, 360], [[0, 0, 0], [-5, -6, -4]]),
    ):
        payload = _enabled_payload(psi, phi, gain, enabled=False)
        assert payload["validation"]["is_valid"] is True
        assert payload["validation"]["warnings"] == ["pattern_present_but_disabled"]


# --- Exact limits (review of #109, F-2) -------------------------------------
# These pin the 1e-9 dB duplicate tolerance, the 0.1 dB pole limit and the
# strict '>' of the pole check.  Loosening either limit, or changing '>' to
# '>=', makes one of them fail.


def test_csv_duplicate_1e8_db_apart_is_rejected():
    text = _csv(
        [(1, 0, 30), (1, 90, 30), (10, 0, 10), (10, 90, 12), (10, 0, 10.00000001)]
    )
    assert _csv_error(text).code == CsvErrorCode.DUPLICATE_GRID_POINT


def test_ffd_duplicate_1e8_db_apart_is_rejected():
    e = 0.5
    e_far = e * 10 ** (1e-8 / 20)  # 1e-8 dB in power
    text = _ffd([(0, 0, 1.0), (0, 90, 1.0), (10, 0, e), (10, 0, e_far), (10, 90, 0.25)])
    err = _ffd_error(text)
    assert err.code == CustomAntennaErrorCode.FFD_DUPLICATE_GRID_POINT
    assert err.line_no == 6


def test_schema_conflict_1e8_db_apart_is_reported():
    payload = _enabled_payload(
        [0, 10], [0, 180, 360], [[0, 0, 0], [-5, -6, -5.00000001]]
    )
    assert "phi_grid_duplicate_conflict" in payload["validation"]["errors"]


def test_csv_pole_row_spread_0p15_db_is_rejected():
    text = _csv([(0, 0, 30), (0, 180, 29.85), (10, 0, 20), (10, 180, 15)])
    assert _csv_error(text).code == CsvErrorCode.POLE_ROW_INCONSISTENT


def test_ffd_pole_row_spread_0p15_db_is_rejected():
    e_low = 10 ** (-0.15 / 20)
    text = _ffd([(0, 0, 1.0), (0, 180, e_low), (10, 0, 0.5), (10, 180, 0.5)])
    assert _ffd_error(text).code == CustomAntennaErrorCode.POLE_ROW_INCONSISTENT


def test_csv_pole_row_spread_of_exactly_0p1_db_is_accepted():
    # 0.1 - 0.0 is exactly 0.1 in double precision.  NetSim rejects only a
    # spread above 0.1 dB.
    assert 0.1 - 0.0 == 0.1
    text = _csv([(0, 0, 0.1), (0, 180, 0.0), (10, 0, -10), (10, 180, -15)])
    payload = parse_custom_antenna_csv_text(text)
    assert payload["psi_deg"] == [0.0, 10.0]


def test_ffd_pole_row_spread_of_exactly_0p1_db_is_accepted():
    # This pair is exactly 0.1 dB apart as the parser computes it,
    # 10*log10(E^2).  (The review's pair, 1.010165808235598 and
    # 0.9986025358479282, comes out at 0.09999999999999998 dB that way, so it
    # would not tell '>' from '>='.)
    e_hi, e_lo = 0.9848678477056328, 0.9735941586775221
    assert 10 * math.log10(e_hi * e_hi) - 10 * math.log10(e_lo * e_lo) == 0.1
    text = _ffd([(0, 0, e_hi), (0, 180, e_lo), (10, 0, 0.5), (10, 180, 0.5)])
    out = parse_custom_antenna_ffd_text(text)
    assert out["validation"]["is_valid"] is True


# --- Extreme FFD values (review of #109, F-1 and F-6) -----------------------


def test_ffd_duplicate_far_below_first_copy_is_rejected_not_crash():
    # F-1: E = 1e10 then 1e-160 is about 3400 dB apart.  The ratio of the two
    # powers underflows to 0; the parser must still reject the file cleanly.
    text = _ffd(
        [(0, 0, 1.0), (0, 90, 1.0), (10, 0, 1.0), (10, 90, 1e10), (10, 90, 1e-160)]
    )
    err = _ffd_error(text)
    assert err.code == CustomAntennaErrorCode.FFD_DUPLICATE_GRID_POINT
    assert err.line_no == 7


def test_ffd_pole_row_near_max_float_is_accepted_not_crash():
    # F-6: E = 1e154 gives a power near 1e308; summing the pole row overflowed
    # to inf.  main accepts this file, so a saved project may already use it.
    text = _ffd([(0, 0, 1e154), (0, 90, 1e154), (10, 0, 1.0), (10, 90, 1.0)])
    out = parse_custom_antenna_ffd_text(text)
    assert out["validation"]["is_valid"] is True
    assert out["gain_dbi"][0] == [0.0, 0.0]
    assert out["gain_dbi"][1] == pytest.approx([-3080.0, -3080.0])
