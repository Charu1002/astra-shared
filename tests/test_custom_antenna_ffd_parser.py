#!/usr/bin/env python3

import pytest

from astra_shared.custom_antenna_errors import CustomAntennaErrorCode
from astra_shared.custom_antenna_ffd_parser import (
    CustomAntennaFfdError,
    parse_custom_antenna_ffd_text,
)


def test_ffd_parser_minimal_single_frequency_success():
    text = """
frequency_hz=12000000000
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 1.0 0.0
10 0 0.7 0.0
10 180 0.5 0.0
"""
    out = parse_custom_antenna_ffd_text(text, filename="a.ffd")
    assert out["enabled"] is True
    assert out["source_format"] == "ffd"
    assert out["frequency_hz"] == 12_000_000_000
    assert out["validation"]["is_valid"] is True
    assert len(out["psi_deg"]) == 2
    assert len(out["phi_deg"]) == 2


def test_ffd_parser_requires_frequency_if_header_missing():
    text = """
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 0.9 0.0
"""
    with pytest.raises(CustomAntennaFfdError) as exc:
        parse_custom_antenna_ffd_text(text)
    assert exc.value.code == CustomAntennaErrorCode.FFD_MISSING_FREQUENCY_SELECTION


def test_ffd_parser_detects_frequency_mismatch():
    text = """
frequency_hz=12000000000
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 0.9 0.0
"""
    with pytest.raises(CustomAntennaFfdError) as exc:
        parse_custom_antenna_ffd_text(text, frequency_hz=11_000_000_000)
    assert exc.value.code == CustomAntennaErrorCode.FFD_FREQUENCY_MISMATCH


def test_ffd_parser_accepts_frequency_selection_when_header_is_absent():
    text = """
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 1.0 0.0
10 0 0.7 0.0
10 180 0.5 0.0
"""
    out = parse_custom_antenna_ffd_text(text, frequency_hz=12_000_000_000)
    assert out["frequency_hz"] == 12_000_000_000.0
    assert out["validation"]["is_valid"] is True


def test_ffd_parser_accepts_small_relative_frequency_difference():
    text = """
frequency_hz=11999999000
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 1.0 0.0
10 0 0.7 0.0
10 180 0.5 0.0
"""
    out = parse_custom_antenna_ffd_text(text, frequency_hz=12_000_000_000)
    assert out["frequency_hz"] == 12_000_000_000.0
    assert out["validation"]["is_valid"] is True


def test_ffd_parser_treats_phi_360_as_periodic_boundary_duplicate():
    text = """
frequency_hz=12000000000
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 1.0 0.0
0 360 1.0 0.0
10 0 0.7 0.0
10 180 0.5 0.0
10 360 0.7 0.0
"""
    out = parse_custom_antenna_ffd_text(text)
    assert out["phi_deg"] == [0.0, 180.0]
    assert out["validation"]["is_valid"] is True


def test_ffd_parser_rejects_conflicting_zero_and_360_degree_samples():
    text = """
frequency_hz=12000000000
theta phi e_theta e_phi
0 0 1.0 0.0
0 180 1.0 0.0
0 360 2.0 0.0
10 0 0.7 0.0
10 180 0.5 0.0
10 360 0.7 0.0
"""
    with pytest.raises(CustomAntennaFfdError) as exc:
        parse_custom_antenna_ffd_text(text)
    assert exc.value.code == CustomAntennaErrorCode.FFD_DUPLICATE_GRID_POINT
