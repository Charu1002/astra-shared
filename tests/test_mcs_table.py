from dataclasses import FrozenInstanceError

import pytest

from astra_shared.mcs_table import MCS_TABLE, MCS_TABLE_VERSION


EXPECTED = (
    (0, "QPSK", 2, 120, "0.2344"),
    (1, "QPSK", 2, 193, "0.3770"),
    (2, "QPSK", 2, 308, "0.6016"),
    (3, "QPSK", 2, 449, "0.8770"),
    (4, "QPSK", 2, 602, "1.1758"),
    (5, "16QAM", 4, 378, "1.4766"),
    (6, "16QAM", 4, 434, "1.6953"),
    (7, "16QAM", 4, 490, "1.9141"),
    (8, "16QAM", 4, 553, "2.1602"),
    (9, "16QAM", 4, 616, "2.4063"),
    (10, "16QAM", 4, 658, "2.5703"),
    (11, "64QAM", 6, 466, "2.7305"),
    (12, "64QAM", 6, 517, "3.0293"),
    (13, "64QAM", 6, 567, "3.3223"),
    (14, "64QAM", 6, 616, "3.6094"),
    (15, "64QAM", 6, 666, "3.9023"),
    (16, "64QAM", 6, 719, "4.2129"),
    (17, "64QAM", 6, 772, "4.5234"),
    (18, "64QAM", 6, 822, "4.8164"),
    (19, "64QAM", 6, 873, "5.1152"),
    (20, "256QAM", 8, 682.5, "5.3320"),
    (21, "256QAM", 8, 711, "5.5547"),
    (22, "256QAM", 8, 754, "5.8906"),
    (23, "256QAM", 8, 797, "6.2266"),
    (24, "256QAM", 8, 841, "6.5703"),
    (25, "256QAM", 8, 885, "6.9141"),
    (26, "256QAM", 8, 916.5, "7.1602"),
    (27, "256QAM", 8, 948, "7.4063"),
)


def test_mcs_table_has_exact_version_and_rows_0_through_27():
    assert MCS_TABLE_VERSION == "TS 38.214 V18.11.0 Table 5.1.3.1-2"
    assert tuple(
        (row.index, row.modulation, row.qm, row.r_x1024, row.printed_se)
        for row in MCS_TABLE
    ) == EXPECTED


def test_mcs_table_uses_exact_efficiency_for_calculation():
    for row in MCS_TABLE:
        assert row.spectral_efficiency == row.qm * row.r_x1024 / 1024
        assert row.code_rate == row.r_x1024 / 1024


def test_mcs_table_and_rows_are_immutable():
    assert isinstance(MCS_TABLE, tuple)
    with pytest.raises(FrozenInstanceError):
        MCS_TABLE[0].modulation = "BPSK"
