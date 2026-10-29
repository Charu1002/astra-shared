"""Immutable 3GPP TS 38.214 MCS table 2 rows used by Astra."""

from __future__ import annotations

from dataclasses import dataclass

MCS_TABLE_VERSION = "TS 38.214 V18.11.0 Table 5.1.3.1-2"


@dataclass(frozen=True)
class McsRow:
    index: int
    modulation: str
    qm: int
    r_x1024: float
    printed_se: str

    @property
    def code_rate(self) -> float:
        return self.r_x1024 / 1024.0

    @property
    def spectral_efficiency(self) -> float:
        return self.qm * self.r_x1024 / 1024.0


MCS_TABLE = (
    McsRow(0, "QPSK", 2, 120, "0.2344"),
    McsRow(1, "QPSK", 2, 193, "0.3770"),
    McsRow(2, "QPSK", 2, 308, "0.6016"),
    McsRow(3, "QPSK", 2, 449, "0.8770"),
    McsRow(4, "QPSK", 2, 602, "1.1758"),
    McsRow(5, "16QAM", 4, 378, "1.4766"),
    McsRow(6, "16QAM", 4, 434, "1.6953"),
    McsRow(7, "16QAM", 4, 490, "1.9141"),
    McsRow(8, "16QAM", 4, 553, "2.1602"),
    McsRow(9, "16QAM", 4, 616, "2.4063"),
    McsRow(10, "16QAM", 4, 658, "2.5703"),
    McsRow(11, "64QAM", 6, 466, "2.7305"),
    McsRow(12, "64QAM", 6, 517, "3.0293"),
    McsRow(13, "64QAM", 6, 567, "3.3223"),
    McsRow(14, "64QAM", 6, 616, "3.6094"),
    McsRow(15, "64QAM", 6, 666, "3.9023"),
    McsRow(16, "64QAM", 6, 719, "4.2129"),
    McsRow(17, "64QAM", 6, 772, "4.5234"),
    McsRow(18, "64QAM", 6, 822, "4.8164"),
    McsRow(19, "64QAM", 6, 873, "5.1152"),
    McsRow(20, "256QAM", 8, 682.5, "5.3320"),
    McsRow(21, "256QAM", 8, 711, "5.5547"),
    McsRow(22, "256QAM", 8, 754, "5.8906"),
    McsRow(23, "256QAM", 8, 797, "6.2266"),
    McsRow(24, "256QAM", 8, 841, "6.5703"),
    McsRow(25, "256QAM", 8, 885, "6.9141"),
    McsRow(26, "256QAM", 8, 916.5, "7.1602"),
    McsRow(27, "256QAM", 8, 948, "7.4063"),
)
