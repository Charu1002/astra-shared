"""CSV parser for custom antenna pattern import (Step 3.3).

This module parses CSV text with required semantic columns:
- psi_deg (aliases supported)
- phi_deg (aliases supported)
- gain_dbi (aliases supported)

Step 3.3 adds structured deterministic error codes while preserving strict
validation behavior from Step 3.1/3.2.

CSV input gains are absolute dBi. Returned pattern gains are peak-relative dB,
which is the convention used by the RF link budget and FFD importer.
"""

from __future__ import annotations

import csv
import math
from io import StringIO
from typing import Any

from .custom_antenna_errors import CustomAntennaErrorCode
from .custom_antenna_schema import (
    DUPLICATE_GAIN_TOL_DB,
    POLE_ROW_MAX_SPREAD_DB,
    POLE_ROW_PSI_DEG,
    normalize_custom_antenna,
    pole_row_representative_db,
    pole_row_spread_db,
    wrap_phi_deg,
)


class CsvErrorCode:
    EMPTY_CSV = CustomAntennaErrorCode.CSV_EMPTY
    HEADER_NOT_FOUND = CustomAntennaErrorCode.CSV_HEADER_NOT_FOUND
    MISSING_REQUIRED_COLUMNS = CustomAntennaErrorCode.CSV_MISSING_REQUIRED_COLUMNS
    NON_NUMERIC_VALUE = CustomAntennaErrorCode.CSV_NON_NUMERIC_VALUE
    UNEXPECTED_COLUMN_COUNT = CustomAntennaErrorCode.CSV_UNEXPECTED_COLUMN_COUNT
    NON_FINITE_VALUE = CustomAntennaErrorCode.CSV_NON_FINITE_VALUE
    PSI_OUT_OF_RANGE = CustomAntennaErrorCode.CSV_PSI_OUT_OF_RANGE
    PHI_OUT_OF_RANGE = CustomAntennaErrorCode.CSV_PHI_OUT_OF_RANGE
    NO_DATA_ROWS = CustomAntennaErrorCode.CSV_NO_DATA_ROWS
    INSUFFICIENT_PSI_POINTS = CustomAntennaErrorCode.CSV_INSUFFICIENT_PSI_POINTS
    INSUFFICIENT_PHI_POINTS = CustomAntennaErrorCode.CSV_INSUFFICIENT_PHI_POINTS
    DUPLICATE_GRID_POINT = CustomAntennaErrorCode.CSV_DUPLICATE_GRID_POINT
    INCOMPLETE_TABLE = CustomAntennaErrorCode.CSV_INCOMPLETE_TABLE
    MIXED_PHI_CONVENTION = CustomAntennaErrorCode.CSV_MIXED_PHI_CONVENTION
    POLE_ROW_INCONSISTENT = CustomAntennaErrorCode.POLE_ROW_INCONSISTENT


class CustomAntennaCsvError(ValueError):
    """Deterministic parse/validation error for custom antenna CSV input."""

    def __init__(
        self, code: str, *, line_no: int | None = None, detail: str | None = None
    ):
        self.code = code
        self.line_no = line_no
        self.detail = detail

        parts: list[str] = []
        if line_no is not None:
            parts.append(f"line_{line_no}")
        parts.append(code)
        if detail:
            parts.append(detail)

        super().__init__(":".join(parts))


_HEADER_ALIASES: dict[str, set[str]] = {
    "psi_deg": {"psi_deg", "psi", "theta", "off_axis_deg"},
    "phi_deg": {"phi_deg", "phi", "azimuth_deg", "az_deg"},
    "gain_dbi": {"gain_dbi", "gain_db", "gain"},
}


def _norm_header_token(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "_").replace("-", "_")


def _is_comment_or_blank(line: str) -> bool:
    t = line.strip()
    return (not t) or t.startswith("#") or t.startswith("//") or t.startswith(";")


def _canonical_field_map(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise CustomAntennaCsvError(CsvErrorCode.HEADER_NOT_FOUND)

    by_norm = {_norm_header_token(h): h for h in fieldnames}
    out: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in _HEADER_ALIASES.items():
        found = None
        for alias in aliases:
            key = _norm_header_token(alias)
            if key in by_norm:
                found = by_norm[key]
                break
        if found is None:
            missing.append(canonical)
        else:
            out[canonical] = found

    if missing:
        raise CustomAntennaCsvError(
            CsvErrorCode.MISSING_REQUIRED_COLUMNS, detail=",".join(sorted(missing))
        )

    return out


def _extract_csv_table_text(csv_text: str) -> str:
    lines = csv_text.splitlines()
    saw_candidate_header = False
    for i, line in enumerate(lines):
        if _is_comment_or_blank(line):
            continue
        header = next(csv.reader([line]), None)
        if not header:
            continue
        saw_candidate_header = True
        try:
            fmap = _canonical_field_map(header)
        except CustomAntennaCsvError as exc:
            if exc.code == CsvErrorCode.MISSING_REQUIRED_COLUMNS:
                # Non-comment metadata rows can resemble a partial data
                # header.  Keep scanning for the first complete header.
                continue
            continue
        if len(fmap) == 3:
            return "\n".join(lines[i:])
    if saw_candidate_header:
        raise CustomAntennaCsvError(
            CsvErrorCode.MISSING_REQUIRED_COLUMNS,
            detail="psi_deg,phi_deg,gain_dbi",
        )
    raise CustomAntennaCsvError(CsvErrorCode.HEADER_NOT_FOUND)


def parse_custom_antenna_csv_text(
    csv_text: str,
    *,
    filename: str | None = None,
    frequency_hz: float | None = None,
    selected_frequency_hz: float | None = None,
) -> dict[str, Any]:
    """Parse custom antenna CSV text into canonical custom_antenna payload."""
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise CustomAntennaCsvError(CsvErrorCode.EMPTY_CSV)

    table_text = _extract_csv_table_text(csv_text)
    reader = csv.DictReader(StringIO(table_text))
    field_map = _canonical_field_map(reader.fieldnames)

    triples: list[tuple[float, float, float]] = []
    saw_negative_phi = False
    saw_phi_above_180 = False
    for line_no, row in enumerate(reader, start=2):
        if None in row:
            raise CustomAntennaCsvError(
                CsvErrorCode.UNEXPECTED_COLUMN_COUNT, line_no=line_no
            )
        if all((v is None or str(v).strip() == "") for v in row.values()):
            continue
        try:
            psi = float(row[field_map["psi_deg"]])
            phi = float(row[field_map["phi_deg"]])
            gain = float(row[field_map["gain_dbi"]])
        except Exception as exc:  # noqa: BLE001
            raise CustomAntennaCsvError(
                CsvErrorCode.NON_NUMERIC_VALUE, line_no=line_no
            ) from exc

        if not (math.isfinite(psi) and math.isfinite(phi) and math.isfinite(gain)):
            raise CustomAntennaCsvError(CsvErrorCode.NON_FINITE_VALUE, line_no=line_no)

        if psi < 0.0 or psi > 180.0:
            raise CustomAntennaCsvError(CsvErrorCode.PSI_OUT_OF_RANGE, line_no=line_no)

        if phi < -180.0 or phi > 360.0:
            raise CustomAntennaCsvError(CsvErrorCode.PHI_OUT_OF_RANGE, line_no=line_no)

        # A file gives phi either as -180..180 or as 0..360.  NetSim rejects
        # a mix at the first row where both a negative phi and a phi above
        # 180 have been seen.  This must run on the raw phi, before wrapping.
        if phi < 0.0:
            saw_negative_phi = True
        if phi > 180.0:
            saw_phi_above_180 = True
        if saw_negative_phi and saw_phi_above_180:
            raise CustomAntennaCsvError(
                CsvErrorCode.MIXED_PHI_CONVENTION, line_no=line_no
            )

        # Wrap phi to [0, 360) here, as NetSim does, so that the sample count,
        # the duplicate check and the completeness check below all see 0 and
        # 360 (or -90 and 270) as the same azimuth.
        triples.append((psi, wrap_phi_deg(phi), gain))

    if not triples:
        raise CustomAntennaCsvError(CsvErrorCode.NO_DATA_ROWS)

    psi_grid = sorted({psi for psi, _, _ in triples})
    phi_grid = sorted({phi for _, phi, _ in triples})

    if len(psi_grid) < 2:
        raise CustomAntennaCsvError(CsvErrorCode.INSUFFICIENT_PSI_POINTS)
    if len(phi_grid) < 2:
        raise CustomAntennaCsvError(CsvErrorCode.INSUFFICIENT_PHI_POINTS)

    # A grid point given twice is accepted when both gains agree within
    # DUPLICATE_GAIN_TOL_DB (the first copy is kept) and rejected otherwise.
    cell_map: dict[tuple[float, float], float] = {}
    for psi, phi, gain in triples:
        key = (psi, phi)
        if key in cell_map:
            if abs(gain - cell_map[key]) > DUPLICATE_GAIN_TOL_DB:
                raise CustomAntennaCsvError(CsvErrorCode.DUPLICATE_GRID_POINT)
            continue
        cell_map[key] = gain

    expected_points = len(psi_grid) * len(phi_grid)
    if len(cell_map) != expected_points:
        raise CustomAntennaCsvError(
            CsvErrorCode.INCOMPLETE_TABLE,
            detail=f"expected_{expected_points}_got_{len(cell_map)}",
        )

    gain_table: list[list[float]] = []
    for psi in psi_grid:
        row: list[float] = []
        for phi in phi_grid:
            row.append(cell_map[(psi, phi)])
        gain_table.append(row)

    # A psi = 0 or psi = 180 row is a single direction.  Reject a spread above
    # POLE_ROW_MAX_SPREAD_DB; power-average a smaller one.  This runs before
    # peak normalization, so a pole-row peak is the averaged value, as in
    # NetSim.
    for row_idx, psi in enumerate(psi_grid):
        if psi not in POLE_ROW_PSI_DEG:
            continue
        row = gain_table[row_idx]
        if pole_row_spread_db(row) > POLE_ROW_MAX_SPREAD_DB:
            raise CustomAntennaCsvError(
                CsvErrorCode.POLE_ROW_INCONSISTENT, detail=f"psi_{psi:g}"
            )
        representative = pole_row_representative_db(row)
        gain_table[row_idx] = [representative] * len(row)

    peak_gain_dbi = max(max(row) for row in gain_table)
    gain_table = [[gain_dbi - peak_gain_dbi for gain_dbi in row] for row in gain_table]

    selected_frequency = (
        frequency_hz if frequency_hz is not None else selected_frequency_hz
    )
    raw_payload = {
        "schema_version": 1,
        "enabled": True,
        "source_format": "csv",
        "filename": filename or None,
        "frequency_hz": selected_frequency,
        "psi_deg": psi_grid,
        "phi_deg": phi_grid,
        "gain_dbi": gain_table,
    }

    return normalize_custom_antenna(raw_payload)
