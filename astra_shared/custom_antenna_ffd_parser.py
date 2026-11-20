"""Step 8.2: FFD parser with minimal field extraction + canonical mapping.

Scope (intentional constraints):
- single-frequency selection only,
- scalar gain extraction from E_theta / E_phi magnitudes,
- canonical payload normalization via normalize_custom_antenna(...).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from .custom_antenna_errors import CustomAntennaErrorCode
from .custom_antenna_schema import (
    DUPLICATE_GAIN_TOL_DB,
    POLE_ROW_MAX_SPREAD_DB,
    POLE_ROW_PSI_DEG,
    normalize_custom_antenna,
    pole_row_spread_db,
)


@dataclass
class CustomAntennaFfdError(Exception):
    """Deterministic FFD parsing error for UI/API handling."""

    code: str
    detail: str
    line_no: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "detail": self.detail}
        if self.line_no is not None:
            payload["line_no"] = self.line_no
        return payload


_FREQ_RE = re.compile(r"frequency(?:_hz)?\s*[:=]\s*([0-9.eE+-]+)")
_MULTISPACE_RE = re.compile(r"\s+")


def _parse_frequency_header(text: str) -> float | None:
    m = _FREQ_RE.search(text.lower())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if math.isfinite(val) and val > 0 else None


def parse_custom_antenna_ffd_text(
    ffd_text: str,
    *,
    filename: str | None = None,
    frequency_hz: float | None = None,
) -> dict[str, Any]:
    if not isinstance(ffd_text, str):
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_INVALID_INPUT_TYPE,
            "FFD parser expects text input (str).",
        )

    text = ffd_text.strip()
    if not text:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_EMPTY_FILE, "FFD file is empty."
        )

    lines = [ln.strip() for ln in ffd_text.splitlines()]
    nonempty = [
        (i + 1, ln) for i, ln in enumerate(lines) if ln and not ln.startswith("#")
    ]

    parsed_freq = _parse_frequency_header(ffd_text)
    if frequency_hz is not None:
        try:
            selected = float(frequency_hz)
        except ValueError as exc:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_INVALID_FREQUENCY_SELECTION,
                "frequency_hz must be numeric.",
            ) from exc
        if not math.isfinite(selected) or selected <= 0:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_INVALID_FREQUENCY_SELECTION,
                "frequency_hz must be positive finite.",
            )
        if parsed_freq is not None and not math.isclose(
            parsed_freq, selected, rel_tol=1e-6, abs_tol=1.0
        ):
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_FREQUENCY_MISMATCH,
                f"Requested frequency_hz={selected} does not match FFD frequency {parsed_freq}.",
            )
        out_freq = selected
    else:
        if parsed_freq is None:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_MISSING_FREQUENCY_SELECTION,
                "No frequency header found; provide frequency_hz explicitly for Step 8.2 single-frequency mode.",
            )
        out_freq = parsed_freq

    # detect table header
    header_idx = None
    for line_no, ln in nonempty:
        low = _MULTISPACE_RE.sub(" ", ln.lower())
        if "theta" in low and "phi" in low and "e_theta" in low and "e_phi" in low:
            header_idx = (line_no, ln)
            break
    if header_idx is None:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_FORMAT_NOT_RECOGNIZED,
            "Missing required header columns: theta phi e_theta e_phi",
            1,
        )

    start_line_no = header_idx[0] + 1
    # (theta, phi, power, line_no) in file order.  The sample count, the
    # duplicate check and the completeness check run after the loop, in
    # NetSim's order: count, duplicates, completeness, pole rows.
    samples: list[tuple[float, float, float, int]] = []
    for idx in range(start_line_no - 1, len(lines)):
        raw = lines[idx].strip()
        line_no = idx + 1
        if not raw or raw.startswith("#"):
            continue
        toks = raw.replace(",", " ").split()
        if len(toks) < 4:
            continue
        if len(toks) > 4:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_UNEXPECTED_COLUMN_COUNT,
                "FFD data row must contain exactly 4 columns: theta phi e_theta e_phi.",
                line_no,
            )
        try:
            theta = float(toks[0])
            phi = float(toks[1])
            e_theta = float(toks[2])
            e_phi = float(toks[3])
        except ValueError:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_NON_NUMERIC_FIELD,
                "Could not parse theta/phi/e-field numeric values.",
                line_no,
            )
        if not all(math.isfinite(v) for v in (theta, phi, e_theta, e_phi)):
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_NON_FINITE_FIELD,
                "Encountered non-finite numeric value.",
                line_no,
            )
        if theta < 0 or theta > 180:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_THETA_OUT_OF_RANGE,
                "theta must be in [0, 180] degrees.",
                line_no,
            )
        if phi < 0 or phi > 360:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_PHI_OUT_OF_RANGE,
                "phi must be in [0, 360] degrees.",
                line_no,
            )
        # phi = 360 is the same azimuth as phi = 0.  Wrap only exactly 360,
        # as NetSim does; a phi just below 360 is its own column.
        if phi == 360.0:
            phi = 0.0
        p = (e_theta * e_theta) + (e_phi * e_phi)
        if p <= 0:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_NON_POSITIVE_POWER,
                "Computed field power must be positive.",
                line_no,
            )
        samples.append((theta, phi, p, line_no))

    if not samples:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_NO_DATA_ROWS,
            "No numeric data rows found after header.",
        )

    psi = sorted({s[0] for s in samples})
    ph = sorted({s[1] for s in samples})

    # At least two distinct theta and two distinct phi values, counted before
    # the duplicate check (NetSim's order).  NetSim uses these csv-named codes
    # for FFD files too.
    if len(psi) < 2:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.CSV_INSUFFICIENT_PSI_POINTS,
            "FFD table needs at least two distinct theta values.",
        )
    if len(ph) < 2:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.CSV_INSUFFICIENT_PHI_POINTS,
            "FFD table needs at least two distinct phi values.",
        )

    # A grid point given twice (for example phi = 0 and phi = 360) is
    # accepted when both powers agree within DUPLICATE_GAIN_TOL_DB, compared
    # in dB so that the tolerance holds at any power level.  The first copy is
    # kept.  Otherwise the file is rejected at the conflicting sample.
    lookup: dict[tuple[float, float], float] = {}
    for theta, phi, p, line_no in samples:
        key = (theta, phi)
        if key in lookup:
            if abs(10.0 * math.log10(p / lookup[key])) > DUPLICATE_GAIN_TOL_DB:
                raise CustomAntennaFfdError(
                    CustomAntennaErrorCode.FFD_DUPLICATE_GRID_POINT,
                    "Duplicate theta/phi sample encountered in FFD table.",
                    line_no,
                )
            continue
        lookup[key] = p

    missing = [(t, p) for t in psi for p in ph if (t, p) not in lookup]
    if missing:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_INCOMPLETE_GRID,
            "FFD table does not form a complete rectangular theta/phi grid.",
        )

    # A theta = 0 or theta = 180 row is a single direction.  Reject a spread
    # above POLE_ROW_MAX_SPREAD_DB; power-average a smaller one (the linear
    # mean of E^2) before peak normalization.
    for t in psi:
        if t not in POLE_ROW_PSI_DEG:
            continue
        row_power = [lookup[(t, p)] for p in ph]
        row_db = [10.0 * math.log10(pw) for pw in row_power]
        if pole_row_spread_db(row_db) > POLE_ROW_MAX_SPREAD_DB:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.POLE_ROW_INCONSISTENT,
                f"Pole row theta={t:g} spreads by more than "
                f"{POLE_ROW_MAX_SPREAD_DB:g} dB across phi.",
            )
        mean_power = sum(row_power) / len(row_power)
        for p in ph:
            lookup[(t, p)] = mean_power

    peak = max(lookup.values())
    gain = []
    for t in psi:
        row = []
        for p in ph:
            rel = lookup[(t, p)] / peak
            row.append(10.0 * math.log10(rel))
        gain.append(row)

    raw_payload = {
        "enabled": True,
        "source_format": "ffd",
        "filename": filename,
        "frequency_hz": out_freq,
        "psi_deg": psi,
        "phi_deg": ph,
        "gain_dbi": gain,
    }
    return normalize_custom_antenna(raw_payload)
