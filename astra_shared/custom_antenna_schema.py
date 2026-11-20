"""Canonical schema helpers for external custom antenna patterns (Step 2a).

This module defines the minimal v1 custom_antenna contract and deterministic
normalization/validation utilities. File parsing (CSV/FFD), interpolation, and
runtime integration are intentionally deferred to later steps.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

SUPPORTED_SOURCE_FORMATS = {"csv", "ffd"}
SCHEMA_VERSION = 1
SPARSE_GRID_STEP_DEG = 30.0
MAX_PSI_POINTS = 2000
MAX_PHI_POINTS = 4000
MAX_GAIN_CELLS = 1_000_000

# File acceptance rules shared with NetSim (comparison report rows D8, D17).
# Two samples of the same grid point agree when their gains differ by at most
# DUPLICATE_GAIN_TOL_DB; the first copy is kept.  A pole row (psi = 0 or 180,
# one direction) may spread by at most POLE_ROW_MAX_SPREAD_DB.
DUPLICATE_GAIN_TOL_DB = 1e-9
POLE_ROW_MAX_SPREAD_DB = 0.1
POLE_ROW_PSI_DEG = (0.0, 180.0)


def wrap_phi_deg(value: float) -> float:
    """Reduce phi to [0, 360) exactly as NetSim does.

    Equivalent to ``while phi < 0: phi += 360`` then
    ``while phi >= 360: phi -= 360``.  ``%`` alone is not enough: for a tiny
    negative phi it returns 360.0, so the second step is needed.
    """
    phi = float(value) % 360.0
    if phi >= 360.0:
        phi -= 360.0
    return phi


def pole_row_spread_db(gains_db: list[float]) -> float:
    """Spread (max - min) of the gains of one pole row, in dB."""
    return max(gains_db) - min(gains_db)


def pole_row_representative_db(gains_db: list[float]) -> float:
    """Power average of a pole row, in dB.

    representative = max + 10*log10(mean(10^((g - max)/10))), NetSim's rule.
    """
    peak = max(gains_db)
    mean_rel = sum(10.0 ** ((g - peak) / 10.0) for g in gains_db) / len(gains_db)
    return peak + 10.0 * math.log10(mean_rel)


def default_custom_antenna() -> dict[str, Any]:
    """Return the canonical default custom_antenna payload for schema v1."""
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": False,
        "source_format": None,
        "filename": None,
        "frequency_hz": None,
        "psi_deg": [],
        "phi_deg": [],
        "gain_dbi": [],
        "validation": {
            "is_valid": False,
            "errors": ["no_pattern_loaded"],
            "warnings": [],
        },
    }


def normalize_custom_antenna(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize arbitrary input into canonical custom_antenna schema v1.

    Deterministic behavior:
    - Missing/invalid fields fall back to schema defaults.
    - Numeric arrays are converted to finite floats where possible.
    - Validation metadata is regenerated from normalized content.
    """
    base = default_custom_antenna()
    if not isinstance(raw, dict):
        return base

    normalized = deepcopy(base)

    # Scalar fields
    normalized["schema_version"] = _parse_schema_version(raw.get("schema_version"))
    normalized["enabled"] = _parse_bool(raw.get("enabled"), default=False)
    normalized["source_format"] = _parse_source_format(raw.get("source_format"))
    normalized["filename"] = _parse_optional_string(raw.get("filename"))
    normalized["frequency_hz"] = _parse_optional_finite_float(raw.get("frequency_hz"))

    # Grids/table
    psi_deg = _parse_numeric_list(raw.get("psi_deg"))
    phi_deg = _parse_numeric_list(raw.get("phi_deg"))
    gain_dbi = _parse_numeric_matrix(raw.get("gain_dbi"))
    phi_deg, gain_dbi, phi_conflict = _normalize_phi_grid_and_gain(phi_deg, gain_dbi)

    normalized["psi_deg"] = psi_deg
    normalized["phi_deg"] = phi_deg
    normalized["gain_dbi"] = gain_dbi

    errors, warnings = _validate_shape_and_content(
        enabled=normalized["enabled"],
        source_format=normalized["source_format"],
        frequency_hz=normalized["frequency_hz"],
        psi_deg=psi_deg,
        phi_deg=phi_deg,
        gain_dbi=gain_dbi,
        phi_conflict=phi_conflict,
    )

    # Step 9.5 invariants: deterministic/de-duplicated validation vectors.
    errors = sorted(set(errors))
    warnings = sorted(set(warnings))
    normalized["validation"] = {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    return normalized


def _parse_schema_version(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return SCHEMA_VERSION
    return parsed if parsed > 0 else SCHEMA_VERSION


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if token in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
    return default


def _parse_source_format(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if token in SUPPORTED_SOURCE_FORMATS else None


def _parse_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _parse_optional_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        parsed = _parse_optional_finite_float(item)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def _parse_numeric_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            return []
        parsed_row = _parse_numeric_list(row)
        if len(parsed_row) != len(row):
            return []
        matrix.append(parsed_row)
    return matrix


def _normalize_phi_grid_and_gain(
    phi_deg: list[float], gain_dbi: list[list[float]]
) -> tuple[list[float], list[list[float]], bool]:
    """Wrap phi to [0, 360), sort it, and merge columns that name one azimuth.

    Columns collide only when their wrapped phi values are exactly equal, as
    in NetSim.  The first column (in input order) is kept.  The returned flag
    is True when a dropped column differs from the kept one by more than
    DUPLICATE_GAIN_TOL_DB in some row; the caller reports that as an error
    instead of losing the data silently.  Rows shorter than ``phi_deg`` are
    left for the shape check and skipped here.
    """
    if not phi_deg:
        return [], gain_dbi, False

    # sorted() is stable, so equal wrapped values keep their input order and
    # the first one seen is the first column of the input.
    indexed = sorted(
        ((idx, wrap_phi_deg(phi)) for idx, phi in enumerate(phi_deg)),
        key=lambda item: item[1],
    )
    kept_indices: list[int] = []
    normalized_phi: list[float] = []
    merged: list[tuple[int, int]] = []  # (kept index, dropped index)
    kept_by_phi: dict[float, int] = {}
    for original_idx, phi in indexed:
        if phi in kept_by_phi:
            merged.append((kept_by_phi[phi], original_idx))
            continue
        kept_by_phi[phi] = original_idx
        kept_indices.append(original_idx)
        normalized_phi.append(phi)

    if not gain_dbi:
        return normalized_phi, gain_dbi, False

    conflict = False
    reordered_gain: list[list[float]] = []
    for row in gain_dbi:
        if len(row) < len(phi_deg):
            reordered_gain.append(row)
            continue
        for kept_idx, dropped_idx in merged:
            if abs(row[dropped_idx] - row[kept_idx]) > DUPLICATE_GAIN_TOL_DB:
                conflict = True
        reordered_gain.append([row[idx] for idx in kept_indices])
    return normalized_phi, reordered_gain, conflict


def _validate_shape_and_content(
    *,
    enabled: bool,
    source_format: str | None,
    frequency_hz: float | None,
    psi_deg: list[float],
    phi_deg: list[float],
    gain_dbi: list[list[float]],
    phi_conflict: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not enabled:
        if psi_deg or phi_deg or gain_dbi:
            warnings.append("pattern_present_but_disabled")
        else:
            errors.append("no_pattern_loaded")
        return errors, warnings

    # The checks below apply to enabled patterns only, so a disabled payload
    # that carries a stale table stays valid.

    # Two phi columns named the same azimuth (e.g. 0 and 360) with different
    # gains.  NetSim rejects the file; keeping either column loses data.
    if phi_conflict:
        errors.append("phi_grid_duplicate_conflict")

    # NetSim needs at least two distinct samples per axis.  With one, the
    # lookup cannot interpolate along that axis.  phi_deg is already merged
    # to distinct values; psi_deg may still repeat (the monotonic check below
    # uses '>'), so count distinct values.
    if psi_deg and phi_deg:
        if len(set(psi_deg)) < 2:
            errors.append("insufficient_psi_points")
        if len(set(phi_deg)) < 2:
            errors.append("insufficient_phi_points")

    if source_format is None:
        errors.append("missing_source_format")
    if frequency_hz is None:
        errors.append("missing_frequency_hz")

    if not psi_deg:
        errors.append("missing_psi_grid")
    if not phi_deg:
        errors.append("missing_phi_grid")
    if not gain_dbi:
        errors.append("missing_gain_table")

    # Both interpolation implementations use ordered-grid searches.  A payload
    # supplied outside the CSV/FFD importers must therefore not be accepted
    # when its psi samples are out of order.
    if psi_deg and any(psi_deg[i] > psi_deg[i + 1] for i in range(len(psi_deg) - 1)):
        errors.append("psi_grid_not_monotonic")
    if phi_deg and any(phi_deg[i] > phi_deg[i + 1] for i in range(len(phi_deg) - 1)):
        errors.append("phi_grid_not_monotonic")

    if gain_dbi:
        row_lengths = {len(row) for row in gain_dbi}
        if len(row_lengths) > 1:
            errors.append("gain_table_non_rectangular")
        else:
            cols = row_lengths.pop() if row_lengths else 0
            if psi_deg and len(gain_dbi) != len(psi_deg):
                errors.append("gain_rows_mismatch_psi")
            if phi_deg and cols != len(phi_deg):
                errors.append("gain_cols_mismatch_phi")

    if len(psi_deg) > MAX_PSI_POINTS:
        errors.append("psi_grid_too_large")
    if len(phi_deg) > MAX_PHI_POINTS:
        errors.append("phi_grid_too_large")
    if psi_deg and phi_deg and len(psi_deg) * len(phi_deg) > MAX_GAIN_CELLS:
        errors.append("gain_table_too_large")

    # FFD imports are normalized relative to their global peak.  Flag a
    # shaped pattern whose boresight is not that peak so users know the
    # resulting relative gain is referenced to an off-boresight direction.
    if gain_dbi and 0.0 in psi_deg:
        boresight_row = gain_dbi[psi_deg.index(0.0)]
        all_gain_values = [value for row in gain_dbi for value in row]
        if boresight_row and all_gain_values:
            peak_gain = max(all_gain_values)
            if peak_gain - max(boresight_row) > 1e-9:
                warnings.append("boresight_not_pattern_peak")

    # Step 9.4 confidence checks (warnings only, deterministic)
    if enabled and psi_deg and phi_deg and not errors:
        psi_steps = [psi_deg[i + 1] - psi_deg[i] for i in range(len(psi_deg) - 1)]
        phi_steps = [phi_deg[i + 1] - phi_deg[i] for i in range(len(phi_deg) - 1)]
        if psi_steps and max(psi_steps) > SPARSE_GRID_STEP_DEG:
            warnings.append("sparse_psi_grid")
        if phi_steps and max(phi_steps) > SPARSE_GRID_STEP_DEG:
            warnings.append("sparse_phi_grid")
        if not _phi_grid_is_closed(phi_deg):
            warnings.append("phi_coverage_limited")

    return errors, warnings


def _phi_grid_is_closed(phi_deg: list[float]) -> bool:
    if len(phi_deg) < 2:
        return False
    phi_steps = [phi_deg[i + 1] - phi_deg[i] for i in range(len(phi_deg) - 1)]
    gap = 360.0 - phi_deg[-1] + phi_deg[0]
    return gap <= max(phi_steps) + 1e-9
