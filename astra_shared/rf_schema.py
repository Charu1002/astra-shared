"""Version 1-to-2 RF conversion and Version 2 persistence helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .clutter_config import RF_SCHEMA_VERSION, _require_version
from .defaults import DEFAULT_CODE_RATE, DEFAULT_MODULATION

DEFAULT_MCS_SELECTION = "fixed"
DEFAULT_SHANNON_ATTENUATION_FACTOR = 0.6
DEFAULT_OVERHEAD_PERCENT = 0

_VERSION_2_ONLY_KEYS = (
    "digital_link",
    "mcs_selection",
    "shannon_attenuation_factor",
    "overhead_percent",
)


class RfConfigError(ValueError):
    """Raised when RF conversion, persistence, or run validation fails."""


class RfParamParseError(ValueError):
    """A typed RF request parsing error with stable, user-facing codes."""

    def __init__(self, message: str, *, codes: tuple[str, ...] | list[str]):
        if not codes or any(not isinstance(code, str) or not code for code in codes):
            raise ValueError("RfParamParseError requires at least one error code")
        super().__init__(message)
        self.codes = list(codes)


class RfKeyError(RfParamParseError, RfConfigError):
    """A key-specific RF error usable by both saved-config and route callers."""

    def __init__(self, message: str, *, key: str, code: str):
        if not isinstance(key, str) or not key:
            raise ValueError("RfKeyError requires a non-empty RF key")
        if not isinstance(code, str) or not code:
            raise ValueError("RfKeyError requires a non-empty error code")
        self.key = key
        self.code = code
        super().__init__(message, codes=[code])


@dataclass(frozen=True)
class RfConversion:
    source_version: int
    target_version: int
    converted: bool
    digital_link: bool


def parse_bool_value(value: Any, default: bool = False) -> bool:
    """Apply Astra's existing permissive RF Boolean convention."""
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in (
        "true",
        "1",
        "yes",
        "on",
        "enable",
        "enabled",
    )


def _legacy_data_rate_enables_link(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric) and numeric > 0


def convert_rf_block(
    block: Mapping[str, Any],
) -> tuple[dict[str, Any], RfConversion]:
    """Return a clean Version 2 copy and an explicit conversion result."""
    converted = dict(block)
    try:
        source_version = _require_version(converted)
    except ValueError as exc:
        raise RfKeyError(
            str(exc),
            key="rf_schema_version",
            code="rf_schema_version_invalid",
        ) from exc

    if source_version == 2:
        if "data_rate_bps" in converted:
            raise RfKeyError(
                "data_rate_bps is not a key of rf_schema_version 2; remove it",
                key="data_rate_bps",
                code="data_rate_bps_not_v2_key",
            )
        digital_link = parse_bool_value(converted.get("digital_link"), False)
        return converted, RfConversion(2, 2, False, digital_link)

    for key in _VERSION_2_ONLY_KEYS:
        if key in converted:
            raise RfKeyError(
                f"{key} is not a key of rf_schema_version 1",
                key=key,
                code=f"{key}_not_v1_key",
            )

    digital_link = _legacy_data_rate_enables_link(converted.get("data_rate_bps"))
    converted.pop("data_rate_bps", None)
    converted["rf_schema_version"] = RF_SCHEMA_VERSION
    converted["digital_link"] = digital_link
    converted["mcs_selection"] = DEFAULT_MCS_SELECTION
    return converted, RfConversion(1, 2, True, digital_link)


def complete_saved_rf_keys(block: Mapping[str, Any]) -> dict[str, Any]:
    """Complete the six saved RF keys without performing run validation."""
    completed = dict(block)
    try:
        version = _require_version(completed)
    except ValueError as exc:
        raise RfKeyError(
            str(exc),
            key="rf_schema_version",
            code="rf_schema_version_invalid",
        ) from exc
    if version != RF_SCHEMA_VERSION:
        raise RfKeyError(
            "saved RF block must use rf_schema_version 2",
            key="rf_schema_version",
            code="rf_schema_version_not_v2",
        )
    if "data_rate_bps" in completed:
        raise RfKeyError(
            "data_rate_bps is not a key of rf_schema_version 2; remove it",
            key="data_rate_bps",
            code="data_rate_bps_not_v2_key",
        )

    defaults = {
        "digital_link": False,
        "mcs_selection": DEFAULT_MCS_SELECTION,
        "modulation": DEFAULT_MODULATION,
        "code_rate": DEFAULT_CODE_RATE,
        "shannon_attenuation_factor": DEFAULT_SHANNON_ATTENUATION_FACTOR,
        "overhead_percent": DEFAULT_OVERHEAD_PERCENT,
    }
    for key, default in defaults.items():
        if key not in completed or completed[key] in (None, ""):
            completed[key] = default
    completed["digital_link"] = parse_bool_value(completed["digital_link"], False)
    completed["rf_schema_version"] = RF_SCHEMA_VERSION
    return completed
