"""Versioned clutter configuration normalization for RF payloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .clutter import ClutterConfig, ClutterModel, LookupState, LOOKUP_STATE_TO_CODE
from .defaults import (
    CLUTTER_PERCENTILE_INPUT_MAX,
    CLUTTER_PERCENTILE_INPUT_MIN,
    DEFAULT_CLUTTER_PERCENTILE,
)

RF_SCHEMA_VERSION = 2
CLUTTER_MODE_DISABLED = ClutterModel.DISABLED.value
CLUTTER_MODE_WORLDCOVER = ClutterModel.WORLDCOVER_P2108_P833.value
CLUTTER_MODES = frozenset({CLUTTER_MODE_DISABLED, CLUTTER_MODE_WORLDCOVER})
OLD_CLUTTER_FIELDS = frozenset(
    {"clutter_enable", "clutter_enabled", "clutter_values", "clutter_fallback"}
)
CLUTTER_NOTICE_TEXTS = {
    "clutter.below_0p5ghz": "Clutter is not modelled below 0.5 GHz; 0 dB is applied.",
    "clutter.above_100ghz": "Clutter is not modelled above 100 GHz; 0 dB is applied.",
    "clutter.interim_sub10ghz": (
        "Built-up clutter below 10 GHz is extrapolated from ITU-R P.2108-1 §3.3 and is interim."
    ),
    "clutter.data_unavailable": (
        "WorldCover data unavailable for {count} of {total} grid points; those points use the 0 dB fallback."
    ),
    "clutter.rasterio_unavailable": (
        "The rasterio library is not available, so no WorldCover data could be read; "
        "clutter is 0 dB for the whole run."
    ),
}
CLUTTER_NOTICE_THRESHOLDS_GHZ = {
    "min_model_ghz": 0.5,
    "interim_max_ghz": 10.0,
    "max_model_ghz": 100.0,
}


@dataclass
class ClutterTally:
    total: int = 0
    data_unavailable: int = 0
    rasterio_unavailable: bool = False

    def add_state(self, state: LookupState | str) -> None:
        state = LookupState(state)
        self.total += 1
        self.data_unavailable += state in (LookupState.TILE_MISSING, LookupState.READ_FAILED)
        self.rasterio_unavailable |= state == LookupState.RASTERIO_UNAVAILABLE

    def add_grid(self, states, admitted) -> None:
        states = np.asarray(states)
        admitted = np.asarray(admitted, dtype=bool)
        states, admitted = np.broadcast_arrays(states, admitted)
        self.total += int(np.count_nonzero(admitted))
        if np.issubdtype(states.dtype, np.integer):
            missing = np.isin(
                states,
                [LOOKUP_STATE_TO_CODE[LookupState.TILE_MISSING], LOOKUP_STATE_TO_CODE[LookupState.READ_FAILED]],
            )
            unavailable = states == LOOKUP_STATE_TO_CODE[LookupState.RASTERIO_UNAVAILABLE]
        else:
            missing_states = {
                LookupState.TILE_MISSING,
                LookupState.READ_FAILED,
                LOOKUP_STATE_TO_CODE[LookupState.TILE_MISSING],
                LOOKUP_STATE_TO_CODE[LookupState.READ_FAILED],
            }
            unavailable_states = {
                LookupState.RASTERIO_UNAVAILABLE,
                LOOKUP_STATE_TO_CODE[LookupState.RASTERIO_UNAVAILABLE],
            }
            missing = np.fromiter((state in missing_states for state in states.flat), dtype=bool).reshape(states.shape)
            unavailable = np.fromiter(
                (state in unavailable_states for state in states.flat), dtype=bool
            ).reshape(states.shape)
        self.data_unavailable += int(np.count_nonzero(admitted & missing))
        self.rasterio_unavailable |= bool(np.any(admitted & unavailable))

    def merge(self, other: "ClutterTally") -> None:
        self.total += other.total
        self.data_unavailable += other.data_unavailable
        self.rasterio_unavailable |= other.rasterio_unavailable

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "total": self.total,
            "data_unavailable": self.data_unavailable,
            "rasterio_unavailable": self.rasterio_unavailable,
        }

    @classmethod
    def from_dict(cls, value: dict | None) -> "ClutterTally":
        value = value or {}
        return cls(
            total=int(value.get("total", 0)),
            data_unavailable=int(value.get("data_unavailable", 0)),
            rasterio_unavailable=bool(value.get("rasterio_unavailable", False)),
        )


def merge_lookup_states(current: LookupState | None, incoming: LookupState) -> LookupState:
    if LookupState.RASTERIO_UNAVAILABLE in (current, incoming):
        return LookupState.RASTERIO_UNAVAILABLE
    if current in (LookupState.TILE_MISSING, LookupState.READ_FAILED):
        return current
    if incoming in (LookupState.TILE_MISSING, LookupState.READ_FAILED):
        return incoming
    return current or incoming


def build_clutter_envelope(rf_params: dict[str, Any], tally: ClutterTally | None = None) -> dict[str, Any]:
    fields = normalize_clutter_rf(rf_params)
    envelope = {
        "model": fields["clutter_mode"],
        "rf_schema_version": fields["rf_schema_version"],
        "percentile": fields["clutter_percentile"],
        "notices": [],
    }
    if fields["clutter_mode"] == CLUTTER_MODE_DISABLED:
        return envelope

    frequency_hz = rf_params.get("freq_hz")
    if frequency_hz is None:
        frequency_hz = float(rf_params["frequency_ghz"]) * 1e9
    frequency_ghz = float(frequency_hz) / 1e9
    if frequency_ghz < CLUTTER_NOTICE_THRESHOLDS_GHZ["min_model_ghz"]:
        frequency_code = "clutter.below_0p5ghz"
    elif frequency_ghz > CLUTTER_NOTICE_THRESHOLDS_GHZ["max_model_ghz"]:
        frequency_code = "clutter.above_100ghz"
    elif frequency_ghz < CLUTTER_NOTICE_THRESHOLDS_GHZ["interim_max_ghz"]:
        frequency_code = "clutter.interim_sub10ghz"
    else:
        frequency_code = None
    if frequency_code:
        envelope["notices"].append(
            {"code": frequency_code, "message": CLUTTER_NOTICE_TEXTS[frequency_code], "count": None, "total": None}
        )
    if tally is not None:
        if tally.rasterio_unavailable:
            code = "clutter.rasterio_unavailable"
            envelope["notices"].append(
                {"code": code, "message": CLUTTER_NOTICE_TEXTS[code], "count": None, "total": None}
            )
        elif tally.data_unavailable:
            code = "clutter.data_unavailable"
            envelope["notices"].append(
                {
                    "code": code,
                    "message": CLUTTER_NOTICE_TEXTS[code].format(
                        count=tally.data_unavailable, total=tally.total
                    ),
                    "count": tally.data_unavailable,
                    "total": tally.total,
                }
            )
    return envelope


class ClutterConfigError(ValueError):
    """Raised when a clutter configuration payload has an unsupported version."""


def _require_version(rf_params: dict[str, Any]) -> int:
    raw_version = rf_params.get("rf_schema_version")
    if raw_version is None or raw_version == "":
        raise ClutterConfigError(
            "rf_schema_version is required; run tools/convert_clutter_config.py on saved RF settings"
        )
    if type(raw_version) not in (int, str) or raw_version not in (1, "1", 2, "2"):
        raise ClutterConfigError(f"unsupported rf_schema_version {raw_version!r}")
    return int(raw_version)


def _reject_old_fields(rf_params: dict[str, Any]) -> None:
    present = sorted(field for field in OLD_CLUTTER_FIELDS if field in rf_params)
    if present:
        raise ClutterConfigError(
            "old clutter fields are not accepted: "
            + ", ".join(present)
            + "; run tools/convert_clutter_config.py on saved RF settings"
        )


def _normalize_mode(rf_params: dict[str, Any]) -> str:
    if "clutter_mode" not in rf_params or rf_params.get("clutter_mode") in (None, ""):
        raise ClutterConfigError("clutter_mode is required")
    mode = str(rf_params["clutter_mode"]).strip()
    if mode not in CLUTTER_MODES:
        raise ClutterConfigError(
            f"clutter_mode {mode!r} must be disabled or worldcover_p2108_p833"
        )
    return mode


def _normalize_percentile(rf_params: dict[str, Any], mode: str) -> float | None:
    if mode == CLUTTER_MODE_DISABLED:
        return None
    raw_percentile = rf_params.get("clutter_percentile", DEFAULT_CLUTTER_PERCENTILE)
    if raw_percentile is None or raw_percentile == "":
        raw_percentile = DEFAULT_CLUTTER_PERCENTILE
    if isinstance(raw_percentile, bool):
        raise ClutterConfigError("clutter_percentile must be numeric")
    try:
        percentile = float(raw_percentile)
    except (TypeError, ValueError) as exc:
        raise ClutterConfigError("clutter_percentile must be numeric") from exc
    if not math.isfinite(percentile):
        raise ClutterConfigError("clutter_percentile must be finite")
    if not (
        CLUTTER_PERCENTILE_INPUT_MIN
        <= percentile
        <= CLUTTER_PERCENTILE_INPUT_MAX
    ):
        raise ClutterConfigError(
            "clutter_percentile must be between "
            f"{CLUTTER_PERCENTILE_INPUT_MIN:g} and {CLUTTER_PERCENTILE_INPUT_MAX:g}"
        )
    return percentile


def normalize_clutter_rf(rf_params: dict[str, Any]) -> dict[str, Any]:
    """Normalize supported-version clutter RF payload fields.

    Spec section 8 item 8 owns all mode resolution. Callers must pass only
    versioned input; old table, fallback and enable fields are rejected by name.
    """
    version = _require_version(rf_params)
    _reject_old_fields(rf_params)
    mode = _normalize_mode(rf_params)
    percentile = _normalize_percentile(rf_params, mode)
    return {
        "rf_schema_version": version,
        "clutter_mode": mode,
        "clutter_percentile": percentile,
    }


def build_clutter_config(rf_params: dict[str, Any]) -> ClutterConfig:
    """Build a ClutterConfig from normalized RF payload fields."""
    normalized = normalize_clutter_rf(rf_params)
    return ClutterConfig(
        normalized["clutter_mode"],
        normalized["clutter_percentile"],
    )
