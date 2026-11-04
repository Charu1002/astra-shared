"""Shared RF parameter parsing helpers used by all Astra services."""

import json
import logging
import math

from .clutter_config import normalize_clutter_rf
from .custom_antenna_schema import normalize_custom_antenna
from .defaults import (
    ADDITIONAL_LOSSES_DB_MAX,
    ADDITIONAL_LOSSES_DB_MIN,
    DEFAULT_BANDWIDTH_HZ,
    DEFAULT_CODE_RATE,
    DEFAULT_COMPUTE_PFD,
    DEFAULT_EIRP_DBW,
    DEFAULT_MODULATION,
    DEFAULT_PFD_LIMIT_BAND,
    DEFAULT_PFD_REF_BW_HZ,
    DEFAULT_RX_GAIN_DBI,
    DEFAULT_SYSTEM_NOISE_TEMP_K,
    POLARIZATION_LOSS_DB_MAX,
    POLARIZATION_LOSS_DB_MIN,
)
from .rf_schema import (
    DEFAULT_OVERHEAD_PERCENT,
    DEFAULT_SHANNON_ATTENUATION_FACTOR,
    RfConfigError,
    RfKeyError,
    RfParamParseError,
    convert_rf_block,
    parse_bool_value,
)

logger = logging.getLogger(__name__)

MODULATIONS = {"BPSK", "QPSK", "OQPSK", "8PSK", "16QAM", "64QAM", "256QAM"}
VALID_CODE_RATES = (
    1.0 / 4.0,
    1.0 / 3.0,
    2.0 / 5.0,
    1.0 / 2.0,
    3.0 / 5.0,
    2.0 / 3.0,
    3.0 / 4.0,
    4.0 / 5.0,
    5.0 / 6.0,
    8.0 / 9.0,
    9.0 / 10.0,
    1.0,
)
VALID_CODE_RATE_LABELS = "1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10, 1"
# TX antenna peak gain limits in dBi (astra-shared#82). ITU-R S.672 keeps the
# 10 dBi minimum: its far sidelobe level L_F is fixed at 0 dBi, so below 10 dBi
# the far sidelobes come within 10 dB of boresight, and rise above boresight
# once the peak gain is below 0 dBi.
MAX_GAIN_DBI_MIN = -10.0
MAX_GAIN_DBI_MAX = 60.0
S672_MAX_GAIN_DBI_MIN = 10.0
CUSTOM_PFD_LIMIT_MIN_DBW_M2 = -200.0
CUSTOM_PFD_LIMIT_MAX_DBW_M2 = 0.0
PFD_LIMIT_PRESETS = {
    "S-2500-2690-FSS": {"l0": -136.0, "l25": -125.0, "ref_bw_hz": 1.0e6},
    "C-3400-4200-GSO": {"l0": -152.0, "l25": -142.0, "ref_bw_hz": 4.0e3},
    "C-4500-4800-FSS": {"l0": -152.0, "l25": -142.0, "ref_bw_hz": 4.0e3},
    "C-5150-5216-FSS": {"l0": -164.0, "l25": -164.0, "ref_bw_hz": 4.0e3},
    "C-6700-6825-FSS": {"l0": -137.0, "l25": -127.0, "ref_bw_hz": 1.0e6},
    "C-6825-7075-FSS": {
        "conjunctive": [
            {"l0": -154.0, "l25": -144.0, "ref_bw_hz": 4.0e3},
            {"l0": -134.0, "l25": -124.0, "ref_bw_hz": 1.0e6},
        ],
    },
    "X-7250-7900-FSS": {"l0": -152.0, "l25": -142.0, "ref_bw_hz": 4.0e3},
    "Ku-10700-11700-GSO": {"l0": -150.0, "l25": -140.0, "ref_bw_hz": 4.0e3},
    "Ku-10700-11700-NGSO-normal": {"l0": -126.0, "l25": -116.0, "ref_bw_hz": 1.0e6},
    "Ka-17700-19300-GSO-or-old-NGSO": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Ka-19300-19700-FSS": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Ka-27500-27501-FSS": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Q-37500-40000-NGSO": {
        "l0": -120.0,
        "l25": -105.0,
        "ref_bw_hz": 1.0e6,
        "slope": 0.75,
    },
    "Q-37500-40000-GSO": {"ref_bw_hz": 1.0e6, "shape": "q_gso_127"},
    "Q-40000-40500-FSS": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Q-40500-42000-NGSO": {"l0": -115.0, "l25": -105.0, "ref_bw_hz": 1.0e6},
    "Q-40500-42000-GSO": {"ref_bw_hz": 1.0e6, "shape": "q_gso_120"},
    "Q-42000-42500-NGSO": {
        "l0": -120.0,
        "l25": -105.0,
        "ref_bw_hz": 1.0e6,
        "slope": 0.75,
    },
    "Q-42000-42500-GSO": {"ref_bw_hz": 1.0e6, "shape": "q_gso_127"},
}


def _get_float(
    params: dict,
    key: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Parse a float parameter with optional clamping."""
    raw_value = params.get(key, default)
    if raw_value in (None, ""):
        raw_value = default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def _get_float_pair(
    params: dict,
    key: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> tuple[float, float]:
    """Parse a finite float once and return its requested and clamped values.

    The requested value is the finite value supplied by the user (or the
    default when the input is absent or invalid).  Keeping it alongside the
    clamped value lets callers report an adjustment without parsing the input
    a second time.
    """
    raw_value = params.get(key, default)
    if raw_value in (None, ""):
        raw_value = default
    try:
        requested = float(raw_value)
    except (TypeError, ValueError):
        requested = float(default)
    if not math.isfinite(requested):
        requested = float(default)

    value = requested
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return requested, value


def _get_int(
    params: dict,
    key: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Parse an integer parameter with optional clamping."""
    raw_value = params.get(key, default)
    if raw_value in (None, ""):
        raw_value = default
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError):
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def _get_str(params: dict, key: str, default: str) -> str:
    """Parse a string parameter."""
    raw_value = params.get(key, default)
    if raw_value is None:
        raw_value = default
    return str(raw_value).strip()


def _parse_eirp(params: dict) -> float:
    """Parse EIRP, supporting both new (eirp_dbw) and legacy (tx_power + tx_gain) formats."""
    if "eirp" in params:
        return _get_float(params, "eirp", DEFAULT_EIRP_DBW)
    if "eirp_dbw" in params:
        return _get_float(params, "eirp_dbw", DEFAULT_EIRP_DBW)
    if "tx_power" in params and "tx_gain" in params:
        return _get_float(params, "tx_power", 40.0) + _get_float(
            params, "tx_gain", 30.0
        )
    if "tx_power_dbw" in params and "tx_gain_dbi" in params:
        return _get_float(params, "tx_power_dbw", 40.0) + _get_float(
            params, "tx_gain_dbi", 30.0
        )
    return DEFAULT_EIRP_DBW


def _parse_custom_antenna_payload(params: dict) -> dict:
    raw = params.get("custom_antenna")
    if raw is None or raw == "" or raw == "null":
        return normalize_custom_antenna(None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return normalize_custom_antenna(None)
    return normalize_custom_antenna(raw)


def _normalize_antenna_model(value: str | None) -> str:
    model = str(value or "gaussian").strip().lower()
    allowed = {"gaussian", "bessel", "itu_s672", "phased_array", "custom"}
    return model if model in allowed else "gaussian"


def _parse_max_gain_dbi(params: dict, antenna_model: str) -> float:
    min_val = S672_MAX_GAIN_DBI_MIN if antenna_model == "itu_s672" else MAX_GAIN_DBI_MIN
    return _get_float(
        params, "max_gain_dbi", 30.0, min_val=min_val, max_val=MAX_GAIN_DBI_MAX
    )


def _parse_modulation(params: dict) -> str:
    raw = params.get("modulation")
    if raw in (None, ""):
        raw = DEFAULT_MODULATION
    modulation = str(raw or DEFAULT_MODULATION).strip().upper()
    if modulation not in MODULATIONS:
        raise RfKeyError(
            f"Unsupported modulation: {modulation}",
            key="modulation",
            code="invalid_modulation",
        )
    return modulation


def _parse_code_rate(params: dict) -> float:
    raw = params.get("code_rate", DEFAULT_CODE_RATE)
    if raw in (None, "", "null"):
        return DEFAULT_CODE_RATE
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RfKeyError(
            f"code_rate must be one of: {VALID_CODE_RATE_LABELS}",
            key="code_rate",
            code="code_rate_invalid",
        ) from exc
    if not any(
        math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-9)
        for allowed in VALID_CODE_RATES
    ):
        raise RfKeyError(
            f"code_rate must be one of: {VALID_CODE_RATE_LABELS}",
            key="code_rate",
            code="code_rate_invalid",
        )
    return value


def _parse_bool(params: dict, key: str, default: bool) -> bool:
    return parse_bool_value(params.get(key, default), default)


def _parse_mcs_selection(params: dict) -> str:
    raw = params.get("mcs_selection", "fixed")
    if raw in (None, "", "null"):
        return "fixed"
    # Plan v6.1 section 2.1: exact match, no case folding or trimming.  The
    # browser applies the same rule, so a saved value runs identically everywhere.
    if raw not in ("fixed", "auto"):
        raise RfKeyError(
            "mcs_selection must be fixed or auto",
            key="mcs_selection",
            code="mcs_selection_invalid",
        )
    return raw


def _parse_conditional_number(
    params: dict,
    key: str,
    default: float,
    *,
    used: bool,
    valid,
    error: str,
    code: str,
) -> float:
    raw = params.get(key, default)
    if raw in (None, "", "null"):
        raw = default
    if isinstance(raw, bool):
        if used:
            raise RfKeyError(error, key=key, code=code)
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        if used:
            raise RfKeyError(error, key=key, code=code)
        return default
    if not math.isfinite(value) or not valid(value):
        if used:
            raise RfKeyError(error, key=key, code=code)
        # Hidden, unused values remain part of the saved project.  Preserve
        # finite values so a Fixed/link-off replay does not rewrite them.
        if math.isfinite(value):
            return value
        return default
    return value


def _resolve_bandwidth_hz(params: dict) -> float:
    if params.get("bandwidth_mhz") not in (None, "", "null"):
        key = "bandwidth_mhz"
        multiplier = 1.0e6
    elif params.get("bandwidth_hz") not in (None, "", "null"):
        key = "bandwidth_hz"
        multiplier = 1.0
    else:
        return float(DEFAULT_BANDWIDTH_HZ)
    raw = params[key]
    code = "bandwidth_mhz_invalid" if key == "bandwidth_mhz" else "bandwidth_hz_invalid"
    if isinstance(raw, bool):
        raise RfKeyError(
            f"{key} must be a positive finite number",
            key=key,
            code=code,
        )
    try:
        value = float(raw) * multiplier
    except (TypeError, ValueError, OverflowError) as exc:
        raise RfKeyError(
            f"{key} must be a positive finite number",
            key=key,
            code=code,
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise RfKeyError(
            f"{key} must be a positive finite number",
            key=key,
            code=code,
        )
    return value


def _parse_optional_float(params: dict, key: str) -> float | None:
    raw = params.get(key)
    if raw in (None, "", "null"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RfKeyError(
            f"{key} must be finite",
            key=key,
            code="pfd_ref_bw_hz_invalid"
            if key == "pfd_ref_bw_hz"
            else "pfd_custom_limit_invalid",
        ) from exc
    if not math.isfinite(value):
        raise RfKeyError(
            f"{key} must be finite",
            key=key,
            code="pfd_ref_bw_hz_invalid"
            if key == "pfd_ref_bw_hz"
            else "pfd_custom_limit_invalid",
        )
    return value


def _parse_pfd_limit_band(params: dict) -> str | None:
    raw = params.get("pfd_limit_band", DEFAULT_PFD_LIMIT_BAND)
    if raw in (None, "", "null", "none"):
        return None
    band = str(raw).strip()
    if band.lower() == "none":
        return None
    if band not in PFD_LIMIT_PRESETS and band != "custom":
        raise RfKeyError(
            f"Unsupported PFD limit band: {raw}",
            key="pfd_limit_band",
            code="pfd_limit_band_invalid",
        )
    return band


def _validate_custom_pfd_limit(value: float, key: str) -> float:
    if value < CUSTOM_PFD_LIMIT_MIN_DBW_M2 or value > CUSTOM_PFD_LIMIT_MAX_DBW_M2:
        raise RfKeyError(
            f"{key} must be between {CUSTOM_PFD_LIMIT_MIN_DBW_M2:g} "
            f"and {CUSTOM_PFD_LIMIT_MAX_DBW_M2:g} dBW/m^2",
            key=key,
            code="pfd_custom_limit_out_of_range",
        )
    return value


def _pfd_checks_for_band(
    band: str | None,
    custom_l0: float | None = None,
    custom_l25: float | None = None,
    custom_ref_bw_hz: float | None = None,
) -> list[dict]:
    """Return the complete PFD mask definition for downstream callers.

    The legacy scalar fields cannot represent conjunctive or shaped masks, so
    keep those fields for compatibility and expose the complete checks here.
    """
    if band in PFD_LIMIT_PRESETS:
        preset = PFD_LIMIT_PRESETS[band]
        checks = preset.get("conjunctive", [preset])
        return [dict(check) for check in checks]
    if band == "custom" and custom_l0 is not None and custom_l25 is not None:
        return [
            {
                "l0": custom_l0,
                "l25": custom_l25,
                "ref_bw_hz": custom_ref_bw_hz,
            }
        ]
    return []


def _parse_pfd_params(
    params: dict,
) -> tuple[bool, str | None, float | None, float | None, float]:
    compute_pfd = _parse_bool(params, "compute_pfd", DEFAULT_COMPUTE_PFD)
    if not compute_pfd:
        return compute_pfd, None, None, None, DEFAULT_PFD_REF_BW_HZ

    pfd_limit_band = _parse_pfd_limit_band(params)
    pfd_ref_bw_hz = _parse_optional_float(params, "pfd_ref_bw_hz")
    if pfd_ref_bw_hz is None:
        pfd_ref_bw_hz = DEFAULT_PFD_REF_BW_HZ
    if pfd_ref_bw_hz <= 0.0:
        raise RfKeyError(
            "pfd_ref_bw_hz must be greater than 0",
            key="pfd_ref_bw_hz",
            code="pfd_ref_bw_hz_out_of_range",
        )

    if pfd_limit_band in PFD_LIMIT_PRESETS:
        preset = PFD_LIMIT_PRESETS[pfd_limit_band]
        if "conjunctive" in preset:
            preset = preset["conjunctive"][0]
        if "shape" in preset:
            return compute_pfd, pfd_limit_band, None, None, preset["ref_bw_hz"]
        return (
            compute_pfd,
            pfd_limit_band,
            preset["l0"],
            preset["l25"],
            preset["ref_bw_hz"],
        )

    if pfd_limit_band == "custom":
        pfd_l0_dbw_m2 = _parse_optional_float(params, "pfd_l0_dbw_m2")
        pfd_l25_dbw_m2 = _parse_optional_float(params, "pfd_l25_dbw_m2")
        if pfd_l0_dbw_m2 is None or pfd_l25_dbw_m2 is None:
            raise RfKeyError(
                "custom PFD limit requires pfd_l0_dbw_m2 and pfd_l25_dbw_m2",
                key="pfd_limit_band",
                code="pfd_custom_limit_invalid",
            )
        pfd_l0_dbw_m2 = _validate_custom_pfd_limit(pfd_l0_dbw_m2, "pfd_l0_dbw_m2")
        pfd_l25_dbw_m2 = _validate_custom_pfd_limit(pfd_l25_dbw_m2, "pfd_l25_dbw_m2")
        return compute_pfd, pfd_limit_band, pfd_l0_dbw_m2, pfd_l25_dbw_m2, pfd_ref_bw_hz

    return compute_pfd, None, None, None, pfd_ref_bw_hz


def _normalize_pointing_mode(value: str | None) -> tuple[str, str | None]:
    if value is None or value == "":
        return "nadir", None
    normalized = {
        "nadir": "nadir",
        "earth_moving": "nadir",
        "earth-moving": "nadir",
        "earth_fixed": "earth_fixed",
        "earth-fixed": "earth_fixed",
        "targeted": "earth_fixed",
    }.get(str(value).strip().lower())
    return (normalized, None) if normalized else ("nadir", "invalid_pointing_mode")


def _normalize_assignment_rule(value: str | None) -> tuple[str, str | None]:
    if value is None or value == "":
        return "highest_elevation", None
    normalized = str(value).strip().lower()
    if normalized in {"highest_elevation", "highest_cn"}:
        return normalized, None
    return "highest_elevation", "invalid_assignment_rule"


def _parse_bounded_float(
    params: dict,
    key: str,
    default: float,
    min_val: float,
    max_val: float,
    error_prefix: str,
) -> tuple[float, str | None]:
    raw = params.get(key)
    if raw is None or raw == "":
        return default, None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default, f"{error_prefix}_invalid"
    if not math.isfinite(value):
        return default, f"{error_prefix}_invalid"
    if value < min_val or value > max_val:
        return default, f"{error_prefix}_out_of_range"
    return value, None


def _parse_min_elevation_deg(params: dict, default: float) -> tuple[float, str | None]:
    key = (
        "min_elevation_deg"
        if params.get("min_elevation_deg") not in (None, "")
        else "min_el_deg"
    )
    return _parse_bounded_float(params, key, default, 0.0, 90.0, key)


def _canonicalize_targets(params: dict, errors: list[str]) -> list[dict]:
    """Decode and normalize the two supported target input representations."""
    sources = []
    if "targets_json" in params and params.get("targets_json") not in (None, ""):
        try:
            decoded = params["targets_json"]
            sources.append(json.loads(decoded) if isinstance(decoded, str) else decoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append("invalid_targets_json")
    if "targets" in params and params.get("targets") is not None:
        sources.append(params.get("targets"))

    canonical_sources = []
    for source in sources:
        if not isinstance(source, list):
            errors.append("targets_not_list")
            continue
        used_ids: set[str] = set()
        canonical = []
        for index, row in enumerate(source):
            if not isinstance(row, dict):
                errors.append("target_not_object")
                continue
            raw_target_id = row.get("target_id")
            # Treat an explicit JSON null as "not supplied" (auto-generate), not as
            # the literal string "None"; an explicit empty string is still a blank error.
            target_id_supplied = "target_id" in row and raw_target_id is not None
            target_id = str(raw_target_id).strip() if raw_target_id is not None else ""
            if target_id_supplied and not target_id:
                errors.append("target_id_blank")
                continue
            if not target_id:
                next_id = index + 1
                target_id = f"target-{next_id}"
                while target_id in used_ids:
                    next_id += 1
                    target_id = f"target-{next_id}"
            if target_id in used_ids:
                errors.append("target_id_duplicate")
                continue
            used_ids.add(target_id)
            label = (
                str(row.get("label") or f"Target-{index + 1}").strip()
                or f"Target-{index + 1}"
            )
            # One conversion per field, so each error names its own field.
            try:
                lat = float(row.get("target_lat"))
            except (TypeError, ValueError, OverflowError):
                errors.append("target_lat_invalid")
                continue
            try:
                lon = float(row.get("target_lon"))
            except (TypeError, ValueError, OverflowError):
                errors.append("target_lon_invalid")
                continue
            try:
                alt = float(row.get("target_alt_km", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                errors.append("target_altitude_unsupported")
                continue
            if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
                errors.append(
                    "target_lat_out_of_range"
                    if math.isfinite(lat)
                    else "target_lat_invalid"
                )
                continue
            if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
                errors.append(
                    "target_lon_out_of_range"
                    if math.isfinite(lon)
                    else "target_lon_invalid"
                )
                continue
            if not math.isfinite(alt) or alt != 0.0:
                errors.append("target_altitude_unsupported")
                continue
            canonical.append(
                {
                    "target_id": target_id,
                    "label": label,
                    "target_lat": lat,
                    "target_lon": lon,
                    "target_alt_km": 0.0,
                }
            )
        canonical_sources.append(canonical)

    if len(canonical_sources) == 2 and canonical_sources[0] != canonical_sources[1]:
        errors.append("conflicting_targets_sources")
    return canonical_sources[0] if canonical_sources else []


# =============================================================================
# Unified RF Parameter Parsing
# =============================================================================


def _parse_rf_params(
    params: dict,
    *,
    path_default_min_elevation_deg: float = 5.0,
    attach_earth_fixed_errors: bool = False,
) -> dict:
    """Parse RF parameters from any source into a canonical dict.

    Accepts form args, config.json, project files, or HTTP request bodies.
    Callers use the subset they need � unused keys are harmless.
    """
    params, _conversion = convert_rf_block(params)

    if params.get("frequency_ghz") not in (None, "") or params.get("frequency") not in (
        None,
        "",
    ):
        freq_ghz = _get_float(
            params,
            "frequency_ghz",
            _get_float(params, "frequency", 12.0),
            min_val=0.001,
        )
        freq_hz = freq_ghz * 1e9
    else:
        freq_hz = _get_float(
            params, "freq_hz", _get_float(params, "frequency_hz", 12.0e9), min_val=1.0e6
        )

    wavelength_m = 3.0e8 / freq_hz
    aperture_radius_m_raw = params.get("aperture_radius_m")
    if aperture_radius_m_raw in (None, ""):
        aperture_radius_wl = _get_float(params, "aperture_radius_wl", 10.0, min_val=1.0)
        aperture_radius_m = aperture_radius_wl * wavelength_m
    else:
        aperture_radius_m = _get_float(
            params, "aperture_radius_m", 10.0 * wavelength_m, min_val=0.0
        )
        aperture_radius_wl = aperture_radius_m / wavelength_m
    system_noise_temp_k = _get_float(
        params,
        "system_noise_temp_k",
        DEFAULT_SYSTEM_NOISE_TEMP_K,
        min_val=10.0,
        max_val=10000.0,
    )

    bandwidth_hz = _resolve_bandwidth_hz(params)

    logger.debug(
        "[RF] noise_temp_k=%s bandwidth_hz=%s cn_enabled=%s",
        system_noise_temp_k,
        bandwidth_hz,
        bandwidth_hz is not None,
    )

    antenna_model = _normalize_antenna_model(
        _get_str(params, "antenna_model", "gaussian")
    )

    earth_fixed_errors: list[str] = []
    pointing_mode, pointing_error = _normalize_pointing_mode(
        params.get("pointing_mode")
    )
    if pointing_error:
        earth_fixed_errors.append(pointing_error)
    assignment_rule, assignment_error = _normalize_assignment_rule(
        params.get("assignment_rule")
    )
    if assignment_error:
        earth_fixed_errors.append(assignment_error)
    max_steer_deg, max_steer_error = _parse_bounded_float(
        params, "max_steer_deg", 60.0, 0.0, 90.0, "max_steer_deg"
    )
    if max_steer_error:
        earth_fixed_errors.append(max_steer_error)
    min_elevation_deg, min_elevation_error = _parse_min_elevation_deg(
        params, float(path_default_min_elevation_deg)
    )
    if min_elevation_error:
        earth_fixed_errors.append(min_elevation_error)
    for elevation_key in ("min_elevation_deg", "min_el_deg"):
        if params.get(elevation_key) not in (None, ""):
            _, alias_error = _parse_bounded_float(
                params,
                elevation_key,
                float(path_default_min_elevation_deg),
                0.0,
                90.0,
                elevation_key,
            )
            if alias_error and alias_error not in earth_fixed_errors:
                earth_fixed_errors.append(alias_error)
    if "debug_target_reachability" in params:
        earth_fixed_errors.append("debug_target_reachability_not_public")
    targets = _canonicalize_targets(params, earth_fixed_errors)
    if pointing_mode == "earth_fixed" and not targets:
        earth_fixed_errors.append("earth_fixed_targets_required")
    if pointing_mode == "earth_fixed" and antenna_model == "phased_array":
        earth_fixed_errors.append("earth_fixed_phased_array_unsupported")
    if pointing_mode != "earth_fixed" and not pointing_error:
        earth_fixed_errors = []
        min_elevation_deg = _get_float(
            params,
            "min_elevation_deg"
            if params.get("min_elevation_deg") not in (None, "")
            else "min_el_deg",
            float(path_default_min_elevation_deg),
            min_val=0.0,
            max_val=90.0,
        )

    digital_link = _parse_bool(params, "digital_link", False)
    mcs_selection = _parse_mcs_selection(params)
    shannon_attenuation_factor = _parse_conditional_number(
        params,
        "shannon_attenuation_factor",
        DEFAULT_SHANNON_ATTENUATION_FACTOR,
        used=digital_link and mcs_selection == "auto",
        valid=lambda value: 0 < value <= 1,
        error="shannon_attenuation_factor must be in (0, 1]",
        code="shannon_attenuation_factor_invalid",
    )
    overhead_percent = _parse_conditional_number(
        params,
        "overhead_percent",
        DEFAULT_OVERHEAD_PERCENT,
        used=digital_link,
        valid=lambda value: 0 <= value < 50,
        error="overhead_percent must be in [0, 50)",
        code="overhead_percent_invalid",
    )
    modulation = _parse_modulation(params)
    code_rate = _parse_code_rate(params)
    compute_pfd, pfd_limit_band, pfd_l0_dbw_m2, pfd_l25_dbw_m2, pfd_ref_bw_hz = (
        _parse_pfd_params(params)
    )
    pfd_checks = _pfd_checks_for_band(
        pfd_limit_band,
        pfd_l0_dbw_m2,
        pfd_l25_dbw_m2,
        pfd_ref_bw_hz,
    )

    clutter_rf = normalize_clutter_rf(params)

    parsed = {
        "eirp_dbw": _parse_eirp(params),
        "rx_gain_dbi": _get_float(
            params, "rx_gain_dbi", _get_float(params, "rx_gain", DEFAULT_RX_GAIN_DBI)
        ),
        "freq_hz": freq_hz,
        "antenna_model": antenna_model,
        "custom_antenna": _parse_custom_antenna_payload(params),
        "beamwidth_deg": _get_float(
            params, "beamwidth_deg", _get_float(params, "beamwidth", 4.5), min_val=0.1
        ),
        "aperture_radius_wl": aperture_radius_wl,
        "aperture_radius_m": aperture_radius_m,
        "max_gain_dbi": _parse_max_gain_dbi(params, antenna_model),
        "ln_db": _get_float(params, "ln_db", -20.0),
        "ellipticity_ratio": _get_float(
            params, "ellipticity_ratio", 1.0, min_val=1.0, max_val=3.0
        ),
        "num_elements_x": _get_int(params, "num_elements_x", 8, min_val=1, max_val=64),
        "num_elements_y": _get_int(params, "num_elements_y", 8, min_val=1, max_val=64),
        "spacing_wl": _get_float(params, "spacing_wl", 0.5, min_val=0.1, max_val=2.0),
        "element_exponent": _get_float(
            params, "element_exponent", 1.3, min_val=0.0, max_val=3.0
        ),
        "rf_schema_version": clutter_rf["rf_schema_version"],
        "clutter_mode": clutter_rf["clutter_mode"],
        "clutter_percentile": clutter_rf["clutter_percentile"],
        "atmospheric_mode": _get_str(params, "atmospheric_mode", "disable").lower(),
        "availability_percent": _get_float(
            params, "availability_percent", 99.0, min_val=90.0, max_val=99.999
        ),
        "additional_losses_db": _get_float(
            params,
            "additional_losses_db",
            2.0,
            min_val=ADDITIONAL_LOSSES_DB_MIN,
            max_val=ADDITIONAL_LOSSES_DB_MAX,
        ),
        "polarization_loss_db": _get_float(
            params,
            "polarization_loss_db",
            0.0,
            min_val=POLARIZATION_LOSS_DB_MIN,
            max_val=POLARIZATION_LOSS_DB_MAX,
        ),
        "system_noise_temp_k": system_noise_temp_k,
        "bandwidth_hz": bandwidth_hz,
        "digital_link": digital_link,
        "mcs_selection": mcs_selection,
        "modulation": modulation,
        "code_rate": code_rate,
        "shannon_attenuation_factor": shannon_attenuation_factor,
        "overhead_percent": overhead_percent,
        "compute_pfd": compute_pfd,
        "pfd_limit_band": pfd_limit_band,
        "pfd_ref_bw_hz": pfd_ref_bw_hz,
        "pfd_l0_dbw_m2": pfd_l0_dbw_m2,
        "pfd_l25_dbw_m2": pfd_l25_dbw_m2,
        "pfd_checks": pfd_checks,
        "min_el_deg": min_elevation_deg,
        "min_elevation_deg": min_elevation_deg,
        "pointing_mode": pointing_mode,
        "assignment_rule": assignment_rule,
        "max_steer_deg": max_steer_deg,
        "targets": targets,
    }
    if earth_fixed_errors:
        if attach_earth_fixed_errors:
            # Collect mode (R8-S1): report the Earth-Fixed codes but keep the
            # parsed pointing mode and the valid target rows, as main did, so
            # callers such as the scalar point fallback still see earth_fixed.
            parsed["_earth_fixed_errors"] = list(dict.fromkeys(earth_fixed_errors))
            return parsed
        raise RfParamParseError(
            "Invalid RF request parameters", codes=earth_fixed_errors
        )
    return parsed


_PFD_KEYS = frozenset(
    {
        "compute_pfd",
        "pfd_limit_band",
        "pfd_ref_bw_hz",
        "pfd_l0_dbw_m2",
        "pfd_l25_dbw_m2",
    }
)
# A bad bandwidth key masks the other one only when both are present; drop both
# so the unused alias is not reported as a second, false error.
_COLLECT_RETRY_DROP = {
    "bandwidth_mhz": ("bandwidth_mhz", "bandwidth_hz"),
    "bandwidth_hz": ("bandwidth_mhz", "bandwidth_hz"),
}


def parse_rf_params(
    params: dict,
    *,
    path_default_min_elevation_deg: float = 5.0,
    collect_errors: bool = False,
) -> dict:
    """Parse RF parameters and optionally collect key-specific parser errors.

    Normal callers keep the existing exception-based API.  Collected callers
    receive a complete, safe RF dictionary plus ``parser_errors`` so that a
    route can report every invalid key in one typed validation envelope.
    """
    if not collect_errors:
        return _parse_rf_params(
            params, path_default_min_elevation_deg=path_default_min_elevation_deg
        )

    working = dict(params)
    parser_errors: list[str] = []
    for _ in range(32):
        try:
            parsed = _parse_rf_params(
                working,
                path_default_min_elevation_deg=path_default_min_elevation_deg,
                attach_earth_fixed_errors=True,
            )
        except RfKeyError as exc:
            parser_errors.extend(
                code for code in exc.codes if code not in parser_errors
            )
            # Retry only after dropping the key-specific bad value.  Never drop
            # Earth-Fixed inputs: their errors are attached, not raised (R8-S1).
            for key in _COLLECT_RETRY_DROP.get(exc.key, (exc.key,)):
                working.pop(key, None)
            if exc.key in _PFD_KEYS:
                # Parse PFD once: a second pass without the bad key would add a
                # false "requires both limits" error.
                working["compute_pfd"] = False
            continue
        except RfParamParseError as exc:
            # A non key-specific parse error cannot be retried meaningfully.
            parser_errors.extend(
                code for code in exc.codes if code not in parser_errors
            )
            break
        except RfConfigError:
            # RF validation exposed through request parsing must always name a
            # field.  Do not turn a future non-key-specific programming error
            # into a generic, un-actionable route response.
            raise
        else:
            earth_fixed_errors = parsed.pop("_earth_fixed_errors", [])
            all_errors = list(dict.fromkeys(parser_errors + earth_fixed_errors))
            # Keep the Earth-Fixed parser's established public key while also
            # exposing the clearer name used by the Automatic MCS route.
            parsed["rf_param_errors"] = all_errors
            parsed["parser_errors"] = all_errors
            return parsed

    if not parser_errors:
        raise RuntimeError(
            "RF parser collection exhausted without a key-specific error"
        )

    # A collected request error still returns a normalized RF dictionary, but
    # only together with the concrete key-specific codes that caused it.
    fallback = _parse_rf_params(
        {"rf_schema_version": 2, "clutter_mode": "disabled"},
        path_default_min_elevation_deg=path_default_min_elevation_deg,
    )
    fallback["rf_param_errors"] = parser_errors
    fallback["parser_errors"] = parser_errors
    return fallback
