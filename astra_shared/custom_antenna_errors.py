"""Structured error taxonomy for custom antenna ingestion (Step 9.1).

This module centralizes deterministic error codes and response-shaping helpers
for CSV/FFD parsing and upload API routes.
"""

from __future__ import annotations

from typing import Any


class CustomAntennaErrorCode:
    # Transport/upload layer
    MISSING_FILE = "ca_upload_missing_file"
    INVALID_EXTENSION = "ca_upload_invalid_extension"
    READ_FAILED = "ca_upload_read_failed"
    EMPTY_FILE = "ca_upload_empty_file"
    FILE_TOO_LARGE = "ca_upload_file_too_large"
    INVALID_ENCODING = "ca_upload_invalid_encoding"
    VIEWER_INVALID_PAYLOAD = "ca_viewer_invalid_payload"

    # CSV parser/validation
    CSV_EMPTY = "ca_parse_csv_empty"
    CSV_HEADER_NOT_FOUND = "ca_parse_csv_header_not_found"
    CSV_MISSING_REQUIRED_COLUMNS = "ca_validate_csv_missing_required_columns"
    CSV_NON_NUMERIC_VALUE = "ca_parse_csv_non_numeric_value"
    CSV_UNEXPECTED_COLUMN_COUNT = "ca_parse_csv_unexpected_column_count"
    CSV_NON_FINITE_VALUE = "ca_validate_csv_non_finite_value"
    CSV_PSI_OUT_OF_RANGE = "ca_validate_csv_psi_out_of_range"
    CSV_PHI_OUT_OF_RANGE = "ca_validate_csv_phi_out_of_range"
    CSV_NO_DATA_ROWS = "ca_validate_csv_no_data_rows"
    CSV_INSUFFICIENT_PSI_POINTS = "ca_validate_csv_insufficient_psi_points"
    CSV_INSUFFICIENT_PHI_POINTS = "ca_validate_csv_insufficient_phi_points"
    CSV_DUPLICATE_GRID_POINT = "ca_validate_csv_duplicate_grid_point"
    CSV_INCOMPLETE_TABLE = "ca_validate_csv_incomplete_table"
    # A CSV may give phi as -180..180 or as 0..360, not both (NetSim's rule).
    CSV_MIXED_PHI_CONVENTION = "ca_validate_mixed_phi_convention"

    # Shared CSV/FFD validation (NetSim's code names)
    # A psi = 0 or psi = 180 row is one direction; its gains may spread by at
    # most POLE_ROW_MAX_SPREAD_DB before the file is rejected.
    POLE_ROW_INCONSISTENT = "ca_validate_pole_row_inconsistent"

    # FFD parser/validation/policy
    FFD_INVALID_INPUT_TYPE = "ca_parse_ffd_invalid_input_type"
    FFD_EMPTY_FILE = "ca_parse_ffd_empty_file"
    FFD_INVALID_FREQUENCY_SELECTION = "ca_policy_ffd_invalid_frequency_selection"
    FFD_FREQUENCY_MISMATCH = "ca_policy_ffd_frequency_mismatch"
    FFD_MISSING_FREQUENCY_SELECTION = "ca_policy_ffd_missing_frequency_selection"
    FFD_FORMAT_NOT_RECOGNIZED = "ca_parse_ffd_format_not_recognized"
    FFD_UNEXPECTED_COLUMN_COUNT = "ca_parse_ffd_unexpected_column_count"
    FFD_NON_NUMERIC_FIELD = "ca_parse_ffd_non_numeric_field"
    FFD_NON_FINITE_FIELD = "ca_validate_ffd_non_finite_field"
    FFD_THETA_OUT_OF_RANGE = "ca_validate_ffd_theta_out_of_range"
    FFD_PHI_OUT_OF_RANGE = "ca_validate_ffd_phi_out_of_range"
    FFD_DUPLICATE_GRID_POINT = "ca_validate_ffd_duplicate_grid_point"
    FFD_NON_POSITIVE_POWER = "ca_validate_ffd_non_positive_power"
    FFD_NO_DATA_ROWS = "ca_validate_ffd_no_data_rows"
    FFD_INCOMPLETE_GRID = "ca_validate_ffd_incomplete_grid"


def build_error_payload(
    *,
    code: str,
    message: str,
    detail: str | None = None,
    line_no: int | None = None,
    filename: str | None = None,
    source_format: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if detail is not None:
        payload["error"]["detail"] = detail
    if line_no is not None:
        payload["error"]["line_no"] = line_no
    if filename is not None:
        payload["error"]["filename"] = filename
    if source_format is not None:
        payload["error"]["source_format"] = source_format
    return payload
