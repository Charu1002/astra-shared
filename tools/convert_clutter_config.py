#!/usr/bin/env python3
"""One-time converter for pre-versioned Astra clutter RF settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import astra_shared
from astra_shared.clutter_config import ClutterConfigError, normalize_clutter_rf

RF_SCHEMA_VERSION = 1
CLUTTER_MODE_DISABLED = "disabled"
CLUTTER_MODE_WORLDCOVER = "worldcover_p2108_p833"
CLUTTER_MODES = {CLUTTER_MODE_DISABLED, CLUTTER_MODE_WORLDCOVER}
DEFAULT_CLUTTER_PERCENTILE = 50.0
OLD_CLUTTER_FIELDS = {
    "clutter_enable",
    "clutter_enabled",
    "clutter_values",
    "clutter_fallback",
}
RF_BLOCK_KEYS = {"rf", "rf_params", "rf_settings"}


class ConverterError(ValueError):
    """Raised when a clutter RF block cannot be converted safely."""


def _old_clutter_enabled(block: dict[str, Any]) -> bool:
    if "clutter_mode" in block:
        value = block["clutter_mode"]
    elif "clutter_enable" in block:
        value = block["clutter_enable"]
    elif "clutter_enabled" in block:
        value = block["clutter_enabled"]
    else:
        return False
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == CLUTTER_MODE_WORLDCOVER:
        return True
    if normalized in {"true", "1", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"false", "0", "no", "off", "disable", "disabled"}:
        return False
    raise ConverterError(f"unsupported old clutter boolean {value!r}")


def _has_current_version(block: dict[str, Any]) -> bool:
    return type(block.get("rf_schema_version")) in (int, str) and block.get("rf_schema_version") in (
        RF_SCHEMA_VERSION,
        "1",
    )


def _has_version_two(block: dict[str, Any]) -> bool:
    return type(block.get("rf_schema_version")) in (int, str) and block.get(
        "rf_schema_version"
    ) in (2, "2")


def _is_clutter_block(block: dict[str, Any], is_rf_block: bool) -> bool:
    return (
        is_rf_block
        or "clutter_mode" in block
        or bool(OLD_CLUTTER_FIELDS.intersection(block))
    )


def _is_clean_current_block(block: dict[str, Any]) -> bool:
    return (
        _has_current_version(block)
        and not OLD_CLUTTER_FIELDS.intersection(block)
        and block.get("clutter_mode") in CLUTTER_MODES
    )


def _validate_converted_block(block: dict[str, Any]) -> None:
    try:
        normalize_clutter_rf(block)
    except ClutterConfigError as exc:
        value = block.get("clutter_percentile")
        raise ConverterError(f"clutter_percentile {value!r} is invalid: {exc}") from exc


def _convert_rf_block(block: dict[str, Any], is_rf_block: bool) -> tuple[bool, bool]:
    if "rf_schema_version" in block:
        if _has_version_two(block):
            return False, False
        if not _has_current_version(block):
            raise ConverterError(
                f"unsupported rf_schema_version {block.get('rf_schema_version')!r}"
            )
    if not _is_clutter_block(block, is_rf_block):
        return False, False
    if _is_clean_current_block(block):
        _validate_converted_block(block)
        return False, False

    if "clutter_mode" in block:
        for field in ("clutter_enable", "clutter_enabled"):
            if field in block and _old_clutter_enabled({field: block[field]}) != _old_clutter_enabled(
                {"clutter_mode": block["clutter_mode"]}
            ):
                print(f"clutter_mode={block['clutter_mode']!r} overrides {field}={block[field]!r}")
    original_mode = block.get("clutter_mode")
    changed = not _has_current_version(block)
    enabled = _old_clutter_enabled(block)
    for key in OLD_CLUTTER_FIELDS:
        if key in block:
            changed = True
            block.pop(key, None)
    if block.get("rf_schema_version") != RF_SCHEMA_VERSION:
        changed = True
    block["rf_schema_version"] = RF_SCHEMA_VERSION
    mode = CLUTTER_MODE_WORLDCOVER if enabled else CLUTTER_MODE_DISABLED
    if block.get("clutter_mode") != mode:
        changed = True
    block["clutter_mode"] = mode
    if block.get("clutter_percentile") is None:
        changed = True
        block["clutter_percentile"] = DEFAULT_CLUTTER_PERCENTILE
    _validate_converted_block(block)
    moved_to_worldcover = changed and enabled and original_mode != CLUTTER_MODE_WORLDCOVER
    return changed, moved_to_worldcover


def convert_payload(value: Any, parent_key: str | None = None) -> tuple[bool, bool]:
    """Convert every clutter RF block inside a decoded JSON payload.

    Returns ``(changed, moved_to_worldcover)``. A ``clutter_values`` block alone
    never enables clutter.
    """
    changed = False
    moved_to_worldcover = False
    if isinstance(value, dict):
        is_rf_block = parent_key in RF_BLOCK_KEYS
        block_changed, block_enabled = _convert_rf_block(value, is_rf_block)
        changed = changed or block_changed
        moved_to_worldcover = moved_to_worldcover or block_enabled
        for key, child in value.items():
            child_changed, child_enabled = convert_payload(child, key)
            changed = changed or child_changed
            moved_to_worldcover = moved_to_worldcover or child_enabled
    elif isinstance(value, list):
        for child in value:
            child_changed, child_enabled = convert_payload(child)
            changed = changed or child_changed
            moved_to_worldcover = moved_to_worldcover or child_enabled
    return changed, moved_to_worldcover


def convert_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed, moved_to_worldcover = convert_payload(payload)
    if not changed:
        return False
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if moved_to_worldcover:
        print(f"{path}: moved to {CLUTTER_MODE_WORLDCOVER}")
    return True


def iter_json_files(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        elif path.suffix.lower() == ".json":
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert old Astra clutter RF settings to schema version 1."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    print(f"using astra_shared from {Path(astra_shared.__file__).resolve()}")
    changed = 0
    examined = 0
    failures = 0
    valid_paths = []
    for path in args.paths:
        if path.is_dir() or (path.is_file() and path.suffix.lower() == ".json"):
            valid_paths.append(path)
        else:
            failures += 1
            print(f"{path}: expected an existing directory or a .json file")
    for path in iter_json_files(valid_paths):
        examined += 1
        try:
            if convert_file(path):
                changed += 1
        except Exception as exc:
            failures += 1
            print(f"{path}: conversion failed: {exc}")
    print(f"examined {examined} file(s)")
    print(f"converted {changed} file(s)")
    if failures:
        print(f"failed {failures} file(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
