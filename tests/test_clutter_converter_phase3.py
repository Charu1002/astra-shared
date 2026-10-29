#!/usr/bin/env python3

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from astra_shared.clutter_config import normalize_clutter_rf

_CONVERTER_PATH = Path(__file__).parents[1] / "tools" / "convert_clutter_config.py"
_SPEC = importlib.util.spec_from_file_location("convert_clutter_config", _CONVERTER_PATH)
converter = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(converter)


def test_converter_rewrites_enabled_old_fields_to_versioned_worldcover():
    payload = {
        "rf": {
            "clutter_enable": True,
            "clutter_values": {"50": 20.0},
            "clutter_fallback": 3.0,
        }
    }

    changed, moved = converter.convert_payload(payload)

    assert changed is True
    assert moved is True
    assert payload["rf"] == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 50.0,
    }


def test_converter_never_uses_table_presence_to_enable_clutter():
    payload = {"rf": {"clutter_values": {"50": 20.0}, "clutter_fallback": 3.0}}

    changed, moved = converter.convert_payload(payload)

    assert changed is True
    assert moved is False
    assert payload["rf"]["clutter_mode"] == "disabled"
    assert payload["rf"]["clutter_percentile"] == 50.0


def test_converter_versions_an_rf_block_without_clutter_fields():
    payload = {"rf": {"frequency_ghz": 12.0, "eirp_dbw": 70.0}}

    assert converter.convert_payload(payload) == (True, False)
    assert payload["rf"] == {
        "frequency_ghz": 12.0,
        "eirp_dbw": 70.0,
        "rf_schema_version": 1,
        "clutter_mode": "disabled",
        "clutter_percentile": 50.0,
    }


def test_converter_versions_new_mode_token_without_schema_version():
    payload = {"rf": {"clutter_mode": "worldcover_p2108_p833", "clutter_percentile": 80.0}}

    assert converter.convert_payload(payload) == (True, False)
    assert payload["rf"] == {
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 80.0,
        "rf_schema_version": 1,
    }


def test_converter_repairs_current_mode_mixed_with_retired_fields():
    payload = {
        "rf": {
            "clutter_mode": "disabled",
            "clutter_enable": True,
            "clutter_values": {"50": 8.0},
            "clutter_fallback": 2.0,
        }
    }

    assert converter.convert_payload(payload) == (True, False)

    assert payload["rf"] == {
        "clutter_mode": "disabled",
        "rf_schema_version": 1,
        "clutter_percentile": 50.0,
    }
    assert normalize_clutter_rf(payload["rf"]) == {
        "rf_schema_version": 1,
        "clutter_mode": "disabled",
        "clutter_percentile": None,
    }
    assert converter.convert_payload(payload) == (False, False)


def test_converter_repairs_versioned_blocks_with_retired_fields():
    payload = {
        "rf": {
            "rf_schema_version": 1,
            "clutter_mode": "worldcover_p2108_p833",
            "clutter_enable": False,
            "clutter_values": {"50": 12.0},
            "clutter_percentile": 75.0,
        }
    }

    assert converter.convert_payload(payload) == (True, False)

    assert payload["rf"] == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 75.0,
    }
    assert normalize_clutter_rf(payload["rf"])["clutter_percentile"] == 75.0


def test_converter_converts_new_token_block_outside_rf_parent():
    payload = {
        "saved_clutter": {
            "clutter_mode": "worldcover_p2108_p833",
            "clutter_percentile": 60.0,
        }
    }

    assert converter.convert_payload(payload) == (True, False)

    assert payload["saved_clutter"] == {
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 60.0,
        "rf_schema_version": 1,
    }
    assert normalize_clutter_rf(payload["saved_clutter"])["clutter_mode"] == "worldcover_p2108_p833"


def test_converter_keeps_existing_percentile_when_converting_old_enabled_fields():
    payload = {"rf": {"clutter_enable": True, "clutter_percentile": 80.0}}

    assert converter.convert_payload(payload) == (True, True)
    assert payload["rf"] == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 80.0,
    }


def test_converter_mode_precedes_conflicting_boolean(capsys):
    payload = {"rf": {"clutter_mode": "disable", "clutter_enable": True}}

    changed, moved = converter.convert_payload(payload)

    assert (changed, moved) == (True, False)
    assert payload["rf"]["clutter_mode"] == "disabled"
    assert "clutter_mode='disable' overrides clutter_enable=True" in capsys.readouterr().out


def test_converter_rewrites_nested_saved_project_file(tmp_path, capsys):
    path = tmp_path / "project.json"
    path.write_text(
        json.dumps(
            {
                "settings": {
                    "coverage": {
                        "rf": {
                            "clutter_mode": "enable",
                            "clutter_values": {"50": 20.0},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert converter.convert_file(path) is True

    converted = json.loads(path.read_text(encoding="utf-8"))
    rf = converted["settings"]["coverage"]["rf"]
    assert rf["rf_schema_version"] == 1
    assert rf["clutter_mode"] == "worldcover_p2108_p833"
    assert rf["clutter_percentile"] == 50.0
    assert "clutter_values" not in rf
    assert "moved to worldcover_p2108_p833" in capsys.readouterr().out


def test_converter_repairing_already_worldcover_block_does_not_report_move(tmp_path, capsys):
    path = tmp_path / "project.json"
    path.write_text(
        json.dumps(
            {
                "rf": {
                    "rf_schema_version": 1,
                    "clutter_mode": "worldcover_p2108_p833",
                    "clutter_values": {"50": 20.0},
                    "clutter_percentile": 80.0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert converter.convert_file(path) is True

    rf = json.loads(path.read_text(encoding="utf-8"))["rf"]
    assert rf == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 80.0,
    }
    assert "moved to worldcover_p2108_p833" not in capsys.readouterr().out


def test_converter_is_idempotent_for_versioned_payload():
    payload = {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 50.0,
    }

    assert converter.convert_payload(payload) == (False, False)
    assert payload == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 50.0,
    }


def test_converter_leaves_version_two_unchanged_and_rejects_unknown_versions():
    payload = {"rf": {"rf_schema_version": 2, "clutter_mode": "disabled"}}

    assert converter.convert_payload(payload) == (False, False)
    assert payload["rf"]["rf_schema_version"] == 2

    with pytest.raises(converter.ConverterError, match="unsupported rf_schema_version"):
        converter.convert_payload(
            {"rf": {"rf_schema_version": 3, "clutter_mode": "disabled"}}
        )


def test_converter_rejects_boolean_schema_version():
    payload = {"rf": {"rf_schema_version": True, "clutter_mode": "disabled"}}

    with pytest.raises(converter.ConverterError, match="unsupported rf_schema_version"):
        converter.convert_payload(payload)


def test_converter_rejects_unknown_old_boolean_string():
    payload = {"rf": {"clutter_enable": "maybe"}}

    with pytest.raises(converter.ConverterError, match="unsupported old clutter boolean"):
        converter.convert_payload(payload)


def test_converter_cli_reports_bad_files_and_keeps_going(tmp_path, capsys):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text('{"rf":{"clutter_enable":true}}', encoding="utf-8")
    bad.write_text('{"rf":{"clutter_enable":"maybe"}}', encoding="utf-8")

    assert converter.main([str(tmp_path)]) == 1

    assert json.loads(good.read_text(encoding="utf-8"))["rf"]["clutter_mode"] == "worldcover_p2108_p833"
    output = capsys.readouterr().out
    assert "bad.json: conversion failed: unsupported old clutter boolean 'maybe'" in output
    assert "converted 1 file(s)" in output
    assert "failed 1 file(s)" in output


@pytest.mark.parametrize(
    "percentile",
    ["abc", 150, 0, True, {}],
)
def test_converter_rejects_invalid_enabled_percentile_without_rewriting(percentile, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    good = tmp_path / "good.json"
    bad.write_text(
        json.dumps({"rf": {"clutter_mode": "enable", "clutter_percentile": percentile}}),
        encoding="utf-8",
    )
    original_bad = bad.read_text(encoding="utf-8")
    good.write_text('{"rf":{"clutter_enable":true}}', encoding="utf-8")

    assert converter.main([str(tmp_path)]) == 1

    assert bad.read_text(encoding="utf-8") == original_bad
    assert json.loads(good.read_text(encoding="utf-8"))["rf"] == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 50.0,
    }
    output = capsys.readouterr().out
    assert "bad.json: conversion failed:" in output
    assert repr(percentile) in output
    assert "converted 1 file(s)" in output
    assert "failed 1 file(s)" in output


@pytest.mark.parametrize("percentile", [0.001, 99.999, None])
def test_converter_accepts_enabled_percentile_endpoints_and_default(percentile, tmp_path):
    path = tmp_path / "project.json"
    rf = {"clutter_mode": "enable"}
    if percentile is not None:
        rf["clutter_percentile"] = percentile
    path.write_text(json.dumps({"rf": rf}), encoding="utf-8")

    assert converter.convert_file(path) is True

    converted = json.loads(path.read_text(encoding="utf-8"))["rf"]
    assert converted["clutter_mode"] == "worldcover_p2108_p833"
    assert converted["clutter_percentile"] == (50.0 if percentile is None else percentile)
    normalize_clutter_rf(converted)


def test_converter_keeps_disabled_invalid_percentile_when_converting(tmp_path):
    path = tmp_path / "project.json"
    path.write_text('{"rf":{"clutter_mode":"disable","clutter_percentile":"abc"}}', encoding="utf-8")

    assert converter.convert_file(path) is True

    converted = json.loads(path.read_text(encoding="utf-8"))["rf"]
    assert converted == {
        "clutter_mode": "disabled",
        "clutter_percentile": "abc",
        "rf_schema_version": 1,
    }
    assert normalize_clutter_rf(converted)["clutter_percentile"] is None


@pytest.mark.parametrize(
    "rf",
    [
        {"rf_schema_version": 1, "clutter_mode": "worldcover_p2108_p833"},
        {"rf_schema_version": 1, "clutter_mode": "worldcover_p2108_p833", "clutter_percentile": None},
        {"rf_schema_version": 1, "clutter_mode": "disabled"},
        {"rf_schema_version": "1", "clutter_mode": "disabled", "metadata": "Bengaluru Δ"},
    ],
)
def test_converter_leaves_clean_version_one_files_byte_identical(rf, tmp_path, capsys):
    path = tmp_path / "project.json"
    original = json.dumps({"rf": rf, "name": "Project Δ"}, ensure_ascii=False, separators=(",", ":"))
    path.write_text(original, encoding="utf-8")

    assert converter.convert_file(path) is False

    assert path.read_text(encoding="utf-8") == original
    assert capsys.readouterr().out == ""


def test_converter_script_runs_from_outside_repo_with_repo_local_import(tmp_path):
    path = tmp_path / "project.json"
    path.write_text('{"rf":{"clutter_enable":true}}', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, str(_CONVERTER_PATH), str(path)],
        cwd=outside,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "using astra_shared from" in result.stdout
    assert str(Path(__file__).parents[1] / "astra_shared") in result.stdout
    assert json.loads(path.read_text(encoding="utf-8"))["rf"] == {
        "rf_schema_version": 1,
        "clutter_mode": "worldcover_p2108_p833",
        "clutter_percentile": 50.0,
    }


def test_converter_prefers_repo_package_over_decoy_on_pythonpath(tmp_path):
    decoy_root = tmp_path / "decoy"
    decoy_package = decoy_root / "astra_shared"
    decoy_package.mkdir(parents=True)
    (decoy_package / "__init__.py").write_text("DECOY = True\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    path = tmp_path / "invalid.json"
    original = '{"rf":{"rf_schema_version":1,"clutter_mode":"worldcover_p2108_p833","clutter_percentile":150}}'
    path.write_text(original, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join((str(decoy_root), str(Path(__file__).parents[1])))

    result = subprocess.run(
        [sys.executable, str(_CONVERTER_PATH), str(path)],
        cwd=outside,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert f"using astra_shared from {Path(__file__).parents[1] / 'astra_shared'}" in result.stdout
    assert "clutter_percentile 150 is invalid" in result.stdout
    assert path.read_text(encoding="utf-8") == original


def test_converter_reports_invalid_clean_version_one_enabled_file_without_rewriting(tmp_path, capsys):
    path = tmp_path / "project.json"
    original = '{"rf":{"rf_schema_version":1,"clutter_mode":"worldcover_p2108_p833","clutter_percentile":150}}'
    path.write_text(original, encoding="utf-8")

    assert converter.main([str(path)]) == 1

    assert path.read_text(encoding="utf-8") == original
    output = capsys.readouterr().out
    assert "clutter_percentile 150 is invalid" in output
    assert "converted clutter_percentile" not in output


def test_converter_cli_catches_non_converter_exceptions_per_file(tmp_path, monkeypatch, capsys):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text('{"rf":{"clutter_enable":true}}', encoding="utf-8")
    bad.write_text('{"rf":{"clutter_enable":false}}', encoding="utf-8")

    original = converter.convert_file

    def convert_or_type_error(path):
        if path == bad:
            raise TypeError("object-mode membership failed")
        return original(path)

    monkeypatch.setattr(converter, "convert_file", convert_or_type_error)

    assert converter.main([str(tmp_path)]) == 1

    assert json.loads(good.read_text(encoding="utf-8"))["rf"]["clutter_mode"] == "worldcover_p2108_p833"
    output = capsys.readouterr().out
    assert "bad.json: conversion failed: object-mode membership failed" in output
    assert "converted 1 file(s)" in output
    assert "failed 1 file(s)" in output


@pytest.mark.parametrize("bad_path", ["missing-directory", "missing.json", "notes.txt"])
def test_converter_cli_rejects_paths_that_do_not_match_json_inputs(tmp_path, bad_path, capsys):
    path = tmp_path / bad_path
    if bad_path.endswith(".txt"):
        path.write_text("not a config", encoding="utf-8")

    assert converter.main([str(path)]) == 1

    output = capsys.readouterr().out
    assert "expected an existing directory or a .json file" in output
    assert "examined 0 file(s)" in output
    assert "converted 0 file(s)" in output


def test_converter_cli_converts_valid_file_and_fails_for_invalid_argument(tmp_path, capsys):
    valid = tmp_path / "legacy.json"
    valid.write_text('{"rf":{"clutter_enable":true}}', encoding="utf-8")
    invalid = tmp_path / "missing.json"

    assert converter.main([str(valid), str(invalid)]) == 1

    assert json.loads(valid.read_text(encoding="utf-8"))["rf"]["clutter_mode"] == "worldcover_p2108_p833"
    output = capsys.readouterr().out
    assert "missing.json: expected an existing directory or a .json file" in output
    assert "examined 1 file(s)" in output
    assert "converted 1 file(s)" in output
    assert "failed 1 file(s)" in output


def test_converter_preserves_order_and_unicode(tmp_path):
    path = tmp_path / "project.json"
    path.write_text(
        '{"name":"Bengaluru Δ","rf":{"frequency_ghz":12.0,"clutter_enable":"on"}}',
        encoding="utf-8",
    )

    assert converter.convert_file(path) is True

    text = path.read_text(encoding="utf-8")
    assert "Bengaluru Δ" in text
    assert text.index('"frequency_ghz"') < text.index('"rf_schema_version"')


def test_converter_does_not_version_marker_only_non_rf_blocks():
    payload = {"display": {"frequency_ghz": 12.0, "label": "metadata only"}}

    assert converter.convert_payload(payload) == (False, False)
    assert payload == {"display": {"frequency_ghz": 12.0, "label": "metadata only"}}


@pytest.mark.parametrize(
    ("old", "expected_mode"),
    [
        ({"clutter_mode": "disable", "clutter_values": {"50": 8.0}}, "disabled"),
        ({"clutter_mode": "enable", "clutter_values": {"50": 8.0}}, "worldcover_p2108_p833"),
        ({"clutter_enable": "yes", "clutter_values": {"50": 8.0}}, "worldcover_p2108_p833"),
        ({"clutter_enabled": "on", "clutter_values": {"50": 8.0}}, "worldcover_p2108_p833"),
        ({"clutter_enable": False, "clutter_values": {"50": 8.0}}, "disabled"),
        ({"clutter_enabled": True, "clutter_values": {"50": 8.0}}, "worldcover_p2108_p833"),
        ({"clutter_values": {"50": 8.0}}, "disabled"),
    ],
)
def test_converter_old_state_rules(old, expected_mode):
    payload = {"rf": dict(old)}

    assert converter.convert_payload(payload)[0] is True
    assert payload["rf"] == {
        "rf_schema_version": 1,
        "clutter_mode": expected_mode,
        "clutter_percentile": 50.0,
    }
