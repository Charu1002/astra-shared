"""store_run writes a Point run's initial projection rows in the record's own
transaction: both exist together, or neither does."""

import pytest

from astra_shared import live_source_store


def _row(key, sat_id="SAT-1"):
    return {"key": key, "time_iso": "2026-01-01T00:00:00Z", "columns": ["sat_id"], "row": [sat_id]}


def test_store_run_inserts_initial_projection_rows_with_the_record(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"}, run_id="run-1", projection_rows=[_row("k1")])

    assert run_id == "run-1"
    assert live_source_store.retrieve_run("run-1")["kind"] == "test"
    assert [row["key"] for row in live_source_store.retrieve_point_projection_rows("run-1")] == ["k1"]


def test_store_run_with_duplicate_projection_keys_stores_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        live_source_store.store_run({"kind": "test"}, run_id="run-2", projection_rows=[_row("k1"), _row("k1", "SAT-2")])

    assert live_source_store.retrieve_run("run-2") is None
    assert live_source_store.retrieve_point_projection_rows("run-2") == []


def test_store_run_rolls_back_the_record_when_a_projection_insert_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    # A row without a JSON-serialisable body fails inside the transaction.
    with pytest.raises(TypeError, match="not JSON serializable"):
        live_source_store.store_run({"kind": "test"}, run_id="run-3", projection_rows=[{"key": "k1", "row": object()}])

    assert live_source_store.retrieve_run("run-3") is None
    assert live_source_store.retrieve_point_projection_rows("run-3") == []
