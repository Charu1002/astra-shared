"""A lost Point CAS is "already_committed" only for an exact completion of the
commit that advanced the version: same four-part keys AND same row content."""

from astra_shared import live_source_store


def _row(rx):
    return {"key": "run:2026-01-01T00:00:00.000Z:SAT-1:hash", "time_iso": "2026-01-01T00:00:00Z", "columns": ["sat_id", "rx"], "row": ["SAT-1", rx]}


def test_stale_cas_with_the_same_keys_but_different_content_is_a_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[_row(-80.0)]) == "committed"

    # A second writer at the same stale version with the same key but different content.
    status = live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[_row(-90.0)])

    assert status == "conflict"
    assert [row["row"] for row in live_source_store.retrieve_point_projection_rows(run_id)] == [["SAT-1", -80.0]]


def test_stale_cas_that_exactly_repeats_the_winning_commit_is_already_committed(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[_row(-80.0)]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[_row(-80.0)]) == "already_committed"


def _sample(time_iso, rx):
    return {
        "key": f"run:{time_iso}:SAT-1:hash",
        "time_iso": time_iso,
        "columns": ["sat_id", "rx"],
        "row": ["SAT-1", rx],
    }


def test_a_delayed_retry_of_an_older_sample_is_a_conflict_after_newer_samples(tmp_path, monkeypatch):
    # R2-07: matching ANY stored sample is not enough; only the run's last
    # commit can be completed idempotently.
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    first = _sample("2026-01-01T00:00:00.000Z", -80.0)
    second = _sample("2026-01-01T00:00:05.000Z", -81.0)
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[first]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=1, projection_rows=[second]) == "committed"

    # The first sample's attempt arrives late, from the snapshot it was computed on.
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[first]) == "conflict"
    # The last commit can still be completed idempotently.
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=1, projection_rows=[second]) == "already_committed"
