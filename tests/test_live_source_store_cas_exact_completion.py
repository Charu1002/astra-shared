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


def test_an_empty_batch_retry_is_identified_only_at_the_same_next_version(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[]) == "already_committed"
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=1, projection_rows=[]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, {"kind": "test"}, expected_version=0, projection_rows=[]) == "conflict"


# --- R7-03: the marker identifies the committed state, not only its rows -----


def _run(instant_ms):
    return {"kind": "test", "point_analysis": {"last_instant_ms": instant_ms}}


def test_two_different_empty_samples_from_one_version_are_not_one_commit(tmp_path, monkeypatch):
    # Every empty batch used to hash the literal []: a second, different empty
    # sample computed on the same snapshot read as an already-committed retry.
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    assert live_source_store.commit_point_run_cas(run_id, _run(1000), expected_version=0, projection_rows=[]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, _run(2000), expected_version=0, projection_rows=[]) == "conflict"
    assert live_source_store.commit_point_run_cas(run_id, _run(1000), expected_version=0, projection_rows=[]) == "already_committed"
    assert live_source_store.retrieve_run(run_id)["point_analysis"]["last_instant_ms"] == 1000


def test_the_same_rows_with_a_different_snapshot_are_a_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    row = _row(-80.0)
    assert live_source_store.commit_point_run_cas(run_id, _run(1000), expected_version=0, projection_rows=[row]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, _run(1001), expected_version=0, projection_rows=[row]) == "conflict"


def test_the_same_rows_in_another_order_are_the_same_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    rows = [
        {"key": f"run:2026-01-01T00:00:00.000Z:{sat}:hash", "time_iso": "2026-01-01T00:00:00Z", "columns": ["sat_id"], "row": [sat]}
        for sat in ("SAT-1", "SAT-2", "SAT-3")
    ]
    assert live_source_store.commit_point_run_cas(run_id, _run(1), expected_version=0, projection_rows=rows) == "committed"
    reordered = [rows[2], rows[0], rows[1]]
    assert live_source_store.commit_point_run_cas(run_id, _run(1), expected_version=0, projection_rows=reordered) == "already_committed"
    assert len(live_source_store.retrieve_point_projection_rows(run_id)) == 3


def test_a_marker_read_back_from_the_store_does_not_change_the_commit(tmp_path, monkeypatch):
    # A caller that passes the run as it read it back (with the previous
    # commit's marker) commits the same state as one that does not.
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    assert live_source_store.commit_point_run_cas(run_id, _run(1), expected_version=0, projection_rows=[]) == "committed"
    stored = live_source_store.retrieve_run(run_id)
    stored["point_analysis"] = {"last_instant_ms": 2}
    assert live_source_store.commit_point_run_cas(run_id, stored, expected_version=1, projection_rows=[]) == "committed"
    assert live_source_store.commit_point_run_cas(run_id, _run(2), expected_version=1, projection_rows=[]) == "already_committed"


class _RacingConnection:
    """A connection on which the identical commit lands from another
    connection between this one's SELECT and its UPDATE."""

    def __init__(self, conn, race):
        self._conn = conn
        self._race = race

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)

    def execute(self, sql, *args):
        if sql.lstrip().startswith("UPDATE live_runs") and self._race:
            self._race.pop()()
        return self._conn.execute(sql, *args)


def test_a_retry_that_loses_the_race_to_its_own_commit_is_already_committed(tmp_path, monkeypatch):
    # The completion check used to read the row selected BEFORE the UPDATE: a
    # genuine retry whose UPDATE found the version already advanced by the same
    # commit was reported as a conflict.
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    real_connect = live_source_store._connect
    rows = [_row(-80.0)]

    def commit_elsewhere():
        monkeypatch.setattr(live_source_store, "_connect", real_connect)
        assert live_source_store.commit_point_run_cas(run_id, _run(1), expected_version=0, projection_rows=rows) == "committed"

    race = [commit_elsewhere]
    monkeypatch.setattr(live_source_store, "_connect", lambda: _RacingConnection(real_connect(), race))
    assert live_source_store.commit_point_run_cas(run_id, _run(1), expected_version=0, projection_rows=rows) == "already_committed"
    assert not race  # the race did happen
    assert live_source_store.retrieve_run(run_id)["_state_version"] == 1
    assert len(live_source_store.retrieve_point_projection_rows(run_id)) == 1


def test_a_different_commit_that_wins_the_race_is_a_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    run_id = live_source_store.store_run({"kind": "test"})
    real_connect = live_source_store._connect

    def commit_elsewhere():
        monkeypatch.setattr(live_source_store, "_connect", real_connect)
        assert live_source_store.commit_point_run_cas(run_id, _run(2), expected_version=0, projection_rows=[]) == "committed"

    race = [commit_elsewhere]
    monkeypatch.setattr(live_source_store, "_connect", lambda: _RacingConnection(real_connect(), race))
    assert live_source_store.commit_point_run_cas(run_id, _run(1), expected_version=0, projection_rows=[]) == "conflict"
    assert live_source_store.retrieve_run(run_id)["point_analysis"]["last_instant_ms"] == 2
