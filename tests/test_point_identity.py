import pytest

from astra_shared.point_identity import canonical_point_sat_id, prepare_point_satellite_ids


def test_point_identity_is_canonical_and_unique():
    assert canonical_point_sat_id({"canonical_sat_id": " sat-1 "}, source="tle", order_index=0) == "sat-1"
    rows = prepare_point_satellite_ids([{"name": "A"}, {"name": "B"}], source="static")
    assert [row["id"] for row in rows] == ["A", "B"]
    assert all("canonical_sat_id" not in row for row in rows)


@pytest.mark.parametrize("bad", [None, "SAT-1", 7, 1.5, True, ["SAT-1"]])
def test_a_non_object_satellite_is_a_value_error(bad):
    # R5-04 defense in depth: callers catch ValueError, never AttributeError.
    with pytest.raises(ValueError, match="point_satellite_not_object"):
        prepare_point_satellite_ids([{"name": "A"}, bad], source="rf_log")
    with pytest.raises(ValueError, match="point_satellite_not_object"):
        canonical_point_sat_id(bad, source="rf_log", order_index=0)
