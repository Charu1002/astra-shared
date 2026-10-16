from astra_shared.coverage_identity import (
    build_fixed_manifest,
    build_walker_manifest,
    canonical_coverage_cell_key,
    canonical_coverage_sat_id,
)


def test_coverage_ids_and_manifests_are_deterministic():
    assert canonical_coverage_sat_id({"id": "  SAT-1  "}, source="static") == "SAT-1"
    assert canonical_coverage_cell_key({"row": 2, "col": 4}) == (2, 4)
    assert build_fixed_manifest(1, 2, 550)[0]["id"] == "fixed-0"
    first = build_walker_manifest(2, 3, 550, 53, 30, 0, "delta")
    second = build_walker_manifest(2, 3, 550, 53, 30, 0, "delta")
    assert first == second
    assert len({row["id"] for row in first}) == 6


def test_rf_identity_tracks_versioned_clutter_configuration():
    from astra_shared.coverage_identity import build_earth_fixed_rf_identity_v1

    off = build_earth_fixed_rf_identity_v1({"clutter_mode": "disabled", "clutter_percentile": 50.0})
    p50 = build_earth_fixed_rf_identity_v1({"clutter_mode": "worldcover_p2108_p833", "clutter_percentile": 50.0})
    p90 = build_earth_fixed_rf_identity_v1({"clutter_mode": "worldcover_p2108_p833", "clutter_percentile": 90.0})
    assert off != p50 != p90
    assert off["clutter_percentile"] is None
    assert "clutter_enable" not in p50


def test_rf_identity_treats_null_min_elevation_deg_as_not_supplied():
    from astra_shared.coverage_identity import build_earth_fixed_rf_identity_v1

    legacy = build_earth_fixed_rf_identity_v1({"min_el_deg": 12.0})
    null_canonical = build_earth_fixed_rf_identity_v1({"min_elevation_deg": None, "min_el_deg": 12.0})
    canonical = build_earth_fixed_rf_identity_v1({"min_elevation_deg": 12.0})
    assert legacy["min_elevation_deg"] == null_canonical["min_elevation_deg"] == canonical["min_elevation_deg"] == 12.0
    both = build_earth_fixed_rf_identity_v1({"min_elevation_deg": 20.0, "min_el_deg": 12.0})
    assert both["min_elevation_deg"] == 20.0
