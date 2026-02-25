"""
Tests for galaxy catalog data integrity and SPARC quality validation.

Validates that all galaxy entries have required fields, sensible values,
mass models are consistent with stated total masses, and the quality
validator correctly classifies good vs. bad galaxy data.
"""

import copy
import math
import pytest
from data.galaxies import (
    PREDICTION_GALAXIES,
    SIMPLE_PREDICTION_GALAXIES,
    INFERENCE_GALAXIES,
    get_prediction_galaxies,
    get_inference_galaxies,
    get_galaxy_by_id,
    get_all_galaxies,
    get_galaxy_catalog,
    _invalidate_cache,
)
from physics.mass_model import total_mass
from physics.services.sparc.sparc_parser import validate_galaxy_quality


class TestCatalogStructure:
    """Test catalog completeness and field presence."""

    def test_hardcoded_prediction_galaxies_not_empty(self):
        assert len(PREDICTION_GALAXIES) >= 6

    def test_sparc_prediction_galaxies_not_empty(self):
        galaxies = get_prediction_galaxies()
        assert len(galaxies) >= 170

    def test_inference_galaxies_not_empty(self):
        assert len(INFERENCE_GALAXIES) >= 4

    def test_all_hardcoded_prediction_have_required_fields(self):
        for g in PREDICTION_GALAXIES:
            assert "id" in g, f"Missing id in {g.get('name', '?')}"
            assert "name" in g
            assert "distance" in g
            assert "mass" in g
            assert "accel" in g
            assert "mass_model" in g
            assert "observations" in g

    def test_all_sparc_prediction_have_required_fields(self):
        for g in get_prediction_galaxies():
            assert "id" in g, f"Missing id in {g.get('name', '?')}"
            assert "name" in g
            assert "distance" in g
            assert "mass" in g
            assert "accel" in g
            assert "mass_model" in g
            assert "observations" in g

    def test_all_inference_have_required_fields(self):
        for g in INFERENCE_GALAXIES:
            assert "id" in g
            assert "name" in g
            assert "distance" in g
            assert "velocity" in g
            assert "accel" in g
            assert "mass_model" in g, f"{g['id']} missing mass_model shape"

    def test_unique_hardcoded_ids(self):
        """Hardcoded galaxy IDs must be unique within their own lists."""
        all_galaxies = (
            PREDICTION_GALAXIES
            + SIMPLE_PREDICTION_GALAXIES
            + INFERENCE_GALAXIES
        )
        ids = [g["id"] for g in all_galaxies]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_unique_sparc_ids(self):
        """All sparc galaxy IDs must be unique."""
        galaxies = get_prediction_galaxies()
        ids = [g["id"] for g in galaxies]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in sparc: {[x for x in ids if ids.count(x) > 1]}"


class TestMassModelConsistency:
    """Verify mass models are consistent with stated log10 masses."""

    @pytest.mark.parametrize("galaxy", PREDICTION_GALAXIES, ids=lambda g: g["id"])
    def test_total_mass_matches_log10(self, galaxy):
        mm = galaxy["mass_model"]
        m_total = total_mass(mm)
        log10_actual = math.log10(m_total)
        log10_stated = galaxy["mass"]
        # Should agree within 0.1 dex (factor of ~1.26)
        assert abs(log10_actual - log10_stated) < 0.15, \
            f"{galaxy['id']}: log10(total)={log10_actual:.3f} vs stated={log10_stated}"


class TestObservationalData:
    """Validate observational data arrays."""

    @pytest.mark.parametrize("galaxy", PREDICTION_GALAXIES, ids=lambda g: g["id"])
    def test_observations_have_fields(self, galaxy):
        for obs in galaxy["observations"]:
            assert "r" in obs
            assert "v" in obs
            assert "err" in obs

    @pytest.mark.parametrize("galaxy", PREDICTION_GALAXIES, ids=lambda g: g["id"])
    def test_radii_positive_and_sorted(self, galaxy):
        radii = [obs["r"] for obs in galaxy["observations"]]
        assert all(r > 0 for r in radii)
        assert radii == sorted(radii), f"{galaxy['id']}: radii not sorted"

    @pytest.mark.parametrize("galaxy", PREDICTION_GALAXIES, ids=lambda g: g["id"])
    def test_velocities_positive(self, galaxy):
        for obs in galaxy["observations"]:
            assert obs["v"] > 0

    @pytest.mark.parametrize("galaxy", PREDICTION_GALAXIES, ids=lambda g: g["id"])
    def test_errors_non_negative(self, galaxy):
        for obs in galaxy["observations"]:
            assert obs["err"] >= 0


class TestLookupFunctions:
    """Test galaxy lookup helpers."""

    def test_get_by_id_sparc_galaxy(self):
        g = get_galaxy_by_id("ic2574")
        assert g is not None
        assert g["id"] == "ic2574"

    def test_get_by_id_hardcoded_galaxy(self):
        """Hardcoded galaxies should still be findable via get_galaxy_by_id."""
        g = get_galaxy_by_id("milky_way")
        assert g is not None
        assert g["id"] == "milky_way"

    def test_get_by_id_not_found(self):
        g = get_galaxy_by_id("nonexistent_galaxy_xyz")
        assert g is None

    def test_get_all_galaxies_structure(self):
        result = get_all_galaxies()
        assert "prediction" in result
        assert "inference" in result
        assert len(result["prediction"]) > 0
        assert len(result["inference"]) > 0

    def test_get_prediction_returns_sparc_galaxies(self):
        """get_prediction_galaxies returns galaxies from the sparc/ folder."""
        galaxies = get_prediction_galaxies()
        ids = [g["id"] for g in galaxies]
        assert "ic2574" in ids
        assert "ngc3198" in ids
        assert "ddo154" in ids


class TestGalaxyCatalog:
    """Test the lightweight galaxy catalog (sparc/catalog.json)."""

    def test_catalog_not_empty(self):
        catalog = get_galaxy_catalog()
        assert len(catalog) >= 170

    def test_catalog_entries_have_id_and_name(self):
        for entry in get_galaxy_catalog():
            assert "id" in entry
            assert "name" in entry
            assert isinstance(entry["id"], str)
            assert isinstance(entry["name"], str)
            assert len(entry["id"]) > 0
            assert len(entry["name"]) > 0

    def test_catalog_ids_match_prediction_galaxies(self):
        """Every galaxy in the catalog should be loadable."""
        catalog_ids = {e["id"] for e in get_galaxy_catalog()}
        prediction_ids = {g["id"] for g in get_prediction_galaxies()}
        missing = catalog_ids - prediction_ids
        assert len(missing) == 0, f"Catalog has IDs not in prediction: {missing}"

    def test_catalog_sorted_by_id(self):
        catalog = get_galaxy_catalog()
        ids = [e["id"] for e in catalog]
        assert ids == sorted(ids, key=str.lower)


# ---------------------------------------------------------------------------
# Minimal valid galaxy fixture for quality validator tests
# ---------------------------------------------------------------------------

def _make_valid_galaxy(**overrides):
    base = {
        "id": "test_gal",
        "name": "Test Galaxy",
        "distance": 10.0,
        "galactic_radius": 30.0,
        "mass": 5.0,
        "accel": 1,
        "mass_model": {
            "bulge": {"M": 0, "a": 0.3},
            "disk": {"M": 3_000_000_000, "Rd": 2.0},
            "gas": {"M": 2_000_000_000, "Rd": 4.0},
        },
        "observations": [
            {"r": 1.0, "v": 80.0, "err": 2.0},
            {"r": 5.0, "v": 150.0, "err": 3.0},
            {"r": 10.0, "v": 180.0, "err": 2.5},
            {"r": 15.0, "v": 190.0, "err": 1.5},
        ],
        "references": ["SPARC VizieR J/AJ/152/157 (Lelli+2016)"],
    }
    base.update(overrides)
    return base


class TestQualityValidator:
    """Test validate_galaxy_quality with synthetic galaxy dicts."""

    def test_valid_galaxy_passes(self):
        data = _make_valid_galaxy()
        result = validate_galaxy_quality(data)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_too_few_observations(self):
        data = _make_valid_galaxy(observations=[
            {"r": 1.0, "v": 80.0, "err": 2.0},
            {"r": 5.0, "v": 150.0, "err": 3.0},
        ])
        result = validate_galaxy_quality(data)
        assert result["valid"] is False
        assert any("observation points" in e for e in result["errors"])

    def test_negative_velocity_is_error(self):
        obs = [
            {"r": 1.0, "v": -10.0, "err": 2.0},
            {"r": 5.0, "v": 150.0, "err": 3.0},
            {"r": 10.0, "v": 180.0, "err": 2.5},
        ]
        result = validate_galaxy_quality(_make_valid_galaxy(observations=obs))
        assert result["valid"] is False
        assert any("v is not positive" in e for e in result["errors"])

    def test_negative_radius_is_error(self):
        obs = [
            {"r": -1.0, "v": 80.0, "err": 2.0},
            {"r": 5.0, "v": 150.0, "err": 3.0},
            {"r": 10.0, "v": 180.0, "err": 2.5},
        ]
        result = validate_galaxy_quality(_make_valid_galaxy(observations=obs))
        assert result["valid"] is False
        assert any("r is not positive" in e for e in result["errors"])

    def test_unsorted_radii_warns(self):
        obs = [
            {"r": 10.0, "v": 180.0, "err": 2.5},
            {"r": 5.0, "v": 150.0, "err": 3.0},
            {"r": 15.0, "v": 190.0, "err": 1.5},
        ]
        result = validate_galaxy_quality(_make_valid_galaxy(observations=obs))
        assert result["valid"] is True
        assert any("not sorted" in w for w in result["warnings"])

    def test_duplicate_radii_warns(self):
        obs = [
            {"r": 1.0, "v": 80.0, "err": 2.0},
            {"r": 5.0, "v": 150.0, "err": 3.0},
            {"r": 5.0, "v": 155.0, "err": 3.0},
            {"r": 10.0, "v": 180.0, "err": 2.5},
        ]
        result = validate_galaxy_quality(_make_valid_galaxy(observations=obs))
        assert any("duplicate radii" in w for w in result["warnings"])

    def test_small_radial_span_warns(self):
        obs = [
            {"r": 1.0, "v": 80.0, "err": 2.0},
            {"r": 1.5, "v": 90.0, "err": 3.0},
            {"r": 2.0, "v": 100.0, "err": 2.5},
        ]
        result = validate_galaxy_quality(_make_valid_galaxy(observations=obs))
        assert result["valid"] is True
        assert any("radial span" in w for w in result["warnings"])

    def test_missing_disk_mass_is_error(self):
        data = _make_valid_galaxy()
        data["mass_model"]["disk"]["M"] = -1
        result = validate_galaxy_quality(data)
        assert result["valid"] is False
        assert any("disk.M" in e for e in result["errors"])

    def test_missing_disk_rd_is_error(self):
        data = _make_valid_galaxy()
        data["mass_model"]["disk"]["Rd"] = 0
        result = validate_galaxy_quality(data)
        assert result["valid"] is False
        assert any("disk.Rd" in e for e in result["errors"])

    def test_zero_total_baryonic_mass_is_error(self):
        data = _make_valid_galaxy()
        data["mass_model"]["disk"]["M"] = 0
        data["mass_model"]["gas"]["M"] = 0
        data["mass_model"]["bulge"]["M"] = 0
        result = validate_galaxy_quality(data)
        assert result["valid"] is False
        assert any("total baryonic mass" in e for e in result["errors"])

    def test_mass_consistency_warns_on_large_discrepancy(self):
        data = _make_valid_galaxy()
        data["mass"] = 0.001
        result = validate_galaxy_quality(data)
        assert any("differs from stated mass" in w for w in result["warnings"])

    def test_obs_exceeding_galactic_radius_warns(self):
        data = _make_valid_galaxy(galactic_radius=5.0)
        result = validate_galaxy_quality(data)
        assert any("exceeds galactic_radius" in w for w in result["warnings"])

    def test_invalid_id_is_error(self):
        result = validate_galaxy_quality(_make_valid_galaxy(id="bad id!"))
        assert result["valid"] is False
        assert any("id" in e and "invalid" in e for e in result["errors"])

    def test_empty_name_is_error(self):
        result = validate_galaxy_quality(_make_valid_galaxy(name=""))
        assert result["valid"] is False
        assert any("name" in e for e in result["errors"])

    def test_empty_references_warns(self):
        result = validate_galaxy_quality(_make_valid_galaxy(references=[]))
        assert any("references" in w for w in result["warnings"])

    def test_nan_distance_is_error(self):
        result = validate_galaxy_quality(_make_valid_galaxy(distance=float("nan")))
        assert result["valid"] is False

    def test_negative_err_warns(self):
        obs = [
            {"r": 1.0, "v": 80.0, "err": -1.0},
            {"r": 5.0, "v": 150.0, "err": 3.0},
            {"r": 10.0, "v": 180.0, "err": 2.5},
        ]
        result = validate_galaxy_quality(_make_valid_galaxy(observations=obs))
        assert any("err is negative" in w for w in result["warnings"])
