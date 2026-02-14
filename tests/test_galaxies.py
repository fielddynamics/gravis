"""
Tests for galaxy catalog data integrity.

Validates that all galaxy entries have required fields, sensible values,
and that mass models are consistent with stated total masses.
"""

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
)
from physics.mass_model import total_mass


class TestCatalogStructure:
    """Test catalog completeness and field presence."""

    def test_prediction_galaxies_not_empty(self):
        assert len(PREDICTION_GALAXIES) >= 6

    def test_inference_galaxies_not_empty(self):
        assert len(INFERENCE_GALAXIES) >= 4

    def test_all_prediction_have_required_fields(self):
        for g in PREDICTION_GALAXIES:
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

    def test_unique_ids(self):
        """All galaxy IDs must be unique across entire catalog."""
        all_galaxies = (
            PREDICTION_GALAXIES
            + SIMPLE_PREDICTION_GALAXIES
            + INFERENCE_GALAXIES
        )
        ids = [g["id"] for g in all_galaxies]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


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

    def test_get_by_id_found(self):
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

    def test_get_prediction_includes_simple(self):
        """get_prediction_galaxies should include both detailed and simple entries."""
        galaxies = get_prediction_galaxies()
        ids = [g["id"] for g in galaxies]
        assert "milky_way" in ids
        assert "ic2574" in ids
