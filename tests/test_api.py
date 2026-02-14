"""
Tests for Flask API endpoints.

Integration tests that validate the REST API returns correct
status codes, JSON structure, and physically reasonable values.
"""

import json
import math
import pytest


class TestGalaxiesEndpoint:
    """Test GET /api/galaxies."""

    def test_list_galaxies(self, client):
        resp = client.get("/api/galaxies")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "prediction" in data
        assert "inference" in data
        assert len(data["prediction"]) >= 6
        assert len(data["inference"]) >= 4

    def test_galaxy_by_id(self, client):
        resp = client.get("/api/galaxies/milky_way")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == "milky_way"
        assert "mass_model" in data
        assert "observations" in data

    def test_galaxy_not_found(self, client):
        resp = client.get("/api/galaxies/nonexistent")
        assert resp.status_code == 404


class TestRotationCurveEndpoint:
    """Test POST /api/rotation-curve."""

    def test_milky_way_curve(self, client):
        payload = {
            "max_radius": 30,
            "num_points": 50,
            "accel_ratio": 1.0,
            "mass_model": {
                "bulge": {"M": 1.5e10, "a": 0.6},
                "disk":  {"M": 5.0e10, "Rd": 2.5},
                "gas":   {"M": 1.0e10, "Rd": 5.0}
            }
        }
        resp = client.post(
            "/api/rotation-curve",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "radii" in data
        assert "newtonian" in data
        assert "dtg" in data
        assert "mond" in data
        assert "enclosed_mass" in data
        assert len(data["radii"]) == 50

    def test_dtg_above_newtonian(self, client):
        """DTG velocity should be >= Newtonian at every radius."""
        payload = {
            "max_radius": 30,
            "num_points": 50,
            "accel_ratio": 1.0,
            "mass_model": {
                "bulge": {"M": 1.5e10, "a": 0.6},
                "disk":  {"M": 5.0e10, "Rd": 2.5},
                "gas":   {"M": 1.0e10, "Rd": 5.0}
            }
        }
        resp = client.post(
            "/api/rotation-curve",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        for i in range(len(data["radii"])):
            assert data["dtg"][i] >= data["newtonian"][i] - 0.01, \
                f"At r={data['radii'][i]}: DTG={data['dtg'][i]} < Newton={data['newtonian'][i]}"

    def test_velocities_positive(self, client):
        payload = {
            "max_radius": 20,
            "num_points": 20,
            "accel_ratio": 1.0,
            "mass_model": {
                "bulge": {"M": 1e9, "a": 0.5},
                "disk":  {"M": 5e9, "Rd": 2.0},
                "gas":   {"M": 2e9, "Rd": 4.0}
            }
        }
        resp = client.post(
            "/api/rotation-curve",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        for v in data["newtonian"]:
            assert v > 0
        for v in data["dtg"]:
            assert v > 0
        for v in data["mond"]:
            assert v > 0

    def test_missing_mass_model(self, client):
        payload = {"max_radius": 30, "num_points": 50}
        resp = client.post(
            "/api/rotation-curve",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_num_points_capped(self, client):
        """num_points should be capped at 500."""
        payload = {
            "max_radius": 30,
            "num_points": 10000,
            "accel_ratio": 1.0,
            "mass_model": {
                "bulge": {"M": 1e10, "a": 0.5},
                "disk":  {"M": 5e10, "Rd": 2.5},
                "gas":   {"M": 1e10, "Rd": 5.0}
            }
        }
        resp = client.post(
            "/api/rotation-curve",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert len(data["radii"]) == 500


class TestInferMassEndpoint:
    """Test POST /api/infer-mass."""

    def test_milky_way_inference(self, client):
        payload = {
            "r_kpc": 8.0,
            "v_km_s": 230.0,
            "accel_ratio": 1.0
        }
        resp = client.post(
            "/api/infer-mass",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "inferred_mass_solar" in data
        assert "log10_mass" in data
        # Should infer order of 10^10 M_sun
        assert 10 < data["log10_mass"] < 12

    def test_inference_returns_positive_mass(self, client):
        payload = {
            "r_kpc": 10.0,
            "v_km_s": 150.0,
            "accel_ratio": 1.0
        }
        resp = client.post(
            "/api/infer-mass",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["inferred_mass_solar"] > 0


class TestInferMassModelEndpoint:
    """Test POST /api/infer-mass-model."""

    MW_PAYLOAD = {
        "r_kpc": 8.0,
        "v_km_s": 230.0,
        "accel_ratio": 1.0,
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0}
        }
    }

    def test_basic_response_structure(self, client):
        resp = client.post(
            "/api/infer-mass-model",
            data=json.dumps(self.MW_PAYLOAD),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "inferred_enclosed" in data
        assert "scale_factor" in data
        assert "inferred_mass_model" in data
        assert "inferred_total" in data
        assert "log10_total" in data
        assert "btfr_mass" in data
        assert "log10_btfr" in data

    def test_inferred_model_has_components(self, client):
        resp = client.post(
            "/api/infer-mass-model",
            data=json.dumps(self.MW_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        model = data["inferred_mass_model"]
        assert "bulge" in model
        assert "disk" in model
        assert "gas" in model
        # Scale lengths should be preserved
        assert model["bulge"]["a"] == 0.6
        assert model["disk"]["Rd"] == 2.5
        assert model["gas"]["Rd"] == 5.0

    def test_scale_preserves_proportions(self, client):
        """Component masses should maintain their relative proportions."""
        resp = client.post(
            "/api/infer-mass-model",
            data=json.dumps(self.MW_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        model = data["inferred_mass_model"]
        # Original ratio: bulge/disk = 1.5/5.0 = 0.3
        original_ratio = 1.5e10 / 5.0e10
        inferred_ratio = model["bulge"]["M"] / model["disk"]["M"]
        assert abs(inferred_ratio - original_ratio) / original_ratio < 1e-6

    def test_milky_way_total_reasonable(self, client):
        """MW inference should give order 10^10 M_sun total."""
        resp = client.post(
            "/api/infer-mass-model",
            data=json.dumps(self.MW_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert 10 < data["log10_total"] < 12

    def test_btfr_mass_reasonable(self, client):
        """BTFR mass for MW (230 km/s) should be ~10^10 M_sun."""
        resp = client.post(
            "/api/infer-mass-model",
            data=json.dumps(self.MW_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert 10 < data["log10_btfr"] < 12

    def test_missing_mass_model(self, client):
        payload = {"r_kpc": 8.0, "v_km_s": 230.0}
        resp = client.post(
            "/api/infer-mass-model",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_different_shape_changes_total(self, client):
        """Different scale lengths should produce different inferred totals."""
        payload1 = dict(self.MW_PAYLOAD)
        payload2 = {
            "r_kpc": 8.0,
            "v_km_s": 230.0,
            "accel_ratio": 1.0,
            "mass_model": {
                "bulge": {"M": 1.5e10, "a": 0.6},
                "disk":  {"M": 5.0e10, "Rd": 5.0},
                "gas":   {"M": 1.0e10, "Rd": 10.0}
            }
        }
        resp1 = client.post("/api/infer-mass-model", data=json.dumps(payload1), content_type="application/json")
        resp2 = client.post("/api/infer-mass-model", data=json.dumps(payload2), content_type="application/json")
        total1 = resp1.get_json()["inferred_total"]
        total2 = resp2.get_json()["inferred_total"]
        # More extended disk => less mass enclosed at 8 kpc => higher total needed
        assert total2 > total1


class TestConstantsEndpoint:
    """Test GET /api/constants."""

    def test_get_constants(self, client):
        resp = client.get("/api/constants")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "G" in data
        assert "M_SUN" in data
        assert "A0" in data
        assert data["K_SIMPLEX"] == 4


class TestIndexRoute:
    """Test main page serving."""

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"GRAVIS" in resp.data
