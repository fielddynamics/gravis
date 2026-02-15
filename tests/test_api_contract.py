"""
API contract and numerical robustness tests.

These verify that the API returns consistent, well-formed responses
across normal and extreme inputs, and that no combination of parameters
causes crashes, NaN, or Infinity values.
"""

import math
import json
import pytest


# Standard Milky Way mass model for reuse
MW_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}


class TestRotationCurveArrayConsistency:
    """All output arrays from /api/rotation/curve must have equal length
    and contain no NaN or Infinity values."""

    def test_all_arrays_equal_length(self, client):
        """radii, newtonian, dtg, mond, cdm, enclosed_mass must all
        have the same number of elements."""
        resp = client.post("/api/rotation/curve", json={
            "mass_model": MW_MODEL,
            "max_radius": 30,
            "num_points": 50,
            "accel_ratio": 1.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()

        n = len(data["radii"])
        assert n == 50
        assert len(data["newtonian"]) == n
        assert len(data["dtg"]) == n
        assert len(data["mond"]) == n
        assert len(data["cdm"]) == n
        assert len(data["enclosed_mass"]) == n

    def test_no_nan_or_inf(self, client):
        """No output value should be NaN or Infinity."""
        resp = client.post("/api/rotation/curve", json={
            "mass_model": MW_MODEL,
            "max_radius": 30,
            "num_points": 100,
            "accel_ratio": 1.0,
        })
        data = resp.get_json()
        for key in ["radii", "newtonian", "dtg", "mond", "cdm", "enclosed_mass"]:
            for i, val in enumerate(data[key]):
                assert not math.isnan(val), f"{key}[{i}] is NaN"
                assert not math.isinf(val), f"{key}[{i}] is Inf"

    def test_radii_evenly_spaced(self, client):
        """Radii should be evenly spaced from max_radius/N to max_radius."""
        resp = client.post("/api/rotation/curve", json={
            "mass_model": MW_MODEL,
            "max_radius": 20,
            "num_points": 10,
            "accel_ratio": 1.0,
        })
        data = resp.get_json()
        radii = data["radii"]
        # Expected spacing: 20/10 = 2.0 kpc
        for i in range(1, len(radii)):
            spacing = radii[i] - radii[i - 1]
            assert spacing == pytest.approx(2.0, abs=0.001)


class TestExtremeMasses:
    """Very small and very large mass models should not crash."""

    def test_very_small_mass(self, client):
        """10^4 M_sun (tiny dwarf) should produce valid velocities."""
        tiny_model = {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.0e4, "Rd": 0.5},
            "gas":   {"M": 0, "Rd": 1.0},
        }
        resp = client.post("/api/rotation/curve", json={
            "mass_model": tiny_model,
            "max_radius": 5,
            "num_points": 20,
            "accel_ratio": 1.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # All velocities should be positive (even if small)
        for v in data["dtg"]:
            assert v >= 0 and not math.isnan(v)

    def test_very_large_mass(self, client):
        """10^14 M_sun (galaxy cluster scale) should not overflow."""
        huge_model = {
            "bulge": {"M": 2.0e13, "a": 10.0},
            "disk":  {"M": 5.0e13, "Rd": 50.0},
            "gas":   {"M": 3.0e13, "Rd": 100.0},
        }
        resp = client.post("/api/rotation/curve", json={
            "mass_model": huge_model,
            "max_radius": 500,
            "num_points": 20,
            "accel_ratio": 1.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        for v in data["dtg"]:
            assert not math.isnan(v) and not math.isinf(v)

    def test_inference_extreme_velocity(self, client):
        """Very high velocity (500 km/s) and very low (5 km/s)
        should both return valid masses."""
        for v in [5.0, 500.0]:
            resp = client.post("/api/rotation/infer-mass", json={
                "r_kpc": 10.0,
                "v_km_s": v,
                "accel_ratio": 1.0,
            })
            assert resp.status_code == 200
            data = resp.get_json()
            m = data["inferred_mass_solar"]
            assert m > 0 and not math.isnan(m) and not math.isinf(m), \
                f"v={v}: inferred mass = {m}"


class TestAccelRatioExtremes:
    """The accel_ratio slider should smoothly interpolate between
    Newtonian (accel_ratio -> 0) and deep-field (accel_ratio -> large)."""

    def test_near_zero_approaches_newtonian(self, client):
        """With accel_ratio = 0.001 (a0 very small), GFD should be
        nearly identical to Newtonian."""
        resp = client.post("/api/rotation/curve", json={
            "mass_model": MW_MODEL,
            "max_radius": 30,
            "num_points": 20,
            "accel_ratio": 0.001,
        })
        data = resp.get_json()
        for v_n, v_g in zip(data["newtonian"], data["dtg"]):
            if v_n > 0:
                ratio = v_g / v_n
                assert ratio == pytest.approx(1.0, abs=0.05), \
                    f"accel_ratio=0.001: GFD/Newton={ratio:.4f}, expected ~1.0"

    def test_large_accel_ratio_strong_enhancement(self, client):
        """With accel_ratio = 100 (a0 very large), GFD should be
        much larger than Newtonian everywhere."""
        resp = client.post("/api/rotation/curve", json={
            "mass_model": MW_MODEL,
            "max_radius": 30,
            "num_points": 20,
            "accel_ratio": 100.0,
        })
        data = resp.get_json()
        # Skip the first few inner points where acceleration is high
        for v_n, v_g in zip(data["newtonian"][5:], data["dtg"][5:]):
            if v_n > 0:
                boost = v_g / v_n
                assert boost > 1.5, \
                    f"accel_ratio=100: GFD/Newton={boost:.2f}, expected >1.5"
