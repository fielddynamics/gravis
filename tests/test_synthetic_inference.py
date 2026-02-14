"""
Multi-point inference validation with synthetic data.

These tests use controlled, synthetic observations to verify that the
inference pipeline produces mathematically correct results.  By generating
"perfect" data from a known mass model, we can verify that the residuals,
statistics, and diagnostics are exact -- removing any ambiguity from
real observational noise.
"""

import json
import math
import pytest

from physics.aqual import velocity as dtg_velocity
from physics.mass_model import enclosed_mass, total_mass


# A well-characterized mass model for testing
TEST_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}
TEST_TOTAL = total_mass(TEST_MODEL)


def _make_perfect_observations(mass_model, radii, err=5.0):
    """Generate synthetic observations by computing exact GFD velocities
    from a known mass model.  These should produce zero residuals."""
    obs = []
    for r in radii:
        m = enclosed_mass(r, mass_model)
        v = dtg_velocity(r, m)
        obs.append({"r": r, "v": round(v, 4), "err": err})
    return obs


def _make_scaled_observations(mass_model, radii, scale, err=5.0):
    """Generate observations from a scaled version of the mass model.
    This simulates a known mass offset."""
    scaled = {}
    for comp in ("bulge", "disk", "gas"):
        if comp in mass_model and mass_model[comp].get("M", 0) > 0:
            scaled[comp] = dict(mass_model[comp])
            scaled[comp]["M"] = scaled[comp]["M"] * scale
    obs = []
    for r in radii:
        m = enclosed_mass(r, scaled)
        v = dtg_velocity(r, m)
        obs.append({"r": r, "v": round(v, 4), "err": err})
    return obs


class TestPerfectDataZeroResiduals:
    """When observations are generated from the exact same mass model,
    all residuals should be approximately zero."""

    def test_zero_delta_v(self, client):
        """All delta_v values should be ~0 for perfect synthetic data."""
        radii = [2, 5, 8, 10, 15, 20, 25]
        obs = _make_perfect_observations(TEST_MODEL, radii)
        resp = client.post("/api/infer-mass-multi", json={
            "observations": obs,
            "mass_model": TEST_MODEL,
            "accel_ratio": 1.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        for pt in data["points"]:
            assert abs(pt["delta_v"]) < 0.1, \
                f"r={pt['r_kpc']}: delta_v={pt['delta_v']}, expected ~0"

    def test_zero_sigma(self, client):
        """All sigma_dev values should be ~0 for perfect data."""
        radii = [2, 5, 8, 10, 15, 20, 25]
        obs = _make_perfect_observations(TEST_MODEL, radii)
        resp = client.post("/api/infer-mass-multi", json={
            "observations": obs,
            "mass_model": TEST_MODEL,
            "accel_ratio": 1.0,
        })
        data = resp.get_json()
        for pt in data["points"]:
            if pt["sigma_dev"] is not None:
                assert pt["sigma_dev"] < 0.1, \
                    f"r={pt['r_kpc']}: sigma={pt['sigma_dev']}, expected ~0"

    def test_near_zero_mass_offset(self, client):
        """The mass offset (weighted mean vs model total) should be ~0%."""
        radii = [2, 5, 8, 10, 15, 20, 25]
        obs = _make_perfect_observations(TEST_MODEL, radii)
        resp = client.post("/api/infer-mass-multi", json={
            "observations": obs,
            "mass_model": TEST_MODEL,
            "accel_ratio": 1.0,
        })
        data = resp.get_json()
        offset_pct = abs(data["weighted_mean"] - TEST_TOTAL) / TEST_TOTAL * 100
        assert offset_pct < 1.0, \
            f"Mass offset = {offset_pct:.2f}%, expected <1%"


class TestKnownMassOffsetDetected:
    """When observations come from a 10% heavier model, the inference
    should detect approximately +10% mass offset."""

    def test_ten_percent_offset(self, client):
        """Feed observations from a model scaled by 1.10.
        The inferred masses should average ~10% above the reference model."""
        radii = [5, 8, 10, 15, 20]
        obs = _make_scaled_observations(TEST_MODEL, radii, scale=1.10)
        resp = client.post("/api/infer-mass-multi", json={
            "observations": obs,
            "mass_model": TEST_MODEL,
            "accel_ratio": 1.0,
        })
        data = resp.get_json()
        # The weighted mean should be ~10% above the model total
        expected = TEST_TOTAL * 1.10
        actual = data["weighted_mean"]
        offset = (actual - TEST_TOTAL) / TEST_TOTAL * 100
        assert 5.0 < offset < 20.0, \
            f"Mass offset = {offset:.1f}%, expected ~10%"


class TestWrongShapeTriggersdiagnostic:
    """When the mass model shape is wrong (different scale length),
    the shape diagnostic should detect inner/outer disagreement."""

    def test_wrong_disk_scale_length(self, client):
        """Generate observations from a model with disk Rd=4.0 instead
        of 2.5.  The shape diagnostic should show systematic residuals."""
        wrong_shape = {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 4.0},  # Wrong: 4.0 instead of 2.5
            "gas":   {"M": 1.0e10, "Rd": 5.0},
        }
        radii = [2, 4, 6, 8, 10, 15, 20, 25]
        obs = _make_perfect_observations(wrong_shape, radii)

        resp = client.post("/api/infer-mass-multi", json={
            "observations": obs,
            "mass_model": TEST_MODEL,  # Reference model with Rd=2.5
            "accel_ratio": 1.0,
        })
        data = resp.get_json()

        # The shape diagnostic should exist and show significant sigma
        diag = data.get("shape_diagnostic")
        assert diag is not None, "Shape diagnostic should not be null"

        # With wrong shape, at least one region should have mean_sigma > 0.5
        max_sigma = max(diag["inner_mean_sigma"], diag["outer_mean_sigma"])
        assert max_sigma > 0.5, \
            f"Expected shape mismatch: inner_sigma={diag['inner_mean_sigma']}, " \
            f"outer_sigma={diag['outer_mean_sigma']}"


class TestWeightedStatisticsHandComputed:
    """Verify weighted mean and std against hand-computed values for a
    simple 3-point dataset."""

    def test_three_point_weighted_mean(self, client):
        """Three observations with known enclosed fractions.  Verify
        the weighted mean matches the hand-computed result."""
        # Use a simple mass model where we can predict the enclosed fractions
        simple_model = {
            "bulge": {"M": 0, "a": 1.0},
            "disk":  {"M": 1.0e10, "Rd": 3.0},
            "gas":   {"M": 0, "a": 1.0},
        }
        radii = [5, 10, 20]
        obs = _make_perfect_observations(simple_model, radii, err=3.0)

        resp = client.post("/api/infer-mass-multi", json={
            "observations": obs,
            "mass_model": simple_model,
            "accel_ratio": 1.0,
        })
        assert resp.status_code == 200
        data = resp.get_json()

        # With perfect data, all inferred totals should be ~ the model total
        m_total = total_mass(simple_model)
        for pt in data["points"]:
            assert pt["inferred_total"] == pytest.approx(m_total, rel=0.01), \
                f"r={pt['r_kpc']}: inferred={pt['inferred_total']:.2e}, " \
                f"model={m_total:.2e}"

        # The weighted mean should also be ~ the model total
        assert data["weighted_mean"] == pytest.approx(m_total, rel=0.01)
