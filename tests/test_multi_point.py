"""
Tests for multi-point inference endpoint (/api/infer-mass-multi).

Validates per-point delta-v, sigma, enclosed fractions, aggregate
statistics, band methods, and shape diagnostic -- all computed
server-side and now fully unit-testable.
"""

import json
import math
import pytest


# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

NGC3198_MASS_MODEL = {
    "bulge": {"M": 0, "a": 0.0},
    "disk":  {"M": 1.7e10, "Rd": 3.0},
    "gas":   {"M": 1.5e10, "Rd": 6.0},
}

NGC3198_OBS = [
    {"r": 3,  "v": 110, "err": 8},
    {"r": 5,  "v": 145, "err": 5},
    {"r": 8,  "v": 150, "err": 3},
    {"r": 10, "v": 152, "err": 3},
    {"r": 15, "v": 150, "err": 3},
    {"r": 20, "v": 150, "err": 4},
    {"r": 25, "v": 150, "err": 5},
    {"r": 30, "v": 149, "err": 6},
]

MW_MASS_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}

MW_OBS = [
    {"r": 2,  "v": 206, "err": 25},
    {"r": 5,  "v": 236, "err": 7},
    {"r": 8,  "v": 230, "err": 3},
    {"r": 10, "v": 229, "err": 3},
    {"r": 15, "v": 230, "err": 5},
    {"r": 20, "v": 225, "err": 8},
    {"r": 25, "v": 220, "err": 10},
    {"r": 30, "v": 218, "err": 12},
]


def _post_multi(client, mass_model, observations, accel_ratio=1.0):
    """Helper to POST to /api/infer-mass-multi."""
    payload = {
        "mass_model": mass_model,
        "observations": observations,
        "accel_ratio": accel_ratio,
    }
    resp = client.post(
        "/api/infer-mass-multi",
        data=json.dumps(payload),
        content_type="application/json",
    )
    return resp


# ===========================================================================
# 1. Response structure
# ===========================================================================

class TestMultiPointResponseStructure:
    """Verify every expected field is present in the API response."""

    def test_top_level_fields(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        assert resp.status_code == 200
        data = resp.get_json()

        for key in [
            "points", "n_points",
            "mean_total", "std_total", "log10_mean", "cv_percent",
            "weighted_mean", "weighted_std",
            "log10_weighted_mean", "weighted_cv_percent",
            "min_total", "max_total",
            "band_methods", "shape_diagnostic",
        ]:
            assert key in data, f"Missing top-level key: {key}"

    def test_per_point_fields(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()

        for i, pt in enumerate(data["points"]):
            for key in [
                "r_kpc", "v_km_s", "err",
                "inferred_total", "log10_total", "enclosed_frac",
                "v_gfd", "delta_v", "sigma_dev",
            ]:
                assert key in pt, f"Point {i} missing key: {key}"

    def test_band_methods_present(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        bm = data["band_methods"]
        for key in ["weighted_scatter", "obs_error", "iqr"]:
            assert key in bm, f"Missing band method: {key}"

    def test_shape_diagnostic_fields(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        sd = data["shape_diagnostic"]
        assert sd is not None, "shape_diagnostic should be present for 8 points"
        for key in [
            "inner_r_max", "outer_r_min",
            "inner_mean_dv", "outer_mean_dv",
            "inner_mean_sigma", "outer_mean_sigma",
            "n_inner", "n_outer",
        ]:
            assert key in sd, f"Missing shape diagnostic key: {key}"


# ===========================================================================
# 2. Per-point delta-v and sigma calculations
# ===========================================================================

class TestDeltaVAndSigma:
    """Verify delta-v = v_obs - v_gfd, sigma = |delta_v| / err."""

    def test_delta_v_is_obs_minus_gfd(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        for pt in data["points"]:
            expected = round(pt["v_km_s"] - pt["v_gfd"], 2)
            assert pt["delta_v"] == expected, (
                f"At r={pt['r_kpc']}: delta_v={pt['delta_v']}, "
                f"expected v_obs({pt['v_km_s']}) - v_gfd({pt['v_gfd']}) = {expected}"
            )

    def test_sigma_is_abs_delta_over_err(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        for pt in data["points"]:
            if pt["err"] > 0:
                expected = round(abs(pt["delta_v"]) / pt["err"], 2)
                assert pt["sigma_dev"] == expected, (
                    f"At r={pt['r_kpc']}: sigma_dev={pt['sigma_dev']}, "
                    f"expected |{pt['delta_v']}|/{pt['err']} = {expected}"
                )

    def test_sigma_positive_or_zero(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        for pt in data["points"]:
            assert pt["sigma_dev"] >= 0

    def test_gfd_velocity_positive(self, client):
        """GFD predicted velocity must be positive at every radius."""
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        for pt in data["points"]:
            assert pt["v_gfd"] > 0, (
                f"v_gfd should be positive at r={pt['r_kpc']}"
            )


# ===========================================================================
# 3. Enclosed fraction
# ===========================================================================

class TestEnclosedFraction:
    """Enclosed fraction should be monotonically increasing with radius."""

    def test_monotonically_increasing(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        fracs = [pt["enclosed_frac"] for pt in data["points"]]
        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i - 1], (
                f"Enclosed fraction should increase: "
                f"f({data['points'][i-1]['r_kpc']})={fracs[i-1]} > "
                f"f({data['points'][i]['r_kpc']})={fracs[i]}"
            )

    def test_fractions_between_0_and_1(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        for pt in data["points"]:
            assert 0 < pt["enclosed_frac"] <= 1.0, (
                f"Enclosed frac out of range at r={pt['r_kpc']}: "
                f"{pt['enclosed_frac']}"
            )

    def test_outer_points_have_high_fraction(self, client):
        """At 30 kpc for NGC 3198, most mass should be enclosed."""
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        last = data["points"][-1]
        assert last["enclosed_frac"] > 0.7, (
            f"At 30 kpc enclosed frac = {last['enclosed_frac']}, expected > 0.7"
        )


# ===========================================================================
# 4. Aggregate statistics
# ===========================================================================

class TestAggregateStatistics:
    """Verify mean, std, weighted stats, and band methods."""

    def test_n_points_matches(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        assert data["n_points"] == len(data["points"])
        assert data["n_points"] == len(NGC3198_OBS)

    def test_mean_between_min_and_max(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        assert data["min_total"] <= data["mean_total"] <= data["max_total"]
        assert data["min_total"] <= data["weighted_mean"] <= data["max_total"]

    def test_std_positive(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        assert data["std_total"] >= 0
        assert data["weighted_std"] >= 0

    def test_log10_mean_consistent(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        expected_log = math.log10(data["mean_total"])
        assert abs(data["log10_mean"] - expected_log) < 0.01

    def test_band_methods_non_negative(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        bm = data["band_methods"]
        assert bm["weighted_scatter"] >= 0
        assert bm["obs_error"] >= 0
        assert bm["iqr"] >= 0

    def test_iqr_less_than_or_equal_total_range(self, client):
        """IQR half-width should not exceed half the total range."""
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        half_range = (data["max_total"] - data["min_total"]) / 2.0
        assert data["band_methods"]["iqr"] <= half_range + 1.0  # +1 for rounding


# ===========================================================================
# 5. Shape diagnostic
# ===========================================================================

class TestShapeDiagnostic:
    """Verify the shape diagnostic inner/outer split."""

    def test_inner_outer_counts_sum(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        sd = data["shape_diagnostic"]
        # Should split the diagnosable points (those with enc_frac >= 0.05)
        n_diag = sd["n_inner"] + sd["n_outer"]
        # At least 4 points needed for diagnostic
        assert n_diag >= 4

    def test_inner_r_max_less_than_outer_r_min(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        sd = data["shape_diagnostic"]
        assert sd["inner_r_max"] <= sd["outer_r_min"]

    def test_sigma_values_non_negative(self, client):
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        sd = data["shape_diagnostic"]
        assert sd["inner_mean_sigma"] >= 0
        assert sd["outer_mean_sigma"] >= 0

    def test_no_diagnostic_with_few_points(self, client):
        """With only 2 points, shape diagnostic should be null."""
        obs = [
            {"r": 5, "v": 145, "err": 5},
            {"r": 10, "v": 152, "err": 3},
        ]
        resp = _post_multi(client, NGC3198_MASS_MODEL, obs)
        data = resp.get_json()
        # May be None if fewer than 4 diagnosable points
        # (2 points won't reach threshold of 4 for diagnostic)
        assert data["shape_diagnostic"] is None


# ===========================================================================
# 6. Milky Way cross-check (well-fitting galaxy)
# ===========================================================================

class TestMilkyWayMultiPoint:
    """The MW should show a good GFD fit with low sigma deviations."""

    def test_mw_low_average_sigma(self, client):
        """Average sigma across all MW points should be < 3."""
        resp = _post_multi(client, MW_MASS_MODEL, MW_OBS)
        data = resp.get_json()
        sigmas = [pt["sigma_dev"] for pt in data["points"]
                  if pt["sigma_dev"] is not None]
        avg_sigma = sum(sigmas) / len(sigmas)
        assert avg_sigma < 3.0, (
            f"MW average sigma = {avg_sigma:.2f}, expected < 3.0"
        )

    def test_mw_mass_offset_small(self, client):
        """Weighted mean should be within 20% of the model total."""
        resp = _post_multi(client, MW_MASS_MODEL, MW_OBS)
        data = resp.get_json()
        m_total = sum(
            c["M"] for c in MW_MASS_MODEL.values() if isinstance(c, dict)
        )
        offset_pct = abs(data["weighted_mean"] - m_total) / m_total * 100
        assert offset_pct < 20, (
            f"MW mass offset = {offset_pct:.1f}%, expected < 20%"
        )

    def test_mw_outer_points_within_2sigma(self, client):
        """Outer MW points (r >= 10 kpc) should mostly be within 2 sigma."""
        resp = _post_multi(client, MW_MASS_MODEL, MW_OBS)
        data = resp.get_json()
        outer = [pt for pt in data["points"] if pt["r_kpc"] >= 10]
        within_2sig = sum(1 for pt in outer if pt["sigma_dev"] <= 2.0)
        assert within_2sig >= len(outer) // 2, (
            f"Only {within_2sig}/{len(outer)} outer points within 2 sigma"
        )


# ===========================================================================
# 7. Edge cases and error handling
# ===========================================================================

class TestEdgeCases:
    """Boundary conditions and error paths."""

    def test_minimum_points(self, client):
        """Exactly 2 points should work (minimum allowed)."""
        obs = [
            {"r": 5,  "v": 145, "err": 5},
            {"r": 20, "v": 150, "err": 4},
        ]
        resp = _post_multi(client, NGC3198_MASS_MODEL, obs)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_points"] == 2

    def test_single_point_rejected(self, client):
        """Only 1 observation should return 400."""
        obs = [{"r": 10, "v": 150, "err": 3}]
        resp = _post_multi(client, NGC3198_MASS_MODEL, obs)
        assert resp.status_code == 400

    def test_missing_mass_model(self, client):
        payload = {"observations": NGC3198_OBS}
        resp = client.post(
            "/api/infer-mass-multi",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_missing_observations(self, client):
        payload = {"mass_model": NGC3198_MASS_MODEL}
        resp = client.post(
            "/api/infer-mass-multi",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_zero_radius_skipped(self, client):
        """Points with r=0 should be silently skipped."""
        obs = [
            {"r": 0,  "v": 100, "err": 5},
            {"r": 5,  "v": 145, "err": 5},
            {"r": 10, "v": 152, "err": 3},
            {"r": 20, "v": 150, "err": 4},
        ]
        resp = _post_multi(client, NGC3198_MASS_MODEL, obs)
        data = resp.get_json()
        assert data["n_points"] == 3  # r=0 excluded

    def test_zero_velocity_skipped(self, client):
        """Points with v=0 should be silently skipped."""
        obs = [
            {"r": 5,  "v": 0,   "err": 5},
            {"r": 10, "v": 152, "err": 3},
            {"r": 20, "v": 150, "err": 4},
        ]
        resp = _post_multi(client, NGC3198_MASS_MODEL, obs)
        data = resp.get_json()
        assert data["n_points"] == 2  # v=0 excluded

    def test_accel_ratio_affects_gfd(self, client):
        """Different accel_ratio should change v_gfd and thus delta_v."""
        resp1 = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS, accel_ratio=1.0)
        resp2 = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS, accel_ratio=2.0)
        d1 = resp1.get_json()
        d2 = resp2.get_json()
        # v_gfd should differ when accel_ratio changes
        gfd1 = [pt["v_gfd"] for pt in d1["points"]]
        gfd2 = [pt["v_gfd"] for pt in d2["points"]]
        assert gfd1 != gfd2, "Different accel_ratio should produce different v_gfd"


# ===========================================================================
# 8. Consistency: delta-v sign vs inferred mass offset
# ===========================================================================

class TestConsistency:
    """Cross-check that delta-v direction aligns with mass inference."""

    def test_positive_delta_v_implies_higher_inferred_mass(self, client):
        """
        If v_obs > v_gfd (positive delta_v), then the observation demands
        more mass than the model provides at that radius, so inferred_total
        should exceed the model total.
        """
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        m_total = NGC3198_MASS_MODEL["disk"]["M"] + NGC3198_MASS_MODEL["gas"]["M"]
        for pt in data["points"]:
            if pt["delta_v"] > 1.0:  # meaningfully positive
                assert pt["inferred_total"] > m_total * 0.95, (
                    f"At r={pt['r_kpc']}: positive delta_v={pt['delta_v']} "
                    f"but inferred_total={pt['inferred_total']:.2e} "
                    f"not > model total={m_total:.2e}"
                )

    def test_negative_delta_v_implies_lower_inferred_mass(self, client):
        """
        If v_obs < v_gfd (negative delta_v), inferred_total should be
        below the model total.
        """
        resp = _post_multi(client, NGC3198_MASS_MODEL, NGC3198_OBS)
        data = resp.get_json()
        m_total = NGC3198_MASS_MODEL["disk"]["M"] + NGC3198_MASS_MODEL["gas"]["M"]
        for pt in data["points"]:
            if pt["delta_v"] < -1.0:  # meaningfully negative
                assert pt["inferred_total"] < m_total * 1.05, (
                    f"At r={pt['r_kpc']}: negative delta_v={pt['delta_v']} "
                    f"but inferred_total={pt['inferred_total']:.2e} "
                    f"not < model total={m_total:.2e}"
                )
