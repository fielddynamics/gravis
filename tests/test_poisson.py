"""
Tests for the GFD Poisson: covariant completion composed with kappa operator.

Verifies:
  1. Kappa screening function: boundary values, monotonicity, throat value.
  2. Poisson velocity: composition with covariant (v_cov / sqrt(kappa)).
  3. Traced callable (gfd_poisson_eq): intermediates and guards.
  4. Engine integration: gfd_poisson appears in API, above GFD in inner
     galaxy, converges to GFD at large r.
"""

import math
import pytest

from physics.poisson import (
    kappa,
    KAPPA_FLOOR,
    G_EFF_RATIO,
    velocity as poisson_velocity,
)
from physics.manifold import ALPHA_THROAT
from physics.equations import gfd_poisson_eq
from physics.aqual import velocity as ref_gfd_velocity
from physics.engine import GravisConfig, GravisEngine


MILKY_WAY_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}

R_GAL = 30.0  # Milky Way galactic radius in kpc


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
class TestConstants:

    def test_alpha_throat(self):
        assert ALPHA_THROAT == pytest.approx(0.30, abs=1e-6)

    def test_g_eff_ratio(self):
        assert G_EFF_RATIO == pytest.approx(17.0 / 13.0, rel=1e-10)

    def test_kappa_floor_positive(self):
        assert KAPPA_FLOOR > 0
        assert KAPPA_FLOOR < 0.01


# -----------------------------------------------------------------------
# Kappa screening function
# -----------------------------------------------------------------------
class TestKappa:

    def test_at_zero(self):
        """kappa(0) should be 0 (full manifold coupling)."""
        assert kappa(0.0, R_GAL) == 0.0

    def test_at_throat(self):
        """kappa(r_t) = 1 - exp(-1) = 0.6321."""
        r_t = ALPHA_THROAT * R_GAL
        k = kappa(r_t, R_GAL)
        assert k == pytest.approx(1.0 - math.exp(-1.0), rel=1e-6)

    def test_approaches_one_at_large_r(self):
        """At large r, kappa -> 1 (standard gravity)."""
        k = kappa(50.0, R_GAL)
        assert k > 0.99

    def test_monotonically_increasing(self):
        """kappa should increase with r."""
        prev = 0.0
        for r in [0.1, 1, 3, 5, 9, 15, 20, 30, 50]:
            k = kappa(r, R_GAL)
            assert k >= prev, f"kappa not increasing at r={r}"
            prev = k

    def test_values_between_0_and_1(self):
        for r in [0.0, 0.1, 1, 5, 9, 15, 20, 30, 50]:
            k = kappa(r, R_GAL)
            assert 0.0 <= k <= 1.0

    def test_negative_r(self):
        assert kappa(-1.0, R_GAL) == 0.0

    def test_zero_galactic_radius(self):
        assert kappa(5.0, 0.0) == 0.0

    def test_at_2_r_t(self):
        """kappa(2*r_t) = 1 - exp(-4) ~ 0.9817."""
        r_t = ALPHA_THROAT * R_GAL
        k = kappa(2 * r_t, R_GAL)
        assert k == pytest.approx(1.0 - math.exp(-4.0), rel=1e-6)


# -----------------------------------------------------------------------
# Poisson velocity (composition)
# -----------------------------------------------------------------------
class TestPoissonVelocity:

    def test_zero_radius(self):
        assert poisson_velocity(0.0, 1e10, R_GAL) == 0.0

    def test_zero_mass(self):
        assert poisson_velocity(5.0, 0.0, R_GAL) == 0.0

    def test_above_gfd_inside_throat(self):
        """Inside the throat, kappa < 1, so v_poisson > v_gfd."""
        r = 3.0
        m = 5e10
        v_poi = poisson_velocity(r, m, R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        assert v_poi > v_gfd

    def test_converges_to_gfd_at_large_r(self):
        """At large r, kappa -> 1, so v_poisson -> v_gfd."""
        r = 50.0
        m = 7.5e10
        v_poi = poisson_velocity(r, m, R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        # Should match within 1%
        assert v_poi == pytest.approx(v_gfd, rel=0.01)

    def test_composition_formula(self):
        """v_poisson = v_gfd / sqrt(kappa)."""
        r = 10.0
        m = 5e10
        v_gfd = ref_gfd_velocity(r, m)
        k = kappa(r, R_GAL)
        k_safe = max(k, KAPPA_FLOOR)
        expected = v_gfd / math.sqrt(k_safe)
        actual = poisson_velocity(r, m, R_GAL)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_at_throat_boost(self):
        """At the throat: kappa = 1 - e^(-1) ~ 0.632, boost = 1.26x."""
        r_t = ALPHA_THROAT * R_GAL
        m = 5e10
        v_poi = poisson_velocity(r_t, m, R_GAL)
        v_gfd = ref_gfd_velocity(r_t, m)
        expected_boost = 1.0 / math.sqrt(1.0 - math.exp(-1.0))
        assert v_poi == pytest.approx(v_gfd * expected_boost, rel=0.001)

    def test_smooth_no_kink(self):
        """Velocity should be smooth without sudden jumps."""
        m = 5e10
        velocities = [poisson_velocity(r, m, R_GAL)
                      for r in [3, 5, 7, 9, 11, 13, 15, 17, 20]]
        for i in range(1, len(velocities)):
            assert abs(velocities[i] - velocities[i-1]) < 30

    def test_enhancement_decreases_with_r(self):
        """The boost factor (v_poisson / v_gfd) should decrease with r."""
        m = 5e10
        prev_boost = float('inf')
        for r in [1, 3, 5, 9, 15, 20, 30]:
            v_poi = poisson_velocity(r, m, R_GAL)
            v_gfd = ref_gfd_velocity(r, m)
            if v_gfd > 0:
                boost = v_poi / v_gfd
                assert boost <= prev_boost + 1e-6, (
                    f"Boost not decreasing at r={r}"
                )
                prev_boost = boost


# -----------------------------------------------------------------------
# Traced callable
# -----------------------------------------------------------------------
class TestGfdPoissonEq:

    def test_intermediates_keys(self):
        v, ints = gfd_poisson_eq(5.0, 5e10, galactic_radius_kpc=R_GAL)
        expected = {"kappa", "v_cov", "galactic_radius",
                    "g_N", "y_N", "x", "g_cov"}
        assert set(ints.keys()) == expected

    def test_kappa_at_throat(self):
        r_t = ALPHA_THROAT * R_GAL
        _, ints = gfd_poisson_eq(r_t, 5e10, galactic_radius_kpc=R_GAL)
        assert ints["kappa"] == pytest.approx(
            1.0 - math.exp(-1.0), rel=1e-6
        )

    def test_galactic_radius_recorded(self):
        _, ints = gfd_poisson_eq(10.0, 5e10, galactic_radius_kpc=R_GAL)
        assert ints["galactic_radius"] == R_GAL

    def test_v_cov_matches_gfd(self):
        """The v_cov intermediate should match standard GFD velocity."""
        r, m = 10.0, 5e10
        _, ints = gfd_poisson_eq(r, m, galactic_radius_kpc=R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        assert ints["v_cov"] == pytest.approx(v_gfd, rel=1e-8)

    def test_velocity_matches_module(self):
        r, m = 10.0, 5e10
        v_eq, _ = gfd_poisson_eq(r, m, galactic_radius_kpc=R_GAL)
        v_mod = poisson_velocity(r, m, R_GAL)
        assert v_eq == pytest.approx(v_mod, rel=1e-10)

    def test_zero_guards(self):
        v, _ = gfd_poisson_eq(0.0, 5e10, galactic_radius_kpc=R_GAL)
        assert v == 0.0
        v, _ = gfd_poisson_eq(5.0, 0.0, galactic_radius_kpc=R_GAL)
        assert v == 0.0


# -----------------------------------------------------------------------
# Engine integration
# -----------------------------------------------------------------------
class TestEngineIntegration:

    @pytest.fixture
    def config(self):
        return GravisConfig(
            mass_model=MILKY_WAY_MODEL,
            max_radius=30.0,
            num_points=100,
            galactic_radius=30.0,
        )

    def test_gfd_poisson_in_api(self, config):
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        assert "gfd_poisson" in api
        assert len(api["gfd_poisson"]) == 100

    def test_poisson_above_gfd_in_inner_galaxy(self, config):
        """Poisson should be above GFD everywhere (kappa < 1 inside)."""
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        for i, r in enumerate(api["radii"]):
            if r < R_GAL * 0.5:
                assert api["gfd_poisson"][i] >= api["dtg"][i] - 0.1, (
                    f"Poisson should be >= GFD at r={r}"
                )

    def test_poisson_converges_to_gfd_at_large_r(self, config):
        """At large r, Poisson should nearly match GFD."""
        config_ext = GravisConfig(
            mass_model=MILKY_WAY_MODEL,
            max_radius=60.0,
            num_points=100,
            galactic_radius=30.0,
        )
        api = GravisEngine.rotation_curve(config_ext).run().to_api_response()
        for i, r in enumerate(api["radii"]):
            if r > 50.0:
                ratio = api["gfd_poisson"][i] / api["dtg"][i]
                assert ratio == pytest.approx(1.0, abs=0.02), (
                    f"Poisson should match GFD at r={r}"
                )

    def test_all_six_theories(self, config):
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        for key in ["newtonian", "dtg", "gfd_manifold", "gfd_poisson",
                     "mond", "cdm"]:
            assert key in api

    def test_poisson_smooth_no_bump(self, config):
        """Poisson curve should not have sudden drops."""
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        vals = api["gfd_poisson"]
        for i in range(5, len(vals) - 1):
            drop = vals[i] - vals[i+1]
            assert drop < 20, (
                f"Sudden drop of {drop:.1f} km/s at r={api['radii'][i]}"
            )
