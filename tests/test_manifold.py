"""
Tests for the GFD Manifold smooth vortex-distributed weighting.

Verifies:
  1. Topological constant ALPHA_THROAT.
  2. Core weight function W(r): smooth distribution over full galaxy,
     boundary values, monotonicity, C1 continuity.
  3. Manifold velocity: smooth boost tapering from center to horizon.
  4. Traced callable (gfd_manifold_eq): intermediates and guards.
  5. Engine integration: gfd_manifold produces a smooth, elevated curve
     that tapers toward GFD at the galactic horizon.
"""

import math
import pytest

from physics.manifold import (
    ALPHA_THROAT,
    throat_radius,
    core_weight,
    velocity as manifold_velocity,
)
from physics.equations import gfd_manifold_eq
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


# -----------------------------------------------------------------------
# Core weight: distributed over full galaxy
# -----------------------------------------------------------------------
class TestCoreWeight:

    def test_at_center(self):
        assert core_weight(0.0, R_GAL) == 1.0

    def test_at_horizon(self):
        assert core_weight(R_GAL, R_GAL) == pytest.approx(0.0, abs=1e-15)

    def test_outside_horizon(self):
        assert core_weight(35.0, R_GAL) == 0.0

    def test_at_throat(self):
        """At the throat (30% of R), W should be approximately 0.784."""
        r_throat = 0.30 * R_GAL
        w = core_weight(r_throat, R_GAL)
        assert w == pytest.approx(0.784, abs=0.001)

    def test_at_half_galaxy(self):
        """At half the galaxy, W should be exactly 0.5."""
        w = core_weight(R_GAL / 2.0, R_GAL)
        assert w == pytest.approx(0.5, abs=1e-10)

    def test_monotonically_decreasing(self):
        prev = 1.0
        for r in [0.5, 2, 5, 9, 15, 20, 25, 29, 30]:
            w = core_weight(r, R_GAL)
            assert w <= prev, f"W not decreasing at r={r}"
            prev = w

    def test_values_between_0_and_1(self):
        for r in [0, 0.1, 3, 9, 15, 20, 25, 29.9, 30, 35]:
            w = core_weight(r, R_GAL)
            assert 0.0 <= w <= 1.0

    def test_nonzero_at_20_kpc(self):
        """The weight is still significant at 20 kpc (2/3 of galaxy)."""
        w = core_weight(20.0, R_GAL)
        # x = 20/30 = 0.667, W = 1 - 3*(0.444) + 2*(0.296) = 0.259
        assert w > 0.2

    def test_c1_continuous_at_horizon(self):
        """Derivative at the horizon should be zero (C1 continuous)."""
        eps = 1e-6
        w_near = core_weight(R_GAL - eps, R_GAL)
        w_at = core_weight(R_GAL, R_GAL)
        deriv = (w_at - w_near) / eps
        assert abs(deriv) < 0.01


# -----------------------------------------------------------------------
# Manifold velocity
# -----------------------------------------------------------------------
class TestManifoldVelocity:

    def test_zero_radius(self):
        assert manifold_velocity(0.0, 1e10, R_GAL) == 0.0

    def test_zero_mass(self):
        assert manifold_velocity(5.0, 0.0, R_GAL) == 0.0

    def test_at_horizon_matches_gfd(self):
        """At the horizon, W=0, so manifold = GFD."""
        r = R_GAL
        m = 7.5e10
        v_man = manifold_velocity(r, m, R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        assert v_man == pytest.approx(v_gfd, rel=1e-8)

    def test_beyond_horizon_matches_gfd(self):
        r = 35.0
        m = 7.5e10
        v_man = manifold_velocity(r, m, R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        assert v_man == pytest.approx(v_gfd, rel=1e-8)

    def test_boosted_at_small_r(self):
        """Near center, boost ~ sqrt(2)."""
        r = 0.3
        m = 1e9
        v_man = manifold_velocity(r, m, R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        assert v_man == pytest.approx(v_gfd * math.sqrt(2), rel=0.02)

    def test_boosted_at_15_kpc(self):
        """At 15 kpc (half the galaxy), W=0.5, boost = sqrt(1.5)."""
        r = 15.0
        m = 6e10
        v_man = manifold_velocity(r, m, R_GAL)
        v_gfd = ref_gfd_velocity(r, m)
        assert v_man == pytest.approx(v_gfd * math.sqrt(1.5), rel=0.001)

    def test_smooth_no_kink(self):
        """Velocity should increase smoothly, no sudden drop or bump."""
        m = 5e10
        velocities = [manifold_velocity(r, m, R_GAL)
                      for r in [5, 6, 7, 8, 9, 10, 11, 12]]
        # Check no sudden drops (each step change < 20 km/s)
        for i in range(1, len(velocities)):
            assert abs(velocities[i] - velocities[i-1]) < 20


# -----------------------------------------------------------------------
# Traced callable
# -----------------------------------------------------------------------
class TestGfdManifoldEq:

    def test_intermediates_keys(self):
        v, ints = gfd_manifold_eq(5.0, 5e10, galactic_radius_kpc=R_GAL)
        expected = {"W", "galactic_radius", "g_base", "g_tot", "g_N", "y_N", "x"}
        assert set(ints.keys()) == expected

    def test_weight_at_midgalaxy(self):
        _, ints = gfd_manifold_eq(15.0, 6e10, galactic_radius_kpc=R_GAL)
        assert ints["W"] == pytest.approx(0.5, abs=0.01)
        assert ints["g_tot"] > ints["g_base"]

    def test_weight_at_horizon(self):
        _, ints = gfd_manifold_eq(30.0, 7.5e10, galactic_radius_kpc=R_GAL)
        assert ints["W"] == pytest.approx(0.0, abs=1e-10)

    def test_velocity_matches_module(self):
        r, m = 10.0, 5e10
        v_eq, _ = gfd_manifold_eq(r, m, galactic_radius_kpc=R_GAL)
        v_mod = manifold_velocity(r, m, R_GAL)
        assert v_eq == pytest.approx(v_mod, rel=1e-10)

    def test_zero_guards(self):
        v, _ = gfd_manifold_eq(0.0, 5e10, galactic_radius_kpc=R_GAL)
        assert v == 0.0
        v, _ = gfd_manifold_eq(5.0, 0.0, galactic_radius_kpc=R_GAL)
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

    def test_gfd_manifold_in_api(self, config):
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        assert "gfd_manifold" in api
        assert len(api["gfd_manifold"]) == 100

    def test_manifold_above_gfd_everywhere_inside(self, config):
        """Manifold should be above GFD at every point inside the horizon."""
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        for i, r in enumerate(api["radii"]):
            if r < R_GAL * 0.95:
                assert api["gfd_manifold"][i] >= api["dtg"][i] - 0.1, (
                    f"Manifold should be >= GFD at r={r}"
                )

    def test_manifold_smooth_no_bump(self, config):
        """The manifold curve should not have a sudden drop or bump."""
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        vals = api["gfd_manifold"]
        # After the initial rise, check no sudden velocity drops > 15 km/s
        for i in range(5, len(vals) - 1):
            drop = vals[i] - vals[i+1]
            assert drop < 15, (
                f"Sudden drop of {drop:.1f} km/s at r={api['radii'][i]}"
            )

    def test_all_six_theories(self, config):
        api = GravisEngine.rotation_curve(config).run().to_api_response()
        for key in ["newtonian", "dtg", "gfd_manifold", "gfd_poisson",
                     "mond", "cdm"]:
            assert key in api

    def test_galactic_radius_defaults(self):
        config = GravisConfig(MILKY_WAY_MODEL, 25.0, 50)
        assert config.galactic_radius == 25.0
