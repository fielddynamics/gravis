"""
Tests for Dual Tetrad Gravity AQUAL solver.

Validates the constitutive law mu(x) = x/(1+x), the AQUAL field equation
solver, and the rotation velocity predictions against known limits.
"""

import math
import pytest
from physics.aqual import mu, solve_x, velocity
from physics.constants import A0


class TestConstitutiveLaw:
    """Test mu(x) = x / (1 + x)."""

    def test_mu_at_zero(self):
        assert mu(0) == 0.0

    def test_mu_at_one(self):
        assert abs(mu(1.0) - 0.5) < 1e-15

    def test_mu_deep_mond(self):
        """In deep MOND (x << 1), mu(x) ~ x."""
        x = 1e-6
        assert abs(mu(x) - x) / x < 1e-5

    def test_mu_newtonian(self):
        """In Newtonian regime (x >> 1), mu(x) ~ 1."""
        x = 1e6
        assert abs(mu(x) - 1.0) < 1e-5

    def test_mu_monotonic(self):
        """mu must be monotonically increasing."""
        prev = 0
        for x in [0.01, 0.1, 1, 10, 100, 1000]:
            val = mu(x)
            assert val > prev
            prev = val


class TestSolveX:
    """Test AQUAL field equation solver: mu(x)*x = y_N."""

    def test_solve_zero(self):
        assert solve_x(0) == 0.0

    def test_solve_tiny(self):
        assert solve_x(1e-40) == 0.0

    def test_solve_consistency(self):
        """Verify that mu(x)*x = y_N after solving."""
        for y_N in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            x = solve_x(y_N)
            # mu(x) * x should equal y_N
            reconstructed = mu(x) * x
            assert abs(reconstructed - y_N) / y_N < 1e-10, \
                f"y_N={y_N}: mu({x})*{x} = {reconstructed}"

    def test_deep_mond_limit(self):
        """In deep MOND: x ~ sqrt(y_N) when y_N << 1."""
        y_N = 1e-6
        x = solve_x(y_N)
        expected = math.sqrt(y_N)
        assert abs(x - expected) / expected < 0.01

    def test_newtonian_limit(self):
        """In Newtonian limit: x ~ y_N when y_N >> 1."""
        y_N = 1e6
        x = solve_x(y_N)
        assert abs(x - y_N) / y_N < 0.01


class TestDTGVelocity:
    """Test DTG rotation curve predictions."""

    def test_milky_way_solar_radius(self):
        """MW at ~8 kpc with ~5e10 M_sun enclosed should give ~200-240 km/s."""
        v = velocity(8.0, 5e10)
        assert 180 < v < 260, f"v = {v} km/s at 8 kpc"

    def test_velocity_increases_with_mass(self):
        """Higher enclosed mass => higher velocity at same radius."""
        v_low = velocity(10.0, 1e10)
        v_high = velocity(10.0, 1e11)
        assert v_high > v_low

    def test_zero_mass(self):
        assert velocity(10.0, 0) == 0.0

    def test_zero_radius(self):
        assert velocity(0, 1e10) == 0.0

    def test_dtg_vs_newtonian_deep_mond(self):
        """In deep MOND regime, DTG velocity > Newtonian velocity."""
        from physics.newtonian import velocity as newt_v
        # Low mass at large radius = deep MOND
        v_dtg = velocity(20.0, 1e8)
        v_newt = newt_v(20.0, 1e8)
        assert v_dtg > v_newt

    def test_dtg_approaches_newtonian_high_accel(self):
        """In high-acceleration regime, DTG ~ Newtonian."""
        from physics.newtonian import velocity as newt_v
        # High mass at small radius = Newtonian regime
        v_dtg = velocity(0.5, 1e12)
        v_newt = newt_v(0.5, 1e12)
        # Should agree within 5%
        assert abs(v_dtg - v_newt) / v_newt < 0.05

    def test_accel_ratio_effect(self):
        """Changing accel_ratio should shift the transition scale."""
        v_base = velocity(10.0, 5e10, accel_ratio=1.0)
        v_high = velocity(10.0, 5e10, accel_ratio=10.0)
        # Higher a0 means more Newtonian behavior at same radius
        # So velocity should be different
        assert v_base != v_high
