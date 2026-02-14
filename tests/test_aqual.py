"""
Tests for Gravity Field Dynamics covariant field equation solver.

Validates the field equation x^2/(1+x) = y_N, the analytic solver,
and the rotation velocity predictions against known limits.

The field equation derives from the Lagrangian F(y) = y/2 - sqrt(y) + ln(1+sqrt(y)),
which is uniquely determined by the dual tetrad topology.
"""

import math
import pytest
from physics.aqual import solve_x, velocity
from physics.constants import A0


class TestFieldEquation:
    """Test the field equation x^2 / (1 + x) = y_N."""

    def test_field_eq_at_zero(self):
        """x=0 => x^2/(1+x) = 0."""
        assert 0.0 * 0.0 / (1.0 + 0.0) == 0.0

    def test_field_eq_at_one(self):
        """x=1 => x^2/(1+x) = 1/2."""
        val = 1.0 * 1.0 / (1.0 + 1.0)
        assert abs(val - 0.5) < 1e-15

    def test_field_eq_deep_mond_limit(self):
        """When x << 1: x^2/(1+x) ~ x^2, so y_N ~ x^2, i.e. x ~ sqrt(y_N)."""
        x = 1e-6
        y_N = x * x / (1.0 + x)
        assert abs(y_N - x * x) / (x * x) < 1e-5

    def test_field_eq_newtonian_limit(self):
        """When x >> 1: x^2/(1+x) ~ x, so y_N ~ x, i.e. g ~ g_N."""
        x = 1e6
        y_N = x * x / (1.0 + x)
        assert abs(y_N - x) / x < 1e-5

    def test_field_eq_monotonic(self):
        """y_N = x^2/(1+x) must be monotonically increasing in x."""
        prev = 0
        for x in [0.01, 0.1, 1, 10, 100, 1000]:
            val = x * x / (1.0 + x)
            assert val > prev
            prev = val


class TestSolveX:
    """Test the analytic solver for x^2/(1+x) = y_N."""

    def test_solve_zero(self):
        assert solve_x(0) == 0.0

    def test_solve_tiny(self):
        assert solve_x(1e-40) == 0.0

    def test_solve_consistency(self):
        """Verify that x^2/(1+x) = y_N after solving."""
        for y_N in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            x = solve_x(y_N)
            # Evaluate the field equation: should recover y_N
            reconstructed = x * x / (1.0 + x)
            assert abs(reconstructed - y_N) / y_N < 1e-10, \
                f"y_N={y_N}: x^2/(1+x) at x={x} = {reconstructed}"

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


class TestGFDVelocity:
    """Test GFD rotation curve predictions."""

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

    def test_gfd_vs_newtonian_deep_mond(self):
        """In deep MOND regime, GFD velocity > Newtonian velocity."""
        from physics.newtonian import velocity as newt_v
        # Low mass at large radius = deep MOND
        v_gfd = velocity(20.0, 1e8)
        v_newt = newt_v(20.0, 1e8)
        assert v_gfd > v_newt

    def test_gfd_approaches_newtonian_high_accel(self):
        """In high-acceleration regime, GFD ~ Newtonian."""
        from physics.newtonian import velocity as newt_v
        # High mass at small radius = Newtonian regime
        v_gfd = velocity(0.5, 1e12)
        v_newt = newt_v(0.5, 1e12)
        # Should agree within 5%
        assert abs(v_gfd - v_newt) / v_newt < 0.05

    def test_accel_ratio_effect(self):
        """Changing accel_ratio should shift the transition scale."""
        v_base = velocity(10.0, 5e10, accel_ratio=1.0)
        v_high = velocity(10.0, 5e10, accel_ratio=10.0)
        # Higher a0 means more Newtonian behavior at same radius
        # So velocity should be different
        assert v_base != v_high
