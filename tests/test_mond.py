"""
Tests for classical MOND solver (Bekenstein & Milgrom interpolating function).

Validates mu(x) = x/sqrt(1+x^2) and the corresponding field equation solver.
"""

import math
import pytest
from physics.mond import mu_mond, solve_x, velocity


class TestMONDInterpolatingFunction:
    """Test mu(x) = x / sqrt(1 + x^2)."""

    def test_mu_at_zero(self):
        assert mu_mond(0) == 0.0

    def test_mu_at_one(self):
        expected = 1.0 / math.sqrt(2.0)
        assert abs(mu_mond(1.0) - expected) < 1e-15

    def test_mu_deep_mond(self):
        """In deep MOND (x << 1), mu(x) ~ x."""
        x = 1e-6
        assert abs(mu_mond(x) - x) / x < 1e-5

    def test_mu_newtonian(self):
        """In Newtonian regime (x >> 1), mu(x) ~ 1."""
        x = 1e6
        assert abs(mu_mond(x) - 1.0) < 1e-5


class TestMONDSolveX:
    """Test MOND field equation solver."""

    def test_solve_zero(self):
        assert solve_x(0) == 0.0

    def test_solve_consistency(self):
        """Verify that mu(x)*x = y_N after solving."""
        for y_N in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            x = solve_x(y_N)
            reconstructed = mu_mond(x) * x
            assert abs(reconstructed - y_N) / y_N < 1e-8, \
                f"y_N={y_N}: mu_MOND({x})*{x} = {reconstructed}"


class TestMONDVelocity:
    """Test classical MOND rotation curve predictions."""

    def test_milky_way_solar_radius(self):
        v = velocity(8.0, 5e10)
        assert 180 < v < 260

    def test_zero_mass(self):
        assert velocity(10.0, 0) == 0.0

    def test_zero_radius(self):
        assert velocity(0, 1e10) == 0.0

    def test_mond_vs_newtonian_deep_mond(self):
        """In deep MOND, MOND velocity > Newtonian."""
        from physics.newtonian import velocity as newt_v
        v_mond = velocity(20.0, 1e8)
        v_newt = newt_v(20.0, 1e8)
        assert v_mond > v_newt

    def test_mond_vs_dtg_same_ballpark(self):
        """MOND and DTG should give similar (not identical) predictions."""
        from physics.aqual import velocity as dtg_v
        v_mond = velocity(10.0, 5e10)
        v_dtg = dtg_v(10.0, 5e10)
        # Should be within 20% of each other
        diff = abs(v_mond - v_dtg) / max(v_mond, v_dtg)
        assert diff < 0.20
