"""
Tests for physical constants module.

Validates that all constants match CODATA 2022 / IAU 2015 values
and that the topological acceleration scale a0 is correctly derived.
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, M_E, R_E, K_SIMPLEX, A0, verify_a0


class TestPhysicalConstants:
    """Verify fundamental constants are at expected values."""

    def test_gravitational_constant(self):
        assert abs(G - 6.67430e-11) / 6.67430e-11 < 1e-6

    def test_solar_mass(self):
        assert abs(M_SUN - 1.98892e30) / 1.98892e30 < 1e-4

    def test_kpc_to_meters(self):
        assert abs(KPC_TO_M - 3.0857e19) / 3.0857e19 < 1e-3

    def test_electron_mass(self):
        assert abs(M_E - 9.1093837139e-31) / 9.1093837139e-31 < 1e-10

    def test_classical_electron_radius(self):
        assert abs(R_E - 2.8179403205e-15) / 2.8179403205e-15 < 1e-10

    def test_simplex_number(self):
        assert K_SIMPLEX == 4


class TestTopologicalAcceleration:
    """Verify a0 derivation from topological parameters."""

    def test_a0_formula(self):
        """a0 = k^2 * G * m_e / r_e^2"""
        expected = K_SIMPLEX**2 * G * M_E / (R_E**2)
        assert abs(A0 - expected) / expected < 1e-12

    def test_a0_order_of_magnitude(self):
        """a0 should be approximately 1.2e-10 m/s^2."""
        assert 1.0e-10 < A0 < 1.5e-10

    def test_verify_a0_function(self):
        ok, val = verify_a0()
        assert ok is True
        assert val == A0

    def test_a0_no_free_parameters(self):
        """
        Verify a0 is computed entirely from fundamental constants
        and the simplex number k=4 (topological, not fitted).
        """
        a0_check = 16 * G * M_E / (R_E * R_E)
        assert abs(A0 - a0_check) < 1e-20
