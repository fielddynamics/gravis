"""
Tests for mass inference (inverse problem solver).
"""

import math
import pytest
from physics.inference import infer_mass
from physics.aqual import velocity as dtg_velocity


class TestMassInference:
    """Test the inverse problem: v -> M."""

    def test_round_trip_consistency(self):
        """
        Forward: M -> v via DTG.
        Inverse: v -> M via inference.
        Should recover original M.
        """
        M_original = 5e10
        r = 10.0
        # Forward
        v = dtg_velocity(r, M_original)
        # Inverse
        M_inferred = infer_mass(r, v)
        # Should match within 1%
        assert abs(M_inferred - M_original) / M_original < 0.01, \
            f"M_orig={M_original:.3e}, M_inferred={M_inferred:.3e}"

    def test_round_trip_various_masses(self):
        """Round-trip test across several mass scales."""
        for log_m in [8, 9, 10, 11, 12]:
            M = 10 ** log_m
            r = 10.0
            v = dtg_velocity(r, M)
            M_inf = infer_mass(r, v)
            assert abs(M_inf - M) / M < 0.01

    def test_milky_way_inference(self):
        """Infer mass from MW rotation: 230 km/s at 8 kpc."""
        M = infer_mass(8.0, 230.0)
        # Should be order of 10^10 M_sun
        assert 1e10 < M < 1e12

    def test_zero_velocity(self):
        assert infer_mass(10.0, 0) == 0.0

    def test_zero_radius(self):
        assert infer_mass(0, 230.0) == 0.0

    def test_accel_ratio_effect(self):
        M1 = infer_mass(8.0, 230.0, accel_ratio=1.0)
        M2 = infer_mass(8.0, 230.0, accel_ratio=2.0)
        assert M1 != M2
