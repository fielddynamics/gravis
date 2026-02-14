"""
Tests for distributed mass model (Hernquist bulge + exponential disks).

Validates enclosed mass calculations match analytic expectations and
the methodology in run_all_covariant_predictions.py.
"""

import math
import pytest
from physics.mass_model import enclosed_mass, total_mass


# Milky Way mass model (enterprise reference)
MW_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}


class TestEnclosedMass:
    """Test M(<r) computation for various mass distributions."""

    def test_zero_radius(self):
        assert enclosed_mass(0, MW_MODEL) == 0.0

    def test_negative_radius(self):
        assert enclosed_mass(-1, MW_MODEL) == 0.0

    def test_monotonically_increasing(self):
        """Enclosed mass must monotonically increase with radius."""
        prev = 0
        for r in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
            m = enclosed_mass(r, MW_MODEL)
            assert m >= prev, f"M(<{r}) = {m} < M(<prev) = {prev}"
            prev = m

    def test_approaches_total_at_large_radius(self):
        """At very large r, enclosed mass should approach total mass."""
        m_total = total_mass(MW_MODEL)
        m_at_100 = enclosed_mass(100, MW_MODEL)
        # At 100 kpc, should have >99% of total mass enclosed
        assert m_at_100 / m_total > 0.99

    def test_milky_way_at_8kpc(self):
        """At Solar radius (~8 kpc), MW should have significant enclosed mass."""
        m = enclosed_mass(8.0, MW_MODEL)
        # Should be between 3e10 and 7e10 M_sun at 8 kpc
        assert 3e10 < m < 7e10

    def test_milky_way_at_25kpc(self):
        """At 25 kpc, most of the MW mass should be enclosed."""
        m = enclosed_mass(25.0, MW_MODEL)
        m_total = total_mass(MW_MODEL)
        assert m / m_total > 0.90


class TestHernquistBulge:
    """Test Hernquist bulge profile in isolation."""

    def test_hernquist_half_mass(self):
        """Hernquist half-mass radius is at r = a."""
        model = {"bulge": {"M": 1e10, "a": 1.0}}
        m_at_a = enclosed_mass(1.0, model)
        # M(<a) = M * a^2 / (a+a)^2 = M/4 for Hernquist
        expected = 1e10 * 1.0 / (2.0 * 2.0)
        assert abs(m_at_a - expected) / expected < 1e-10

    def test_hernquist_asymptote(self):
        """At large r, Hernquist M(<r) -> M_total."""
        model = {"bulge": {"M": 1e10, "a": 1.0}}
        m = enclosed_mass(1000.0, model)
        assert abs(m - 1e10) / 1e10 < 0.01


class TestExponentialDisk:
    """Test exponential disk profile in isolation."""

    def test_disk_at_zero(self):
        model = {"disk": {"M": 5e10, "Rd": 2.5}}
        assert enclosed_mass(0, model) == 0.0

    def test_disk_at_large_radius(self):
        """At r >> Rd, disk M(<r) -> M_total."""
        model = {"disk": {"M": 5e10, "Rd": 2.5}}
        m = enclosed_mass(100.0, model)
        assert abs(m - 5e10) / 5e10 < 0.01

    def test_disk_at_scale_length(self):
        """At r = Rd: M(<Rd) = M * [1 - 2*exp(-1)] ~ 0.264 * M."""
        model = {"disk": {"M": 1e10, "Rd": 3.0}}
        m = enclosed_mass(3.0, model)
        expected_fraction = 1.0 - 2.0 * math.exp(-1.0)
        expected = 1e10 * expected_fraction
        assert abs(m - expected) / expected < 1e-10


class TestGasDisk:
    """Test gas disk (same functional form as stellar disk)."""

    def test_gas_extended(self):
        """Gas disk with Rd=7 kpc should have significant mass at 15 kpc."""
        model = {"gas": {"M": 3e9, "Rd": 7.0}}
        m = enclosed_mass(15.0, model)
        # At r = 2.14 * Rd, should have majority enclosed
        assert m > 0.5 * 3e9


class TestTotalMass:
    """Test total_mass helper."""

    def test_milky_way_total(self):
        assert total_mass(MW_MODEL) == 1.5e10 + 5.0e10 + 1.0e10

    def test_empty_model(self):
        assert total_mass({}) == 0.0

    def test_partial_model(self):
        model = {"disk": {"M": 5e10, "Rd": 2.5}}
        assert total_mass(model) == 5e10
