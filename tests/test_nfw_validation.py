"""
NFW / CDM module validation.

These tests verify the Lambda-CDM components against published results:
  - Dutton & Maccio 2014 concentration-mass relation
  - Moster+ 2013 abundance matching (stellar-halo mass relation)
  - NFW profile mathematical properties
  - CDM fit quality relative to Newtonian-only
"""

import math
import pytest

from physics.nfw import (
    concentration, r200_kpc, nfw_enclosed_mass, nfw_velocity,
    cdm_velocity, abundance_matching, fit_halo,
)
from physics.newtonian import velocity as newtonian_velocity
from physics.mass_model import enclosed_mass
from data.galaxies import get_prediction_galaxies


class TestConcentrationMassRelation:
    """Verify the Dutton & Maccio 2014 c-M relation against their
    published values (Table 3, z=0, relaxed halos, Planck cosmology)."""

    def test_milky_way_scale_halo(self):
        """M200 = 10^12 M_sun should give c ~ 7-10."""
        c = concentration(1.0e12)
        assert 6.0 < c < 12.0, f"c(10^12) = {c:.2f}, expected ~7-10"

    def test_cluster_scale_halo(self):
        """M200 = 10^14 M_sun should give c ~ 4-7 (less concentrated)."""
        c = concentration(1.0e14)
        assert 3.0 < c < 9.0, f"c(10^14) = {c:.2f}, expected ~4-7"

    def test_dwarf_scale_halo(self):
        """M200 = 10^10 M_sun should give c ~ 12-20 (more concentrated)."""
        c = concentration(1.0e10)
        assert 10.0 < c < 25.0, f"c(10^10) = {c:.2f}, expected ~12-20"

    def test_concentration_decreases_with_mass(self):
        """c(M) is a decreasing function: larger halos are less concentrated."""
        masses = [1e10, 1e11, 1e12, 1e13, 1e14]
        concs = [concentration(m) for m in masses]
        for i in range(len(concs) - 1):
            assert concs[i] > concs[i + 1], \
                f"c({masses[i]:.0e})={concs[i]:.2f} <= " \
                f"c({masses[i+1]:.0e})={concs[i+1]:.2f}"


class TestAbundanceMatching:
    """Verify the Moster+2013 stellar-halo mass relation."""

    def test_milky_way_halo_mass(self):
        """MW baryonic mass ~7.5e10 M_sun should give M200 ~ 10^12-10^13.

        Note: we feed total baryonic mass (stellar + gas) into a relation
        calibrated on stellar mass.  The gas contribution (~1e10) pushes
        the inferred halo mass higher than the observational MW halo mass
        (~1-2e12).  This is a known systematic of abundance matching."""
        m200 = abundance_matching(7.5e10)
        assert 5e11 < m200 < 2e13, \
            f"MW M200 = {m200:.2e}, expected ~10^12-10^13"

    def test_dwarf_galaxy_halo_mass(self):
        """DDO 154 baryonic mass ~5e8 should give M200 ~ 10^10-10^11."""
        m200 = abundance_matching(5.0e8)
        assert 1e9 < m200 < 5e11, \
            f"DDO 154 M200 = {m200:.2e}, expected ~10^10"

    def test_massive_spiral_halo_mass(self):
        """UGC 2885 baryonic mass ~2.5e11 should give large halo."""
        m200 = abundance_matching(2.5e11)
        assert m200 > 1e12, \
            f"UGC 2885 M200 = {m200:.2e}, expected >10^12"

    def test_monotonically_increasing(self):
        """More baryonic mass -> more halo mass."""
        masses = [1e8, 1e9, 1e10, 1e11]
        halos = [abundance_matching(m) for m in masses]
        for i in range(len(halos) - 1):
            assert halos[i] < halos[i + 1], \
                f"Not monotonic: M200({masses[i]:.0e}) >= M200({masses[i+1]:.0e})"


class TestNFWProfile:
    """Verify mathematical properties of the NFW enclosed mass profile."""

    def test_enclosed_approaches_m200_at_r200(self):
        """NFW enclosed mass at r200 should be close to M200."""
        for m200 in [1e10, 1e12, 1e14]:
            r200 = r200_kpc(m200)
            m_enc = nfw_enclosed_mass(r200, m200)
            ratio = m_enc / m200
            # Should be close to 1.0 (exact at r200 by definition)
            assert ratio == pytest.approx(1.0, rel=0.02), \
                f"M200={m200:.0e}: M_enc(r200)/M200 = {ratio:.4f}"

    def test_enclosed_monotonically_increasing(self):
        """NFW enclosed mass must increase with radius."""
        m200 = 1.0e12
        prev = 0.0
        for r in [1, 5, 10, 50, 100, 200]:
            m = nfw_enclosed_mass(r, m200)
            assert m > prev, f"Not increasing at r={r}: {m} <= {prev}"
            prev = m


class TestCDMFitQuality:
    """The CDM halo fit should improve chi-squared vs Newtonian-only."""

    def test_cdm_reduces_chi2_vs_newtonian(self):
        """For galaxies with observations, CDM chi-squared should be
        lower than Newtonian-only chi-squared."""
        galaxies = [g for g in get_prediction_galaxies()
                    if g.get("mass_model") and g.get("observations")
                    and len(g["observations"]) >= 3]

        for galaxy in galaxies[:3]:  # Test first 3 for speed
            mm = galaxy["mass_model"]
            obs = galaxy["observations"]

            def mass_at_r(r):
                return enclosed_mass(r, mm)

            # Newtonian-only chi-squared
            chi2_newton = 0.0
            for o in obs:
                r, v, err = o["r"], o["v"], o.get("err", 5.0)
                m = enclosed_mass(r, mm)
                v_n = newtonian_velocity(r, m)
                sigma = max(err, 1.0)
                chi2_newton += ((v - v_n) / sigma) ** 2

            # CDM fit
            fit = fit_halo(obs, mass_at_r)
            assert fit is not None, f"{galaxy['id']}: fit_halo returned None"
            assert fit["chi2"] < chi2_newton, \
                f"{galaxy['id']}: CDM chi2={fit['chi2']:.1f} >= " \
                f"Newton chi2={chi2_newton:.1f}"
