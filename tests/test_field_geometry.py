"""
Tests for topological field geometry prediction.

The solve_field_geometry function derives the Field Origin (R_t) and
Field Horizon (R_env) from a galaxy's baryonic mass model using the
throat condition y_N = (4/13)(9/10) = 18/65. These tests verify the
prediction against catalog values for well-constrained galaxies and
examine behavior across the full mass spectrum.
"""

import math
import pytest

from physics.constants import A0, K_SIMPLEX
from physics.services.rotation.inference import (
    solve_field_geometry,
    THROAT_YN,
    THROAT_FRAC,
)
from data.galaxies import get_galaxy_by_id


# -----------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------

def _geometry_for_galaxy(galaxy_id):
    """Run solve_field_geometry for a catalog galaxy (published masses)."""
    gal = get_galaxy_by_id(galaxy_id)
    mm = gal["mass_model"]
    b = mm.get("bulge", {})
    d = mm.get("disk", {})
    g = mm.get("gas", {})
    a0_eff = A0 * gal.get("accel", 1.0)
    fg = solve_field_geometry(
        b.get("M", 0), b.get("a", 0),
        d.get("M", 0), d.get("Rd", 0),
        g.get("M", 0), g.get("Rd", 0),
        a0_eff,
    )
    return fg, gal


# -----------------------------------------------------------------
# MODULE CONSTANTS
# -----------------------------------------------------------------

class TestThroatConstants:
    """Verify the topological constants are correctly defined."""

    def test_throat_yn_value(self):
        """THROAT_YN = (4/13)(9/10) = 18/65."""
        assert abs(THROAT_YN - 18.0 / 65.0) < 1e-12

    def test_throat_frac_value(self):
        """THROAT_FRAC = 0.30 (from coupled tetrahedral face model)."""
        assert THROAT_FRAC == 0.30

    def test_structural_fraction_origin(self):
        """4/13 comes from 4 structural channels in 1+3+9=13 closure."""
        assert abs((4.0 / 13.0) * (9.0 / 10.0) - THROAT_YN) < 1e-12


# -----------------------------------------------------------------
# MILKY WAY (best-constrained galaxy)
# -----------------------------------------------------------------

class TestMilkyWayGeometry:
    """
    Milky Way: the gold-standard test.  Published masses give R_t
    within ~5% of catalog; fitted masses (post-inference) within ~1%.
    """

    def test_throat_exists(self):
        fg, _ = _geometry_for_galaxy("milky_way")
        assert fg["throat_radius_kpc"] is not None
        assert fg["envelope_radius_kpc"] is not None

    def test_throat_near_catalog(self):
        """Published-mass R_t should be within 6% of catalog 18.0 kpc."""
        fg, gal = _geometry_for_galaxy("milky_way")
        rt_cat = THROAT_FRAC * gal["galactic_radius"]
        err = abs(fg["throat_radius_kpc"] - rt_cat) / rt_cat
        assert err < 0.06, (
            "R_t predicted %.2f, catalog %.1f, error %.1f%%"
            % (fg["throat_radius_kpc"], rt_cat, err * 100)
        )

    def test_envelope_near_catalog(self):
        """Published-mass R_env should be within 6% of catalog 60.0 kpc."""
        fg, gal = _geometry_for_galaxy("milky_way")
        err = abs(fg["envelope_radius_kpc"] - gal["galactic_radius"]) / gal["galactic_radius"]
        assert err < 0.06

    def test_yn_at_throat_matches_target(self):
        """y_N at the solved R_t must equal THROAT_YN exactly."""
        fg, _ = _geometry_for_galaxy("milky_way")
        assert abs(fg["yN_at_throat"] - THROAT_YN) < 1e-4

    def test_envelope_equals_throat_over_frac(self):
        """R_env ~ R_t / 0.30 for 3-cycle galaxies (emerges, not forced)."""
        fg, _ = _geometry_for_galaxy("milky_way")
        expected = fg["throat_radius_kpc"] / THROAT_FRAC
        # For 3-cycle, R_t and R_env are solved independently.
        # The ratio emerges from the mass profile shape and is
        # typically within 1% of 0.30 but not exact (it depends
        # on the mass concentration). Allow 1% tolerance.
        assert abs(fg["envelope_radius_kpc"] - expected) / expected < 0.01


# -----------------------------------------------------------------
# OTHER WELL-CONSTRAINED SPIRALS
# -----------------------------------------------------------------

class TestSpiralGeometry:
    """Galaxies where published mass models are well-constrained."""

    @pytest.mark.parametrize("galaxy_id, tol", [
        # Published masses are systematically ~10% below fitted masses
        # (photometric M/L uncertainty), so R_t from published masses
        # is smaller than catalog.  Post-inference these tighten to <5%.
        ("ngc2403", 0.15),    # 1.3e10 Msun
        ("ngc6503", 0.06),    # 8.7e9 Msun
    ])
    def test_throat_near_catalog(self, galaxy_id, tol):
        fg, gal = _geometry_for_galaxy(galaxy_id)
        rt_cat = THROAT_FRAC * gal["galactic_radius"]
        assert fg["throat_radius_kpc"] is not None
        err = abs(fg["throat_radius_kpc"] - rt_cat) / rt_cat
        assert err < tol, (
            "%s: R_t predicted %.2f, catalog %.2f, error %.1f%%"
            % (galaxy_id, fg["throat_radius_kpc"], rt_cat, err * 100)
        )


# -----------------------------------------------------------------
# DWARF GALAXIES (deep-field regime)
# -----------------------------------------------------------------

class TestDwarfGeometry:
    """
    Dwarf galaxies where the total baryonic mass is so low that g_N
    never reaches a_0 * THROAT_YN at any radius.

    These systems are fully in the GFD deep-field regime: y_N < 0.07
    everywhere.  The current throat condition y_N = 18/65 has no
    solution because the enclosed mass is insufficient to produce
    the required Newtonian acceleration at any radius.

    IMPORTANT: These galaxies are still gravitational vortices.  The
    field origin and horizon exist as topological features.  The
    GFD-sigma correction is applied to them and produces accurate
    rotation curves.  Returning None here indicates that the y_N =
    18/65 condition (derived for galaxies with a Newtonian-to-deep-
    field transition) does not apply to systems that are ENTIRELY
    in the deep-field.  A generalized throat condition for fully
    deep-field systems remains an open question.
    """

    @pytest.mark.parametrize("galaxy_id", [
        "ddo154",
        "ic2574",
        "ngc3109",
    ])
    def test_peak_yn_below_threshold(self, galaxy_id):
        """
        Verify that the peak y_N for each dwarf is genuinely below
        THROAT_YN, confirming no crossing exists.
        """
        gal = get_galaxy_by_id(galaxy_id)
        mm = gal["mass_model"]
        b = mm.get("bulge", {})
        d = mm.get("disk", {})
        g = mm.get("gas", {})
        a0 = A0 * gal.get("accel", 1.0)

        def _enc(r):
            enc = 0.0
            if b.get("M", 0) > 0 and b.get("a", 0) > 0:
                enc += b["M"] * r * r / ((r + b["a"]) ** 2)
            if d.get("M", 0) > 0 and d.get("Rd", 0) > 0:
                x = r / d["Rd"]
                enc += d["M"] * (1 - (1 + x) * math.exp(-x))
            if g.get("M", 0) > 0 and g.get("Rd", 0) > 0:
                x = r / g["Rd"]
                enc += g["M"] * (1 - (1 + x) * math.exp(-x))
            return enc

        from physics.equations import G, M_SUN, KPC_TO_M
        peak = 0
        for ri in range(1, 5000):
            r = ri * 0.005
            enc = _enc(r)
            r_m = r * KPC_TO_M
            if r_m > 0 and enc > 0:
                yn = G * enc * M_SUN / (r_m * r_m * a0)
                if yn > peak:
                    peak = yn

        assert peak < THROAT_YN, (
            "%s: peak y_N = %.4f exceeds THROAT_YN = %.4f"
            % (galaxy_id, peak, THROAT_YN)
        )

    @pytest.mark.parametrize("galaxy_id", [
        "ddo154",
        "ic2574",
        "ngc3109",
    ])
    def test_2cycle_geometry(self, galaxy_id):
        """2-cycle galaxies: R_env from horizon, R_t = 0.30 * R_env."""
        fg, _ = _geometry_for_galaxy(galaxy_id)
        # Every vortex has a throat and horizon.
        # 2-cycle galaxies get R_t from the topological constant.
        assert fg["envelope_radius_kpc"] is not None
        assert fg["envelope_radius_kpc"] > 0
        assert fg["throat_radius_kpc"] is not None
        assert fg["throat_radius_kpc"] > 0
        assert fg["cycle"] == 2
        # R_t = 0.30 * R_env exactly for 2-cycle
        assert abs(fg["throat_fraction"] - THROAT_FRAC) < 1e-6
        # yN at throat is below the 3-cycle threshold
        assert fg["yN_at_throat"] < THROAT_YN


# -----------------------------------------------------------------
# FITTED MASSES (post-inference, higher accuracy)
# -----------------------------------------------------------------

class TestFittedMassGeometry:
    """
    After Stage 3 inference adjusts the stellar masses, the throat
    prediction tightens from ~5% to ~1% for the Milky Way.
    """

    def test_mw_fitted_within_2pct(self):
        """MW fitted masses: R_t within 2% of catalog."""
        fg = solve_field_geometry(
            1.663e10, 0.60,   # bulge (fitted)
            5.068e10, 2.20,   # disk  (fitted)
            1.50e10,  7.00,   # gas   (fixed)
            A0,
        )
        rt_cat = 18.0
        assert fg["throat_radius_kpc"] is not None
        err = abs(fg["throat_radius_kpc"] - rt_cat) / rt_cat
        assert err < 0.02, (
            "Fitted R_t = %.4f, catalog = %.1f, error = %.1f%%"
            % (fg["throat_radius_kpc"], rt_cat, err * 100)
        )

    def test_mw_fitted_envelope_within_2pct(self):
        """MW fitted masses: R_env within 2% of catalog."""
        fg = solve_field_geometry(
            1.663e10, 0.60,
            5.068e10, 2.20,
            1.50e10,  7.00,
            A0,
        )
        renv_cat = 60.0
        err = abs(fg["envelope_radius_kpc"] - renv_cat) / renv_cat
        assert err < 0.02


# -----------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_zero_mass_returns_none(self):
        """A massless system has no throat."""
        fg = solve_field_geometry(0, 0, 0, 0, 0, 0, A0)
        assert fg["throat_radius_kpc"] is None

    def test_single_component_bulge_only(self):
        """A galaxy with only a bulge should still produce a throat."""
        fg = solve_field_geometry(
            5e10, 1.0,    # bulge
            0, 0,         # no disk
            0, 0,         # no gas
            A0,
        )
        assert fg["throat_radius_kpc"] is not None
        assert fg["throat_radius_kpc"] > 0
        assert abs(fg["yN_at_throat"] - THROAT_YN) < 1e-4

    def test_single_component_disk_only(self):
        """A pure disk galaxy should still produce a throat."""
        fg = solve_field_geometry(
            0, 0,           # no bulge
            5e10, 3.0,      # disk
            0, 0,           # no gas
            A0,
        )
        assert fg["throat_radius_kpc"] is not None
        assert fg["throat_radius_kpc"] > 0

    def test_more_mass_means_larger_throat(self):
        """Doubling total mass should push R_t outward."""
        fg1 = solve_field_geometry(1e10, 0.5, 3e10, 2.0, 1e10, 5.0, A0)
        fg2 = solve_field_geometry(2e10, 0.5, 6e10, 2.0, 2e10, 5.0, A0)
        assert fg2["throat_radius_kpc"] > fg1["throat_radius_kpc"]

    def test_accel_ratio_effect(self):
        """Higher a0 should shrink R_t (easier to reach threshold)."""
        fg1 = solve_field_geometry(1.5e10, 0.6, 4.57e10, 2.2, 1.5e10, 7.0, A0 * 1.0)
        fg2 = solve_field_geometry(1.5e10, 0.6, 4.57e10, 2.2, 1.5e10, 7.0, A0 * 2.0)
        # Higher a0 means y_N = g_N / a0 is smaller, so you need
        # to go further in before y_N reaches the threshold.
        # Wait: higher a0 means the threshold g_N = a0 * THROAT_YN is
        # higher, so you need MORE acceleration -> smaller radius.
        # But g_N decreases outward, so smaller R_t.
        # Actually: g_N is fixed by mass. y_N = g_N / a0.
        # Higher a0 -> lower y_N at every radius -> need to go
        # further IN to reach THROAT_YN -> SMALLER R_t.
        assert fg2["throat_radius_kpc"] < fg1["throat_radius_kpc"]
