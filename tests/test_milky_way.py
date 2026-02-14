"""
Enterprise-grade Milky Way rotation curve tests.

Uses the Milky Way as a reference galaxy to cross-validate all three
gravitational theories (Newtonian, DTG, MOND) with and without the
distributed baryonic mass model.

Reference values computed from the physics engine itself and validated
against analytic limits and observational constraints.

Milky Way mass model (independently measured, NOT fitted):
  Bulge: 1.5e10 M_sun, a=0.6 kpc  (Bland-Hawthorn & Gerhard 2016)
  Disk:  5.0e10 M_sun, Rd=2.5 kpc  (McMillan 2017)
  Gas:   1.0e10 M_sun, Rd=5.0 kpc  (Kalberla & Kerp 2009)
  Total: 7.5e10 M_sun

Key observational constraint:
  v_c(R_sun ~ 8 kpc) = 230 +/- 10 km/s (Eilers+2019, Mroz+2019)
"""

import math
import pytest
from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.mass_model import enclosed_mass, total_mass
from physics.newtonian import velocity as newt_v
from physics.aqual import velocity as dtg_v, solve_x as dtg_solve_x, mu as dtg_mu
from physics.mond import velocity as mond_v, solve_x as mond_solve_x, mu_mond


# =====================================================================
# MILKY WAY REFERENCE DATA
# =====================================================================

MW_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}
MW_TOTAL = 7.5e10  # M_sun

# Enclosed mass at key radii (pre-computed reference values)
MW_ENCLOSED = {
    1:  9.112203e9,
    2:  1.905165e10,
    5:  4.430003e10,
    8:  5.917062e10,
    10: 6.471098e10,
    15: 7.100930e10,
    20: 7.307220e10,
    25: 7.387587e10,
}

# Observational constraint: Eilers+2019, Mroz+2019
OBS_V_AT_8KPC = 230.0  # km/s +/- 10
OBS_V_TOLERANCE = 15.0  # generous 1-sigma for model comparison


# =====================================================================
# NEWTONIAN: 10 MILKY WAY TESTS
# =====================================================================

class TestNewtonianMilkyWay:
    """
    Newtonian gravity applied to the Milky Way.
    v_c(r) = sqrt(G * M(<r) / r)

    10 tests covering point mass, distributed mass, Keplerian scaling,
    and cross-checks against analytic formulas.
    """

    def test_01_point_mass_at_8kpc(self):
        """Point mass (total MW): v at 8 kpc should be ~201 km/s."""
        v = newt_v(8.0, MW_TOTAL)
        assert abs(v - 200.83) < 0.5

    def test_02_distributed_mass_at_8kpc(self):
        """Distributed mass: v at 8 kpc should be ~178 km/s."""
        m_enc = enclosed_mass(8.0, MW_MODEL)
        v = newt_v(8.0, m_enc)
        assert abs(v - 178.38) < 0.5

    def test_03_distributed_lower_than_point_mass(self):
        """Distributed mass v < point mass v at all MW radii (mass not yet enclosed)."""
        for r in [2, 5, 8, 10, 15, 20]:
            m_enc = enclosed_mass(r, MW_MODEL)
            v_dist = newt_v(r, m_enc)
            v_point = newt_v(r, MW_TOTAL)
            assert v_dist < v_point, \
                f"r={r}: distributed {v_dist:.1f} >= point {v_point:.1f}"

    def test_04_converge_at_large_radius(self):
        """At 25 kpc (98.5% enclosed), distributed ~ point mass within 1%."""
        m_enc = enclosed_mass(25.0, MW_MODEL)
        v_dist = newt_v(25.0, m_enc)
        v_point = newt_v(25.0, MW_TOTAL)
        assert abs(v_dist - v_point) / v_point < 0.01

    def test_05_keplerian_decline_point_mass(self):
        """Point mass: v(r) ~ 1/sqrt(r), so v(20)/v(40) = sqrt(2)."""
        v20 = newt_v(20.0, MW_TOTAL)
        v40 = newt_v(40.0, MW_TOTAL)
        ratio = v20 / v40
        assert abs(ratio - math.sqrt(2)) < 1e-4

    def test_06_analytic_formula_check(self):
        """Cross-check: v = sqrt(G*M/r) computed manually at 10 kpc."""
        r_m = 10.0 * KPC_TO_M
        M_kg = MW_TOTAL * M_SUN
        v_expected_ms = math.sqrt(G * M_kg / r_m)
        v_expected_kms = v_expected_ms / 1000.0
        v_computed = newt_v(10.0, MW_TOTAL)
        assert abs(v_computed - v_expected_kms) < 1e-8

    def test_07_distributed_peak_velocity(self):
        """Distributed Newtonian curve should peak around 2-3 kpc for MW."""
        velocities = {}
        for r_10 in range(5, 100, 5):  # 0.5 to 10 kpc in 0.5 steps
            r = r_10 / 10.0
            m = enclosed_mass(r, MW_MODEL)
            velocities[r] = newt_v(r, m)
        peak_r = max(velocities, key=velocities.get)
        # Peak should be in 1-4 kpc range for MW
        assert 1.0 <= peak_r <= 4.0, f"Peak at r={peak_r} kpc"

    def test_08_inner_curve_rising(self):
        """Distributed: velocity must rise from center to peak."""
        v1 = newt_v(0.5, enclosed_mass(0.5, MW_MODEL))
        v2 = newt_v(1.0, enclosed_mass(1.0, MW_MODEL))
        v3 = newt_v(2.0, enclosed_mass(2.0, MW_MODEL))
        assert v1 < v2 < v3

    def test_09_outer_curve_declining(self):
        """Distributed: velocity must decline beyond ~5 kpc (Newtonian)."""
        v5 = newt_v(5.0, enclosed_mass(5.0, MW_MODEL))
        v10 = newt_v(10.0, enclosed_mass(10.0, MW_MODEL))
        v20 = newt_v(20.0, enclosed_mass(20.0, MW_MODEL))
        assert v5 > v10 > v20

    def test_10_newtonian_below_observed(self):
        """Newtonian prediction at 8 kpc must be BELOW observed 230 km/s.
        This is the 'missing mass' problem that motivates DTG/MOND."""
        m_enc = enclosed_mass(8.0, MW_MODEL)
        v = newt_v(8.0, m_enc)
        assert v < OBS_V_AT_8KPC, \
            f"Newton {v:.1f} >= observed {OBS_V_AT_8KPC} (no missing mass?!)"


# =====================================================================
# DUAL TETRAD GRAVITY (COVARIANT COMPLETION): 10 MILKY WAY TESTS
# =====================================================================

class TestDTGMilkyWay:
    """
    Dual Tetrad Gravity (AQUAL field equation with mu(x) = x/(1+x))
    applied to the Milky Way.

    10 tests covering point mass, distributed mass, boost factors,
    the BTFR, and cross-checks against Newtonian and observational limits.
    """

    def test_01_point_mass_at_8kpc(self):
        """Point mass (total MW): DTG v at 8 kpc should be ~246 km/s."""
        v = dtg_v(8.0, MW_TOTAL)
        assert abs(v - 245.95) < 0.5

    def test_02_distributed_mass_at_8kpc(self):
        """Distributed mass: DTG v at 8 kpc should be ~225 km/s."""
        m_enc = enclosed_mass(8.0, MW_MODEL)
        v = dtg_v(8.0, m_enc)
        assert abs(v - 225.32) < 0.5

    def test_03_matches_observed_at_solar_radius(self):
        """DTG with distributed mass should match v=230+/-15 km/s at 8 kpc."""
        m_enc = enclosed_mass(8.0, MW_MODEL)
        v = dtg_v(8.0, m_enc)
        assert abs(v - OBS_V_AT_8KPC) < OBS_V_TOLERANCE, \
            f"DTG distributed = {v:.1f}, observed = {OBS_V_AT_8KPC} +/- {OBS_V_TOLERANCE}"

    def test_04_always_above_newtonian(self):
        """DTG v >= Newtonian v at every MW radius (DTG adds to gravity)."""
        for r in [1, 2, 5, 8, 10, 15, 20, 25]:
            m = enclosed_mass(r, MW_MODEL)
            v_dtg = dtg_v(r, m)
            v_newt = newt_v(r, m)
            assert v_dtg >= v_newt - 0.01, \
                f"r={r}: DTG {v_dtg:.2f} < Newton {v_newt:.2f}"

    def test_05_boost_increases_with_radius(self):
        """DTG/Newtonian boost factor must increase with radius
        (deeper MOND regime at larger r)."""
        prev_boost = 0
        for r in [1, 2, 5, 8, 10, 15, 20, 25]:
            m = enclosed_mass(r, MW_MODEL)
            boost = dtg_v(r, m) / newt_v(r, m)
            assert boost > prev_boost, \
                f"r={r}: boost {boost:.4f} <= prev {prev_boost:.4f}"
            prev_boost = boost

    def test_06_inner_boost_small(self):
        """At r=1 kpc (high acceleration), DTG boost < 5%."""
        m = enclosed_mass(1.0, MW_MODEL)
        boost = dtg_v(1.0, m) / newt_v(1.0, m)
        assert boost < 1.05, f"Inner boost = {boost:.4f}, too large"

    def test_07_outer_boost_large(self):
        """At r=25 kpc (low acceleration), DTG boost > 50%."""
        m = enclosed_mass(25.0, MW_MODEL)
        boost = dtg_v(25.0, m) / newt_v(25.0, m)
        assert boost > 1.50, f"Outer boost = {boost:.4f}, too small"

    def test_08_btfr_deep_mond_limit(self):
        """At very large r (point mass), v^4 -> G*M*a0 (BTFR).
        DTG mu(x)=x/(1+x) approaches this as x->0."""
        v_200 = dtg_v(200.0, MW_TOTAL)
        v4 = (v_200 * 1000.0) ** 4
        gma0 = G * MW_TOTAL * M_SUN * A0
        # Should be approaching 1.0 (within ~5% at 200 kpc)
        ratio = v4 / gma0
        assert abs(ratio - 1.0) < 0.10, f"BTFR ratio = {ratio:.4f}"

    def test_09_flat_rotation_curve_region(self):
        """Distributed DTG curve should be approximately flat from 5-25 kpc.
        Variation < 15% across this range."""
        velocities = []
        for r in [5, 8, 10, 15, 20, 25]:
            m = enclosed_mass(r, MW_MODEL)
            velocities.append(dtg_v(r, m))
        v_max = max(velocities)
        v_min = min(velocities)
        variation = (v_max - v_min) / ((v_max + v_min) / 2)
        assert variation < 0.15, \
            f"DTG curve variation = {variation:.3f} ({v_min:.1f}-{v_max:.1f} km/s)"

    def test_10_distributed_vs_point_mass_crosscheck(self):
        """At r where 99% of mass is enclosed, distributed ~ point mass."""
        # At 25 kpc, 98.5% enclosed
        m_enc = enclosed_mass(25.0, MW_MODEL)
        v_dist = dtg_v(25.0, m_enc)
        v_point = dtg_v(25.0, MW_TOTAL)
        rel_diff = abs(v_dist - v_point) / v_point
        assert rel_diff < 0.02, \
            f"At 25 kpc: distributed={v_dist:.2f}, point={v_point:.2f}, diff={rel_diff:.4f}"


# =====================================================================
# CLASSICAL MOND: 10 MILKY WAY TESTS
# =====================================================================

class TestMONDMilkyWay:
    """
    Classical MOND (mu(x) = x/sqrt(1+x^2), Bekenstein & Milgrom)
    applied to the Milky Way with same a0 as DTG.

    10 tests covering point mass, distributed mass, comparison with DTG,
    and cross-checks against observational constraints.
    """

    def test_01_point_mass_at_8kpc(self):
        """Point mass (total MW): MOND v at 8 kpc should be ~218 km/s."""
        v = mond_v(8.0, MW_TOTAL)
        assert abs(v - 218.50) < 0.5

    def test_02_distributed_mass_at_8kpc(self):
        """Distributed mass: MOND v at 8 kpc should be ~200 km/s."""
        m_enc = enclosed_mass(8.0, MW_MODEL)
        v = mond_v(8.0, m_enc)
        assert abs(v - 199.80) < 0.5

    def test_03_mond_below_dtg_at_all_radii(self):
        """MOND v < DTG v at every MW radius (different interpolating functions).
        DTG mu(x)=x/(1+x) gives stronger boost than MOND mu(x)=x/sqrt(1+x^2)."""
        for r in [1, 2, 5, 8, 10, 15, 20, 25]:
            m = enclosed_mass(r, MW_MODEL)
            v_mond = mond_v(r, m)
            v_dtg = dtg_v(r, m)
            assert v_mond <= v_dtg + 0.01, \
                f"r={r}: MOND {v_mond:.2f} > DTG {v_dtg:.2f}"

    def test_04_mond_above_newtonian(self):
        """MOND v >= Newtonian v at every MW radius."""
        for r in [1, 2, 5, 8, 10, 15, 20, 25]:
            m = enclosed_mass(r, MW_MODEL)
            v_mond = mond_v(r, m)
            v_newt = newt_v(r, m)
            assert v_mond >= v_newt - 0.01, \
                f"r={r}: MOND {v_mond:.2f} < Newton {v_newt:.2f}"

    def test_05_ordering_newton_mond_dtg(self):
        """At every MW radius: Newton < MOND < DTG (for these mu functions)."""
        for r in [2, 5, 8, 10, 15, 20]:
            m = enclosed_mass(r, MW_MODEL)
            vn = newt_v(r, m)
            vm = mond_v(r, m)
            vd = dtg_v(r, m)
            assert vn < vm < vd, \
                f"r={r}: ordering violated: N={vn:.1f}, M={vm:.1f}, D={vd:.1f}"

    def test_06_mond_converges_to_newtonian_inner(self):
        """At r=1 kpc (high acceleration), MOND ~ Newtonian within 1%."""
        m = enclosed_mass(1.0, MW_MODEL)
        v_mond = mond_v(1.0, m)
        v_newt = newt_v(1.0, m)
        rel_diff = abs(v_mond - v_newt) / v_newt
        assert rel_diff < 0.01, f"MOND/Newton diff = {rel_diff:.4f} at 1 kpc"

    def test_07_mond_dtg_gap_peaks_then_converges(self):
        """DTG-MOND gap peaks at intermediate radii then shrinks.
        Both theories converge in both limits:
          - High acceleration (Newtonian): both -> Newton, gap -> 0
          - Deep MOND (r -> inf): both -> v^4 = G*M*a0, gap -> 0
        So the gap has a peak in the transition regime (~5-10 kpc for MW)."""
        # Inner galaxy: small gap (both near Newtonian)
        gap_1 = dtg_v(1, MW_TOTAL) - mond_v(1, MW_TOTAL)
        # Transition region: peak gap
        gap_8 = dtg_v(8, MW_TOTAL) - mond_v(8, MW_TOTAL)
        # Far out: gap shrinking toward zero
        gap_500 = dtg_v(500, MW_TOTAL) - mond_v(500, MW_TOTAL)
        # Peak should be larger than inner
        assert gap_8 > gap_1, \
            f"Transition gap {gap_8:.2f} should > inner gap {gap_1:.2f}"
        # Far-field gap should be small (converging to same BTFR)
        assert gap_500 < gap_8, \
            f"Far-field gap {gap_500:.2f} should < peak gap {gap_8:.2f}"
        # At 500 kpc, gap should be < 3 km/s (both near BTFR)
        assert gap_500 < 3.0, f"Far-field gap = {gap_500:.2f} km/s, too large"

    def test_08_mond_flat_curve_region(self):
        """Distributed MOND curve should be approximately flat from 5-25 kpc.
        Variation < 10% across this range."""
        velocities = []
        for r in [5, 8, 10, 15, 20, 25]:
            m = enclosed_mass(r, MW_MODEL)
            velocities.append(mond_v(r, m))
        v_max = max(velocities)
        v_min = min(velocities)
        variation = (v_max - v_min) / ((v_max + v_min) / 2)
        assert variation < 0.10, \
            f"MOND curve variation = {variation:.3f} ({v_min:.1f}-{v_max:.1f} km/s)"

    def test_09_mond_mu_stronger_transition(self):
        """MOND mu transitions more sharply than DTG mu at x=1.
        MOND mu(1) = 1/sqrt(2) ~ 0.707 vs DTG mu(1) = 0.5."""
        mond_at_1 = mu_mond(1.0)
        dtg_at_1 = dtg_mu(1.0)
        assert mond_at_1 > dtg_at_1, \
            f"MOND mu(1)={mond_at_1:.4f} should > DTG mu(1)={dtg_at_1:.4f}"
        assert abs(mond_at_1 - 1.0 / math.sqrt(2)) < 1e-15
        assert abs(dtg_at_1 - 0.5) < 1e-15

    def test_10_both_same_deep_mond_asymptote(self):
        """MOND and DTG converge to same v^4 = G*M*a0 at very large r.
        Both mu functions -> x as x -> 0, giving identical deep-MOND limit."""
        r = 500.0  # Very large radius, deep MOND
        v_mond = mond_v(r, MW_TOTAL)
        v_dtg = dtg_v(r, MW_TOTAL)
        # Both should give v ~ (G*M*a0)^(1/4) ~ 189 km/s
        v4_mond = (v_mond * 1000) ** 4
        v4_dtg = (v_dtg * 1000) ** 4
        gma0 = G * MW_TOTAL * M_SUN * A0
        # Both should be within 3% of BTFR at 500 kpc
        assert abs(v4_mond / gma0 - 1.0) < 0.03
        assert abs(v4_dtg / gma0 - 1.0) < 0.03


# =====================================================================
# CROSS-THEORY MILKY WAY TESTS
# =====================================================================

class TestMilkyWayCrossChecks:
    """
    Cross-theory validation: ensures internal consistency between
    Newtonian, DTG, and MOND applied to the same Milky Way mass model.
    """

    def test_all_agree_at_high_acceleration(self):
        """At very small r with high enclosed mass (high a), all three converge.
        Use a large point mass at 0.5 kpc to ensure deep Newtonian regime."""
        r = 0.5
        m = 1e12  # Very high mass -> very high acceleration
        vn = newt_v(r, m)
        vd = dtg_v(r, m)
        vm = mond_v(r, m)
        # All within 1% of each other
        assert abs(vd - vn) / vn < 0.01, \
            f"DTG={vd:.2f} vs Newton={vn:.2f}"
        assert abs(vm - vn) / vn < 0.01, \
            f"MOND={vm:.2f} vs Newton={vn:.2f}"

    def test_enclosed_mass_consistency(self):
        """Verify enclosed mass approaches total at large radius.
        Exponential disk has long tail; at 1000 kpc should be > 99.99%."""
        m_enc = enclosed_mass(1000.0, MW_MODEL)
        assert abs(m_enc - MW_TOTAL) / MW_TOTAL < 0.001

    def test_distributed_always_less_than_point(self):
        """For all theories, distributed v <= point mass v at every r."""
        for r in [2, 5, 8, 10, 15, 20]:
            m = enclosed_mass(r, MW_MODEL)
            for theory_v in [newt_v, dtg_v, mond_v]:
                v_dist = theory_v(r, m)
                v_point = theory_v(r, MW_TOTAL)
                assert v_dist <= v_point + 0.01, \
                    f"r={r}, {theory_v.__module__}: dist={v_dist:.1f} > point={v_point:.1f}"

    def test_only_dtg_matches_observed_curve(self):
        """DTG (distributed) is closest to observed 230 km/s at 8 kpc.
        Newton underpredicts, MOND is closer but DTG is best."""
        m = enclosed_mass(8.0, MW_MODEL)
        vn = newt_v(8.0, m)
        vm = mond_v(8.0, m)
        vd = dtg_v(8.0, m)
        # Newton clearly below
        assert vn < 190
        # DTG closest to 230
        err_n = abs(vn - OBS_V_AT_8KPC)
        err_m = abs(vm - OBS_V_AT_8KPC)
        err_d = abs(vd - OBS_V_AT_8KPC)
        assert err_d < err_m, f"DTG err={err_d:.1f} should < MOND err={err_m:.1f}"
        assert err_d < err_n, f"DTG err={err_d:.1f} should < Newton err={err_n:.1f}"
