"""
Diagnostic: compare GFD vs GFD+ across all prediction galaxies.

For each galaxy, compute:
  1. GFD and GFD+ velocities at each observation point
  2. Chi-squared residuals against observations
  3. Structural fraction of total acceleration
  4. Key ratios: Rd/R_env, M_stellar/M_total, R_t position

Also explores alternative structural formulations:
  A. Current: g_struct uses total M_stellar
  B. Alternative: g_struct uses enclosed M_stellar at R_t
  C. Alternative: g_struct uses total M_baryonic
  D. Alternative: modulate by gas fraction

No unicode. ASCII only.
"""

import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from physics.equations import mass_model_eq, gfd_eq, gfd_structure_eq
from data.galaxies import PREDICTION_GALAXIES


# -- Helpers --

def enclosed_stellar(mass_model, r_kpc):
    """Enclosed stellar mass (bulge + disk) at radius r_kpc."""
    m = 0.0
    bulge = mass_model.get("bulge", {})
    if bulge.get("M", 0) > 0:
        a = bulge["a"]
        m += bulge["M"] * r_kpc**2 / (r_kpc + a)**2
    disk = mass_model.get("disk", {})
    if disk.get("M", 0) > 0:
        x = r_kpc / disk["Rd"]
        m += disk["M"] * (1.0 - (1.0 + x) * math.exp(-x))
    return m


def total_stellar(mass_model):
    """Total stellar mass (bulge + disk)."""
    m = 0.0
    bulge = mass_model.get("bulge", {})
    m += bulge.get("M", 0)
    disk = mass_model.get("disk", {})
    m += disk.get("M", 0)
    return m


def total_baryonic(mass_model):
    """Total baryonic mass (bulge + disk + gas)."""
    m = total_stellar(mass_model)
    gas = mass_model.get("gas", {})
    m += gas.get("M", 0)
    return m


def gas_fraction(mass_model):
    """Gas fraction."""
    m_total = total_baryonic(mass_model)
    if m_total <= 0:
        return 0.0
    gas = mass_model.get("gas", {})
    return gas.get("M", 0) / m_total


def compute_gfd_velocity(r_kpc, mass_model, accel_ratio=1.0):
    """GFD velocity at a single radius."""
    m_enc, _ = mass_model_eq(r_kpc, mass_model)
    v, intermediates = gfd_eq(r_kpc, m_enc, accel_ratio=accel_ratio)
    return v, intermediates


def compute_gfd_plus_velocity(r_kpc, mass_model, galactic_radius_kpc,
                              m_stellar, accel_ratio=1.0):
    """GFD+ velocity at a single radius."""
    m_enc, _ = mass_model_eq(r_kpc, mass_model)
    v, intermediates = gfd_structure_eq(
        r_kpc, m_enc, accel_ratio=accel_ratio,
        galactic_radius_kpc=galactic_radius_kpc,
        m_stellar=m_stellar,
    )
    return v, intermediates


def chi_squared(obs_list, model_func):
    """Reduced chi-squared of model vs observations."""
    chi2 = 0.0
    n = 0
    for o in obs_list:
        v_model = model_func(o["r"])
        err = max(o["err"], 1.0)  # floor at 1 km/s
        chi2 += ((o["v"] - v_model) / err) ** 2
        n += 1
    if n <= 1:
        return 0.0
    return chi2 / (n - 1)


def rms_residual(obs_list, model_func):
    """RMS residual in km/s."""
    ss = 0.0
    n = 0
    for o in obs_list:
        v_model = model_func(o["r"])
        ss += (o["v"] - v_model) ** 2
        n += 1
    if n == 0:
        return 0.0
    return math.sqrt(ss / n)


def mean_signed_residual(obs_list, model_func):
    """Mean signed residual (positive = model overshoots)."""
    total = 0.0
    n = 0
    for o in obs_list:
        v_model = model_func(o["r"])
        total += (v_model - o["v"])
        n += 1
    if n == 0:
        return 0.0
    return total / n


# -- Main diagnostic --

def run_diagnostic():
    print("=" * 100)
    print("GFD+ STRUCTURAL TERM DIAGNOSTIC")
    print("=" * 100)
    print()

    # Summary table header
    fmt = "{:<20s} {:>6s} {:>6s} {:>8s} {:>8s} {:>7s} {:>7s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format(
        "Galaxy", "R_env", "R_t", "M_star", "Rd_disk",
        "Rd/Renv", "f_gas",
        "chi2_GFD", "chi2_+", "rms_GFD", "rms_+",
    ))
    print("-" * 100)

    results = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
        accel = gal.get("accel", 1.0)
        R_env = gal.get("galactic_radius")
        if R_env is None:
            continue

        R_t = 0.30 * R_env
        m_star = total_stellar(mm)
        m_bary = total_baryonic(mm)
        f_gas = gas_fraction(mm)
        Rd_disk = mm.get("disk", {}).get("Rd", 0)
        Rd_ratio = Rd_disk / R_env if R_env > 0 else 0

        # Model functions
        def gfd_model(r, _mm=mm, _a=accel):
            v, _ = compute_gfd_velocity(r, _mm, accel_ratio=_a)
            return v

        def gfd_plus_model(r, _mm=mm, _a=accel, _R=R_env, _ms=m_star):
            v, _ = compute_gfd_plus_velocity(r, _mm, _R, _ms, accel_ratio=_a)
            return v

        chi2_gfd = chi_squared(obs, gfd_model)
        chi2_plus = chi_squared(obs, gfd_plus_model)
        rms_gfd = rms_residual(obs, gfd_model)
        rms_plus = rms_residual(obs, gfd_plus_model)

        print(fmt.format(
            gid[:20],
            "{:.0f}".format(R_env),
            "{:.1f}".format(R_t),
            "{:.1e}".format(m_star),
            "{:.1f}".format(Rd_disk),
            "{:.3f}".format(Rd_ratio),
            "{:.0%}".format(f_gas),
            "{:.2f}".format(chi2_gfd),
            "{:.2f}".format(chi2_plus),
            "{:.1f}".format(rms_gfd),
            "{:.1f}".format(rms_plus),
        ))

        results.append({
            "id": gid, "R_env": R_env, "R_t": R_t,
            "m_star": m_star, "m_bary": m_bary, "f_gas": f_gas,
            "Rd_disk": Rd_disk, "Rd_ratio": Rd_ratio,
            "chi2_gfd": chi2_gfd, "chi2_plus": chi2_plus,
            "rms_gfd": rms_gfd, "rms_plus": rms_plus,
        })

    print()
    print("=" * 100)
    print("DETAILED PER-POINT COMPARISON (GFD vs GFD+ vs Observed)")
    print("=" * 100)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
        accel = gal.get("accel", 1.0)
        R_env = gal.get("galactic_radius")
        if R_env is None:
            continue

        R_t = 0.30 * R_env
        m_star = total_stellar(mm)

        print()
        print("-" * 80)
        print("  {}  (R_env={}, R_t={:.1f}, f_gas={:.0%})".format(
            gid, R_env, R_t, gas_fraction(mm)))
        print("-" * 80)
        pfmt = "  r={:>6.1f}  v_obs={:>6.1f}  v_GFD={:>6.1f}  v_GFD+={:>6.1f}  " \
               "dv_GFD={:>+6.1f}  dv_GFD+={:>+6.1f}  g_struct/g_tot={:>5.1%}"

        for o in obs:
            r = o["r"]
            v_obs = o["v"]

            v_gfd, _ = compute_gfd_velocity(r, mm, accel_ratio=accel)
            v_plus, ints = compute_gfd_plus_velocity(
                r, mm, R_env, m_star, accel_ratio=accel)

            g_struct = ints.get("g_struct", 0)
            g_total = ints.get("g_total", 0)
            frac = g_struct / g_total if g_total > 0 else 0

            print(pfmt.format(
                r, v_obs, v_gfd, v_plus,
                v_gfd - v_obs, v_plus - v_obs,
                frac))

    # -- Alternative formulations --
    print()
    print("=" * 100)
    print("ALTERNATIVE FORMULATIONS (Milky Way focus)")
    print("=" * 100)

    mw = None
    for g in PREDICTION_GALAXIES:
        if g["id"] == "milky_way":
            mw = g
            break

    if mw is None:
        print("Milky Way not found!")
        return

    mm = mw["mass_model"]
    obs = mw["observations"]
    accel = mw.get("accel", 1.0)
    R_env = mw["galactic_radius"]
    R_t = 0.30 * R_env
    m_star_total = total_stellar(mm)
    m_bary_total = total_baryonic(mm)
    m_star_at_Rt = enclosed_stellar(mm, R_t)
    f_gas_val = gas_fraction(mm)

    print()
    print("  Milky Way properties:")
    print("    R_env = {} kpc, R_t = {:.1f} kpc".format(R_env, R_t))
    print("    M_stellar_total = {:.2e}".format(m_star_total))
    print("    M_stellar_enclosed(R_t) = {:.2e}  ({:.1%} of total)".format(
        m_star_at_Rt, m_star_at_Rt / m_star_total if m_star_total > 0 else 0))
    print("    M_baryonic_total = {:.2e}".format(m_bary_total))
    print("    f_gas = {:.1%}".format(f_gas_val))
    print("    Rd_disk = {} kpc, Rd/R_env = {:.3f}".format(
        mm["disk"]["Rd"], mm["disk"]["Rd"] / R_env))
    print()

    # Define alternative structural models
    THROAT_FRAC = 0.30
    STRUCT_FRAC = 4.0 / 13.0
    P_STRUCT = 3.0 / 4.0

    def struct_accel(r_kpc, R_env_val, m_struct):
        """Raw structural acceleration for a given mass and R_env."""
        R_t_val = THROAT_FRAC * R_env_val
        if r_kpc <= R_t_val or R_env_val <= R_t_val or m_struct <= 0:
            return 0.0
        R_t_m = R_t_val * KPC_TO_M
        g0 = STRUCT_FRAC * G * m_struct * M_SUN / (R_t_m * R_t_m)
        xi = (r_kpc - R_t_val) / (R_env_val - R_t_val)
        return g0 * (xi ** P_STRUCT)

    def velocity_with_struct(r_kpc, mass_model, R_env_val, m_struct,
                             accel_ratio=1.0):
        """GFD + arbitrary structural mass."""
        m_enc, _ = mass_model_eq(r_kpc, mass_model)
        r = r_kpc * KPC_TO_M
        M = m_enc * M_SUN
        if r <= 0 or M <= 0:
            return 0.0
        gN = G * M / (r * r)
        a0_eff = A0 * accel_ratio
        y_N = gN / a0_eff
        x = aqual_solve_x(y_N)
        g_dtg = a0_eff * x
        g_s = struct_accel(r_kpc, R_env_val, m_struct)
        g_tot = g_dtg + g_s
        return math.sqrt(g_tot * r) / 1000.0

    alternatives = [
        ("A: Current (total M_stellar)", m_star_total),
        ("B: Enclosed M_stellar at R_t", m_star_at_Rt),
        ("C: Total M_baryonic", m_bary_total),
        ("D: M_stellar * (1 - f_gas)", m_star_total * (1.0 - f_gas_val)),
        ("E: M_stellar * f_gas", m_star_total * f_gas_val),
        ("F: M_stellar * 0.5", m_star_total * 0.5),
    ]

    print("  Model variants for Milky Way:")
    for label, m_val in alternatives:
        print("    {}: m_struct = {:.2e}".format(label, m_val))
    print()

    hdr = "  {:>6s}  {:>6s}  {:>6s}".format("r_kpc", "v_obs", "v_GFD")
    for label, _ in alternatives:
        tag = label.split(":")[0]
        hdr += "  {:>8s}".format(tag)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for o in obs:
        r = o["r"]
        v_obs = o["v"]
        v_gfd, _ = compute_gfd_velocity(r, mm, accel_ratio=accel)
        line = "  {:6.1f}  {:6.1f}  {:6.1f}".format(r, v_obs, v_gfd)
        for label, m_val in alternatives:
            v_alt = velocity_with_struct(r, mm, R_env, m_val, accel_ratio=accel)
            line += "  {:8.1f}".format(v_alt)
        print(line)

    # Chi-squared for each alternative
    print()
    print("  Chi-squared (reduced) and RMS for each variant:")
    print("  {:<40s} {:>10s} {:>10s} {:>12s}".format(
        "Variant", "chi2_red", "RMS(km/s)", "Mean_bias"))
    print("  " + "-" * 72)

    # GFD baseline
    def gfd_func(r):
        v, _ = compute_gfd_velocity(r, mm, accel_ratio=accel)
        return v

    chi2_gfd_mw = chi_squared(obs, gfd_func)
    rms_gfd_mw = rms_residual(obs, gfd_func)
    bias_gfd_mw = mean_signed_residual(obs, gfd_func)
    print("  {:<40s} {:>10.2f} {:>10.1f} {:>+12.1f}".format(
        "GFD (no structural)", chi2_gfd_mw, rms_gfd_mw, bias_gfd_mw))

    for label, m_val in alternatives:
        def alt_func(r, _m=m_val):
            return velocity_with_struct(r, mm, R_env, _m, accel_ratio=accel)

        chi2_val = chi_squared(obs, alt_func)
        rms_val = rms_residual(obs, alt_func)
        bias_val = mean_signed_residual(obs, alt_func)
        print("  {:<40s} {:>10.2f} {:>10.1f} {:>+12.1f}".format(
            label, chi2_val, rms_val, bias_val))

    # -- R_env sensitivity --
    print()
    print("=" * 100)
    print("R_ENV SENSITIVITY (Milky Way)")
    print("=" * 100)
    print()
    print("  How chi2 and RMS change with R_env (using total M_stellar):")
    print("  {:>6s}  {:>6s}  {:>10s}  {:>10s}  {:>12s}".format(
        "R_env", "R_t", "chi2_red", "RMS(km/s)", "Mean_bias"))
    print("  " + "-" * 50)

    for R_test in [20, 25, 30, 35, 40, 50, 60, 80, 100]:
        def test_func(r, _R=R_test):
            return velocity_with_struct(r, mm, _R, m_star_total,
                                        accel_ratio=accel)

        chi2_t = chi_squared(obs, test_func)
        rms_t = rms_residual(obs, test_func)
        bias_t = mean_signed_residual(obs, test_func)
        R_t_test = 0.30 * R_test
        print("  {:6.0f}  {:6.1f}  {:10.2f}  {:10.1f}  {:>+12.1f}".format(
            R_test, R_t_test, chi2_t, rms_t, bias_t))

    # -- Cross-galaxy summary with improvement/degradation markers --
    print()
    print("=" * 100)
    print("CROSS-GALAXY: Does GFD+ improve or degrade fit?")
    print("=" * 100)
    print()
    sfmt = "  {:<20s} {:>6s} {:>7s} {:>8s} {:>8s} {:>8s} {:>10s}"
    print(sfmt.format("Galaxy", "f_gas", "Rd/Renv", "rms_GFD", "rms_GFD+",
                       "delta", "verdict"))
    print("  " + "-" * 75)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm_g = gal["mass_model"]
        obs_g = gal["observations"]
        accel_g = gal.get("accel", 1.0)
        R_env_g = gal.get("galactic_radius")
        if R_env_g is None:
            continue

        m_star_g = total_stellar(mm_g)
        Rd_g = mm_g.get("disk", {}).get("Rd", 0)

        def gfd_g(r, _m=mm_g, _a=accel_g):
            v, _ = compute_gfd_velocity(r, _m, accel_ratio=_a)
            return v

        def plus_g(r, _m=mm_g, _a=accel_g, _R=R_env_g, _ms=m_star_g):
            v, _ = compute_gfd_plus_velocity(r, _m, _R, _ms, accel_ratio=_a)
            return v

        rms_gfd_g = rms_residual(obs_g, gfd_g)
        rms_plus_g = rms_residual(obs_g, plus_g)
        delta = rms_plus_g - rms_gfd_g
        fg = gas_fraction(mm_g)
        rd_ratio = Rd_g / R_env_g if R_env_g > 0 else 0

        if delta < -2:
            verdict = "++ BETTER"
        elif delta < 0:
            verdict = "+ better"
        elif delta < 2:
            verdict = "~ similar"
        elif delta < 5:
            verdict = "- worse"
        else:
            verdict = "-- WORSE"

        print(sfmt.format(
            gid[:20], "{:.0%}".format(fg), "{:.3f}".format(rd_ratio),
            "{:.1f}".format(rms_gfd_g), "{:.1f}".format(rms_plus_g),
            "{:>+.1f}".format(delta), verdict))


if __name__ == "__main__":
    run_diagnostic()
