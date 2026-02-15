"""
Milky Way GFD+ diagnostic: test M_disk and R_env combinations.

The MW is the only galaxy with a declining rotation curve in our sample.
Two hypotheses for the GFD+ overshoot:
  1. M_disk too high (5.0e10 vs literature range 3.5-5.0e10)
  2. R_env too small (30 vs ~50 kpc including stellar halo + extended HI)

Tests all combinations and reports chi2, RMS, and per-point residuals.

No unicode. ASCII only.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from physics.equations import mass_model_eq, gfd_eq


# MW observations (Ou+2023)
MW_OBS = [
    {"r": 6.3,  "v": 231, "err": 1},
    {"r": 7.9,  "v": 234, "err": 1},
    {"r": 9.2,  "v": 230, "err": 1},
    {"r": 10.2, "v": 229, "err": 1},
    {"r": 11.2, "v": 227, "err": 1},
    {"r": 12.2, "v": 227, "err": 1},
    {"r": 13.2, "v": 225, "err": 1},
    {"r": 14.2, "v": 222, "err": 1},
    {"r": 15.2, "v": 218, "err": 1},
    {"r": 16.2, "v": 218, "err": 2},
    {"r": 17.2, "v": 220, "err": 2},
    {"r": 18.2, "v": 215, "err": 2},
    {"r": 19.2, "v": 208, "err": 2},
    {"r": 20.2, "v": 203, "err": 2},
    {"r": 20.7, "v": 195, "err": 2},
    {"r": 21.2, "v": 200, "err": 2},
    {"r": 21.7, "v": 201, "err": 3},
    {"r": 22.3, "v": 197, "err": 6},
    {"r": 23.4, "v": 192, "err": 5},
    {"r": 25.0, "v": 191, "err": 8},
]

# Fixed MW components
MW_BULGE = {"M": 1.5e10, "a": 0.6}
MW_GAS = {"M": 1.5e10, "Rd": 7.0}

THROAT_FRAC = 0.30
STRUCT_FRAC = 4.0 / 13.0
P_STRUCT = 3.0 / 4.0


def make_mass_model(m_disk, rd_disk=2.5):
    return {
        "bulge": MW_BULGE.copy(),
        "disk": {"M": m_disk, "Rd": rd_disk},
        "gas": MW_GAS.copy(),
    }


def gfd_plus_v(r_kpc, mass_model, R_env, m_stellar, accel_ratio=1.0):
    """Compute GFD+ velocity."""
    m_enc, _ = mass_model_eq(r_kpc, mass_model)
    r = r_kpc * KPC_TO_M
    M = m_enc * M_SUN
    if r <= 0 or M <= 0:
        return 0.0, 0.0, 0.0

    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    y_N = gN / a0_eff
    x = aqual_solve_x(y_N)
    g_dtg = a0_eff * x

    R_t = THROAT_FRAC * R_env
    g_struct = 0.0
    if r_kpc > R_t and R_env > R_t and m_stellar > 0:
        R_t_m = R_t * KPC_TO_M
        g0 = STRUCT_FRAC * G * m_stellar * M_SUN / (R_t_m * R_t_m)
        xi = (r_kpc - R_t) / (R_env - R_t)
        g_struct = g0 * (xi ** P_STRUCT)

    g_total = g_dtg + g_struct
    v = math.sqrt(g_total * r) / 1000.0
    v_gfd = math.sqrt(g_dtg * r) / 1000.0
    return v, v_gfd, g_struct / g_total if g_total > 0 else 0


def chi2_rms(obs, mass_model, R_env, m_stellar):
    """Compute chi2, RMS, and mean bias."""
    chi2 = 0.0
    ss = 0.0
    bias = 0.0
    n = len(obs)
    for o in obs:
        v_model, _, _ = gfd_plus_v(o["r"], mass_model, R_env, m_stellar)
        err = max(o["err"], 1.0)
        chi2 += ((o["v"] - v_model) / err) ** 2
        ss += (o["v"] - v_model) ** 2
        bias += (v_model - o["v"])
    return chi2 / max(n - 1, 1), math.sqrt(ss / n), bias / n


def gfd_only_chi2_rms(obs, mass_model):
    """GFD only (no structural term)."""
    chi2 = 0.0
    ss = 0.0
    bias = 0.0
    n = len(obs)
    for o in obs:
        m_enc, _ = mass_model_eq(o["r"], mass_model)
        v, _ = gfd_eq(o["r"], m_enc)
        err = max(o["err"], 1.0)
        chi2 += ((o["v"] - v) / err) ** 2
        ss += (o["v"] - v) ** 2
        bias += (v - o["v"])
    return chi2 / max(n - 1, 1), math.sqrt(ss / n), bias / n


def run():
    print("=" * 100)
    print("MILKY WAY GFD+ PARAMETER EXPLORATION")
    print("=" * 100)

    # Test grid
    disk_masses = [3.5e10, 4.0e10, 4.5e10, 5.0e10]
    r_envs = [30, 40, 50, 60, 80]

    # -- Part 1: GFD-only baseline for each M_disk --
    print()
    print("PART 1: GFD-only (no structural term) sensitivity to M_disk")
    print("-" * 60)
    print("{:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}".format(
        "M_disk", "M_total", "chi2_red", "RMS(km/s)", "Bias"))
    print("-" * 60)

    for md in disk_masses:
        mm = make_mass_model(md)
        m_total = MW_BULGE["M"] + md + MW_GAS["M"]
        c2, rms, bias = gfd_only_chi2_rms(MW_OBS, mm)
        print("{:>10.1e}  {:>8.1e}  {:>10.2f}  {:>10.1f}  {:>+10.1f}".format(
            md, m_total, c2, rms, bias))

    # -- Part 2: GFD+ grid search --
    print()
    print("PART 2: GFD+ sensitivity to M_disk x R_env")
    print("-" * 80)
    print("{:>10s}  {:>6s}  {:>6s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}".format(
        "M_disk", "R_env", "R_t", "M_star", "chi2_red", "RMS(km/s)", "Bias", "better?"))
    print("-" * 80)

    best_chi2 = 1e9
    best_combo = None

    for md in disk_masses:
        mm = make_mass_model(md)
        m_star = MW_BULGE["M"] + md
        c2_gfd, rms_gfd, _ = gfd_only_chi2_rms(MW_OBS, mm)

        for renv in r_envs:
            c2, rms, bias = chi2_rms(MW_OBS, mm, renv, m_star)
            better = "YES" if rms < rms_gfd else "no"
            marker = " ***" if c2 < best_chi2 else ""
            if c2 < best_chi2:
                best_chi2 = c2
                best_combo = (md, renv)
            print("{:>10.1e}  {:>6.0f}  {:>6.1f}  {:>8.1e}  {:>10.2f}  {:>10.1f}  {:>+10.1f}  {:>8s}{}".format(
                md, renv, 0.30 * renv, m_star, c2, rms, bias, better, marker))
        print()

    print("BEST COMBO: M_disk={:.1e}, R_env={:.0f}  (chi2={:.2f})".format(
        best_combo[0], best_combo[1], best_chi2))

    # -- Part 3: Per-point comparison for key combos --
    combos = [
        ("Current: Md=5e10, Re=30", 5.0e10, 30),
        ("Fix1: Md=4e10, Re=30", 4.0e10, 30),
        ("Fix2: Md=5e10, Re=50", 5.0e10, 50),
        ("Fix3: Md=4e10, Re=50", 4.0e10, 50),
        ("Fix4: Md=4.5e10, Re=50", 4.5e10, 50),
        ("Fix5: Md=4e10, Re=40", 4.0e10, 40),
    ]

    print()
    print("=" * 100)
    print("PART 3: Per-point residuals (v_model - v_obs) for key combos")
    print("=" * 100)

    # Header
    hdr = "{:>6s}  {:>6s}".format("r", "v_obs")
    for label, _, _ in combos:
        tag = label.split(":")[0]
        hdr += "  {:>10s}".format(tag)
    hdr += "  {:>10s}".format("GFD only")
    print(hdr)
    print("-" * len(hdr))

    for o in MW_OBS:
        r = o["r"]
        line = "{:6.1f}  {:6.1f}".format(r, o["v"])
        for label, md, renv in combos:
            mm = make_mass_model(md)
            m_star = MW_BULGE["M"] + md
            v_plus, v_gfd, frac = gfd_plus_v(r, mm, renv, m_star)
            line += "  {:>+10.1f}".format(v_plus - o["v"])
        # GFD only (with current 5e10)
        mm5 = make_mass_model(5.0e10)
        m_enc, _ = mass_model_eq(r, mm5)
        v_gfd_only, _ = gfd_eq(r, m_enc)
        line += "  {:>+10.1f}".format(v_gfd_only - o["v"])
        print(line)

    # Summary
    print()
    print("SUMMARY:")
    print("-" * 80)
    sfmt = "  {:<35s}  {:>10s}  {:>10s}  {:>10s}"
    print(sfmt.format("Combo", "chi2_red", "RMS(km/s)", "Bias"))
    print("  " + "-" * 75)

    # GFD only baselines
    for md in [5.0e10, 4.0e10]:
        mm = make_mass_model(md)
        c2, rms, bias = gfd_only_chi2_rms(MW_OBS, mm)
        print(sfmt.format(
            "GFD only (Md={:.0e})".format(md),
            "{:.2f}".format(c2), "{:.1f}".format(rms), "{:+.1f}".format(bias)))

    for label, md, renv in combos:
        mm = make_mass_model(md)
        m_star = MW_BULGE["M"] + md
        c2, rms, bias = chi2_rms(MW_OBS, mm, renv, m_star)
        print(sfmt.format(label, "{:.2f}".format(c2),
                          "{:.1f}".format(rms), "{:+.1f}".format(bias)))

    # -- Part 4: Verify other galaxies still work with any MW changes --
    print()
    print("=" * 100)
    print("PART 4: Sanity check - does changing R_env concept affect others?")
    print("(Other galaxies unchanged, just checking current values)")
    print("=" * 100)


if __name__ == "__main__":
    run()
