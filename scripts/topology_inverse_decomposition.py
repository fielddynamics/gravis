#!/usr/bin/env python3
"""
topology_inverse_decomposition.py
===================================
Test whether the 3-constraint linear system can recover Mb, Md, Mg
from topology-derived R_env, R_t, cycle, and SPARC scale lengths.

The system:
  Eq1 (horizon): fb(R_env)*Mb + fd(R_env)*Md + fg(R_env)*Mg = M_horizon
  Eq2 (throat):  fb(R_t)*Mb  + fd(R_t)*Md  + fg(R_t)*Mg  = M_throat
  Eq3 (closure): Md = ratio * Mg   (3-cycle: 3.0, 2-cycle: 0.6)

Where:
  fb(r) = r^2 / (r + ab)^2           (Hernquist enclosed fraction)
  fd(r) = 1 - (1 + r/Rd)*exp(-r/Rd)  (exponential enclosed fraction)
  fg(r) = 1 - (1 + r/Rg)*exp(-r/Rg)  (exponential enclosed fraction)
  M_horizon = (36/1365) * R_env^2 * a0 / G
  M_throat  = (18/65)   * R_t^2   * a0 / G   (3-cycle)

This is a 3x3 linear system solved directly via Cramer's rule.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0, G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.services.rotation.inference import solve_field_geometry


def hernquist_frac(r, a):
    """Fraction of Hernquist mass enclosed at radius r."""
    if r <= 0 or a <= 0:
        return 0.0
    return r * r / ((r + a) * (r + a))


def disk_frac(r, Rd):
    """Fraction of exponential disk mass enclosed at radius r."""
    if r <= 0 or Rd <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return 1.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def solve_3x3(A, b):
    """Solve 3x3 system Ax = b via Cramer's rule. Returns None if singular."""
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if abs(det) < 1e-30:
        return None

    x0 = (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2])
        + A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) / det
    x1 = (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2])
        - b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) / det
    x2 = (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1])
        - A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0])
        + b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det
    return (x0, x1, x2)


def pct_err(pred, actual):
    """Percentage error, handling zero."""
    if actual == 0:
        return 0.0 if pred == 0 else 999.9
    return (pred - actual) / actual * 100


def fmt_mass(m):
    """Format mass in scientific notation."""
    if m == 0:
        return "      0"
    exp = int(math.floor(math.log10(abs(m))))
    coeff = m / 10**exp
    return "%5.2fe%d" % (coeff, exp)


def main():
    print("=" * 120)
    print("  TOPOLOGY INVERSE DECOMPOSITION: Solve for Mb, Md, Mg from R_env, R_t, cycle + SPARC scale lengths")
    print("=" * 120)

    # Closure cycle ratios
    RATIO_3CYC = 3.0   # Md/Mg for 3-cycle
    RATIO_2CYC = 0.6   # Md/Mg for 2-cycle

    print()
    print("  %-12s %3s | %10s %10s %10s %10s | %10s %10s %10s %10s | %6s %6s %6s %6s" % (
        "Galaxy", "Cyc",
        "Mb_sparc", "Md_sparc", "Mg_sparc", "Mt_sparc",
        "Mb_pred", "Md_pred", "Mg_pred", "Mt_pred",
        "dMb%", "dMd%", "dMg%", "dMt%"))
    print("  " + "-" * 116)

    err_Mb = []
    err_Md = []
    err_Mg = []
    err_Mt = []
    results_3cyc = []
    results_2cyc = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        # SPARC published values
        Mb_s = mm["bulge"]["M"]
        ab = mm["bulge"]["a"]
        Md_s = mm["disk"]["M"]
        Rd = mm["disk"]["Rd"]
        Mg_s = mm["gas"]["M"]
        Rg = mm["gas"]["Rd"]
        Mt_s = Mb_s + Md_s + Mg_s

        # Get topology-derived geometry from SPARC mass model
        fg = solve_field_geometry(Mb_s, ab, Md_s, Rd, Mg_s, Rg, a0_eff)
        r_t = fg.get("throat_radius_kpc")
        r_env = fg.get("envelope_radius_kpc")
        cycle = fg.get("cycle", 3)

        if r_env is None or r_env <= 0:
            continue
        if r_t is None or r_t <= 0:
            continue

        # Compute M_horizon and M_throat from topology
        r_env_m = r_env * KPC_TO_M
        M_horizon = HORIZON_YN * r_env_m * r_env_m * a0_eff / (G * M_SUN)

        if cycle == 3:
            r_t_m = r_t * KPC_TO_M
            M_throat = THROAT_YN * r_t_m * r_t_m * a0_eff / (G * M_SUN)
            ratio = RATIO_3CYC
        else:
            # For 2-cycle, use actual yN at throat
            yN_at_rt = fg.get("yN_at_throat", 0.0)
            r_t_m = r_t * KPC_TO_M
            M_throat = yN_at_rt * r_t_m * r_t_m * a0_eff / (G * M_SUN)
            ratio = RATIO_2CYC

        # Build the 3x3 system using SPARC scale lengths
        # Row 0: horizon condition
        fb_env = hernquist_frac(r_env, ab)
        fd_env = disk_frac(r_env, Rd)
        fg_env = disk_frac(r_env, Rg)

        # Row 1: throat condition
        fb_t = hernquist_frac(r_t, ab)
        fd_t = disk_frac(r_t, Rd)
        fg_t = disk_frac(r_t, Rg)

        # Row 2: closure ratio Md = ratio * Mg => Md - ratio*Mg = 0
        # Variables: [Mb, Md, Mg]
        A = [
            [fb_env, fd_env, fg_env],
            [fb_t,   fd_t,   fg_t],
            [0.0,    1.0,    -ratio],
        ]
        b = [M_horizon, M_throat, 0.0]

        sol = solve_3x3(A, b)
        if sol is None:
            print("  %-12s %3d | SINGULAR MATRIX" % (gid, cycle))
            continue

        Mb_p, Md_p, Mg_p = sol
        Mt_p = Mb_p + Md_p + Mg_p

        dMb = pct_err(Mb_p, Mb_s)
        dMd = pct_err(Md_p, Md_s)
        dMg = pct_err(Mg_p, Mg_s)
        dMt = pct_err(Mt_p, Mt_s)

        # Flag negative masses
        flag = ""
        if Mb_p < 0 or Md_p < 0 or Mg_p < 0:
            flag = " NEG"

        print("  %-12s %3d | %10s %10s %10s %10s | %10s %10s %10s %10s | %+6.0f %+6.0f %+6.0f %+6.0f%s" % (
            gid, cycle,
            fmt_mass(Mb_s), fmt_mass(Md_s), fmt_mass(Mg_s), fmt_mass(Mt_s),
            fmt_mass(Mb_p), fmt_mass(Md_p), fmt_mass(Mg_p), fmt_mass(Mt_p),
            dMb, dMd, dMg, dMt, flag))

        if Mb_p >= 0 and Md_p >= 0 and Mg_p >= 0:
            err_Mt.append(abs(dMt))
            if Mb_s > 0:
                err_Mb.append(abs(dMb))
            err_Md.append(abs(dMd))
            if Mg_s > 0:
                err_Mg.append(abs(dMg))
            if cycle == 3:
                results_3cyc.append((gid, dMb, dMd, dMg, dMt))
            else:
                results_2cyc.append((gid, dMb, dMd, dMg, dMt))

    print()
    print("=" * 120)
    print("  STATISTICS (excluding galaxies with negative mass predictions)")
    print("=" * 120)

    def show_stats(label, arr):
        if not arr:
            print("  %s: no data" % label)
            return
        n = len(arr)
        mean = sum(arr) / n
        med = sorted(arr)[n // 2]
        print("  %s (N=%d): median |err| = %.1f%%,  mean |err| = %.1f%%" % (
            label, n, med, mean))

    show_stats("M_total", err_Mt)
    show_stats("M_bulge (where Mb > 0)", err_Mb)
    show_stats("M_disk", err_Md)
    show_stats("M_gas (where Mg > 0)", err_Mg)

    print()
    print("  3-cycle results (N=%d):" % len(results_3cyc))
    for gid, dMb, dMd, dMg, dMt in results_3cyc:
        print("    %-12s  dMt=%+6.1f%%  dMb=%+6.1f%%  dMd=%+6.1f%%  dMg=%+6.1f%%" % (
            gid, dMt, dMb, dMd, dMg))

    print()
    print("  2-cycle results (N=%d):" % len(results_2cyc))
    for gid, dMb, dMd, dMg, dMt in results_2cyc:
        print("    %-12s  dMt=%+6.1f%%  dMb=%+6.1f%%  dMd=%+6.1f%%  dMg=%+6.1f%%" % (
            gid, dMt, dMb, dMd, dMg))


if __name__ == "__main__":
    main()
