#!/usr/bin/env python3
"""
bulge_self_correction.py
=========================
Test whether the Hernquist bulge profile can self-correct M_bulge
when M_total and M_gas are already known from topology.

Chain:
  1. Topology: M_total (0.1% accurate), M_gas (2.5% from Method B)
  2. Bayesian fit: scale lengths a_b, Rd, Rg
  3. One free parameter: M_b (with M_d = M_total - M_b - M_g)
  4. Compute v_GFD(r) for each trial M_b
  5. Compare against observation data
  6. Pick M_b that minimizes velocity chi-squared

Because the Hernquist profile concentrates mass at small radii,
the inner velocity curve is sensitive to M_b vs M_d split.
This is a 1D optimization that leverages the full rotation curve.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0, G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry


def hernquist_enc(r, M, a):
    if M <= 0 or a <= 0 or r <= 0:
        return 0.0
    return M * r * r / ((r + a) * (r + a))


def disk_enc(r, M, Rd):
    if M <= 0 or Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return M
    return M * (1.0 - (1.0 + x) * math.exp(-x))


def gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """GFD circular velocity from 3-component mass model."""
    enc = hernquist_enc(r_kpc, Mb, ab) + disk_enc(r_kpc, Md, Rd) + disk_enc(r_kpc, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0


def hernquist_frac(r, a):
    if r <= 0 or a <= 0:
        return 0.0
    return r * r / ((r + a) * (r + a))


def disk_frac(r, Rd):
    if r <= 0 or Rd <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return 1.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def solve_3x3(A, b):
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


def method_b_get_mg(r_env, r_t, cycle, yN_at_rt, ab, Rd, Rg, a0_eff):
    """Get M_total and M_gas from Method B (topology + continuous relationship)."""
    r_env_m = r_env * KPC_TO_M
    M_horizon = HORIZON_YN * r_env_m**2 * a0_eff / (G * M_SUN)

    if cycle == 3:
        r_t_m = r_t * KPC_TO_M
        M_throat = THROAT_YN * r_t_m**2 * a0_eff / (G * M_SUN)
    else:
        r_t_m = r_t * KPC_TO_M
        M_throat = yN_at_rt * r_t_m**2 * a0_eff / (G * M_SUN)

    fb_env = hernquist_frac(r_env, ab)
    fd_env = disk_frac(r_env, Rd)
    fg_env = disk_frac(r_env, Rg)
    fb_t = hernquist_frac(r_t, ab)
    fd_t = disk_frac(r_t, Rd)
    fg_t = disk_frac(r_t, Rg)

    ratio = 2.0
    sol = None
    for _ in range(30):
        A = [
            [fb_env, fd_env, fg_env],
            [fb_t,   fd_t,   fg_t],
            [0.0,    1.0,    -ratio],
        ]
        b = [M_horizon, M_throat, 0.0]
        sol = solve_3x3(A, b)
        if sol is None:
            return None, None, None
        Mb, Md, Mg = sol
        Mt = Mb + Md + Mg
        if Mt <= 0:
            break
        fg = max(Mg, 0) / Mt
        fg = max(0.01, min(fg, 0.99))
        fd_pred = -0.826 * fg + 0.863
        fd_pred = max(0.02, min(fd_pred, 0.95))
        new_ratio = fd_pred / fg
        new_ratio = max(0.01, min(new_ratio, 20.0))
        if abs(new_ratio - ratio) < 0.001:
            break
        ratio = ratio * 0.5 + new_ratio * 0.5

    if sol is None:
        return None, None, None
    Mb, Md, Mg = sol
    Mt = Mb + Md + Mg
    return Mt, Mg, sol


def scan_bulge_fraction(obs, Mt, Mg, ab, Rd, Rg, a0_eff, n_steps=200):
    """
    1D scan: M_b from 0 to (Mt - Mg), M_d = Mt - M_b - Mg.
    Find M_b that minimizes weighted velocity chi-squared.
    """
    max_Mb = max(Mt - Mg, 0) * 0.5  # bulge can't exceed 50% of stellar
    if max_Mb <= 0:
        return 0.0, Mt - Mg, 999.0

    best_chi2 = float('inf')
    best_Mb = 0.0

    for i in range(n_steps + 1):
        Mb = max_Mb * i / n_steps
        Md = Mt - Mb - Mg
        if Md < 0:
            continue

        chi2 = 0.0
        for pt in obs:
            r, v_obs, err = pt["r"], pt["v"], pt["err"]
            if r <= 0 or v_obs <= 0:
                continue
            v_pred = gfd_velocity(r, Mb, ab, Md, Rd, Mg, Rg, a0_eff)
            w = 1.0 / (err * err) if err > 0 else 1.0
            chi2 += w * (v_pred - v_obs) ** 2

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_Mb = Mb

    # Refine with golden section around best
    lo = max(0, best_Mb - max_Mb / n_steps)
    hi = min(max_Mb, best_Mb + max_Mb / n_steps)
    gr = (math.sqrt(5) + 1) / 2
    for _ in range(50):
        if hi - lo < 1.0:  # 1 solar mass precision
            break
        c = hi - (hi - lo) / gr
        d = lo + (hi - lo) / gr

        def cost(mb):
            md = Mt - mb - Mg
            if md < 0:
                return 1e30
            chi = 0.0
            for pt in obs:
                r, v_obs, err = pt["r"], pt["v"], pt["err"]
                if r <= 0 or v_obs <= 0:
                    continue
                v_pred = gfd_velocity(r, mb, ab, md, Rd, Mg, Rg, a0_eff)
                w = 1.0 / (err * err) if err > 0 else 1.0
                chi += w * (v_pred - v_obs) ** 2
            return chi

        if cost(c) < cost(d):
            hi = d
        else:
            lo = c

    best_Mb = (lo + hi) / 2.0
    best_Md = Mt - best_Mb - Mg
    return best_Mb, best_Md, best_chi2


def pct(pred, actual):
    if actual == 0:
        return 0.0 if pred == 0 else 999.9
    return (pred - actual) / actual * 100


def fmt(m):
    if abs(m) < 1e3:
        return "%8.0f" % m
    exp = int(math.floor(math.log10(max(abs(m), 1))))
    coeff = m / 10**exp
    return "%5.2fe%d" % (coeff, exp)


def main():
    print("=" * 150)
    print("  BULGE SELF-CORRECTION VIA VELOCITY CURVE")
    print("  Fix M_total and M_gas from topology. Scan M_b as 1 free parameter.")
    print("  M_d = M_total - M_b - M_g. Pick M_b minimizing velocity chi-squared.")
    print("=" * 150)
    print()

    hdr = "  %-12s %3s | %9s %9s %9s %9s" % ("Galaxy", "Cyc", "Mb_sparc", "Md_sparc", "Mg_sparc", "Mt_sparc")
    hdr += " | %9s %9s %9s %9s" % ("Mb_methB", "Md_methB", "Mg_methB", "Mt_methB")
    hdr += " | %9s %9s %9s" % ("Mb_corr", "Md_corr", "Mg_corr")
    hdr += " | %6s %6s %6s" % ("eB_B%", "eB_C%", "eD_C%")
    print(hdr)
    print("  " + "-" * 146)

    # Accumulators
    b_mb_errs = []
    c_mb_errs = []
    c_md_errs = []
    c_mg_errs = []
    c_mt_errs = []
    b_neg = 0
    c_neg = 0

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        Mb_s = mm["bulge"]["M"]
        ab = mm["bulge"]["a"]
        Md_s = mm["disk"]["M"]
        Rd = mm["disk"]["Rd"]
        Mg_s = mm["gas"]["M"]
        Rg = mm["gas"]["Rd"]
        Mt_s = Mb_s + Md_s + Mg_s

        # Step 1: topology geometry
        geom = solve_field_geometry(Mb_s, ab, Md_s, Rd, Mg_s, Rg, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)
        yN_at_rt = geom.get("yN_at_throat", 0.0)
        if not r_env or not r_t or r_env <= 0 or r_t <= 0:
            print("  %-12s  SKIP (no geometry)" % gid)
            continue

        # Step 2: Method B for M_total and M_gas
        Mt_b, Mg_b, sol_b = method_b_get_mg(
            r_env, r_t, cycle, yN_at_rt, ab, Rd, Rg, a0_eff)
        if Mt_b is None or sol_b is None:
            print("  %-12s  SKIP (singular)" % gid)
            continue

        Mb_b, Md_b, Mg_b_val = sol_b
        if Mb_b < 0 or Md_b < 0 or Mg_b_val < 0:
            b_neg += 1

        # Step 3: Self-correction scan
        # Use Method B's M_total and M_gas, scan M_b using velocity curve
        Mb_c, Md_c, chi2_c = scan_bulge_fraction(
            obs, Mt_b, Mg_b_val, ab, Rd, Rg, a0_eff)
        Mg_c = Mg_b_val  # gas unchanged

        if Mb_c < 0 or Md_c < 0:
            c_neg += 1

        # Errors
        e_mb_b = pct(Mb_b, Mb_s) if Mb_s > 0 else 0
        e_mb_c = pct(Mb_c, Mb_s) if Mb_s > 0 else 0
        e_md_c = pct(Md_c, Md_s) if Md_s > 0 else 0
        e_mg_c = pct(Mg_c, Mg_s) if Mg_s > 0 else 0
        e_mt_c = pct(Mb_c + Md_c + Mg_c, Mt_s) if Mt_s > 0 else 0

        if Mb_s > 0:
            b_mb_errs.append(abs(e_mb_b))
            c_mb_errs.append(abs(e_mb_c))
        if Md_s > 0:
            c_md_errs.append(abs(e_md_c))
        if Mg_s > 0:
            c_mg_errs.append(abs(e_mg_c))
        c_mt_errs.append(abs(e_mt_c))

        print("  %-12s %3d | %9s %9s %9s %9s | %9s %9s %9s %9s | %9s %9s %9s | %+5.0f%% %+5.0f%% %+5.0f%%" % (
            gid, cycle,
            fmt(Mb_s), fmt(Md_s), fmt(Mg_s), fmt(Mt_s),
            fmt(Mb_b), fmt(Md_b), fmt(Mg_b_val), fmt(Mt_b),
            fmt(Mb_c), fmt(Md_c), fmt(Mg_c),
            e_mb_b, e_mb_c, e_md_c))

    # Summary
    print()
    print("=" * 150)
    print("  SUMMARY")
    print("=" * 150)
    print()

    def median(arr):
        if not arr:
            return 0
        s = sorted(arr)
        return s[len(s) // 2]

    def mean(arr):
        if not arr:
            return 0
        return sum(arr) / len(arr)

    print("  Method B (residual): M_bulge median|err| = %.1f%%  (N=%d galaxies with Mb>0)" % (
        median(b_mb_errs), len(b_mb_errs)))
    print("  Self-corrected:      M_bulge median|err| = %.1f%%  (N=%d)" % (
        median(c_mb_errs), len(c_mb_errs)))
    print("  Self-corrected:      M_disk  median|err| = %.1f%%  (N=%d)" % (
        median(c_md_errs), len(c_md_errs)))
    print("  Self-corrected:      M_gas   median|err| = %.1f%%  (N=%d, unchanged from Method B)" % (
        median(c_mg_errs), len(c_mg_errs)))
    print("  Self-corrected:      M_total median|err| = %.1f%%  (N=%d, budget-constrained)" % (
        median(c_mt_errs), len(c_mt_errs)))
    print()
    print("  Method B negatives: %d / 22" % b_neg)
    print("  Self-corrected negatives: %d / 22" % c_neg)
    print()
    print("  The key question: does using the velocity curve shape")
    print("  to find the M_b vs M_d split beat the 67%% residual method?")


if __name__ == "__main__":
    main()
