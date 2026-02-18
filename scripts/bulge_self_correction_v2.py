#!/usr/bin/env python3
"""
bulge_self_correction_v2.py
============================
Three approaches to improving M_bulge once M_total and M_gas are known:

  Method B: Original topology 3x3 system (baseline, ~67% median error on Mb)

  Method C (throat extraction):
    With M_d from fd-fg relationship and M_g from topology, extract M_b
    directly from the throat equation:
    M_b = (M_throat - M_d*h_d(Rt) - M_g*h_g(Rt)) / h_b(Rt)

  Method D (inner-weighted velocity scan):
    Same 1D scan as v1, but weight inner points (r < R_t) much more heavily,
    since that's where the Hernquist bulge shape has actual leverage.

  Method E (combined: throat prior + velocity refinement):
    Use Method C's M_b as a Bayesian prior, then scan with velocity data
    to refine within the prior's uncertainty band.
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

def gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    enc = hernquist_enc(r_kpc, Mb, ab) + disk_enc(r_kpc, Md, Rd) + disk_enc(r_kpc, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0

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


def method_b(r_env, r_t, cycle, yN_at_rt, ab, Rd, Rg, a0_eff):
    """Method B: 3x3 with continuous fd-fg relationship."""
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
        A = [[fb_env, fd_env, fg_env],
             [fb_t,   fd_t,   fg_t],
             [0.0,    1.0,    -ratio]]
        b = [M_horizon, M_throat, 0.0]
        sol = solve_3x3(A, b)
        if sol is None:
            return None, M_horizon, M_throat
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
    return sol, M_horizon, M_throat


def method_c_throat_extract(Mt, M_throat, Mg, ab, Rd, Rg, r_t, r_env):
    """
    Method C: Direct throat extraction.
    
    1. fd = -0.826 * (Mg/Mt) + 0.863  ->  Md = fd * Mt
    2. M_b_at_throat = M_throat - Md*h_d(Rt) - Mg*h_g(Rt)
    3. M_b = M_b_at_throat / h_b(Rt)
    4. Rebalance: M_d = Mt - M_b - Mg
    
    This uses the Hernquist enclosure fraction h_b(Rt) as a LEVER:
    since a_b << R_t, h_b(Rt) ~ 0.8-0.95, so dividing by it amplifies
    the throat signal into M_b.
    """
    if Mt <= 0:
        return None

    fg = max(Mg, 0) / Mt
    fg = max(0.01, min(fg, 0.99))
    fd = -0.826 * fg + 0.863
    fd = max(0.02, min(fd, 0.95))

    Md_init = fd * Mt
    Mg_used = Mg

    # Throat extraction
    hb_t = hernquist_frac(r_t, ab)
    hd_t = disk_frac(r_t, Rd)
    hg_t = disk_frac(r_t, Rg)

    if hb_t < 0.01:
        return None

    Mb_throat = (M_throat - Md_init * hd_t - Mg_used * hg_t) / hb_t
    Mb_throat = max(0, Mb_throat)

    # Budget: Md = Mt - Mb - Mg
    Md_final = Mt - Mb_throat - Mg_used

    if Md_final < 0:
        # Bulge was too big, clamp
        Mb_throat = Mt - Mg_used
        Md_final = 0

    return (Mb_throat, Md_final, Mg_used)


def method_e_combined(obs, Mt, M_throat, Mg_b, ab, Rd, Rg, r_t, a0_eff):
    """
    Method E: Use throat extraction as prior, velocity scan to refine.
    
    1. Get M_b_prior from throat extraction (Method C)
    2. Scan M_b in range [0.3*prior, 3*prior], using inner-weighted chi2
    3. Combine: chi2_total = chi2_velocity + lambda * (M_b - M_b_prior)^2
    """
    # Throat prior
    sol_c = method_c_throat_extract(Mt, M_throat, Mg_b, ab, Rd, Rg, r_t, 0)
    if sol_c is None:
        return None
    Mb_prior = sol_c[0]

    # If prior is 0, just use budget
    if Mb_prior <= 0:
        Md = Mt - Mg_b
        return (0, max(Md, 0), Mg_b)

    # Scan range
    lo = 0
    hi = min(Mb_prior * 4, (Mt - Mg_b) * 0.5)
    if hi <= lo:
        return sol_c

    # Prior strength: sigma_prior = 50% of Mb_prior
    sigma_prior = max(Mb_prior * 0.5, 1e6)

    n_steps = 200
    best_cost = float('inf')
    best_Mb = Mb_prior

    for i in range(n_steps + 1):
        Mb = lo + (hi - lo) * i / n_steps
        Md = Mt - Mb - Mg_b
        if Md < 0:
            continue

        # Velocity chi2, inner-weighted
        chi2 = 0.0
        for pt in obs:
            r, v_obs, err = pt["r"], pt["v"], pt["err"]
            if r <= 0 or v_obs <= 0:
                continue
            v_pred = gfd_velocity(r, Mb, ab, Md, Rd, Mg_b, Rg, a0_eff)
            w = 1.0 / (err * err) if err > 0 else 1.0
            # Upweight inner points where bulge matters
            if r < r_t:
                w *= 3.0
            chi2 += w * (v_pred - v_obs) ** 2

        # Prior penalty
        prior_cost = ((Mb - Mb_prior) / sigma_prior) ** 2
        total = chi2 + prior_cost

        if total < best_cost:
            best_cost = total
            best_Mb = Mb

    best_Md = Mt - best_Mb - Mg_b
    return (best_Mb, max(best_Md, 0), Mg_b)


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
    print("=" * 130)
    print("  BULGE SELF-CORRECTION v2: Three Strategies")
    print("  All use M_total and M_gas from topology (Method B baseline)")
    print("=" * 130)
    print()

    # Collect per-method errors for galaxies with Mb > 0
    errs_b = {"Mb": [], "Md": [], "Mg": []}
    errs_c = {"Mb": [], "Md": [], "Mg": []}
    errs_e = {"Mb": [], "Md": [], "Mg": []}
    neg_b = 0
    neg_c = 0
    neg_e = 0

    print("  %-12s %3s  %9s %9s %9s | ---- Method B ---- | -- C: Throat Ext -- | -- E: Combined ---" % (
        "Galaxy", "Cyc", "Mb_sparc", "Md_sparc", "Mg_sparc"))
    print("  %-12s %3s  %9s %9s %9s | %7s %7s %7s | %7s %7s %7s | %7s %7s %7s" % (
        "", "", "", "", "", "eMb%", "eMd%", "eMg%", "eMb%", "eMd%", "eMg%", "eMb%", "eMd%", "eMg%"))
    print("  " + "-" * 126)

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

        geom = solve_field_geometry(Mb_s, ab, Md_s, Rd, Mg_s, Rg, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)
        yN_at_rt = geom.get("yN_at_throat", 0.0)
        if not r_env or not r_t or r_env <= 0 or r_t <= 0:
            continue

        # Method B baseline
        sol_b, M_horizon, M_throat = method_b(
            r_env, r_t, cycle, yN_at_rt, ab, Rd, Rg, a0_eff)
        if sol_b is None:
            continue
        Mb_b, Md_b, Mg_b = sol_b
        Mt_b = Mb_b + Md_b + Mg_b

        if Mb_b < 0 or Md_b < 0 or Mg_b < 0:
            neg_b += 1

        # Method C: Throat extraction (using Method B's Mt and Mg)
        sol_c = method_c_throat_extract(Mt_b, M_throat, Mg_b, ab, Rd, Rg, r_t, r_env)

        # Method E: Combined throat prior + inner velocity
        sol_e = method_e_combined(obs, Mt_b, M_throat, Mg_b, ab, Rd, Rg, r_t, a0_eff)

        # Compute errors
        def get_errs(sol, errs_dict, label):
            nonlocal neg_c, neg_e
            if sol is None:
                return "   N/A  ", "   N/A  ", "   N/A  "
            Mb_p, Md_p, Mg_p = sol
            if Mb_p < 0 or Md_p < 0 or Mg_p < 0:
                if label == "C":
                    neg_c += 1
                elif label == "E":
                    neg_e += 1

            eMb = pct(Mb_p, Mb_s) if Mb_s > 0 else 0
            eMd = pct(Md_p, Md_s) if Md_s > 0 else 0
            eMg = pct(Mg_p, Mg_s) if Mg_s > 0 else 0

            # Only track galaxies with actual bulge (Mb_s > 0)
            if Mb_s > 0 and Mb_p >= 0 and Md_p >= 0:
                errs_dict["Mb"].append(abs(eMb))
            if Md_s > 0:
                errs_dict["Md"].append(abs(eMd))
            if Mg_s > 0:
                errs_dict["Mg"].append(abs(eMg))

            def cfmt(v):
                if abs(v) > 999:
                    return " >999%"
                return "%+6.0f%%" % v
            return cfmt(eMb), cfmt(eMd), cfmt(eMg)

        b_str = get_errs(sol_b, errs_b, "B")
        c_str = get_errs(sol_c, errs_c, "C")
        e_str = get_errs(sol_e, errs_e, "E")

        print("  %-12s %3d  %9s %9s %9s | %7s %7s %7s | %7s %7s %7s | %7s %7s %7s" % (
            gid, cycle, fmt(Mb_s), fmt(Md_s), fmt(Mg_s),
            b_str[0], b_str[1], b_str[2],
            c_str[0], c_str[1], c_str[2],
            e_str[0], e_str[1], e_str[2]))

    # Summary
    print()
    print("=" * 130)
    print("  SUMMARY (median |error| for galaxies with published Mb > 0)")
    print("=" * 130)
    print()

    def median(arr):
        if not arr:
            return float('nan')
        s = sorted(arr)
        return s[len(s) // 2]

    for label, errs, neg in [("Method B (3x3 topology)", errs_b, neg_b),
                              ("Method C (throat extraction)", errs_c, neg_c),
                              ("Method E (combined prior+vel)", errs_e, neg_e)]:
        print("  %s:" % label)
        for k in ["Mb", "Md", "Mg"]:
            arr = errs[k]
            if arr:
                print("    M_%-6s  median|err| = %6.1f%%  mean = %6.1f%%  (N=%d)" % (
                    k[1:], median(arr), sum(arr)/len(arr), len(arr)))
            else:
                print("    M_%-6s  no data" % k[1:])
        print("    Negatives: %d / 22" % neg)
        print()

    # Detailed comparison for galaxies with Mb > 0
    print("=" * 130)
    print("  ACTUAL vs PREDICTED MASSES (galaxies with Mb > 0 only)")
    print("=" * 130)
    print()
    print("  %-12s | %9s | %9s %6s | %9s %6s | %9s %6s" % (
        "Galaxy", "Mb_sparc", "Mb_B", "err", "Mb_C", "err", "Mb_E", "err"))
    print("  " + "-" * 90)

    for gal in PREDICTION_GALAXIES:
        mm = gal["mass_model"]
        Mb_s = mm["bulge"]["M"]
        if Mb_s <= 0:
            continue

        ab = mm["bulge"]["a"]
        Md_s = mm["disk"]["M"]
        Rd = mm["disk"]["Rd"]
        Mg_s = mm["gas"]["M"]
        Rg = mm["gas"]["Rd"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        geom = solve_field_geometry(Mb_s, ab, Md_s, Rd, Mg_s, Rg, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)
        yN_at_rt = geom.get("yN_at_throat", 0.0)
        if not r_env or not r_t:
            continue

        sol_b, M_hor, M_thr = method_b(r_env, r_t, cycle, yN_at_rt, ab, Rd, Rg, a0_eff)
        if sol_b is None:
            continue
        Mt_b = sol_b[0] + sol_b[1] + sol_b[2]
        sol_c = method_c_throat_extract(Mt_b, M_thr, sol_b[2], ab, Rd, Rg, r_t, r_env)
        sol_e = method_e_combined(gal["observations"], Mt_b, M_thr, sol_b[2], ab, Rd, Rg, r_t, a0_eff)

        def val_err(sol, idx):
            if sol is None:
                return "   N/A   ", "  N/A"
            return fmt(sol[idx]), "%+.0f%%" % pct(sol[idx], Mb_s)

        b_v, b_e = val_err(sol_b, 0)
        c_v, c_e = val_err(sol_c, 0)
        e_v, e_e = val_err(sol_e, 0)

        print("  %-12s | %9s | %9s %6s | %9s %6s | %9s %6s" % (
            gal["id"], fmt(Mb_s), b_v, b_e, c_v, c_e, e_v, e_e))

    # Key insight section
    print()
    print("=" * 130)
    print("  KEY INSIGHT: Hernquist leverage at throat")
    print("=" * 130)
    print()
    print("  For each galaxy, show h_b(R_t) = R_t^2/(R_t+a_b)^2")
    print("  This is the fraction of bulge mass enclosed at the throat.")
    print("  Higher values = more leverage for throat extraction.")
    print()
    print("  %-12s  %6s  %6s  %7s  %6s  %7s" % (
        "Galaxy", "a_b", "R_t", "R_t/a_b", "h_b(Rt)", "fb"))
    print("  " + "-" * 55)

    for gal in PREDICTION_GALAXIES:
        mm = gal["mass_model"]
        Mb_s = mm["bulge"]["M"]
        if Mb_s <= 0:
            continue
        ab = mm["bulge"]["a"]
        Md_s = mm["disk"]["M"]
        Rd = mm["disk"]["Rd"]
        Mg_s = mm["gas"]["M"]
        Rg = mm["gas"]["Rd"]
        Mt_s = Mb_s + Md_s + Mg_s
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        geom = solve_field_geometry(Mb_s, ab, Md_s, Rd, Mg_s, Rg, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        if not r_t or r_t <= 0:
            continue
        hb = hernquist_frac(r_t, ab)
        print("  %-12s  %6.2f  %6.2f  %7.1f  %6.3f  %7.3f" % (
            gal["id"], ab, r_t, r_t / ab, hb, Mb_s / Mt_s))


if __name__ == "__main__":
    main()
