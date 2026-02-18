#!/usr/bin/env python3
"""
photometric_correction.py
==========================
Test: use SPARC photometric Mb/Md ratio to split topology-derived M_stellar.

The chain:
  1. M_total from topology (0.1%)
  2. M_gas from topology (2.5-7%)
  3. M_stellar = M_total - M_gas
  4. Mb/Md = SPARC photometric ratio (from 3.6um luminosity decomposition)
  5. Mb = M_stellar * ratio / (1 + ratio)
  6. Md = M_stellar - Mb

The ONLY errors come from M_total and M_gas. The photometric ratio
is an independent measurement, not fitted to the rotation curve.

Also tests: what if we use our ab/Rd scale length predictor INSTEAD
of the actual photometric ratio? Comparison shows how much the
photometry adds beyond what scale lengths already give.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0, G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.services.rotation.inference import solve_field_geometry


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

def method_b(r_env, r_t, cycle, yN_at_rt, ab, Rd, Rg, a0_eff):
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
            return None
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
    return sol


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
    print("=" * 145)
    print("  PHOTOMETRIC CORRECTION TEST")
    print("  Topology: M_total, M_gas. Photometry: Mb/Md ratio. Combined: full decomposition.")
    print("=" * 145)
    print()

    errs_B = {"Mb": [], "Md": [], "Mg": [], "Mt": []}
    errs_SL = {"Mb": [], "Md": [], "Mg": [], "Mt": []}
    errs_PH = {"Mb": [], "Md": [], "Mg": [], "Mt": []}

    print("  %-12s %3s %5s | %6s | --- Method B --- | -- Scale (ab/Rd) -- | -- Photometric ----" % (
        "Galaxy", "Cyc", "fb%", "Mb/Md"))
    print("  %-12s %3s %5s | %6s | %6s %6s %6s | %6s %6s %6s  | %6s %6s %6s" % (
        "", "", "", "",
        "eMb%", "eMd%", "eMg%",
        "eMb%", "eMd%", "eMg%",
        "eMb%", "eMd%", "eMg%"))
    print("  " + "-" * 141)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel

        Mb_s = mm["bulge"]["M"]
        ab_s = mm["bulge"]["a"]
        Md_s = mm["disk"]["M"]
        Rd_s = mm["disk"]["Rd"]
        Mg_s = mm["gas"]["M"]
        Rg_s = mm["gas"]["Rd"]
        Mt_s = Mb_s + Md_s + Mg_s
        fb_s = Mb_s / Mt_s * 100 if Mt_s > 0 else 0
        photo_ratio = Mb_s / Md_s if Md_s > 0 else 0

        geom = solve_field_geometry(Mb_s, ab_s, Md_s, Rd_s, Mg_s, Rg_s, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)
        yN_at_rt = geom.get("yN_at_throat", 0.0)
        if not r_env or not r_t or r_env <= 0 or r_t <= 0:
            continue

        # Method B: topology baseline
        sol_b = method_b(r_env, r_t, cycle, yN_at_rt, ab_s, Rd_s, Rg_s, a0_eff)
        if sol_b is None:
            continue
        Mb_b, Md_b, Mg_b = sol_b
        Mt_b = Mb_b + Md_b + Mg_b
        Mg_locked = max(Mg_b, 0)
        M_stellar = Mt_b - Mg_locked

        # Scale length predictor: Mb/Md = 2.007 * (ab/Rd) - 0.124
        ab_Rd = ab_s / Rd_s if Rd_s > 0 else 0
        sl_ratio = max(2.007 * ab_Rd - 0.124, 0)
        if sl_ratio > 0 and M_stellar > 0:
            Mb_sl = M_stellar * sl_ratio / (1.0 + sl_ratio)
            Md_sl = M_stellar - Mb_sl
        else:
            Mb_sl = 0
            Md_sl = M_stellar

        # Photometric: use SPARC Mb/Md ratio directly
        if photo_ratio > 0 and M_stellar > 0:
            Mb_ph = M_stellar * photo_ratio / (1.0 + photo_ratio)
            Md_ph = M_stellar - Mb_ph
        else:
            Mb_ph = 0
            Md_ph = M_stellar

        # Errors
        def get_errs(Mb_p, Md_p, Mg_p, errs_dict):
            eMt = pct(Mb_p + Md_p + Mg_p, Mt_s) if Mt_s > 0 else 0
            eMb = pct(Mb_p, Mb_s) if Mb_s > 0 else 0
            eMd = pct(Md_p, Md_s) if Md_s > 0 else 0
            eMg = pct(Mg_p, Mg_s) if Mg_s > 0 else 0
            errs_dict["Mt"].append(abs(eMt))
            if Mb_s > 0:
                errs_dict["Mb"].append(abs(eMb))
            if Md_s > 0:
                errs_dict["Md"].append(abs(eMd))
            if Mg_s > 0:
                errs_dict["Mg"].append(abs(eMg))
            return eMb, eMd, eMg

        eMb_b, eMd_b, eMg_b = get_errs(Mb_b, Md_b, Mg_b, errs_B)
        eMb_sl, eMd_sl, eMg_sl = get_errs(Mb_sl, Md_sl, Mg_locked, errs_SL)
        eMb_ph, eMd_ph, eMg_ph = get_errs(Mb_ph, Md_ph, Mg_locked, errs_PH)

        def cfmt(v):
            if abs(v) > 999:
                return ">999%"
            return "%+5.0f%%" % v

        print("  %-12s %3d %4.1f%% | %6.3f | %6s %6s %6s | %6s %6s %6s  | %6s %6s %6s" % (
            gid, cycle, fb_s, photo_ratio,
            cfmt(eMb_b), cfmt(eMd_b), cfmt(eMg_b),
            cfmt(eMb_sl), cfmt(eMd_sl), cfmt(eMg_sl),
            cfmt(eMb_ph), cfmt(eMd_ph), cfmt(eMg_ph)))

    # Summary
    print()
    print("=" * 145)
    print("  SUMMARY (all 22 galaxies)")
    print("=" * 145)
    print()

    def median(arr):
        if not arr:
            return float('nan')
        s = sorted(arr)
        return s[len(s) // 2]
    def mean(arr):
        if not arr:
            return float('nan')
        return sum(arr) / len(arr)

    for label, errs in [
        ("Method B (topology 3x3, no photometry)", errs_B),
        ("Scale length predictor (ab/Rd proxy)", errs_SL),
        ("Photometric ratio (SPARC Mb/Md exact)", errs_PH),
    ]:
        print("  %s:" % label)
        for k, name in [("Mt", "M_total"), ("Mb", "M_bulge"), ("Md", "M_disk"), ("Mg", "M_gas")]:
            arr = errs[k]
            if arr:
                print("    %-10s  median = %5.1f%%   mean = %5.1f%%   (N=%d)" % (
                    name, median(arr), mean(arr), len(arr)))
        print()

    # Excluding outliers
    EXCLUDE = {"ngc891", "ngc5055", "ic2574"}
    print("  " + "-" * 60)
    print("  Excluding NGC891, NGC5055, IC2574:")
    print()

    for label, errs_full in [
        ("Scale length (ab/Rd)", errs_SL),
        ("Photometric (Mb/Md)", errs_PH),
    ]:
        # Rebuild excluding outliers
        errs_c = {"Mb": [], "Md": [], "Mg": [], "Mt": []}
        for gal in PREDICTION_GALAXIES:
            if gal["id"] in EXCLUDE:
                continue
            mm = gal["mass_model"]
            accel = gal.get("accel", 1.0)
            a0_eff = A0 * accel
            Mb_s = mm["bulge"]["M"]
            ab_s = mm["bulge"]["a"]
            Md_s = mm["disk"]["M"]
            Rd_s = mm["disk"]["Rd"]
            Mg_s = mm["gas"]["M"]
            Rg_s = mm["gas"]["Rd"]
            Mt_s = Mb_s + Md_s + Mg_s

            geom = solve_field_geometry(Mb_s, ab_s, Md_s, Rd_s, Mg_s, Rg_s, a0_eff)
            r_t = geom.get("throat_radius_kpc")
            r_env = geom.get("envelope_radius_kpc")
            cycle = geom.get("cycle", 3)
            yN_at_rt = geom.get("yN_at_throat", 0.0)
            if not r_env or not r_t or r_env <= 0 or r_t <= 0:
                continue

            sol_b = method_b(r_env, r_t, cycle, yN_at_rt, ab_s, Rd_s, Rg_s, a0_eff)
            if sol_b is None:
                continue
            Mt_b = sol_b[0] + sol_b[1] + sol_b[2]
            Mg_locked = max(sol_b[2], 0)
            M_stellar = Mt_b - Mg_locked

            if "Scale" in label:
                ab_Rd = ab_s / Rd_s if Rd_s > 0 else 0
                r = max(2.007 * ab_Rd - 0.124, 0)
            else:
                r = Mb_s / Md_s if Md_s > 0 else 0

            if r > 0 and M_stellar > 0:
                Mb_p = M_stellar * r / (1.0 + r)
                Md_p = M_stellar - Mb_p
            else:
                Mb_p = 0
                Md_p = M_stellar

            eMt = pct(Mb_p + Md_p + Mg_locked, Mt_s)
            errs_c["Mt"].append(abs(eMt))
            if Mb_s > 0:
                errs_c["Mb"].append(abs(pct(Mb_p, Mb_s)))
            if Md_s > 0:
                errs_c["Md"].append(abs(pct(Md_p, Md_s)))
            if Mg_s > 0:
                errs_c["Mg"].append(abs(pct(Mg_locked, Mg_s)))

        print("  %s (clean 19):" % label)
        for k, name in [("Mt", "M_total"), ("Mb", "M_bulge"), ("Md", "M_disk"), ("Mg", "M_gas")]:
            arr = errs_c[k]
            if arr:
                print("    %-10s  median = %5.1f%%   mean = %5.1f%%   (N=%d)" % (
                    name, median(arr), mean(arr), len(arr)))
        print()


if __name__ == "__main__":
    main()
