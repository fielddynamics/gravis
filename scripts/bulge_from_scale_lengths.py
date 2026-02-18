#!/usr/bin/env python3
"""
bulge_from_scale_lengths.py
============================
Test: can we predict Mb from the scale length ratio ab/Rd?

Chain:
  1. M_total from topology (0.1%)
  2. M_gas from topology (2.5-7%)
  3. M_stellar = M_total - M_gas
  4. Mb/Md = 2.007 * (ab/Rd) - 0.124
  5. Mb = M_stellar * ratio / (1 + ratio)
  6. Md = M_stellar - Mb

Compare against SPARC published values for all 22 galaxies.
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
        A_mat = [[fb_env, fd_env, fg_env],
                  [fb_t,   fd_t,   fg_t],
                  [0.0,    1.0,    -ratio]]
        b = [M_horizon, M_throat, 0.0]
        sol = solve_3x3(A_mat, b)
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
    print("=" * 140)
    print("  BULGE FROM SCALE LENGTH RATIO: Mb/Md = 2.007 * (ab/Rd) - 0.124")
    print("  M_total and M_gas from topology. Mb and Md from scale length ratio.")
    print("=" * 140)
    print()

    errs_methB = {"Mb": [], "Md": [], "Mg": []}
    errs_scale = {"Mb": [], "Md": [], "Mg": []}

    print("  %-12s %3s | %5s %5s %6s | --- Method B --- | -- Scale Length -- | Improvement" % (
        "Galaxy", "Cyc", "fb%", "ab/Rd", "Mb/Md",))
    print("  %-12s %3s | %5s %5s %6s | %6s %6s %6s | %6s %6s %6s | %6s %6s" % (
        "", "", "", "", "",
        "eMb%", "eMd%", "eMg%",
        "eMb%", "eMd%", "eMg%",
        "dMb", "dMd"))
    print("  " + "-" * 136)

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

        geom = solve_field_geometry(Mb_s, ab_s, Md_s, Rd_s, Mg_s, Rg_s, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)
        yN_at_rt = geom.get("yN_at_throat", 0.0)
        if not r_env or not r_t or r_env <= 0 or r_t <= 0:
            continue

        # Method B baseline
        sol_b = method_b(r_env, r_t, cycle, yN_at_rt, ab_s, Rd_s, Rg_s, a0_eff)
        if sol_b is None:
            continue
        Mb_b, Md_b, Mg_b = sol_b
        Mt_b = Mb_b + Md_b + Mg_b

        # Scale length prediction
        Mg_locked = max(Mg_b, 0)
        M_stellar = Mt_b - Mg_locked

        ab_Rd = ab_s / Rd_s if Rd_s > 0 else 0
        mb_md_ratio = 2.007 * ab_Rd - 0.124
        mb_md_ratio = max(mb_md_ratio, 0)

        if mb_md_ratio > 0 and M_stellar > 0:
            Mb_pred = M_stellar * mb_md_ratio / (1.0 + mb_md_ratio)
            Md_pred = M_stellar - Mb_pred
        else:
            Mb_pred = 0
            Md_pred = M_stellar

        # Errors
        eMb_b = pct(Mb_b, Mb_s) if Mb_s > 0 else 0
        eMd_b = pct(Md_b, Md_s) if Md_s > 0 else 0
        eMg_b = pct(Mg_b, Mg_s) if Mg_s > 0 else 0

        eMb_p = pct(Mb_pred, Mb_s) if Mb_s > 0 else 0
        eMd_p = pct(Md_pred, Md_s) if Md_s > 0 else 0
        eMg_p = pct(Mg_locked, Mg_s) if Mg_s > 0 else 0

        # Track
        if Mb_s > 0:
            errs_methB["Mb"].append(abs(eMb_b))
            errs_scale["Mb"].append(abs(eMb_p))
        if Md_s > 0:
            errs_methB["Md"].append(abs(eMd_b))
            errs_scale["Md"].append(abs(eMd_p))
        if Mg_s > 0:
            errs_methB["Mg"].append(abs(eMg_b))
            errs_scale["Mg"].append(abs(eMg_p))

        def cfmt(v):
            if abs(v) > 999:
                return ">999%"
            return "%+5.0f%%" % v

        # Did scale method improve over Method B?
        imp_mb = ""
        imp_md = ""
        if Mb_s > 0:
            if abs(eMb_p) < abs(eMb_b):
                imp_mb = "  YES"
            else:
                imp_mb = "   no"
        if Md_s > 0:
            if abs(eMd_p) < abs(eMd_b):
                imp_md = "  YES"
            else:
                imp_md = "   no"

        actual_mb_md = Mb_s / Md_s if Md_s > 0 else 0

        print("  %-12s %3d | %4.1f%% %5.3f %6.3f | %6s %6s %6s | %6s %6s %6s | %5s %5s" % (
            gid, cycle, fb_s, ab_Rd, actual_mb_md,
            cfmt(eMb_b), cfmt(eMd_b), cfmt(eMg_b),
            cfmt(eMb_p), cfmt(eMd_p), cfmt(eMg_p),
            imp_mb, imp_md))

    # Summary
    print()
    print("=" * 140)
    print("  SUMMARY")
    print("=" * 140)
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

    for label, errs in [("Method B (topology 3x3)", errs_methB),
                         ("Scale length predictor (ab/Rd)", errs_scale)]:
        print("  %s:" % label)
        for k, name in [("Mb", "M_bulge"), ("Md", "M_disk"), ("Mg", "M_gas")]:
            arr = errs[k]
            if arr:
                print("    %-10s  median = %5.1f%%   mean = %5.1f%%   (N=%d)" % (
                    name, median(arr), mean(arr), len(arr)))
        print()

    # Exclude outliers
    EXCLUDE = {"ngc891", "ngc5055", "ic2574"}
    errs_clean = {"Mb": [], "Md": [], "Mg": []}

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid in EXCLUDE:
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
        Mb_b, Md_b, Mg_b = sol_b
        Mt_b = Mb_b + Md_b + Mg_b
        Mg_locked = max(Mg_b, 0)
        M_stellar = Mt_b - Mg_locked
        ab_Rd = ab_s / Rd_s if Rd_s > 0 else 0
        mb_md_ratio = max(2.007 * ab_Rd - 0.124, 0)
        if mb_md_ratio > 0 and M_stellar > 0:
            Mb_pred = M_stellar * mb_md_ratio / (1.0 + mb_md_ratio)
            Md_pred = M_stellar - Mb_pred
        else:
            Mb_pred = 0
            Md_pred = M_stellar

        if Mb_s > 0:
            errs_clean["Mb"].append(abs(pct(Mb_pred, Mb_s)))
        if Md_s > 0:
            errs_clean["Md"].append(abs(pct(Md_pred, Md_s)))
        if Mg_s > 0:
            errs_clean["Mg"].append(abs(pct(Mg_locked, Mg_s)))

    print("  Scale length predictor (excluding NGC891, NGC5055, IC2574):")
    for k, name in [("Mb", "M_bulge"), ("Md", "M_disk"), ("Mg", "M_gas")]:
        arr = errs_clean[k]
        if arr:
            print("    %-10s  median = %5.1f%%   mean = %5.1f%%   (N=%d)" % (
                name, median(arr), mean(arr), len(arr)))


if __name__ == "__main__":
    main()
