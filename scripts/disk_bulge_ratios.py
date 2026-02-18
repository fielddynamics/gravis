#!/usr/bin/env python3
"""
disk_bulge_ratios.py
=====================
With M_total and M_gas nailed, explore whether disk and bulge
have a predictable relationship: Mb/Md, Mb/(Mb+Md), Md/M_stellar,
or any ratio that correlates with known quantities.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0
from physics.services.rotation.inference import solve_field_geometry


def pearson(x, y):
    n = len(x)
    if n < 3:
        return 0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sxx * syy == 0:
        return 0
    return sxy / math.sqrt(sxx * syy)


def main():
    data = []
    for gal in PREDICTION_GALAXIES:
        mm = gal["mass_model"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel
        Mb = mm["bulge"]["M"]
        ab = mm["bulge"]["a"]
        Md = mm["disk"]["M"]
        Rd = mm["disk"]["Rd"]
        Mg = mm["gas"]["M"]
        Rg = mm["gas"]["Rd"]
        Mt = Mb + Md + Mg
        M_stellar = Mb + Md
        if Mt <= 0:
            continue

        geom = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)

        fg = Mg / Mt
        fd = Md / Mt
        fb = Mb / Mt
        f_stellar = M_stellar / Mt

        data.append({
            "id": gal["id"], "cycle": cycle,
            "Mb": Mb, "Md": Md, "Mg": Mg, "Mt": Mt,
            "M_stellar": M_stellar,
            "ab": ab, "Rd": Rd, "Rg": Rg,
            "r_t": r_t or 0, "r_env": r_env or 0,
            "fb": fb, "fd": fd, "fg": fg,
            "f_stellar": f_stellar,
            "Mb_Md": Mb / Md if Md > 0 else 0,
            "Mb_Mstar": Mb / M_stellar if M_stellar > 0 else 0,
            "ab_Rd": ab / Rd if Rd > 0 else 0,
        })

    data.sort(key=lambda d: d["fb"])

    # Table 1: All ratios
    print("=" * 130)
    print("  DISK/BULGE RATIO SURVEY (sorted by bulge fraction)")
    print("=" * 130)
    print()
    print("  %-12s %3s  %6s %6s %6s | %6s %7s %7s | %6s %6s %6s | %6s" % (
        "Galaxy", "Cyc", "fb%", "fd%", "fg%",
        "Mb/Md", "Mb/Mst", "ab/Rd",
        "ab", "Rd", "Rg", "Rt"))
    print("  " + "-" * 126)

    for d in data:
        print("  %-12s %3d  %5.1f%% %5.1f%% %5.1f%% | %6.3f %7.3f %7.3f | %6.2f %6.2f %6.2f | %6.2f" % (
            d["id"], d["cycle"],
            d["fb"] * 100, d["fd"] * 100, d["fg"] * 100,
            d["Mb_Md"], d["Mb_Mstar"], d["ab_Rd"],
            d["ab"], d["Rd"], d["Rg"], d["r_t"]))

    # Only galaxies with actual bulge (Mb > 0)
    bulge_data = [d for d in data if d["Mb"] > 0]

    print()
    print("=" * 130)
    print("  CORRELATIONS (galaxies with Mb > 0, N=%d)" % len(bulge_data))
    print("=" * 130)
    print()

    # Extract arrays for correlation
    fb_arr = [d["fb"] for d in bulge_data]
    fd_arr = [d["fd"] for d in bulge_data]
    fg_arr = [d["fg"] for d in bulge_data]
    mb_md_arr = [d["Mb_Md"] for d in bulge_data]
    mb_mstar_arr = [d["Mb_Mstar"] for d in bulge_data]
    ab_rd_arr = [d["ab_Rd"] for d in bulge_data]
    fstar_arr = [d["f_stellar"] for d in bulge_data]
    rt_arr = [d["r_t"] for d in bulge_data]
    renv_arr = [d["r_env"] for d in bulge_data]
    rd_arr = [d["Rd"] for d in bulge_data]
    ab_arr = [d["ab"] for d in bulge_data]
    rt_rd_arr = [d["r_t"] / d["Rd"] if d["Rd"] > 0 else 0 for d in bulge_data]
    rt_ab_arr = [d["r_t"] / d["ab"] if d["ab"] > 0 else 0 for d in bulge_data]
    md_arr = [d["Md"] for d in bulge_data]
    mb_arr = [d["Mb"] for d in bulge_data]
    log_mt_arr = [math.log10(d["Mt"]) for d in bulge_data]
    log_mb_arr = [math.log10(max(d["Mb"], 1)) for d in bulge_data]
    log_md_arr = [math.log10(max(d["Md"], 1)) for d in bulge_data]

    pairs = [
        ("fb vs fg", fb_arr, fg_arr),
        ("fb vs fd", fb_arr, fd_arr),
        ("fb vs f_stellar", fb_arr, fstar_arr),
        ("Mb/Md vs fg", mb_md_arr, fg_arr),
        ("Mb/Md vs fd", mb_md_arr, fd_arr),
        ("Mb/Md vs fb", mb_md_arr, fb_arr),
        ("Mb/M_star vs fg", mb_mstar_arr, fg_arr),
        ("Mb/M_star vs fd", mb_mstar_arr, fd_arr),
        ("Mb/M_star vs f_stellar", mb_mstar_arr, fstar_arr),
        ("ab/Rd vs fb", ab_rd_arr, fb_arr),
        ("ab/Rd vs Mb/Md", ab_rd_arr, mb_md_arr),
        ("ab/Rd vs Mb/M_star", ab_rd_arr, mb_mstar_arr),
        ("Rt/Rd vs fb", rt_rd_arr, fb_arr),
        ("Rt/Rd vs Mb/Md", rt_rd_arr, mb_md_arr),
        ("Rt/ab vs fb", rt_ab_arr, fb_arr),
        ("log(Mb) vs log(Md)", log_mb_arr, log_md_arr),
        ("log(Mb) vs log(Mt)", log_mb_arr, log_mt_arr),
        ("fb vs log(Mt)", fb_arr, log_mt_arr),
        ("Mb/Md vs log(Mt)", mb_md_arr, log_mt_arr),
    ]

    print("  %-25s  Pearson r  Interpretation" % "Pair")
    print("  " + "-" * 65)
    for name, x, y in pairs:
        r = pearson(x, y)
        strength = ""
        if abs(r) > 0.9:
            strength = "*** STRONG ***"
        elif abs(r) > 0.7:
            strength = "** good **"
        elif abs(r) > 0.5:
            strength = "* moderate *"
        print("  %-25s  %+6.3f     %s" % (name, r, strength))

    # Detailed look at the strongest correlations
    print()
    print("=" * 130)
    print("  BEST CORRELATIONS: DETAILED")
    print("=" * 130)

    # ab/Rd vs Mb/Md
    print()
    print("  ab/Rd vs Mb/Md (scale length ratio vs mass ratio):")
    print("  %-12s  %6s  %6s  %7s" % ("Galaxy", "ab/Rd", "Mb/Md", "fb%"))
    print("  " + "-" * 40)
    for d in sorted(bulge_data, key=lambda d: d["ab_Rd"]):
        print("  %-12s  %6.3f  %6.3f  %5.1f%%" % (
            d["id"], d["ab_Rd"], d["Mb_Md"], d["fb"] * 100))

    # Mb/M_stellar: does it cluster?
    print()
    print("  Mb/M_stellar (bulge-to-stellar ratio):")
    print("  %-12s  %7s  %5s  %5s" % ("Galaxy", "Mb/Mst", "fb%", "fg%"))
    print("  " + "-" * 40)
    for d in sorted(bulge_data, key=lambda d: d["Mb_Mstar"]):
        print("  %-12s  %7.4f  %5.1f  %5.1f" % (
            d["id"], d["Mb_Mstar"], d["fb"] * 100, d["fg"] * 100))

    # Linear fit: Mb/Md = a * (ab/Rd) + b
    if len(bulge_data) > 2:
        x = [d["ab_Rd"] for d in bulge_data]
        y = [d["Mb_Md"] for d in bulge_data]
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        sxx = sum((xi - mx) ** 2 for xi in x)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        if sxx > 0:
            slope = sxy / sxx
            intercept = my - slope * mx
            print()
            print("  Linear fit: Mb/Md = %.3f * (ab/Rd) + %.3f" % (slope, intercept))
            print("  Pearson r = %.3f" % pearson(x, y))

    # log-log: log(Mb) vs log(Md)
    if len(bulge_data) > 2:
        x = log_md_arr
        y = log_mb_arr
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        sxx = sum((xi - mx) ** 2 for xi in x)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        if sxx > 0:
            slope = sxy / sxx
            intercept = my - slope * mx
            print()
            print("  Power law: log(Mb) = %.3f * log(Md) + %.3f" % (slope, intercept))
            print("  => Mb = 10^(%.3f) * Md^(%.3f)" % (intercept, slope))
            print("  Pearson r = %.3f" % pearson(x, y))


if __name__ == "__main__":
    main()
