#!/usr/bin/env python3
"""
mass_ratio_survey.py
=====================
Survey the mass ratios (Mb/Mt, Md/Mt, Mg/Mt, Md/Mg, Mb/Md)
across all 22 SPARC galaxies, separated by closure cycle.

Goal: with M_gas effectively solved by topology, identify patterns
in the remaining 2 parameters (Mb, Md) that could be predicted.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0
from physics.services.rotation.inference import solve_field_geometry


def main():
    print("=" * 130)
    print("  MASS RATIO SURVEY: All 22 SPARC galaxies")
    print("  M_gas is solved by topology (~7%%). Two parameters remain: Mb, Md")
    print("=" * 130)
    print()

    header = "  %-12s %3s | %6s %6s %6s | %6s %6s %6s | %7s %7s | %7s %7s %7s | %10s"
    print(header % (
        "Galaxy", "Cyc",
        "fb", "fd", "fg",
        "Rd/Rg", "a/Rd", "a/Rg",
        "Md/Mg", "Mb/Md",
        "Rt/Renv", "Rd/Renv", "Rg/Renv",
        "M_total"))
    print("  " + "-" * 126)

    data_3 = []
    data_2 = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
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

        if Mt <= 0:
            continue

        fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        r_t = fg.get("throat_radius_kpc")
        r_env = fg.get("envelope_radius_kpc")
        cycle = fg.get("cycle", 3)

        fb = Mb / Mt
        fd = Md / Mt
        fgas = Mg / Mt

        rd_rg = Rd / Rg if Rg > 0 else 0
        a_rd = ab / Rd if Rd > 0 else 0
        a_rg = ab / Rg if Rg > 0 else 0

        md_mg = Md / Mg if Mg > 0 else 999
        mb_md = Mb / Md if Md > 0 else 0

        rt_renv = r_t / r_env if r_t and r_env and r_env > 0 else 0
        rd_renv = Rd / r_env if r_env and r_env > 0 else 0
        rg_renv = Rg / r_env if r_env and r_env > 0 else 0

        row = {
            "id": gid, "cycle": cycle,
            "fb": fb, "fd": fd, "fg": fgas,
            "Mb": Mb, "Md": Md, "Mg": Mg, "Mt": Mt,
            "ab": ab, "Rd": Rd, "Rg": Rg,
            "rd_rg": rd_rg, "a_rd": a_rd, "a_rg": a_rg,
            "md_mg": md_mg, "mb_md": mb_md,
            "rt_renv": rt_renv, "rd_renv": rd_renv, "rg_renv": rg_renv,
            "r_t": r_t, "r_env": r_env,
        }

        if cycle == 3:
            data_3.append(row)
        else:
            data_2.append(row)

        print("  %-12s %3d | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f | %7.3f %7.3f | %7.4f %7.4f %7.4f | %10.3e" % (
            gid, cycle,
            fb, fd, fgas,
            rd_rg, a_rd, a_rg,
            md_mg, mb_md,
            rt_renv, rd_renv, rg_renv,
            Mt))

    # Statistics by cycle
    def show_stats(label, data):
        if not data:
            return
        n = len(data)
        print()
        print("  %s (N=%d):" % (label, n))
        print("  " + "-" * 90)

        for key, name in [
            ("fb", "Mb/Mt (bulge fraction)"),
            ("fd", "Md/Mt (disk fraction)"),
            ("fg", "Mg/Mt (gas fraction)"),
            ("md_mg", "Md/Mg"),
            ("mb_md", "Mb/Md"),
            ("rd_rg", "Rd/Rg"),
            ("a_rd", "a/Rd"),
            ("a_rg", "a/Rg"),
            ("rt_renv", "R_t/R_env"),
            ("rd_renv", "Rd/R_env"),
            ("rg_renv", "Rg/R_env"),
        ]:
            vals = [d[key] for d in data if d[key] < 900]
            if not vals:
                continue
            vals_s = sorted(vals)
            med = vals_s[len(vals_s) // 2]
            mean = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            std = math.sqrt(sum((x - mean)**2 for x in vals) / len(vals))
            print("    %-25s  median=%7.4f  mean=%7.4f  std=%7.4f  range=[%.4f, %.4f]" % (
                name, med, mean, std, mn, mx))

    print()
    print("=" * 130)
    print("  STATISTICS BY CYCLE")
    print("=" * 130)
    show_stats("3-CYCLE (disk/bulge-dominated)", data_3)
    show_stats("2-CYCLE (gas-dominated)", data_2)
    show_stats("ALL GALAXIES", data_3 + data_2)

    # Scatter: Md/Mg vs fg (gas fraction)
    print()
    print("=" * 130)
    print("  SCATTER: Md/Mg vs gas fraction (fg) and vs Rd/Rg")
    print("  Looking for a predictor of the closure ratio")
    print("=" * 130)
    print()
    print("  %-12s %3s | %6s %7s %7s %7s %7s" % (
        "Galaxy", "Cyc", "fg", "Md/Mg", "Rd/Rg", "Rd/Renv", "Rg/Renv"))
    print("  " + "-" * 65)

    all_data = sorted(data_3 + data_2, key=lambda d: d["fg"])
    for d in all_data:
        print("  %-12s %3d | %6.3f %7.3f %7.3f %7.4f %7.4f" % (
            d["id"], d["cycle"],
            d["fg"], d["md_mg"], d["rd_rg"],
            d["rd_renv"], d["rg_renv"]))

    # Check if Md/Mg correlates with Rd/Rg
    print()
    print("  " + "-" * 65)
    print("  CORRELATION CHECK: Md/Mg vs Rd/Rg")
    xs = [d["rd_rg"] for d in all_data if d["md_mg"] < 900]
    ys = [d["md_mg"] for d in all_data if d["md_mg"] < 900]
    n = len(xs)
    if n > 2:
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx)**2 for x in xs)
        syy = sum((y - my)**2 for y in ys)
        sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        if sxx > 0 and syy > 0:
            r = sxy / math.sqrt(sxx * syy)
            slope = sxy / sxx if sxx > 0 else 0
            intercept = my - slope * mx
            print("    Pearson r = %.4f" % r)
            print("    Linear fit: Md/Mg = %.3f * (Rd/Rg) + %.3f" % (slope, intercept))

    # Check Md/Mg vs fg
    print()
    print("  CORRELATION CHECK: Md/Mg vs fg (gas fraction)")
    xs2 = [d["fg"] for d in all_data if d["md_mg"] < 900]
    ys2 = [d["md_mg"] for d in all_data if d["md_mg"] < 900]
    n2 = len(xs2)
    if n2 > 2:
        mx2 = sum(xs2) / n2
        my2 = sum(ys2) / n2
        sxx2 = sum((x - mx2)**2 for x in xs2)
        syy2 = sum((y - my2)**2 for y in ys2)
        sxy2 = sum((xs2[i] - mx2) * (ys2[i] - my2) for i in range(n2))
        if sxx2 > 0 and syy2 > 0:
            r2 = sxy2 / math.sqrt(sxx2 * syy2)
            slope2 = sxy2 / sxx2 if sxx2 > 0 else 0
            intercept2 = my2 - slope2 * mx2
            print("    Pearson r = %.4f" % r2)
            print("    Linear fit: Md/Mg = %.3f * fg + %.3f" % (slope2, intercept2))

    # Check Mb/Mt vs something
    print()
    print("  CORRELATION CHECK: Mb/Mt vs Rt/Renv")
    xs3 = [d["rt_renv"] for d in all_data if d["fb"] > 0 and d["rt_renv"] > 0]
    ys3 = [d["fb"] for d in all_data if d["fb"] > 0 and d["rt_renv"] > 0]
    n3 = len(xs3)
    if n3 > 2:
        mx3 = sum(xs3) / n3
        my3 = sum(ys3) / n3
        sxx3 = sum((x - mx3)**2 for x in xs3)
        syy3 = sum((y - my3)**2 for y in ys3)
        sxy3 = sum((xs3[i] - mx3) * (ys3[i] - my3) for i in range(n3))
        if sxx3 > 0 and syy3 > 0:
            r3 = sxy3 / math.sqrt(sxx3 * syy3)
            slope3 = sxy3 / sxx3 if sxx3 > 0 else 0
            intercept3 = my3 - slope3 * mx3
            print("    Pearson r = %.4f" % r3)
            print("    Linear fit: Mb/Mt = %.3f * (Rt/Renv) + %.3f" % (slope3, intercept3))

    # Check fd vs fg (are they complementary?)
    print()
    print("  CORRELATION CHECK: fd vs fg")
    xs4 = [d["fg"] for d in all_data]
    ys4 = [d["fd"] for d in all_data]
    n4 = len(xs4)
    if n4 > 2:
        mx4 = sum(xs4) / n4
        my4 = sum(ys4) / n4
        sxx4 = sum((x - mx4)**2 for x in xs4)
        syy4 = sum((y - my4)**2 for y in ys4)
        sxy4 = sum((xs4[i] - mx4) * (ys4[i] - my4) for i in range(n4))
        if sxx4 > 0 and syy4 > 0:
            r4 = sxy4 / math.sqrt(sxx4 * syy4)
            slope4 = sxy4 / sxx4 if sxx4 > 0 else 0
            intercept4 = my4 - slope4 * mx4
            print("    Pearson r = %.4f" % r4)
            print("    Linear fit: fd = %.3f * fg + %.3f" % (slope4, intercept4))
            print("    (If fd + fg ~ 1.0 with fb ~ 0, slope should be ~ -1.0)")


if __name__ == "__main__":
    main()
