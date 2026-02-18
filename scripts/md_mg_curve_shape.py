#!/usr/bin/env python3
"""
md_mg_curve_shape.py
=====================
Check whether Md/Mg follows a hyperbola (C/fg - 1) or a line.
Also check where Md/Mg = pi, and whether the hyperbola constant
C relates to any GFD number.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0
from physics.services.rotation.inference import solve_field_geometry


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
        if Mt <= 0 or Mg <= 0:
            continue
        fg = Mg / Mt
        fd = Md / Mt
        fb = Mb / Mt
        md_mg = Md / Mg
        geom = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        cycle = geom.get("cycle", 3)
        data.append({
            "id": gal["id"], "fg": fg, "fd": fd, "fb": fb,
            "md_mg": md_mg, "cycle": cycle, "Mt": Mt
        })

    data.sort(key=lambda d: d["fg"])

    print("=" * 90)
    print("  Md/Mg CURVE SHAPE ANALYSIS")
    print("=" * 90)
    print()

    # Model 1: Linear  Md/Mg = a*fg + b
    # Model 2: Hyperbola  Md/Mg = C/fg - 1  =>  C = (Md/Mg + 1) * fg = (fd + fg)
    # Model 3: Md/Mg = C/fg - D

    print("  %-12s %3s  %6s  %6s  %6s  %7s | %7s | %7s %7s" % (
        "Galaxy", "Cyc", "fb", "fd", "fg", "Md/Mg",
        "fd+fg", "C=(1+R)*fg", "err_hyp"))
    print("  " + "-" * 86)

    C_vals = []
    fd_fg_sums = []
    for d in data:
        fd_fg = d["fd"] + d["fg"]
        C = (d["md_mg"] + 1.0) * d["fg"]  # = fd + fg = 1 - fb
        pred_hyp = C / d["fg"] - 1.0  # should equal md_mg by definition
        # Use the MEAN C to predict
        C_vals.append(C)
        fd_fg_sums.append(fd_fg)

        print("  %-12s %3d  %6.3f  %6.3f  %6.3f  %7.3f | %7.4f | %7.4f %7.3f" % (
            d["id"], d["cycle"], d["fb"], d["fd"], d["fg"], d["md_mg"],
            fd_fg, C, 0.0))

    # C = fd + fg = 1 - fb
    mean_C = sum(C_vals) / len(C_vals)
    med_C = sorted(C_vals)[len(C_vals) // 2]

    print()
    print("  C = (Md/Mg + 1) * fg = fd + fg = 1 - fb")
    print("  Mean C:   %.4f" % mean_C)
    print("  Median C: %.4f" % med_C)
    print()

    # Now predict Md/Mg using a FIXED C
    print("  " + "-" * 86)
    print("  PREDICTION: Md/Mg = C/fg - 1  with C = %.4f (median)" % med_C)
    print("  " + "-" * 86)
    print()

    errs_hyp = []
    errs_lin = []
    # Linear fit coefficients from previous analysis
    lin_a = -9.436
    lin_b = 7.098

    print("  %-12s %6s  %7s  %7s %7s  %7s %7s" % (
        "Galaxy", "fg", "actual", "hyp", "err_h%", "linear", "err_l%"))
    print("  " + "-" * 70)

    for d in data:
        actual = d["md_mg"]
        pred_h = med_C / d["fg"] - 1.0
        pred_l = lin_a * d["fg"] + lin_b
        err_h = (pred_h - actual) / actual * 100 if actual > 0 else 0
        err_l = (pred_l - actual) / actual * 100 if actual > 0 else 0
        errs_hyp.append(abs(err_h))
        errs_lin.append(abs(err_l))

        print("  %-12s %6.3f  %7.3f  %7.3f %+6.1f%%  %7.3f %+6.1f%%" % (
            d["id"], d["fg"], actual, pred_h, err_h, pred_l, err_l))

    print()
    med_eh = sorted(errs_hyp)[len(errs_hyp) // 2]
    med_el = sorted(errs_lin)[len(errs_lin) // 2]
    mean_eh = sum(errs_hyp) / len(errs_hyp)
    mean_el = sum(errs_lin) / len(errs_lin)
    print("  Hyperbola (C/fg - 1): median|err| = %.1f%%  mean = %.1f%%" % (med_eh, mean_eh))
    print("  Linear (-9.4*fg+7.1): median|err| = %.1f%%  mean = %.1f%%" % (med_el, mean_el))

    # Where does Md/Mg = pi?
    print()
    print("  " + "=" * 70)
    print("  WHERE DOES Md/Mg = pi?")
    print("  " + "=" * 70)
    pi = math.pi

    fg_at_pi_hyp = med_C / (pi + 1.0)
    fg_at_pi_lin = (lin_b - pi) / (-lin_a)

    print("  Hyperbola: fg = C/(pi+1) = %.4f / %.4f = %.4f" % (med_C, pi + 1, fg_at_pi_hyp))
    print("  Linear:    fg = (7.098 - pi)/9.436 = %.4f" % fg_at_pi_lin)
    print()
    print("  At fg = %.4f, the Milky Way has fg = 0.198" % fg_at_pi_hyp)

    # Check interesting GFD numbers for C
    print()
    print("  " + "=" * 70)
    print("  WHAT IS C?")
    print("  " + "=" * 70)
    print("  C = 1 - fb = %.4f" % med_C)
    print()
    print("  Candidate matches:")
    candidates = [
        ("1 - 1/21 = 20/21", 20.0 / 21.0),
        ("1 - 1/13 = 12/13", 12.0 / 13.0),
        ("1 - 4/21 = 17/21", 17.0 / 21.0),
        ("13/14", 13.0 / 14.0),
        ("20/21", 20.0 / 21.0),
        ("6/7", 6.0 / 7.0),
        ("18/19", 18.0 / 19.0),
        ("pi/k = pi/4", math.pi / 4.0),
        ("1 - 1/k = 3/4", 3.0 / 4.0),
        ("1 - 1/(k+d) = 6/7", 6.0 / 7.0),
        ("1 - 1/f(k) = 20/21", 20.0 / 21.0),
        ("sqrt(k/pi)/k = 0.282", math.sqrt(4.0/math.pi)/4.0),
    ]
    for name, val in candidates:
        dev = abs(val - med_C) / med_C * 100
        marker = " <--" if dev < 3 else ""
        print("    %-25s = %.4f  (dev from C: %+.2f%%)%s" % (name, val, dev, marker))

    # What if C varies by galaxy? Is it correlated with anything?
    print()
    print("  " + "=" * 70)
    print("  DOES C = 1 - fb CORRELATE WITH ANYTHING?")
    print("  " + "=" * 70)
    print()
    print("  %-12s  %6s  %6s  %7s  %3s" % ("Galaxy", "fb", "C=1-fb", "Mt", "Cyc"))
    print("  " + "-" * 45)
    for d in data:
        print("  %-12s  %6.3f  %6.4f  %7.1e  %3d" % (
            d["id"], d["fb"], 1.0 - d["fb"], d["Mt"], d["cycle"]))


if __name__ == "__main__":
    main()
