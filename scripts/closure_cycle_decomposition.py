"""
Closure cycle mass decomposition.

Given:
  1. M_total from topology (horizon condition)
  2. M_enc(R_t) from throat condition (yN = 18/65)
  3. Md/Mg = ratio from closure cycle prediction
  4. Scale lengths from topology (a, Rd, Rg from R_env)

Solve analytically for Mb, Md, Mg. No fitting needed.

Tests multiple closure ratios to find which works best.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import A0, G, M_SUN, KPC_TO_M
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.pure_inference import (
    horizon_enclosed_mass,
    throat_enclosed_mass,
)
from data.galaxies import PREDICTION_GALAXIES


def hernquist_frac(r, a):
    """Fraction of Hernquist mass enclosed at radius r."""
    if a <= 0 or r <= 0:
        return 0.0
    return r * r / ((r + a) * (r + a))


def disk_frac(r, Rd):
    """Fraction of exponential disk mass enclosed at radius r."""
    if Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return 1.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def solve_3cycle(M_total, M_enc_throat, r_t, a, Rd, Rg, ratio):
    """Solve for Mb, Md, Mg analytically given Md/Mg = ratio.

    Three equations:
      (1) Mt = Mb + Md + Mg
      (2) M_enc(Rt) = Mb*fb(Rt) + Md*fd(Rt) + Mg*fg(Rt)
      (3) Md = ratio * Mg

    Substituting (3) into (1): Mb = Mt - (1+ratio)*Mg
    Substituting into (2):
      M_enc(Rt) = (Mt - (1+ratio)*Mg)*fb + ratio*Mg*fd + Mg*fg
      M_enc(Rt) = Mt*fb + Mg*(-(1+ratio)*fb + ratio*fd + fg)

    Solving for Mg:
      Mg = (M_enc(Rt) - Mt*fb) / (-(1+ratio)*fb + ratio*fd + fg)
    """
    fb = hernquist_frac(r_t, a)
    fd = disk_frac(r_t, Rd)
    fg = disk_frac(r_t, Rg)

    denom = -(1.0 + ratio) * fb + ratio * fd + fg
    if abs(denom) < 1e-20:
        return None, None, None

    Mg = (M_enc_throat - M_total * fb) / denom
    Md = ratio * Mg
    Mb = M_total - Md - Mg

    # Check physical validity (all masses >= 0)
    if Mg < 0 or Md < 0 or Mb < 0:
        return Mb, Md, Mg  # Return anyway, flag negative

    return Mb, Md, Mg


def solve_2cycle(M_total, ratio):
    """Solve for 2-cycle: Mb = 0, Md/Mg = ratio."""
    Mg = M_total / (1.0 + ratio)
    Md = ratio * Mg
    return 0.0, Md, Mg


def pct(fit, pub):
    if abs(pub) < 1e3:
        if abs(fit) < 1e3:
            return 0.0
        return None
    return (fit - pub) / abs(pub) * 100.0


def fmt_m(v):
    if abs(v) < 1:
        return "0"
    if v < 0:
        return "NEG"
    return "%.3e" % v


def fmt_p(v):
    if v is None:
        return "N/A"
    return "%+.1f%%" % v


def run_test(ratio_3c, ratio_2c, label):
    """Run decomposition with given closure ratios."""
    mt_d = []
    mb_d = []
    md_d = []
    mg_d = []
    neg_count = 0

    results = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid.endswith("_inference"):
            continue
        mm = gal.get("mass_model", {})
        accel = gal.get("accel", 1.0)
        a0 = A0 * accel

        if not mm:
            continue

        sb = mm.get("bulge", {})
        sd = mm.get("disk", {})
        sg = mm.get("gas", {})
        Mb_s = sb.get("M", 0)
        Md_s = sd.get("M", 0)
        Mg_s = sg.get("M", 0)
        Mt_s = Mb_s + Md_s + Mg_s

        geom = solve_field_geometry(
            Mb_s, sb.get("a", 0.1),
            Md_s, sd.get("Rd", 1),
            Mg_s, sg.get("Rd", 1),
            a0,
        )
        r_env = geom.get("envelope_radius_kpc", 0) or 0
        r_t = geom.get("throat_radius_kpc", 0) or 0
        cycle = geom.get("cycle", 3)

        if r_env <= 0:
            continue

        # Topology masses
        M_total = horizon_enclosed_mass(r_env, a0)

        if cycle == 3:
            M_enc_throat = throat_enclosed_mass(r_t, a0)
            a_pred = 0.01 * r_env
            Rd_pred = 0.09 * r_env
            Rg_pred = 0.21 * r_env
            Mb_f, Md_f, Mg_f = solve_3cycle(
                M_total, M_enc_throat, r_t,
                a_pred, Rd_pred, Rg_pred, ratio_3c)
        else:
            a_pred = 0.01 * r_env
            Rd_pred = 0.30 * r_env
            Rg_pred = 1.00 * r_env
            Mb_f, Md_f, Mg_f = solve_2cycle(M_total, ratio_2c)

        if Mb_f is None:
            continue

        has_neg = Mb_f < -1e3 or Md_f < -1e3 or Mg_f < -1e3
        if has_neg:
            neg_count += 1

        Mt_f = Mb_f + Md_f + Mg_f

        dMb = pct(Mb_f, Mb_s)
        dMd = pct(Md_f, Md_s)
        dMg = pct(Mg_f, Mg_s)
        dMt = pct(Mt_f, Mt_s)

        if dMt is not None:
            mt_d.append(dMt)
        if dMb is not None:
            mb_d.append(dMb)
        if dMd is not None:
            md_d.append(dMd)
        if dMg is not None:
            mg_d.append(dMg)

        results.append((gid, cycle, r_env, Mb_f, Mb_s, dMb,
                         Md_f, Md_s, dMd, Mg_f, Mg_s, dMg,
                         Mt_f, Mt_s, dMt, has_neg))

    return results, mt_d, mb_d, md_d, mg_d, neg_count


def print_results(results, mt_d, mb_d, md_d, mg_d, neg_count, label):
    print()
    print("=" * 160)
    print("CLOSURE CYCLE DECOMPOSITION: %s" % label)
    print("=" * 160)
    print()

    header = (
        "{:<13s} {:>5s} {:>7s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s} {:>4s}"
    ).format(
        "Galaxy", "Cycle", "R_env",
        "Mb_fit", "Mb_sp", "dMb%",
        "Md_fit", "Md_sp", "dMd%",
        "Mg_fit", "Mg_sp", "dMg%",
        "Mt_fit", "Mt_sp", "dMt%", "neg",
    )
    print(header)
    print("-" * 160)

    for r in results:
        (gid, cycle, r_env, Mb_f, Mb_s, dMb,
         Md_f, Md_s, dMd, Mg_f, Mg_s, dMg,
         Mt_f, Mt_s, dMt, has_neg) = r

        neg_flag = " *" if has_neg else ""

        row = (
            "{:<13s} {:>5d} {:>7.1f}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s} {:>4s}"
        ).format(
            gid, cycle, r_env,
            fmt_m(Mb_f), fmt_m(Mb_s), fmt_p(dMb),
            fmt_m(Md_f), fmt_m(Md_s), fmt_p(dMd),
            fmt_m(Mg_f), fmt_m(Mg_s), fmt_p(dMg),
            fmt_m(Mt_f), fmt_m(Mt_s), fmt_p(dMt),
            neg_flag,
        )
        print(row)

    print()
    print("SUMMARY  (%d galaxies with negative masses)" % neg_count)
    print("-" * 80)

    def stats(lst, name):
        if not lst:
            return
        s = sorted(lst)
        n = len(s)
        med = s[n // 2]
        abs_s = sorted([abs(x) for x in lst])
        abs_med = abs_s[n // 2]
        abs_avg = sum(abs(x) for x in lst) / n
        print(
            "  %-15s  median=%+7.1f%%  |median|=%5.1f%%  |mean|=%5.1f%%  (n=%d)"
            % (name, med, abs_med, abs_avg, n)
        )

    stats(mt_d, "M_total")
    stats(mb_d, "M_bulge")
    stats(md_d, "M_disk")
    stats(mg_d, "M_gas")


def main():
    # Test multiple closure ratios
    ratios_to_test = [
        (3.0, 0.1, "Md/Mg=3.0 (3-cyc), Md/Mg=0.1 (2-cyc)"),
        (2.0, 0.1, "Md/Mg=2.0 (3-cyc), Md/Mg=0.1 (2-cyc)"),
        (4.0, 0.1, "Md/Mg=4.0 (3-cyc), Md/Mg=0.1 (2-cyc)"),
        (5.0, 0.1, "Md/Mg=5.0 (3-cyc), Md/Mg=0.1 (2-cyc)"),
        (1.0, 0.1, "Md/Mg=1.0 (3-cyc), Md/Mg=0.1 (2-cyc)"),
    ]

    # First show full detail for ratio=3
    results, mt_d, mb_d, md_d, mg_d, neg = run_test(3.0, 0.1, ratios_to_test[0][2])
    print_results(results, mt_d, mb_d, md_d, mg_d, neg, ratios_to_test[0][2])

    # Then show summary comparison for all ratios
    print()
    print()
    print("=" * 120)
    print("COMPARISON ACROSS CLOSURE RATIOS")
    print("=" * 120)
    print()

    header = "{:<45s}  {:>7s} {:>7s} {:>7s} {:>7s}  {:>4s}".format(
        "Ratio", "|Mt|%", "|Mb|%", "|Md|%", "|Mg|%", "neg")
    print(header)
    print("-" * 120)

    for r3, r2, label in ratios_to_test:
        res, mt_d, mb_d, md_d, mg_d, neg = run_test(r3, r2, label)

        def abs_med(lst):
            if not lst:
                return 999
            return sorted([abs(x) for x in lst])[len(lst) // 2]

        row = "{:<45s}  {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}%  {:>4d}".format(
            label,
            abs_med(mt_d), abs_med(mb_d), abs_med(md_d), abs_med(mg_d),
            neg)
        print(row)

    # Also show what the ACTUAL Md/Mg ratios are from SPARC
    print()
    print()
    print("=" * 120)
    print("REFERENCE: Actual SPARC Md/Mg ratios")
    print("=" * 120)
    print()

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid.endswith("_inference"):
            continue
        mm = gal.get("mass_model", {})
        if not mm:
            continue

        sd = mm.get("disk", {})
        sg = mm.get("gas", {})
        Md = sd.get("M", 0)
        Mg = sg.get("M", 0)

        a0 = A0 * gal.get("accel", 1.0)
        sb = mm.get("bulge", {})
        geom = solve_field_geometry(
            sb.get("M", 0), sb.get("a", 0.1),
            Md, sd.get("Rd", 1),
            Mg, sg.get("Rd", 1),
            a0)
        cycle = geom.get("cycle", 3)

        ratio = Md / Mg if Mg > 0 else 999
        print("  {:<13s}  cycle={:d}  Md/Mg={:>7.2f}".format(gid, cycle, ratio))


if __name__ == "__main__":
    main()
