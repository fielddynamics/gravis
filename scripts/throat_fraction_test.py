#!/usr/bin/env python3
"""
throat_fraction_test.py
========================
Test whether M_enc(R_t) / M_total clusters near 13/21 = 0.619
across all 22 SPARC galaxies, using the actual topology-derived
R_t, R_env, and cycle classification from solve_field_geometry
and derive_mass_from_topology.

The three structural states of f(k) = 1 + s*k + k^2:
  s=+1 -> 21 (full coupling)    -> M_total
  s= 0 -> 17 (capacity)         -> ?
  s=-1 -> 13 (spatial structure) -> M_enc(R_t)?

Also: at what radii (as fraction of R_env) do the polynomial
mass fractions (1/21, 4/21, 13/21, 16/21, 17/21) fall?

Uses solve_field_geometry for topology-derived radii,
derive_mass_from_topology for M_total and cycle classification.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.pure_inference import (
    derive_mass_from_topology, _hernquist_enc, _disk_enc, _model_enc
)


def total_enclosed(mm, r):
    """Total enclosed mass from SPARC 3-component model at radius r."""
    Mb = mm["bulge"]["M"]
    ab = mm["bulge"]["a"]
    Md = mm["disk"]["M"]
    Rd = mm["disk"]["Rd"]
    Mg = mm["gas"]["M"]
    Rg = mm["gas"]["Rd"]
    return _model_enc(r, Mb, ab, Md, Rd, Mg, Rg)


def find_radius_for_fraction(mm, frac, M_total, r_max):
    """Bisect to find radius where M_enc = frac * M_total."""
    target = frac * M_total
    r_lo, r_hi = 0.001, r_max * 3.0
    for _ in range(80):
        r_mid = (r_lo + r_hi) / 2.0
        if total_enclosed(mm, r_mid) < target:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return r_mid


def main():
    print("=" * 100)
    print("  THROAT MASS FRACTION TEST (using topology-derived R_t, R_env, cycle)")
    print("  Hypothesis: M_enc(R_t) / M_total = 13/21 = %.5f" % (13.0 / 21.0))
    print("=" * 100)
    print()

    header = "  %-12s %5s %7s %7s %10s %10s %10s %8s %8s %+8s"
    print(header % (
        "Galaxy", "Cycle", "R_env", "R_t", "M_total",
        "M_enc(Rt)", "M_topo_tot", "Rt_frac", "Ratio", "vs13/21"))
    print("  " + "-" * 96)

    TARGET_13 = 13.0 / 21.0
    TARGET_17 = 17.0 / 21.0

    ratios_3cyc = []
    ratios_2cyc = []
    ratios_all = []

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
        M_total_sparc = Mb + Md + Mg

        # Solve topology from the SPARC mass model
        fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        r_t = fg.get("throat_radius_kpc")
        r_env = fg.get("envelope_radius_kpc")
        cycle = fg.get("cycle", 3)

        if r_env is None or r_env <= 0:
            continue

        # Derive M_total from topology
        topo = derive_mass_from_topology(
            fg, a0_eff, mass_params=(Mb, ab, Md, Rd, Mg, Rg))
        M_topo_total = topo["M_total"]

        # Compute actual M_enc at R_t from the SPARC mass model
        if r_t is not None and r_t > 0:
            M_at_Rt = total_enclosed(mm, r_t)
        else:
            M_at_Rt = 0.0

        # Ratio: M_enc(R_t) / M_total (using SPARC asymptotic total)
        ratio = M_at_Rt / M_total_sparc if M_total_sparc > 0 else 0.0
        delta_13 = (ratio - TARGET_13) / TARGET_13 * 100

        rt_frac = r_t / r_env if r_t and r_env > 0 else 0.0

        ratios_all.append(ratio)
        if cycle == 3:
            ratios_3cyc.append(ratio)
        else:
            ratios_2cyc.append(ratio)

        print("  %-12s %5d %7.2f %7.2f %10.3e %10.3e %10.3e %8.4f %8.4f %+7.1f%%" % (
            gid, cycle, r_env, r_t or 0, M_total_sparc,
            M_at_Rt, M_topo_total, rt_frac, ratio, delta_13))

    # Statistics
    def stats(label, arr):
        if not arr:
            print("  %s: no data" % label)
            return
        n = len(arr)
        mean = sum(arr) / n
        med = sorted(arr)[n // 2]
        std = math.sqrt(sum((x - mean)**2 for x in arr) / n)
        print("  %s (N=%d):" % (label, n))
        print("    Mean:   %.4f  (vs 13/21=%.4f: %+.1f%%)" % (
            mean, TARGET_13, (mean - TARGET_13) / TARGET_13 * 100))
        print("    Median: %.4f  (vs 13/21=%.4f: %+.1f%%)" % (
            med, TARGET_13, (med - TARGET_13) / TARGET_13 * 100))
        print("    Std:    %.4f" % std)
        print("    Range:  %.4f to %.4f" % (min(arr), max(arr)))
        print("    vs 17/21=%.4f: mean dev %+.1f%%" % (
            TARGET_17, (mean - TARGET_17) / TARGET_17 * 100))

    print()
    print("=" * 100)
    print("  STATISTICS")
    print("=" * 100)
    stats("ALL galaxies", ratios_all)
    print()
    stats("3-CYCLE (disk/bulge-dominated)", ratios_3cyc)
    print()
    stats("2-CYCLE (gas-dominated)", ratios_2cyc)

    # ================================================================
    # BONUS: At what R/R_env does each polynomial fraction fall?
    # ================================================================
    print()
    print("=" * 100)
    print("  BONUS: Radius (as R/R_env) where M_enc = fraction * M_total")
    print("  Using topology-derived R_env for each galaxy")
    print("=" * 100)
    print()

    fractions = [
        (1.0 / 21.0, " 1/21 = 0.048 (k^0/f(k), Field Origin)"),
        (4.0 / 21.0, " 4/21 = 0.190 (k^1/f(k), propagation)"),
        (13.0 / 21.0, "13/21 = 0.619 (f(k,s=-1)/f(k), structure)"),
        (16.0 / 21.0, "16/21 = 0.762 (k^2/f(k), interaction)"),
        (17.0 / 21.0, "17/21 = 0.810 (f(k,s=0)/f(k), capacity)"),
    ]

    for frac_val, label in fractions:
        r_fracs_3 = []
        r_fracs_2 = []
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
            M_total = Mb + Md + Mg
            if M_total <= 0:
                continue

            fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
            r_env = fg.get("envelope_radius_kpc")
            cycle = fg.get("cycle", 3)
            if r_env is None or r_env <= 0:
                continue

            r_frac = find_radius_for_fraction(mm, frac_val, M_total, r_env)
            r_over_renv = r_frac / r_env

            if cycle == 3:
                r_fracs_3.append(r_over_renv)
            else:
                r_fracs_2.append(r_over_renv)

        all_fracs = r_fracs_3 + r_fracs_2
        med_all = sorted(all_fracs)[len(all_fracs) // 2] if all_fracs else 0
        med_3 = sorted(r_fracs_3)[len(r_fracs_3) // 2] if r_fracs_3 else 0
        med_2 = sorted(r_fracs_2)[len(r_fracs_2) // 2] if r_fracs_2 else 0

        print("  %s" % label)
        print("    All:    median R/R_env = %.4f  (N=%d)" % (med_all, len(all_fracs)))
        if r_fracs_3:
            print("    3-cyc:  median R/R_env = %.4f  (N=%d)" % (med_3, len(r_fracs_3)))
        if r_fracs_2:
            print("    2-cyc:  median R/R_env = %.4f  (N=%d)" % (med_2, len(r_fracs_2)))
        print()

    # ================================================================
    # Per-galaxy detail for the 13/21 radius
    # ================================================================
    print("=" * 100)
    print("  DETAIL: Radius where M_enc = 13/21 * M_total (vs R_t)")
    print("=" * 100)
    print()
    print("  %-12s %5s %7s %7s %7s %8s" % (
        "Galaxy", "Cycle", "R_env", "R_t", "R_13", "R_13/R_t"))
    print("  " + "-" * 55)

    for gal in PREDICTION_GALAXIES:
        mm = gal["mass_model"]
        gid = gal["id"]
        accel = gal.get("accel", 1.0)
        a0_eff = A0 * accel
        Mb = mm["bulge"]["M"]
        ab = mm["bulge"]["a"]
        Md = mm["disk"]["M"]
        Rd = mm["disk"]["Rd"]
        Mg = mm["gas"]["M"]
        Rg = mm["gas"]["Rd"]
        M_total = Mb + Md + Mg
        if M_total <= 0:
            continue

        fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        r_env = fg.get("envelope_radius_kpc")
        r_t = fg.get("throat_radius_kpc")
        cycle = fg.get("cycle", 3)
        if r_env is None or r_env <= 0:
            continue

        r_13 = find_radius_for_fraction(mm, 13.0 / 21.0, M_total, r_env)
        r13_over_rt = r_13 / r_t if r_t and r_t > 0 else 0.0

        print("  %-12s %5d %7.2f %7.2f %7.2f %8.4f" % (
            gid, cycle, r_env, r_t or 0, r_13, r13_over_rt))


if __name__ == "__main__":
    main()
