"""
Full validation: pure observation-driven topology vs SPARC published data.

For each of the 22 SPARC galaxies:
  1. Invert observations (r, v, err) -> M_enc(r) at each point
  2. Fit parametric mass model (geometry ladder)
  3. Solve field geometry: R_t, R_env from yN conditions
  4. Derive M_total from horizon + enclosure correction
  5. Compare derived R_t, R_env, M_total against SPARC published values

Zero SPARC mass data used in the derivation. Only (r, v, err).

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import A0, G, M_SUN, KPC_TO_M
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.pure_inference import (
    invert_observations,
    fit_mass_model,
    derive_mass_from_topology,
)
from data.galaxies import PREDICTION_GALAXIES


def pct(derived, published):
    if published == 0:
        return 0.0 if derived == 0 else float('inf')
    return (derived - published) / published * 100.0


def fmt_mass(m):
    if m == 0:
        return "       0    "
    exp = int(math.floor(math.log10(abs(m))))
    coeff = m / (10 ** exp)
    return "{:.2f}e{:+d}".format(coeff, exp)


def main():
    a0_eff = A0

    print("=" * 140)
    print("PURE OBSERVATION-DRIVEN TOPOLOGY vs SPARC PUBLISHED DATA")
    print("Input: observations (r, v, err) only. Zero SPARC mass data.")
    print("=" * 140)
    print()

    # Header
    print("{:<14s} {:>5s}  {:>7s} {:>7s} {:>7s} {:>7s}  "
          "{:>11s} {:>11s} {:>7s}  {:>6s}  "
          "{:>7s} {:>7s} {:>7s}".format(
              "Galaxy", "Cycle",
              "Rt_der", "Rt_pub", "Re_der", "Re_pub",
              "Mt_derived", "Mt_pubSPARC", "Mt_%err",
              "enc_r",
              "Rt/Re_d", "Rt/Re_p", "Rt%err"))
    print("-" * 140)

    summary = {"n": 0, "n_3cycle": 0, "n_2cycle": 0,
               "mt_errs": [], "rt_errs": [], "re_errs": []}

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid.endswith("_inference"):
            continue

        observations = gal.get("observations", [])
        if len(observations) < 3:
            continue

        mm = gal.get("mass_model", {})
        b = mm.get("bulge", {})
        d = mm.get("disk", {})
        g = mm.get("gas", {})

        # Published SPARC values (for comparison only)
        Mb_pub = b.get("M", 0)
        ab_pub = b.get("a", 0.1)
        Md_pub = d.get("M", 0)
        Rd_pub = d.get("Rd", 1.0)
        Mg_pub = g.get("M", 0)
        Rg_pub = g.get("Rd", 1.0)
        Mt_pub = Mb_pub + Md_pub + Mg_pub

        # Published field geometry (for comparison)
        geom_pub = solve_field_geometry(
            Mb_pub, ab_pub, Md_pub, Rd_pub, Mg_pub, Rg_pub, a0_eff)
        Rt_pub = geom_pub.get("throat_radius_kpc") or 0
        Re_pub = geom_pub.get("envelope_radius_kpc") or 0
        tf_pub = geom_pub.get("throat_fraction") or 0

        # =============================================================
        # PURE PIPELINE: observations only, no SPARC mass data
        # =============================================================

        # Step 1: Invert observations -> M_enc(r)
        inverted = invert_observations(observations, a0_eff)
        if len(inverted) < 3:
            print("{:<14s}   --  (insufficient valid observations)".format(gid))
            continue

        # Step 2: Fit parametric mass model (geometry ladder)
        fit = fit_mass_model(inverted, a0_eff)
        if fit is None:
            print("{:<14s}   --  (mass model fit failed)".format(gid))
            continue

        params = fit["params"]
        Mb, ab, Md, Rd, Mg, Rg = params

        # Step 3: Solve field geometry from derived mass model
        geom_der = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        Rt_der = geom_der.get("throat_radius_kpc") or 0
        Re_der = geom_der.get("envelope_radius_kpc") or 0
        cycle = geom_der.get("cycle", 0)

        # Step 4: Derive M_total from topology + enclosure correction
        topo = derive_mass_from_topology(geom_der, a0_eff, mass_params=params)
        Mt_der = topo["M_total"]
        enc_r = topo["enclosure_ratio"]

        # Derived throat fraction
        tf_der = Rt_der / Re_der if Re_der > 0 else 0

        # Compute errors vs SPARC published
        mt_err = pct(Mt_der, Mt_pub) if Mt_pub > 0 else 0
        rt_err = pct(Rt_der, Rt_pub) if Rt_pub > 0 else 0
        re_err = pct(Re_der, Re_pub) if Re_pub > 0 else 0

        print("{:<14s} {:>5d}  {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f}  "
              "{:>11s} {:>11s} {:>+7.1f}  {:>6.2f}  "
              "{:>7.4f} {:>7.4f} {:>+6.1f}%".format(
                  gid, cycle,
                  Rt_der, Rt_pub, Re_der, Re_pub,
                  fmt_mass(Mt_der), fmt_mass(Mt_pub), mt_err,
                  enc_r,
                  tf_der, tf_pub, rt_err))

        summary["n"] += 1
        if cycle == 3:
            summary["n_3cycle"] += 1
        else:
            summary["n_2cycle"] += 1
        if Mt_pub > 0:
            summary["mt_errs"].append(abs(mt_err))
        if Rt_pub > 0:
            summary["rt_errs"].append(abs(rt_err))
        if Re_pub > 0:
            summary["re_errs"].append(abs(re_err))

    # Summary statistics
    print()
    print("=" * 140)
    print("SUMMARY")
    print("-" * 60)
    print("  Galaxies tested: {}  (3-cycle: {}, 2-cycle: {})".format(
        summary["n"], summary["n_3cycle"], summary["n_2cycle"]))

    if summary["mt_errs"]:
        errs = summary["mt_errs"]
        median = sorted(errs)[len(errs) // 2]
        print("  M_total error vs SPARC:")
        print("    Median: {:.1f}%".format(median))
        print("    Mean:   {:.1f}%".format(sum(errs) / len(errs)))
        print("    Max:    {:.1f}%".format(max(errs)))
        print("    < 5%%:   {}/{}".format(
            sum(1 for e in errs if e < 5), len(errs)))
        print("    < 10%%:  {}/{}".format(
            sum(1 for e in errs if e < 10), len(errs)))
        print("    < 20%%:  {}/{}".format(
            sum(1 for e in errs if e < 20), len(errs)))

    if summary["rt_errs"]:
        errs = summary["rt_errs"]
        median = sorted(errs)[len(errs) // 2]
        print("  R_t error vs SPARC:")
        print("    Median: {:.1f}%".format(median))
        print("    Mean:   {:.1f}%".format(sum(errs) / len(errs)))

    if summary["re_errs"]:
        errs = summary["re_errs"]
        median = sorted(errs)[len(errs) // 2]
        print("  R_env error vs SPARC:")
        print("    Median: {:.1f}%".format(median))
        print("    Mean:   {:.1f}%".format(sum(errs) / len(errs)))

    print()
    print("  Note: 'published' R_t, R_env are derived from SPARC mass model")
    print("  via solve_field_geometry. 'derived' values come purely from")
    print("  (r, v, err) observations with zero SPARC mass data.")
    print("=" * 140)


if __name__ == "__main__":
    main()
