"""
Full validation: Bayesian observation-driven topology vs SPARC published data.

Same as prove_horizon_vs_sparc.py but uses the Bayesian optimizer
(differential_evolution + L-BFGS-B polish) instead of the grid search
mass model. This is a much more accurate geometry ladder because the
optimizer fits the GFD velocity curve directly, not the enclosed mass.

For each of the 22 SPARC galaxies:
  1. Extract observations (r, v, err) only
  2. Bayesian Pass 1: unconstrained differential_evolution fits the
     6-parameter mass model (Mb, ab, Md, Rd, Mg, Rg) to the GFD
     velocity curve. No SPARC mass data used.
  3. Solve field geometry from the fitted mass model
  4. Derive M_total from horizon + enclosure correction
  5. Compare derived R_t, R_env, M_total against SPARC published values

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.optimize import differential_evolution, minimize

from physics.constants import A0, G, M_SUN, KPC_TO_M
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.pure_inference import derive_mass_from_topology
from physics.services.sandbox.bayesian_fit import (
    _chi2_cost,
    _estimate_bounds,
    gfd_velocity,
)
from data.galaxies import PREDICTION_GALAXIES


def bayesian_pass1(obs_r, obs_v, obs_err, a0_eff, seed=42):
    """Run unconstrained Bayesian fit (Pass 1 only).

    Returns the best-fit (Mb, ab, Md, Rd, Mg, Rg) and RMS.
    """
    n = len(obs_r)
    obs_w = [1.0 / (e * e) for e in obs_err]
    bounds = _estimate_bounds(obs_r, obs_v, a0_eff)

    result = differential_evolution(
        _chi2_cost,
        bounds=bounds,
        args=(obs_r, obs_v, obs_w, a0_eff),
        seed=seed,
        maxiter=300,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    polished = minimize(
        _chi2_cost,
        result.x,
        args=(obs_r, obs_v, obs_w, a0_eff),
        method='L-BFGS-B',
        bounds=bounds,
    )

    best = polished.x if polished.success else result.x
    Mb, ab, Md, Rd, Mg, Rg = best

    # Compute RMS
    ss = 0.0
    for j in range(n):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        ss += delta * delta
    rms = math.sqrt(ss / n) if n > 0 else 0.0

    return (Mb, ab, Md, Rd, Mg, Rg), rms


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

    print("=" * 145)
    print("BAYESIAN OBSERVATION-DRIVEN TOPOLOGY vs SPARC PUBLISHED DATA")
    print("Input: observations (r, v, err) only. Bayesian optimizer (DE + L-BFGS-B). Zero SPARC mass data.")
    print("=" * 145)
    print()

    print("{:<14s} {:>5s}  {:>7s} {:>7s} {:>7s} {:>7s}  "
          "{:>11s} {:>11s} {:>7s}  {:>6s}  "
          "{:>7s} {:>7s}  {:>5s}".format(
              "Galaxy", "Cycle",
              "Rt_der", "Rt_pub", "Re_der", "Re_pub",
              "Mt_derived", "Mt_pubSPARC", "Mt_%err",
              "enc_r",
              "Rt/Re_d", "Rt/Re_p", "RMS"))
    print("-" * 145)

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
        # PURE BAYESIAN PIPELINE: observations only
        # =============================================================
        obs_r = []
        obs_v = []
        obs_err = []
        for o in observations:
            r = float(o.get("r", 0))
            v = float(o.get("v", 0))
            if r > 0 and v > 0:
                obs_r.append(r)
                obs_v.append(v)
                obs_err.append(max(float(o.get("err", 5.0)), 1.0))

        if len(obs_r) < 3:
            print("{:<14s}   --  (insufficient valid observations)".format(gid))
            continue

        # Bayesian Pass 1: unconstrained fit
        params, rms = bayesian_pass1(obs_r, obs_v, obs_err, a0_eff)
        Mb, ab, Md, Rd, Mg, Rg = params

        # Solve field geometry from Bayesian mass model
        geom_der = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        Rt_der = geom_der.get("throat_radius_kpc") or 0
        Re_der = geom_der.get("envelope_radius_kpc") or 0
        cycle = geom_der.get("cycle", 0)

        # Derive M_total from topology + enclosure correction
        topo = derive_mass_from_topology(geom_der, a0_eff, mass_params=params)
        Mt_der = topo["M_total"]
        enc_r = topo["enclosure_ratio"]

        tf_der = Rt_der / Re_der if Re_der > 0 else 0

        mt_err = pct(Mt_der, Mt_pub) if Mt_pub > 0 else 0
        rt_err = pct(Rt_der, Rt_pub) if Rt_pub > 0 else 0
        re_err = pct(Re_der, Re_pub) if Re_pub > 0 else 0

        print("{:<14s} {:>5d}  {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f}  "
              "{:>11s} {:>11s} {:>+7.1f}  {:>6.2f}  "
              "{:>7.4f} {:>7.4f}  {:>5.1f}".format(
                  gid, cycle,
                  Rt_der, Rt_pub, Re_der, Re_pub,
                  fmt_mass(Mt_der), fmt_mass(Mt_pub), mt_err,
                  enc_r,
                  tf_der, tf_pub, rms))

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
    print("=" * 145)
    print("SUMMARY")
    print("-" * 60)
    print("  Galaxies tested: {}  (3-cycle: {}, 2-cycle: {})".format(
        summary["n"], summary["n_3cycle"], summary["n_2cycle"]))

    if summary["mt_errs"]:
        errs = summary["mt_errs"]
        errs_s = sorted(errs)
        median = errs_s[len(errs) // 2]
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
    print("  Note: 'published' R_t, R_env derived from SPARC mass model")
    print("  via solve_field_geometry. 'derived' values come purely from")
    print("  (r, v, err) observations with zero SPARC mass data.")
    print("=" * 145)


if __name__ == "__main__":
    main()
