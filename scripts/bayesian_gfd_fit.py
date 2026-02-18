"""
Bayesian GFD base fit to observations.

Fits the 3-component mass model (Hernquist bulge + 2 exponential disks)
directly to observed rotation curves using the GFD covariant field equation.
Uses scipy.optimize.differential_evolution for global optimization
(maximum likelihood = MAP with flat priors).

No published masses used. Observations are the only input.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.optimize import differential_evolution, minimize
from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry
from data.galaxies import get_all_galaxies


# =====================================================================
# MASS MODEL -> GFD BASE VELOCITY
# =====================================================================

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


def model_enc(r, Mb, ab, Md, Rd, Mg, Rg):
    return hernquist_enc(r, Mb, ab) + disk_enc(r, Md, Rd) + disk_enc(r, Mg, Rg)


def gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """GFD base velocity at radius r_kpc from the covariant field equation."""
    enc = model_enc(r_kpc, Mb, ab, Md, Rd, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0


# =====================================================================
# CHI-SQUARED COST FUNCTION
# =====================================================================

def chi2_cost(params, obs_r, obs_v, obs_w, a0_eff):
    """Weighted chi-squared: sum of w_j * (v_obs - v_pred)^2."""
    Mb, ab, Md, Rd, Mg, Rg = params
    cost = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        cost += obs_w[j] * delta * delta
    return cost


# =====================================================================
# PARAMETER BOUNDS ESTIMATION
# =====================================================================

def estimate_bounds(obs_r, obs_v, a0_eff):
    """Estimate reasonable parameter bounds from observation data.

    Uses the field equation inversion to get total enclosed mass,
    then sets bounds around plausible ranges for each component.
    """
    r_max = max(obs_r)
    v_max = max(obs_v)

    # Invert outermost observation to get total mass estimate
    r_m = r_max * KPC_TO_M
    v_m = v_max * 1000.0
    x = (v_m * v_m) / (a0_eff * r_m)
    y = (x * x) / (1.0 + x)
    M_total_est = y * r_m * r_m * a0_eff / (G * M_SUN)

    # Mass bounds: each component can be 0 to 2x total
    M_hi = M_total_est * 2.0
    M_lo = 0.0

    # Scale bounds: bulge is compact, disk is medium, gas is extended
    ab_bounds = (0.01, r_max * 0.3)
    Rd_bounds = (0.1, r_max * 0.8)
    Rg_bounds = (0.2, r_max * 1.5)

    return [
        (M_lo, M_hi),      # Mb
        ab_bounds,          # ab
        (M_lo, M_hi),      # Md
        Rd_bounds,          # Rd
        (M_lo, M_hi),      # Mg
        Rg_bounds,          # Rg
    ]


# =====================================================================
# BAYESIAN FIT (differential_evolution + polish)
# =====================================================================

def fit_gfd_bayesian(obs_r, obs_v, obs_err, a0_eff, seed=42):
    """Find the maximum likelihood (MAP) mass model parameters.

    Stage 1: differential_evolution for global search (robust, no
             local minima traps, handles bounds naturally).
    Stage 2: L-BFGS-B polish for precision near the optimum.

    Returns dict with fitted parameters, RMS, per-point residuals.
    """
    n = len(obs_r)
    obs_w = [1.0 / (e * e) for e in obs_err]

    bounds = estimate_bounds(obs_r, obs_v, a0_eff)

    # Stage 1: Global search
    result = differential_evolution(
        chi2_cost,
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

    # Stage 2: Local polish with L-BFGS-B
    polished = minimize(
        chi2_cost,
        result.x,
        args=(obs_r, obs_v, obs_w, a0_eff),
        method='L-BFGS-B',
        bounds=bounds,
    )

    best = polished.x if polished.success else result.x
    Mb, ab, Md, Rd, Mg, Rg = best

    # Compute per-point residuals
    residuals = []
    ss = 0.0
    for j in range(n):
        v_pred = gfd_velocity(obs_r[j], Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        ss += delta * delta
        residuals.append({
            "r": obs_r[j],
            "v_obs": obs_v[j],
            "v_pred": round(v_pred, 2),
            "delta": round(delta, 2),
            "err": obs_err[j],
        })
    rms = math.sqrt(ss / n) if n > 0 else 0.0
    chi2_dof = polished.fun / max(n - 6, 1)

    return {
        "params": {
            "Mb": round(Mb, 2),
            "ab": round(ab, 4),
            "Md": round(Md, 2),
            "Rd": round(Rd, 4),
            "Mg": round(Mg, 2),
            "Rg": round(Rg, 4),
        },
        "M_total": round(Mb + Md + Mg, 2),
        "rms": round(rms, 2),
        "chi2_dof": round(chi2_dof, 4),
        "n_obs": n,
        "residuals": residuals,
        "optimizer_converged": bool(polished.success),
    }


# =====================================================================
# MAIN: Run on all galaxies
# =====================================================================

def main():
    all_grouped = get_all_galaxies()
    galaxies = []
    for group in all_grouped.values():
        if isinstance(group, list):
            galaxies.extend(group)

    # Only galaxies with observations
    galaxies = [g for g in galaxies if len(g.get('observations', [])) >= 3]

    # Header
    print("=" * 140)
    print("BAYESIAN GFD BASE FIT: Covariant completion mapped to observations")
    print("Optimizer: differential_evolution (global) + L-BFGS-B (polish)")
    print("Free parameters: Mb, ab, Md, Rd, Mg, Rg (6 parameters)")
    print("=" * 140)
    print()
    print("%-22s | %6s | %8s | %12s %8s %8s %8s %8s %8s | %8s %8s" % (
        "Galaxy", "N_obs", "RMS",
        "M_total", "Mb", "Md", "Mg", "Rd", "Rg",
        "Renv", "Rt"))
    print("-" * 140)

    total_rms = 0.0
    n_galaxies = 0
    results = []

    for g in galaxies:
        name = g.get('name', g.get('id', '?')).split('(')[0].strip()
        obs = g.get('observations', [])
        accel = g.get('accel', 1.0)
        a0_eff = A0 * accel

        # Extract observation arrays
        obs_r = []
        obs_v = []
        obs_err = []
        for o in obs:
            r = float(o.get('r', 0))
            v = float(o.get('v', 0))
            if r > 0 and v > 0:
                obs_r.append(r)
                obs_v.append(v)
                obs_err.append(max(float(o.get('err', 5.0)), 1.0))

        if len(obs_r) < 3:
            continue

        t0 = time.time()
        fit = fit_gfd_bayesian(obs_r, obs_v, obs_err, a0_eff)
        elapsed = time.time() - t0

        p = fit["params"]

        # Derive field geometry from the fitted mass model
        geom = solve_field_geometry(
            p["Mb"], p["ab"], p["Md"], p["Rd"], p["Mg"], p["Rg"], a0_eff)
        r_env = geom.get("envelope_radius_kpc") or 0
        r_t = geom.get("throat_radius_kpc") or 0

        short_name = name[:22]
        print("%-22s | %6d | %8.2f | %12.2e %8.2e %8.2e %8.2e %8.2f %8.2f | %8.2f %8.2f  (%.1fs)" % (
            short_name, fit["n_obs"], fit["rms"],
            fit["M_total"], p["Mb"], p["Md"], p["Mg"], p["Rd"], p["Rg"],
            r_env, r_t, elapsed))

        total_rms += fit["rms"]
        n_galaxies += 1
        results.append((name, fit, geom))

    print("-" * 140)
    if n_galaxies > 0:
        print("Mean RMS across %d galaxies: %.2f km/s" % (
            n_galaxies, total_rms / n_galaxies))
    print()

    # Per-point residual detail for a few galaxies
    detail_galaxies = ["M33 Triangulum", "Milky Way", "NGC 2841", "NGC 3521"]
    for name, fit, geom in results:
        clean = name.split('(')[0].strip()
        if clean not in detail_galaxies:
            continue
        print("=" * 80)
        print("DETAIL: %s (RMS = %.2f km/s, chi2/dof = %.4f)" % (
            clean, fit["rms"], fit["chi2_dof"]))
        print("%-8s  %8s  %8s  %8s  %8s" % (
            "r(kpc)", "v_obs", "v_pred", "delta", "err"))
        print("-" * 48)
        for res in fit["residuals"]:
            print("%8.2f  %8.1f  %8.1f  %+8.2f  %8.1f" % (
                res["r"], res["v_obs"], res["v_pred"],
                res["delta"], res["err"]))
        print()


if __name__ == "__main__":
    main()
