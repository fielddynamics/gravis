#!/usr/bin/env python3
"""
bulge_bayesian_budget.py
=========================
Constrained Bayesian: fix M_total and M_gas from topology.
The only freedom is the bulge/disk split within the stellar budget.

  M_stellar = M_total - M_gas           (fixed from topology)
  M_bulge   = f_b * M_stellar           (single free fraction, 0 to 1)
  M_disk    = (1 - f_b) * M_stellar     (determined by budget)

The velocity curve + scale length perturbations pick the best f_b.

Free parameters: f_b, delta_Rd, delta_ab  (3 parameters)
Fixed:           M_total, M_gas, Rg       (from topology)
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.optimize import differential_evolution
from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0, G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry


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

def gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    enc = hernquist_enc(r_kpc, Mb, ab) + disk_enc(r_kpc, Md, Rd) + disk_enc(r_kpc, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0

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
    """Method B: topology 3x3 with continuous fd-fg."""
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


def budget_fit_1d(obs, M_stellar, Mg_fixed, Md_init,
                   ab_init, Rd_init, Rg_fixed, a0_eff):
    """
    Constrained Bayesian: M_total and M_gas locked.
    M_disk allowed to wiggle +-5% (median error is ~10%, so 1-sigma ~ 5%).
    M_bulge = M_stellar - M_disk (absorbs the residual).

    Free parameters:
      [0] f_Md     : fractional perturbation to M_disk  (+-5%)
      [1] f_Rd     : fractional perturbation to Rd      (+-10%)
      [2] f_ab     : fractional perturbation to ab      (+-50%)
    """
    obs_r = [p["r"] for p in obs]
    obs_v = [p["v"] for p in obs]
    obs_e = [p["err"] for p in obs]
    obs_w = [1.0 / (e * e) if e > 0 else 1.0 for e in obs_e]

    sigma_Md = 0.05
    sigma_Rd = 0.10
    sigma_ab = 0.50

    def cost(params):
        f_Md, f_Rd, f_ab = params

        Md = Md_init * (1.0 + f_Md)
        Mb = M_stellar - Md
        if Mb < 0:
            return 1e20

        cur_Rd = Rd_init * (1.0 + f_Rd)
        cur_ab = ab_init * (1.0 + f_ab)

        if cur_Rd <= 0 or cur_ab <= 0:
            return 1e20

        chi2 = 0.0
        for j in range(len(obs_r)):
            v_pred = gfd_velocity(
                obs_r[j], Mb, cur_ab, Md, cur_Rd, Mg_fixed, Rg_fixed, a0_eff)
            delta = obs_v[j] - v_pred
            chi2 += obs_w[j] * delta * delta

        # Gaussian prior penalizing perturbations from topology values
        prior = (f_Md / sigma_Md) ** 2
        prior += (f_Rd / sigma_Rd) ** 2 + (f_ab / sigma_ab) ** 2
        return chi2 + prior

    bounds = [
        (-sigma_Md, sigma_Md),  # f_Md: disk mass +-5%
        (-sigma_Rd, sigma_Rd),  # f_Rd: disk scale length +-10%
        (-sigma_ab, sigma_ab),  # f_ab: bulge scale length +-50%
    ]

    result = differential_evolution(
        cost, bounds,
        seed=42, maxiter=300, tol=1e-9,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9,
        polish=True
    )

    f_Md, f_Rd, f_ab = result.x
    Md_out = Md_init * (1.0 + f_Md)
    Mb_out = M_stellar - Md_out
    Rd_out = Rd_init * (1.0 + f_Rd)
    ab_out = ab_init * (1.0 + f_ab)

    rms = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], Mb_out, ab_out, Md_out, Rd_out,
                               Mg_fixed, Rg_fixed, a0_eff)
        rms += (obs_v[j] - v_pred) ** 2
    rms = math.sqrt(rms / len(obs_r))

    Mt = Mb_out + Md_out + Mg_fixed
    fb = Mb_out / Mt if Mt > 0 else 0

    return {
        "Mb": Mb_out, "ab": ab_out,
        "Md": Md_out, "Rd": Rd_out,
        "Mg": Mg_fixed, "Rg": Rg_fixed,
        "fb": fb,
        "rms": rms,
        "deltas": {"Md": f_Md, "Rd": f_Rd, "ab": f_ab},
    }


def pass2_refine(obs, Mt_fixed,
                  Mg_center, Md_center, Mb_center,
                  ab_center, Rd_center, Rg_center,
                  a0_eff):
    """
    Pass 2: Lock gas +-3.5%, lock disk +-7.5%, let bulge absorb residual.
    All 3 scale lengths can adjust within tight bands.
    M_total stays fixed (budget: Mb = Mt - Md - Mg).

    Free parameters:
      [0] f_Mg  : fractional perturbation to M_gas  (+-3.5%)
      [1] f_Md  : fractional perturbation to M_disk (+-7.5%)
      [2] f_Rd  : fractional perturbation to Rd     (+-10%)
      [3] f_Rg  : fractional perturbation to Rg     (+-10%)
      [4] f_ab  : fractional perturbation to ab     (+-40%)
    """
    obs_r = [p["r"] for p in obs]
    obs_v = [p["v"] for p in obs]
    obs_e = [p["err"] for p in obs]
    obs_w = [1.0 / (e * e) if e > 0 else 1.0 for e in obs_e]

    s_Mg = 0.035
    s_Md = 0.075
    s_Rd = 0.10
    s_Rg = 0.10
    s_ab = 0.40

    def cost(params):
        f_Mg, f_Md, f_Rd, f_Rg, f_ab = params

        Mg = Mg_center * (1.0 + f_Mg)
        Md = Md_center * (1.0 + f_Md)
        Mb = Mt_fixed - Mg - Md

        if Mb < 0 or Mg < 0 or Md < 0:
            return 1e20

        cur_Rd = Rd_center * (1.0 + f_Rd)
        cur_Rg = Rg_center * (1.0 + f_Rg)
        cur_ab = ab_center * (1.0 + f_ab)

        if cur_Rd <= 0 or cur_Rg <= 0 or cur_ab <= 0:
            return 1e20

        chi2 = 0.0
        for j in range(len(obs_r)):
            v_pred = gfd_velocity(
                obs_r[j], Mb, cur_ab, Md, cur_Rd, Mg, cur_Rg, a0_eff)
            delta = obs_v[j] - v_pred
            chi2 += obs_w[j] * delta * delta

        # Prior: penalize perturbations from Pass 1 center
        prior = (f_Mg / s_Mg) ** 2 + (f_Md / s_Md) ** 2
        prior += (f_Rd / s_Rd) ** 2 + (f_Rg / s_Rg) ** 2
        prior += (f_ab / s_ab) ** 2
        return chi2 + prior

    bounds = [
        (-s_Mg, s_Mg),
        (-s_Md, s_Md),
        (-s_Rd, s_Rd),
        (-s_Rg, s_Rg),
        (-s_ab, s_ab),
    ]

    result = differential_evolution(
        cost, bounds,
        seed=42, maxiter=300, tol=1e-9,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9,
        polish=True
    )

    f_Mg, f_Md, f_Rd, f_Rg, f_ab = result.x
    Mg_out = Mg_center * (1.0 + f_Mg)
    Md_out = Md_center * (1.0 + f_Md)
    Mb_out = Mt_fixed - Mg_out - Md_out
    Rd_out = Rd_center * (1.0 + f_Rd)
    Rg_out = Rg_center * (1.0 + f_Rg)
    ab_out = ab_center * (1.0 + f_ab)

    rms = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], max(Mb_out, 0), ab_out,
                               Md_out, Rd_out, Mg_out, Rg_out, a0_eff)
        rms += (obs_v[j] - v_pred) ** 2
    rms = math.sqrt(rms / len(obs_r))

    Mt = Mb_out + Md_out + Mg_out
    fb = Mb_out / Mt if Mt > 0 else 0

    return {
        "Mb": Mb_out, "ab": ab_out,
        "Md": Md_out, "Rd": Rd_out,
        "Mg": Mg_out, "Rg": Rg_out,
        "fb": fb, "rms": rms,
        "deltas": {"Mg": f_Mg, "Md": f_Md, "Rd": f_Rd, "Rg": f_Rg, "ab": f_ab},
    }


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
    print("=" * 135)
    print("  BUDGET BAYESIAN: M_total and M_gas LOCKED")
    print("  Only freedom: bulge fraction f_b of stellar mass (M_total - M_gas)")
    print("  M_bulge = f_b * M_stellar, M_disk = (1-f_b) * M_stellar")
    print("=" * 135)
    print()

    errs_b = {"Mb": [], "Md": [], "Mg": []}
    errs_f = {"Mb": [], "Md": [], "Mg": []}

    print("  %-12s %3s %5s | ---- SPARC ---- | --- Method B --- | --- Budget Fit --- | --- Deltas ---" % (
        "Galaxy", "Cyc", "fb_s%"))
    print("  %-12s %3s %5s | %7s %7s %7s | %6s %6s %6s | %6s %6s %6s %5s | %5s %5s %5s %5s" % (
        "", "", "", "Mb", "Md", "Mg",
        "eMb%", "eMd%", "eMg%",
        "eMb%", "eMd%", "eMg%", "RMS",
        "fb%", "dMd%", "dRd%", "dab%"))
    print("  " + "-" * 135)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
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
        M_stellar_s = Mb_s + Md_s

        geom = solve_field_geometry(Mb_s, ab_s, Md_s, Rd_s, Mg_s, Rg_s, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")
        cycle = geom.get("cycle", 3)
        yN_at_rt = geom.get("yN_at_throat", 0.0)
        if not r_env or not r_t or r_env <= 0 or r_t <= 0:
            print("  %-12s  SKIP" % gid)
            continue

        # Method B: topology baseline
        sol_b = method_b(r_env, r_t, cycle, yN_at_rt, ab_s, Rd_s, Rg_s, a0_eff)
        if sol_b is None:
            print("  %-12s  SINGULAR" % gid)
            continue
        Mb_b, Md_b, Mg_b = sol_b
        Mt_b = Mb_b + Md_b + Mg_b

        # LOCK M_total and M_gas from topology
        Mt_locked = Mt_b
        Mg_locked = max(Mg_b, 0)
        M_stellar = Mt_locked - Mg_locked

        if M_stellar <= 0:
            print("  %-12s  SKIP (M_stellar <= 0)" % gid)
            continue

        # Md from Method B (locked gas, so Md_b is the topology estimate)
        Md_locked = max(Md_b, 1e3)

        # Budget fit: let Md wiggle +-5%, bulge absorbs residual
        fit = budget_fit_1d(
            obs, M_stellar, Mg_locked, Md_locked,
            ab_s, Rd_s, Rg_s, a0_eff
        )

        Mb_f = fit["Mb"]
        Md_f = fit["Md"]
        Mg_f = fit["Mg"]
        fb_f = fit["fb"] * 100
        rms_f = fit["rms"]

        # Method B errors
        eMb_b = pct(Mb_b, Mb_s) if Mb_s > 0 else 0
        eMd_b = pct(Md_b, Md_s) if Md_s > 0 else 0
        eMg_b = pct(Mg_b, Mg_s) if Mg_s > 0 else 0

        # Budget fit errors
        eMb_f = pct(Mb_f, Mb_s) if Mb_s > 0 else 0
        eMd_f = pct(Md_f, Md_s) if Md_s > 0 else 0
        eMg_f = pct(Mg_f, Mg_s) if Mg_s > 0 else 0

        # Track (for galaxies with actual bulge)
        if Mb_s > 0:
            errs_b["Mb"].append(abs(eMb_b))
            errs_f["Mb"].append(abs(eMb_f))
        if Md_s > 0:
            errs_b["Md"].append(abs(eMd_b))
            errs_f["Md"].append(abs(eMd_f))
        if Mg_s > 0:
            errs_b["Mg"].append(abs(eMg_b))
            errs_f["Mg"].append(abs(eMg_f))

        def cfmt(v):
            if abs(v) > 999:
                return ">999%"
            return "%+5.0f%%" % v

        print("  %-12s %3d %4.1f%% | %7s %7s %7s | %6s %6s %6s | %6s %6s %6s %5.1f | %4.1f%% %+4.0f%% %+4.0f%% %+4.0f%%" % (
            gid, cycle, fb_s,
            fmt(Mb_s), fmt(Md_s), fmt(Mg_s),
            cfmt(eMb_b), cfmt(eMd_b), cfmt(eMg_b),
            cfmt(eMb_f), cfmt(eMd_f), cfmt(eMg_f), rms_f,
            fit["fb"] * 100,
            fit["deltas"]["Md"] * 100,
            fit["deltas"]["Rd"] * 100,
            fit["deltas"]["ab"] * 100))

    # ===== PASS 2: Refine with tighter bands from Pass 1 =====
    print()
    print("=" * 135)
    print("  PASS 2: Lock gas +-3.5%, lock disk +-7.5%, let bulge move freely")
    print("  Uses Pass 1 results as starting point. Velocity curve resolves final split.")
    print("=" * 135)
    print()

    errs_p2 = {"Mb": [], "Md": [], "Mg": []}

    print("  %-12s %3s %5s | --- Pass 1 ---- | --- Pass 2 (final) --- | --- Deltas ---" % (
        "Galaxy", "Cyc", "fb_s%"))
    print("  %-12s %3s %5s | %6s %6s %6s | %6s %6s %6s %5s | %5s %5s %5s %5s %5s" % (
        "", "", "", "eMb%", "eMd%", "eMg%",
        "eMb%", "eMd%", "eMg%", "RMS",
        "fb%", "dMg%", "dMd%", "dRd%", "dab%"))
    print("  " + "-" * 131)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        mm = gal["mass_model"]
        obs = gal["observations"]
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

        sol_b = method_b(r_env, r_t, cycle, yN_at_rt, ab_s, Rd_s, Rg_s, a0_eff)
        if sol_b is None:
            continue
        Mb_b, Md_b, Mg_b = sol_b
        Mt_b = Mb_b + Md_b + Mg_b
        Mg_locked = max(Mg_b, 0)
        M_stellar = Mt_b - Mg_locked
        if M_stellar <= 0:
            continue
        Md_locked = max(Md_b, 1e3)

        # Pass 1
        fit1 = budget_fit_1d(
            obs, M_stellar, Mg_locked, Md_locked,
            ab_s, Rd_s, Rg_s, a0_eff
        )

        # Pass 2: use Pass 1 outputs as center, tighten bands
        fit2 = pass2_refine(
            obs, Mt_b,
            fit1["Mg"], fit1["Md"], fit1["Mb"],
            fit1["ab"], fit1["Rd"], fit1["Rg"],
            a0_eff
        )

        # Pass 1 errors (for display)
        eMb_1 = pct(fit1["Mb"], Mb_s) if Mb_s > 0 else 0
        eMd_1 = pct(fit1["Md"], Md_s) if Md_s > 0 else 0
        eMg_1 = pct(fit1["Mg"], Mg_s) if Mg_s > 0 else 0

        # Pass 2 errors
        eMb_2 = pct(fit2["Mb"], Mb_s) if Mb_s > 0 else 0
        eMd_2 = pct(fit2["Md"], Md_s) if Md_s > 0 else 0
        eMg_2 = pct(fit2["Mg"], Mg_s) if Mg_s > 0 else 0

        if Mb_s > 0:
            errs_p2["Mb"].append(abs(eMb_2))
        if Md_s > 0:
            errs_p2["Md"].append(abs(eMd_2))
        if Mg_s > 0:
            errs_p2["Mg"].append(abs(eMg_2))

        def cfmt(v):
            if abs(v) > 999:
                return ">999%"
            return "%+5.0f%%" % v

        print("  %-12s %3d %4.1f%% | %6s %6s %6s | %6s %6s %6s %5.1f | %4.1f%% %+4.0f%% %+4.0f%% %+4.0f%% %+4.0f%%" % (
            gid, cycle, fb_s,
            cfmt(eMb_1), cfmt(eMd_1), cfmt(eMg_1),
            cfmt(eMb_2), cfmt(eMd_2), cfmt(eMg_2), fit2["rms"],
            fit2["fb"] * 100,
            fit2["deltas"]["Mg"] * 100,
            fit2["deltas"]["Md"] * 100,
            fit2["deltas"]["Rd"] * 100,
            fit2["deltas"]["ab"] * 100))

    # Summary
    print()
    print("=" * 135)
    print("  FINAL SUMMARY")
    print("=" * 135)
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

    for label, errs in [("METHOD B (topology 3x3)", errs_b),
                         ("PASS 1 (M_total+M_gas locked, M_disk +-5%)", errs_f),
                         ("PASS 2 (gas +-3.5%, disk +-7.5%, bulge free)", errs_p2)]:
        print("  %s:" % label)
        for k, name in [("Mb", "M_bulge"), ("Md", "M_disk"), ("Mg", "M_gas")]:
            arr = errs[k]
            if arr:
                print("    %-10s  median|err| = %6.1f%%  mean = %6.1f%%  (N=%d)" % (
                    name, median(arr), mean(arr), len(arr)))
        print()


if __name__ == "__main__":
    main()
