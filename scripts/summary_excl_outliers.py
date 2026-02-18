#!/usr/bin/env python3
"""
Quick summary: Pass 1 results excluding NGC891, NGC5055, IC2574.
"""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.galaxies import PREDICTION_GALAXIES
from physics.constants import A0, G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry

EXCLUDE = {"ngc891", "ngc5055", "ic2574"}


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
    from scipy.optimize import differential_evolution
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
        prior = (f_Md / sigma_Md) ** 2
        prior += (f_Rd / sigma_Rd) ** 2 + (f_ab / sigma_ab) ** 2
        return chi2 + prior
    bounds = [(-sigma_Md, sigma_Md), (-sigma_Rd, sigma_Rd), (-sigma_ab, sigma_ab)]
    result = differential_evolution(
        cost, bounds, seed=42, maxiter=300, tol=1e-9,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9, polish=True)
    f_Md, f_Rd, f_ab = result.x
    Md_out = Md_init * (1.0 + f_Md)
    Mb_out = M_stellar - Md_out
    Rd_out = Rd_init * (1.0 + f_Rd)
    ab_out = ab_init * (1.0 + f_ab)
    return {"Mb": Mb_out, "Md": Md_out, "Mg": Mg_fixed}

def pct(pred, actual):
    if actual == 0:
        return 0.0 if pred == 0 else 999.9
    return (pred - actual) / actual * 100

def main():
    errs_all = {"Mb": [], "Md": [], "Mg": [], "Mt": []}
    errs_clean = {"Mb": [], "Md": [], "Mg": [], "Mt": []}

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

        fit = budget_fit_1d(obs, M_stellar, Mg_locked, Md_locked,
                             ab_s, Rd_s, Rg_s, a0_eff)

        eMt = pct(Mt_b, Mt_s)
        eMb = pct(fit["Mb"], Mb_s) if Mb_s > 0 else None
        eMd = pct(fit["Md"], Md_s) if Md_s > 0 else None
        eMg = pct(fit["Mg"], Mg_s) if Mg_s > 0 else None

        # All galaxies
        if eMt is not None:
            errs_all["Mt"].append(abs(eMt))
        if eMb is not None:
            errs_all["Mb"].append(abs(eMb))
        if eMd is not None:
            errs_all["Md"].append(abs(eMd))
        if eMg is not None:
            errs_all["Mg"].append(abs(eMg))

        # Excluding outliers
        if gid not in EXCLUDE:
            if eMt is not None:
                errs_clean["Mt"].append(abs(eMt))
            if eMb is not None:
                errs_clean["Mb"].append(abs(eMb))
            if eMd is not None:
                errs_clean["Md"].append(abs(eMd))
            if eMg is not None:
                errs_clean["Mg"].append(abs(eMg))

    def median(arr):
        if not arr:
            return float('nan')
        s = sorted(arr)
        return s[len(s) // 2]
    def mean(arr):
        if not arr:
            return float('nan')
        return sum(arr) / len(arr)

    print("=" * 80)
    print("  PASS 1 RESULTS: ALL 22 GALAXIES")
    print("=" * 80)
    for k, name in [("Mt", "M_total"), ("Mg", "M_gas"), ("Md", "M_disk"), ("Mb", "M_bulge")]:
        arr = errs_all[k]
        if arr:
            print("    %-10s  median = %5.1f%%   mean = %5.1f%%   (N=%d)" % (
                name, median(arr), mean(arr), len(arr)))

    print()
    print("=" * 80)
    print("  PASS 1 RESULTS: 19 GALAXIES (excluding NGC891, NGC5055, IC2574)")
    print("=" * 80)
    for k, name in [("Mt", "M_total"), ("Mg", "M_gas"), ("Md", "M_disk"), ("Mb", "M_bulge")]:
        arr = errs_clean[k]
        if arr:
            print("    %-10s  median = %5.1f%%   mean = %5.1f%%   (N=%d)" % (
                name, median(arr), mean(arr), len(arr)))


if __name__ == "__main__":
    main()
