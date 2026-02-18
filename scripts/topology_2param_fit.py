"""
2-parameter mass fraction fit with topology-fixed scale lengths and M_total.

Strategy:
  1. Use SPARC R_env (best case test) to get M_total from topology
  2. Fix scale lengths: a=0.01*R_env, Rd=0.09*R_env, Rg=0.21*R_env
  3. Search only 2 free parameters: fb, fd (fg = 1-fb-fd)
  4. All masses: Mb=fb*Mt, Md=fd*Mt, Mg=fg*Mt
  5. Minimize velocity chi-squared through the GFD equation

This is a 2D search vs the original 6D, leveraging topology constraints.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.optimize import differential_evolution, minimize
from physics.constants import A0, G, M_SUN, KPC_TO_M
from physics.aqual import solve_x as aqual_solve_x
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.pure_inference import (
    horizon_enclosed_mass,
    derive_mass_from_topology,
)
from data.galaxies import PREDICTION_GALAXIES


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


def gfd_velocity(r_kpc, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    enc = hernquist_enc(r_kpc, Mb, ab) + disk_enc(r_kpc, Md, Rd) + disk_enc(r_kpc, Mg, Rg)
    r_m = r_kpc * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    gN = G * enc * M_SUN / (r_m * r_m)
    y = gN / a0_eff
    x = aqual_solve_x(y)
    return math.sqrt(a0_eff * x * r_m) / 1000.0


def chi2_cost_2param(params, obs_r, obs_v, obs_w, M_total, a, Rd, Rg, a0_eff):
    """Chi-squared with only 2 free parameters: fb, fd."""
    fb, fd = params
    fg = 1.0 - fb - fd
    if fg < 0:
        return 1e20

    Mb = fb * M_total
    Md = fd * M_total
    Mg = fg * M_total

    cost = 0.0
    for j in range(len(obs_r)):
        v_pred = gfd_velocity(obs_r[j], Mb, a, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - v_pred
        cost += obs_w[j] * delta * delta
    return cost


def fit_2param(obs_r, obs_v, obs_err, M_total, a, Rd, Rg, a0_eff, seed=42):
    """Fit mass fractions fb, fd via differential evolution (2D)."""
    obs_w = [1.0 / (e * e) for e in obs_err]

    # Bounds: fb in [0, 0.50], fd in [0, 0.95], with fg = 1-fb-fd >= 0
    bounds = [
        (0.0, 0.50),   # fb (bulge fraction)
        (0.0, 0.95),   # fd (disk fraction)
    ]

    result = differential_evolution(
        chi2_cost_2param,
        bounds=bounds,
        args=(obs_r, obs_v, obs_w, M_total, a, Rd, Rg, a0_eff),
        seed=seed,
        maxiter=200,
        tol=1e-10,
        popsize=30,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
    )

    # Local polish
    polished = minimize(
        chi2_cost_2param,
        result.x,
        args=(obs_r, obs_v, obs_w, M_total, a, Rd, Rg, a0_eff),
        method="L-BFGS-B",
        bounds=bounds,
    )

    best = polished.x if polished.success else result.x
    fb, fd = best
    fg = 1.0 - fb - fd

    Mb = fb * M_total
    Md = fd * M_total
    Mg = fg * M_total

    # Compute RMS
    n = len(obs_r)
    ss = 0.0
    for j in range(n):
        vp = gfd_velocity(obs_r[j], Mb, a, Md, Rd, Mg, Rg, a0_eff)
        delta = obs_v[j] - vp
        ss += delta * delta
    rms = math.sqrt(ss / n) if n > 0 else 0.0

    return fb, fd, fg, Mb, Md, Mg, rms


def pct(fit, pub):
    if abs(pub) < 1e3:
        if abs(fit) < 1e3:
            return 0.0
        return None
    return (fit - pub) / abs(pub) * 100.0


def fmt_m(v):
    if abs(v) < 1:
        return "0"
    return "%.3e" % v


def fmt_p(v):
    if v is None:
        return "N/A"
    return "%+.1f%%" % v


def main():
    print("=" * 160)
    print("2-PARAMETER FIT: Topology-fixed scale lengths + M_total, search fb/fd only")
    print("Using SPARC R_env (best case)")
    print("3-cycle: a=0.01*Re, Rd=0.09*Re, Rg=0.21*Re")
    print("2-cycle: a=0.01*Re, Rd=0.30*Re, Rg=1.00*Re")
    print("=" * 160)
    print()

    header = (
        "{:<13s} {:>5s}  "
        "{:>5s} {:>5s} {:>5s}  "
        "{:>5s} {:>5s} {:>5s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>6s} {:>6s}"
    ).format(
        "Galaxy", "Cycle",
        "fb_f", "fd_f", "fg_f",
        "fb_s", "fd_s", "fg_s",
        "Md_fit", "Md_sp", "dMd%",
        "Mg_fit", "Mg_sp", "dMg%",
        "Mt_fit", "Mt_sp", "dMt%",
        "RMS", "BayRMS",
    )
    print(header)
    print("-" * 160)

    mt_d = []; md_d = []; mg_d = []; fb_d = []; fd_d = []; fg_d = []

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid.endswith("_inference"):
            continue
        observations = gal.get("observations", [])
        mm = gal.get("mass_model", {})
        accel = gal.get("accel", 1.0)
        a0 = A0 * accel

        if not observations or len(observations) < 3 or not mm:
            continue

        sb = mm.get("bulge", {})
        sd = mm.get("disk", {})
        sg = mm.get("gas", {})
        Mb_s = sb.get("M", 0)
        Md_s = sd.get("M", 0)
        Mg_s = sg.get("M", 0)
        Mt_s = Mb_s + Md_s + Mg_s
        fb_s = Mb_s / Mt_s if Mt_s > 0 else 0
        fd_s = Md_s / Mt_s if Mt_s > 0 else 0
        fg_s = Mg_s / Mt_s if Mt_s > 0 else 0

        # Field geometry from SPARC
        geom = solve_field_geometry(
            Mb_s, sb.get("a", 0.1),
            Md_s, sd.get("Rd", 1),
            Mg_s, sg.get("Rd", 1),
            a0,
        )
        r_env = geom.get("envelope_radius_kpc", 0) or 0
        cycle = geom.get("cycle", 3)

        if r_env <= 0:
            continue

        # M_total from topology
        mass_params = (Mb_s, sb.get("a", 0.1), Md_s, sd.get("Rd", 1),
                        Mg_s, sg.get("Rd", 1))
        topo = derive_mass_from_topology(geom, a0, mass_params=mass_params)
        M_total = topo.get("M_total", 0)

        # Scale lengths from topology
        if cycle == 3:
            a_pred = 0.01 * r_env
            Rd_pred = 0.09 * r_env
            Rg_pred = 0.21 * r_env
        else:
            a_pred = 0.01 * r_env
            Rd_pred = 0.30 * r_env
            Rg_pred = 1.00 * r_env

        # Extract observations
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
            continue

        # 6-param Bayesian RMS for comparison
        from physics.services.sandbox.bayesian_fit import fit_gfd_to_observations_with_bayesian
        bay = fit_gfd_to_observations_with_bayesian(obs_r, obs_v, obs_err, a0)
        bay_rms = bay.get("rms", -1)

        # 2-param fit
        fb_f, fd_f, fg_f, Mb_f, Md_f, Mg_f, rms = fit_2param(
            obs_r, obs_v, obs_err, M_total, a_pred, Rd_pred, Rg_pred, a0)
        Mt_f = Mb_f + Md_f + Mg_f

        dMd = pct(Md_f, Md_s)
        dMg = pct(Mg_f, Mg_s)
        dMt = pct(Mt_f, Mt_s)

        if dMt is not None: mt_d.append(dMt)
        if dMd is not None: md_d.append(dMd)
        if dMg is not None: mg_d.append(dMg)
        fb_d.append(abs(fb_f - fb_s) * 100)
        fd_d.append(abs(fd_f - fd_s) * 100)
        fg_d.append(abs(fg_f - fg_s) * 100)

        row = (
            "{:<13s} {:>5d}  "
            "{:>4.1f}% {:>4.1f}% {:>4.1f}%  "
            "{:>4.1f}% {:>4.1f}% {:>4.1f}%  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>6.2f} {:>6.2f}"
        ).format(
            gid, cycle,
            fb_f * 100, fd_f * 100, fg_f * 100,
            fb_s * 100, fd_s * 100, fg_s * 100,
            fmt_m(Md_f), fmt_m(Md_s), fmt_p(dMd),
            fmt_m(Mg_f), fmt_m(Mg_s), fmt_p(dMg),
            fmt_m(Mt_f), fmt_m(Mt_s), fmt_p(dMt),
            rms, bay_rms,
        )
        print(row)

    print()
    print("=" * 160)
    print("SUMMARY")
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
            "  %-25s  median=%+7.1f%%  |median|=%5.1f%%  |mean|=%5.1f%%  (n=%d)"
            % (name, med, abs_med, abs_avg, n)
        )

    def stats_pp(lst, name):
        if not lst:
            return
        s = sorted(lst)
        n = len(s)
        med = s[n // 2]
        avg = sum(lst) / n
        print(
            "  %-25s  median=%5.1f pp  mean=%5.1f pp  (n=%d)"
            % (name, med, avg, n)
        )

    stats(mt_d, "M_total")
    stats(md_d, "M_disk")
    stats(mg_d, "M_gas")
    print()
    print("  Fraction errors (percentage points):")
    stats_pp(fb_d, "fb (bulge fraction)")
    stats_pp(fd_d, "fd (disk fraction)")
    stats_pp(fg_d, "fg (gas fraction)")
    print()
    print("=" * 160)


if __name__ == "__main__":
    main()
