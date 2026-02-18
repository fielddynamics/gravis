"""
Topology-constrained linear fit for mass decomposition.

Strategy:
1. Invert observations through GFD to get M_enc(r) at each point (exact)
2. Use SPARC R_env (best case) to predict scale lengths from topology
3. With scale lengths fixed, solve for Mb, Md, Mg via linear least squares
4. Compare to SPARC published masses

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import A0, G, M_SUN, KPC_TO_M
from physics.services.rotation.inference import solve_field_geometry
from physics.services.sandbox.pure_inference import invert_velocity_to_mass
from data.galaxies import PREDICTION_GALAXIES


HORIZON_YN = 36.0 / 1365.0
THROAT_YN = 18.0 / 65.0


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


def fit_masses_linear(obs_r, obs_Menc, a, Rd, Rg):
    """Fit Mb, Md, Mg by non-negative linear least squares.

    Given fixed scale lengths (a, Rd, Rg), the enclosed mass at
    each radius is linear in the three masses:

        M_enc(r) = Mb * f_b(r,a) + Md * f_d(r,Rd) + Mg * f_g(r,Rg)

    where f_b, f_d, f_g are the normalized enclosed fractions.
    This is a standard NNLS problem (masses must be >= 0).
    """
    n = len(obs_r)
    A = np.zeros((n, 3))
    b = np.array(obs_Menc, dtype=np.float64)

    for i in range(n):
        A[i, 0] = hernquist_frac(obs_r[i], a)
        A[i, 1] = disk_frac(obs_r[i], Rd)
        A[i, 2] = disk_frac(obs_r[i], Rg)

    x, residual = nnls(A, b)
    return x[0], x[1], x[2], residual


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
    print("TOPOLOGY-CONSTRAINED LINEAR FIT: Fix scale lengths from R_env, solve masses linearly")
    print("Using SPARC R_env (best case) to establish what this approach can achieve")
    print("=" * 160)
    print()

    header = (
        "{:<13s} {:>5s} {:>7s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}  "
        "{:>10s} {:>10s} {:>6s}"
    ).format(
        "Galaxy", "Cycle", "R_env",
        "Mb_fit", "Mb_sp", "dMb%",
        "Md_fit", "Md_sp", "dMd%",
        "Mg_fit", "Mg_sp", "dMg%",
        "Mt_fit", "Mt_sp", "dMt%",
    )
    print(header)
    print("-" * 160)

    mt_deltas = []
    mb_deltas = []
    md_deltas = []
    mg_deltas = []

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

        # SPARC published
        sb = mm.get("bulge", {})
        sd = mm.get("disk", {})
        sg = mm.get("gas", {})
        Mb_s = sb.get("M", 0)
        Md_s = sd.get("M", 0)
        Mg_s = sg.get("M", 0)
        Mt_s = Mb_s + Md_s + Mg_s

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

        # Topology-predicted scale lengths
        if cycle == 3:
            a_pred = 0.01 * r_env
            Rd_pred = 0.09 * r_env
            Rg_pred = 0.21 * r_env
        else:
            a_pred = 0.01 * r_env
            Rd_pred = 0.30 * r_env
            Rg_pred = 1.00 * r_env

        # Invert observations to get M_enc(r)
        obs_r = []
        obs_Menc = []
        for o in observations:
            r = float(o.get("r", 0))
            v = float(o.get("v", 0))
            if r > 0 and v > 0:
                M = invert_velocity_to_mass(r, v, a0)
                obs_r.append(r)
                obs_Menc.append(M)

        if len(obs_r) < 3:
            continue

        # Linear fit for masses
        Mb_f, Md_f, Mg_f, resid = fit_masses_linear(
            obs_r, obs_Menc, a_pred, Rd_pred, Rg_pred)
        Mt_f = Mb_f + Md_f + Mg_f

        dMb = pct(Mb_f, Mb_s)
        dMd = pct(Md_f, Md_s)
        dMg = pct(Mg_f, Mg_s)
        dMt = pct(Mt_f, Mt_s)

        if dMt is not None:
            mt_deltas.append(dMt)
        if dMb is not None:
            mb_deltas.append(dMb)
        if dMd is not None:
            md_deltas.append(dMd)
        if dMg is not None:
            mg_deltas.append(dMg)

        row = (
            "{:<13s} {:>5d} {:>7.1f}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}  "
            "{:>10s} {:>10s} {:>6s}"
        ).format(
            gid, cycle, r_env,
            fmt_m(Mb_f), fmt_m(Mb_s), fmt_p(dMb),
            fmt_m(Md_f), fmt_m(Md_s), fmt_p(dMd),
            fmt_m(Mg_f), fmt_m(Mg_s), fmt_p(dMg),
            fmt_m(Mt_f), fmt_m(Mt_s), fmt_p(dMt),
        )
        print(row)

    print()
    print("=" * 160)
    print("SUMMARY")
    print("-" * 160)

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

    print()
    stats(mt_deltas, "M_total")
    stats(mb_deltas, "M_bulge")
    stats(md_deltas, "M_disk")
    stats(mg_deltas, "M_gas")
    print()
    print("=" * 160)


if __name__ == "__main__":
    main()
