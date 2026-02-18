"""
Prove: horizon M_total accuracy for ALL galaxies (2-cycle and 3-cycle).

The current solve_field_geometry bails out for dwarf galaxies when
yN never reaches 18/65 (no throat). But the horizon condition
yN = 36/1365 can still be reached by any galaxy with sufficient
mass for yN_peak > 36/1365.

This script solves for R_env independently of the throat, by:
  1. Scanning yN(r) from the published mass model
  2. Finding r_peak where yN is maximum
  3. If yN_peak >= 36/1365, binary searching the descending side
     for where yN = 36/1365 -> that is R_env
  4. Computing M_total = (36/1365) * R_env^2 * a0 / G
  5. Comparing to published M_total

This proves the horizon condition works for BOTH cycle types.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import G, M_SUN, KPC_TO_M, A0, THROAT_YN, HORIZON_YN
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


def total_enc(r, Mb, ab, Md, Rd, Mg, Rg):
    return hernquist_enc(r, Mb, ab) + disk_enc(r, Md, Rd) + disk_enc(r, Mg, Rg)


def yN_at(r, Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    enc = total_enc(r, Mb, ab, Md, Rd, Mg, Rg)
    r_m = r * KPC_TO_M
    if r_m <= 0 or enc <= 0:
        return 0.0
    return G * enc * M_SUN / (r_m * r_m * a0_eff)


def find_yN_peak(Mb, ab, Md, Rd, Mg, Rg, a0_eff, r_max=500.0, steps=5000):
    """Find radius where yN is maximum by scanning."""
    best_r = 0.01
    best_yN = 0.0
    dr = r_max / steps
    for i in range(1, steps + 1):
        r = dr * i
        y = yN_at(r, Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        if y > best_yN:
            best_yN = y
            best_r = r
    return best_r, best_yN


def solve_horizon_independent(Mb, ab, Md, Rd, Mg, Rg, a0_eff):
    """Solve for R_env where yN = 36/1365 on the descending side.

    Works for both 2-cycle and 3-cycle galaxies. Does not require
    the throat condition to be met first.

    Returns (r_env, r_peak, yN_peak, has_throat).
    """
    r_peak, yN_peak = find_yN_peak(Mb, ab, Md, Rd, Mg, Rg, a0_eff)

    has_throat = yN_peak >= THROAT_YN

    if yN_peak < HORIZON_YN:
        # Galaxy too small even for a horizon
        return None, r_peak, yN_peak, has_throat

    # Binary search on the descending side: from r_peak outward
    lo = r_peak
    hi = 2000.0  # generous upper bound
    for _ in range(150):
        mid = (lo + hi) / 2.0
        y = yN_at(mid, Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        if y > HORIZON_YN:
            lo = mid
        else:
            hi = mid
    r_env = (lo + hi) / 2.0

    return r_env, r_peak, yN_peak, has_throat


def solve_throat(Mb, ab, Md, Rd, Mg, Rg, a0_eff, r_peak):
    """Solve for R_t where yN = 18/65 on the descending side.

    Only called for 3-cycle galaxies.
    """
    lo = r_peak
    hi = 500.0
    for _ in range(150):
        mid = (lo + hi) / 2.0
        y = yN_at(mid, Mb, ab, Md, Rd, Mg, Rg, a0_eff)
        if y > THROAT_YN:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def horizon_mass(r_env_kpc, a0_eff):
    r_m = r_env_kpc * KPC_TO_M
    return HORIZON_YN * r_m * r_m * a0_eff / (G * M_SUN)


def pct(derived, published):
    if published == 0:
        return 0.0 if derived == 0 else float('inf')
    return (derived - published) / published * 100.0


def fmt(m):
    if m == 0:
        return "    0       "
    exp = int(math.floor(math.log10(abs(m))))
    coeff = m / (10 ** exp)
    return "{:.2f}e{:+d}".format(coeff, exp)


def main():
    a0_eff = A0

    print("=" * 110)
    print("HORIZON M_TOTAL ACCURACY: ALL GALAXIES (2-CYCLE AND 3-CYCLE)")
    print("=" * 110)
    print()
    print(
        "{:<14s} {:>5s}  {:>8s}  {:>8s}  {:>8s}  "
        "{:>12s}  {:>12s}  {:>7s}  {:>8s}".format(
            "Galaxy", "Cycle", "yN_peak", "R_peak", "R_env",
            "M_pub", "M_horizon", "%err", "R_t/Renv"
        ))
    print("-" * 110)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid.endswith("_inference"):
            continue

        mm = gal.get("mass_model", {})
        b = mm.get("bulge", {})
        d = mm.get("disk", {})
        g = mm.get("gas", {})

        Mb = b.get("M", 0)
        ab = b.get("a", 0.1)
        Md = d.get("M", 0)
        Rd = d.get("Rd", 1.0)
        Mg = g.get("M", 0)
        Rg = g.get("Rd", 1.0)
        Mt_pub = Mb + Md + Mg

        r_env, r_peak, yN_peak, has_throat = solve_horizon_independent(
            Mb, ab, Md, Rd, Mg, Rg, a0_eff)

        cycle = 3 if has_throat else 2

        if r_env is None:
            print("{:<14s} {:>5d}  {:>8.4f}  {:>8.2f}  {:>8s}  "
                  "{:>12s}  {:>12s}  {:>7s}  {:>8s}".format(
                      gid, cycle, yN_peak, r_peak, "N/A",
                      fmt(Mt_pub), "N/A", "N/A", "N/A"))
            continue

        Mt_hor = horizon_mass(r_env, a0_eff)
        err = pct(Mt_hor, Mt_pub)

        rt_str = "N/A"
        if has_throat:
            r_t = solve_throat(Mb, ab, Md, Rd, Mg, Rg, a0_eff, r_peak)
            rt_str = "{:.4f}".format(r_t / r_env)

        print(
            "{:<14s} {:>5d}  {:>8.4f}  {:>8.2f}  {:>8.2f}  "
            "{:>12s}  {:>12s}  {:>+7.1f}  {:>8s}".format(
                gid, cycle, yN_peak, r_peak, r_env,
                fmt(Mt_pub), fmt(Mt_hor), err, rt_str))

    print()
    print("=" * 110)
    print("Key:")
    print("  Cycle: 3 = throat exists (yN_peak >= 18/65 = {:.4f})".format(
        THROAT_YN))
    print("         2 = no throat (yN_peak < 18/65), gas-dominated")
    print("  M_horizon = (36/1365) * R_env^2 * a0 / G")
    print("  R_t/Renv should be ~0.30 for 3-cycle")
    print("=" * 110)


if __name__ == "__main__":
    main()
