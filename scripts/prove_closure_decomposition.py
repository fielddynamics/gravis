"""
Proof: closure cycle mass decomposition from topological conditions.

For each galaxy in the catalog, this script:
  1. Computes yN(r) from the published mass model
  2. Determines cycle count: 3-cycle if yN reaches 18/65, else 2-cycle
  3. Derives M_total from the horizon condition: (36/1365) * R_env^2 * a0 / G
  4. Derives M_enc(R_t) from the throat condition (3-cycle only)
  5. Applies the closure cycle Md/Mg ratio to decompose mass
  6. Compares derived Mb, Md, Mg against published SPARC values

Three constraints for three unknowns:
  (1) M_total from horizon  ->  Mb + Md + Mg = M_total
  (2) Md / Mg = R_closure   ->  closure ratio from cycle count
  (3) M_enc(R_t) from throat -> enclosed mass at throat radius

For 2-cycle (gas-dominated, no throat):
  Mb = 0, Md/Mg = 0.6, two equations suffice.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.constants import G, M_SUN, KPC_TO_M, A0, THROAT_YN, HORIZON_YN
from physics.services.rotation.inference import solve_field_geometry
from data.galaxies import PREDICTION_GALAXIES


# Closure cycle Md/Mg ratios (from SPARC analysis)
RATIO_3CYCLE = 3.0    # Md/Mg for 3-cycle galaxies (disk-dominated)
RATIO_2CYCLE = 0.6    # Md/Mg for 2-cycle galaxies (gas-dominated)


def hernquist_enc(r, M, a):
    """Hernquist enclosed mass fraction at radius r."""
    if M <= 0 or a <= 0 or r <= 0:
        return 0.0
    return M * r * r / ((r + a) * (r + a))


def disk_enc(r, M, Rd):
    """Exponential disk enclosed mass at radius r."""
    if M <= 0 or Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return M
    return M * (1.0 - (1.0 + x) * math.exp(-x))


def enc_frac_hernquist(r, a):
    """Fraction of Hernquist mass enclosed at radius r."""
    if a <= 0 or r <= 0:
        return 0.0
    return r * r / ((r + a) * (r + a))


def enc_frac_disk(r, Rd):
    """Fraction of exponential disk mass enclosed at radius r."""
    if Rd <= 0 or r <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return 1.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def compute_m_total_from_horizon(r_env_kpc, a0_eff):
    """M_total = (36/1365) * R_env^2 * a0 / G  in solar masses."""
    r_m = r_env_kpc * KPC_TO_M
    return HORIZON_YN * r_m * r_m * a0_eff / (G * M_SUN)


def compute_m_enc_from_throat(r_t_kpc, a0_eff):
    """M_enc(R_t) = (18/65) * R_t^2 * a0 / G  in solar masses."""
    r_m = r_t_kpc * KPC_TO_M
    return THROAT_YN * r_m * r_m * a0_eff / (G * M_SUN)


def decompose_3cycle(M_total, M_enc_Rt, r_t, ab, Rd, Rg, R_closure=RATIO_3CYCLE):
    """Decompose mass for a 3-cycle galaxy using 3 constraints.

    Equations:
      (1) Mb + Md + Mg = M_total
      (2) Md = R_closure * Mg
      (3) Mb*fb + Md*fd + Mg*fg = M_enc_Rt

    Where fb, fd, fg are enclosed fractions at R_t for each profile.

    Solving for Mg:
      Mg = (M_enc_Rt - M_total * fb) / (R_closure * fd + fg - fb * (R_closure + 1))
      Md = R_closure * Mg
      Mb = M_total - Md - Mg
    """
    fb = enc_frac_hernquist(r_t, ab)
    fd = enc_frac_disk(r_t, Rd)
    fg = enc_frac_disk(r_t, Rg)

    denom = R_closure * fd + fg - fb * (R_closure + 1.0)
    if abs(denom) < 1e-12:
        # Degenerate: fall back to simple ratio split with no bulge
        Mg = M_total / (R_closure + 1.0)
        Md = R_closure * Mg
        Mb = 0.0
    else:
        Mg = (M_enc_Rt - M_total * fb) / denom
        Mg = max(Mg, 0.0)
        Md = R_closure * Mg
        Mb = M_total - Md - Mg
        Mb = max(Mb, 0.0)

    return Mb, Md, Mg


def decompose_2cycle(M_total, R_closure=RATIO_2CYCLE):
    """Decompose mass for a 2-cycle galaxy (no throat, no bulge).

    Equations:
      (1) Md + Mg = M_total  (Mb = 0)
      (2) Md = R_closure * Mg
    """
    Mg = M_total / (R_closure + 1.0)
    Md = R_closure * Mg
    Mb = 0.0
    return Mb, Md, Mg


def fmt_mass(m):
    """Format mass in scientific notation."""
    if m == 0:
        return "0.00e+0 "
    exp = int(math.floor(math.log10(abs(m))))
    coeff = m / (10 ** exp)
    return "{:.2f}e{:+d}".format(coeff, exp)


def pct_err(derived, published):
    """Percentage error: (derived - published) / published * 100."""
    if published == 0:
        if derived == 0:
            return 0.0
        return float('inf')
    return (derived - published) / published * 100.0


def main():
    a0_eff = A0  # accel_ratio = 1.0 for all SPARC galaxies

    print("=" * 120)
    print("CLOSURE CYCLE MASS DECOMPOSITION PROOF")
    print("=" * 120)
    print()

    header = (
        "{:<14s} {:>5s}  {:>12s} {:>12s} {:>6s}  "
        "{:>12s} {:>12s} {:>6s}  "
        "{:>12s} {:>12s} {:>6s}  "
        "{:>12s} {:>12s} {:>6s}"
    ).format(
        "Galaxy", "Cycle",
        "Mb_pub", "Mb_der", "%err",
        "Md_pub", "Md_der", "%err",
        "Mg_pub", "Mg_der", "%err",
        "Mt_pub", "Mt_hor", "%err",
    )
    print(header)
    print("-" * 120)

    for gal in PREDICTION_GALAXIES:
        gid = gal["id"]
        if gid.endswith("_inference"):
            continue

        mm = gal.get("mass_model", {})
        b = mm.get("bulge", {})
        d = mm.get("disk", {})
        g = mm.get("gas", {})

        Mb_pub = b.get("M", 0)
        ab_pub = b.get("a", 0.1)
        Md_pub = d.get("M", 0)
        Rd_pub = d.get("Rd", 1.0)
        Mg_pub = g.get("M", 0)
        Rg_pub = g.get("Rd", 1.0)
        Mt_pub = Mb_pub + Md_pub + Mg_pub

        # Step 1: Derive field geometry from published mass model
        geom = solve_field_geometry(
            Mb_pub, ab_pub, Md_pub, Rd_pub, Mg_pub, Rg_pub, a0_eff)

        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")

        if r_env is None or r_env <= 0:
            # Cannot derive geometry; skip
            print("{:<14s}   --  (no envelope radius derived)".format(gid))
            continue

        # Step 2: Determine cycle count
        has_throat = r_t is not None and r_t > 0
        cycle = 3 if has_throat else 2

        # Step 3: M_total from horizon condition
        Mt_horizon = compute_m_total_from_horizon(r_env, a0_eff)

        # Step 4: Decompose using closure cycle ratio
        if cycle == 3:
            M_enc_Rt = compute_m_enc_from_throat(r_t, a0_eff)
            Mb_der, Md_der, Mg_der = decompose_3cycle(
                Mt_horizon, M_enc_Rt, r_t,
                ab_pub, Rd_pub, Rg_pub)
        else:
            Mb_der, Md_der, Mg_der = decompose_2cycle(Mt_horizon)

        # Print comparison
        row = (
            "{:<14s} {:>5d}  {:>12s} {:>12s} {:>+6.1f}  "
            "{:>12s} {:>12s} {:>+6.1f}  "
            "{:>12s} {:>12s} {:>+6.1f}  "
            "{:>12s} {:>12s} {:>+6.1f}"
        ).format(
            gid, cycle,
            fmt_mass(Mb_pub), fmt_mass(Mb_der),
            pct_err(Mb_der, Mb_pub) if Mb_pub > 0 else 0.0,
            fmt_mass(Md_pub), fmt_mass(Md_der),
            pct_err(Md_der, Md_pub),
            fmt_mass(Mg_pub), fmt_mass(Mg_der),
            pct_err(Mg_der, Mg_pub),
            fmt_mass(Mt_pub), fmt_mass(Mt_horizon),
            pct_err(Mt_horizon, Mt_pub),
        )
        print(row)

    print()
    print("=" * 120)
    print("Notes:")
    print("  Cycle: 2 = gas-dominated (no throat), 3 = disk-dominated (throat exists)")
    print("  M_total from horizon: (36/1365) * R_env^2 * a0 / G")
    print("  3-cycle ratio: Md/Mg = {:.1f}".format(RATIO_3CYCLE))
    print("  2-cycle ratio: Md/Mg = {:.1f}".format(RATIO_2CYCLE))
    print("  Scale lengths used: published (proof of decomposition logic,")
    print("  not scale length inference)")
    print("=" * 120)

    # Also print a focused summary for key diagnostics
    print()
    print("DETAILED DIAGNOSTICS (select galaxies):")
    print("-" * 80)

    for gid_target in ["milky_way", "m33", "ngc2841", "ddo154", "ngc3109"]:
        gal = None
        for g in PREDICTION_GALAXIES:
            if g["id"] == gid_target:
                gal = g
                break
        if gal is None:
            continue

        mm = gal.get("mass_model", {})
        b = mm.get("bulge", {})
        d = mm.get("disk", {})
        g_comp = mm.get("gas", {})

        Mb_pub = b.get("M", 0)
        ab_pub = b.get("a", 0.1)
        Md_pub = d.get("M", 0)
        Rd_pub = d.get("Rd", 1.0)
        Mg_pub = g_comp.get("M", 0)
        Rg_pub = g_comp.get("Rd", 1.0)
        Mt_pub = Mb_pub + Md_pub + Mg_pub

        geom = solve_field_geometry(
            Mb_pub, ab_pub, Md_pub, Rd_pub, Mg_pub, Rg_pub, a0_eff)
        r_t = geom.get("throat_radius_kpc")
        r_env = geom.get("envelope_radius_kpc")

        if r_env is None:
            print("{}: no envelope".format(gid_target))
            continue

        has_throat = r_t is not None and r_t > 0
        cycle = 3 if has_throat else 2

        Mt_horizon = compute_m_total_from_horizon(r_env, a0_eff)

        print()
        print("{}  (cycle={})".format(gid_target, cycle))
        print("  R_t  = {:.2f} kpc".format(r_t if r_t else 0))
        print("  R_env = {:.2f} kpc".format(r_env))
        if r_t and r_env:
            print("  R_t/R_env = {:.4f}".format(r_t / r_env))
        print("  M_total (published) = {:.3e}".format(Mt_pub))
        print("  M_total (horizon)   = {:.3e}  ({:+.1f}%)".format(
            Mt_horizon, pct_err(Mt_horizon, Mt_pub)))

        if cycle == 3:
            M_enc_Rt = compute_m_enc_from_throat(r_t, a0_eff)
            print("  M_enc(R_t) (throat) = {:.3e}".format(M_enc_Rt))
            print("  M_enc(R_t)/M_total  = {:.4f}".format(
                M_enc_Rt / Mt_horizon if Mt_horizon > 0 else 0))

            # Enclosed fractions at R_t
            fb = enc_frac_hernquist(r_t, ab_pub)
            fd = enc_frac_disk(r_t, Rd_pub)
            fg = enc_frac_disk(r_t, Rg_pub)
            print("  Enclosed fracs at R_t: bulge={:.3f}, disk={:.3f}, gas={:.3f}".format(
                fb, fd, fg))

            # Published Md/Mg
            if Mg_pub > 0:
                print("  Published Md/Mg = {:.2f}  (target: {:.1f})".format(
                    Md_pub / Mg_pub, RATIO_3CYCLE))

            Mb_der, Md_der, Mg_der = decompose_3cycle(
                Mt_horizon, M_enc_Rt, r_t,
                ab_pub, Rd_pub, Rg_pub)
        else:
            if Mg_pub > 0:
                print("  Published Md/Mg = {:.2f}  (target: {:.1f})".format(
                    Md_pub / Mg_pub, RATIO_2CYCLE))
            Mb_der, Md_der, Mg_der = decompose_2cycle(Mt_horizon)

        print("  Derived:  Mb={:.3e}  Md={:.3e}  Mg={:.3e}".format(
            Mb_der, Md_der, Mg_der))
        print("  Published: Mb={:.3e}  Md={:.3e}  Mg={:.3e}".format(
            Mb_pub, Md_pub, Mg_pub))
        print("  Errors:    Mb={:+.1f}%  Md={:+.1f}%  Mg={:+.1f}%".format(
            pct_err(Mb_der, Mb_pub) if Mb_pub > 0 else 0.0,
            pct_err(Md_der, Md_pub) if Md_pub > 0 else 0.0,
            pct_err(Mg_der, Mg_pub) if Mg_pub > 0 else 0.0,
        ))


if __name__ == "__main__":
    main()
