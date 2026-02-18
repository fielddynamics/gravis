"""Check if the coupling polynomial f(k) = 1+k+k^2 = 21 governs
the mass scaling between GFD base and GFD sigma across the field.

If the structural correction accumulates through three coupling modes
(1, k, k^2), the mass excess at fractional radii should follow
the polynomial terms.
"""

import math

from data.galaxies import get_prediction_galaxies
from physics.services.sandbox.pure_inference import (
    invert_observations, fit_mass_model, _model_enc, _gfd_vel,
    search_throughput, _build_sigma_stage, invert_velocity_to_mass,
    THROAT_FRAC
)
from physics.services.rotation.inference import solve_field_geometry
from physics.sigma import GfdSymmetricStage
from physics.constants import A0

FK = 21.0   # f(k) = 1 + k + k^2 at k=4
K = 4


def main():
    all_gals = get_prediction_galaxies()
    results = []

    for gal in all_gals:
        nm = gal.get('name', '?')
        if 'observations' not in gal:
            continue
        try:
            obs = gal['observations']
            accel = gal.get('accel', 1.0)
            a0_eff = A0 * accel
            inverted = invert_observations(obs, a0_eff)
            fit = fit_mass_model(inverted, a0_eff)
            if fit is None:
                continue
            params = fit['params']
            Mb, ab, Md, Rd, Mg, Rg = params
            fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
            pure_rt = fg['throat_radius_kpc']
            pure_renv = fg['envelope_radius_kpc']
            if pure_rt is None or pure_rt <= 0:
                continue

            tp = search_throughput(
                inverted, params, pure_renv, accel, a0_eff, 500)
            sigma = tp['throughput']
            if abs(sigma) < 0.01:
                continue
            f_gas_val = tp['f_gas']
            m_total = Mb + Md + Mg
            ms = m_total * (1.0 - f_gas_val)

            # Build grid to R_env
            grid_max = pure_renv * 1.05
            num_pts = 3000
            radii = [
                (grid_max / num_pts) * (i + 1) for i in range(num_pts)
            ]
            enc = [
                _model_enc(r, Mb, ab, Md, Rd, Mg, Rg) for r in radii
            ]
            v_base = [
                _gfd_vel(r, Mb, ab, Md, Rd, Mg, Rg, a0_eff)
                for r in radii
            ]
            stage = _build_sigma_stage(
                sigma, pure_renv, ms, f_gas_val, accel)
            result = stage.process(radii, enc)
            v_sigma = result.series

            enc_base = [
                invert_velocity_to_mass(radii[i], v_base[i], a0_eff)
                for i in range(num_pts)
            ]
            enc_sigma = [
                invert_velocity_to_mass(radii[i], v_sigma[i], a0_eff)
                for i in range(num_pts)
            ]
            rat = [
                enc_sigma[i] / enc_base[i] if enc_base[i] > 0 else 1.0
                for i in range(num_pts)
            ]

            def ratio_at(r):
                return GfdSymmetricStage._interp(radii, rat, r)

            # Sample at multiple fractions of the outer field
            # (from throat to horizon)
            outer_span = pure_renv - pure_rt
            samples = {}
            for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                r = pure_rt + frac * outer_span
                samples[frac] = ratio_at(r) - 1.0

            # Derivative at R_env
            dr = radii[1] - radii[0]
            idx = min(int(pure_renv / dr) - 1, num_pts - 2)
            idx = max(1, idx)
            deriv = (rat[idx + 1] - rat[idx - 1]) / (2 * dr)

            results.append((
                nm[:20], pure_rt, pure_renv, sigma,
                samples, deriv
            ))
        except Exception:
            pass

    # Table 1: Mass excess at fractions of outer field, normalized
    print("Mass excess (M_sig/M_base - 1) at fractions of outer field")
    print("Outer field = R_t to R_env")
    print("sigma/f(k) = sigma/21 is the unit of coupling")
    print("=" * 120)

    header = "%20s %+6s" % ("Galaxy", "sigma")
    fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for f in fracs:
        header += " %7s" % ("%.0f%%" % (f * 100))
    print(header)
    print("-" * 120)

    for row in sorted(results, key=lambda r: abs(r[3]), reverse=True):
        nm, rt, renv, sig, samples, deriv = row
        line = "%20s %+5.2f" % (nm, sig)
        for f in fracs:
            line += " %+6.2f%%" % (samples[f] * 100)
        print(line)

    # Table 2: Ratios relative to the 10% mark
    print()
    print()
    print("Ratio: Excess(frac) / Excess(10%)")
    print("If polynomial coupling: expect growth as 1+k+k^2 terms")
    print("=" * 120)

    header = "%20s %+6s" % ("Galaxy", "sigma")
    for f in fracs:
        header += " %7s" % ("%.0f%%" % (f * 100))
    print(header)
    print("-" * 120)

    for row in sorted(results, key=lambda r: abs(r[3]), reverse=True):
        nm, rt, renv, sig, samples, deriv = row
        base = samples[0.1]
        line = "%20s %+5.2f" % (nm, sig)
        for f in fracs:
            if abs(base) > 1e-10:
                r = samples[f] / base
                line += " %7.2f" % r
            else:
                line += " %7s" % "N/A"
        print(line)

    # Table 3: Normalized by sigma / f(k)
    print()
    print()
    print("Excess / (sigma/21) at each fraction")
    print("= how many units of sigma/f(k) the excess represents")
    print("=" * 120)

    header = "%20s %+6s" % ("Galaxy", "sigma")
    for f in fracs:
        header += " %7s" % ("%.0f%%" % (f * 100))
    print(header)
    print("-" * 120)

    for row in sorted(results, key=lambda r: abs(r[3]), reverse=True):
        nm, rt, renv, sig, samples, deriv = row
        sfk = sig / FK
        line = "%20s %+5.2f" % (nm, sig)
        for f in fracs:
            if abs(sfk) > 1e-10:
                n = samples[f] / sfk
                line += " %7.3f" % n
            else:
                line += " %7s" % "N/A"
        print(line)

    # Check: at R_env, is excess = sigma * some_function_of_k?
    print()
    print()
    print("Excess at R_env: what multiple of sigma?")
    print("=" * 70)
    print("%20s %+6s %8s %8s %8s" % (
        "Galaxy", "sigma", "Exc@env", "Exc/sig", "Exc*21/s"))
    print("-" * 70)
    for row in sorted(results, key=lambda r: abs(r[3]), reverse=True):
        nm, rt, renv, sig, samples, deriv = row
        ee = samples[1.0]
        es = ee / sig if abs(sig) > 1e-10 else 0
        esfk = ee * FK / sig if abs(sig) > 1e-10 else 0
        print("%20s %+5.2f %+7.2f%% %+7.4f %+7.2f" % (
            nm, sig, ee * 100, es, esfk))


if __name__ == "__main__":
    main()
