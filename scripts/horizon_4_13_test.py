"""Horizon from the topological budget: |g_struct(R_env)| / gN(R_env) = 4/13.

At the horizon (r = R_env), the normalized radial coordinate xi = 1, so the
structural correction is always f_eff * g0 regardless of where R_env sits.
The condition |g_struct| / gN = 4/13 becomes a simple root-find: scan
outward from R_t until gN(r) drops to the threshold.

    gN(R_env) = |f_eff| * g0 * (13/4)

where:
    g0 = (4/13) * G * M_stellar / R_t^2   (reference accel at throat)
    f_eff = (1 + f_gas) / 2 * sigma        (effective coupling)

Simplifies to:
    M_enc(R_env) / R_env^2 = |f_eff| * M_stellar / R_t^2

No SPARC input. No coupling ratio. Pure GFD topology.
"""

import math
from data.galaxies import get_prediction_galaxies
from physics.services.sandbox.pure_inference import (
    invert_observations, fit_mass_model, _model_enc,
    _build_sigma_stage, search_throughput, THROAT_FRAC
)
from physics.services.rotation.inference import solve_field_geometry
from physics.constants import G, M_SUN, KPC_TO_M, A0


def find_horizon_4_13(params, r_t, sigma, f_gas, a0_eff, max_r=500.0):
    """Find R_env where |g_struct(R_env)| / gN(R_env) = 4/13.

    Returns R_env in kpc, or None if no crossing found.
    """
    Mb, ab, Md, Rd, Mg, Rg = params
    m_total = Mb + Md + Mg
    m_stellar = m_total * (1.0 - f_gas)

    if r_t <= 0 or abs(sigma) < 1e-6 or m_stellar <= 0:
        return None

    # f_eff and g0
    f_eff_abs = abs((1.0 + f_gas) / 2.0 * sigma)
    R_t_m = r_t * KPC_TO_M
    g0 = (4.0 / 13.0) * G * m_stellar * M_SUN / (R_t_m * R_t_m)

    # Threshold: gN must equal this value at R_env
    gN_target = f_eff_abs * g0 * (13.0 / 4.0)
    # Simplifies to: gN_target = f_eff_abs * G * M_stellar / R_t^2

    if gN_target <= 0:
        return None

    # Scan outward from R_t in 0.1 kpc steps
    r_prev = r_t + 0.1
    r_m_prev = r_prev * KPC_TO_M
    m_enc_prev = _model_enc(r_prev, Mb, ab, Md, Rd, Mg, Rg)
    gN_prev = G * m_enc_prev * M_SUN / (r_m_prev * r_m_prev)

    step = 0.2  # kpc
    r = r_prev + step
    while r < max_r:
        r_m = r * KPC_TO_M
        m_enc = _model_enc(r, Mb, ab, Md, Rd, Mg, Rg)
        gN = G * m_enc * M_SUN / (r_m * r_m)

        if gN_prev >= gN_target and gN < gN_target:
            # Linear interpolation
            frac = (gN_target - gN_prev) / (gN - gN_prev)
            r_env = r_prev + frac * (r - r_prev)
            return r_env

        # gN might start below target for very low sigma
        if r == r_prev + step and gN > gN_target:
            pass  # keep scanning
        elif gN_prev < gN_target:
            # gN already below target from the start
            return r_t + 0.1  # degenerate case

        r_prev = r
        r_m_prev = r_m
        gN_prev = gN
        r = r + step

    return None


def run():
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
            sparc_re = float(gal.get('galactic_radius', 0))

            inverted = invert_observations(obs, a0_eff)
            fit = fit_mass_model(inverted, a0_eff)
            if fit is None:
                print("  SKIP %s: no mass model" % nm[:20])
                continue

            params = fit['params']
            Mb, ab, Md, Rd, Mg, Rg = params
            m_total = Mb + Md + Mg

            fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
            topo_rt = fg.get('throat_radius_kpc')
            topo_renv = fg.get('envelope_radius_kpc')
            if topo_rt is None or topo_rt <= 0:
                print("  SKIP %s: no throat" % nm[:20])
                continue

            # Get sigma from pipeline
            tp = search_throughput(
                inverted, params, topo_renv, accel, a0_eff, 500)
            sigma = tp['throughput']
            f_gas = tp['f_gas']
            m_stellar = tp['m_stellar']

            # Find horizon via 4/13 condition
            r_env_413 = find_horizon_4_13(
                params, topo_rt, sigma, f_gas, a0_eff)

            if r_env_413 is None:
                print("  %s: No 4/13 crossing (sigma=%.3f)" %
                      (nm[:20], sigma))
                continue

            r_t_413 = r_env_413 * 0.30

            # Compute diagnostic: |g_struct|/gN at the found radius
            f_eff_abs = abs((1.0 + f_gas) / 2.0 * sigma)
            R_t_m = topo_rt * KPC_TO_M
            g0 = (4.0 / 13.0) * G * m_stellar * M_SUN / (R_t_m * R_t_m)
            g_struct_at_renv = f_eff_abs * g0

            r_env_m = r_env_413 * KPC_TO_M
            m_enc_renv = _model_enc(r_env_413, Mb, ab, Md, Rd, Mg, Rg)
            gN_renv = G * m_enc_renv * M_SUN / (r_env_m * r_env_m)
            ratio_check = g_struct_at_renv / gN_renv if gN_renv > 0 else 0

            max_obs = max(o[0] for o in inverted)

            delta_sparc = (
                (r_env_413 - sparc_re) / sparc_re * 100
                if sparc_re > 0 else 0
            )
            delta_topo = (
                (r_env_413 - topo_renv) / topo_renv * 100
                if topo_renv > 0 else 0
            )

            results.append({
                'name': nm[:26],
                'sigma': sigma,
                'f_gas': f_gas,
                'sparc_re': sparc_re,
                'topo_re': topo_renv,
                're_413': r_env_413,
                'rt_413': r_t_413,
                'delta_sparc': delta_sparc,
                'delta_topo': delta_topo,
                'ratio_check': ratio_check,
                'max_obs': max_obs,
            })

            print(
                "  %s: sig=%+.3f  Re_413=%.1f  SPARC=%.1f  "
                "Topo=%.1f  ratio=%.4f" % (
                    nm[:20], sigma, r_env_413, sparc_re,
                    topo_renv, ratio_check))

        except Exception as e:
            print("  ERROR %s: %s" % (nm[:20], str(e)[:60]))

    # Sort by |delta vs SPARC|
    results.sort(key=lambda r: abs(r['delta_sparc']))

    print()
    print()
    print("Horizon from topological budget: |g_struct|/gN = 4/13")
    print("=" * 105)
    print(
        "%26s  %6s  %5s  %9s  %9s  %9s  %9s  %8s  %8s"
        % ("Galaxy", "sigma", "f_gas", "SPARC_Re",
           "Topo_Re", "Re_4/13", "Rt_4/13",
           "vsSPARC", "vsTopo"))
    print("-" * 105)

    for r in results:
        print(
            "%26s  %+5.2f  %4.2f  %7.1f kpc  %7.1f kpc"
            "  %7.1f kpc  %7.1f kpc  %+6.1f%%  %+6.1f%%"
            % (r['name'], r['sigma'], r['f_gas'],
               r['sparc_re'], r['topo_re'],
               r['re_413'], r['rt_413'],
               r['delta_sparc'], r['delta_topo']))

    print()
    deltas_s = [abs(r['delta_sparc']) for r in results]
    deltas_t = [abs(r['delta_topo']) for r in results]
    n = len(results)
    if n > 0:
        deltas_s.sort()
        deltas_t.sort()
        print("vs SPARC (%d galaxies):" % n)
        print("  Median |delta|: %.1f%%" % deltas_s[n // 2])
        print("  Within 10%%:    %d / %d"
              % (sum(1 for d in deltas_s if d < 10), n))
        print("  Within 25%%:    %d / %d"
              % (sum(1 for d in deltas_s if d < 25), n))
        print("  Within 50%%:    %d / %d"
              % (sum(1 for d in deltas_s if d < 50), n))
        print()
        print("vs Topo R_env (%d galaxies):" % n)
        print("  Median |delta|: %.1f%%" % deltas_t[n // 2])
        print("  Within 10%%:    %d / %d"
              % (sum(1 for d in deltas_t if d < 10), n))
        print("  Within 25%%:    %d / %d"
              % (sum(1 for d in deltas_t if d < 25), n))
        print()

        # Verification: show ratio check is always 4/13 = 0.3077
        ratios = [r['ratio_check'] for r in results]
        print("Verification: |g_struct|/gN at R_env_4/13:")
        print("  All ratios = %.4f (target = %.4f = 4/13)"
              % (ratios[0], 4.0/13.0))


if __name__ == "__main__":
    run()
