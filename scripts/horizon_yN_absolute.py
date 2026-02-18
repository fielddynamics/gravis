"""Horizon from absolute yN condition: yN(R_env) = 36/1365.

The throat is defined by an absolute acceleration: yN(Rt) = 18/65.
The horizon is defined by the same topology at a deeper level:

    yN(R_env) = yN(Rt) * 2/f(k) = (18/65) * (2/21) = 36/1365

This says: the horizon is where the Newtonian acceleration has dropped
to 2/f(k) = 2/21 of its throat value. No sigma, no coupling ratio.
Pure mass model + topology.

Equivalently: gN(R_env) = a0 * 36/1365
"""

import math
from data.galaxies import get_prediction_galaxies
from physics.services.sandbox.pure_inference import (
    invert_observations, fit_mass_model, _model_enc,
    search_throughput, THROAT_FRAC
)
from physics.services.rotation.inference import solve_field_geometry
from physics.constants import G, M_SUN, KPC_TO_M, A0


# Topological constants
FK = 21          # f(k) = 1 + k + k^2 for k=4
YN_THROAT = 18.0 / 65.0                    # 0.27692
YN_HORIZON = 2.0 * 18.0 / (65.0 * FK)     # 36/1365 = 0.02637
RATIO_CHECK = 2.0 / FK                     # 2/21 = 0.09524


def find_horizon_yN(params, a0_eff, max_r=500.0):
    """Find R_env where yN(r) = 36/1365.

    Returns (R_env_kpc, yN_at_renv) or (None, None).
    """
    Mb, ab, Md, Rd, Mg, Rg = params
    gN_target = a0_eff * YN_HORIZON

    step = 0.2  # kpc
    r_prev = 0.5
    r_m_prev = r_prev * KPC_TO_M
    m_prev = _model_enc(r_prev, Mb, ab, Md, Rd, Mg, Rg)
    gN_prev = G * m_prev * M_SUN / (r_m_prev * r_m_prev) if r_m_prev > 0 else 0

    r = r_prev + step
    while r < max_r:
        r_m = r * KPC_TO_M
        m_enc = _model_enc(r, Mb, ab, Md, Rd, Mg, Rg)
        gN = G * m_enc * M_SUN / (r_m * r_m)

        if gN_prev >= gN_target and gN < gN_target:
            frac = (gN_target - gN_prev) / (gN - gN_prev)
            r_env = r_prev + frac * step
            yN_at = gN_prev + frac * (gN - gN_prev)
            return r_env, yN_at / a0_eff

        r_prev = r
        gN_prev = gN
        r += step

    return None, None


def run():
    all_gals = get_prediction_galaxies()
    results = []

    print("yN_throat = 18/65 = %.6f" % YN_THROAT)
    print("yN_horizon = 36/1365 = %.6f" % YN_HORIZON)
    print("Ratio = 2/f(k) = 2/21 = %.6f" % RATIO_CHECK)
    print()

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

            fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
            topo_rt = fg.get('throat_radius_kpc')
            topo_renv = fg.get('envelope_radius_kpc')

            if topo_rt is None or topo_rt <= 0:
                print("  SKIP %s: no throat" % nm[:20])
                continue

            # Get sigma for reference
            tp = search_throughput(
                inverted, params, topo_renv, accel, a0_eff, 500)
            sigma = tp['throughput']

            # Find horizon via yN = 36/1365
            r_env_yN, yN_check = find_horizon_yN(params, a0_eff)

            if r_env_yN is None:
                print("  %s: No yN crossing found" % nm[:20])
                continue

            r_t_yN = r_env_yN * 0.30

            # Verify: gN ratio at throat vs horizon
            r_t_m = topo_rt * KPC_TO_M
            m_t = _model_enc(topo_rt, Mb, ab, Md, Rd, Mg, Rg)
            gN_throat = G * m_t * M_SUN / (r_t_m * r_t_m)
            yN_throat_actual = gN_throat / a0_eff

            r_h_m = r_env_yN * KPC_TO_M
            m_h = _model_enc(r_env_yN, Mb, ab, Md, Rd, Mg, Rg)
            gN_horizon = G * m_h * M_SUN / (r_h_m * r_h_m)
            yN_horizon_actual = gN_horizon / a0_eff

            accel_ratio_actual = (
                yN_horizon_actual / yN_throat_actual
                if yN_throat_actual > 0 else 0
            )

            max_obs = max(o[0] for o in inverted)

            delta_sparc = (
                (r_env_yN - sparc_re) / sparc_re * 100
                if sparc_re > 0 else 0
            )
            delta_topo = (
                (r_env_yN - topo_renv) / topo_renv * 100
                if topo_renv > 0 else 0
            )

            results.append({
                'name': nm[:26],
                'sigma': sigma,
                'sparc_re': sparc_re,
                'topo_rt': topo_rt,
                'topo_re': topo_renv,
                're_yN': r_env_yN,
                'rt_yN': r_t_yN,
                'delta_sparc': delta_sparc,
                'delta_topo': delta_topo,
                'yN_check': yN_check,
                'accel_ratio': accel_ratio_actual,
                'max_obs': max_obs,
                'yN_throat': yN_throat_actual,
            })

            print(
                "  %s: Re_yN=%.1f  SPARC=%.1f  Topo=%.1f  "
                "yN=%.5f  gRatio=%.4f  sig=%+.2f" % (
                    nm[:20], r_env_yN, sparc_re, topo_renv,
                    yN_check, accel_ratio_actual, sigma))

        except Exception as e:
            print("  ERROR %s: %s" % (nm[:20], str(e)[:60]))

    # Sort by |delta vs SPARC|
    results.sort(key=lambda r: abs(r['delta_sparc']))

    print()
    print()
    print(
        "Horizon from absolute yN: yN(R_env) = 36/1365 = %.6f"
        % YN_HORIZON)
    print(
        "Throat condition: yN(R_t) = 18/65 = %.6f" % YN_THROAT)
    print(
        "Acceleration ratio at horizon/throat = 2/f(k) = 2/21 = %.6f"
        % RATIO_CHECK)
    print("=" * 120)
    print(
        "%26s  %6s  %9s  %9s  %9s  %9s  %8s  %8s  %8s"
        % ("Galaxy", "sigma", "SPARC_Re", "Topo_Re",
           "Re_yN", "Rt_yN", "vsSPARC", "vsTopo",
           "gH/gT"))
    print("-" * 120)

    for r in results:
        print(
            "%26s  %+5.2f  %7.1f kpc  %7.1f kpc"
            "  %7.1f kpc  %7.1f kpc  %+6.1f%%  %+6.1f%%"
            "  %.4f"
            % (r['name'], r['sigma'],
               r['sparc_re'], r['topo_re'],
               r['re_yN'], r['rt_yN'],
               r['delta_sparc'], r['delta_topo'],
               r['accel_ratio']))

    print()
    n = len(results)
    if n > 0:
        deltas_s = sorted(abs(r['delta_sparc']) for r in results)
        deltas_t = sorted(abs(r['delta_topo']) for r in results)
        print("vs SPARC (%d galaxies):" % n)
        print("  Median |delta|: %.1f%%" % deltas_s[n // 2])
        print("  Mean   |delta|: %.1f%%"
              % (sum(deltas_s) / n))
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

        # Also show yN at horizon for each to verify
        print()
        print("Verification: yN at R_env_yN for each galaxy:")
        for r in results:
            print("  %26s: yN = %.6f (target %.6f)"
                  % (r['name'], r['yN_check'], YN_HORIZON))


if __name__ == "__main__":
    run()
