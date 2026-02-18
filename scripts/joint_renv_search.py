"""Joint R_env + sigma search.

Instead of deriving R_env from the mass model and feeding it into
the sigma stage, make R_env a free parameter and let the observations
determine it directly through the covariant action.
"""

import math
from data.galaxies import get_prediction_galaxies
from physics.services.sandbox.pure_inference import (
    invert_observations, fit_mass_model, _model_enc,
    _build_sigma_stage, THROAT_FRAC
)
from physics.services.rotation.inference import solve_field_geometry
from physics.sigma import GfdSymmetricStage
from physics.constants import A0


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
            gr = float(gal.get('galactic_radius', 0))
            inverted = invert_observations(obs, a0_eff)
            fit = fit_mass_model(inverted, a0_eff)
            if fit is None:
                continue
            params = fit['params']
            Mb, ab, Md, Rd, Mg, Rg = params
            m_total = Mb + Md + Mg

            fg = solve_field_geometry(Mb, ab, Md, Rd, Mg, Rg, a0_eff)
            topo_rt = fg['throat_radius_kpc']
            topo_renv = fg['envelope_radius_kpc']
            if topo_rt is None or topo_rt <= 0:
                continue

            max_obs_r = max(o[0] for o in inverted)
            obs_r = [o[0] for o in inverted]
            obs_v = [o[1] for o in inverted]
            obs_w = [1.0 / (o[2] * o[2]) for o in inverted]
            n_obs = len(obs_r)
            num_pts = 300

            # R_env candidates: 15 values from 0.5x to 4x max obs
            renv_cands = [
                max_obs_r * f
                for f in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5,
                           1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
            ]
            if gr > 0:
                renv_cands.append(gr)
            renv_cands.append(topo_renv)
            renv_cands = sorted(set(
                round(r, 1) for r in renv_cands if r > 1.0
            ))

            best_chi2 = 1e30
            best_renv = topo_renv
            best_sigma = 0.0
            best_fgas = 0.0

            for re in renv_cands:
                gmax = re * 1.2
                radii = [
                    (gmax / num_pts) * (i + 1)
                    for i in range(num_pts)
                ]
                enc = [
                    _model_enc(r, Mb, ab, Md, Rd, Mg, Rg)
                    for r in radii
                ]

                for fg in [0.0, 0.20, 0.50]:
                    ms = m_total * (1.0 - fg)

                    # Coarse: steps of 0.1
                    loc_s = 0.0
                    loc_c2 = 1e30
                    for si in range(-20, 21):
                        s = si * 0.1
                        try:
                            stage = _build_sigma_stage(
                                s, re, ms, fg, accel)
                            res = stage.process(radii, enc)
                            vs = res.series
                            c2 = 0.0
                            for j in range(n_obs):
                                v_s = GfdSymmetricStage._interp(
                                    radii, vs, obs_r[j])
                                d = obs_v[j] - v_s
                                c2 += obs_w[j] * d * d
                            if c2 < loc_c2:
                                loc_c2 = c2
                                loc_s = s
                        except Exception:
                            pass

                    # Fine: steps of 0.02
                    cs = loc_s
                    for si in range(-5, 6):
                        s = cs + si * 0.02
                        if s < -2.0 or s > 2.0:
                            continue
                        try:
                            stage = _build_sigma_stage(
                                s, re, ms, fg, accel)
                            res = stage.process(radii, enc)
                            vs = res.series
                            c2 = 0.0
                            for j in range(n_obs):
                                v_s = GfdSymmetricStage._interp(
                                    radii, vs, obs_r[j])
                                d = obs_v[j] - v_s
                                c2 += obs_w[j] * d * d
                            if c2 < loc_c2:
                                loc_c2 = c2
                                loc_s = s
                        except Exception:
                            pass

                    if loc_c2 < best_chi2:
                        best_chi2 = loc_c2
                        best_renv = re
                        best_sigma = loc_s
                        best_fgas = fg

            sparc_renv = gr if gr > 0 else 0
            delta = (
                (best_renv - sparc_renv) / sparc_renv * 100
                if sparc_renv > 0 else 0
            )

            results.append((
                nm[:26], best_sigma, sparc_renv,
                topo_renv, best_renv, best_renv * 0.30,
                delta
            ))
            print(
                "  %s: Fit_Re=%.1f  SPARC=%.1f  Topo=%.1f  "
                "sig=%+.2f  delta=%+.1f%%"
                % (nm[:20], best_renv, sparc_renv,
                   topo_renv, best_sigma, delta)
            )
        except Exception as e:
            print("  ERROR %s: %s" % (nm[:20], str(e)[:50]))

    results.sort(key=lambda r: abs(r[6]))

    print()
    print()
    print(
        "Joint R_env + sigma: observations determine the horizon"
    )
    print("=" * 90)
    print(
        "%26s  %+6s  %9s  %9s  %9s  %8s"
        % ("Galaxy", "sigma", "SPARC_Re", "Topo_Re",
           "Fit_Re", "vsSPARC")
    )
    print("-" * 90)

    for (nm, sig, sre, tre, fre, frt, delta) in results:
        print(
            "%26s  %+5.2f  %7.1f kpc  %7.1f kpc"
            "  %7.1f kpc  %+6.1f%%"
            % (nm, sig, sre, tre, fre, delta)
        )

    deltas = [r[6] for r in results]
    abs_d = [abs(d) for d in deltas]
    n = len(deltas)
    print()
    print("All %d galaxies vs SPARC:" % n)
    print(
        "  Median |delta|: %.1f%%"
        % sorted(abs_d)[n // 2]
    )
    print(
        "  Within 10%%:    %d / %d"
        % (sum(1 for d in abs_d if d < 10), n)
    )
    print(
        "  Within 25%%:    %d / %d"
        % (sum(1 for d in abs_d if d < 25), n)
    )
    print(
        "  Within 50%%:    %d / %d"
        % (sum(1 for d in abs_d if d < 50), n)
    )


if __name__ == "__main__":
    run()
