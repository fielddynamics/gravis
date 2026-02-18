"""Validate the sigma-squared displacement law across all galaxies.

The topological throat (Rt_bary) comes from the smooth parametric
model. The observational throat (Rt_obs) comes from raw v -> M_enc
inversion. The sigma-squared law predicts:

    Rt_obs = Rt_bary * (1 + sigma^2)

The cleanliness index = actual_shift / sigma^2 measures how well
the smooth GFD model captures the real galactic structure.
  1.0 = ideal exponential disk (M33)
  deviations = bars, warps, projection effects, HI holes
"""

import math

from data.galaxies import get_prediction_galaxies
from physics.services.sandbox.pure_inference import (
    invert_observations, fit_mass_model,
    search_throughput, compute_yN_profile, find_throat_direct,
)
from physics.services.rotation.inference import solve_field_geometry
from physics.constants import A0


def get_state(nm):
    nm_up = nm.upper()
    if any(x in nm_up for x in ['MILKY', 'M31', 'ANDROMEDA']):
        return 'S3'
    if '2841' in nm_up or '891' in nm_up:
        return 'S3'
    if any(x in nm_up for x in ['DDO', 'IC 2574', '3109', '5585',
                                 'UGC 128']):
        return 'S1'
    return 'S2'


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
            sig_sq = sigma * sigma

            yN_profile = compute_yN_profile(inverted, a0_eff)
            obs_geo = find_throat_direct(yN_profile)
            obs_rt = obs_geo.get('throat_radius_kpc')

            state = get_state(nm)

            if obs_rt and obs_rt > 0 and abs(sig_sq) > 1e-8:
                actual_shift = (obs_rt - pure_rt) / pure_rt
                predicted_shift = sig_sq
                rt_pred = pure_rt * (1.0 + sig_sq)
                cleanliness = actual_shift / sig_sq
                pred_err_kpc = rt_pred - obs_rt

                results.append((
                    state, nm[:22], pure_rt, obs_rt, rt_pred,
                    sigma, sig_sq, actual_shift, predicted_shift,
                    cleanliness, pred_err_kpc
                ))
        except Exception as e:
            print("ERROR %s: %s" % (nm[:22], str(e)[:60]))

    results.sort(key=lambda x: (x[0], abs(x[9] - 1.0)))

    print("Sigma-squared displacement law: Rt_obs = Rt_bary * (1 + sig^2)")
    print("=" * 125)
    print(
        "%3s %22s %7s %7s %7s %7s %7s %7s %7s %10s %8s"
        % ("St", "Galaxy", "Rt_bry", "Rt_obs", "Rt_prd",
           "sigma", "sig^2", "Act_%", "Prd_%", "Clean_Idx", "Err_kpc")
    )
    print("-" * 125)

    for st_label in ["S3", "S2", "S1"]:
        group = [r for r in results if r[0] == st_label]
        if not group:
            continue
        label = {
            "S3": "COMPLETE CLOSURE (1:3:1)",
            "S2": "INCOMPLETE CLOSURE",
            "S1": "PRE-CLOSURE",
        }[st_label]
        print("--- %s ---" % label)
        for row in group:
            (state, nm, prt, ort, rpred, sig, s2,
             act_s, prd_s, ci, err) = row
            print(
                "%3s %22s %7.2f %7.2f %7.2f %+6.2f %6.4f"
                " %+6.1f%% %+6.1f%% %+9.4f %+7.3f"
                % (state, nm, prt, ort, rpred, sig, s2,
                   act_s * 100, prd_s * 100, ci, err)
            )
        cis = [r[9] for r in group]
        abs_dev = [abs(c - 1.0) for c in cis]
        print(
            "    --> cleanliness: mean=%+.4f  "
            "closest to 1.0: %s (%.4f)"
            % (
                sum(cis) / len(cis),
                min(group, key=lambda r: abs(r[9] - 1.0))[1].strip(),
                min(cis, key=lambda c: abs(c - 1.0)),
            )
        )
        print()

    # Summary table sorted by cleanliness (closest to 1.0 first)
    print()
    print("MORPHOLOGICAL CLEANLINESS RANKING")
    print("(sorted by |cleanliness_index - 1.0|, best first)")
    print("=" * 90)
    print(
        "%22s %3s %7s %+8s %+8s %10s %s"
        % ("Galaxy", "St", "sigma", "Act_%", "Prd_%", "Clean_Idx",
           "Interpretation")
    )
    print("-" * 90)

    ranked = sorted(results, key=lambda r: abs(r[9] - 1.0))
    for row in ranked:
        (state, nm, prt, ort, rpred, sig, s2,
         act_s, prd_s, ci, err) = row
        dev = abs(ci - 1.0)
        if dev < 0.1:
            interp = "Textbook exponential disk"
        elif dev < 0.5:
            interp = "Minor morphological features"
        elif dev < 1.0:
            interp = "Significant structure"
        elif ci < 0:
            interp = "Sign reversal (complex morphology)"
        else:
            interp = "Strong morphological contamination"
        print(
            "%22s %3s %+6.2f %+7.1f%% %+7.1f%% %+9.4f  %s"
            % (nm, state, sig, act_s * 100, prd_s * 100, ci, interp)
        )


if __name__ == "__main__":
    main()
