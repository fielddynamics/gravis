"""
Diagnostic test: compare two-stage GA optimization against brute-force
grid search to verify the GA is finding the true minimum.

Also tests that the optimization actually changes the throughput from
the theoretical prediction when observations are present.
"""
import sys
import os
import math
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.engine import (
    auto_vortex_strength,
    optimize_two_stage_ga,
    GfdSymmetricStage,
)
from physics.equations import (
    mass_model_eq, aqual_solve_x, G, M_SUN, A0, KPC_TO_M,
)


# =====================================================================
# M33 Galaxy data (from data/galaxies.py)
# =====================================================================
M33_MASS_MODEL = {
    "bulge": {"M": 0.4e9, "a": 0.18},
    "disk":  {"M": 3.5e9, "Rd": 1.6},
    "gas":   {"M": 3.2e9, "Rd": 4.0},
}
M33_GALACTIC_RADIUS = 17.0
M33_MAX_RADIUS = 20.0
M33_NUM_POINTS = 100
M33_ACCEL_RATIO = 1.0
M33_OBSERVATIONS = [
    {"r": 0.5, "v": 30,  "err": 5},
    {"r": 1,   "v": 38,  "err": 8},
    {"r": 1.5, "v": 55,  "err": 7},
    {"r": 2,   "v": 68,  "err": 8},
    {"r": 3,   "v": 88,  "err": 6},
    {"r": 4,   "v": 100, "err": 7},
    {"r": 5,   "v": 105, "err": 5},
    {"r": 6,   "v": 108, "err": 6},
    {"r": 7,   "v": 108, "err": 5},
    {"r": 8,   "v": 108, "err": 6},
    {"r": 9,   "v": 110, "err": 6},
    {"r": 10,  "v": 115, "err": 7},
    {"r": 11,  "v": 117, "err": 6},
    {"r": 12,  "v": 120, "err": 7},
    {"r": 14,  "v": 125, "err": 7},
    {"r": 15,  "v": 128, "err": 8},
    {"r": 17,  "v": 130, "err": 8},
]

# Milky Way data
MW_MASS_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 4.57e10, "Rd": 2.2},
    "gas":   {"M": 1.5e10, "Rd": 7.0},
}
MW_GALACTIC_RADIUS = 60.0
MW_MAX_RADIUS = 30.0
MW_OBSERVATIONS = [
    {"r": 2,  "v": 210, "err": 10},
    {"r": 4,  "v": 225, "err": 8},
    {"r": 6,  "v": 230, "err": 6},
    {"r": 8,  "v": 230, "err": 5},
    {"r": 10, "v": 228, "err": 5},
    {"r": 12, "v": 225, "err": 6},
    {"r": 15, "v": 222, "err": 7},
    {"r": 20, "v": 218, "err": 8},
    {"r": 25, "v": 215, "err": 10},
]


def _build_velocities(mass_model, galactic_radius, max_radius, num_points,
                      accel_ratio, observations, mass_scale, throughput):
    """
    Build predicted velocities at each observation radius for the
    GFD-sigma curve at a given mass_scale and throughput.
    Returns (radii, model_v, obs_list) where obs_list has valid points.
    """
    radii = [(max_radius / num_points) * (i + 1)
             for i in range(num_points)]
    base_enc = [mass_model_eq(r, mass_model)[0] for r in radii]
    scaled_enc = [m * mass_scale for m in base_enc]

    mm = mass_model
    m_stellar = 0.0
    m_total = 0.0
    for comp in ("bulge", "disk"):
        if mm.get(comp) and mm[comp].get("M"):
            m_stellar += mm[comp]["M"]
            m_total += mm[comp]["M"]
    m_gas = 0.0
    if mm.get("gas") and mm["gas"].get("M"):
        m_gas = mm["gas"]["M"]
        m_total += m_gas
    m_stellar_s = m_stellar * mass_scale
    m_gas_s = m_gas * mass_scale
    m_total_s = m_total * mass_scale
    f_gas = m_gas_s / m_total_s if m_total_s > 0 else 0.0

    stage = GfdSymmetricStage(
        name="_test",
        equation_label="",
        parameters={
            "accel_ratio": accel_ratio,
            "galactic_radius_kpc": galactic_radius,
            "m_stellar": m_stellar_s,
            "f_gas": f_gas,
            "vortex_strength": throughput,
        },
    )
    result = stage.process(radii, scaled_enc)
    return radii, result.series, observations


def compute_rms(mass_model, galactic_radius, max_radius, num_points,
                accel_ratio, observations, mass_scale, throughput):
    """Unweighted RMS between GFD-sigma and observations."""
    radii, model_v, _ = _build_velocities(
        mass_model, galactic_radius, max_radius, num_points,
        accel_ratio, observations, mass_scale, throughput)
    ss = 0.0
    n = 0
    for obs in observations:
        r, v = obs["r"], obs["v"]
        if r <= 0 or v <= 0:
            continue
        v_pred = GfdSymmetricStage._interp(radii, model_v, r)
        ss += (v - v_pred) ** 2
        n += 1
    return math.sqrt(ss / n) if n > 0 else 999.0


def compute_chi2(mass_model, galactic_radius, max_radius, num_points,
                 accel_ratio, observations, mass_scale, throughput):
    """Weighted chi-squared (same metric the GA uses)."""
    radii, model_v, _ = _build_velocities(
        mass_model, galactic_radius, max_radius, num_points,
        accel_ratio, observations, mass_scale, throughput)
    chi2 = 0.0
    for obs in observations:
        r, v, err = obs["r"], obs["v"], obs.get("err", 1.0)
        if r <= 0 or v <= 0 or err <= 0:
            continue
        v_pred = GfdSymmetricStage._interp(radii, model_v, r)
        chi2 += ((v - v_pred) / err) ** 2
    return chi2


def grid_search(mass_model, galactic_radius, max_radius, num_points,
                accel_ratio, observations):
    """
    Sequential brute-force grid search matching the two-stage optimizer:
    1. Find best mass_scale for GFD base curve (throughput=0, i.e. no sigma)
    2. With that scale locked, find best throughput for GFD-sigma.
    """
    # Stage 1: optimize mass_scale for GFD base (throughput=0 means
    # GFD-sigma collapses to GFD base since structural term vanishes)
    best_chi2_gfd = 1e30
    best_scale = 1.0

    # Coarse
    for scale_i in range(6, 61):
        scale = scale_i * 0.05
        chi2 = compute_chi2(
            mass_model, galactic_radius, max_radius, num_points,
            accel_ratio, observations, scale, 0.0,
        )
        if chi2 < best_chi2_gfd:
            best_chi2_gfd = chi2
            best_scale = scale

    # Fine
    coarse_scale = best_scale
    for si in range(-10, 11):
        scale = coarse_scale + si * 0.005
        if scale < 0.3 or scale > 3.0:
            continue
        chi2 = compute_chi2(
            mass_model, galactic_radius, max_radius, num_points,
            accel_ratio, observations, scale, 0.0,
        )
        if chi2 < best_chi2_gfd:
            best_chi2_gfd = chi2
            best_scale = scale

    # Stage 2: optimize throughput at locked scale
    best_chi2 = 1e30
    best_throughput = 0.0

    # Coarse
    for tp_i in range(-30, 31):
        tp = tp_i * 0.1
        chi2 = compute_chi2(
            mass_model, galactic_radius, max_radius, num_points,
            accel_ratio, observations, best_scale, tp,
        )
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_throughput = tp

    # Fine
    coarse_tp = best_throughput
    for ti in range(-10, 11):
        tp = coarse_tp + ti * 0.01
        if tp < -3.0 or tp > 3.0:
            continue
        chi2 = compute_chi2(
            mass_model, galactic_radius, max_radius, num_points,
            accel_ratio, observations, best_scale, tp,
        )
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_throughput = tp

    best_rms = compute_rms(
        mass_model, galactic_radius, max_radius, num_points,
        accel_ratio, observations, best_scale, best_throughput,
    )
    return best_scale, best_throughput, best_rms, best_chi2


def test_galaxy(name, mass_model, galactic_radius, max_radius,
                num_points, accel_ratio, observations):
    """Run full diagnostic for one galaxy."""
    print("=" * 60)
    print("GALAXY: %s" % name)
    print("=" * 60)

    # 1. Theoretical prediction (no observations)
    theo_ot = auto_vortex_strength(mass_model, galactic_radius)
    theo_rms = compute_rms(
        mass_model, galactic_radius, max_radius, num_points,
        accel_ratio, observations, 1.0, theo_ot,
    )
    print("\nTheoretical (gas leverage):")
    print("  throughput = %.2f" % theo_ot)
    print("  mass_scale = 1.00 (unchanged)")
    print("  RMS        = %.2f km/s" % theo_rms)

    # 2. GA optimization
    t0 = time.time()
    ga = optimize_two_stage_ga(
        mass_model, max_radius, num_points,
        observations, accel_ratio, galactic_radius,
    )
    ga_time = time.time() - t0
    ga_rms = compute_rms(
        mass_model, galactic_radius, max_radius, num_points,
        accel_ratio, observations, ga["mass_scale"], ga["throughput"],
    )
    print("\nTwo-stage GA:")
    print("  throughput  = %.2f" % ga["throughput"])
    print("  mass_scale  = %.4f" % ga["mass_scale"])
    print("  RMS (GA)    = %.2f km/s" % ga["rms"])
    print("  RMS (verify)= %.2f km/s" % ga_rms)
    print("  GFD-only RMS= %.2f km/s" % (ga["gfd_rms"] or 0))
    print("  chi2/dof    = %.2f" % (ga["chi2_dof"] or 0))
    print("  time        = %.3f s" % ga_time)

    # 3. Brute-force grid search (same chi2 metric as GA)
    t0 = time.time()
    grid_scale, grid_tp, grid_rms, grid_chi2 = grid_search(
        mass_model, galactic_radius, max_radius, num_points,
        accel_ratio, observations,
    )
    grid_time = time.time() - t0
    n_obs = sum(1 for o in observations if o["r"] > 0 and o["v"] > 0)
    dof = max(n_obs - 2, 1)
    print("\nBrute-force sequential grid search:")
    print("  throughput  = %.2f" % grid_tp)
    print("  mass_scale  = %.2f" % grid_scale)
    print("  RMS         = %.2f km/s" % grid_rms)
    print("  chi2/dof    = %.2f" % (grid_chi2 / dof))
    print("  time        = %.1f s" % grid_time)

    # 4. Comparison
    print("\n--- COMPARISON ---")
    delta_tp = abs(ga["throughput"] - grid_tp)
    delta_scale = abs(ga["mass_scale"] - grid_scale)
    delta_rms = abs(ga_rms - grid_rms)
    print("  throughput diff: %.2f" % delta_tp)
    print("  scale diff:      %.4f" % delta_scale)
    print("  RMS diff:        %.2f km/s" % delta_rms)

    # Did GA improve over theoretical?
    improvement = theo_rms - ga_rms
    print("  GA improvement over theoretical: %.2f km/s" % improvement)

    ok = True
    if delta_rms > 2.0:
        print("  ** WARNING: GA RMS is %.2f km/s worse than grid **" % delta_rms)
        ok = False
    if improvement < 0:
        print("  ** WARNING: GA is WORSE than theoretical **")
        ok = False
    if delta_tp > 0.5:
        print("  ** WARNING: throughput differs by > 0.5 from grid **")
        ok = False

    if ok:
        print("  PASS")
    else:
        print("  FAIL")

    return ok


if __name__ == "__main__":
    print("Two-stage GA Optimization Diagnostic Test")
    print("=========================================\n")

    results = []

    results.append(test_galaxy(
        "M33 Triangulum",
        M33_MASS_MODEL, M33_GALACTIC_RADIUS, M33_MAX_RADIUS,
        M33_NUM_POINTS, M33_ACCEL_RATIO, M33_OBSERVATIONS,
    ))

    print()

    results.append(test_galaxy(
        "Milky Way",
        MW_MASS_MODEL, MW_GALACTIC_RADIUS, MW_MAX_RADIUS,
        MW_NUM_POINTS := 100, MW_ACCEL_RATIO := 1.0, MW_OBSERVATIONS,
    ))

    print("\n" + "=" * 60)
    print("OVERALL: %d/%d passed" % (sum(results), len(results)))
    print("=" * 60)
