"""
Investigate whether filtering high-error observations improves
inference accuracy, and show the error distribution across MW data.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(__file__))

from data.galaxies import PREDICTION_GALAXIES
from physics.services.rotation.inference import optimize_inference

mw = next(g for g in PREDICTION_GALAXIES if g["id"] == "milky_way")
mm = mw["mass_model"]
obs_all = [o for o in mw["observations"] if o.get("r", 0) > 0 and o.get("v", 0) > 0]

PUB_TOTAL = 1.50e10 + 4.57e10 + 1.50e10  # 7.57e10

# --- Part 1: Show every observation with its error ---
print()
print("  Milky Way Observation Error Distribution")
print("  " + "=" * 65)
print()
print("  {:<5s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {}".format(
    "Obs#", "r (kpc)", "v (km/s)", "err", "weight", "Source"))
print("  " + "-" * 65)

errors = [o["err"] for o in obs_all]
for i, o in enumerate(obs_all):
    r, v, err = o["r"], o["v"], o["err"]
    w = 1.0 / (err * err)
    src = ""
    if r <= 25:
        src = "Gaia DR3 (Ou+2023)"
    elif r == 30 or r == 40:
        src = "Halo K giants (Huang+2016)"
    elif r == 50:
        src = "RR Lyrae (Ablimit+Zhao 2017)"
    print("  {:<5d}  {:>8.1f}  {:>8.0f}  {:>8.1f}  {:>10.4f}  {}".format(
        i + 1, r, v, err, w, src))

mean_err = sum(errors) / len(errors)
var_err = sum((e - mean_err) ** 2 for e in errors) / len(errors)
std_err = math.sqrt(var_err)
median_err = sorted(errors)[len(errors) // 2]

print()
print("  Error statistics:")
print("    Mean:   {:.1f} km/s".format(mean_err))
print("    Median: {:.1f} km/s".format(median_err))
print("    Std:    {:.1f} km/s".format(std_err))

# --- Part 2: Run inference with different sigma cutoffs ---
print()
print()
print("  Inference Accuracy vs Error Tolerance Cutoff")
print("  " + "=" * 75)
print()
print("  {:<20s}  {:>6s}  {:>14s}  {:>10s}  {:>10s}".format(
    "Filter", "N_obs", "Total Mass", "Error %", "Comp Err %"))
print("  " + "-" * 75)


def run_filtered(label, obs_list):
    n = len(obs_list)
    if n < 3:
        print("  {:<20s}  {:>6d}  {:>14s}  {:>10s}  {:>10s}".format(
            label, n, "too few", "", ""))
        return

    result = optimize_inference(
        mass_model=mm,
        max_radius=mw["distance"],
        num_points=500,
        observations=obs_list,
        accel_ratio=mw["accel"],
        galactic_radius_kpc=mw["galactic_radius"],
        constrained=True,
    )
    opt = result["mass_model"]
    m_total = (opt.get("bulge", {}).get("M", 0)
               + opt.get("disk", {}).get("M", 0)
               + opt.get("gas", {}).get("M", 0))
    total_err = (m_total - PUB_TOTAL) / PUB_TOTAL * 100.0
    # Closure worst component
    c_b = 0.20 * m_total
    c_d = 0.60 * m_total
    c_g = 0.20 * m_total
    max_comp = max(
        abs(c_b - 1.50e10) / 1.50e10 * 100,
        abs(c_d - 4.57e10) / 4.57e10 * 100,
        abs(c_g - 1.50e10) / 1.50e10 * 100,
    )
    print("  {:<20s}  {:>6d}  {:>14s}  {:>+9.1f}%  {:>9.1f}%".format(
        label, n, "{:.3e}".format(m_total), total_err, max_comp))


# All observations
run_filtered("All (no filter)", obs_all)

# Fixed thresholds
for cutoff in [2, 3, 5, 8, 10, 15]:
    filtered = [o for o in obs_all if o["err"] <= cutoff]
    run_filtered("err <= {} km/s".format(cutoff), filtered)

# Sigma-based thresholds
for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
    threshold = median_err + k * std_err
    filtered = [o for o in obs_all if o["err"] <= threshold]
    label = "med + {:.1f}*std ({:.0f})".format(k, threshold)
    run_filtered(label, filtered)

# IQR-based (outlier detection)
sorted_errs = sorted(errors)
q1 = sorted_errs[len(sorted_errs) // 4]
q3 = sorted_errs[3 * len(sorted_errs) // 4]
iqr = q3 - q1
for k in [1.5, 2.0, 3.0]:
    threshold = q3 + k * iqr
    filtered = [o for o in obs_all if o["err"] <= threshold]
    label = "IQR x{:.1f} ({:.0f})".format(k, threshold)
    run_filtered(label, filtered)

print()
print("  Notes:")
print("  - 'Comp Err %' = worst individual component error using closure partition")
print("  - Published total: 7.57e10 M_sun")
print("  - The optimizer already uses 1/err^2 weighting internally")
print()
