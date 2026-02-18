"""
Convergence test: how many observation points does the GFD inference
pipeline need before the total baryonic mass (and closure decomposition)
reaches a given confidence level?

Milky Way test case, constrained mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.galaxies import PREDICTION_GALAXIES
from physics.services.rotation.inference import optimize_inference

# --- Load Milky Way ---
mw = next(g for g in PREDICTION_GALAXIES if g["id"] == "milky_way")
mm = mw["mass_model"]
obs_all = [o for o in mw["observations"] if o.get("r", 0) > 0 and o.get("v", 0) > 0]
max_obs = len(obs_all)

# Published values
PUB_BULGE = 1.50e10
PUB_DISK = 4.57e10
PUB_GAS = 1.50e10
PUB_TOTAL = PUB_BULGE + PUB_DISK + PUB_GAS  # 7.57e10

# Closure partition (State 3: complete closure, 1:3:1)
F_BULGE = 0.20
F_DISK = 0.60
F_GAS = 0.20

# Header
print()
print("  Milky Way Inference Convergence (Constrained Mode + Field Closure Partition)")
print("  " + "=" * 90)
print()
print("  {:<6s}  {:>14s}  {:>10s}  {:>12s}  {:>12s}  {:>12s}  {:>10s}".format(
    "N_obs",
    "Total Mass",
    "Error %",
    "Bulge",
    "Disk",
    "Gas",
    "Comp Err %",
))
print("  {:<6s}  {:>14s}  {:>10s}  {:>12s}  {:>12s}  {:>12s}  {:>10s}".format(
    "",
    "(M_sun)",
    "(vs pub)",
    "(closure)",
    "(closure)",
    "(closure)",
    "(max comp)",
))
print("  " + "-" * 90)

for n in range(3, max_obs + 1):
    obs_subset = obs_all[:n]

    result = optimize_inference(
        mass_model=mm,
        max_radius=mw["distance"],
        num_points=500,
        observations=obs_subset,
        accel_ratio=mw["accel"],
        galactic_radius_kpc=mw["galactic_radius"],
        constrained=True,
    )

    opt = result["mass_model"]
    m_bulge = opt.get("bulge", {}).get("M", 0)
    m_disk = opt.get("disk", {}).get("M", 0)
    m_gas = opt.get("gas", {}).get("M", 0)
    m_total = m_bulge + m_disk + m_gas

    total_err = (m_total - PUB_TOTAL) / PUB_TOTAL * 100.0

    # Closure decomposition
    c_bulge = F_BULGE * m_total
    c_disk = F_DISK * m_total
    c_gas = F_GAS * m_total

    # Max component error from closure
    err_b = abs(c_bulge - PUB_BULGE) / PUB_BULGE * 100.0
    err_d = abs(c_disk - PUB_DISK) / PUB_DISK * 100.0
    err_g = abs(c_gas - PUB_GAS) / PUB_GAS * 100.0
    max_err = max(err_b, err_d, err_g)

    print("  {:<6d}  {:>14s}  {:>+9.1f}%  {:>12s}  {:>12s}  {:>12s}  {:>9.1f}%".format(
        n,
        "{:.3e}".format(m_total),
        total_err,
        "{:.3e}".format(c_bulge),
        "{:.3e}".format(c_disk),
        "{:.3e}".format(c_gas),
        max_err,
    ))

print()
print("  Notes:")
print("  - Total Mass: sum of optimizer outputs (Stage 1 Bayesian MAP)")
print("  - Bulge/Disk/Gas: derived from Total Mass via 1:3:1 closure partition (0.20:0.60:0.20)")
print("  - Comp Err %: largest individual component error from closure decomposition")
print("  - Published: Bulge=1.50e10, Disk=4.57e10, Gas=1.50e10, Total=7.57e10 M_sun")
print()
