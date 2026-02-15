"""
Field Origin Shielding and Release.

The field origin boundary at the throat acts as a shield:
- Inside the throat: the 30% structural coupling is CONSUMED
  by the field origin's own domain. It maintains the merkaba
  structure. The effective gravitational mass is REDUCED.
- Outside the throat: the consumed coupling is RELEASED and
  amplified. Two tetrahedra, each releasing 30%, giving ~60%
  or some function of it.

Physical picture: the field origin is a gravitational capacitor.
It stores coupling energy inside its domain, releases it outside.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# Constants
# =====================================================================
G = 6.67430e-11
M_SUN = 1.98892e30
KPC_TO_M = 3.0857e19
M_E = 9.1093837139e-31
R_E = 2.8179403205e-15
K = 4
D = 3
A0 = K * K * G * M_E / (R_E * R_E)
ALPHA_THROAT = 4.0 / 13.0
P = float(K) / float(D)  # 4/3


GALAXIES = {
    "M33": {
        "bulge": {"M": 0.1e9, "a": 0.2},
        "disk":  {"M": 2.0e9, "Rd": 1.4},
        "gas":   {"M": 1.86e9, "Rd": 7.0},
        "R_env": 16.0,
        "obs": [
            (1, 45, 10), (2, 68, 8), (4, 100, 5), (6, 108, 5),
            (8, 112, 5), (10, 117, 5), (12, 122, 6), (14, 128, 8),
            (16, 130, 10),
        ],
    },
    "Milky Way": {
        "bulge": {"M": 1.5e10, "a": 0.6},
        "disk":  {"M": 5.0e10, "Rd": 2.5},
        "gas":   {"M": 1.0e10, "Rd": 5.0},
        "R_env": 30.0,
        "obs": [
            (3, 213, 7), (5, 225, 5), (8, 230, 3),
            (10, 232, 5), (14, 232, 8), (20, 220, 15),
        ],
    },
    "DDO 154": {
        "bulge": {"M": 0.0, "a": 0.1},
        "disk":  {"M": 3.0e7, "Rd": 0.8},
        "gas":   {"M": 3.6e8, "Rd": 2.5},
        "R_env": 8.0,
        "obs": [
            (0.5, 15, 3), (1.0, 26, 3), (2.0, 40, 3),
            (3.0, 47, 3), (5.0, 47, 4), (7.0, 46, 5),
        ],
    },
}


def solve_x(y_N):
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def v_from_mass(r_kpc, m_solar):
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    gN = G * m_solar * M_SUN / (r_m * r_m)
    x = solve_x(gN / A0)
    return math.sqrt(A0 * x * r_m) / 1000.0


def enclosed_baryon(r_kpc, model):
    r = r_kpc
    m = 0.0
    b = model["bulge"]
    if b["M"] > 0 and r > 0:
        m += b["M"] * r**2 / (r + b["a"])**2
    d = model["disk"]
    if d["M"] > 0 and r > 0:
        x = r / d["Rd"]
        m += d["M"] * (1.0 - (1.0 + x) * math.exp(-x))
    g = model["gas"]
    if g["M"] > 0 and r > 0:
        x = r / g["Rd"]
        m += g["M"] * (1.0 - (1.0 + x) * math.exp(-x))
    return m


def total_baryon(model):
    return model["bulge"]["M"] + model["disk"]["M"] + model["gas"]["M"]


def chi2(v_pred, obs):
    n = len(obs)
    if n == 0:
        return float('inf')
    return sum((v_pred[i] - obs[i][1])**2 / obs[i][2]**2
               for i in range(n)) / n


# =====================================================================
# SHIELD + RELEASE MODEL
#
# Inside throat (r <= r_t):
#   The field origin's domain CONSUMES a fraction of the coupling.
#   Shield fraction varies with depth into the field origin:
#   shield(r) = sigma * (1 - (r/r_t)^2)
#     Maximum at center (r=0): shield = sigma
#     Zero at throat (r=r_t): shield = 0
#   M_eff = M_baryon * (1 - shield)
#
# Outside throat (r > r_t):
#   The consumed coupling is RELEASED + amplified.
#   M_eff = M_baryon + amp * M_total * ((r/r_t)^{4/3} - 1)
#
# sigma = inner shielding strength
# amp = outer amplification factor
# =====================================================================

def model_shield_release(r_kpc, model, R_env, sigma, amp):
    """Shield + Release model."""
    r_t = ALPHA_THROAT * R_env
    M_b = enclosed_baryon(r_kpc, model)

    if r_kpc <= r_t:
        # Inside throat: shielding
        depth = 1.0 - (r_kpc / r_t) ** 2 if r_t > 0 else 0
        shield = sigma * depth
        M_eff = M_b * (1.0 - shield)
    else:
        # Outside throat: release + amplification
        M_total = total_baryon(model)
        M_struct = amp * M_total * ((r_kpc / r_t) ** P - 1.0)
        M_eff = M_b + M_struct

    return v_from_mass(r_kpc, max(M_eff, 1e-6))


# =====================================================================
# PARAMETER SCAN
# =====================================================================
print("=" * 80)
print("FIELD ORIGIN SHIELDING + RELEASE")
print("=" * 80)
print()
print("sigma = inner shielding fraction (0 = no shielding, 0.30 = full)")
print("amp   = outer amplification factor for structural mass")
print()

# Scan parameters
sigmas = [0.0, 0.10, 0.20, 0.30, 4.0/13.0]
amps = [0.5, 0.75, 1.0, 1.25, 1.5, 8.0/13.0, 2.0]

# For each galaxy, find best (sigma, amp)
for gname, gal in GALAXIES.items():
    model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    obs = gal["obs"]

    print(f"\n{'='*60}")
    print(f"{gname} (throat at r_t = {ALPHA_THROAT * R_env:.2f} kpc)")
    print(f"{'='*60}")

    # Standard GFD baseline
    v_std = [v_from_mass(o[0], enclosed_baryon(o[0], model)) for o in obs]
    c2_std = chi2(v_std, obs)
    print(f"  Standard GFD chi2 = {c2_std:.1f}")
    print()

    print(f"{'sigma':>6} {'amp':>6} {'chi2':>8}", end="")
    for o in obs:
        print(f"  r={o[0]}", end="")
    print()
    print("-" * (22 + 7 * len(obs)))

    best_chi2 = 999
    best_params = (0, 0)
    best_vpred = None

    for sigma in sigmas:
        for amp in amps:
            v_pred = [model_shield_release(o[0], model, R_env, sigma, amp)
                      for o in obs]
            c2 = chi2(v_pred, obs)

            if c2 < best_chi2:
                best_chi2 = c2
                best_params = (sigma, amp)
                best_vpred = v_pred

            # Print a selection
            if sigma in [0.0, 4.0/13.0] or amp in [1.0, 8.0/13.0]:
                print(f"{sigma:6.3f} {amp:6.3f} {c2:8.2f}", end="")
                for v in v_pred:
                    print(f"  {v:5.1f}", end="")
                print()

    print()
    print(f"  Best: sigma={best_params[0]:.3f}, amp={best_params[1]:.3f}, "
          f"chi2={best_chi2:.2f}")
    print(f"  {'Observed:':>28}", end="")
    for o in obs:
        print(f"  {o[1]:5.0f}", end="")
    print()
    print(f"  {'Best pred:':>28}", end="")
    for v in best_vpred:
        print(f"  {v:5.1f}", end="")
    print()


# =====================================================================
# FINE-GRAINED SCAN for topologically motivated values
# =====================================================================
print()
print("=" * 80)
print("TOPOLOGICALLY MOTIVATED PARAMETER PAIRS")
print("=" * 80)
print()
print("Candidates derived from the coupling polynomial:")
print("  4/13 = 0.3077 (structural fraction)")
print("  8/13 = 0.6154 (two tetrahedra releasing)")
print("  4/17 = 0.2353 (structural / total coupling)")
print("  8/17 = 0.4706 (double structural / total)")
print("  4/21 = 0.1905 (structural / full depth)")
print("  1/(1+k) = 1/5 = 0.2000")
print("  k/(k^2+1) = 4/17 = 0.2353")
print()

# Topologically motivated (sigma, amp) pairs
topo_pairs = [
    (0.0,      1.0,    "No shield, 1x release"),
    (4.0/13,   1.0,    "4/13 shield, 1x release"),
    (0.0,      8.0/13, "No shield, 8/13 release"),
    (4.0/13,   8.0/13, "4/13 shield, 8/13 release"),
    (4.0/17,   8.0/13, "4/17 shield, 8/13 release"),
    (4.0/17,   1.0,    "4/17 shield, 1x release"),
    (4.0/13,   4.0/13, "4/13 shield, 4/13 release"),
    (4.0/21,   4.0/13, "4/21 shield, 4/13 release"),
    (0.0,      4.0/13, "No shield, 4/13 release"),
    (1.0/5,    1.0,    "1/5 shield, 1x release"),
    (4.0/13,   17.0/13, "4/13 shield, 17/13 release"),
]

print(f"{'Description':>30} {'sig':>5} {'amp':>5}", end="")
for gname in GALAXIES:
    print(f"  {gname:>8}", end="")
print(f"  {'AVG':>6}")
print("-" * (50 + 10 * len(GALAXIES)))

for sigma, amp, desc in topo_pairs:
    print(f"{desc:>30} {sigma:5.3f} {amp:5.3f}", end="")
    chi2_vals = []
    for gname, gal in GALAXIES.items():
        mdl = {k: gal[k] for k in ["bulge", "disk", "gas"]}
        R_env = gal["R_env"]
        obs = gal["obs"]
        v_pred = [model_shield_release(o[0], mdl, R_env, sigma, amp)
                  for o in obs]
        c2 = chi2(v_pred, obs)
        chi2_vals.append(c2)
        print(f"  {c2:8.1f}", end="")
    avg = sum(chi2_vals) / len(chi2_vals)
    print(f"  {avg:6.1f}")


# =====================================================================
# BRUTE FORCE: find global best across all galaxies
# =====================================================================
print()
print("=" * 80)
print("OPTIMAL PARAMETERS (minimizing mean chi2 across all galaxies)")
print("=" * 80)
print()

best_overall = 999
best_s = 0
best_a = 0

for sigma_100 in range(0, 51, 1):      # 0.00 to 0.50
    sigma = sigma_100 / 100.0
    for amp_100 in range(10, 251, 1):    # 0.10 to 2.50
        amp = amp_100 / 100.0
        total_chi2 = 0
        for gname, gal in GALAXIES.items():
            mdl = {k: gal[k] for k in ["bulge", "disk", "gas"]}
            R_env = gal["R_env"]
            obs = gal["obs"]
            v_pred = [model_shield_release(o[0], mdl, R_env, sigma, amp)
                      for o in obs]
            total_chi2 += chi2(v_pred, obs)
        avg_chi2 = total_chi2 / 3.0

        if avg_chi2 < best_overall:
            best_overall = avg_chi2
            best_s = sigma
            best_a = amp

print(f"Best overall: sigma = {best_s:.3f}, amp = {best_a:.3f}")
print(f"Mean chi2 = {best_overall:.2f}")
print()

# Detail for the best
print(f"{'Galaxy':>12} {'chi2':>8}")
for gname, gal in GALAXIES.items():
    mdl = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    obs = gal["obs"]
    v_pred = [model_shield_release(o[0], mdl, R_env, best_s, best_a)
              for o in obs]
    c2 = chi2(v_pred, obs)
    print(f"{gname:>12} {c2:8.2f}")

    r_t = ALPHA_THROAT * R_env
    print(f"  {'r':>5} {'obs':>5} {'err':>4} {'pred':>6} {'res':>6} {'zone':>8}")
    for i, o in enumerate(obs):
        zone = "THROAT" if o[0] <= r_t else "OUTER"
        res = v_pred[i] - o[1]
        print(f"  {o[0]:5.1f} {o[1]:5.0f} {o[2]:4.0f} {v_pred[i]:6.1f} "
              f"{res:+6.1f} {zone:>8}")
    print()


# =====================================================================
# Check: what topological value is closest to the optimal?
# =====================================================================
print("=" * 80)
print("NEAREST TOPOLOGICAL VALUES")
print("=" * 80)
print()
print(f"Optimal sigma = {best_s:.3f}")
topo_s = [(4.0/13, "4/13"), (4.0/17, "4/17"), (4.0/21, "4/21"),
          (1.0/5, "1/5"), (0.2, "0.2"), (0.0, "0")]
for val, name in topo_s:
    print(f"  {name:>6} = {val:.4f}  delta = {abs(best_s - val):.4f}")

print(f"\nOptimal amp = {best_a:.3f}")
topo_a = [(4.0/13, "4/13"), (8.0/13, "8/13"), (1.0, "1"),
          (17.0/13, "17/13"), (4.0/9, "4/9"), (8.0/9, "8/9"),
          (13.0/9, "13/9")]
for val, name in topo_a:
    print(f"  {name:>6} = {val:.4f}  delta = {abs(best_a - val):.4f}")


# =====================================================================
# PLOT for the best parameters
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Shield + Release: sigma={best_s:.2f}, amp={best_a:.2f}',
             fontsize=14, fontweight='bold')

for col, (gname, gal) in enumerate(GALAXIES.items()):
    mdl = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    r_t = ALPHA_THROAT * R_env
    obs = gal["obs"]

    r_dense = np.linspace(0.1, R_env * 1.2, 300)

    v_std = [v_from_mass(r, enclosed_baryon(r, mdl)) for r in r_dense]
    v_sr = [model_shield_release(r, mdl, R_env, best_s, best_a) for r in r_dense]

    obs_r = [o[0] for o in obs]
    obs_v = [o[1] for o in obs]
    obs_e = [o[2] for o in obs]

    c2_std = chi2([v_from_mass(o[0], enclosed_baryon(o[0], mdl))
                   for o in obs], obs)
    c2_sr = chi2([model_shield_release(o[0], mdl, R_env, best_s, best_a)
                  for o in obs], obs)

    ax = axes[col]
    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt='o', color='orange',
                markersize=6, capsize=3, zorder=10, label='Observed')
    ax.plot(r_dense, v_std, '-', color='purple', linewidth=1.5,
            label=f'Standard GFD ({c2_std:.1f})', alpha=0.7)
    ax.plot(r_dense, v_sr, '-', color='cyan', linewidth=2.5,
            label=f'Shield+Release ({c2_sr:.1f})')

    ax.axvline(r_t, color='yellow', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvspan(0, r_t, alpha=0.06, color='yellow')

    ax.set_title(gname, fontsize=12)
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('shield_release.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: shield_release.png")
