"""
DTG + Growing Structural Acceleration

From the user's M33 result:
  g_struct(r) = A * (r - r_t)^{4/5}   for r > r_t
  g_struct(r) = 0                       for r <= r_t
  g_total(r) = g_covariant(r) + g_struct(r)
  v(r) = sqrt(r * g_total(r))

Exponent: 4/5 = k/(k+1) = 4/5 = 0.8
Throat:   r_t = (4/13) * R_env

Test on:
  1. Milky Way (major spiral)
  2. DDO 154 (gas-dominated dwarf)
  3. NGC 3109 (gas-dominated irregular)
  4. IC 2574 (gas-dominated dwarf)
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
P_STRUCT = float(K) / float(K + 1)  # 4/5 = 0.8


# =====================================================================
# Galaxy data from galaxies.py
# =====================================================================
GALAXIES = {
    "milky_way": {
        "name": "Milky Way",
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0},
        },
        "R_env": 30.0,
        "obs": [
            (2, 206, 25), (5, 236, 7), (8, 230, 3),
            (10, 228, 5), (15, 221, 7), (20, 213, 10), (25, 200, 15),
        ],
    },
    "ddo154": {
        "name": "DDO 154 (gas dominated)",
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.95e7, "Rd": 0.40},
            "gas":   {"M": 4.83e8, "Rd": 2.0},
        },
        "R_env": 10.0,
        "obs": [
            (0.5, 15, 3), (1.0, 24, 2), (1.5, 31, 2), (2.0, 35, 2),
            (2.5, 38, 2), (3.0, 39, 2), (4.0, 42, 2), (5.0, 44, 2),
            (6.0, 45, 3), (7.0, 46, 4), (7.5, 46, 5),
        ],
    },
    "ngc3109": {
        "name": "NGC 3109 (gas dominated)",
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 1.0e8, "Rd": 1.5},
            "gas":   {"M": 1.0e9, "Rd": 3.5},
        },
        "R_env": 8.0,
        "obs": [
            (0.26, 6, 2), (0.52, 11, 1), (0.77, 15, 2), (1.03, 19, 2),
            (1.29, 24, 1), (1.55, 30, 2), (1.81, 34, 2), (2.06, 38, 4),
            (2.32, 43, 6), (2.58, 42, 3), (2.84, 47, 4), (3.10, 49, 4),
            (3.35, 51, 4), (3.61, 53, 4), (3.87, 55, 3), (4.13, 55, 3),
            (4.38, 58, 3), (4.64, 59, 2), (4.90, 61, 2), (5.16, 63, 2),
            (5.42, 64, 3), (5.67, 67, 3), (5.93, 66, 3), (6.19, 66, 3),
            (6.45, 67, 3),
        ],
    },
    "ic2574": {
        "name": "IC 2574 (gas dominated)",
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 9.31e7, "Rd": 1.70},
            "gas":   {"M": 1.032e9, "Rd": 4.0},
        },
        "R_env": 13.0,
        "obs": [
            (0.5, 10, 5), (1.0, 17, 4), (2.0, 30, 3), (3.0, 40, 3),
            (4.0, 48, 3), (5.0, 54, 3), (6.0, 59, 3), (7.0, 62, 3),
            (8.0, 65, 3), (9.0, 66, 4), (10.0, 67, 4), (11.0, 66, 5),
        ],
    },
}


# =====================================================================
# Physics
# =====================================================================
def solve_x(y_N):
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


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


def g_covariant(r_kpc, model):
    """Covariant (DTG base) acceleration in m/s^2."""
    if r_kpc <= 0:
        return 0.0
    m = enclosed_baryon(r_kpc, model)
    if m <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    gN = G * m * M_SUN / (r_m * r_m)
    x = solve_x(gN / A0)
    return A0 * x


def v_from_g(r_kpc, g_ms2):
    """Velocity in km/s from acceleration."""
    if r_kpc <= 0 or g_ms2 <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    return math.sqrt(g_ms2 * r_m) / 1000.0


def v_standard(r_kpc, model):
    """Standard DTG velocity."""
    return v_from_g(r_kpc, g_covariant(r_kpc, model))


def rms(v_pred, obs):
    n = len(obs)
    if n == 0:
        return float('inf')
    return math.sqrt(sum((v_pred[i] - obs[i][1])**2 for i in range(n)) / n)


def chi2(v_pred, obs):
    n = len(obs)
    if n == 0:
        return float('inf')
    return sum((v_pred[i] - obs[i][1])**2 / obs[i][2]**2
               for i in range(n)) / n


# =====================================================================
# STRUCTURAL ACCELERATION MODEL
#
# g_struct(r) = A * ((r - r_t) / r_t)^{4/5}   for r > r_t
#             = 0                                 for r <= r_t
#
# The amplitude A must have units of m/s^2.
# Natural scale: A = C * a0, where C is a dimensionless
# topological constant.
#
# We scan C to find the best fit, then identify the
# nearest topological value.
# =====================================================================

def model_dtg_struct(r_kpc, model, R_env, C):
    """DTG + growing structural acceleration."""
    r_t = ALPHA_THROAT * R_env
    g_cov = g_covariant(r_kpc, model)

    if r_kpc <= r_t or r_t <= 0:
        return v_from_g(r_kpc, g_cov)

    # Structural acceleration
    g_s = C * A0 * ((r_kpc - r_t) / r_t) ** P_STRUCT
    g_total = g_cov + g_s
    return v_from_g(r_kpc, g_total)


# =====================================================================
# ANALYSIS
# =====================================================================
print("=" * 80)
print("DTG + GROWING STRUCTURAL ACCELERATION")
print("=" * 80)
print()
print(f"g_struct(r) = C * a0 * ((r - r_t) / r_t)^{{k/(k+1)}}")
print(f"  k/(k+1) = {K}/{K+1} = {P_STRUCT}")
print(f"  r_t = (4/13) * R_env = {ALPHA_THROAT:.4f} * R_env")
print(f"  a0 = {A0:.4e} m/s^2")
print(f"  C = dimensionless amplitude (to be determined)")
print()


# =====================================================================
# Scan C for each galaxy
# =====================================================================
C_values = np.arange(0.001, 0.30, 0.001)

print("=" * 80)
print("AMPLITUDE SCAN")
print("=" * 80)
print()

results = {}
for gid, gal in GALAXIES.items():
    model = gal["mass_model"]
    R_env = gal["R_env"]
    r_t = ALPHA_THROAT * R_env
    obs = gal["obs"]
    name = gal["name"]

    # Standard DTG
    v_std = [v_standard(o[0], model) for o in obs]
    rms_std = rms(v_std, obs)
    chi2_std = chi2(v_std, obs)

    # Scan C
    best_rms = 999
    best_chi2 = 999
    best_C_rms = 0
    best_C_chi2 = 0

    for C in C_values:
        v_pred = [model_dtg_struct(o[0], model, R_env, C) for o in obs]
        r = rms(v_pred, obs)
        c2 = chi2(v_pred, obs)
        if r < best_rms:
            best_rms = r
            best_C_rms = C
        if c2 < best_chi2:
            best_chi2 = c2
            best_C_chi2 = C

    results[gid] = {
        "best_C_rms": best_C_rms,
        "best_rms": best_rms,
        "best_C_chi2": best_C_chi2,
        "best_chi2": best_chi2,
        "rms_std": rms_std,
        "chi2_std": chi2_std,
    }

    print(f"{name}")
    print(f"  R_env = {R_env} kpc, r_t = {r_t:.2f} kpc")
    print(f"  Standard DTG:  RMS = {rms_std:.1f} km/s, chi2 = {chi2_std:.1f}")
    print(f"  Best (RMS):    C = {best_C_rms:.4f}, RMS = {best_rms:.1f}")
    print(f"  Best (chi2):   C = {best_C_chi2:.4f}, chi2 = {best_chi2:.1f}")
    print()


# =====================================================================
# Topological candidates for C
# =====================================================================
print("=" * 80)
print("TOPOLOGICAL CANDIDATES FOR C")
print("=" * 80)
print()

topo_C = [
    (1.0 / 13.0,    "1/13 = 1/(kd+1)"),
    (1.0 / 17.0,    "1/17 = 1/(k^2+1)"),
    (1.0 / 21.0,    "1/21 = 1/f(k)"),
    (4.0 / 169.0,   "4/169 = k/(kd+1)^2"),
    (4.0 / 289.0,   "4/289 = k/(k^2+1)^2"),
    (1.0 / 16.0,    "1/16 = 1/k^2"),
    (1.0 / (K*K+K), "1/20 = 1/(k^2+k)"),
    (4.0 / 63.0,    "4/63 = k/(d*f(k))"),
    (1.0 / 5.0,     "1/5 = 1/(k+1)"),
    (4.0 / 65.0,    "4/65 = 4/(k^2+1 + 48)...no"),
    (1.0 / 9.0,     "1/9 = 1/d^2"),
    (1.0 / 4.0,     "1/4 = 1/k"),
    (0.05,           "0.05"),
    (0.08,           "~1/13"),
]

# Test each topological C on all galaxies
print(f"{'C value':>8} {'description':>22}", end="")
for gid in GALAXIES:
    print(f"  {gid:>10}", end="")
print(f"  {'mean':>6}")
print("-" * (32 + 12 * len(GALAXIES) + 8))

for C_val, desc in sorted(topo_C, key=lambda x: x[0]):
    print(f"{C_val:8.5f} {desc:>22}", end="")
    chi2_vals = []
    for gid, gal in GALAXIES.items():
        model = gal["mass_model"]
        R_env = gal["R_env"]
        obs = gal["obs"]
        v_pred = [model_dtg_struct(o[0], model, R_env, C_val) for o in obs]
        c2 = chi2(v_pred, obs)
        chi2_vals.append(c2)
        print(f"  {c2:10.1f}", end="")
    print(f"  {sum(chi2_vals)/len(chi2_vals):6.1f}")


# =====================================================================
# Detail for each galaxy with best C
# =====================================================================
print()
print("=" * 80)
print("DETAILED RESULTS PER GALAXY")
print("=" * 80)

for gid, gal in GALAXIES.items():
    model = gal["mass_model"]
    R_env = gal["R_env"]
    r_t = ALPHA_THROAT * R_env
    obs = gal["obs"]
    name = gal["name"]
    best_C = results[gid]["best_C_chi2"]

    print(f"\n--- {name} (C = {best_C:.4f}, r_t = {r_t:.2f} kpc) ---")
    print(f"{'r':>6} {'v_obs':>6} {'err':>4} {'v_std':>7} {'v_dtg+s':>7} "
          f"{'g_cov':>10} {'g_struct':>10} {'boost%':>7} {'zone':>8}")
    print("-" * 75)

    for o in obs:
        r, vo, ve = o
        v_s = v_standard(r, model)
        v_ds = model_dtg_struct(r, model, R_env, best_C)
        gc = g_covariant(r, model)
        if r > r_t and r_t > 0:
            gs = best_C * A0 * ((r - r_t) / r_t) ** P_STRUCT
            boost = 100.0 * gs / gc if gc > 0 else 0
            zone = "OUTER"
        else:
            gs = 0
            boost = 0
            zone = "THROAT"
        print(f"{r:6.2f} {vo:6.0f} {ve:4.0f} {v_s:7.1f} {v_ds:7.1f} "
              f"{gc:10.3e} {gs:10.3e} {boost:6.1f}% {zone:>8}")


# =====================================================================
# UNIVERSAL C: what single value works best across all galaxies?
# =====================================================================
print()
print("=" * 80)
print("UNIVERSAL C SEARCH (mean chi2 across all galaxies)")
print("=" * 80)
print()

best_mean = 999
best_C_universal = 0

for C_100 in range(0, 3000):
    C = C_100 / 10000.0
    total_chi2 = 0
    for gid, gal in GALAXIES.items():
        model = gal["mass_model"]
        R_env = gal["R_env"]
        obs = gal["obs"]
        v_pred = [model_dtg_struct(o[0], model, R_env, C) for o in obs]
        total_chi2 += chi2(v_pred, obs)
    mean_chi2 = total_chi2 / len(GALAXIES)
    if mean_chi2 < best_mean:
        best_mean = mean_chi2
        best_C_universal = C

print(f"Best universal C = {best_C_universal:.4f}")
print(f"Mean chi2 = {best_mean:.2f}")
print()

# Report per galaxy with universal C
print(f"{'Galaxy':>20} {'chi2_std':>10} {'chi2_struct':>12} {'RMS_std':>8} {'RMS_struct':>10}")
print("-" * 65)

for gid, gal in GALAXIES.items():
    model = gal["mass_model"]
    R_env = gal["R_env"]
    obs = gal["obs"]
    v_std = [v_standard(o[0], model) for o in obs]
    v_ds = [model_dtg_struct(o[0], model, R_env, best_C_universal) for o in obs]
    c2_s = chi2(v_std, obs)
    c2_d = chi2(v_ds, obs)
    r_s = rms(v_std, obs)
    r_d = rms(v_ds, obs)
    print(f"{gal['name']:>20} {c2_s:10.1f} {c2_d:12.1f} {r_s:8.1f} {r_d:10.1f}")


# Nearest topological values
print(f"\nUniversal C = {best_C_universal:.4f}")
for C_val, desc in sorted(topo_C, key=lambda x: abs(x[0] - best_C_universal)):
    delta = abs(C_val - best_C_universal)
    print(f"  {desc:>25} = {C_val:.5f}  (delta = {delta:.5f})")
    if delta > 0.1:
        break


# =====================================================================
# PLOT
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'DTG + Growing Structural Acceleration  |  g_struct ~ (r-r_t)^{{4/5}}\n'
             f'Universal C = {best_C_universal:.4f}',
             fontsize=13, fontweight='bold')

for idx, (gid, gal) in enumerate(GALAXIES.items()):
    row, col = divmod(idx, 2)
    ax = axes[row][col]

    model = gal["mass_model"]
    R_env = gal["R_env"]
    r_t = ALPHA_THROAT * R_env
    obs = gal["obs"]
    name = gal["name"]

    r_dense = np.linspace(0.1, R_env * 1.1, 300)

    v_std_arr = [v_standard(r, model) for r in r_dense]
    v_struct_arr = [model_dtg_struct(r, model, R_env, best_C_universal)
                    for r in r_dense]
    # Also with best per-galaxy C
    best_C_local = results[gid]["best_C_chi2"]
    v_local_arr = [model_dtg_struct(r, model, R_env, best_C_local)
                   for r in r_dense]

    obs_r = [o[0] for o in obs]
    obs_v = [o[1] for o in obs]
    obs_e = [o[2] for o in obs]

    c2_std = chi2([v_standard(o[0], model) for o in obs], obs)
    c2_uni = chi2([model_dtg_struct(o[0], model, R_env, best_C_universal)
                   for o in obs], obs)
    rms_std = rms([v_standard(o[0], model) for o in obs], obs)
    rms_uni = rms([model_dtg_struct(o[0], model, R_env, best_C_universal)
                   for o in obs], obs)

    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt='o', color='#FFD700',
                markersize=5, capsize=3, zorder=10, label='Observed')
    ax.plot(r_dense, v_std_arr, '--', color='#00CED1', linewidth=1.5,
            label=f'DTG base (RMS={rms_std:.1f})', alpha=0.8)
    ax.plot(r_dense, v_struct_arr, '-', color='#00FFFF', linewidth=2.5,
            label=f'GFD+ (RMS={rms_uni:.1f})')

    if abs(best_C_local - best_C_universal) > 0.005:
        rms_loc = rms([model_dtg_struct(o[0], model, R_env, best_C_local)
                       for o in obs], obs)
        ax.plot(r_dense, v_local_arr, ':', color='#FF69B4', linewidth=2,
                label=f'Best local C={best_C_local:.3f} (RMS={rms_loc:.1f})',
                alpha=0.8)

    # Throat marker
    ax.axvline(r_t, color='yellow', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvspan(0, r_t, alpha=0.04, color='yellow')

    ax.set_title(f'{name}\nchi2: std={c2_std:.1f}, struct={c2_uni:.1f}',
                 fontsize=10)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('structural_accel.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: structural_accel.png")
