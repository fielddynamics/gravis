"""
Zero-Parameter DTG: Covariant Completion + Recursive Structure.

g(r) = g_DTG(r) + g_struct(r)

  g_DTG  = (1/2)[g_N + sqrt(g_N^2 + 4*g_N*a0)]
  g_struct = (4/13) * G*M_star/R_t^2 * [(r - R_t)/(R_env - R_t)]^(d/k)
           = 0  for r <= R_t

  a0 = k^2 * G * m_e / r_e^2
  R_t = 0.30 * R_env
  d/k = 3/4  (spatial dims / tetrahedral faces)
  M_star = M_bulge + M_disk (stars have field origins, gas does not)
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# Constants
# =====================================================================
G_SI = 6.67430e-11
M_SUN = 1.98892e30
KPC_TO_M = 3.0857e19
M_E = 9.1093837139e-31
R_E = 2.8179403205e-15
K = 4
D = 3
A0 = K * K * G_SI * M_E / (R_E * R_E)

THROAT_FRAC = 0.30                     # R_t = 0.30 * R_env
STRUCT_FRAC = 4.0 / 13.0              # 0.3077 amplitude factor
P_STRUCT = float(D) / float(K)        # d/k = 3/4 = 0.75


# =====================================================================
# Four test galaxies matching user's validated mass models
# =====================================================================
GALAXIES = {
    "M33": {
        "name": "M33",
        "subtitle": "7.1x10$^9$ M$_\\odot$, 45% gas",
        "mass_model": {
            "bulge": {"M": 0.4e9, "a": 0.18},
            "disk":  {"M": 3.5e9, "Rd": 1.6},
            "gas":   {"M": 3.2e9, "Rd": 4.0},
        },
        "R_env": 20.0,
        "obs": [
            (1, 45, 10), (2, 68, 8), (4, 100, 5), (6, 108, 5),
            (8, 112, 5), (10, 117, 5), (12, 122, 6), (14, 128, 8),
            (16, 130, 10),
        ],
    },
    "Milky Way": {
        "name": "Milky Way",
        "subtitle": "7.5x10$^{10}$ M$_\\odot$, 20% gas",
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 4.5e10, "Rd": 2.5},
            "gas":   {"M": 1.5e10, "Rd": 7.0},
        },
        "R_env": 30.0,
        "obs": [
            (2, 206, 25), (5, 236, 7), (8, 230, 3),
            (10, 228, 5), (15, 221, 7), (20, 213, 10), (25, 200, 15),
        ],
    },
    "NGC 2403": {
        "name": "NGC 2403",
        "subtitle": "13.2x10$^9$ M$_\\odot$, 42% gas",
        "mass_model": {
            "bulge": {"M": 0.2e9, "a": 0.15},
            "disk":  {"M": 7.5e9, "Rd": 3.25},
            "gas":   {"M": 5.5e9, "Rd": 3.0},
        },
        "R_env": 22.0,
        "obs": [
            (1, 65, 10), (2, 100, 8), (5, 130, 5), (8, 134, 3),
            (10, 136, 3), (15, 135, 4), (20, 134, 5),
        ],
    },
    "DDO 154": {
        "name": "DDO 154",
        "subtitle": "4.3x10$^8$ M$_\\odot$, 93% gas",
        "mass_model": {
            "bulge": {"M": 0, "a": 0.1},
            "disk":  {"M": 3.0e7, "Rd": 0.7},
            "gas":   {"M": 4.0e8, "Rd": 2.5},
        },
        "R_env": 8.0,
        "obs": [
            (0.5, 15, 3), (1.0, 24, 2), (1.5, 31, 2), (2.0, 35, 2),
            (2.5, 38, 2), (3.0, 39, 2), (4.0, 42, 2), (5.0, 44, 2),
            (6.0, 45, 3), (7.0, 46, 4), (7.5, 46, 5),
        ],
    },
}


# =====================================================================
# Physics
# =====================================================================
def solve_x(y_N):
    """Solve x^2/(1+x) = y_N for x >= 0."""
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


def g_newtonian(r_kpc, model):
    if r_kpc <= 0:
        return 0.0
    m = enclosed_baryon(r_kpc, model)
    if m <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    return G_SI * m * M_SUN / (r_m * r_m)


def g_dtg(r_kpc, model):
    gN = g_newtonian(r_kpc, model)
    x = solve_x(gN / A0)
    return A0 * x


def v_from_g(r_kpc, g_ms2):
    if r_kpc <= 0 or g_ms2 <= 0:
        return 0.0
    return math.sqrt(g_ms2 * r_kpc * KPC_TO_M) / 1000.0


def stellar_mass(model):
    return model["bulge"]["M"] + model["disk"]["M"]


def total_baryon(model):
    return model["bulge"]["M"] + model["disk"]["M"] + model["gas"]["M"]


def gas_fraction(model):
    M_tot = total_baryon(model)
    return model["gas"]["M"] / M_tot if M_tot > 0 else 0


# =====================================================================
# STRUCTURAL TERM
# =====================================================================
def compute_g0(model, R_env):
    """(4/13) * G * M_stellar / R_t^2"""
    M_star = stellar_mass(model)
    r_t_kpc = THROAT_FRAC * R_env
    r_t_m = r_t_kpc * KPC_TO_M
    if r_t_m <= 0 or M_star <= 0:
        return 0.0
    return STRUCT_FRAC * G_SI * M_star * M_SUN / (r_t_m * r_t_m)


def g_structural(r_kpc, model, R_env):
    r_t = THROAT_FRAC * R_env
    if r_kpc <= r_t or R_env <= r_t:
        return 0.0
    g0 = compute_g0(model, R_env)
    xi = (r_kpc - r_t) / (R_env - r_t)
    return g0 * xi ** P_STRUCT


def v_dtg_struct(r_kpc, model, R_env):
    g_cov = g_dtg(r_kpc, model)
    g_s = g_structural(r_kpc, model, R_env)
    return v_from_g(r_kpc, g_cov + g_s)


def v_dtg_only(r_kpc, model):
    return v_from_g(r_kpc, g_dtg(r_kpc, model))


def v_newton(r_kpc, model):
    return v_from_g(r_kpc, g_newtonian(r_kpc, model))


def rms_err(v_pred, obs):
    n = len(obs)
    if n == 0:
        return float('inf')
    return math.sqrt(sum((v_pred[i] - obs[i][1])**2
                         for i in range(n)) / n)


# =====================================================================
# ANALYSIS
# =====================================================================
print("=" * 80)
print("ZERO-PARAMETER DTG: COVARIANT + RECURSIVE STRUCTURE")
print("=" * 80)
print()
print("g(r) = g_DTG(r) + (4/13)*G*M_star/R_t^2 * "
      "[(r-R_t)/(R_env-R_t)]^(d/k)")
print(f"  d/k = {D}/{K} = {P_STRUCT}")
print(f"  R_t = {THROAT_FRAC} * R_env")
print(f"  a0 = {A0:.4e} m/s^2")
print()

# Summary
print(f"{'Galaxy':>12} {'M_total':>10} {'f_gas':>5} {'M_star':>10} "
      f"{'R_env':>5} {'R_t':>5} {'g0/a0':>8} "
      f"{'RMS_DTG':>8} {'RMS_D+S':>8}")
print("-" * 80)

results = {}
for gid, gal in GALAXIES.items():
    model = gal["mass_model"]
    R_env = gal["R_env"]
    obs = gal["obs"]
    r_t = THROAT_FRAC * R_env
    M_star = stellar_mass(model)
    M_tot = total_baryon(model)
    fg = gas_fraction(model)
    g0 = compute_g0(model, R_env)

    v_d = [v_dtg_only(o[0], model) for o in obs]
    v_ds = [v_dtg_struct(o[0], model, R_env) for o in obs]
    rms_d = rms_err(v_d, obs)
    rms_ds = rms_err(v_ds, obs)

    results[gid] = {"rms_d": rms_d, "rms_ds": rms_ds, "g0": g0}

    print(f"{gid:>12} {M_tot:10.2e} {fg:4.0%} {M_star:10.2e} "
          f"{R_env:5.0f} {r_t:5.1f} {g0/A0:8.5f} "
          f"{rms_d:8.1f} {rms_ds:8.1f}")


# =====================================================================
# Point-by-point
# =====================================================================
for gid, gal in GALAXIES.items():
    model = gal["mass_model"]
    R_env = gal["R_env"]
    r_t = THROAT_FRAC * R_env
    obs = gal["obs"]
    g0 = results[gid]["g0"]

    rms_d = results[gid]["rms_d"]
    rms_ds = results[gid]["rms_ds"]

    print()
    print(f"--- {gal['name']} ({gal['subtitle'].replace('$', '')}) ---")
    print(f"  R_t = {r_t:.1f} kpc, g0/a0 = {g0/A0:.5f}")
    print(f"  RMS: DTG = {rms_d:.1f}, GFD+ = {rms_ds:.1f}")
    print()
    print(f"  {'r':>6} {'v_obs':>6} {'err':>4} {'v_N':>6} {'v_DTG':>6} "
          f"{'v_D+S':>6} {'g_str/g_cov':>11} {'zone':>7}")
    print("  " + "-" * 60)

    for o in obs:
        r, vo, ve = o
        vn = v_newton(r, model)
        vd = v_dtg_only(r, model)
        vds = v_dtg_struct(r, model, R_env)
        gc = g_dtg(r, model)
        gs = g_structural(r, model, R_env)
        pct = 100.0 * gs / gc if gc > 0 else 0
        zone = "THROAT" if r <= r_t else "OUTER"
        print(f"  {r:6.2f} {vo:6.0f} {ve:4.0f} {vn:6.1f} {vd:6.1f} "
              f"{vds:6.1f} {pct:+10.1f}% {zone:>7}")


# =====================================================================
# PLOT: four-panel layout matching user's reference
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    'Zero-Parameter DTG: Covariant Completion + Recursive Structure\n'
    '$g = g_{DTG} + \\frac{4}{13}\\frac{GM_\\star}{R_t^2}'
    '\\left[\\frac{r - R_t}{R_{env} - R_t}\\right]^{d/k}$'
    '   $d/k = 3/4$,  $R_t = 0.30\\,R_{env}$',
    fontsize=12, fontweight='bold')

gal_order = ["M33", "Milky Way", "NGC 2403", "DDO 154"]

for idx, gid in enumerate(gal_order):
    gal = GALAXIES[gid]
    model = gal["mass_model"]
    R_env = gal["R_env"]
    r_t = THROAT_FRAC * R_env
    obs = gal["obs"]
    fg = gas_fraction(model)
    M_tot = total_baryon(model)

    row, col = divmod(idx, 2)
    ax = axes[row][col]

    r_max = max(o[0] for o in obs) * 1.2
    r_dense = np.linspace(0.05, r_max, 400)

    v_n_arr = [v_newton(r, model) for r in r_dense]
    v_d_arr = [v_dtg_only(r, model) for r in r_dense]
    v_ds_arr = [v_dtg_struct(r, model, R_env) for r in r_dense]

    obs_r = [o[0] for o in obs]
    obs_v = [o[1] for o in obs]
    obs_e = [o[2] for o in obs]

    rms_d = results[gid]["rms_d"]
    rms_ds = results[gid]["rms_ds"]

    # Plot
    ax.plot(r_dense, v_n_arr, '-', color='#FF6B8A', linewidth=1.2,
            label='Newtonian', alpha=0.7)
    ax.plot(r_dense, v_d_arr, '--', color='#00CED1', linewidth=1.8,
            label=f'DTG covariant (RMS={rms_d:.1f})')
    ax.plot(r_dense, v_ds_arr, '-', color='#00FFFF', linewidth=2.5,
            label=f'GFD+ (RMS={rms_ds:.1f})')
    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt='o', color='#FFD700',
                markersize=5, capsize=3, zorder=10, label='Observed')

    # Throat line
    if r_t < r_max:
        ax.axvline(r_t, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    # Title: galaxy name, total mass, gas fraction
    ax.set_title(f'{gal["name"]}\n{gal["subtitle"]}',
                 fontsize=11)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.2)
    ax.set_ylim(0, None)
    ax.set_xlim(0, r_max)

plt.tight_layout()
plt.savefig('absorb_release.png', dpi=150, bbox_inches='tight')
print()
print("Plot saved: absorb_release.png")


# =====================================================================
# SUMMARY TABLE
# =====================================================================
print()
print("=" * 60)
print(f"{'Galaxy':>12} {'DTG only':>10} {'GFD+':>14} {'Notes':>20}")
print("-" * 60)
for gid in gal_order:
    rms_d = results[gid]["rms_d"]
    rms_ds = results[gid]["rms_ds"]
    fg = gas_fraction(GALAXIES[gid]["mass_model"])
    if abs(rms_ds - rms_d) < 0.5:
        note = "negligible"
    elif rms_ds < rms_d:
        note = "improved"
    else:
        note = "worsened"
    print(f"{gid:>12} {rms_d:10.1f} {rms_ds:14.1f}  {note:>20}")
