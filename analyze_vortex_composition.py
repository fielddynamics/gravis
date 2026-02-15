"""
Systematic search for the correct composition of GFD + Vortex.

Quadrature overshoots high-mass galaxies. Need a composition that:
  1. Enhances M33 outer velocities (where GFD underpredicts)
  2. Does NOT overshoot Milky Way (where GFD already fits)
  3. Uses zero fitted parameters

Candidate compositions to test:
  A. Feed v_vortex^2/r as gN into covariant (vortex as source)
  B. BTFR-style 4th power: v^4 = v_gfd^4 + v_vortex^4
  C. Replace Newtonian input: gN -> max(gN, gN_vortex), then covariant
  D. Smooth crossover: GFD inside throat, vortex outside
  E. Vortex modifies gN before covariant: gN_mod = gN * (1 + v_vtx^2/(v_gfd^2))
  F. Feed vortex as modified Newtonian into covariant solve

No unicode characters (Windows charmap constraint).
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
A0 = K * K * G * M_E / (R_E * R_E)
ALPHA = 0.30

# =====================================================================
# Galaxy Data (same as before, abbreviated for key galaxies)
# =====================================================================
GALAXIES = {
    "M33": {
        "model": {
            "bulge": {"M": 0.1e9, "a": 0.2},
            "disk":  {"M": 2.0e9, "Rd": 1.4},
            "gas":   {"M": 1.86e9, "Rd": 7.0},
        },
        "R_env": 16.0,
        "obs": [
            {"r": 1,  "v": 45,  "err": 10},
            {"r": 2,  "v": 68,  "err": 8},
            {"r": 4,  "v": 100, "err": 5},
            {"r": 6,  "v": 108, "err": 5},
            {"r": 8,  "v": 112, "err": 5},
            {"r": 10, "v": 117, "err": 5},
            {"r": 12, "v": 122, "err": 6},
            {"r": 14, "v": 128, "err": 8},
            {"r": 16, "v": 130, "err": 10},
        ],
    },
    "Milky Way": {
        "model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0},
        },
        "R_env": 30.0,
        "obs": [
            {"r": 3,   "v": 213, "err": 7},
            {"r": 4,   "v": 220, "err": 6},
            {"r": 5,   "v": 225, "err": 5},
            {"r": 6,   "v": 228, "err": 4},
            {"r": 7,   "v": 229, "err": 4},
            {"r": 8,   "v": 230, "err": 3},
            {"r": 9,   "v": 231, "err": 4},
            {"r": 10,  "v": 232, "err": 5},
            {"r": 12,  "v": 233, "err": 6},
            {"r": 14,  "v": 232, "err": 8},
            {"r": 16,  "v": 230, "err": 10},
            {"r": 18,  "v": 228, "err": 12},
            {"r": 20,  "v": 220, "err": 15},
        ],
    },
    "NGC 3198": {
        "model": {
            "bulge": {"M": 0.5e9, "a": 0.3},
            "disk":  {"M": 2.5e10, "Rd": 3.0},
            "gas":   {"M": 1.0e10, "Rd": 6.0},
        },
        "R_env": 30.0,
        "obs": [
            {"r": 3,  "v": 120, "err": 6},
            {"r": 5,  "v": 147, "err": 5},
            {"r": 7,  "v": 155, "err": 4},
            {"r": 10, "v": 157, "err": 4},
            {"r": 15, "v": 155, "err": 5},
            {"r": 20, "v": 150, "err": 6},
            {"r": 25, "v": 148, "err": 8},
            {"r": 30, "v": 150, "err": 10},
        ],
    },
    "DDO 154": {
        "model": {
            "bulge": {"M": 0.0, "a": 0.1},
            "disk":  {"M": 3.0e7, "Rd": 0.8},
            "gas":   {"M": 3.6e8, "Rd": 2.5},
        },
        "R_env": 8.0,
        "obs": [
            {"r": 0.5, "v": 15,  "err": 3},
            {"r": 1.0, "v": 26,  "err": 3},
            {"r": 1.5, "v": 35,  "err": 3},
            {"r": 2.0, "v": 40,  "err": 3},
            {"r": 3.0, "v": 47,  "err": 3},
            {"r": 4.0, "v": 50,  "err": 3},
            {"r": 5.0, "v": 47,  "err": 4},
            {"r": 6.0, "v": 46,  "err": 5},
            {"r": 7.0, "v": 46,  "err": 5},
        ],
    },
    "NGC 3109": {
        "model": {
            "bulge": {"M": 0.0, "a": 0.1},
            "disk":  {"M": 6.4e8, "Rd": 1.8},
            "gas":   {"M": 4.8e8, "Rd": 3.5},
        },
        "R_env": 10.0,
        "obs": [
            {"r": 1,  "v": 20, "err": 5},
            {"r": 2,  "v": 37, "err": 4},
            {"r": 3,  "v": 48, "err": 3},
            {"r": 4,  "v": 55, "err": 3},
            {"r": 5,  "v": 60, "err": 3},
            {"r": 6,  "v": 63, "err": 4},
            {"r": 7,  "v": 65, "err": 4},
            {"r": 8,  "v": 67, "err": 5},
            {"r": 9,  "v": 67, "err": 6},
        ],
    },
}


# =====================================================================
# Physics
# =====================================================================
def enclosed_mass(r_kpc, model):
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


def total_model_mass(model):
    return model["bulge"]["M"] + model["disk"]["M"] + model["gas"]["M"]


def solve_x(y_N):
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def gfd_velocity(r_kpc, m_solar):
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
    y_N = gN / A0
    x = solve_x(y_N)
    return math.sqrt(A0 * x * r_m) / 1000.0


def gfd_from_gN(gN_ms2, r_m):
    """Given Newtonian acceleration, compute GFD velocity."""
    if gN_ms2 <= 0 or r_m <= 0:
        return 0.0
    y_N = gN_ms2 / A0
    x = solve_x(y_N)
    return math.sqrt(A0 * x * r_m) / 1000.0


def newtonian_gN(r_kpc, m_solar):
    """Newtonian acceleration in m/s^2."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    return G * M_kg / (r_m * r_m)


def vortex_velocity(r_kpc, M_total_solar, R_env_kpc):
    """v_vortex = v_throat * (r/r_throat)^{1/3}"""
    r_throat_kpc = ALPHA * R_env_kpc
    r_throat_m = r_throat_kpc * KPC_TO_M
    M_kg = M_total_solar * M_SUN
    if r_throat_m <= 0 or M_kg <= 0 or r_kpc <= 0:
        return 0.0
    v_throat_kms = math.sqrt(G * M_kg / r_throat_m) / 1000.0
    return v_throat_kms * (r_kpc / r_throat_kpc) ** (1.0 / 3.0)


def vortex_gN(r_kpc, M_total_solar, R_env_kpc):
    """Newtonian acceleration equivalent of the vortex profile."""
    v_kms = vortex_velocity(r_kpc, M_total_solar, R_env_kpc)
    if v_kms <= 0 or r_kpc <= 0:
        return 0.0
    v_ms = v_kms * 1000.0
    r_m = r_kpc * KPC_TO_M
    return v_ms * v_ms / r_m


def reduced_chi2(predicted, observed, errors, n_params=0):
    dof = len(observed) - n_params
    if dof <= 0:
        return float('inf')
    return sum((p - o)**2 / e**2
               for p, o, e in zip(predicted, observed, errors)) / dof


# =====================================================================
# Composition Methods
# =====================================================================

def method_gfd_only(r_kpc, model, R_env):
    """Baseline: pure GFD covariant."""
    return gfd_velocity(r_kpc, enclosed_mass(r_kpc, model))


def method_quadrature(r_kpc, model, R_env):
    """v^2 = v_gfd^2 + v_vtx^2 (add accelerations)."""
    v_g = gfd_velocity(r_kpc, enclosed_mass(r_kpc, model))
    v_v = vortex_velocity(r_kpc, total_model_mass(model), R_env)
    return math.sqrt(v_g**2 + v_v**2)


def method_4th_power(r_kpc, model, R_env):
    """v^4 = v_gfd^4 + v_vtx^4 (BTFR inspired)."""
    v_g = gfd_velocity(r_kpc, enclosed_mass(r_kpc, model))
    v_v = vortex_velocity(r_kpc, total_model_mass(model), R_env)
    return (v_g**4 + v_v**4) ** 0.25


def method_covariant_of_vortex(r_kpc, model, R_env):
    """Feed vortex gN into covariant solver (vortex replaces Newton)."""
    gN_v = vortex_gN(r_kpc, total_model_mass(model), R_env)
    r_m = r_kpc * KPC_TO_M
    return gfd_from_gN(gN_v, r_m)


def method_covariant_of_max_gN(r_kpc, model, R_env):
    """Feed max(gN_baryon, gN_vortex) into covariant."""
    m_enc = enclosed_mass(r_kpc, model)
    gN_b = newtonian_gN(r_kpc, m_enc)
    gN_v = vortex_gN(r_kpc, total_model_mass(model), R_env)
    gN_max = max(gN_b, gN_v)
    r_m = r_kpc * KPC_TO_M
    return gfd_from_gN(gN_max, r_m)


def method_covariant_of_sum_gN(r_kpc, model, R_env):
    """Feed gN_baryon + gN_vortex into covariant."""
    m_enc = enclosed_mass(r_kpc, model)
    gN_b = newtonian_gN(r_kpc, m_enc)
    gN_v = vortex_gN(r_kpc, total_model_mass(model), R_env)
    gN_total = gN_b + gN_v
    r_m = r_kpc * KPC_TO_M
    return gfd_from_gN(gN_total, r_m)


def method_vortex_only(r_kpc, model, R_env):
    """Pure vortex profile (no GFD)."""
    return vortex_velocity(r_kpc, total_model_mass(model), R_env)


def method_smooth_crossover(r_kpc, model, R_env):
    """GFD inside throat, vortex outside, sigmoid blend."""
    v_g = gfd_velocity(r_kpc, enclosed_mass(r_kpc, model))
    v_v = vortex_velocity(r_kpc, total_model_mass(model), R_env)
    r_t = ALPHA * R_env
    w = 1.0 / (1.0 + math.exp(-(r_kpc - r_t) / (0.2 * r_t)))
    return (1 - w) * v_g + w * v_v


def method_max(r_kpc, model, R_env):
    """max(v_gfd, v_vortex)."""
    v_g = gfd_velocity(r_kpc, enclosed_mass(r_kpc, model))
    v_v = vortex_velocity(r_kpc, total_model_mass(model), R_env)
    return max(v_g, v_v)


def method_geometric_mean(r_kpc, model, R_env):
    """sqrt(v_gfd * v_vortex) -- geometric mean."""
    v_g = gfd_velocity(r_kpc, enclosed_mass(r_kpc, model))
    v_v = vortex_velocity(r_kpc, total_model_mass(model), R_env)
    if v_g <= 0 or v_v <= 0:
        return max(v_g, v_v)
    return math.sqrt(v_g * v_v)


def method_covariant_of_vortex_gN_plus_baryon_gN(r_kpc, model, R_env):
    """
    Hmm, same as covariant_of_sum_gN, but let me be explicit.
    gN_effective = G*M_enc/r^2 + v_vtx^2/r
    Then solve x^2/(1+x) = gN_effective/a0
    """
    return method_covariant_of_sum_gN(r_kpc, model, R_env)


def method_vortex_replaces_gN_in_covariant(r_kpc, model, R_env):
    """
    Use the vortex gN INSTEAD of baryonic gN in the covariant eq.
    This means: the vortex defines the "Newtonian" potential,
    and then the covariant equation enhances it further.

    gN_eff = v_vortex^2 / r (which encodes M_total and r^{1/3} profile)
    x^2/(1+x) = gN_eff / a0
    v = sqrt(a0 * x * r)
    """
    return method_covariant_of_vortex(r_kpc, model, R_env)


def method_vortex_replaces_newton_keep_field(r_kpc, model, R_env):
    """
    The "field dynamics excess" of GFD over Newton, added to vortex.
    v_field = sqrt(v_gfd^2 - v_newton^2) (the extra from field dynamics)
    v_total = sqrt(v_vortex^2 + v_field^2)

    This preserves the field dynamics contribution but swaps the
    Newtonian base with the vortex profile.
    """
    m_enc = enclosed_mass(r_kpc, model)
    v_gfd_val = gfd_velocity(r_kpc, m_enc)
    r_m = r_kpc * KPC_TO_M
    M_kg = m_enc * M_SUN
    if r_m <= 0 or M_kg <= 0:
        return vortex_velocity(r_kpc, total_model_mass(model), R_env)
    v_newton = math.sqrt(G * M_kg / r_m) / 1000.0
    v_field_sq = max(v_gfd_val**2 - v_newton**2, 0)
    v_vtx = vortex_velocity(r_kpc, total_model_mass(model), R_env)
    return math.sqrt(v_vtx**2 + v_field_sq)


# =====================================================================
# Run all methods on all galaxies
# =====================================================================
METHODS = {
    "GFD only":          method_gfd_only,
    "Vortex only":       method_vortex_only,
    "Quadrature":        method_quadrature,
    "4th power":         method_4th_power,
    "Cov(vortex_gN)":    method_covariant_of_vortex,
    "Cov(max gN)":       method_covariant_of_max_gN,
    "Cov(sum gN)":       method_covariant_of_sum_gN,
    "Crossover":         method_smooth_crossover,
    "Max(GFD,vtx)":      method_max,
    "Geometric mean":    method_geometric_mean,
    "Vtx+field excess":  method_vortex_replaces_newton_keep_field,
}

print("=" * 110)
print("SYSTEMATIC COMPOSITION SEARCH")
print("v_vortex = v_throat * (r/r_throat)^{1/3}, r_throat = 0.30 * R_env")
print("=" * 110)
print()

# Compute chi2 for each method on each galaxy
results = {}
for method_name, method_fn in METHODS.items():
    results[method_name] = {}
    for gal_name, gal_data in GALAXIES.items():
        model = gal_data["model"]
        R_env = gal_data["R_env"]
        obs = gal_data["obs"]

        try:
            predicted = [method_fn(o["r"], model, R_env) for o in obs]
            observed = [o["v"] for o in obs]
            errors = [o["err"] for o in obs]
            chi2 = reduced_chi2(predicted, observed, errors)
        except Exception as e:
            chi2 = float('inf')

        results[method_name][gal_name] = chi2

# Print table
gal_names = list(GALAXIES.keys())
header = f"{'Method':>20} " + " ".join(f"{g:>10}" for g in gal_names) + f" {'MEAN':>10}"
print(header)
print("-" * len(header))

best_mean = float('inf')
best_method = None

for method_name in METHODS:
    chi2_vals = [results[method_name][g] for g in gal_names]
    mean_chi2 = sum(chi2_vals) / len(chi2_vals)
    row = f"{method_name:>20} " + " ".join(f"{c:10.1f}" for c in chi2_vals) + f" {mean_chi2:10.1f}"
    print(row)
    if mean_chi2 < best_mean:
        best_mean = mean_chi2
        best_method = method_name

print()
print(f"BEST METHOD: {best_method} (mean chi2 = {best_mean:.1f})")
print()


# =====================================================================
# Deep dive into the best methods
# =====================================================================
# Look at top 3
sorted_methods = sorted(METHODS.keys(),
                        key=lambda m: sum(results[m].values()) / len(results[m]))

print("=" * 80)
print("TOP 5 METHODS (by mean chi2)")
print("=" * 80)
print()

for rank, method_name in enumerate(sorted_methods[:5], 1):
    chi2_vals = [results[method_name][g] for g in gal_names]
    mean_chi2 = sum(chi2_vals) / len(chi2_vals)
    max_chi2 = max(chi2_vals)
    worst_gal = gal_names[chi2_vals.index(max_chi2)]
    print(f"#{rank}: {method_name}")
    print(f"    Mean chi2 = {mean_chi2:.1f}, Worst = {max_chi2:.1f} ({worst_gal})")
    for g in gal_names:
        print(f"    {g:>12}: chi2 = {results[method_name][g]:.1f}")
    print()


# =====================================================================
# PLOT: Top 3 methods across all galaxies
# =====================================================================
top3 = sorted_methods[:3]
colors_map = {
    "GFD only": "purple",
    "Vortex only": "lightblue",
    "Quadrature": "red",
    "4th power": "orange",
    "Cov(vortex_gN)": "green",
    "Cov(max gN)": "darkgreen",
    "Cov(sum gN)": "brown",
    "Crossover": "cyan",
    "Max(GFD,vtx)": "magenta",
    "Geometric mean": "gold",
    "Vtx+field excess": "deepskyblue",
}

n_gals = len(GALAXIES)
fig, axes = plt.subplots(2, n_gals, figsize=(5 * n_gals, 10))
fig.suptitle('Composition Method Comparison (top 3 + GFD baseline)',
             fontsize=14, fontweight='bold')

for col, (gal_name, gal_data) in enumerate(GALAXIES.items()):
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    obs_r = np.array([o["r"] for o in obs])
    obs_v = np.array([o["v"] for o in obs])
    obs_err = np.array([o["err"] for o in obs])

    r_max = max(o["r"] for o in obs) * 1.2
    r_plot = np.linspace(0.1, r_max, 200)

    # Top row: curves
    ax = axes[0, col]
    ax.errorbar(obs_r, obs_v, yerr=obs_err, fmt='o', color='orange',
                markersize=5, capsize=3, zorder=5, label='Obs')

    # Always show GFD baseline
    v_gfd = [method_gfd_only(r, model, R_env) for r in r_plot]
    ax.plot(r_plot, v_gfd, '-', color='purple', linewidth=1, alpha=0.5,
            label='GFD')

    # Show top 3
    for method_name in top3:
        method_fn = METHODS[method_name]
        v_arr = [method_fn(r, model, R_env) for r in r_plot]
        c = colors_map.get(method_name, 'gray')
        chi2 = results[method_name][gal_name]
        ax.plot(r_plot, v_arr, '-', color=c, linewidth=2,
                label=f'{method_name} ({chi2:.1f})')

    ax.set_title(gal_name)
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    # Bottom row: residuals for top 3
    ax = axes[1, col]
    for method_name in top3:
        method_fn = METHODS[method_name]
        resid = [method_fn(o["r"], model, R_env) - o["v"] for o in obs]
        c = colors_map.get(method_name, 'gray')
        ax.plot(obs_r, resid, 'o-', color=c, markersize=4,
                label=method_name)
    ax.fill_between(obs_r, -obs_err, obs_err, alpha=0.15, color='orange')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('Residual (km/s)')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_title('Residuals')

plt.tight_layout()
plt.savefig('m33_composition_search.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to m33_composition_search.png")


# =====================================================================
# DETAILED per-point for best method
# =====================================================================
print()
print("=" * 80)
print(f"DETAILED: {best_method}")
print("=" * 80)
print()

best_fn = METHODS[best_method]
for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]
    chi2 = results[best_method][gal_name]

    print(f"--- {gal_name} (chi2={chi2:.1f}) ---")
    print(f"{'r':>6} {'v_obs':>6} {'err':>4} {'v_pred':>7} {'resid':>7} {'sigma':>5}")
    for o in obs:
        v_pred = best_fn(o["r"], model, R_env)
        resid = v_pred - o["v"]
        sigma = abs(resid) / o["err"] if o["err"] > 0 else 0
        print(f"{o['r']:6.1f} {o['v']:6.0f} {o['err']:4.0f} "
              f"{v_pred:7.1f} {resid:+7.1f} {sigma:5.1f}")
    print()
