"""
Recursive Scale-Invariant Field Solution.

The F(y) Lagrangian is the complete equation at every scale.
The 17/13 Poisson factor means 4/13 ~ 30% of the coupling comes
from the field origin itself. The field origin is recursive:
  field origins within field origins, each with k=4 closure.

The covariant equation x^2/(1+x) = gN/a0 produces a "phantom mass":
  M_phantom(r) = g(r)*r^2/G - M_baryon(r)

This phantom mass IS the gravitational field's own mass-energy.
If the field origin is recursive, then alpha = 4/13 of this
phantom mass acts as additional source, generating more field,
generating more phantom mass, etc.

Self-consistent iteration:
  1. g_0(r) from standard covariant equation
  2. M_phantom(r) = g_0*r^2/G - M_baryon
  3. M_eff(r) = M_baryon + alpha * M_phantom
  4. gN_eff = G * M_eff / r^2
  5. Solve x^2/(1+x) = gN_eff/a0 for g_1
  6. Repeat until convergence

Key question: does this recursive field energy provide the
outer galaxy support that M33 needs?

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
ALPHA_THROAT = 0.30
ALPHA_RECURSIVE = 4.0 / 13.0  # Recursive coupling fraction from 17/13


# =====================================================================
# Galaxy Data
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
            {"r": 5,   "v": 225, "err": 5},
            {"r": 8,   "v": 230, "err": 3},
            {"r": 10,  "v": 232, "err": 5},
            {"r": 14,  "v": 232, "err": 8},
            {"r": 20,  "v": 220, "err": 15},
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
            {"r": 2.0, "v": 40,  "err": 3},
            {"r": 3.0, "v": 47,  "err": 3},
            {"r": 5.0, "v": 47,  "err": 4},
            {"r": 7.0, "v": 46,  "err": 5},
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


def solve_x(y_N):
    """Solve x^2/(1+x) = y_N. The F(y) field equation."""
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def gfd_acceleration_ms2(r_kpc, m_solar):
    """GFD acceleration in m/s^2."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
    x = solve_x(gN / A0)
    return A0 * x


def gfd_velocity(r_kpc, m_solar):
    """GFD velocity in km/s."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    g = gfd_acceleration_ms2(r_kpc, m_solar)
    return math.sqrt(g * r_m) / 1000.0


def reduced_chi2(predicted, observed, errors):
    dof = len(observed)
    if dof <= 0:
        return float('inf')
    return sum((p - o)**2 / e**2
               for p, o, e in zip(predicted, observed, errors)) / dof


# =====================================================================
# RECURSIVE FIELD SOLVER
#
# The phantom mass at radius r is:
#   M_phantom(r) = g(r) * r^2 / G - M_baryon(r)
#
# This is the mass-energy of the gravitational field itself.
# If the field origin is recursive (self-similar), a fraction
# alpha = 4/13 of this field mass acts as additional source.
#
# Iterate:
#   M_eff^(n+1)(r) = M_baryon(r) + alpha * M_phantom^(n)(r)
#   gN^(n+1) = G * M_eff^(n+1) / r^2
#   x^2/(1+x) = gN^(n+1) / a0
#   g^(n+1) = a0 * x
#   M_phantom^(n+1) = g^(n+1) * r^2 / G - M_baryon
# =====================================================================

def solve_recursive(r_arr, m_baryon_arr, alpha=ALPHA_RECURSIVE,
                    max_iter=50, tol=1e-6, verbose=False):
    """
    Self-consistent recursive field solution.

    Parameters
    ----------
    r_arr : array
        Radii in kpc.
    m_baryon_arr : array
        Enclosed baryonic mass at each radius in solar masses.
    alpha : float
        Recursive coupling fraction (4/13 from the Poisson equation).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance (relative change in v).

    Returns
    -------
    dict with keys:
        v_kms : velocity array in km/s
        g_ms2 : acceleration array in m/s^2
        m_phantom : phantom mass array in solar masses
        m_eff : effective mass array in solar masses
        n_iter : number of iterations
        history : list of velocity arrays at each iteration
    """
    n = len(r_arr)
    r_m = np.array([r * KPC_TO_M for r in r_arr])
    M_baryon_kg = np.array([m * M_SUN for m in m_baryon_arr])

    # Iteration 0: standard GFD
    g = np.zeros(n)
    for i in range(n):
        g[i] = gfd_acceleration_ms2(r_arr[i], m_baryon_arr[i])

    v_prev = np.sqrt(g * r_m) / 1000.0
    history = [v_prev.copy()]

    for iteration in range(max_iter):
        # Compute phantom mass: M_phantom = g*r^2/G - M_baryon
        M_phantom_kg = np.maximum(g * r_m**2 / G - M_baryon_kg, 0.0)
        M_phantom_solar = M_phantom_kg / M_SUN

        # Effective mass: M_eff = M_baryon + alpha * M_phantom
        M_eff_kg = M_baryon_kg + alpha * M_phantom_kg
        M_eff_solar = M_eff_kg / M_SUN

        # New Newtonian acceleration from effective mass
        g_new = np.zeros(n)
        for i in range(n):
            if r_m[i] > 0 and M_eff_kg[i] > 0:
                gN_eff = G * M_eff_kg[i] / (r_m[i]**2)
                x = solve_x(gN_eff / A0)
                g_new[i] = A0 * x

        v_new = np.sqrt(np.maximum(g_new * r_m, 0.0)) / 1000.0
        history.append(v_new.copy())

        # Check convergence
        max_change = np.max(np.abs(v_new - v_prev) /
                            np.maximum(v_prev, 1.0))
        if verbose:
            print(f"  Iteration {iteration+1}: max change = {max_change:.6f}")

        if max_change < tol:
            if verbose:
                print(f"  Converged after {iteration+1} iterations")
            break

        g = g_new
        v_prev = v_new

    return {
        "v_kms": v_new,
        "g_ms2": g_new,
        "m_phantom": M_phantom_solar,
        "m_eff": M_eff_solar,
        "n_iter": iteration + 1,
        "history": history,
    }


# =====================================================================
# ANALYSIS
# =====================================================================
print("=" * 80)
print("RECURSIVE SCALE-INVARIANT FIELD SOLUTION")
print("=" * 80)
print()
print(f"Alpha (recursive coupling) = 4/13 = {ALPHA_RECURSIVE:.4f}")
print(f"This means {ALPHA_RECURSIVE*100:.1f}% of the phantom mass feeds back")
print(f"as additional gravitational source at each recursion level.")
print()
print(f"Geometric sum: 1/(1-alpha) = {1/(1-ALPHA_RECURSIVE):.4f}")
print(f"So total enhancement converges to G_eff = {1/(1-ALPHA_RECURSIVE):.4f} * G")
print(f"vs single-level: G_eff = 17/13 = {17/13:.4f} * G")
print()


# =====================================================================
# Test different alpha values
# =====================================================================
print("=" * 80)
print("ALPHA SENSITIVITY")
print("=" * 80)
print()

alphas_to_test = [0.0, 0.1, ALPHA_RECURSIVE, 0.5, 0.6, 0.7]

for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    obs_r = np.array([o["r"] for o in obs])
    obs_v = np.array([o["v"] for o in obs])
    obs_err = np.array([o["err"] for o in obs])

    # Dense radial grid for the solver
    r_dense = np.linspace(0.1, R_env * 1.3, 300)
    m_dense = np.array([enclosed_mass(r, model) for r in r_dense])

    print(f"--- {gal_name} ---")
    print(f"{'alpha':>8} {'chi2':>8} {'n_iter':>7} {'v@max_r':>8} {'M_ph/M_b':>10}")

    for alpha in alphas_to_test:
        result = solve_recursive(r_dense, m_dense, alpha=alpha, verbose=False)

        # Interpolate to observation radii
        v_pred = np.interp(obs_r, r_dense, result["v_kms"])
        chi2 = reduced_chi2(v_pred, obs_v, obs_err)

        # Phantom-to-baryon ratio at max obs radius
        r_max_obs = obs_r[-1]
        idx_max = np.argmin(np.abs(r_dense - r_max_obs))
        m_ph = result["m_phantom"][idx_max]
        m_b = m_dense[idx_max]
        ratio = m_ph / m_b if m_b > 0 else 0

        v_at_max = np.interp(r_max_obs, r_dense, result["v_kms"])

        print(f"{alpha:8.3f} {chi2:8.1f} {result['n_iter']:7d} "
              f"{v_at_max:8.1f} {ratio:10.2f}")

    print()


# =====================================================================
# Detailed results for alpha = 4/13
# =====================================================================
print("=" * 80)
print(f"DETAILED RESULTS: alpha = 4/13 = {ALPHA_RECURSIVE:.4f}")
print("=" * 80)
print()

all_results = {}
for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    obs_r = np.array([o["r"] for o in obs])
    obs_v = np.array([o["v"] for o in obs])
    obs_err = np.array([o["err"] for o in obs])

    r_dense = np.linspace(0.1, R_env * 1.3, 300)
    m_dense = np.array([enclosed_mass(r, model) for r in r_dense])

    result = solve_recursive(r_dense, m_dense, alpha=ALPHA_RECURSIVE,
                              verbose=True)
    all_results[gal_name] = result

    v_pred_obs = np.interp(obs_r, r_dense, result["v_kms"])
    v_std_obs = np.array([gfd_velocity(o["r"], enclosed_mass(o["r"], model))
                           for o in obs])

    chi2_std = reduced_chi2(v_std_obs, obs_v, obs_err)
    chi2_rec = reduced_chi2(v_pred_obs, obs_v, obs_err)

    print(f"\n--- {gal_name} ---")
    print(f"chi2: standard GFD = {chi2_std:.1f}, recursive = {chi2_rec:.1f}")
    print()
    print(f"{'r':>6} {'v_obs':>6} {'err':>4} {'v_std':>7} {'v_rec':>7} "
          f"{'M_bar':>10} {'M_ph':>10} {'M_eff':>10} {'res_rec':>8} {'sig':>5}")

    for i, o in enumerate(obs):
        r = o["r"]
        idx = np.argmin(np.abs(r_dense - r))
        v_s = gfd_velocity(r, enclosed_mass(r, model))
        v_r = np.interp(r, r_dense, result["v_kms"])
        m_b = enclosed_mass(r, model)
        m_ph = np.interp(r, r_dense, result["m_phantom"])
        m_eff = np.interp(r, r_dense, result["m_eff"])
        res = v_r - o["v"]
        sig = abs(res) / o["err"] if o["err"] > 0 else 0

        print(f"{r:6.1f} {o['v']:6.0f} {o['err']:4.0f} {v_s:7.1f} {v_r:7.1f} "
              f"{m_b:10.2e} {m_ph:10.2e} {m_eff:10.2e} {res:+8.1f} {sig:5.1f}")

    print()


# =====================================================================
# What does the phantom mass profile look like?
# =====================================================================
print("=" * 80)
print("PHANTOM MASS PROFILE (the field's own mass-energy)")
print("=" * 80)
print()

for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]

    r_dense = np.linspace(0.1, R_env * 1.3, 300)
    m_dense = np.array([enclosed_mass(r, model) for r in r_dense])
    result = all_results[gal_name]

    print(f"--- {gal_name} ---")
    print(f"{'r':>6} {'M_baryon':>12} {'M_phantom':>12} {'M_eff':>12} {'ratio':>8}")

    for r_check in [0.5, 1, 2, 4, 6, 8, 10, 14, 20]:
        if r_check > R_env * 1.3:
            continue
        idx = np.argmin(np.abs(r_dense - r_check))
        m_b = m_dense[idx]
        m_ph = result["m_phantom"][idx]
        m_eff = result["m_eff"][idx]
        ratio = m_ph / m_b if m_b > 0 else 0
        print(f"{r_check:6.1f} {m_b:12.2e} {m_ph:12.2e} {m_eff:12.2e} {ratio:8.2f}")
    print()


# =====================================================================
# Convergence visualization: how does the curve evolve per iteration?
# =====================================================================
print("=" * 80)
print("CONVERGENCE: M33 iteration history")
print("=" * 80)
print()

model = GALAXIES["M33"]["model"]
R_env = GALAXIES["M33"]["R_env"]
r_dense = np.linspace(0.1, R_env * 1.3, 300)
m_dense = np.array([enclosed_mass(r, model) for r in r_dense])

# Run with verbose
result_m33 = solve_recursive(r_dense, m_dense, alpha=ALPHA_RECURSIVE,
                              verbose=True, max_iter=20)

# Print convergence at r=10 kpc
print(f"\nVelocity at r=10 kpc per iteration:")
for i, v_hist in enumerate(result_m33["history"]):
    v_at_10 = np.interp(10.0, r_dense, v_hist)
    label = "standard GFD" if i == 0 else f"iteration {i}"
    print(f"  {label:>15}: v = {v_at_10:.1f} km/s")

print(f"\n  Observed: v = 117 km/s")


# =====================================================================
# PLOT
# =====================================================================
n_gals = len(GALAXIES)
fig, axes = plt.subplots(2, n_gals, figsize=(6 * n_gals, 10))
fig.suptitle('Recursive Scale-Invariant Field (alpha = 4/13)',
             fontsize=14, fontweight='bold')

for col, (gal_name, gal_data) in enumerate(GALAXIES.items()):
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    obs_r = np.array([o["r"] for o in obs])
    obs_v = np.array([o["v"] for o in obs])
    obs_err = np.array([o["err"] for o in obs])

    r_dense = np.linspace(0.1, R_env * 1.3, 300)
    m_dense = np.array([enclosed_mass(r, model) for r in r_dense])

    # Standard GFD
    v_std = np.array([gfd_velocity(r, enclosed_mass(r, model))
                       for r in r_dense])

    # Recursive with different alphas
    result_0 = solve_recursive(r_dense, m_dense, alpha=0.0)
    result_4_13 = solve_recursive(r_dense, m_dense, alpha=ALPHA_RECURSIVE)
    result_0_5 = solve_recursive(r_dense, m_dense, alpha=0.5)

    # Top: rotation curves
    ax = axes[0, col]
    ax.errorbar(obs_r, obs_v, yerr=obs_err, fmt='o', color='orange',
                markersize=5, capsize=3, zorder=5, label='Observed')
    ax.plot(r_dense, v_std, '-', color='purple', linewidth=1.5,
            label='Standard GFD')
    ax.plot(r_dense, result_4_13["v_kms"], '-', color='cyan', linewidth=2.5,
            label=f'Recursive a=4/13')
    ax.plot(r_dense, result_0_5["v_kms"], '--', color='lime', linewidth=2,
            label=f'Recursive a=0.5')

    chi2_std = reduced_chi2(
        np.interp(obs_r, r_dense, v_std), obs_v, obs_err)
    chi2_rec = reduced_chi2(
        np.interp(obs_r, r_dense, result_4_13["v_kms"]), obs_v, obs_err)

    ax.set_title(f'{gal_name}\nchi2: std={chi2_std:.1f}, rec={chi2_rec:.1f}')
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    # Bottom: phantom mass profile
    ax2 = axes[1, col]
    m_baryon_plot = m_dense / 1e9  # in 10^9 M_sun
    m_phantom_plot = result_4_13["m_phantom"] / 1e9
    m_eff_plot = result_4_13["m_eff"] / 1e9

    ax2.plot(r_dense, m_baryon_plot, '-', color='red', linewidth=1.5,
             label='M_baryon')
    ax2.plot(r_dense, m_phantom_plot, '-', color='blue', linewidth=1.5,
             label='M_phantom (field)')
    ax2.plot(r_dense, m_eff_plot, '-', color='cyan', linewidth=2,
             label='M_eff = M_b + a*M_ph')
    ax2.set_xlabel('r (kpc)')
    if col == 0:
        ax2.set_ylabel('Mass (10^9 M_sun)')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_title('Mass profiles')

plt.tight_layout()
plt.savefig('m33_recursive_field.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to m33_recursive_field.png")


# =====================================================================
# KEY ANALYSIS: Why phantom mass grows linearly with r
# =====================================================================
print()
print("=" * 80)
print("WHY THE PHANTOM MASS GROWS LINEARLY WITH r")
print("=" * 80)
print()
print("In the deep field regime (gN << a0):")
print("  x ~ sqrt(gN/a0)")
print("  g = a0*x ~ sqrt(a0*gN) = sqrt(a0*G*M_baryon/r^2)")
print("  g*r^2 ~ r * sqrt(a0*G*M_baryon)")
print()
print("  M_total_Newton = g*r^2/G = r * sqrt(a0*M_baryon/G)")
print("  M_phantom = M_total - M_baryon = r*sqrt(a0*M_baryon/G) - M_baryon")
print()
print("For large r where M_baryon plateaus:")
print("  M_phantom ~ r * sqrt(a0*M_baryon/G)")
print()
print("This LINEAR growth with r is why flat rotation curves exist!")
print("And it is why the recursive feeding (adding alpha*M_phantom)")
print("provides INCREASING support at larger r.")
print()
print("Each recursion level adds alpha * r * sqrt(...) more mass,")
print("which grows linearly with r, compounding the outer enhancement.")
print()

# Summary table
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"{'Galaxy':>12} {'chi2_std':>10} {'chi2_4/13':>10} {'chi2_0.5':>10}")
print("-" * 48)

for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    obs_r = np.array([o["r"] for o in obs])
    obs_v = np.array([o["v"] for o in obs])
    obs_err = np.array([o["err"] for o in obs])

    r_dense = np.linspace(0.1, R_env * 1.3, 300)
    m_dense = np.array([enclosed_mass(r, model) for r in r_dense])

    v_std = np.array([gfd_velocity(r, enclosed_mass(r, model))
                       for r in r_dense])
    r_4_13 = solve_recursive(r_dense, m_dense, alpha=ALPHA_RECURSIVE)
    r_0_5 = solve_recursive(r_dense, m_dense, alpha=0.5)

    chi2_std = reduced_chi2(np.interp(obs_r, r_dense, v_std), obs_v, obs_err)
    chi2_413 = reduced_chi2(np.interp(obs_r, r_dense, r_4_13["v_kms"]),
                            obs_v, obs_err)
    chi2_05 = reduced_chi2(np.interp(obs_r, r_dense, r_0_5["v_kms"]),
                           obs_v, obs_err)

    print(f"{gal_name:>12} {chi2_std:10.1f} {chi2_413:10.1f} {chi2_05:10.1f}")
