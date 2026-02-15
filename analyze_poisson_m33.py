"""
Standalone analysis: GFD Covariant + Poisson Operator on M33 data.

This script computes and plots:
  1. Standard GFD covariant velocity (baseline)
  2. GFD Poisson (covariant composed with kappa operator)
  3. Observed M33 rotation curve data
  4. Diagnostic panels: kappa(r), boost factor, residuals

The goal is to understand the composition formula and iterate on the
physics before integrating into the web app.

Run from the gravis directory:
    python analyze_poisson_m33.py

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# Constants (from physics/constants.py)
# =====================================================================
G = 6.67430e-11        # m^3 kg^-1 s^-2
M_SUN = 1.98892e30     # kg
KPC_TO_M = 3.0857e19   # meters per kpc
M_E = 9.1093837139e-31 # kg
R_E = 2.8179403205e-15 # m
K = 4
A0 = K * K * G * M_E / (R_E * R_E)

print(f"a0 = {A0:.4e} m/s^2")

# =====================================================================
# M33 Galaxy Data (from data/galaxies.py)
# =====================================================================
M33_MASS_MODEL = {
    "bulge": {"M": 0.1e9, "a": 0.2},    # Hernquist
    "disk":  {"M": 2.0e9, "Rd": 1.4},    # Exponential
    "gas":   {"M": 1.86e9, "Rd": 7.0},   # Exponential (HI+He)
}

M33_OBS = [
    {"r": 1,  "v": 45,  "err": 10},
    {"r": 2,  "v": 68,  "err": 8},
    {"r": 4,  "v": 100, "err": 5},
    {"r": 6,  "v": 108, "err": 5},
    {"r": 8,  "v": 112, "err": 5},
    {"r": 10, "v": 117, "err": 5},
    {"r": 12, "v": 122, "err": 6},
    {"r": 14, "v": 128, "err": 8},
    {"r": 16, "v": 130, "err": 10},
]

R_GALACTIC = 20.0  # galactic radius (gravitational horizon) in kpc
ALPHA = 0.30       # throat / horizon ratio

obs_r   = np.array([o["r"]   for o in M33_OBS])
obs_v   = np.array([o["v"]   for o in M33_OBS])
obs_err = np.array([o["err"] for o in M33_OBS])


# =====================================================================
# Mass Model: Enclosed Mass at radius r (kpc)
# =====================================================================
def enclosed_mass(r_kpc, model):
    """Total enclosed baryonic mass (solar masses) at r_kpc."""
    r = r_kpc
    m = 0.0

    # Hernquist bulge: M(<r) = M * r^2 / (r + a)^2
    b = model["bulge"]
    if b["M"] > 0 and r > 0:
        m += b["M"] * r**2 / (r + b["a"])**2

    # Exponential disk: M(<r) = M * [1 - (1 + r/Rd) * exp(-r/Rd)]
    d = model["disk"]
    if d["M"] > 0 and r > 0:
        x = r / d["Rd"]
        m += d["M"] * (1.0 - (1.0 + x) * math.exp(-x))

    # Gas disk: same exponential form
    g = model["gas"]
    if g["M"] > 0 and r > 0:
        x = r / g["Rd"]
        m += g["M"] * (1.0 - (1.0 + x) * math.exp(-x))

    return m


# =====================================================================
# GFD Covariant Field Equation
# =====================================================================
def solve_x(y_N):
    """Solve x^2/(1+x) = y_N for the physical root."""
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def gfd_velocity(r_kpc, m_solar):
    """Standard GFD covariant velocity in km/s."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
    y_N = gN / A0
    x = solve_x(y_N)
    g_eff = A0 * x
    v_ms = math.sqrt(g_eff * r_m)
    return v_ms / 1000.0


def gfd_acceleration(r_kpc, m_solar):
    """Standard GFD covariant acceleration in m/s^2."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
    y_N = gN / A0
    x = solve_x(y_N)
    return A0 * x


# =====================================================================
# Kappa Operator (Poisson unified manifold)
# =====================================================================
def kappa(r_kpc, r_t_kpc):
    """
    kappa(r) = 1 - exp(-r^2 / r_t^2)

    Properties:
        kappa(0) = 0  (full manifold coupling)
        kappa(r_t) = 1 - e^(-1) ~ 0.632
        kappa(inf) = 1 (standard gravity recovered)
    """
    if r_kpc <= 0 or r_t_kpc <= 0:
        return 0.0
    x = r_kpc / r_t_kpc
    return 1.0 - math.exp(-x * x)


# =====================================================================
# Composition: GFD Poisson = covariant / sqrt(kappa)
# =====================================================================
def poisson_velocity(r_kpc, m_solar, r_galactic, kappa_floor=1e-4):
    """
    GFD Poisson: v_poisson = v_cov / sqrt(kappa).

    Composes the covariant completion with the kappa operator
    derived from the Poisson equation (Section IX).
    """
    v_cov = gfd_velocity(r_kpc, m_solar)
    if v_cov <= 0:
        return 0.0
    r_t = ALPHA * r_galactic
    k = kappa(r_kpc, r_t)
    k_safe = max(k, kappa_floor)
    return v_cov / math.sqrt(k_safe)


# =====================================================================
# Compute curves
# =====================================================================
r_arr = np.linspace(0.1, 20.0, 200)

# Enclosed mass at each radius
m_arr = np.array([enclosed_mass(r, M33_MASS_MODEL) for r in r_arr])

# Standard GFD
v_gfd = np.array([gfd_velocity(r, enclosed_mass(r, M33_MASS_MODEL))
                   for r in r_arr])

# GFD Poisson with different kappa floors
v_poisson_raw = np.array([poisson_velocity(r, enclosed_mass(r, M33_MASS_MODEL),
                                            R_GALACTIC, kappa_floor=1e-4)
                           for r in r_arr])

v_poisson_01 = np.array([poisson_velocity(r, enclosed_mass(r, M33_MASS_MODEL),
                                           R_GALACTIC, kappa_floor=0.1)
                          for r in r_arr])

v_poisson_03 = np.array([poisson_velocity(r, enclosed_mass(r, M33_MASS_MODEL),
                                           R_GALACTIC, kappa_floor=0.3)
                          for r in r_arr])

# Newtonian
v_newton = np.array([
    math.sqrt(max(G * enclosed_mass(r, M33_MASS_MODEL) * M_SUN
                  / (r * KPC_TO_M), 0.0)) / 1000.0
    for r in r_arr
])

# Kappa profile
r_t = ALPHA * R_GALACTIC
kappa_arr = np.array([kappa(r, r_t) for r in r_arr])

# Boost factor
boost_arr = np.where(kappa_arr > 1e-6, 1.0 / np.sqrt(kappa_arr), np.nan)


# =====================================================================
# Print diagnostics
# =====================================================================
print(f"\nM33 Analysis")
print(f"{'='*60}")
print(f"Galactic radius: {R_GALACTIC} kpc")
print(f"Throat radius:   {r_t:.1f} kpc (alpha={ALPHA})")
print(f"Total mass at 16 kpc: {enclosed_mass(16, M33_MASS_MODEL):.3e} M_sun")
print()

print(f"{'r(kpc)':>8} {'M_enc':>12} {'v_GFD':>8} {'kappa':>8} {'boost':>8} {'v_Poi':>8} {'v_obs':>8}")
print(f"{'-'*8:>8} {'-'*12:>12} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")

for o in M33_OBS:
    r = o["r"]
    m = enclosed_mass(r, M33_MASS_MODEL)
    v_g = gfd_velocity(r, m)
    k = kappa(r, r_t)
    k_safe = max(k, 1e-4)
    b = 1.0 / math.sqrt(k_safe)
    v_p = poisson_velocity(r, m, R_GALACTIC)
    print(f"{r:8.1f} {m:12.3e} {v_g:8.1f} {k:8.4f} {b:8.2f} {v_p:8.1f} {o['v']:8.0f}")


# =====================================================================
# Plot
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('M33: GFD Covariant + Poisson Analysis', fontsize=14, fontweight='bold')

# --- Panel 1: Rotation curves ---
ax1 = axes[0, 0]
ax1.errorbar(obs_r, obs_v, yerr=obs_err, fmt='o', color='orange',
             label='Observed', markersize=6, capsize=3, zorder=5)
ax1.plot(r_arr, v_newton, '--', color='red', alpha=0.6, label='Newtonian')
ax1.plot(r_arr, v_gfd, '-', color='dodgerblue', linewidth=2, label='GFD Covariant')
ax1.plot(r_arr, v_poisson_raw, '-', color='magenta', linewidth=1.5,
         alpha=0.4, label='Poisson (floor=1e-4)')
ax1.plot(r_arr, v_poisson_01, '-', color='hotpink', linewidth=2,
         label='Poisson (floor=0.1)')
ax1.plot(r_arr, v_poisson_03, '-.', color='crimson', linewidth=2,
         label='Poisson (floor=0.3)')
ax1.set_xlabel('r (kpc)')
ax1.set_ylabel('v (km/s)')
ax1.set_title('Rotation Curves')
ax1.set_ylim(0, 250)
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(alpha=0.3)
ax1.axvline(r_t, color='gray', linestyle=':', alpha=0.5, label=f'r_t = {r_t:.1f} kpc')

# --- Panel 2: Kappa profile ---
ax2 = axes[0, 1]
ax2.plot(r_arr, kappa_arr, '-', color='purple', linewidth=2)
ax2.axhline(1 - math.exp(-1), color='gray', linestyle=':', alpha=0.5,
            label=f'kappa(r_t) = {1-math.exp(-1):.3f}')
ax2.axvline(r_t, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('r (kpc)')
ax2.set_ylabel('kappa(r)')
ax2.set_title(f'Screening Function (r_t = {r_t:.1f} kpc)')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1.1)

# --- Panel 3: Boost factor ---
ax3 = axes[1, 0]
ax3.plot(r_arr, boost_arr, '-', color='green', linewidth=2)
ax3.axvline(r_t, color='gray', linestyle=':', alpha=0.5)
ax3.axhline(1.0, color='gray', linestyle='-', alpha=0.3)
ax3.set_xlabel('r (kpc)')
ax3.set_ylabel('v_poisson / v_gfd')
ax3.set_title('Velocity Boost Factor = 1/sqrt(kappa)')
ax3.set_ylim(0, 10)
ax3.grid(alpha=0.3)
# Mark key values
ax3.annotate(f'At r_t: {1/math.sqrt(1-math.exp(-1)):.2f}x',
             xy=(r_t, 1/math.sqrt(1-math.exp(-1))),
             xytext=(r_t + 2, 3), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='green'))

# --- Panel 4: Residuals at observed points ---
ax4 = axes[1, 1]
for floor_val, color, label in [(None, 'dodgerblue', 'GFD'),
                                  (0.1, 'hotpink', 'Poisson (0.1)'),
                                  (0.3, 'crimson', 'Poisson (0.3)')]:
    residuals = []
    for o in M33_OBS:
        r = o["r"]
        m = enclosed_mass(r, M33_MASS_MODEL)
        if floor_val is None:
            v_pred = gfd_velocity(r, m)
        else:
            v_pred = poisson_velocity(r, m, R_GALACTIC, kappa_floor=floor_val)
        residuals.append(v_pred - o["v"])
    ax4.plot(obs_r, residuals, 'o-', color=color, label=label, markersize=5)

ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
ax4.fill_between(obs_r, -obs_err, obs_err, alpha=0.15, color='orange',
                 label='1-sigma obs error')
ax4.set_xlabel('r (kpc)')
ax4.set_ylabel('v_pred - v_obs (km/s)')
ax4.set_title('Residuals (predicted - observed)')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('m33_poisson_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to m33_poisson_analysis.png")
print("Open the PNG to view the analysis.")
