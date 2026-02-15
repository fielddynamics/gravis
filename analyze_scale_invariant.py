"""
Scale-Invariant Field Profile: One Lagrangian, One Curve.

The Poisson equation is scale-invariant. The same F(y) Lagrangian
that produces the field profile of a helium atom produces the
rotation curve of a galaxy. k=4, d=3, alpha=0.30 at every scale.

This means:
  - NOT GFD + vortex (two pieces stitched together)
  - NOT a composition of two models
  - ONE self-similar profile from the field equation

The covariant field equation x^2/(1+x) = gN/a0 already IS this
Lagrangian. But it uses gN = G*M_enc(r)/r^2 as input, which comes
from the STANDARD Poisson equation nabla^2 Phi = 4*pi*G*rho.

The DERIVED Poisson equation (Section IX) replaces this:
  nabla . (kappa(r) * nabla Phi) = 4*pi * G_eff * rho
  G_eff = (17/13) * G

This modifies gN BEFORE it enters the covariant equation.
The modification is not additive. It changes the source term.

The scale-invariant profile means: the same equation at the
galaxy scale as the atom scale. The throat at 0.30*R_env is
where the field transitions from the inner "bound" regime to
the outer "propagating" regime. The 1/3 power law in the
user's vortex plot is what the field profile looks like in
the transition region.

Let me work out what happens when we use the modified Poisson
equation to compute gN, then feed it into the covariant equation.

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
G_EFF_RATIO = 17.0 / 13.0  # From the derived Poisson equation


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
    """Solve x^2/(1+x) = y_N. This IS the F(y) Lagrangian."""
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def standard_gfd(r_kpc, m_solar):
    """Standard: gN from Poisson, then covariant F(y)."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
    y_N = gN / A0
    x = solve_x(y_N)
    return math.sqrt(A0 * x * r_m) / 1000.0


# =====================================================================
# THE SCALE-INVARIANT APPROACH
#
# The derived Poisson equation modifies the source:
#   nabla . (kappa(r) * nabla Phi) = 4*pi * G_eff * rho
#
# In spherical symmetry:
#   d/dr (kappa(r) * r^2 * dPhi/dr) = 4*pi * G_eff * rho * r^2
#
# The effective Newtonian acceleration becomes:
#   gN_eff(r) = G_eff * M_enc(r) / (r^2 * kappa(r))
#
# where kappa(r) = 1 - exp(-r^2/r_t^2) is the manifold operator,
# and G_eff = (17/13) * G.
#
# Then this gN_eff feeds into the SAME covariant equation:
#   x^2/(1+x) = gN_eff / a0
#   v = sqrt(a0 * x * r)
#
# ONE equation. ONE profile. The kappa modification is part of
# the Poisson equation, not a separate model.
# =====================================================================

def kappa(r_kpc, R_env_kpc):
    """
    Manifold operator from the derived Poisson equation.
    kappa(r) = 1 - exp(-r^2 / r_t^2)
    r_t = alpha * R_env
    """
    r_t = ALPHA * R_env_kpc
    if r_kpc <= 0 or r_t <= 0:
        return 0.0
    x = r_kpc / r_t
    return 1.0 - math.exp(-x * x)


def scale_invariant_gfd(r_kpc, m_solar, R_env_kpc, kappa_floor=0.01):
    """
    The full scale-invariant prediction. One equation, one profile.

    1. Compute gN from the DERIVED Poisson equation:
       gN_eff = G_eff * M_enc / (r^2 * kappa(r))
       G_eff = (17/13) * G

    2. Feed into the SAME covariant field equation:
       x^2/(1+x) = gN_eff / a0

    3. v = sqrt(a0 * x * r)

    The kappa operator is NOT a separate model. It is the
    Poisson equation's own structure at the galactic scale,
    the same structure that exists at the atomic scale.
    """
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0

    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN

    # Step 1: Modified Newtonian acceleration from derived Poisson
    k = kappa(r_kpc, R_env_kpc)
    k_safe = max(k, kappa_floor)
    G_eff = G_EFF_RATIO * G

    gN_eff = G_eff * M_kg / (r_m * r_m * k_safe)

    # Step 2: Same covariant field equation
    y_N = gN_eff / A0
    x = solve_x(y_N)

    # Step 3: Circular velocity
    return math.sqrt(A0 * x * r_m) / 1000.0


def reduced_chi2(predicted, observed, errors):
    dof = len(observed)
    if dof <= 0:
        return float('inf')
    return sum((p - o)**2 / e**2
               for p, o, e in zip(predicted, observed, errors)) / dof


# =====================================================================
# ANALYSIS: What kappa_floor gives the best results?
#
# The kappa_floor controls the maximum enhancement at small r.
# Physically, it represents the minimum value of kappa at the
# field origin. In a real field, the coupling never drops to
# exactly zero -- there is always SOME screening.
#
# kappa_floor = 1 means no modification (standard GFD)
# kappa_floor = 0.01 means up to 100x enhancement on gN
# But the covariant equation compresses this: v ~ (gN)^{1/4}
# in the deep regime, so 100x on gN is only ~3.2x on v.
# =====================================================================

print("=" * 80)
print("SCALE-INVARIANT FIELD PROFILE")
print("One Lagrangian. One Poisson equation. One curve.")
print("=" * 80)
print()
print("Modified Poisson: nabla.(kappa * nabla Phi) = 4*pi * G_eff * rho")
print(f"G_eff = (17/13) * G = {G_EFF_RATIO:.4f} * G")
print(f"kappa(r) = 1 - exp(-r^2/r_t^2), r_t = {ALPHA} * R_env")
print(f"Then: x^2/(1+x) = gN_eff/a0 (same covariant as always)")
print()

# Test different kappa floors
print("=" * 80)
print("KAPPA FLOOR SENSITIVITY")
print("=" * 80)
print()

floors = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005]

for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    print(f"--- {gal_name} (R_env={R_env}, M={total_model_mass(model):.1e}) ---")

    # Standard GFD for reference
    v_std = [standard_gfd(o["r"], enclosed_mass(o["r"], model)) for o in obs]
    chi2_std = reduced_chi2(v_std, [o["v"] for o in obs], [o["err"] for o in obs])
    print(f"  Standard GFD: chi2 = {chi2_std:.1f}")

    best_floor = None
    best_chi2 = float('inf')

    for floor in floors:
        v_pred = [scale_invariant_gfd(o["r"], enclosed_mass(o["r"], model),
                                       R_env, kappa_floor=floor)
                  for o in obs]
        chi2 = reduced_chi2(v_pred, [o["v"] for o in obs], [o["err"] for o in obs])

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_floor = floor

        # Only print interesting floors
        if floor in [1.0, 0.3, 0.1, 0.05, 0.01]:
            print(f"  floor={floor:5.2f}: chi2 = {chi2:.1f}")

    print(f"  BEST: floor={best_floor} -> chi2 = {best_chi2:.1f}")
    print()


# =====================================================================
# KEY INSIGHT: What does kappa DO to the field equation?
#
# In the deep field regime (gN << a0):
#   x ~ sqrt(gN/a0) (from x^2/(1+x) ~ x^2 ~ gN/a0)
#   g = a0*x ~ sqrt(a0*gN)
#   v^2 = g*r ~ sqrt(a0*gN)*r = sqrt(a0 * G*M/(r^2*kappa)) * r
#   v^2 ~ sqrt(a0*G*M) * r^{1/2} / kappa^{1/4}
#   v ~ (a0*G*M)^{1/4} * r^{1/4} / kappa^{1/8}
#
# Wait, let me be more careful.
# In deep MOND: x^2 ~ gN/a0, so x = sqrt(gN/a0)
#   v^2 = a0*x*r = a0*sqrt(gN/a0)*r = sqrt(a0*gN)*r
#   v^2 = sqrt(a0 * G_eff*M/(r^2*kappa)) * r
#   v^2 = sqrt(a0*G_eff*M/kappa) * r / r = sqrt(a0*G_eff*M/kappa)
#   Wait: v^2 = sqrt(a0 * gN_eff) * r
#   gN_eff = G_eff*M/(r^2*kappa)
#   v^2 = sqrt(a0*G_eff*M / (r^2*kappa)) * r
#   v^2 = r * sqrt(a0*G_eff*M) / (r * sqrt(kappa))
#   v^2 = sqrt(a0*G_eff*M) / sqrt(kappa)
#   v^2 = sqrt(a0*G_eff*M) * kappa^{-1/2}
#   v = (a0*G_eff*M)^{1/4} * kappa^{-1/4}
#
# So in the deep regime: v ~ kappa^{-1/4}.
# The 1/4 power compresses the divergence enormously!
#
# At the throat (kappa=0.632): v_boost = 0.632^{-1/4} = 1.12x
# At r=0.5*r_t (kappa=0.221): v_boost = 0.221^{-1/4} = 1.46x
# At r=0.3*r_t (kappa=0.086): v_boost = 0.086^{-1/4} = 1.85x
# At r=0.1*r_t (kappa=0.010): v_boost = 0.010^{-1/4} = 3.16x
#
# The covariant equation NATURALLY suppresses the kappa divergence.
# The 1/kappa in the Poisson becomes 1/kappa^{1/4} in the velocity.
# This is why the same equation works at both scales.
# =====================================================================

print("=" * 80)
print("THE COVARIANT COMPRESSION")
print("=" * 80)
print()
print("In the deep field regime (gN << a0):")
print("  v ~ (a0 * G_eff * M)^{1/4} * kappa^{-1/4}")
print()
print("The 1/kappa divergence from the Poisson equation is compressed")
print("to kappa^{-1/4} by the covariant field equation!")
print()
print(f"{'r/r_t':>8} {'kappa':>8} {'1/kappa':>10} {'kappa^{-1/4}':>12} {'v_boost':>10}")
print("-" * 55)
for ratio in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    k = 1.0 - math.exp(-ratio**2)
    k_safe = max(k, 0.001)
    inv_k = 1.0 / k_safe
    boost = k_safe ** (-0.25)
    print(f"{ratio:8.1f} {k:8.4f} {inv_k:10.1f} {boost:12.2f} {boost:10.2f}x")

print()
print("At r/r_t = 0.1: kappa = 0.01, but velocity boost is only 3.16x")
print("At r/r_t = 1.0 (throat): kappa = 0.632, boost = 1.12x")
print("At r/r_t = 2.0: kappa = 0.982, boost = 1.005x (negligible)")
print()
print("The covariant equation ensures the profile is ONE smooth curve,")
print("not a divergent spike. The 'vortex' shape emerges naturally.")
print()


# =====================================================================
# DETAILED PREDICTIONS: Scale-invariant vs Standard GFD
# =====================================================================
print("=" * 80)
print("DETAILED PREDICTIONS (floor=0.01)")
print("=" * 80)
print()

for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]
    r_t = ALPHA * R_env

    print(f"--- {gal_name} (r_throat={r_t:.1f} kpc) ---")
    print(f"{'r':>6} {'r/r_t':>6} {'kappa':>7} {'v_std':>7} {'v_SI':>7} "
          f"{'v_obs':>6} {'err':>4} {'res_std':>8} {'res_SI':>7} {'sig_SI':>6}")

    for o in obs:
        r = o["r"]
        m = enclosed_mass(r, model)
        k = kappa(r, R_env)
        v_std = standard_gfd(r, m)
        v_si = scale_invariant_gfd(r, m, R_env, kappa_floor=0.01)
        res_std = v_std - o["v"]
        res_si = v_si - o["v"]
        sig_si = abs(res_si) / o["err"] if o["err"] > 0 else 0

        print(f"{r:6.1f} {r/r_t:6.2f} {k:7.4f} {v_std:7.1f} {v_si:7.1f} "
              f"{o['v']:6.0f} {o['err']:4.0f} {res_std:+8.1f} {res_si:+7.1f} {sig_si:6.1f}")

    v_std_arr = [standard_gfd(o["r"], enclosed_mass(o["r"], model)) for o in obs]
    v_si_arr = [scale_invariant_gfd(o["r"], enclosed_mass(o["r"], model),
                                     R_env, kappa_floor=0.01) for o in obs]
    chi2_std = reduced_chi2(v_std_arr, [o["v"] for o in obs], [o["err"] for o in obs])
    chi2_si = reduced_chi2(v_si_arr, [o["v"] for o in obs], [o["err"] for o in obs])
    print(f"  chi2: standard={chi2_std:.1f}, scale-invariant={chi2_si:.1f}")
    print()


# =====================================================================
# PLOT: Scale-invariant field profile across all galaxies
# =====================================================================
n = len(GALAXIES)
fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
fig.suptitle('Scale-Invariant Field Profile: Modified Poisson -> Covariant F(y)',
             fontsize=13, fontweight='bold')

for col, (gal_name, gal_data) in enumerate(GALAXIES.items()):
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]
    r_t = ALPHA * R_env

    obs_r = np.array([o["r"] for o in obs])
    obs_v = np.array([o["v"] for o in obs])
    obs_err = np.array([o["err"] for o in obs])

    r_max = max(o["r"] for o in obs) * 1.2
    r_plot = np.linspace(0.1, r_max, 200)

    v_std = [standard_gfd(r, enclosed_mass(r, model)) for r in r_plot]

    # Top: rotation curves with different floors
    ax = axes[0, col]
    ax.errorbar(obs_r, obs_v, yerr=obs_err, fmt='o', color='orange',
                markersize=5, capsize=3, zorder=5, label='Observed')
    ax.plot(r_plot, v_std, '-', color='purple', linewidth=1.5,
            label='Standard GFD')

    for floor, color, ls in [(0.1, 'deepskyblue', '--'),
                               (0.05, 'cyan', '-'),
                               (0.01, 'lime', '-')]:
        v_si = [scale_invariant_gfd(r, enclosed_mass(r, model),
                                     R_env, kappa_floor=floor)
                for r in r_plot]
        chi2 = reduced_chi2(
            [scale_invariant_gfd(o["r"], enclosed_mass(o["r"], model),
                                 R_env, kappa_floor=floor) for o in obs],
            [o["v"] for o in obs], [o["err"] for o in obs])
        ax.plot(r_plot, v_si, color=color, linestyle=ls, linewidth=2,
                label=f'SI floor={floor} (chi2={chi2:.1f})')

    ax.axvline(r_t, color='gray', linestyle=':', alpha=0.4,
               label=f'r_t={r_t:.1f}')
    ax.set_title(f'{gal_name}\nR_env={R_env}, M={total_model_mass(model):.1e}')
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=6, loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    # Bottom: kappa profile and boost
    ax2 = axes[1, col]
    kappa_arr = [kappa(r, R_env) for r in r_plot]
    boost_arr = [max(k, 0.01)**(-0.25) for k in kappa_arr]

    ax2.plot(r_plot, boost_arr, '-', color='green', linewidth=2,
             label='v_boost = kappa^{-1/4}')
    ax2.axvline(r_t, color='gray', linestyle=':', alpha=0.4)
    ax2.axhline(1.0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('r (kpc)')
    if col == 0:
        ax2.set_ylabel('Velocity boost factor')
    ax2.set_title('Covariant compression')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.9, 4)

plt.tight_layout()
plt.savefig('m33_scale_invariant.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to m33_scale_invariant.png")


# =====================================================================
# SUMMARY
# =====================================================================
print()
print("=" * 80)
print("SUMMARY: SCALE-INVARIANT FIELD PROFILE")
print("=" * 80)
print()
print("The modified Poisson equation changes gN BEFORE the covariant equation:")
print("  gN_eff = (17/13) * G * M_enc / (r^2 * kappa(r))")
print("  kappa(r) = 1 - exp(-r^2/r_t^2), r_t = 0.30 * R_env")
print()
print("The covariant equation x^2/(1+x) = gN_eff/a0 compresses the")
print("1/kappa divergence to kappa^{-1/4} in the velocity.")
print()
print("This produces ONE smooth profile from a SINGLE equation:")
print("  - Inside throat: mild enhancement (kappa small, but compressed)")
print("  - At throat: ~12% boost")
print("  - Outside throat: kappa -> 1, recovers standard GFD")
print()
print("The profile shape depends on the regime:")
print("  - Deep field (gN << a0): v ~ (GM)^{1/4} * kappa^{-1/4}")
print("  - Newtonian (gN >> a0): v ~ sqrt(GM/r) / sqrt(kappa)")
print("  - Transition: smoothly interpolated by F(y)")
print()

# Final chi2 table
print(f"{'Galaxy':>12} {'chi2_std':>10} {'chi2_SI':>10} {'improvement':>12}")
print("-" * 50)
for gal_name, gal_data in GALAXIES.items():
    model = gal_data["model"]
    R_env = gal_data["R_env"]
    obs = gal_data["obs"]

    v_std = [standard_gfd(o["r"], enclosed_mass(o["r"], model)) for o in obs]
    v_si = [scale_invariant_gfd(o["r"], enclosed_mass(o["r"], model),
                                 R_env, kappa_floor=0.01) for o in obs]
    chi2_std = reduced_chi2(v_std, [o["v"] for o in obs], [o["err"] for o in obs])
    chi2_si = reduced_chi2(v_si, [o["v"] for o in obs], [o["err"] for o in obs])
    improvement = ((chi2_std - chi2_si) / chi2_std) * 100

    print(f"{gal_name:>12} {chi2_std:10.1f} {chi2_si:10.1f} {improvement:+11.0f}%")
