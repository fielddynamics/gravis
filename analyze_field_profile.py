"""
One Field Object. One Profile.

A gravity field has:
  - A horizon (envelope)
  - A throat at 0.30 * horizon (the Field Origin)
  - A merkaba (dual tetrahedron) coupled at the origin inside

An electron has this structure at R_e scale.
A galaxy has this IDENTICAL structure at galactic scale.
Same topology. Same profile. Different power output.

The field origin's self-similar coupling polynomial
f(k) = 1 + k + k^2 determines how mass-energy is
distributed within the envelope. The k=4 topology
in d=3 dimensions distributes the gravitational
mass as M(r) ~ r^{k/d} = r^{4/3}.

This is the vortex distribution:
  v(r) ~ r^{1/3} in the deep field regime

because v^4 ~ M(r) * a0, and M ~ r^{4/3} gives v ~ r^{1/3}.

No fitting. No stitching. One equation.
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
ALPHA_THROAT = 4.0 / 13.0  # = 0.3077 (from the Poisson equation 17/13 - 1)
P_EXPONENT = float(K) / float(D)  # = 4/3 (simplex faces / dimensions)


# =====================================================================
# Galaxy Data
# =====================================================================
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


# =====================================================================
# Physics: the one equation
# =====================================================================
def solve_x(y_N):
    """x^2/(1+x) = y_N. The F(y) field equation."""
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def v_from_mass(r_kpc, m_solar):
    """Velocity from enclosed mass through the covariant field equation."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
    x = solve_x(gN / A0)
    g = A0 * x
    return math.sqrt(g * r_m) / 1000.0


def enclosed_baryon(r_kpc, model):
    """Standard baryonic enclosed mass in solar masses."""
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
    """Reduced chi-squared."""
    n = len(obs)
    if n == 0:
        return float('inf')
    return sum((v_pred[i] - obs[i][1])**2 / obs[i][2]**2
               for i in range(n)) / n


# =====================================================================
# THE MODELS
# =====================================================================

def model_standard_gfd(r_kpc, model):
    """Standard GFD: distributed baryonic mass into F(y)."""
    return v_from_mass(r_kpc, enclosed_baryon(r_kpc, model))


def model_field_profile(r_kpc, model, R_env):
    """
    The field object profile.

    The galaxy IS a field object. Its gravitational mass follows
    the self-similar distribution from the k=4 topology:

        M_grav(r) = M_total * (r / r_t)^{4/3}

    where r_t = 0.30 * R_env is the throat.

    This IS the rotation curve of a scaled-up electron.
    One profile. One equation. Same topology at every scale.
    """
    M_total = total_baryon(model)
    r_t = ALPHA_THROAT * R_env
    if r_kpc <= 0 or r_t <= 0:
        return 0.0

    # The field origin distributes mass as r^{k/d}
    M_grav = M_total * (r_kpc / r_t) ** P_EXPONENT
    return v_from_mass(r_kpc, M_grav)


def model_baryon_plus_field(r_kpc, model, R_env):
    """
    Baryonic mass PLUS the field origin's mass contribution.

    The Poisson equation says 4/13 of the coupling comes from
    the field origin. This means the field origin generates an
    effective mass = (4/13) * M_total distributed as r^{4/3}.

    Total: M_enc = M_baryon(r) + M_field_origin(r)
    """
    M_total = total_baryon(model)
    r_t = ALPHA_THROAT * R_env
    if r_kpc <= 0 or r_t <= 0:
        return 0.0

    M_baryon = enclosed_baryon(r_kpc, model)
    M_FO = ALPHA_THROAT * M_total * (r_kpc / r_t) ** P_EXPONENT
    return v_from_mass(r_kpc, M_baryon + M_FO)


def model_field_dominant(r_kpc, model, R_env):
    """
    The gravitational mass at any radius is the LARGER of:
    - The baryonic enclosed mass
    - The field origin's self-similar mass M_total * (r/r_t)^{4/3}

    At small r: baryonic disk/bulge dominates (Keplerian rise)
    At large r: field profile dominates (r^{1/3} rise)
    """
    M_total = total_baryon(model)
    r_t = ALPHA_THROAT * R_env
    if r_kpc <= 0 or r_t <= 0:
        return 0.0

    M_baryon = enclosed_baryon(r_kpc, model)
    M_field = M_total * (r_kpc / r_t) ** P_EXPONENT
    return v_from_mass(r_kpc, max(M_baryon, M_field))


# =====================================================================
# ANALYSIS
# =====================================================================
print("=" * 72)
print("ONE FIELD OBJECT. ONE PROFILE.")
print("=" * 72)
print()
print(f"Topology: k = {K} faces, d = {D} dimensions")
print(f"Throat fraction: 4/13 = {ALPHA_THROAT:.4f}")
print(f"Mass exponent: k/d = {P_EXPONENT:.4f}")
print(f"a0 = {A0:.4e} m/s^2")
print()
print("The field origin distributes mass as M(r) ~ r^{4/3}")
print("In deep field regime: v ~ M^{1/4} ~ r^{1/3}")
print()


for gal_name, gal in GALAXIES.items():
    model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    obs = gal["obs"]
    M_total = total_baryon(model)
    r_t = ALPHA_THROAT * R_env

    print("=" * 72)
    print(f"{gal_name}")
    print(f"  M_total = {M_total:.2e} M_sun")
    print(f"  R_env   = {R_env} kpc")
    print(f"  r_t     = {r_t:.2f} kpc (throat)")
    print()

    # Compute predictions for each model
    v_std = [model_standard_gfd(o[0], model) for o in obs]
    v_field = [model_field_profile(o[0], model, R_env) for o in obs]
    v_bpf = [model_baryon_plus_field(o[0], model, R_env) for o in obs]
    v_dom = [model_field_dominant(o[0], model, R_env) for o in obs]

    chi2_std = chi2(v_std, obs)
    chi2_field = chi2(v_field, obs)
    chi2_bpf = chi2(v_bpf, obs)
    chi2_dom = chi2(v_dom, obs)

    print(f"  chi2: Standard GFD      = {chi2_std:.1f}")
    print(f"  chi2: Field Profile Only = {chi2_field:.1f}")
    print(f"  chi2: Baryon + FO Mass   = {chi2_bpf:.1f}")
    print(f"  chi2: max(Baryon, Field) = {chi2_dom:.1f}")
    print()

    header = (f"{'r':>5} {'v_obs':>5} {'err':>4} {'v_std':>7} "
              f"{'v_field':>7} {'v_b+fo':>7} {'v_dom':>7} "
              f"{'M_b':>10} {'M_fo':>10}")
    print(header)
    print("-" * len(header))

    for i, o in enumerate(obs):
        r, vo, ve = o
        M_b = enclosed_baryon(r, model)
        M_fo = M_total * (r / r_t) ** P_EXPONENT
        print(f"{r:5.1f} {vo:5.0f} {ve:4.0f} {v_std[i]:7.1f} "
              f"{v_field[i]:7.1f} {v_bpf[i]:7.1f} {v_dom[i]:7.1f} "
              f"{M_b:10.2e} {M_fo:10.2e}")
    print()

    # What is the "dark matter fraction" at the envelope?
    M_grav_env = M_total * (R_env / r_t) ** P_EXPONENT
    dm_fraction = (M_grav_env - M_total) / M_grav_env
    print(f"  M_grav at R_env = {M_grav_env:.2e} M_sun")
    print(f"  'Dark matter' fraction = {dm_fraction:.1%}")
    print(f"  Baryon fraction = {1-dm_fraction:.1%}")
    print(f"  (Compare cosmic: ~16% baryon, ~84% dark)")
    print()


# =====================================================================
# The electron comparison
# =====================================================================
print("=" * 72)
print("THE ELECTRON AS A FIELD OBJECT")
print("=" * 72)
print()
print(f"  Horizon = R_e = {R_E:.4e} m")
print(f"  Throat  = 0.30 * R_e = {ALPHA_THROAT * R_E:.4e} m")
print(f"  Mass    = m_e = {M_E:.4e} kg")
print()

r_t_e = ALPHA_THROAT * R_E

# At the electron horizon: what is gN/a0?
gN_horizon = G * M_E / (R_E * R_E)
y_horizon = gN_horizon / A0
print(f"  gN at R_e = {gN_horizon:.4e} m/s^2")
print(f"  gN/a0 at R_e = {y_horizon:.4f}")
print(f"  Expected: 1/k^2 = 1/16 = {1.0/16:.4f}")
print()

# Profile: at the throat
gN_throat = G * M_E / (r_t_e * r_t_e)
y_throat = gN_throat / A0
x_throat = solve_x(y_throat)
print(f"  gN/a0 at throat = {y_throat:.4f}")
print(f"  x at throat = {x_throat:.4f}")
print()

# Now: for a galaxy, map the same dimensionless profile
print("=" * 72)
print("SCALING: Electron -> Galaxy (M33)")
print("=" * 72)
print()
print("The dimensionless profile x(r/R) is identical.")
print("Only the mass and scale change.")
print()

M33 = GALAXIES["M33"]
model_m33 = {k: M33[k] for k in ["bulge", "disk", "gas"]}
R_env_m33 = M33["R_env"]
r_t_m33 = ALPHA_THROAT * R_env_m33
M_total_m33 = total_baryon(model_m33)

# Dimensionless radii sampled: r/R_env from 0.05 to 1.2
xi_values = np.linspace(0.05, 1.2, 24)

print(f"{'r/R_env':>8} {'r_gal':>6} {'y_N':>10} {'x':>8} {'v_gal':>7} {'v_obs':>6}")
print("-" * 55)

obs_dict = {o[0]: o[1] for o in M33["obs"]}

for xi in xi_values:
    r_gal = xi * R_env_m33
    M_grav = M_total_m33 * (r_gal / r_t_m33) ** P_EXPONENT
    r_m = r_gal * KPC_TO_M
    M_kg = M_grav * M_SUN

    if r_m > 0:
        gN = G * M_kg / (r_m * r_m)
        y = gN / A0
        x = solve_x(y)
        v = math.sqrt(A0 * x * r_m) / 1000.0
    else:
        y, x, v = 0, 0, 0

    # Find closest observed point
    closest_r = min(obs_dict.keys(), key=lambda rr: abs(rr - r_gal))
    v_obs_str = ""
    if abs(closest_r - r_gal) < 0.5:
        v_obs_str = f"{obs_dict[closest_r]:6.0f}"

    print(f"{xi:8.3f} {r_gal:6.1f} {y:10.4f} {x:8.4f} {v:7.1f} {v_obs_str}")

print()

# =====================================================================
# v_flat for the field profile
# =====================================================================
v_flat = (G * M_total_m33 * M_SUN * A0) ** 0.25 / 1000.0
print(f"  v_flat (MOND asymptotic) = {v_flat:.1f} km/s")
print(f"  v(r_t) from field profile = {model_field_profile(r_t_m33, model_m33, R_env_m33):.1f} km/s")
print(f"  v(R_env) from field profile = {model_field_profile(R_env_m33, model_m33, R_env_m33):.1f} km/s")
print()


# =====================================================================
# PLOT
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('One Field Object, One Profile: v(r) from the k=4 topology',
             fontsize=13, fontweight='bold')

for col, (gal_name, gal) in enumerate(GALAXIES.items()):
    model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    obs = gal["obs"]
    r_t = ALPHA_THROAT * R_env

    r_dense = np.linspace(0.1, R_env * 1.2, 200)

    v_std_arr = [model_standard_gfd(r, model) for r in r_dense]
    v_field_arr = [model_field_profile(r, model, R_env) for r in r_dense]
    v_bpf_arr = [model_baryon_plus_field(r, model, R_env) for r in r_dense]
    v_dom_arr = [model_field_dominant(r, model, R_env) for r in r_dense]

    obs_r = [o[0] for o in obs]
    obs_v = [o[1] for o in obs]
    obs_e = [o[2] for o in obs]

    ax = axes[col]
    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt='o', color='orange',
                markersize=5, capsize=3, zorder=10, label='Observed')

    ax.plot(r_dense, v_std_arr, '-', color='purple', linewidth=1.5,
            label='Standard GFD (baryons)', alpha=0.8)
    ax.plot(r_dense, v_field_arr, '-', color='cyan', linewidth=2.5,
            label='Field Profile: M~r^{4/3}')
    ax.plot(r_dense, v_bpf_arr, '--', color='lime', linewidth=2,
            label='Baryon + FO mass', alpha=0.8)
    ax.plot(r_dense, v_dom_arr, ':', color='red', linewidth=2,
            label='max(Baryon, Field)')

    # Mark throat
    ax.axvline(r_t, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(r_t + 0.2, ax.get_ylim()[0] if col > 0 else 10,
            f'r_t={r_t:.1f}', fontsize=8, color='white', alpha=0.7)

    chi2_s = chi2([model_standard_gfd(o[0], model) for o in obs], obs)
    chi2_f = chi2([model_field_profile(o[0], model, R_env) for o in obs], obs)
    chi2_d = chi2([model_field_dominant(o[0], model, R_env) for o in obs], obs)

    ax.set_title(f'{gal_name}\n'
                 f'chi2: std={chi2_s:.1f}, field={chi2_f:.1f}, dom={chi2_d:.1f}',
                 fontsize=10)
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('field_profile.png', dpi=150, bbox_inches='tight')
print("Plot saved: field_profile.png")


# =====================================================================
# MASS PROFILE PLOT
# =====================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Mass Profiles: Baryonic vs Field Origin Distribution',
              fontsize=13, fontweight='bold')

for col, (gal_name, gal) in enumerate(GALAXIES.items()):
    model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    M_total = total_baryon(model)
    r_t = ALPHA_THROAT * R_env

    r_dense = np.linspace(0.1, R_env * 1.2, 200)

    m_baryon = [enclosed_baryon(r, model) for r in r_dense]
    m_field = [M_total * (r / r_t) ** P_EXPONENT for r in r_dense]
    m_fo = [ALPHA_THROAT * M_total * (r / r_t) ** P_EXPONENT for r in r_dense]
    m_bpf = [m_baryon[i] + m_fo[i] for i in range(len(r_dense))]
    m_dom = [max(m_baryon[i], m_field[i]) for i in range(len(r_dense))]

    ax = axes2[col]
    ax.semilogy(r_dense, m_baryon, '-', color='red', linewidth=1.5,
                label='M_baryon(r)')
    ax.semilogy(r_dense, m_field, '-', color='cyan', linewidth=2.5,
                label='M_field = M_tot * (r/r_t)^{4/3}')
    ax.semilogy(r_dense, m_fo, '--', color='lime', linewidth=1.5,
                label='M_FO = (4/13)*M_tot*(r/r_t)^{4/3}')

    ax.axvline(r_t, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title(f'{gal_name}', fontsize=11)
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('Enclosed Mass (M_sun)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('field_profile_mass.png', dpi=150, bbox_inches='tight')
print("Plot saved: field_profile_mass.png")

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print()
print("The field profile M(r) = M_total * (r/r_t)^{4/3} produces")
print("v ~ r^{1/3} in the deep field regime. This is the")
print("vortex distribution from the merkaba structure.")
print()
print(f"{'Galaxy':>12} {'chi2_std':>9} {'chi2_field':>10} {'chi2_dom':>9}")
print("-" * 45)
for gal_name, gal in GALAXIES.items():
    model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    obs = gal["obs"]
    c_s = chi2([model_standard_gfd(o[0], model) for o in obs], obs)
    c_f = chi2([model_field_profile(o[0], model, R_env) for o in obs], obs)
    c_d = chi2([model_field_dominant(o[0], model, R_env) for o in obs], obs)
    print(f"{gal_name:>12} {c_s:9.1f} {c_f:10.1f} {c_d:9.1f}")
