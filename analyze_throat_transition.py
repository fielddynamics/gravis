"""
Structural State Transition at the Throat.

Inside the throat (r < r_t):
  Structural state s = 0 (the Field Origin).
  Both tetrahedra meet. beta_1 = beta_3 = 0.
  The 17/13 coupling ratio has not yet differentiated.
  The coupling is the "1" in f(k) = 1 + k + k^2.
  -> Standard GFD with baryonic mass. Already fits perfectly.

Outside the throat (r > r_t):
  Structural state s = -1 (spatial projection).
  The full 17/13 ratio activates.
  The 4/13 structural excess propagates from the field origin
  into space, growing as r^{4/3} from the throat outward.
  -> Standard GFD + structural mass.

The throat is the boundary: r_t = (4/13) * R_env.
This is not a stitching parameter. It is where the
structural state changes from s=0 to s=-1.

Every star also has a 30% field origin. We don't have
missing mass; we have missing structure. The structural
contribution only appears outside the galactic throat.
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
ALPHA_THROAT = 4.0 / 13.0  # 0.3077
P = float(K) / float(D)    # 4/3


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


def solve_x(y_N):
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def v_from_mass(r_kpc, m_solar):
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = m_solar * M_SUN
    gN = G * M_kg / (r_m * r_m)
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
# THE THROAT TRANSITION MODELS
# =====================================================================

def model_standard(r_kpc, model):
    """Standard GFD: baryonic mass only."""
    return v_from_mass(r_kpc, enclosed_baryon(r_kpc, model))


def model_throat_A(r_kpc, model, R_env):
    """
    Throat transition: structural mass grows from r_t outward.

    Inside throat: M_enc = M_baryon (standard GFD)
    Outside throat: M_enc = M_baryon + M_structural

    M_structural(r) = M_total * ((r/r_t)^{4/3} - 1)

    Starts at 0 at r_t, grows as r^{4/3} beyond the throat.
    This is the full field profile minus the throat normalization.
    """
    r_t = ALPHA_THROAT * R_env
    M_b = enclosed_baryon(r_kpc, model)
    if r_kpc <= r_t or r_t <= 0:
        return v_from_mass(r_kpc, M_b)

    M_total = total_baryon(model)
    M_struct = M_total * ((r_kpc / r_t) ** P - 1.0)
    return v_from_mass(r_kpc, M_b + M_struct)


def model_throat_B(r_kpc, model, R_env):
    """
    Throat transition with 4/13 coupling fraction.

    M_structural(r) = (4/13) * M_total * ((r/r_t)^{4/3} - 1)

    The 4/13 is the structural excess: 17 - 13 = 4 modes
    out of 13 observable modes.
    """
    r_t = ALPHA_THROAT * R_env
    M_b = enclosed_baryon(r_kpc, model)
    if r_kpc <= r_t or r_t <= 0:
        return v_from_mass(r_kpc, M_b)

    M_total = total_baryon(model)
    M_struct = ALPHA_THROAT * M_total * ((r_kpc / r_t) ** P - 1.0)
    return v_from_mass(r_kpc, M_b + M_struct)


def model_throat_C(r_kpc, model, R_env):
    """
    Throat transition: each star's 30% structure is missing
    from M_baryon. Outside the throat, the full stellar
    structural contribution activates.

    Inside: M_enc = M_baryon (bare stellar mass)
    Outside: M_enc = (17/13) * M_baryon + field profile growth

    The 17/13 accounts for each star's 30% field origin.
    The field profile growth accounts for the galactic-scale
    self-similar distribution.
    """
    r_t = ALPHA_THROAT * R_env
    M_b = enclosed_baryon(r_kpc, model)
    if r_kpc <= r_t or r_t <= 0:
        return v_from_mass(r_kpc, M_b)

    M_total = total_baryon(model)
    # Each star's structural contribution: 4/13 of M_baryon
    M_stellar_struct = (4.0 / 13.0) * M_b
    # Galactic field origin profile growing from throat
    M_galactic_struct = M_total * ((r_kpc / r_t) ** P - 1.0)
    return v_from_mass(r_kpc, M_b + M_stellar_struct + M_galactic_struct)


def model_throat_D(r_kpc, model, R_env):
    """
    Smooth transition using Hermite weight at the throat.

    Instead of a hard boundary, the structural coupling
    activates smoothly from 0 at r_t to 1 at R_env.

    W(x) = 3x^2 - 2x^3 where x = (r - r_t)/(R_env - r_t)

    M_enc = M_baryon + W(x) * M_total * ((r/r_t)^{4/3} - 1)
    """
    r_t = ALPHA_THROAT * R_env
    M_b = enclosed_baryon(r_kpc, model)
    if r_kpc <= r_t or r_t <= 0:
        return v_from_mass(r_kpc, M_b)

    M_total = total_baryon(model)
    x = min(1.0, (r_kpc - r_t) / (R_env - r_t))
    W = 3.0 * x * x - 2.0 * x * x * x
    M_struct = W * M_total * ((r_kpc / r_t) ** P - 1.0)
    return v_from_mass(r_kpc, M_b + M_struct)


def model_throat_E(r_kpc, model, R_env):
    """
    Structural mass measured from the throat (distance from r_t).

    M_structural = M_total * ((r - r_t)/r_t)^{4/3} for r > r_t

    The structural field has propagated a distance (r - r_t)
    from the throat. Normalized by the throat scale.
    """
    r_t = ALPHA_THROAT * R_env
    M_b = enclosed_baryon(r_kpc, model)
    if r_kpc <= r_t or r_t <= 0:
        return v_from_mass(r_kpc, M_b)

    M_total = total_baryon(model)
    M_struct = M_total * ((r_kpc - r_t) / r_t) ** P
    return v_from_mass(r_kpc, M_b + M_struct)


# =====================================================================
# All models
# =====================================================================
MODELS = {
    "Standard GFD": lambda r, m, R: model_standard(r, m),
    "A: M_tot*((r/rt)^p-1)": model_throat_A,
    "B: (4/13)*A": model_throat_B,
    "C: stellar+galactic": model_throat_C,
    "D: smooth Hermite": model_throat_D,
    "E: M_tot*((r-rt)/rt)^p": model_throat_E,
}


# =====================================================================
# RESULTS
# =====================================================================
print("=" * 80)
print("STRUCTURAL STATE TRANSITION AT THE THROAT")
print("=" * 80)
print()
print(f"Throat: r_t = (4/13) * R_env = {ALPHA_THROAT:.4f} * R_env")
print(f"Inside throat (s=0):  standard GFD with baryonic mass")
print(f"Outside throat (s=-1): structural mass activates, grows as r^{{4/3}}")
print()

# Summary table
print(f"{'Model':>25}", end="")
for gname in GALAXIES:
    print(f"  {gname:>10}", end="")
print()
print("-" * (25 + 12 * len(GALAXIES)))

for mname, mfunc in MODELS.items():
    print(f"{mname:>25}", end="")
    for gname, gal in GALAXIES.items():
        model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
        R_env = gal["R_env"]
        obs = gal["obs"]
        v_pred = [mfunc(o[0], model, R_env) for o in obs]
        c2 = chi2(v_pred, obs)
        print(f"  {c2:10.1f}", end="")
    print()

print()

# =====================================================================
# Detailed point-by-point for M33
# =====================================================================
gal = GALAXIES["M33"]
model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
R_env = gal["R_env"]
r_t = ALPHA_THROAT * R_env
obs = gal["obs"]

print("=" * 80)
print(f"M33 DETAIL (throat at r_t = {r_t:.2f} kpc)")
print("=" * 80)
print()

header = (f"{'r':>5} {'obs':>5} {'err':>4} {'std':>6} "
          f"{'A':>6} {'B':>6} {'C':>6} {'D':>6} {'E':>6} {'zone':>8}")
print(header)
print("-" * len(header))

for o in obs:
    r, vo, ve = o
    v_std = model_standard(r, model)
    v_a = model_throat_A(r, model, R_env)
    v_b = model_throat_B(r, model, R_env)
    v_c = model_throat_C(r, model, R_env)
    v_d = model_throat_D(r, model, R_env)
    v_e = model_throat_E(r, model, R_env)
    zone = "THROAT" if r <= r_t else "OUTER"
    print(f"{r:5.1f} {vo:5.0f} {ve:4.0f} {v_std:6.1f} "
          f"{v_a:6.1f} {v_b:6.1f} {v_c:6.1f} {v_d:6.1f} {v_e:6.1f} {zone:>8}")

print()

# =====================================================================
# Detailed for Milky Way
# =====================================================================
gal = GALAXIES["Milky Way"]
model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
R_env = gal["R_env"]
r_t = ALPHA_THROAT * R_env
obs = gal["obs"]

print("=" * 80)
print(f"MILKY WAY DETAIL (throat at r_t = {r_t:.2f} kpc)")
print("=" * 80)
print()

header = (f"{'r':>5} {'obs':>5} {'err':>4} {'std':>6} "
          f"{'A':>6} {'B':>6} {'C':>6} {'D':>6} {'E':>6} {'zone':>8}")
print(header)
print("-" * len(header))

for o in obs:
    r, vo, ve = o
    v_std = model_standard(r, model)
    v_a = model_throat_A(r, model, R_env)
    v_b = model_throat_B(r, model, R_env)
    v_c = model_throat_C(r, model, R_env)
    v_d = model_throat_D(r, model, R_env)
    v_e = model_throat_E(r, model, R_env)
    zone = "THROAT" if r <= r_t else "OUTER"
    print(f"{r:5.1f} {vo:5.0f} {ve:4.0f} {v_std:6.1f} "
          f"{v_a:6.1f} {v_b:6.1f} {v_c:6.1f} {v_d:6.1f} {v_e:6.1f} {zone:>8}")

print()

# =====================================================================
# Detailed for DDO 154
# =====================================================================
gal = GALAXIES["DDO 154"]
model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
R_env = gal["R_env"]
r_t = ALPHA_THROAT * R_env
obs = gal["obs"]

print("=" * 80)
print(f"DDO 154 DETAIL (throat at r_t = {r_t:.2f} kpc)")
print("=" * 80)
print()

header = (f"{'r':>5} {'obs':>5} {'err':>4} {'std':>6} "
          f"{'A':>6} {'B':>6} {'C':>6} {'D':>6} {'E':>6} {'zone':>8}")
print(header)
print("-" * len(header))

for o in obs:
    r, vo, ve = o
    v_std = model_standard(r, model)
    v_a = model_throat_A(r, model, R_env)
    v_b = model_throat_B(r, model, R_env)
    v_c = model_throat_C(r, model, R_env)
    v_d = model_throat_D(r, model, R_env)
    v_e = model_throat_E(r, model, R_env)
    zone = "THROAT" if r <= r_t else "OUTER"
    print(f"{r:5.1f} {vo:5.0f} {ve:4.0f} {v_std:6.1f} "
          f"{v_a:6.1f} {v_b:6.1f} {v_c:6.1f} {v_d:6.1f} {v_e:6.1f} {zone:>8}")

print()


# =====================================================================
# MASS BREAKDOWN for best model
# =====================================================================
print("=" * 80)
print("MASS BREAKDOWN: Model A for M33")
print("=" * 80)
print()

gal = GALAXIES["M33"]
model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
R_env = gal["R_env"]
M_total = total_baryon(model)
r_t = ALPHA_THROAT * R_env

print(f"{'r':>5} {'M_baryon':>12} {'M_struct':>12} {'M_total':>12} "
      f"{'struct%':>8} {'v_pred':>7} {'v_obs':>6}")

for r in [1, 2, 3, 4, r_t, 5, 6, 8, 10, 12, 14, 16, 18, 20]:
    M_b = enclosed_baryon(r, model)
    if r <= r_t:
        M_s = 0.0
    else:
        M_s = M_total * ((r / r_t) ** P - 1.0)
    M_tot = M_b + M_s
    pct = 100.0 * M_s / M_tot if M_tot > 0 else 0
    v = v_from_mass(r, M_tot)

    obs_v = ""
    for o in gal["obs"]:
        if abs(o[0] - r) < 0.1:
            obs_v = f"{o[1]:6.0f}"

    marker = " <-- THROAT" if abs(r - r_t) < 0.1 else ""
    print(f"{r:5.1f} {M_b:12.2e} {M_s:12.2e} {M_tot:12.2e} "
          f"{pct:7.1f}% {v:7.1f} {obs_v}{marker}")

print()


# =====================================================================
# PLOT
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Structural State Transition at the Throat',
             fontsize=14, fontweight='bold')

colors = {
    "Standard GFD": ('purple', '-', 1.5),
    "A: M_tot*((r/rt)^p-1)": ('cyan', '-', 2.5),
    "B: (4/13)*A": ('lime', '--', 2.0),
    "E: M_tot*((r-rt)/rt)^p": ('red', ':', 2.0),
}

for col, (gname, gal) in enumerate(GALAXIES.items()):
    model = {k: gal[k] for k in ["bulge", "disk", "gas"]}
    R_env = gal["R_env"]
    r_t = ALPHA_THROAT * R_env
    obs = gal["obs"]

    r_dense = np.linspace(0.1, R_env * 1.2, 300)

    ax = axes[col]

    # Observed data
    obs_r = [o[0] for o in obs]
    obs_v = [o[1] for o in obs]
    obs_e = [o[2] for o in obs]
    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt='o', color='orange',
                markersize=5, capsize=3, zorder=10, label='Observed')

    # Model curves
    for mname, (clr, ls, lw) in colors.items():
        mfunc = MODELS[mname]
        v_arr = [mfunc(r, model, R_env) for r in r_dense]
        c2 = chi2([mfunc(o[0], model, R_env) for o in obs], obs)
        label = f'{mname} ({c2:.1f})'
        ax.plot(r_dense, v_arr, linestyle=ls, color=clr,
                linewidth=lw, label=label, alpha=0.9)

    # Mark throat
    ax.axvline(r_t, color='yellow', linestyle='--', alpha=0.6,
               linewidth=1.5, label=f'Throat r_t={r_t:.1f}')

    # Shade throat region
    ax.axvspan(0, r_t, alpha=0.08, color='yellow')
    mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 if col > 0 else 30
    ax.text(r_t * 0.5, mid_y, 's=0\nField\nOrigin',
            fontsize=8, ha='center', color='yellow', alpha=0.6)
    ax.text(r_t * 1.5, mid_y, 's=-1\nSpatial\nProjection',
            fontsize=8, ha='center', color='yellow', alpha=0.6)

    ax.set_title(gname, fontsize=12)
    ax.set_xlabel('r (kpc)')
    if col == 0:
        ax.set_ylabel('v (km/s)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('throat_transition.png', dpi=150, bbox_inches='tight')
print("Plot saved: throat_transition.png")


# =====================================================================
# KEY INSIGHT
# =====================================================================
print()
print("=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print()
print("The throat at r_t = 0.30 * R_env is not an arbitrary boundary.")
print("It is where the structural state changes from s=0 to s=-1.")
print()
print("Inside s=0:  beta_1 = beta_3 = 0. The 17/13 split")
print("             has not differentiated. Standard GFD is exact.")
print()
print("Outside s=-1: The full coupling activates.")
print("              Structural mass grows as r^{4/3} from the throat.")
print("              This is the 'missing structure', not 'missing mass'.")
print()
print("Not stitched. The same F(y) equation everywhere.")
print("The only thing that changes is the source: inside the throat,")
print("the source is baryonic mass. Outside, the source includes")
print("the structural contribution that has propagated from the")
print("field origin into the spatial projection.")
