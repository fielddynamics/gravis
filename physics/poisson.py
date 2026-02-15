"""
GFD Poisson: unified manifold operator composed with the covariant completion.

From Section IX of the Dual Tetrad Gravity paper, the derived Poisson
equation with field closure normalization is:

    nabla^2 Phi = k*pi * (k^2+1)/(k*d+1) * G_bare * rho

Substituting k=4, d=3:

    nabla^2 Phi = 4*pi * (17/13) * G_bare * rho

The unified manifold upgrades this to a modified Poisson operator:

    nabla . (kappa(r) * nabla Phi) = 4*pi * G_eff * rho

where kappa is the Lamb-Oseen vortex-style distribution:

    kappa(r) = 1 - exp(-r^2 / r_t^2)
    r_t = 0.30 * R_env

This operator COMPOSES with the existing covariant completion. Since
our covariant engine returns velocity v_cov(r), the composition in
velocity space is:

    v_poisson(r) = v_cov(r) / sqrt(kappa(r))

At large r: kappa -> 1, v_poisson -> v_cov (flat rotation curves preserved)
At the throat (r_t): kappa = 0.632, boost = 1/sqrt(0.632) = 1.26x
At r -> 0: kappa -> 0, strong enhancement (capped numerically)

The 0.30 throat ratio is the same topological constant from coupled
tetrahedral face transport weighting. R_env is the galactic radius
(gravitational horizon), an observable, not a fitted parameter.

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x, velocity as gfd_velocity
from physics.manifold import ALPHA_THROAT

# ---------------------------------------------------------------------------
# Poisson normalization from field closure (Section IX, Equation 74)
# G_eff = (17/13) * G_bare. If G = G_eff (the measured constant),
# then this factor is already absorbed. The kappa operator is the
# new physics beyond the standard Poisson form.
# ---------------------------------------------------------------------------
G_EFF_RATIO = 17.0 / 13.0

# Minimum kappa for numerical stability. Caps the maximum velocity
# enhancement at 1/sqrt(KAPPA_FLOOR). This is numerical only, not
# a physics parameter.
KAPPA_FLOOR = 1e-4


def kappa(r_kpc, galactic_radius_kpc):
    """
    Unified manifold screening function (Lamb-Oseen vortex distribution).

    kappa(r) = 1 - exp(-r^2 / r_t^2)
    r_t = 0.30 * R_env

    Properties:
        kappa(0) = 0        (full manifold coupling at center)
        kappa(r_t) = 0.632  (throat boundary)
        kappa(inf) = 1      (standard gravity)

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    galactic_radius_kpc : float
        Galactic gravitational horizon in kiloparsecs.

    Returns
    -------
    float
        Screening value in [0, 1].
    """
    if galactic_radius_kpc <= 0 or r_kpc <= 0:
        return 0.0
    r_t = ALPHA_THROAT * galactic_radius_kpc
    if r_t <= 0:
        return 1.0
    x = r_kpc / r_t
    return 1.0 - math.exp(-x * x)


def velocity(r_kpc, m_solar, galactic_radius_kpc, accel_ratio=1.0):
    """
    GFD Poisson circular velocity: covariant completion composed with kappa.

    Computes the standard GFD covariant velocity, then applies the
    unified manifold composition:

        v_poisson(r) = v_cov(r) / sqrt(kappa(r))

    At large r: kappa -> 1, returns standard GFD velocity.
    At small r: kappa < 1, velocity is enhanced.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed baryonic mass at r_kpc in solar masses.
    galactic_radius_kpc : float
        Galactic gravitational horizon in kiloparsecs.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0).

    Returns
    -------
    float
        Circular velocity in km/s.
    """
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0

    # Step 1: get the base covariant velocity
    v_cov = gfd_velocity(r_kpc, m_solar, accel_ratio)

    if v_cov <= 0:
        return 0.0

    # Step 2: compute kappa and apply composition
    k = kappa(r_kpc, galactic_radius_kpc)
    k_safe = max(k, KAPPA_FLOOR)

    # Step 3: compose
    v_poisson = v_cov / math.sqrt(k_safe)

    return v_poisson
