"""
GFD Manifold: smooth vortex-distributed weighting for the covariant
field equation.

A galaxy is a gravitational vortex. The galactic radius defines the
outer horizon. The manifold effect is distributed smoothly from center
to horizon, just as a water vortex distributes angular momentum from
core to edge with no abrupt boundary.

The manifold modifies the base GFD acceleration:

    g_tot(r) = g_base(r) * [1 + W(r)]
    v(r) = sqrt(r * g_tot(r))

where W(r) is a smooth (C1) Hermite polynomial over [0, R_galaxy]:

    W(r) = 1 - 3*(r/R)^2 + 2*(r/R)^3

This produces:
    W(0) = 1.0   at center (full manifold coupling, doubles acceleration)
    W(0.30*R) = 0.784   at the throat (78% coupling)
    W(0.50*R) = 0.500   halfway (50% coupling)
    W(R) = 0.0   at the horizon (standard GFD)

The 30% throat is a characteristic scale on this smooth curve, not a
cutoff. It marks where the collective coupling is about 78% of its
central value, corresponding to the coupled tetrahedral face transport
ratio (0.30), which matches the proton-to-electron radius ratio.

Zero free parameters:
  - The galactic radius is an observable (like mass), not a fitted parameter
  - The amplitude is fixed: one additional copy of g_base at center
  - The shape is the standard Hermite smoothstep (determined by C1 conditions)

All galaxies have this manifold, whether a bulge is defined or not.

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x

# ---------------------------------------------------------------------------
# Topological constant (derived, not fitted)
# ---------------------------------------------------------------------------

# Throat-to-horizon ratio from coupled tetrahedral face transport.
# The throat is where W = 0.784 on the smooth distribution.
ALPHA_THROAT = 0.30


def throat_radius(galactic_radius_kpc):
    """
    Compute the transport throat radius from the galactic horizon.

    Parameters
    ----------
    galactic_radius_kpc : float
        Galactic radius (horizon scale) in kiloparsecs.

    Returns
    -------
    float
        Throat radius r_throat in kiloparsecs.
    """
    return ALPHA_THROAT * galactic_radius_kpc


def core_weight(r_kpc, galactic_radius_kpc):
    """
    Smooth manifold weighting distributed over the full galaxy.

    Like a water vortex, the manifold effect is strongest at the center
    and tapers smoothly to zero at the galactic horizon. There is no
    hard cutoff. The throat (30% of the horizon) is simply the point
    on this smooth curve where the weight is approximately 0.784.

    W(r) is a C1-continuous Hermite smoothstep over [0, R_galaxy]:

        W(r) = 1 - 3*x^2 + 2*x^3,   x = r / R_galaxy,  0 <= x <= 1
        W(r) = 0,                     x > 1

    At the center:                 W(0) = 1.0   (full manifold)
    At the throat (x=0.30):        W = 0.784    (78%)
    At half the galaxy (x=0.50):   W = 0.500    (50%)
    At the galactic horizon (x=1): W = 0.0      (standard GFD)

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    galactic_radius_kpc : float
        Galactic gravitational horizon in kiloparsecs.

    Returns
    -------
    float
        Manifold weight in [0, 1].
    """
    if galactic_radius_kpc <= 0 or r_kpc >= galactic_radius_kpc:
        return 0.0
    if r_kpc <= 0:
        return 1.0
    x = r_kpc / galactic_radius_kpc
    return 1.0 - 3.0 * x * x + 2.0 * x * x * x


def velocity(r_kpc, m_solar, galactic_radius_kpc, accel_ratio=1.0):
    """
    GFD Manifold circular velocity with smooth vortex weighting.

    Computes the base GFD acceleration, then boosts it smoothly:

        g_base = a0 * x   (where x^2/(1+x) = G*M / (r^2 * a0))
        g_tot  = g_base * (1 + W(r))
        v      = sqrt(r * g_tot)

    The boost tapers from sqrt(2) at center to 1.0 at the horizon.

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

    # Convert to SI
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN

    # Base GFD field equation
    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    x = solve_x(gN / a0_eff)
    g_base = a0_eff * x

    # Smooth manifold weighting over the full galaxy
    W = core_weight(r_kpc, galactic_radius_kpc)
    g_tot = g_base * (1.0 + W)

    # Velocity from boosted acceleration
    v = math.sqrt(g_tot * r)
    return v / 1000.0  # m/s -> km/s
