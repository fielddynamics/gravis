"""
Gravity Field Dynamics -- covariant field equation solver.

Scalar Lagrangian:  F(y) = y/2 - sqrt(y) + ln(1 + sqrt(y))
  where y = |grad Phi|^2 / a0^2

The Euler-Lagrange equation from this Lagrangian, in spherical symmetry,
reduces to the algebraic field equation:

    x^2 / (1 + x) = g_N / a0

where x = g/a0 is the dimensionless gravitational field strength and
g_N = GM/r^2 is the Newtonian acceleration.

Analytic solution (quadratic in x):
    x = (y_N + sqrt(y_N^2 + 4*y_N)) / 2,   y_N = g_N / a0

The Lagrangian F(y) is uniquely determined by the coupling polynomial
f(k) = 1 + k + k^2 of the stellated octahedron (dual tetrad topology).
Its three terms map to the three structural levels: k^2 = 16 (quadratic),
k = 4 (square root), k^0 = 1 (logarithm / Field Origin).

Nothing here is empirical.  The Lagrangian, the field equation, and the
acceleration scale a0 are all derived from the topology.
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, A0


def solve_x(gN_over_a0):
    """
    Solve the covariant field equation for the true gravitational field.

    Given the Newtonian acceleration ratio y_N = g_N / a0, solve:

        x^2 / (1 + x) = y_N

    This is a quadratic: x^2 - y_N*x - y_N = 0, with the physical root:

        x = (y_N + sqrt(y_N^2 + 4*y_N)) / 2

    Parameters
    ----------
    gN_over_a0 : float
        Newtonian acceleration divided by a0.

    Returns
    -------
    float
        The true gravitational acceleration divided by a0 (i.e., g/a0).
    """
    y_N = gN_over_a0
    if y_N < 1e-30:
        return 0.0
    return (y_N + math.sqrt(y_N * y_N + 4.0 * y_N)) / 2.0


def velocity(r_kpc, m_solar, accel_ratio=1.0):
    """
    GFD circular velocity at radius r_kpc for enclosed mass m_solar.

    Computation chain (forward problem):
      1. g_N = G*M / r^2            (Newtonian acceleration)
      2. y_N = g_N / a0             (dimensionless)
      3. Solve x^2/(1+x) = y_N     (field equation from the Lagrangian)
      4. g = a0 * x                 (true gravitational acceleration)
      5. v = sqrt(g * r)            (circular orbit)

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed mass in solar masses.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0). Allows exploring acceleration regimes.

    Returns
    -------
    float
        Circular velocity in km/s.
    """
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN
    if r <= 0 or M <= 0:
        return 0.0
    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    x = solve_x(gN / a0_eff)
    g_eff = a0_eff * x
    v = math.sqrt(g_eff * r)
    return v / 1000.0  # m/s -> km/s
