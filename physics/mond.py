"""
Classical MOND (Modified Newtonian Dynamics) -- "simple" interpolating function.

Interpolating function: mu(x) = x / sqrt(1 + x^2)  (Bekenstein & Milgrom)
Field equation:          mu(a/a0) * a = a_N

This uses the SAME topologically derived a0 as DTG (zero free parameters).
The difference is purely in the interpolating function shape.
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, A0


def mu_mond(x):
    """Classical MOND interpolating function: mu(x) = x / sqrt(1 + x^2)."""
    return x / math.sqrt(1.0 + x * x)


def solve_x(gN_over_a0):
    """
    Solve the classical MOND field equation mu(x)*x = y_N for x.

    With mu(x) = x/sqrt(1+x^2), solving gives:
    x^2 = (y_N^2 + sqrt(y_N^4 + 4*y_N^2)) / 2

    Parameters
    ----------
    gN_over_a0 : float
        Newtonian acceleration divided by a0.

    Returns
    -------
    float
        The true gravitational acceleration divided by a0.
    """
    y_N = gN_over_a0
    if y_N < 1e-30:
        return 0.0
    y2 = y_N * y_N
    discriminant = y2 * y2 + 4.0 * y2
    x_squared = (y2 + math.sqrt(discriminant)) / 2.0
    return math.sqrt(x_squared)


def velocity(r_kpc, m_solar, accel_ratio=1.0):
    """
    Classical MOND circular velocity at radius r_kpc for enclosed mass m_solar.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed mass in solar masses.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0).

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
