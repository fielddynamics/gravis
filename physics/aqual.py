"""
Dual Tetrad Gravity (DTG) -- AQUAL field equation solver.

Constitutive law: mu(x) = x / (1 + x)
Field equation:   mu(g/a0) * g = g_N
Analytic solution: x = (y_N + sqrt(y_N^2 + 4*y_N)) / 2, where y_N = g_N/a0

This is the covariant scalar-tensor completion of Dual Tetrad Gravity.
The interpolating function mu(x) = x/(1+x) is NOT empirical -- it is
derived from the topological structure of the tetrad field.
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, A0


def mu(x):
    """Constitutive law: mu(x) = x / (1 + x)."""
    return x / (1.0 + x)


def solve_x(gN_over_a0):
    """
    Solve the AQUAL field equation mu(x)*x = y_N for x.

    With mu(x) = x/(1+x), the equation x^2/(1+x) = y_N has the
    analytic solution: x = (y_N + sqrt(y_N^2 + 4*y_N)) / 2.

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
    DTG circular velocity at radius r_kpc for enclosed mass m_solar.

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
