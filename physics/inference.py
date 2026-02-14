"""
Mass inference: inverse problem solver.

Given observed velocity v at radius r, infer the enclosed baryonic mass
using the DTG field equation (AQUAL formulation).
"""

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import mu


def infer_mass(r_kpc, v_km_s, accel_ratio=1.0):
    """
    Infer enclosed baryonic mass from observed rotation velocity.

    Uses the DTG field equation in reverse:
      v^2/r = g_eff  =>  x = g_eff/a0  =>  g_N = a0 * mu(x) * x  =>  M = g_N * r^2 / G

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    v_km_s : float
        Observed circular velocity in km/s.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0).

    Returns
    -------
    float
        Inferred enclosed mass in solar masses.
    """
    r = r_kpc * KPC_TO_M
    v = v_km_s * 1000.0  # km/s -> m/s

    if r <= 0 or v <= 0:
        return 0.0

    # From v^2 = g_eff * r => g_eff = v^2 / r
    g_eff = (v * v) / r

    a0_eff = A0 * accel_ratio

    # x = g_eff / a0
    x = g_eff / a0_eff

    # From field equation: mu(x) * x * a0 = g_N
    mu_x = mu(x)
    gN = a0_eff * mu_x * x

    # From g_N = G*M/r^2 => M = g_N * r^2 / G
    M = (gN * r * r) / G

    return M / M_SUN
