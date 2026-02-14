"""
Newtonian gravity: circular velocity from enclosed mass.

v_c(r) = sqrt(G * M(<r) / r)
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M


def velocity(r_kpc, m_solar):
    """
    Newtonian circular velocity at radius r_kpc for enclosed mass m_solar.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed mass in solar masses.

    Returns
    -------
    float
        Circular velocity in km/s.
    """
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN
    if r <= 0 or M <= 0:
        return 0.0
    v = math.sqrt(G * M / r)
    return v / 1000.0  # m/s -> km/s
