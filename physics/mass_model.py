"""
Distributed mass model: Hernquist bulge + exponential stellar disk + gas disk.

Computes enclosed mass M(<r) for a 3-component baryonic mass distribution.
Matches the methodology in run_all_covariant_predictions.py (GD-2 test).

Components:
  1. Stellar bulge: Hernquist (1990 ApJ 356 359) profile
     M(<r) = M_total * r^2 / (r + a)^2
  2. Stellar disk: Spherical approximation of Freeman (1970) exponential disk
     M(<r) = M_total * [1 - (1 + r/Rd) * exp(-r/Rd)]
  3. Gas disk (HI + H2 + He): Same exponential form, typically more extended

All masses in solar masses, all radii in kiloparsecs.
"""

import math


def enclosed_mass(r_kpc, mass_model):
    """
    Compute enclosed baryonic mass at galactocentric radius r_kpc.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    mass_model : dict
        Dictionary with keys 'bulge', 'disk', 'gas', each containing:
          - bulge: {'M': total_mass_Msun, 'a': scale_radius_kpc}
          - disk:  {'M': total_mass_Msun, 'Rd': scale_length_kpc}
          - gas:   {'M': total_mass_Msun, 'Rd': scale_length_kpc}

    Returns
    -------
    float
        Enclosed mass in solar masses.
    """
    if r_kpc <= 0:
        return 0.0

    m_enc = 0.0
    r = r_kpc

    # Hernquist bulge: M(<r) = M * r^2 / (r + a)^2
    bulge = mass_model.get("bulge")
    if bulge and bulge.get("M", 0) > 0:
        a = bulge["a"]
        m_enc += bulge["M"] * r * r / ((r + a) * (r + a))

    # Exponential stellar disk: M(<r) = M * [1 - (1 + r/Rd) * exp(-r/Rd)]
    disk = mass_model.get("disk")
    if disk and disk.get("M", 0) > 0:
        x = r / disk["Rd"]
        m_enc += disk["M"] * (1.0 - (1.0 + x) * math.exp(-x))

    # Exponential gas disk: same functional form
    gas = mass_model.get("gas")
    if gas and gas.get("M", 0) > 0:
        x = r / gas["Rd"]
        m_enc += gas["M"] * (1.0 - (1.0 + x) * math.exp(-x))

    return m_enc


def total_mass(mass_model):
    """Return total baryonic mass (all components) in solar masses."""
    total = 0.0
    for component in ("bulge", "disk", "gas"):
        comp = mass_model.get(component)
        if comp and comp.get("M", 0) > 0:
            total += comp["M"]
    return total
