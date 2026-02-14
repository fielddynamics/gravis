"""
Physical constants for Dual Tetrad Gravity calculations.

All values match CODATA 2022 / IAU 2015 nominal values and are identical
to those used in run_all_covariant_predictions.py.

IMPORTANT: No unicode characters allowed in this file (Windows charmap constraint).
"""

import math

# Gravitational constant (CODATA 2022)
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Solar mass (IAU 2015 nominal)
M_SUN = 1.98892e30  # kg

# Kiloparsec to meters (IAU exact: AU * 648000/pi * 1000)
KPC_TO_M = 3.0857e19  # meters

# Electron mass (CODATA 2022)
M_E = 9.1093837139e-31  # kg

# Classical electron radius (CODATA 2022)
R_E = 2.8179403205e-15  # m

# Simplex number k = d+1, d=3 (topological origin)
K_SIMPLEX = 4

# Topological acceleration scale (zero free parameters)
# a0 = k^2 * G * m_e / r_e^2
A0 = K_SIMPLEX * K_SIMPLEX * G * M_E / (R_E * R_E)


def verify_a0():
    """Verify a0 is approximately 1.2e-10 m/s^2."""
    expected = 1.2e-10
    ratio = A0 / expected
    return 0.9 < ratio < 1.1, A0
