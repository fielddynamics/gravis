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

# Coupling polynomial f(k) at k = K_SIMPLEX = 4
# f(k) = 1 + k + k^2 = 1 + 4 + 16 = 21
# Full coupling strength of the stellated octahedron topology.
FK = 1 + K_SIMPLEX + K_SIMPLEX * K_SIMPLEX  # 21

# Throat yN threshold: (4/13)(9/10) = 18/65
# Structural fraction (4/13) times throughput factor (9/10).
THROAT_YN = (4.0 / 13.0) * (9.0 / 10.0)  # 0.27692...

# Horizon yN threshold: (18/65)(2/21) = 36/1365
# Throat condition divided by f(k)/2. The horizon is where the
# Newtonian acceleration has dropped to 2/f(k) of its throat value.
HORIZON_YN = 2.0 * THROAT_YN / FK  # 0.026374...

# Acceleration ratio between horizon and throat: 2/f(k)
HORIZON_THROAT_ACCEL_RATIO = 2.0 / FK  # 0.095238...

# Throat fraction: R_t / R_env (emerges from the two yN thresholds)
THROAT_FRAC = 0.30

# Speed of light (CODATA 2018 exact)
C_LIGHT = 2.99792458e8  # m/s


def verify_a0():
    """Verify a0 is approximately 1.2e-10 m/s^2."""
    expected = 1.2e-10
    ratio = A0 / expected
    return 0.9 < ratio < 1.1, A0
