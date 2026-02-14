"""
Mass inference: inverse problem solver.

Given observed velocity v at radius r, infer the enclosed baryonic mass
by evaluating the covariant field equation in the inverse direction.

Forward (prediction):  M -> g_N -> solve field equation for x -> g = a0*x -> v
Inverse (inference):   v -> g = v^2/r -> x = g/a0 -> evaluate y_N = x^2/(1+x) -> g_N -> M

The field equation from the Lagrangian F(y) = y/2 - sqrt(y) + ln(1+sqrt(y))
reduces in spherical symmetry to:  x^2 / (1 + x) = g_N / a0

Forward requires solving a quadratic (given g_N, find x).
Inverse just evaluates the left side (given x, compute g_N).  No solver needed.
"""

from physics.constants import G, M_SUN, KPC_TO_M, A0


def infer_mass(r_kpc, v_km_s, accel_ratio=1.0):
    """
    Infer enclosed baryonic mass from observed rotation velocity.

    Evaluates the covariant field equation in the inverse direction:
      v^2/r = g  =>  x = g/a0  =>  g_N = a0 * x^2/(1+x)  =>  M = g_N * r^2 / G

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

    # From circular orbit: v^2 = g * r  =>  g = v^2 / r
    g_eff = (v * v) / r

    a0_eff = A0 * accel_ratio

    # Dimensionless field strength: x = g / a0
    x = g_eff / a0_eff

    # Field equation evaluated inversely: g_N / a0 = x^2 / (1 + x)
    gN = a0_eff * x * x / (1.0 + x)

    # From g_N = G*M/r^2  =>  M = g_N * r^2 / G
    M = (gN * r * r) / G

    return M / M_SUN
