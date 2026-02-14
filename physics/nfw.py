"""
NFW dark matter halo model for Lambda-CDM rotation curves.

Implements the Navarro-Frenk-White (1996, 1997) universal dark matter halo
profile, the Dutton & Maccio (2014) concentration-mass relation, and a
chi-squared fitter that finds the best-fit halo mass given observed data.

This module exists to provide a direct visual contrast: CDM requires fitting
halo parameters to each galaxy, while the DTG covariant completion predicts
rotation curves with zero free parameters.

Cosmological parameters: Planck 2018 (TT,TE,EE+lowE+lensing)
  H0 = 67.4 km/s/Mpc
  Omega_m = 0.315

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

from physics.constants import G, M_SUN, KPC_TO_M

# ---------------------------------------------------------------------------
# Cosmological parameters (Planck 2018)
# ---------------------------------------------------------------------------

H0_KM_S_MPC = 67.4                          # Hubble constant
MPC_TO_M = 3.0857e22                         # meters per Megaparsec
H0_SI = H0_KM_S_MPC * 1000.0 / MPC_TO_M     # s^-1
RHO_CRIT = 3.0 * H0_SI * H0_SI / (8.0 * math.pi * G)   # kg/m^3
LITTLE_H = H0_KM_S_MPC / 100.0              # dimensionless h


def concentration(m200_solar):
    """
    Dutton & Maccio (2014) concentration-mass relation at z=0.

    log10(c) = 0.905 - 0.101 * log10(M200 / (10^12 h^-1 M_sun))

    Parameters
    ----------
    m200_solar : float
        Virial mass M_200 in solar masses.

    Returns
    -------
    float
        NFW concentration parameter c.
    """
    x = m200_solar / (1.0e12 / LITTLE_H)
    if x <= 0:
        return 10.0  # fallback
    log_c = 0.905 - 0.101 * math.log10(x)
    return 10.0 ** log_c


def r200_kpc(m200_solar):
    """
    Virial radius r_200 in kiloparsecs.

    r_200 = (3 M_200 / (4 pi 200 rho_crit))^(1/3)
    """
    m200_kg = m200_solar * M_SUN
    r200_m = (3.0 * m200_kg / (4.0 * math.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)
    return r200_m / KPC_TO_M


def nfw_enclosed_mass(r_kpc, m200_solar):
    """
    NFW enclosed dark matter mass at radius r in solar masses.

    M_NFW(<r) = M_200 * g(c) * [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]

    where r_s = r_200/c and g(c) = 1/[ln(1+c) - c/(1+c)].
    """
    if r_kpc <= 0 or m200_solar <= 0:
        return 0.0

    c = concentration(m200_solar)
    r200 = r200_kpc(m200_solar)
    rs = r200 / c

    x = r_kpc / rs
    gc = 1.0 / (math.log(1.0 + c) - c / (1.0 + c))
    m_enc = m200_solar * gc * (math.log(1.0 + x) - x / (1.0 + x))
    return max(m_enc, 0.0)


def nfw_velocity(r_kpc, m200_solar):
    """
    NFW halo circular velocity at radius r in km/s.

    v(r) = sqrt(G * M_NFW(<r) / r)
    """
    m_enc = nfw_enclosed_mass(r_kpc, m200_solar)
    if m_enc <= 0 or r_kpc <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    v = math.sqrt(G * m_enc * M_SUN / r_m)
    return v / 1000.0


def cdm_velocity(r_kpc, m_baryon_enclosed, m200_solar):
    """
    Total CDM rotation velocity: baryonic (Newtonian) + NFW halo.

    v_CDM(r) = sqrt(v_baryon^2 + v_NFW^2)

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kpc.
    m_baryon_enclosed : float
        Enclosed baryonic mass at this radius in solar masses.
    m200_solar : float
        NFW halo virial mass M_200 in solar masses.

    Returns
    -------
    float
        Total circular velocity in km/s.
    """
    if r_kpc <= 0:
        return 0.0

    r_m = r_kpc * KPC_TO_M

    # Baryonic (Newtonian) contribution
    v2_baryon = 0.0
    if m_baryon_enclosed > 0:
        v2_baryon = G * m_baryon_enclosed * M_SUN / r_m

    # NFW halo contribution
    m_nfw = nfw_enclosed_mass(r_kpc, m200_solar) * M_SUN
    v2_nfw = G * m_nfw / r_m if m_nfw > 0 else 0.0

    v_total = math.sqrt(v2_baryon + v2_nfw) / 1000.0
    return v_total


# ---------------------------------------------------------------------------
# Abundance matching: estimate halo mass from baryonic mass
# ---------------------------------------------------------------------------

def abundance_matching(m_baryon_solar):
    """
    Estimate M_200 from total baryonic mass using the Moster et al. (2013)
    stellar-mass--halo-mass relation (inverted).

    The relation: M_star / M_halo = 2N * [(M_halo/M1)^-beta + (M_halo/M1)^gamma]^-1

    Parameters at z=0 (Moster+ 2013 Table 1):
      log10(M1) = 11.590
      N = 0.0351
      beta = 1.376
      gamma = 0.608

    We invert numerically to find M_halo given M_star ~ M_baryon.

    Parameters
    ----------
    m_baryon_solar : float
        Total baryonic mass in solar masses.

    Returns
    -------
    float
        Estimated M_200 in solar masses.
    """
    if m_baryon_solar <= 0:
        return 1.0e10  # fallback

    log_m1 = 11.590
    m1 = 10.0 ** log_m1
    n_param = 0.0351
    beta = 1.376
    gamma = 0.608

    def smhm_ratio(m_halo):
        """Stellar-to-halo mass ratio from Moster+2013."""
        x = m_halo / m1
        if x <= 0:
            return 0.0
        return 2.0 * n_param / (x ** (-beta) + x ** gamma)

    # Bracket search: M_halo from 10^9 to 10^15
    # Use bisection to find M_halo where M_star/M_halo * M_halo = M_baryon
    log_lo = 9.0
    log_hi = 15.0
    for _ in range(80):
        log_mid = 0.5 * (log_lo + log_hi)
        m_halo = 10.0 ** log_mid
        m_star_pred = smhm_ratio(m_halo) * m_halo
        if m_star_pred < m_baryon_solar:
            log_lo = log_mid
        else:
            log_hi = log_mid

    return 10.0 ** (0.5 * (log_lo + log_hi))


# ---------------------------------------------------------------------------
# Halo fitting: find best-fit M_200 given observations
# ---------------------------------------------------------------------------

def fit_halo(observations, mass_model_func, accel_ratio=1.0):
    """
    Find the best-fit NFW halo mass M_200 that minimizes chi-squared
    between the CDM model (baryonic + NFW) and observed rotation data.

    Uses golden-section search on a single parameter (M_200), with the
    concentration fixed by the Dutton & Maccio (2014) relation.

    Parameters
    ----------
    observations : list of dict
        Each dict has 'r' (kpc), 'v' (km/s), 'err' (km/s, optional).
    mass_model_func : callable
        Function(r_kpc) -> enclosed baryonic mass in M_sun.
    accel_ratio : float
        Unused (included for API consistency).

    Returns
    -------
    dict
        {'m200': best-fit M_200 in M_sun,
         'c': concentration,
         'r200_kpc': virial radius in kpc,
         'chi2': chi-squared,
         'chi2_reduced': reduced chi-squared,
         'n_params': 2 (M_200 + c, though c is derived)}
    """
    if not observations:
        return None

    # Filter valid observations
    obs = [(o['r'], o['v'], o.get('err', 5.0)) for o in observations
           if o.get('r', 0) > 0 and o.get('v', 0) > 0]
    if len(obs) < 2:
        return None

    def chi_squared(log_m200):
        m200 = 10.0 ** log_m200
        chi2 = 0.0
        for r, v, err in obs:
            m_bar = mass_model_func(r)
            v_cdm = cdm_velocity(r, m_bar, m200)
            sigma = max(err, 1.0)
            chi2 += ((v - v_cdm) / sigma) ** 2
        return chi2

    # Golden-section search over log10(M_200) in [9, 14]
    a = 9.0
    b = 14.0
    gr = (math.sqrt(5.0) + 1.0) / 2.0  # golden ratio

    c_gs = b - (b - a) / gr
    d_gs = a + (b - a) / gr
    tol = 1.0e-6

    for _ in range(100):
        if abs(b - a) < tol:
            break
        if chi_squared(c_gs) < chi_squared(d_gs):
            b = d_gs
        else:
            a = c_gs
        c_gs = b - (b - a) / gr
        d_gs = a + (b - a) / gr

    best_log_m200 = 0.5 * (a + b)
    best_m200 = 10.0 ** best_log_m200
    best_c = concentration(best_m200)
    best_r200 = r200_kpc(best_m200)
    best_chi2 = chi_squared(best_log_m200)

    n_dof = max(len(obs) - 2, 1)  # 2 effective params: M_200 (fitted) + c (derived)

    return {
        'm200': best_m200,
        'c': round(best_c, 2),
        'r200_kpc': round(best_r200, 2),
        'chi2': round(best_chi2, 4),
        'chi2_reduced': round(best_chi2 / n_dof, 4),
        'n_params_fitted': 1,
        'n_params_total': 2,
    }
