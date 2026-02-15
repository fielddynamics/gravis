"""
Traced equation callables for the GravisEngine pipeline.

Each function in this module is a self-contained computation that:
  1. Takes (r_kpc, m_solar, **params) as input (theory equations), or
     (r_kpc, mass_model) for the mass stage, or
     (r_kpc, v_km_s, **params) for the inference stage.
  2. Calls the existing tested solver (e.g. aqual.solve_x) for the
     core numerical step.
  3. Returns (output_value, intermediates_dict) where intermediates
     captures every physically meaningful intermediate variable.

The intermediates are what the verbose/advanced API mode will expose,
giving the user the full chain of math at every sample point.

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
IMPORTANT: These functions must produce numerically identical results
to the existing physics module functions. They replicate the same
arithmetic in the same order, calling the same solvers.
"""

import math

from physics.constants import G, M_SUN, KPC_TO_M, A0
from physics.aqual import solve_x as aqual_solve_x
from physics.mond import solve_x as mond_solve_x
from physics.mond import mu_mond
from physics.nfw import nfw_enclosed_mass
from physics.manifold import ALPHA_THROAT, core_weight
from physics.poisson import kappa as poisson_kappa, KAPPA_FLOOR


# ----------------------------------------------------------------------
# Mass model equation
# ----------------------------------------------------------------------

def mass_model_eq(r_kpc, mass_model):
    """
    Compute enclosed baryonic mass with per-component breakdown.

    Replicates physics.mass_model.enclosed_mass() exactly, but also
    returns the individual component contributions as intermediates.

    Signature differs from theory equations: takes the full mass_model
    dict instead of enclosed mass. The engine handles this via
    _run_mass_stage() rather than GravisStage.process().

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    mass_model : dict
        Three-component mass model with 'bulge', 'disk', 'gas' keys.

    Returns
    -------
    tuple of (float, dict)
        (total_enclosed_mass, {"m_bulge": ..., "m_disk": ..., "m_gas": ...})
    """
    if r_kpc <= 0:
        return 0.0, {"m_bulge": 0.0, "m_disk": 0.0, "m_gas": 0.0}

    r = r_kpc
    m_bulge = 0.0
    m_disk = 0.0
    m_gas = 0.0

    # Hernquist bulge: M(<r) = M * r^2 / (r + a)^2
    bulge = mass_model.get("bulge")
    if bulge and bulge.get("M", 0) > 0:
        a = bulge["a"]
        m_bulge = bulge["M"] * r * r / ((r + a) * (r + a))

    # Exponential stellar disk: M(<r) = M * [1 - (1 + r/Rd) * exp(-r/Rd)]
    disk = mass_model.get("disk")
    if disk and disk.get("M", 0) > 0:
        x = r / disk["Rd"]
        m_disk = disk["M"] * (1.0 - (1.0 + x) * math.exp(-x))

    # Exponential gas disk: same functional form as stellar disk
    gas = mass_model.get("gas")
    if gas and gas.get("M", 0) > 0:
        x = r / gas["Rd"]
        m_gas = gas["M"] * (1.0 - (1.0 + x) * math.exp(-x))

    m_total = m_bulge + m_disk + m_gas
    return m_total, {"m_bulge": m_bulge, "m_disk": m_disk, "m_gas": m_gas}


# ----------------------------------------------------------------------
# Theory equations (all share signature: r_kpc, m_solar, **params)
# ----------------------------------------------------------------------

def newtonian_eq(r_kpc, m_solar):
    """
    Newtonian circular velocity.

    v = sqrt(G * M / r)

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed mass in solar masses.

    Returns
    -------
    tuple of (float, dict)
        (v_km_s, {"g_N": ...})
    """
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN
    if r <= 0 or M <= 0:
        return 0.0, {"g_N": 0.0}

    gN = G * M / (r * r)
    v = math.sqrt(G * M / r) / 1000.0
    return v, {"g_N": gN}


def gfd_eq(r_kpc, m_solar, accel_ratio=1.0):
    """
    GFD (Gravity Field Dynamics) circular velocity.

    Solves the covariant field equation derived from the scalar-tensor
    action of the dual tetrad topology:

        x^2 / (1 + x) = g_N / a0

    where x = g/a0 and g_N = GM/r^2. The analytic solution is:

        x = (y_N + sqrt(y_N^2 + 4*y_N)) / 2,   y_N = g_N / a0

    Calls aqual.solve_x() for the numerical step.

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
    tuple of (float, dict)
        (v_km_s, {"g_N": ..., "y_N": ..., "x": ..., "g_eff": ...})
    """
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN
    if r <= 0 or M <= 0:
        return 0.0, {"g_N": 0.0, "y_N": 0.0, "x": 0.0, "g_eff": 0.0}

    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    y_N = gN / a0_eff
    x = aqual_solve_x(y_N)
    g_eff = a0_eff * x
    v = math.sqrt(g_eff * r) / 1000.0

    return v, {"g_N": gN, "y_N": y_N, "x": x, "g_eff": g_eff}


def gfd_manifold_eq(r_kpc, m_solar, accel_ratio=1.0, galactic_radius_kpc=0.0):
    """
    GFD Manifold circular velocity with smooth vortex weighting.

    Computes the base GFD acceleration, then boosts it with a smooth
    Hermite weight distributed over the full galaxy:

        g_base = a0 * x              (standard covariant field equation)
        g_tot  = g_base * (1 + W(r)) (manifold-boosted acceleration)
        v      = sqrt(r * g_tot)

    W(r) = 1 - 3*(r/R)^2 + 2*(r/R)^3  for r <= R_galaxy
    W(r) = 0                            for r > R_galaxy

    The weight tapers smoothly from center to horizon with no hard
    cutoff. The throat at 30% of R_galaxy is where W = 0.784.

    Zero free parameters. The amplitude (one additional copy of g_base
    at center) is fixed by the theory. The galactic radius is an
    observable, not a fitted parameter.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed mass at r_kpc in solar masses.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0).
    galactic_radius_kpc : float, optional
        Galactic gravitational horizon in kiloparsecs.

    Returns
    -------
    tuple of (float, dict)
        (v_km_s, {"W": ..., "galactic_radius": ..., "g_base": ...,
                   "g_tot": ..., "g_N": ..., "y_N": ..., "x": ...})
    """
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0, {
            "W": 0.0, "galactic_radius": galactic_radius_kpc,
            "g_base": 0.0, "g_tot": 0.0,
            "g_N": 0.0, "y_N": 0.0, "x": 0.0,
        }

    # Convert to SI
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN

    # Base GFD field equation
    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    y_N = gN / a0_eff
    x = aqual_solve_x(y_N)
    g_base = a0_eff * x

    # Smooth manifold weighting over full galaxy
    W = core_weight(r_kpc, galactic_radius_kpc)
    g_tot = g_base * (1.0 + W)

    # Velocity from boosted acceleration
    v = math.sqrt(g_tot * r) / 1000.0

    return v, {
        "W": W,
        "galactic_radius": galactic_radius_kpc,
        "g_base": g_base,
        "g_tot": g_tot,
        "g_N": gN,
        "y_N": y_N,
        "x": x,
    }


def gfd_poisson_eq(r_kpc, m_solar, accel_ratio=1.0, galactic_radius_kpc=0.0):
    """
    GFD Poisson circular velocity: covariant + unified manifold operator.

    Computes the base GFD covariant velocity, then applies the Poisson
    unified manifold composition:

        v_poisson(r) = v_cov(r) / sqrt(kappa(r))

    where kappa(r) = 1 - exp(-r^2/r_t^2), r_t = 0.30 * R_env.

    This preserves flat rotation curves at large r (kappa -> 1) while
    enhancing the inner galaxy where the manifold operator is active.

    The composition comes from the derived Poisson equation (Section IX):
        nabla . (kappa * nabla Phi) = 4*pi * G_eff * rho
    which in spherical symmetry gives g_unified = g_cov / kappa, and
    therefore v_unified = v_cov / sqrt(kappa).

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed mass at r_kpc in solar masses.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0).
    galactic_radius_kpc : float, optional
        Galactic gravitational horizon in kiloparsecs.

    Returns
    -------
    tuple of (float, dict)
        (v_km_s, {"kappa": ..., "v_cov": ..., "galactic_radius": ...,
                   "g_N": ..., "y_N": ..., "x": ..., "g_cov": ...})
    """
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0, {
            "kappa": 0.0, "v_cov": 0.0, "galactic_radius": galactic_radius_kpc,
            "g_N": 0.0, "y_N": 0.0, "x": 0.0, "g_cov": 0.0,
        }

    # Step 1: base GFD covariant field equation
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN

    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    y_N = gN / a0_eff
    x = aqual_solve_x(y_N)
    g_cov = a0_eff * x
    v_cov = math.sqrt(g_cov * r) / 1000.0  # km/s

    # Step 2: kappa screening from unified manifold operator
    k = poisson_kappa(r_kpc, galactic_radius_kpc)
    k_safe = max(k, KAPPA_FLOOR)

    # Step 3: compose
    v_poisson = v_cov / math.sqrt(k_safe)

    return v_poisson, {
        "kappa": k,
        "v_cov": v_cov,
        "galactic_radius": galactic_radius_kpc,
        "g_N": gN,
        "y_N": y_N,
        "x": x,
        "g_cov": g_cov,
    }


def gfd_structure_eq(r_kpc, m_solar, accel_ratio=1.0,
                     galactic_radius_kpc=0.0, m_stellar=0.0):
    """
    GFD+: zero-parameter galactic rotation.

    Combines the DTG covariant completion with a structural release
    term that activates outside the throat:

        g(r) = g_DTG(r) + g_struct(r)

        g_DTG   = (1/2)[g_N + sqrt(g_N^2 + 4*g_N*a0)]
        g_struct = (4/13) * G*M_star/R_t^2 * [(r-R_t)/(R_env-R_t)]^(3/4)
                 = 0  for r <= R_t

    Every piece derived from topology:
        4/13 = structural excess from Poisson equation (17/13 - 1)
        3/4  = d/k = spatial dimensions / tetrahedral faces
        R_t  = 0.30 * R_env (throat radius)
        M_star = M_bulge + M_disk (bound field origins, not gas)

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed baryonic mass at r_kpc in solar masses.
    accel_ratio : float, optional
        Multiplier on a0 (default 1.0).
    galactic_radius_kpc : float, optional
        Galactic gravitational horizon R_env in kiloparsecs.
    m_stellar : float, optional
        Total stellar mass (bulge + disk) in solar masses. Not enclosed
        mass, but total, since it represents the entire field origin's
        structural contribution.

    Returns
    -------
    tuple of (float, dict)
        (v_km_s, {"g_N": ..., "g_DTG": ..., "g_struct": ..., "g_total": ...,
                   "R_t": ..., "xi": ...})
    """
    THROAT_FRAC = 0.30
    STRUCT_FRAC = 4.0 / 13.0   # 0.3077
    P_STRUCT = 3.0 / 4.0       # d/k = 0.75

    R_env = galactic_radius_kpc
    R_t = THROAT_FRAC * R_env

    if r_kpc <= 0 or m_solar <= 0:
        return 0.0, {
            "g_N": 0.0, "g_DTG": 0.0, "g_struct": 0.0, "g_total": 0.0,
            "R_t": R_t, "xi": 0.0,
        }

    # Step 1: DTG covariant completion (same as gfd_eq)
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN
    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    y_N = gN / a0_eff
    x = aqual_solve_x(y_N)
    g_dtg = a0_eff * x

    # Step 2: Structural release (outside throat only)
    g_struct = 0.0
    xi = 0.0
    if r_kpc > R_t and R_env > R_t and m_stellar > 0:
        R_t_m = R_t * KPC_TO_M
        g0 = STRUCT_FRAC * G * m_stellar * M_SUN / (R_t_m * R_t_m)
        xi = (r_kpc - R_t) / (R_env - R_t)
        g_struct = g0 * (xi ** P_STRUCT)

    g_total = g_dtg + g_struct
    v = math.sqrt(g_total * r) / 1000.0

    return v, {
        "g_N": gN, "g_DTG": g_dtg, "g_struct": g_struct,
        "g_total": g_total, "R_t": R_t, "xi": xi,
    }


def mond_eq(r_kpc, m_solar, accel_ratio=1.0):
    """
    Classical MOND circular velocity.

    Uses the Bekenstein-Milgrom interpolating function:

        mu(x) = x / sqrt(1 + x^2)

    and solves the field equation mu(x)*x = g_N/a0.

    Calls mond.solve_x() for the numerical step.

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
    tuple of (float, dict)
        (v_km_s, {"g_N": ..., "y_N": ..., "x": ..., "mu_x": ..., "g_eff": ...})
    """
    r = r_kpc * KPC_TO_M
    M = m_solar * M_SUN
    if r <= 0 or M <= 0:
        return 0.0, {
            "g_N": 0.0, "y_N": 0.0, "x": 0.0,
            "mu_x": 0.0, "g_eff": 0.0,
        }

    gN = G * M / (r * r)
    a0_eff = A0 * accel_ratio
    y_N = gN / a0_eff
    x = mond_solve_x(y_N)
    mu_x = mu_mond(x)
    g_eff = a0_eff * x
    v = math.sqrt(g_eff * r) / 1000.0

    return v, {
        "g_N": gN, "y_N": y_N, "x": x,
        "mu_x": mu_x, "g_eff": g_eff,
    }


def cdm_eq(r_kpc, m_solar, m200=None):
    """
    CDM (baryonic + NFW halo) circular velocity.

    v_CDM = sqrt(v_baryon^2 + v_NFW^2)

    Works in SI units (m^2/s^2) internally to match nfw.cdm_velocity()
    exactly. The baryonic contribution is Newtonian; the NFW halo uses
    the Dutton-Maccio concentration relation.

    Parameters
    ----------
    r_kpc : float
        Galactocentric radius in kiloparsecs.
    m_solar : float
        Enclosed baryonic mass in solar masses.
    m200 : float, optional
        NFW halo virial mass M200 in solar masses.

    Returns
    -------
    tuple of (float, dict)
        (v_km_s, {"v_baryon_km_s": ..., "v_nfw_km_s": ..., "m_nfw_enclosed": ...})
    """
    if r_kpc <= 0:
        return 0.0, {
            "v_baryon_km_s": 0.0, "v_nfw_km_s": 0.0,
            "m_nfw_enclosed": 0.0,
        }

    m200_val = m200 or 0.0
    r_m = r_kpc * KPC_TO_M

    # Baryonic (Newtonian) contribution in SI: v^2 in m^2/s^2
    v2_baryon = 0.0
    if m_solar > 0:
        v2_baryon = G * m_solar * M_SUN / r_m

    # NFW halo contribution in SI
    # nfw_enclosed_mass returns solar masses; convert to kg for SI
    m_nfw = nfw_enclosed_mass(r_kpc, m200_val)
    v2_nfw = G * m_nfw * M_SUN / r_m if m_nfw > 0 else 0.0

    # Combined velocity
    v_total = math.sqrt(v2_baryon + v2_nfw) / 1000.0

    # Component velocities for intermediates (in km/s)
    v_baryon_kms = math.sqrt(v2_baryon) / 1000.0 if v2_baryon > 0 else 0.0
    v_nfw_kms = math.sqrt(v2_nfw) / 1000.0 if v2_nfw > 0 else 0.0

    return v_total, {
        "v_baryon_km_s": v_baryon_kms,
        "v_nfw_km_s": v_nfw_kms,
        "m_nfw_enclosed": m_nfw,
    }


# ----------------------------------------------------------------------
# Inference equation (different signature: r_kpc, v_km_s, **params)
# ----------------------------------------------------------------------

def inference_eq(r_kpc, v_km_s, accel_ratio=1.0):
    """
    Inverse field equation: infer enclosed mass from observed velocity.

    Evaluates the covariant field equation in the inverse direction:
        v -> g = v^2/r -> x = g/a0 -> g_N = a0 * x^2/(1+x) -> M = g_N * r^2 / G

    The forward direction requires solving a quadratic. The inverse
    direction just evaluates the left side, so no solver is needed.

    Replicates physics.inference.infer_mass() exactly.

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
    tuple of (float, dict)
        (m_solar, {"g_eff": ..., "x": ..., "g_N": ...})
    """
    r = r_kpc * KPC_TO_M
    v = v_km_s * 1000.0

    if r <= 0 or v <= 0:
        return 0.0, {"g_eff": 0.0, "x": 0.0, "g_N": 0.0}

    # From circular orbit: g = v^2 / r
    g_eff = (v * v) / r

    # Dimensionless field strength
    a0_eff = A0 * accel_ratio
    x = g_eff / a0_eff

    # Field equation evaluated inversely: g_N = a0 * x^2 / (1 + x)
    gN = a0_eff * x * x / (1.0 + x)

    # Inferred enclosed mass: M = g_N * r^2 / G
    M = (gN * r * r) / G / M_SUN

    return M, {"g_eff": g_eff, "x": x, "g_N": gN}
