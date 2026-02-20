"""
Photometric mass parameter derivation for the rotation service.

Derives all 6 mass model parameters from 3.6um photometric decomposition
and GFD topology (horizon/throat conditions). Shared by the enterprise
rotation service for the unified photometric chart and by the sandbox
for inference. No unicode (Windows charmap).
"""

import math

from physics.constants import G, M_SUN, KPC_TO_M, THROAT_YN, HORIZON_YN
from physics.services.rotation.inference import solve_field_geometry


def _hernquist_frac(r, a):
    """Fraction of Hernquist profile enclosed within radius r."""
    if r <= 0 or a <= 0:
        return 0.0
    return r * r / ((r + a) * (r + a))


def _disk_frac(r, Rd):
    """Fraction of exponential disk enclosed within radius r."""
    if r <= 0 or Rd <= 0:
        return 0.0
    x = r / Rd
    if x > 50:
        return 1.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def _solve_3x3_cramer(A, b):
    """Solve 3x3 linear system using Cramer's rule."""
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if abs(det) < 1e-30:
        return None
    x0 = (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2])
        + A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) / det
    x1 = (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2])
        - b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) / det
    x2 = (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1])
        - A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0])
        + b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det
    return (x0, x1, x2)


def derive_mass_parameters_from_photometry(photometry, a0_eff):
    """Derive all 6 mass model parameters from photometric data + topology.

    Independent of the Bayesian velocity fit. Uses GFD topology
    (horizon/throat conditions) to correct the photometric M/L = 0.5
    assumption, then the photometric Lb/Ld ratio to split stellar mass.

    Parameters
    ----------
    photometry : dict
        Photometric decomposition from 3.6um imaging:
            Mb, ab, Md, Rd, Mg, Rg (solar masses, kpc).
        Lb/Ld is inferred from Mb/Md (constant M/L at 3.6um).
    a0_eff : float
        Effective acceleration scale (A0 * accel_ratio) in m/s^2.

    Returns
    -------
    dict with keys:
        mass_model : dict (bulge, disk, gas with M, a/Rd)
        M_total : float
        field_geometry : dict (R_env, R_t, cycle from solve_field_geometry)
        topology : dict (intermediate values)
        or "error" key on failure
    """
    Mb_ph = photometry.get("Mb", 0)
    ab_ph = photometry.get("ab", 0)
    Md_ph = photometry.get("Md", 0)
    Rd_ph = photometry.get("Rd", 0)
    Mg_ph = photometry.get("Mg", 0)
    Rg_ph = photometry.get("Rg", 0)

    if ab_ph <= 0 or Rd_ph <= 0 or Rg_ph <= 0:
        return {"error": "Scale lengths must be positive"}

    geom = solve_field_geometry(
        Mb_ph, ab_ph, Md_ph, Rd_ph, Mg_ph, Rg_ph, a0_eff)

    r_env = geom.get("envelope_radius_kpc")
    r_t = geom.get("throat_radius_kpc")
    cycle = geom.get("cycle", 3)
    yN_at_rt = geom.get("yN_at_throat", 0.0)

    if not r_env or not r_t or r_env <= 0 or r_t <= 0:
        return {"error": "Field geometry failed (no R_env or R_t)"}

    r_env_m = r_env * KPC_TO_M
    M_horizon = HORIZON_YN * r_env_m**2 * a0_eff / (G * M_SUN)

    r_t_m = r_t * KPC_TO_M
    if cycle == 3:
        M_throat = THROAT_YN * r_t_m**2 * a0_eff / (G * M_SUN)
    else:
        M_throat = yN_at_rt * r_t_m**2 * a0_eff / (G * M_SUN)

    fb_env = _hernquist_frac(r_env, ab_ph)
    fd_env = _disk_frac(r_env, Rd_ph)
    fg_env = _disk_frac(r_env, Rg_ph)
    fb_t = _hernquist_frac(r_t, ab_ph)
    fd_t = _disk_frac(r_t, Rd_ph)
    fg_t = _disk_frac(r_t, Rg_ph)

    ratio = 2.0
    sol = None
    for _ in range(30):
        A = [[fb_env, fd_env, fg_env],
             [fb_t,   fd_t,   fg_t],
             [0.0,    1.0,    -ratio]]
        b = [M_horizon, M_throat, 0.0]
        sol = _solve_3x3_cramer(A, b)
        if sol is None:
            return {"error": "Topology 3x3 system is singular"}
        Mb_sol, Md_sol, Mg_sol = sol
        Mt = Mb_sol + Md_sol + Mg_sol
        if Mt <= 0:
            break
        fg = max(Mg_sol, 0) / Mt
        fg = max(0.01, min(fg, 0.99))
        fd_pred = -0.826 * fg + 0.863
        fd_pred = max(0.02, min(fd_pred, 0.95))
        new_ratio = fd_pred / fg
        new_ratio = max(0.01, min(new_ratio, 20.0))
        if abs(new_ratio - ratio) < 0.001:
            break
        ratio = ratio * 0.5 + new_ratio * 0.5

    if sol is None:
        return {"error": "Topology iteration failed"}

    M_total = sol[0] + sol[1] + sol[2]
    M_gas = max(sol[2], 0)

    M_stellar = M_total - M_gas
    photo_ratio = Mb_ph / Md_ph if Md_ph > 0 else 0.0

    if M_stellar > 0 and photo_ratio > 0:
        M_bulge = M_stellar * photo_ratio / (1.0 + photo_ratio)
        M_disk = M_stellar - M_bulge
    elif M_stellar > 0:
        M_bulge = 0.0
        M_disk = M_stellar
    else:
        M_bulge = 0.0
        M_disk = 0.0

    M_bulge = max(M_bulge, 0.0)
    M_disk = max(M_disk, 0.0)

    return {
        "mass_model": {
            "bulge": {"M": round(M_bulge, 2), "a": round(ab_ph, 4)},
            "disk":  {"M": round(M_disk, 2),  "Rd": round(Rd_ph, 4)},
            "gas":   {"M": round(M_gas, 2),   "Rd": round(Rg_ph, 4)},
        },
        "M_total": round(M_bulge + M_disk + M_gas, 2),
        "field_geometry": geom,
        "topology": {
            "M_horizon": round(M_horizon, 2),
            "M_throat": round(M_throat, 2),
            "r_env": round(r_env, 4),
            "r_t": round(r_t, 4),
            "cycle": cycle,
        },
    }
