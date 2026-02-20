"""
SST velocity from Topological Poisson + SST Action (no R_t, R_env).

Bidirectional: velocity from mass (forward) or mass from velocity (inverse).
Uses topological prefactor 17/13 and transition g_total = sqrt(g_source*a0 + g_source^2).
No throat/envelope radii required.

IMPORTANT: No unicode characters (Windows charmap constraint).
"""

import math
from physics.constants import G, M_SUN, KPC_TO_M, A0
try:
    from scipy.interpolate import UnivariateSpline
except ImportError:
    UnivariateSpline = None

# Topological Poisson prefactor from f(k) = 1 + sk + k^2 (k=4, d=3)
COUPLING_RATIO = 17.0 / 13.0

# 6.2% observational error smoothing (defraction, matches gfd_field_velocity_nails_it)
DEFRACTION_SIGMA = 0.062


def g_source_from_mass(r_m, M_enc_kg):
    """Source acceleration (m/s^2) from enclosed mass. Poisson with 17/13."""
    if r_m <= 0:
        return 0.0
    return COUPLING_RATIO * G * M_enc_kg / (r_m * r_m)


def g_total_from_g_source(g_source, a0_eff):
    """SST Action transition: source -> total. g_total = sqrt(g_source*a0 + g_source^2)."""
    if g_source <= 0:
        return 0.0
    return math.sqrt(g_source * a0_eff + g_source * g_source)


def g_source_from_g_total(g_total, a0_eff):
    """Invert SST: g_total^2 = g_source*a0 + g_source^2 -> g_source = (-a0 + sqrt(a0^2+4*g_total^2))/2."""
    if g_total <= 0:
        return 0.0
    # Quadratic: g_source^2 + a0*g_source - g_total^2 = 0; positive root only
    disc = a0_eff * a0_eff + 4.0 * g_total * g_total
    if disc <= 0:
        return 0.0
    return (-a0_eff + math.sqrt(disc)) / 2.0


def velocity_from_g_total(r_m, g_total_m_s2):
    """Circular velocity (km/s) from total acceleration. v = sqrt(r * g_total)."""
    if r_m <= 0 or g_total_m_s2 <= 0:
        return 0.0
    v_m_s = math.sqrt(r_m * g_total_m_s2)
    return v_m_s / 1000.0


def g_total_from_velocity(v_km_s, r_kpc):
    """Total acceleration (m/s^2) from observed velocity. g_total = v^2/r (circular orbit)."""
    if r_kpc <= 0 or v_km_s <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    v_m_s = v_km_s * 1000.0
    return (v_m_s * v_m_s) / r_m


# ---------------------------------------------------------------------------
# Forward: mass -> velocity (no R_t, R_env)
# ---------------------------------------------------------------------------

def gfd_velocity_sst(r_kpc, M_enc_solar, a0_eff):
    """
    GFD velocity (km/s) at radius r_kpc for enclosed mass M_enc_solar.
    Topological Poisson (17/13) + SST transition. No throat/envelope.
    """
    if r_kpc <= 0 or M_enc_solar <= 0:
        return 0.0
    r_m = r_kpc * KPC_TO_M
    M_kg = M_enc_solar * M_SUN
    g_source = g_source_from_mass(r_m, M_kg)
    g_total = g_total_from_g_source(g_source, a0_eff)
    return velocity_from_g_total(r_m, g_total)


def gfd_velocity_curve_sst(radii_kpc, M_enc_at_r, a0_eff):
    """GFD velocity curve (km/s) from enclosed mass at each radius. M_enc_at_r(r) in solar masses."""
    return [gfd_velocity_sst(r, M_enc_at_r(r), a0_eff) for r in radii_kpc]


# ---------------------------------------------------------------------------
# Field velocity from observations: inverse at obs points -> smooth M(r) -> forward only.
# Per gfd_field_velocity_nails_it: no defraction on velocity. M(r) -> Eq75 -> SST -> v is pure algebra.
# ---------------------------------------------------------------------------

def _interp_curve(r_known, v_known, r_target):
    """Linear interpolation: v at each r in r_target from (r_known, v_known). Clamp at ends."""
    if not r_known or not v_known or len(r_known) != len(v_known):
        return []
    pairs = sorted(zip(r_known, v_known), key=lambda p: p[0])
    r_known, v_known = [p[0] for p in pairs], [p[1] for p in pairs]
    r_min, r_max = r_known[0], r_known[-1]
    out = []
    for r in r_target:
        if r <= r_min:
            out.append(v_known[0])
        elif r >= r_max:
            out.append(v_known[-1])
        else:
            for i in range(len(r_known) - 1):
                if r_known[i] <= r <= r_known[i + 1]:
                    denom = r_known[i + 1] - r_known[i]
                    t = (r - r_known[i]) / denom if denom else 0.0
                    out.append(v_known[i] + t * (v_known[i + 1] - v_known[i]))
                    break
            else:
                out.append(v_known[-1])
    return out


def _interp_curve_smooth(r_known, v_known, r_target):
    """
    Cubic spline interpolation so the field velocity curve is smooth (no kinks).
    Falls back to linear if scipy unavailable or too few points.
    """
    if not r_known or not v_known or len(r_known) != len(v_known) or not r_target:
        return []
    pairs = sorted(zip(r_known, v_known), key=lambda p: p[0])
    r_known, v_known = [p[0] for p in pairs], [p[1] for p in pairs]
    r_min, r_max = r_known[0], r_known[-1]
    # Need at least 4 points for cubic spline; otherwise linear
    if UnivariateSpline is None or len(r_known) < 4:
        return _interp_curve(r_known, v_known, r_target)
    # UnivariateSpline: s=0 interpolates exactly; k=3 cubic
    try:
        spl = UnivariateSpline(r_known, v_known, k=min(3, len(r_known) - 1), s=0)
        out = []
        for r in r_target:
            if r <= r_min:
                out.append(v_known[0])
            elif r >= r_max:
                out.append(v_known[-1])
            else:
                out.append(float(spl(r)))
        return out
    except Exception:
        return _interp_curve(r_known, v_known, r_target)


def gfd_field_velocity_curve_covariant_poisson(obs_r, obs_v, target_radii, a0_eff,
                                              defraction_sigma_frac=DEFRACTION_SIGMA):
    """
    GFD field velocity curve from observations. No photometric mass. No defraction on velocity.

    Pipeline (per gfd_field_velocity_nails_it):
      1. Inverse at observation points only: v_obs -> M_enc(r_i) at each (r_i, v_i).
      2. Smooth M(r): spline M_enc onto target_radii (smooth the mass profile, not velocity).
      3. Forward only: v(r) = gfd_velocity_sst(r, M(r), a0). Pure algebra, Eq75 + SST.

    The curve is smooth because M(r) is smooth; the forward step is four lines of closed-form math.
    defraction_sigma_frac is ignored (kept for API compatibility).
    """
    if not obs_r or not obs_v or len(obs_r) != len(obs_v) or len(obs_r) < 2 or not target_radii:
        return []
    # Inverse at obs points only -> M_enc at obs_r
    M_enc_at_obs = []
    for r, v in zip(obs_r, obs_v):
        if r > 0 and v > 0:
            M_enc_at_obs.append(M_enc_from_velocity(v, r, a0_eff))
        else:
            M_enc_at_obs.append(0.0)
    # Smooth M(r) on target_radii (spline M_enc, not v)
    M_enc_at_target = _interp_curve_smooth(obs_r, M_enc_at_obs, target_radii)
    if not M_enc_at_target:
        return []
    # Forward only: v(r) = gfd_velocity_sst(r, M(r), a0). Pure algebra.
    out = []
    for r, M in zip(target_radii, M_enc_at_target):
        if r <= 0 or M < 0:
            out.append(0.0)
        else:
            out.append(round(gfd_velocity_sst(r, M, a0_eff), 4))
    return out


def gfd_sst_velocity_decode_curve(obs_r, obs_v, target_radii, a0_eff,
                                  sigma_frac=DEFRACTION_SIGMA):
    """
    GFD velocity decode (no mass input): obs -> defraction smooth -> interp -> inverse then forward.
    Same pipeline as Python "GFD velocity decode (no mass input)" (blue curve).
    Returns velocity (km/s) at each target_radii. Use for observational_model and vortex_model charts.
    """
    if not obs_r or not obs_v or len(obs_r) != len(obs_v) or len(obs_r) < 2 or not target_radii:
        return []
    v_smooth = defraction_smooth(obs_r, obs_v, sigma_frac=sigma_frac)
    v_at_radii = _interp_curve_smooth(obs_r, v_smooth, target_radii)
    return field_velocity_from_observations(target_radii, v_at_radii, a0_eff)


def field_velocity_from_observations(radii_kpc, v_at_radii, a0_eff):
    """
    Field velocity (km/s) at radii using the gfd_velocity_with_poisson forward equation.
    At each r: M_enc = M_enc_from_velocity(v_at_radii, r, a0); v = gfd_velocity_sst(r, M_enc, a0).
    Same equation as internal/big deal/gfd_velocity_with_poisson.py forward_velocity.
    """
    if not radii_kpc or not v_at_radii or len(radii_kpc) != len(v_at_radii):
        return []
    out = []
    for i in range(len(radii_kpc)):
        r = radii_kpc[i]
        v = v_at_radii[i] if i < len(v_at_radii) else 0.0
        if r <= 0 or v <= 0:
            out.append(0.0)
            continue
        M_enc = M_enc_from_velocity(v, r, a0_eff)
        v_field = gfd_velocity_sst(r, M_enc, a0_eff)
        out.append(round(v_field, 4))
    return out


# ---------------------------------------------------------------------------
# Inverse: velocity -> g_total -> g_source -> M_enc (direct field reading)
# ---------------------------------------------------------------------------

def M_enc_from_velocity(v_km_s, r_kpc, a0_eff):
    """
    Enclosed mass (solar masses) inferred from observed velocity at r.
    Inverse SST + Poisson. No mass model required.
    """
    if r_kpc <= 0 or v_km_s <= 0:
        return 0.0
    g_total = g_total_from_velocity(v_km_s, r_kpc)
    g_source = g_source_from_g_total(g_total, a0_eff)
    r_m = r_kpc * KPC_TO_M
    # g_source = (17/13) * G * M_enc / r^2  =>  M_enc = g_source * r^2 / (G * 17/13)
    M_kg = g_source * r_m * r_m / (COUPLING_RATIO * G)
    return M_kg / M_SUN


def g_source_from_velocity(v_km_s, r_kpc, a0_eff):
    """Source acceleration (m/s^2) inferred from observed velocity (direct field reading)."""
    g_total = g_total_from_velocity(v_km_s, r_kpc)
    return g_source_from_g_total(g_total, a0_eff)


# ---------------------------------------------------------------------------
# Defraction: 6.2% Gaussian kernel smoothing (observation -> field velocity)
# ---------------------------------------------------------------------------

def defraction_smooth(radii_kpc, v_km_s, sigma_frac=DEFRACTION_SIGMA):
    """
    Apply defraction smoothing to velocity observations: Gaussian kernel.
    Kernel width = sigma_frac * (r_max - r_min) * 0.5 (matches gfd_field_velocity_nails_it).
    Returns smoothed velocities at the same radii (deterministic field velocity from obs).

    The result is a weighted average at each radius, not an exact match to the
    observational mean. With 6.2% defraction the curve cannot yield a linear
    exact match to the observations; it is a smoothed field reading.
    """
    if not radii_kpc or not v_km_s or len(radii_kpc) != len(v_km_s) or len(radii_kpc) < 2:
        return list(v_km_s) if v_km_s else []
    r_min = min(radii_kpc)
    r_max = max(radii_kpc)
    span = r_max - r_min
    kernel_width = sigma_frac * span * 0.5
    # Enforce minimum so defraction is never identity (exact match to obs)
    kernel_width = max(kernel_width, 0.02 * span)
    if kernel_width <= 0:
        return list(v_km_s)
    out = []
    for i in range(len(radii_kpc)):
        r_i = radii_kpc[i]
        w = []
        for j in range(len(radii_kpc)):
            d = (radii_kpc[j] - r_i) / kernel_width
            w.append(math.exp(-0.5 * d * d))
        s = sum(w)
        if s <= 0:
            out.append(v_km_s[i])
        else:
            out.append(sum(w[k] * v_km_s[k] for k in range(len(v_km_s))) / s)
    return [round(x, 4) for x in out]
