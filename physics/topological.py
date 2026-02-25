"""
GFD Topological: signed Burgers vortex velocity stage for GRAVIS.

Two-phase pipeline producing a single velocity series:
  Phase 1 (State Classification):
    Smooth observations with 6.2% Gaussian diffraction, compare to GFD
    photometric, classify interior as ABSORBING, PUMPING, or QUIET.
  Phase 2 (Signed Vortex Fit):
    ABSORBING: fit fractional mass absorption (alpha, r_v).
    PUMPING:   fit additive Burgers vortex (Gamma, r_v).
    QUIET:     return GFD photometric unchanged.

Output is a single velocity curve evaluated at the pipeline radii,
identical in shape to any other GravisEngine stage.

Requires observations. When none are available, falls through to QUIET
(identical to the base GFD photometric curve).

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

from physics.core import StageResult
from physics.constants import G, M_SUN, KPC_TO_M, A0

# G in kpc (km/s)^2 / M_sun for the optimizer (avoids SI conversions
# inside the tight loop). Derived: G_SI * M_SUN / KPC_M / 1e6.
G_KPC = G * M_SUN / KPC_TO_M / 1e6  # ~4.301e-6

# a0 in (km/s)^2 / kpc
A0_KPC = A0 * KPC_TO_M / 1e6  # ~3703


def _enc_mass_kpc(r_kpc, mass_model):
    """Enclosed baryonic mass (solar masses) at r_kpc from mass_model dict."""
    if r_kpc <= 0:
        return 0.0
    m = 0.0
    bulge = mass_model.get("bulge")
    if bulge and bulge.get("M", 0) > 0:
        a = bulge["a"]
        m += bulge["M"] * r_kpc * r_kpc / ((r_kpc + a) * (r_kpc + a))
    disk = mass_model.get("disk")
    if disk and disk.get("M", 0) > 0:
        x = r_kpc / disk["Rd"]
        if x < 50:
            m += disk["M"] * (1.0 - (1.0 + x) * math.exp(-x))
        else:
            m += disk["M"]
    gas = mass_model.get("gas")
    if gas and gas.get("M", 0) > 0:
        x = r_kpc / gas["Rd"]
        if x < 50:
            m += gas["M"] * (1.0 - (1.0 + x) * math.exp(-x))
        else:
            m += gas["M"]
    return m


def _gfd_photometric_kpc(r_kpc, m_solar, a0_kpc):
    """GFD photometric velocity (km/s) from enclosed mass in kpc units."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    gs = G_KPC * m_solar / (r_kpc * r_kpc)
    gt = (gs + math.sqrt(gs * gs + 4.0 * gs * a0_kpc)) / 2.0
    return math.sqrt(gt * r_kpc)


def _gaussian_smooth(radii, values, sigma_frac=0.062):
    """6.2% Gaussian diffraction smoothing (weighted average kernel)."""
    n = len(radii)
    if n < 2:
        return list(values)
    span = radii[-1] - radii[0]
    sigma = sigma_frac * span
    if sigma <= 0:
        return list(values)
    out = []
    for i in range(n):
        wsum = 0.0
        vsum = 0.0
        for j in range(n):
            d = (radii[j] - radii[i]) / sigma
            w = math.exp(-0.5 * d * d)
            wsum += w
            vsum += w * values[j]
        out.append(vsum / wsum if wsum > 0 else values[i])
    return out


# ------------------------------------------------------------------
# Phase 1: State Classification
# ------------------------------------------------------------------

def classify_interior(obs_r, obs_v, mass_model, a0_kpc):
    """Classify interior state from smoothed residual vs photometric.

    Returns (state, delta_mean, delta_rms) where state is one of
    'ABSORBING', 'PUMPING', 'QUIET'.
    """
    n = len(obs_r)
    if n < 3:
        return "QUIET", 0.0, 0.0

    v_smooth = _gaussian_smooth(obs_r, obs_v)
    v_phot = [_gfd_photometric_kpc(obs_r[i],
              _enc_mass_kpc(obs_r[i], mass_model), a0_kpc)
              for i in range(n)]

    delta = [v_smooth[i] - v_phot[i] for i in range(n)]
    n_abs = sum(1 for d in delta if d < 0)
    f_abs = n_abs / n
    f_pump = 1.0 - f_abs
    d_mean = sum(delta) / n
    d_rms = math.sqrt(sum(d * d for d in delta) / n)

    if f_abs > 0.75 and d_mean < -2.0:
        return "ABSORBING", d_mean, d_rms
    if f_pump > 0.75 and d_mean > 2.0:
        return "PUMPING", d_mean, d_rms
    return "QUIET", d_mean, d_rms


# ------------------------------------------------------------------
# Phase 2: Signed Vortex Fit (Nelder-Mead, no gradient needed)
# ------------------------------------------------------------------

def _fit_absorbing(obs_r, obs_v, obs_err, mass_model, a0_kpc):
    """Fit fractional mass absorption: M_eff = M_bar * (1 - alpha*exp(-r^2/2r_v^2))."""
    from scipy.optimize import minimize as sp_minimize

    n = len(obs_r)

    def predict(alpha, r_v):
        out = []
        for i in range(n):
            r = obs_r[i]
            m_bar = _enc_mass_kpc(r, mass_model)
            f_abs = alpha * math.exp(-r * r / (2.0 * r_v * r_v))
            m_eff = m_bar * max(1.0 - f_abs, 0.001)
            out.append(_gfd_photometric_kpc(r, m_eff, a0_kpc))
        return out

    def objective(params):
        alpha, r_v = params
        if not (0.01 < alpha < 1.0 and 0.1 < r_v < 100.0):
            return 1e15
        vp = predict(alpha, r_v)
        return sum(((obs_v[i] - vp[i]) / max(obs_err[i], 1.0)) ** 2
                   for i in range(n))

    best = None
    best_chi2 = 1e15
    for ai in (0.1, 0.3, 0.5, 0.7, 0.9):
        for ri in (0.5, 1.0, 3.0, 8.0, 20.0):
            try:
                res = sp_minimize(objective, [ai, ri], method="Nelder-Mead",
                                  options={"maxiter": 20000, "xatol": 1e-8,
                                           "fatol": 1e-10})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best = res
            except Exception:
                pass

    if best is None:
        return 0.0, 1.0
    return best.x[0], best.x[1]


def _fit_pumping(obs_r, obs_v, obs_err, mass_model, a0_kpc):
    """Fit additive Burgers vortex: g_N = g_bar + g_vortex."""
    from scipy.optimize import minimize as sp_minimize

    n = len(obs_r)

    def predict(gamma, r_v):
        out = []
        for i in range(n):
            r = obs_r[i]
            g_bar = G_KPC * _enc_mass_kpc(r, mass_model) / (r * r) if r > 0 else 0.0
            v_vort = (gamma / max(r, 1e-6)) * (1.0 - math.exp(-r * r / (2.0 * r_v * r_v)))
            g_vort = v_vort * v_vort / max(r, 1e-6)
            g_n = g_bar + g_vort
            gt = (g_n + math.sqrt(g_n * g_n + 4.0 * g_n * a0_kpc)) / 2.0
            out.append(math.sqrt(gt * r) if gt > 0 and r > 0 else 0.0)
        return out

    def objective(params):
        gamma, r_v = params
        if not (0.0 < gamma < 10000.0 and 0.1 < r_v < 50.0):
            return 1e15
        vp = predict(gamma, r_v)
        return sum(((obs_v[i] - vp[i]) / max(obs_err[i], 1.0)) ** 2
                   for i in range(n))

    best = None
    best_chi2 = 1e15
    for gi in (10, 100, 500, 2000):
        for ri in (0.5, 2.0, 5.0, 12.0):
            try:
                res = sp_minimize(objective, [gi, ri], method="Nelder-Mead",
                                  options={"maxiter": 20000, "xatol": 1e-8,
                                           "fatol": 1e-10})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best = res
            except Exception:
                pass

    if best is None:
        return 0.0, 1.0
    return best.x[0], best.x[1]


# ------------------------------------------------------------------
# Per-point evaluation (after fitting)
# ------------------------------------------------------------------

def _eval_absorbing(r_kpc, m_solar, alpha, r_v, a0_kpc):
    """Velocity at one radius for ABSORBING state."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    f_abs = alpha * math.exp(-r_kpc * r_kpc / (2.0 * r_v * r_v))
    m_eff = m_solar * max(1.0 - f_abs, 0.001)
    return _gfd_photometric_kpc(r_kpc, m_eff, a0_kpc)


def _eval_pumping(r_kpc, m_solar, gamma, r_v, a0_kpc):
    """Velocity at one radius for PUMPING state."""
    if r_kpc <= 0 or m_solar <= 0:
        return 0.0
    g_bar = G_KPC * m_solar / (r_kpc * r_kpc)
    v_vort = (gamma / r_kpc) * (1.0 - math.exp(-r_kpc * r_kpc / (2.0 * r_v * r_v)))
    g_vort = v_vort * v_vort / r_kpc
    g_n = g_bar + g_vort
    gt = (g_n + math.sqrt(g_n * g_n + 4.0 * g_n * a0_kpc)) / 2.0
    return math.sqrt(gt * r_kpc) if gt > 0 else 0.0


# ------------------------------------------------------------------
# GfdTopologicalStage: custom process() that fits once, evaluates many
# ------------------------------------------------------------------

class GfdTopologicalStage:
    """Pipeline stage producing the GFD Topological (signed vortex) velocity series.

    Unlike standard GravisStage, this runs a two-phase classification +
    optimization once, then evaluates the fitted model at every pipeline
    radius. Observations are passed in via the parameters dict.
    """

    def __init__(self, name, equation_label, parameters=None):
        self.name = name
        self.equation_label = equation_label
        self.parameters = parameters or {}

    def process(self, radii, enclosed_masses):
        accel_ratio = self.parameters.get("accel_ratio", 1.0)
        observations = self.parameters.get("observations", [])
        mass_model = self.parameters.get("mass_model", {})

        a0_kpc = A0_KPC * accel_ratio

        obs_r = [float(o["r"]) for o in observations
                 if o.get("r") and float(o.get("r", 0)) > 0
                 and o.get("v") and float(o.get("v", 0)) > 0]
        obs_v = [float(o["v"]) for o in observations
                 if o.get("r") and float(o.get("r", 0)) > 0
                 and o.get("v") and float(o.get("v", 0)) > 0]
        obs_err = [float(o.get("err", 5.0)) for o in observations
                   if o.get("r") and float(o.get("r", 0)) > 0
                   and o.get("v") and float(o.get("v", 0)) > 0]

        # Phase 1: classify
        state = "QUIET"
        alpha = 0.0
        r_v = 1.0
        gamma = 0.0

        if len(obs_r) >= 3:
            state, d_mean, d_rms = classify_interior(
                obs_r, obs_v, mass_model, a0_kpc)

            # Phase 2: fit
            if state == "ABSORBING":
                alpha, r_v = _fit_absorbing(
                    obs_r, obs_v, obs_err, mass_model, a0_kpc)
            elif state == "PUMPING":
                gamma, r_v = _fit_pumping(
                    obs_r, obs_v, obs_err, mass_model, a0_kpc)

        # Evaluate at pipeline radii
        n = len(radii)
        output = []
        intermed = {
            "state": [], "alpha": [], "gamma": [], "r_v": [],
        }

        for i in range(n):
            r = radii[i]
            m = enclosed_masses[i]

            if state == "ABSORBING":
                v = _eval_absorbing(r, m, alpha, r_v, a0_kpc)
            elif state == "PUMPING":
                v = _eval_pumping(r, m, gamma, r_v, a0_kpc)
            else:
                v = _gfd_photometric_kpc(r, m, a0_kpc)

            output.append(v)
            intermed["state"].append(state)
            intermed["alpha"].append(alpha)
            intermed["gamma"].append(gamma)
            intermed["r_v"].append(r_v)

        return StageResult(
            name=self.name,
            equation_label=self.equation_label,
            parameters={
                "accel_ratio": accel_ratio,
                "interior_state": state,
                "alpha": round(alpha, 6),
                "gamma": round(gamma, 4),
                "r_v": round(r_v, 4),
            },
            series=output,
            intermediates=intermed,
        )
