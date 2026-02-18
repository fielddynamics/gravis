"""
Field Analysis Service for GRAVIS.

Computes derived GFD field metrics for a galactic system once the
observation mode has resolved the mass model and Origin Throughput.

This module is imported by RotationService and called from the
/api/rotation/field_analysis endpoint. It receives the resolved
mass model and galactic parameters, re-runs the GFD and GFD-sigma
pipelines internally (no data passed from the frontend), and returns
a rich set of field geometry, dynamics, and coupling metrics.

Design constraints:
    - The API self-generates all curves; the frontend sends only the
      mass model + config, never raw series data.
    - All values are plain JSON (no numpy). Math expressions are
      returned as LaTeX strings for KaTeX rendering.

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

from physics.equations import gfd_eq, aqual_solve_x
from physics.constants import G, M_SUN, KPC_TO_M, A0, K_SIMPLEX, M_E, R_E
from physics.sigma import GfdSymmetricStage, auto_vortex_strength
from physics.mass_model import enclosed_mass, total_mass
from physics.services.rotation.inference import solve_field_geometry


def compute_field_analysis(mass_model, galactic_radius, vortex_strength,
                           accel_ratio=1.0, num_points=300):
    """
    Compute comprehensive GFD field metrics for a resolved galactic system.

    Parameters
    ----------
    mass_model : dict
        Resolved mass model with bulge/disk/gas components.
    galactic_radius : float
        Field horizon R_env in kpc.
    vortex_strength : float
        Fitted Origin Throughput value.
    accel_ratio : float
        Multiplier on a0 (default 1.0).
    num_points : int
        Number of radial sample points.

    Returns
    -------
    dict
        Field analysis metrics organized into sections.
    """
    R_env_catalog = galactic_radius
    R_t_catalog = 0.30 * R_env_catalog
    a0_eff = A0 * accel_ratio

    # --- Topological field geometry from the mass model ---
    # solve_field_geometry uses y_N = (4/13)(9/10) = 18/65 to derive
    # R_t and R_env from the baryonic mass profile alone.
    _b = mass_model.get("bulge", {})
    _d = mass_model.get("disk", {})
    _g = mass_model.get("gas", {})
    topo_geom = solve_field_geometry(
        _b.get("M", 0), _b.get("a", 0),
        _d.get("M", 0), _d.get("Rd", 0),
        _g.get("M", 0), _g.get("Rd", 0),
        a0_eff,
    )
    R_t_predicted = topo_geom["throat_radius_kpc"]
    R_env_predicted = topo_geom["envelope_radius_kpc"]

    # Use predicted values when available, fall back to catalog
    R_env = R_env_predicted if R_env_predicted else R_env_catalog
    R_t = R_t_predicted if R_t_predicted else R_t_catalog

    # --- Mass properties ---
    m_bulge = mass_model.get("bulge", {}).get("M", 0)
    m_disk = mass_model.get("disk", {}).get("M", 0)
    m_gas = mass_model.get("gas", {}).get("M", 0)
    m_total = m_bulge + m_disk + m_gas
    m_stellar = m_bulge + m_disk
    f_gas = m_gas / m_total if m_total > 0 else 0.0
    bt_ratio = m_bulge / m_total if m_total > 0 else 0.0

    # --- Generate radial samples ---
    max_r = R_env * 1.05
    radii = [max_r * (i + 1) / num_points for i in range(num_points)]

    # --- Compute enclosed mass at each radius ---
    masses = [enclosed_mass(r, mass_model) for r in radii]

    # --- Compute GFD base velocities ---
    gfd_v = []
    gfd_intermediates = []
    for i, r in enumerate(radii):
        v, inter = gfd_eq(r, masses[i], accel_ratio)
        gfd_v.append(v)
        gfd_intermediates.append(inter)

    # --- Compute Newtonian velocities ---
    newt_v = []
    for i, r in enumerate(radii):
        r_m = r * KPC_TO_M
        M = masses[i] * M_SUN
        if r_m > 0 and M > 0:
            v_n = math.sqrt(G * M / r_m) / 1000.0
        else:
            v_n = 0.0
        newt_v.append(v_n)

    # --- Compute GFD-sigma (Observed) velocities ---
    sigma_stage = GfdSymmetricStage(
        "gfd_symmetric", "GFD (Observed)",
        parameters={
            "accel_ratio": accel_ratio,
            "galactic_radius_kpc": R_env,
            "m_stellar": m_stellar,
            "f_gas": f_gas,
            "vortex_strength": vortex_strength,
        }
    )
    sigma_result = sigma_stage.process(radii, masses)
    sigma_v = sigma_result.series

    # --- Key velocity readings ---
    def interp_v(series, target_r):
        """Linear interpolation of velocity at a specific radius."""
        for i in range(len(radii) - 1):
            if radii[i] <= target_r <= radii[i + 1]:
                t = (target_r - radii[i]) / (radii[i + 1] - radii[i])
                return series[i] + t * (series[i + 1] - series[i])
        return series[-1] if radii else 0.0

    v_peak_gfd = max(gfd_v) if gfd_v else 0.0
    r_peak_gfd = radii[gfd_v.index(v_peak_gfd)] if gfd_v else 0.0

    v_peak_sigma = max(sigma_v) if sigma_v else 0.0
    r_peak_sigma = radii[sigma_v.index(v_peak_sigma)] if sigma_v else 0.0

    v_at_Rt_gfd = interp_v(gfd_v, R_t)
    v_at_Rt_sigma = interp_v(sigma_v, R_t)
    v_at_Renv_gfd = interp_v(gfd_v, R_env)
    v_at_Renv_sigma = interp_v(sigma_v, R_env)

    v_at_Rt_newt = interp_v(newt_v, R_t)
    v_at_Renv_newt = interp_v(newt_v, R_env)

    # --- Field coupling at key radii ---
    m_at_Rt = enclosed_mass(R_t, mass_model)
    m_at_Renv = enclosed_mass(R_env, mass_model)

    r_t_m = R_t * KPC_TO_M
    r_env_m = R_env * KPC_TO_M

    gN_at_Rt = G * m_at_Rt * M_SUN / (r_t_m * r_t_m) if r_t_m > 0 else 0.0
    gN_at_Renv = G * m_at_Renv * M_SUN / (r_env_m * r_env_m) if r_env_m > 0 else 0.0

    yN_at_Rt = gN_at_Rt / a0_eff if a0_eff > 0 else 0.0
    yN_at_Renv = gN_at_Renv / a0_eff if a0_eff > 0 else 0.0

    # --- Structural release metrics ---
    f_eff = (1.0 + f_gas) / 2.0 * vortex_strength
    g0 = 0.0
    if m_stellar > 0 and r_t_m > 0:
        g0 = (4.0 / 13.0) * G * m_stellar * M_SUN / (r_t_m * r_t_m)

    # --- Throughput correction magnitude ---
    delta_v_at_Renv = v_at_Renv_sigma - v_at_Renv_gfd
    delta_v_at_Rt = v_at_Rt_sigma - v_at_Rt_gfd
    delta_pct_Renv = (delta_v_at_Renv / v_at_Renv_gfd * 100.0
                      if v_at_Renv_gfd > 0 else 0.0)

    # --- GFD/Newtonian ratio ---
    gfd_newt_ratio_Rt = v_at_Rt_gfd / v_at_Rt_newt if v_at_Rt_newt > 0 else 0.0
    gfd_newt_ratio_Renv = (v_at_Renv_gfd / v_at_Renv_newt
                           if v_at_Renv_newt > 0 else 0.0)

    # --- Transition radius: where y_N crosses 1 (Newtonian -> deep-field) ---
    r_transition = 0.0
    for i in range(len(radii)):
        inter = gfd_intermediates[i]
        if inter.get("y_N", 0) > 0 and inter["y_N"] <= 1.0:
            if i > 0 and gfd_intermediates[i - 1].get("y_N", 0) > 1.0:
                yA = gfd_intermediates[i - 1]["y_N"]
                yB = inter["y_N"]
                t = (1.0 - yA) / (yB - yA) if yB != yA else 0.5
                r_transition = radii[i - 1] + t * (radii[i] - radii[i - 1])
                break

    # --- Theoretical throughput vs fitted ---
    theoretical_ot = auto_vortex_strength(mass_model, R_env)

    # --- Build LaTeX equations for this system ---
    # Total mass in scientific notation
    m_exp = int(math.floor(math.log10(m_total))) if m_total > 0 else 0
    m_coeff = m_total / (10 ** m_exp) if m_total > 0 else 0

    # The derivation chain follows the topology -> action -> Lagrangian
    # -> field equation -> solution -> velocity pipeline exactly as
    # implemented in physics/aqual.py and physics/constants.py.
    equations = {
        # Step 1: The full scalar-tensor action
        "action": (
            r"S_{\mathrm{ST}} = \int d^4x \sqrt{-g}"
            r"\left["
            r"\frac{R}{16\pi G}"
            r" - \frac{a_0^2}{8\pi G}\,"
            r"\mathcal{F}\!\left(\frac{|\nabla\Phi|^2}{a_0^2}\right)"
            r" - \frac{1}{4}\,e^{-2\Phi/c^2}\,F_{\mu\nu}F^{\mu\nu}"
            r" + \mathcal{L}_{\mathrm{matter}}"
            r"\right]"
        ),
        # Step 2: Coupling polynomial from the stellated octahedron
        "coupling_poly": (
            r"f(k) = 1 + k + k^2"
            r"\quad\quad k = d+1 = 4"
        ),
        # Step 3: Scalar Lagrangian determined by f(k)
        # Three terms map: k^2 -> y/2, k -> -sqrt(y), k^0 -> ln(1+sqrt(y))
        "lagrangian": (
            r"\mathcal{F}(y) = \frac{y}{2}"
            r" - \sqrt{y}"
            r" + \ln\!\left(1 + \sqrt{y}\right)"
            r"\quad\quad y = \frac{|\nabla\Phi|^2}{a_0^2}"
        ),
        # Step 4: The Euler-Lagrange equation (spherical symmetry)
        "field_eq": (
            r"\frac{x^2}{1+x} = \frac{g_N}{a_0}"
            r"\quad\quad x = \frac{g}{a_0}"
        ),
        # Step 5: Analytic solution (physical root of the quadratic)
        "solution": (
            r"x = \frac{y_N + \sqrt{y_N^2 + 4\,y_N}}{2}"
            r"\quad\quad y_N = \frac{g_N}{a_0}"
        ),
        # Step 6: Acceleration scale (zero free parameters)
        "acceleration_scale": (
            r"a_0 = \frac{k^2 \, G \, m_e}{r_e^2}"
            r" = %.4g \;\mathrm{m/s^2}" % a0_eff
        ),
        # Step 7: Circular velocity
        "velocity": (
            r"v(r) = \sqrt{a_0 \cdot x(r) \cdot r}"
        ),
        # Resolved total baryonic mass for this system
        "total_mass": (
            r"M_{\mathrm{bary}} = %.3f \times 10^{%d} \; M_\odot"
            % (m_coeff, m_exp)
        ),
        # Origin Throughput: structural release on the outer arm
        "structural_release": (
            r"g_{\mathrm{struct}} = f_{\mathrm{eff}} \cdot g_0"
            r" \cdot \xi^{3/4}"
            r"\quad (r > R_t)"
        ),
        # Effective coupling for this system
        "f_eff": (
            r"f_{\mathrm{eff}} = \frac{1 + f_{\mathrm{gas}}}{2}"
            r" \cdot \sigma = \frac{1 + %.3f}{2} \cdot %.2f = %.4f"
            % (f_gas, vortex_strength, f_eff)
        ),
    }

    # --- Parametric equation export ---
    # Build the full callable equation with fitted parameters baked in
    # so a researcher can reproduce the curve without the pipeline.
    a_b = mass_model.get("bulge", {}).get("a", 0)
    Rd_d = mass_model.get("disk", {}).get("Rd", 0)
    Rd_g = mass_model.get("gas", {}).get("Rd", 0)

    parametric = {
        # Step 1: Enclosed mass M(r) with the three component profiles
        "mass_bulge": (
            r"M_{\mathrm{b}}(<r) = %.4g \times \frac{r^2}{(r + %.2f)^2}"
            % (m_bulge, a_b)
        ),
        "mass_disk": (
            r"M_{\mathrm{d}}(<r) = %.4g \times "
            r"\left[1 - \left(1 + \frac{r}{%.2f}\right)"
            r"e^{-r/%.2f}\right]"
            % (m_disk, Rd_d, Rd_d)
        ),
        "mass_gas": (
            r"M_{\mathrm{g}}(<r) = %.4g \times "
            r"\left[1 - \left(1 + \frac{r}{%.2f}\right)"
            r"e^{-r/%.2f}\right]"
            % (m_gas, Rd_g, Rd_g)
        ),
        "mass_total": (
            r"M(r) = M_{\mathrm{b}}(<r)"
            r" + M_{\mathrm{d}}(<r)"
            r" + M_{\mathrm{g}}(<r)"
        ),
        # Step 2: Newtonian acceleration
        "g_newtonian": (
            r"g_N(r) = \frac{G \cdot M(r)}{r^2}"
        ),
        # Step 3: GFD field solve
        "gfd_solve": (
            r"y_N = \frac{g_N}{a_0}"
            r"\;\;\rightarrow\;\;"
            r" x = \frac{y_N + \sqrt{y_N^2 + 4\,y_N}}{2}"
            r"\;\;\rightarrow\;\;"
            r" g_{\mathrm{GFD}} = a_0 \cdot x"
        ),
        # Step 4a: Throughput correction (outer arm, r > R_t)
        "throughput_outer": (
            r"g_{\mathrm{struct}}(r) = %.4f \times %.4g"
            r" \times \left(\frac{r - %.2f}{%.2f}\right)^{\!3/4}"
            r"\quad r > %.2f\;\mathrm{kpc}"
            % (f_eff, g0, R_t, R_env - R_t, R_t)
        ),
        # Step 4b: Vortex reflection (inner arm, r <= R_t)
        # The sign is negated: outer boost becomes inner suppression
        "vortex_reflect": (
            r"f = \frac{R_t - r}{R_t}"
            r"\;,\quad r_{\mathrm{mirror}} = R_t + f \cdot (R_{\mathrm{env}} - R_t)"
            r"\;,\quad \Delta v(r) = -\Delta v(r_{\mathrm{mirror}})"
        ),
        # Step 5: Final velocity
        "velocity_final": (
            r"v(r) = \sqrt{\bigl(g_{\mathrm{GFD}}(r)"
            r" + g_{\mathrm{struct}}(r)\bigr) \cdot r}"
        ),
        # Constants used
        "constants": (
            r"a_0 = %.6g \;\mathrm{m/s^2}"
            r"\quad G = 6.6743 \times 10^{-11}\;\mathrm{m^3\,kg^{-1}\,s^{-2}}"
            r"\quad R_t = %.2f\;\mathrm{kpc}"
            r"\quad R_{\mathrm{env}} = %.2f\;\mathrm{kpc}"
            % (a0_eff, R_t, R_env)
        ),
        # Units note
        "units": (
            r"\text{Masses in } M_\odot"
            r"\quad \text{Radii in kpc}"
            r"\quad \text{Velocities in km/s}"
        ),
    }

    # --- Field geometry deltas ---
    def _delta_pct(predicted, catalog):
        if catalog and catalog > 0 and predicted:
            return _r((predicted - catalog) / catalog * 100, 1)
        return None

    rt_delta = _delta_pct(R_t_predicted, R_t_catalog)
    renv_delta = _delta_pct(R_env_predicted, R_env_catalog)

    # --- Assemble response ---
    return {
        "field_geometry": {
            "field_origin_kpc": _r(R_t, 2),
            "field_horizon_kpc": _r(R_env, 2),
            "throat_fraction": _r(R_t / R_env if R_env > 0 else 0, 4),
            "origin_throughput_fitted": _r(vortex_strength, 4),
            "origin_throughput_theoretical": _r(theoretical_ot, 4),
            "throughput_delta_pct": _r(
                (vortex_strength - theoretical_ot) / abs(theoretical_ot) * 100
                if theoretical_ot != 0 else 0, 1
            ),
            "catalog_origin_kpc": _r(R_t_catalog, 2),
            "catalog_horizon_kpc": _r(R_env_catalog, 2),
            "predicted_origin_kpc": _r(R_t_predicted, 2) if R_t_predicted else None,
            "predicted_horizon_kpc": _r(R_env_predicted, 2) if R_env_predicted else None,
            "origin_delta_pct": rt_delta,
            "horizon_delta_pct": renv_delta,
            "yN_at_throat": topo_geom.get("yN_at_throat"),
            "prediction_method": "topological (y_N = 18/65)",
        },
        "mass_properties": {
            "total_baryonic_Msun": _r(m_total, 2),
            "bulge_Msun": _r(m_bulge, 2),
            "disk_Msun": _r(m_disk, 2),
            "gas_Msun": _r(m_gas, 2),
            "gas_fraction": _r(f_gas, 4),
            "bulge_to_total": _r(bt_ratio, 4),
        },
        "dynamics": {
            "v_peak_gfd_kms": _r(v_peak_gfd, 1),
            "r_peak_gfd_kpc": _r(r_peak_gfd, 2),
            "v_peak_observed_kms": _r(v_peak_sigma, 1),
            "r_peak_observed_kpc": _r(r_peak_sigma, 2),
            "v_at_origin_gfd_kms": _r(v_at_Rt_gfd, 1),
            "v_at_origin_observed_kms": _r(v_at_Rt_sigma, 1),
            "v_at_horizon_gfd_kms": _r(v_at_Renv_gfd, 1),
            "v_at_horizon_observed_kms": _r(v_at_Renv_sigma, 1),
            "r_transition_kpc": _r(r_transition, 2),
        },
        "field_coupling": {
            "a0_ms2": float(a0_eff),
            "k_simplex": K_SIMPLEX,
            "yN_at_origin": _r(yN_at_Rt, 4),
            "yN_at_horizon": _r(yN_at_Renv, 4),
            "f_eff": _r(f_eff, 4),
            "g0_ms2": float(g0),
            "structural_frac": "4/13",
            "power_law": "3/4",
        },
        "throughput_effect": {
            "delta_v_at_origin_kms": _r(delta_v_at_Rt, 2),
            "delta_v_at_horizon_kms": _r(delta_v_at_Renv, 2),
            "delta_pct_at_horizon": _r(delta_pct_Renv, 1),
            "gfd_newt_ratio_at_origin": _r(gfd_newt_ratio_Rt, 3),
            "gfd_newt_ratio_at_horizon": _r(gfd_newt_ratio_Renv, 3),
        },
        "equations": equations,
        "parametric": parametric,
    }


def _r(val, decimals):
    """Round a value to the given decimal places."""
    return round(float(val), decimals)
