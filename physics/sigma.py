"""
GFD-sigma: Origin Throughput physics for GRAVIS.

Contains the GfdSymmetricStage (single-pass analytical vortex
correction) and the auto_vortex_strength estimator. These are shared
physics used by both the GravisEngine pipeline and the
RotationInferenceService.

Classes:
    GfdSymmetricStage - Single-pass analytical vortex correction

Functions:
    auto_vortex_strength - Estimate throughput from gas leverage

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math

from physics.core import StageResult


class GfdSymmetricStage:
    """
    Origin Throughput stage (GFD-sigma).

    GFD sigma solves the same covariant field equation as GFD base,
    but with the vortex power output modifying the effective
    acceleration scale a0. From the scalar-tensor action:

        F( |nabla Phi|^2 / a0^2 )

    the power output shifts the effective a0 at each radius:

        a0_sigma(r) = a0 * (1 + sigma_amp * lever(r))

    Then the standard GFD field equation is solved:

        x^2 / (1 + x) = gN / a0_sigma
        g = a0_sigma * x
        v = sqrt(g * r)

    The lever is a smooth step function pivoting at R_t:

        lever(r) = tanh(STEEPNESS * (r - R_t) / R_t)

    This is a smooth sign function: -1 inside R_t, 0 at R_t,
    +1 outside R_t, with a rapid but continuous transition.
    Because the lever is FLAT away from R_t, GFD sigma
    follows the same shape as GFD base (same equation, constant
    a0 shift). The equation's nonlinearity produces the
    characteristic GFD curve at every radius.

    Beyond R_env the lever tapers via Gaussian decay, so the
    field returns to the base solution outside the horizon.
    """

    THROAT_FRAC = 0.30

    def __init__(self, name, equation_label, parameters=None):
        self.name = name
        self.equation_label = equation_label
        self.parameters = parameters or {}

    def process(self, radii, enclosed_masses):
        """
        GFD field equation with vortex power output.

        At each radius: compute the lever, modify a0, solve the
        field equation. The equation itself shapes the curve.
        """
        from physics.equations import aqual_solve_x, G, M_SUN, A0, KPC_TO_M

        accel_ratio = self.parameters.get("accel_ratio", 1.0)
        R_env = self.parameters.get("galactic_radius_kpc", 0.0)
        f_gas = self.parameters.get("f_gas", 0.0)
        vortex_strength = self.parameters.get("vortex_strength", 1.0)

        R_t = self.THROAT_FRAC * R_env
        sigma_amp = (1.0 + f_gas) / 2.0 * vortex_strength
        L_outer = R_env - R_t

        n = len(radii)
        a0_base = A0 * accel_ratio

        output_series = []
        intermed = {
            "g_N": [], "g_DTG": [], "g_struct": [], "g_total": [],
            "R_t": [], "f_eff": [], "v_delta": [],
            "vortex_strength": [],
        }

        for i in range(n):
            r_kpc = radii[i]
            m_solar = enclosed_masses[i]

            if r_kpc <= 0 or m_solar <= 0:
                output_series.append(0.0)
                intermed["g_N"].append(0.0)
                intermed["g_DTG"].append(0.0)
                intermed["g_struct"].append(0.0)
                intermed["g_total"].append(0.0)
                intermed["R_t"].append(R_t)
                intermed["f_eff"].append(sigma_amp)
                intermed["v_delta"].append(0.0)
                intermed["vortex_strength"].append(vortex_strength)
                continue

            r_m = r_kpc * KPC_TO_M
            M = m_solar * M_SUN
            gN = G * M / (r_m * r_m)

            # GFD base: standard covariant field equation
            y_N = gN / a0_base
            x_base = aqual_solve_x(y_N)
            g_dtg = a0_base * x_base
            v_base = math.sqrt(g_dtg * r_m) / 1000.0

            # Smooth step lever: tanh transition at R_t, flat on both sides
            # -1 far inside, 0 at R_t, +1 far outside
            # Steepness 3 gives ~95% of full value by 1 R_t away
            STEEPNESS = 3.0
            lever = 0.0
            if R_t > 0:
                lever = math.tanh(STEEPNESS * (r_kpc - R_t) / R_t)
                # Taper beyond R_env so field returns to base
                if r_kpc > R_env and L_outer > 0:
                    peak = math.tanh(STEEPNESS * (R_env - R_t) / R_t)
                    d = (r_kpc - R_env) / L_outer
                    lever = peak * math.exp(-d * d)

            # Power output modifies the acceleration scale
            a0_sigma = a0_base * (1.0 + sigma_amp * lever)
            if a0_sigma <= 0:
                a0_sigma = a0_base * 0.01

            # Solve the SAME field equation with modified a0
            y_sigma = gN / a0_sigma
            x_sigma = aqual_solve_x(y_sigma)
            g_sigma = a0_sigma * x_sigma

            if g_sigma > 0 and r_m > 0:
                v = math.sqrt(g_sigma * r_m) / 1000.0
            else:
                v = 0.0

            v_delta = v - v_base
            g_struct = g_sigma - g_dtg

            output_series.append(v)
            intermed["g_N"].append(gN)
            intermed["g_DTG"].append(g_dtg)
            intermed["g_struct"].append(g_struct)
            intermed["g_total"].append(g_sigma)
            intermed["R_t"].append(R_t)
            intermed["f_eff"].append(sigma_amp)
            intermed["v_delta"].append(v_delta)
            intermed["vortex_strength"].append(vortex_strength)

        return StageResult(
            name=self.name,
            equation_label=self.equation_label,
            parameters=self.parameters,
            series=output_series,
            intermediates=intermed,
        )

    @staticmethod
    def _interp(xs, ys, x_target):
        """Linear interpolation / extrapolation on sorted (xs, ys)."""
        if not xs:
            return 0.0
        if x_target <= xs[0]:
            return ys[0]
        if x_target >= xs[-1]:
            return ys[-1]
        for j in range(len(xs) - 1):
            if xs[j] <= x_target <= xs[j + 1]:
                t = (x_target - xs[j]) / (xs[j + 1] - xs[j])
                return ys[j] + t * (ys[j + 1] - ys[j])
        return ys[-1]


def auto_vortex_strength(mass_model, galactic_radius_kpc):
    """
    Estimate Origin Throughput from the gas leverage around the
    Field Origin (R_t = 0.30 * R_env).

    Gas leverage measures the fraction of total mass that is gas
    beyond R_t, weighted by how far it extends (its lever arm).
    This captures the bidirectional compression hypothesis: stellar
    mass consumed inside R_t compresses the field, while gas outside
    R_t provides a counter-tension. The net balance determines the
    throughput direction and magnitude.

    Uses a log-linear regression on gas_leverage derived from 8
    galaxies with well-constrained throughput values (Spearman r = 0.90):

        sigma = 1.1545 + 1.4066 * log10(gas_leverage)

    where gas_leverage = f_gas_outside * (2*Rd_gas + R_t) / R_t.

    Returns the estimated throughput (can be negative for highly
    concentrated galaxies like the Milky Way).
    """
    R_env = galactic_radius_kpc
    if R_env <= 0:
        return 0.0

    R_t = 0.30 * R_env
    mm = mass_model

    m_bulge = 0.0
    m_disk = 0.0
    if mm.get("bulge") and mm["bulge"].get("M"):
        m_bulge = mm["bulge"]["M"]
    if mm.get("disk") and mm["disk"].get("M"):
        m_disk = mm["disk"]["M"]
    m_stellar = m_bulge + m_disk

    m_gas = 0.0
    Rd_gas = 1.0
    if mm.get("gas") and mm["gas"].get("M"):
        m_gas = mm["gas"]["M"]
        Rd_gas = mm["gas"].get("Rd", 1.0)

    m_total = m_stellar + m_gas
    if m_total <= 0:
        return 0.0

    m_gas_inside = 0.0
    if m_gas > 0 and Rd_gas > 0:
        x = R_t / Rd_gas
        if x < 50:
            m_gas_inside = m_gas * (1.0 - (1.0 + x) * math.exp(-x))
        else:
            m_gas_inside = m_gas

    m_gas_outside = m_gas - m_gas_inside
    f_gas_outside = m_gas_outside / m_total if m_total > 0 else 0.0

    if R_t > 0 and f_gas_outside > 0:
        gas_lever_arm = (Rd_gas * 2.0 + R_t) / R_t
        gas_leverage = f_gas_outside * gas_lever_arm
    else:
        gas_leverage = 0.001

    if gas_leverage <= 0:
        gas_leverage = 0.001

    sigma = 1.1545 + 1.4066 * math.log10(gas_leverage)
    return round(sigma, 2)
