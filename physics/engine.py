"""
GravisEngine: traceable computation pipeline for rotation curve analysis.

This module provides the core pipeline architecture for GRAVIS. Every
computation step (mass model evaluation, theory velocity prediction,
mass inference) flows through GravisStage objects that record the
equation used, the parameters supplied, all intermediate values, and
the final output series. The result is a GravisResult that can produce
either the existing flat API response or a verbose trace of the full
computation chain.

Classes:
    GravisConfig   - Global pipeline parameters (mass model, radii, etc.)
    GravisStage    - Atomic computation unit with a callable equation
    StageResult    - Immutable record of one stage's execution
    GravisEngine   - Orchestrates stages and produces GravisResult
    GravisResult   - Complete pipeline output with serialization methods

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

import math
from collections import OrderedDict


class GravisConfig:
    """
    Global pipeline configuration.

    Holds all parameters that define a pipeline run: the baryonic mass
    model, the radial range, the number of sample points, and the
    acceleration scale multiplier. Optional fields support CDM halo
    mass and observational data for inference pipelines.

    Parameters
    ----------
    mass_model : dict
        Three-component baryonic mass model with keys 'bulge', 'disk',
        'gas'. Each contains profile parameters (M, a or Rd).
    max_radius : float
        Maximum galactocentric radius in kiloparsecs.
    num_points : int
        Number of radial sample points. Capped at 500, minimum 10.
    accel_ratio : float, optional
        Multiplier on the topological acceleration scale a0 (default 1.0).
    m200 : float, optional
        NFW halo virial mass in solar masses. Required for CDM stage.
    observations : list, optional
        Observational data points for inference pipelines. Each entry
        is a dict with keys 'r', 'v', and optionally 'err'.
    """

    def __init__(self, mass_model, max_radius, num_points,
                 accel_ratio=1.0, m200=None, observations=None,
                 galactic_radius=None, vortex_strength=1.0):
        self.mass_model = mass_model
        self.max_radius = float(max_radius)
        # Enforce bounds on num_points
        self.num_points = max(10, min(int(num_points), 500))
        self.accel_ratio = float(accel_ratio)
        self.m200 = m200
        self.observations = observations
        # Galactic radius (gravitational horizon scale) for the manifold.
        # Defaults to max_radius if not explicitly provided.
        # Track whether it was explicitly set so stages that require a
        # measured R_env (like the structural term) can be skipped when
        # the value is just a chart-range default.
        self.galactic_radius_explicit = galactic_radius is not None
        self.galactic_radius = float(galactic_radius) if galactic_radius else self.max_radius
        # Vortex strength modulator for GFD-sigma (default 1.0 = no change).
        # Scales the structural correction amplitude on both the outer lift
        # and the inner vortex reflection symmetrically.
        self.vortex_strength = float(vortex_strength)

    def to_dict(self):
        """Serialize config for inclusion in verbose output."""
        result = {
            "mass_model": self.mass_model,
            "max_radius": self.max_radius,
            "num_points": self.num_points,
            "accel_ratio": self.accel_ratio,
            "galactic_radius": self.galactic_radius,
        }
        if self.m200 is not None:
            result["m200"] = self.m200
        if self.observations is not None:
            result["observations"] = self.observations
        return result


class StageResult:
    """
    Immutable record of one pipeline stage's execution.

    Captures everything that happened during a single stage: the
    equation label, the parameters used, the output series, and all
    intermediate values at every sample point.

    Parameters
    ----------
    name : str
        Stage identifier (e.g. 'newtonian', 'gfd', 'mass_model').
    equation_label : str
        Human-readable equation string for display and traceability.
    parameters : dict
        The parameters that were passed to the equation callable.
    series : list
        Output values at each sample point (velocities or masses).
    intermediates : dict
        Mapping of intermediate variable names to lists of per-point
        values. For example: {"g_N": [...], "y_N": [...], "x": [...]}.
    """

    def __init__(self, name, equation_label, parameters, series,
                 intermediates):
        self.name = name
        self.equation_label = equation_label
        self.parameters = parameters
        self.series = series
        self.intermediates = intermediates

    def to_dict(self):
        """Serialize the full stage trace for verbose output."""
        return {
            "name": self.name,
            "equation": self.equation_label,
            "parameters": self.parameters,
            "series": self.series,
            "intermediates": self.intermediates,
        }


class GravisStage:
    """
    Atomic computation unit in the pipeline.

    Each stage holds a callable equation, a human-readable label for
    that equation, and the parameters the equation requires. The
    process() method runs the equation at every sample point and
    collects outputs and intermediates into a StageResult.

    The equation callable must have the signature:
        (r_kpc: float, m_solar: float, **params) -> (value, intermediates_dict)

    where value is the scalar output (e.g. velocity in km/s) and
    intermediates_dict maps variable names to their values at that point.

    Parameters
    ----------
    name : str
        Stage identifier. Used as the key in GravisResult.stage_results.
    equation : callable
        The computation function. See signature above.
    equation_label : str
        Human-readable equation string (e.g. "x^2/(1+x) = g_N/a0").
    parameters : dict, optional
        Stage-specific parameters passed as **kwargs to the equation.
    """

    def __init__(self, name, equation, equation_label, parameters=None):
        self.name = name
        self.equation = equation
        self.equation_label = equation_label
        self.parameters = parameters or {}

    def process(self, radii, enclosed_masses):
        """
        Run the equation at each (r, m) sample point.

        Parameters
        ----------
        radii : list of float
            Galactocentric radii in kiloparsecs.
        enclosed_masses : list of float
            Enclosed baryonic mass at each radius in solar masses.

        Returns
        -------
        StageResult
            Complete record of the stage execution.
        """
        output_series = []
        # Accumulator for intermediate arrays, keyed by variable name
        intermed_accum = {}

        for r, m in zip(radii, enclosed_masses):
            value, intermediates = self.equation(r, m, **self.parameters)
            output_series.append(value)

            # On first iteration, initialize accumulator keys
            if not intermed_accum:
                for key in intermediates:
                    intermed_accum[key] = []

            for key, val in intermediates.items():
                intermed_accum[key].append(val)

        return StageResult(
            name=self.name,
            equation_label=self.equation_label,
            parameters=self.parameters,
            series=output_series,
            intermediates=intermed_accum,
        )


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

    # Stellar mass
    m_bulge = 0.0
    m_disk = 0.0
    if mm.get("bulge") and mm["bulge"].get("M"):
        m_bulge = mm["bulge"]["M"]
    if mm.get("disk") and mm["disk"].get("M"):
        m_disk = mm["disk"]["M"]
    m_stellar = m_bulge + m_disk

    # Gas mass and scale length
    m_gas = 0.0
    Rd_gas = 1.0
    if mm.get("gas") and mm["gas"].get("M"):
        m_gas = mm["gas"]["M"]
        Rd_gas = mm["gas"].get("Rd", 1.0)

    m_total = m_stellar + m_gas
    if m_total <= 0:
        return 0.0

    # Gas mass inside R_t (exponential disk profile)
    m_gas_inside = 0.0
    if m_gas > 0 and Rd_gas > 0:
        x = R_t / Rd_gas
        if x < 50:
            m_gas_inside = m_gas * (1.0 - (1.0 + x) * math.exp(-x))
        else:
            m_gas_inside = m_gas

    # Gas mass outside R_t
    m_gas_outside = m_gas - m_gas_inside

    # Fraction of total mass that is gas beyond R_t
    f_gas_outside = m_gas_outside / m_total if m_total > 0 else 0.0

    # Gas leverage: f_gas_outside weighted by lever arm
    # (how far the gas extends beyond R_t relative to R_t)
    if R_t > 0 and f_gas_outside > 0:
        gas_lever_arm = (Rd_gas * 2.0 + R_t) / R_t
        gas_leverage = f_gas_outside * gas_lever_arm
    else:
        gas_leverage = 0.001  # floor to avoid log(0)

    # Clamp to avoid log domain issues
    if gas_leverage <= 0:
        gas_leverage = 0.001

    # Log-linear regression: sigma = 1.1545 + 1.4066 * log10(gl)
    sigma = 1.1545 + 1.4066 * math.log10(gas_leverage)
    return round(sigma, 2)


class GfdSymmetricStage:
    """
    Origin Throughput stage (GFD-sigma).

    The Field Origin (R_t = 0.30 * R_env) acts as an aperture for
    structural energy flow. Stellar mass compressed inside R_t drives
    the field inward; gas extending beyond R_t provides a counter-
    tension (lever arm). The net balance, the Origin Throughput,
    determines the direction and amplitude of the structural correction.

    Outer arm (r > R_t): structural release with 3/4 power law,
    scaled by the effective gas fraction and throughput modulator.

    Inner arm (r < R_t): vortex reflection of the outer delta profile.
    Each inner radius maps to a mirror point on the outer arm:
        f = (R_t - r) / R_t           fractional depth into core
        r_mirror = R_t + f*(R_env-R_t) mirror point on outer arm
        delta_inner(r) = delta_outer(r_mirror)

    The throughput can be auto-calculated from gas leverage
    (Spearman r = 0.90 across 8 calibration galaxies) or set
    manually by the researcher.
    """

    THROAT_FRAC = 0.30
    STRUCT_FRAC = 4.0 / 13.0   # 0.3077
    P_OUTER = 3.0 / 4.0        # d/k = 0.75

    def __init__(self, name, equation_label, parameters=None):
        self.name = name
        self.equation_label = equation_label
        self.parameters = parameters or {}

    def process(self, radii, enclosed_masses):
        """
        Two-pass vortex-reflection computation.

        Pass 1: compute DTG base + outer structural boost for r > R_t.
        Pass 2: reflect outer delta profile into inner region via
                vortex mapping.
        """
        from physics.equations import aqual_solve_x, G, M_SUN, A0, KPC_TO_M

        accel_ratio = self.parameters.get("accel_ratio", 1.0)
        R_env = self.parameters.get("galactic_radius_kpc", 0.0)
        m_stellar = self.parameters.get("m_stellar", 0.0)
        f_gas = self.parameters.get("f_gas", 0.0)
        vortex_strength = self.parameters.get("vortex_strength", 1.0)

        R_t = self.THROAT_FRAC * R_env
        f_eff = (1.0 + f_gas) / 2.0 * vortex_strength
        L_outer = R_env - R_t

        # Pre-compute reference acceleration
        g0 = 0.0
        if m_stellar > 0 and R_t > 0:
            R_t_m = R_t * KPC_TO_M
            g0 = self.STRUCT_FRAC * G * m_stellar * M_SUN / (R_t_m * R_t_m)

        # ----------------------------------------------------------
        # Pass 1: DTG base + outer velocity deltas
        # ----------------------------------------------------------
        n = len(radii)
        v_base = [0.0] * n
        v_outer_delta = [0.0] * n
        g_dtg_arr = [0.0] * n
        g_struct_arr = [0.0] * n

        # Collect outer (r, delta) pairs for interpolation
        outer_r = []
        outer_dv = []

        for i in range(n):
            r_kpc = radii[i]
            m_solar = enclosed_masses[i]

            if r_kpc <= 0 or m_solar <= 0:
                continue

            r_m = r_kpc * KPC_TO_M
            M = m_solar * M_SUN
            gN = G * M / (r_m * r_m)
            a0_eff = A0 * accel_ratio
            y_N = gN / a0_eff
            x = aqual_solve_x(y_N)
            g_dtg = a0_eff * x
            g_dtg_arr[i] = g_dtg
            v_base[i] = math.sqrt(g_dtg * r_m) / 1000.0

            if r_kpc > R_t and R_env > R_t and g0 > 0:
                xi = (r_kpc - R_t) / L_outer
                g_s = f_eff * g0 * (xi ** self.P_OUTER)
                g_struct_arr[i] = g_s
                g_total = g_dtg + g_s
                if g_total > 0:
                    v_enh = math.sqrt(g_total * r_m) / 1000.0
                else:
                    v_enh = 0.0
                dv = v_enh - v_base[i]
                v_outer_delta[i] = dv
                if r_kpc <= R_env:
                    outer_r.append(r_kpc)
                    outer_dv.append(dv)

        # ----------------------------------------------------------
        # Pass 2: vortex reflection into inner region
        # ----------------------------------------------------------
        # For each inner point, map to mirror outer radius and
        # interpolate the outer delta.
        output_series = []
        intermed = {
            "g_N": [], "g_DTG": [], "g_struct": [], "g_total": [],
            "R_t": [], "f_eff": [], "v_delta": [],
            "vortex_strength": [],
        }

        for i in range(n):
            r_kpc = radii[i]
            r_m = r_kpc * KPC_TO_M if r_kpc > 0 else 0.0
            vb = v_base[i]
            v_delta = 0.0
            g_s = 0.0

            if r_kpc <= 0 or enclosed_masses[i] <= 0:
                v = 0.0
            elif r_kpc > R_t:
                # Outer: use computed structural release
                v_delta = v_outer_delta[i]
                v = max(vb + v_delta, 0.0)
                g_s = g_struct_arr[i]
            elif r_kpc < R_t and R_t > 0 and L_outer > 0 and outer_r:
                # Inner: vortex reflection
                f_depth = (R_t - r_kpc) / R_t
                r_mirror = R_t + f_depth * L_outer
                # Linear interpolation of outer delta at r_mirror
                dv_mirror = self._interp(outer_r, outer_dv, r_mirror)
                v_delta = -dv_mirror
                v = max(vb + v_delta, 0.0)
                g_s = -(dv_mirror / vb) * g_dtg_arr[i] if vb > 0 else 0.0
            else:
                v = vb

            output_series.append(v)
            intermed["g_N"].append(
                G * enclosed_masses[i] * M_SUN / (r_m * r_m)
                if r_m > 0 else 0.0
            )
            intermed["g_DTG"].append(g_dtg_arr[i])
            intermed["g_struct"].append(g_s)
            intermed["g_total"].append(g_dtg_arr[i] + g_s)
            intermed["R_t"].append(R_t)
            intermed["f_eff"].append(f_eff)
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


class GravisResult:
    """
    Complete pipeline output.

    Contains the configuration, computed radii, mass model trace, and
    all theory stage results. Provides two serialization methods:
    to_api_response() for the existing flat API format, and
    to_verbose_response() for the full computation chain.

    Parameters
    ----------
    config : GravisConfig
        The pipeline configuration that produced this result.
    radii : list of float
        Galactocentric radii in kiloparsecs.
    mass_result : StageResult
        Mass model evaluation trace with per-component breakdown.
    stage_results : OrderedDict
        Mapping of stage name to StageResult, in execution order.
    """

    def __init__(self, config, radii, mass_result, stage_results):
        self.config = config
        self.radii = radii
        self.mass_result = mass_result
        self.stage_results = stage_results

    def series(self, name):
        """
        Get the output series for a named stage.

        Parameters
        ----------
        name : str
            Stage name (e.g. 'newtonian', 'gfd', 'mond', 'cdm').

        Returns
        -------
        list of float
            Output values at each sample point.

        Raises
        ------
        KeyError
            If no stage with that name exists in the result.
        """
        return self.stage_results[name].series

    def to_api_response(self):
        """
        Produce the existing flat dict for the API.

        Returns the identical JSON structure that the inline loop in
        routes.py previously produced. This ensures zero impact on the
        frontend or existing tests.

        Returns
        -------
        dict
            Flat response with keys: radii, newtonian, dtg, mond, cdm,
            enclosed_mass.
        """
        response = {
            "radii": [round(r, 6) for r in self.radii],
            "enclosed_mass": [round(m, 2) for m in self.mass_result.series],
        }

        # Map internal stage names to API response keys.
        # The GFD stage is named 'gfd' internally but the API key is 'dtg'
        # for historical compatibility.
        api_key_map = {
            "newtonian": "newtonian",
            "gfd": "dtg",
            "gfd_structure": "gfd_structure",
            "gfd_symmetric": "gfd_symmetric",
            "mond": "mond",
            "cdm": "cdm",
        }

        for stage_name, result in self.stage_results.items():
            api_key = api_key_map.get(stage_name, stage_name)
            response[api_key] = [round(v, 4) for v in result.series]

        return response

    def to_verbose_response(self):
        """
        Produce the full computation chain for verbose/advanced mode.

        Returns every stage with its equation, parameters, intermediates,
        and output series. This is the response that the future
        verbose=true query parameter will expose.

        Returns
        -------
        dict
            Complete traced response with config, radii, mass_model
            trace, and per-stage traces.
        """
        return {
            "config": self.config.to_dict(),
            "radii": self.radii,
            "mass_model": self.mass_result.to_dict(),
            "stages": OrderedDict(
                (name, result.to_dict())
                for name, result in self.stage_results.items()
            ),
        }


class GravisEngine:
    """
    Pipeline orchestrator for GRAVIS computations.

    The engine takes a GravisConfig, accepts stages via add_stage(),
    and produces a GravisResult when run() is called. The mass model
    is handled as a special first stage (set via set_mass_stage) whose
    output feeds into all subsequent theory stages.

    Factory methods (rotation_curve, inference) provide pre-wired
    pipelines for common use cases.

    Parameters
    ----------
    config : GravisConfig
        Global pipeline configuration.
    """

    def __init__(self, config):
        self.config = config
        self._mass_equation = None
        self._mass_equation_label = None
        self._mass_parameters = {}
        self._stages = []

    def set_mass_stage(self, equation, equation_label, parameters=None):
        """
        Set the mass model equation.

        The mass stage runs before all theory stages and produces the
        enclosed mass series that feeds into every theory stage. It has
        a different signature from theory stages: (r_kpc, mass_model)
        instead of (r_kpc, m_enclosed, **params).

        Parameters
        ----------
        equation : callable
            Mass model function with signature
            (r_kpc, mass_model) -> (m_total, intermediates_dict).
        equation_label : str
            Human-readable label for the mass model equations.
        parameters : dict, optional
            Additional parameters (reserved for future use).

        Returns
        -------
        GravisEngine
            Self, for method chaining.
        """
        self._mass_equation = equation
        self._mass_equation_label = equation_label
        self._mass_parameters = parameters or {}
        return self

    def add_stage(self, stage):
        """
        Add a theory stage to the pipeline.

        Stages are executed in the order they are added. Each stage
        receives the radii and enclosed masses computed by the mass
        model stage.

        Parameters
        ----------
        stage : GravisStage
            The stage to add.

        Returns
        -------
        GravisEngine
            Self, for method chaining.
        """
        self._stages.append(stage)
        return self

    def run(self):
        """
        Execute the full pipeline.

        Computes radii from the config, runs the mass model stage,
        then runs each theory stage in order. Collects everything
        into a GravisResult.

        Returns
        -------
        GravisResult
            Complete pipeline output with all traces.

        Raises
        ------
        ValueError
            If no mass stage has been set.
        """
        if self._mass_equation is None:
            raise ValueError(
                "No mass stage configured. Call set_mass_stage() or use "
                "a factory method like GravisEngine.rotation_curve()."
            )

        # Step 1: Compute radii from config
        radii = []
        for i in range(self.config.num_points):
            r = (self.config.max_radius / self.config.num_points) * (i + 1)
            radii.append(r)

        # Step 2: Run mass model stage at each radius
        mass_series = []
        mass_intermed_accum = {}
        mass_model = self.config.mass_model

        for r in radii:
            m_total, intermediates = self._mass_equation(r, mass_model)
            mass_series.append(m_total)

            # Initialize accumulator on first iteration
            if not mass_intermed_accum:
                for key in intermediates:
                    mass_intermed_accum[key] = []

            for key, val in intermediates.items():
                mass_intermed_accum[key].append(val)

        # Build mass model parameters dict for traceability.
        # Include the full mass model definition so the trace records
        # exactly which profile parameters were used.
        mass_params = dict(self._mass_parameters)
        mass_params["mass_model"] = mass_model

        mass_result = StageResult(
            name="mass_model",
            equation_label=self._mass_equation_label,
            parameters=mass_params,
            series=mass_series,
            intermediates=mass_intermed_accum,
        )

        # Step 3: Run each theory stage
        stage_results = OrderedDict()
        for stage in self._stages:
            result = stage.process(radii, mass_series)
            stage_results[stage.name] = result

        # Step 4: Assemble and return
        return GravisResult(
            config=self.config,
            radii=radii,
            mass_result=mass_result,
            stage_results=stage_results,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def rotation_curve(cls, config, m200=None):
        """
        Factory: standard 6-theory rotation curve pipeline.

        Creates an engine with the standard mass model stage and six
        theory stages (Newtonian, GFD, GFD Manifold, GFD Poisson,
        MOND, CDM). This is the pipeline that the /api/rotation-curve
        endpoint uses.

        Parameters
        ----------
        config : GravisConfig
            Pipeline configuration with mass model and parameters.
        m200 : float, optional
            NFW halo virial mass for the CDM stage. If None, CDM stage
            uses 0 for M200 (effectively no halo).

        Returns
        -------
        GravisEngine
            Fully configured engine ready to run().
        """
        # Import here to avoid circular imports. The equations module
        # imports from the physics solvers, not from the engine.
        from physics.equations import (
            mass_model_eq,
            newtonian_eq,
            gfd_eq,
            gfd_structure_eq,
            mond_eq,
            cdm_eq,
        )

        engine = cls(config)

        # Mass model stage: Hernquist bulge + exponential disk + gas
        engine.set_mass_stage(
            equation=mass_model_eq,
            equation_label=(
                "M_bulge(<r) = M * r^2 / (r + a)^2; "
                "M_disk(<r) = M * [1 - (1 + r/Rd) * exp(-r/Rd)]; "
                "M_gas(<r) = same exponential form"
            ),
        )

        # Theory stages, in the order they appear in the API response
        engine.add_stage(GravisStage(
            name="newtonian",
            equation=newtonian_eq,
            equation_label="v = sqrt(G * M / r)",
        ))

        engine.add_stage(GravisStage(
            name="gfd",
            equation=gfd_eq,
            equation_label="x^2/(1+x) = g_N/a0, v = sqrt(a0 * x * r)",
            parameters={"accel_ratio": config.accel_ratio},
        ))

        # GFD+: only include when galactic_radius
        # (R_env) was explicitly provided.  The structural term requires a
        # measured baryonic horizon; when galactic_radius just defaults to
        # the chart range (max_radius) the amplitude becomes arbitrary and
        # the curve blows up.
        if config.galactic_radius_explicit:
            mm = config.mass_model
            m_stellar = 0.0
            m_total = 0.0
            if mm.get("bulge") and mm["bulge"].get("M"):
                m_stellar += mm["bulge"]["M"]
                m_total += mm["bulge"]["M"]
            if mm.get("disk") and mm["disk"].get("M"):
                m_stellar += mm["disk"]["M"]
                m_total += mm["disk"]["M"]
            m_gas = 0.0
            if mm.get("gas") and mm["gas"].get("M"):
                m_gas = mm["gas"]["M"]
                m_total += m_gas
            f_gas = m_gas / m_total if m_total > 0 else 0.0

            engine.add_stage(GfdSymmetricStage(
                name="gfd_symmetric",
                equation_label=(
                    "Origin Throughput: structural release xi^(3/4) "
                    "with vortex reflection through R_t = 0.30*R_env, "
                    "auto-scaled by gas leverage"
                ),
                parameters={
                    "accel_ratio": config.accel_ratio,
                    "galactic_radius_kpc": config.galactic_radius,
                    "m_stellar": m_stellar,
                    "f_gas": f_gas,
                    "vortex_strength": config.vortex_strength,
                },
            ))

        engine.add_stage(GravisStage(
            name="mond",
            equation=mond_eq,
            equation_label=(
                "mu(x) = x/sqrt(1+x^2), mu(x)*x = g_N/a0, "
                "v = sqrt(a0 * x * r)"
            ),
            parameters={"accel_ratio": config.accel_ratio},
        ))

        engine.add_stage(GravisStage(
            name="cdm",
            equation=cdm_eq,
            equation_label="v = sqrt(v_baryon^2 + v_NFW^2)",
            parameters={"m200": m200 or 0.0},
        ))

        return engine

    @staticmethod
    def infer_mass(r_kpc, v_km_s, accel_ratio=1.0):
        """
        Single-point mass inference with full intermediate trace.

        Evaluates the covariant field equation in the inverse direction
        to infer the enclosed baryonic mass from an observed velocity.
        Returns both the inferred mass and all intermediate quantities
        so the computation is fully traceable.

        This is a lightweight convenience method. It does not require
        a GravisConfig or multi-stage pipeline since inference at a
        single point is an atomic operation.

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
        StageResult
            Traceable result containing:
              - series: [inferred_mass_solar]
              - intermediates: {"g_eff": [...], "x": [...], "g_N": [...]}
        """
        from physics.equations import inference_eq

        mass, intermediates = inference_eq(r_kpc, v_km_s,
                                           accel_ratio=accel_ratio)

        return StageResult(
            name="inference",
            equation_label=(
                "v -> g=v^2/r -> x=g/a0 -> "
                "g_N=a0*x^2/(1+x) -> M=g_N*r^2/G"
            ),
            parameters={
                "r_kpc": r_kpc,
                "v_km_s": v_km_s,
                "accel_ratio": accel_ratio,
            },
            series=[mass],
            intermediates={k: [v] for k, v in intermediates.items()},
        )
