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
                 galactic_radius=None):
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

            engine.add_stage(GravisStage(
                name="gfd_structure",
                equation=gfd_structure_eq,
                equation_label=(
                    "g = g_DTG + f_gas*(4/13)*G*M_star/R_t^2 "
                    "* [(r-R_t)/(R_env-R_t)]^(3/4), "
                    "R_t = 0.30*R_env"
                ),
                parameters={
                    "accel_ratio": config.accel_ratio,
                    "galactic_radius_kpc": config.galactic_radius,
                    "m_stellar": m_stellar,
                    "f_gas": f_gas,
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
