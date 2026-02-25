"""
GravisEngine: shared pipeline infrastructure for GRAVIS.

ARCHITECTURE RULE: This module is the shared computation core. It
contains ONLY pipeline infrastructure (config, stages, engine, result)
and shared utility functions (compute_fit_metrics, _interpolate).

DO NOT add domain logic, optimization algorithms, physics models, or
service-specific code here. Those belong in their respective service
modules under physics/services/*, following the Registry / Dependency
Injection pattern:

    app.py
      -> GravisRegistry.register(RotationService())
         -> RotationService delegates to:
            - physics.services.rotation.inference  (optimize_inference)
            - physics.sigma                        (GfdSymmetricStage)
            - physics.engine                       (GravisEngine pipeline)

If you need to add new physics:
    1. Create a module under physics/ for shared equations/stages
    2. Create or extend a service under physics/services/ for the logic
    3. Register that service in app.py via the GravisRegistry
    4. The service calls GravisEngine to run the pipeline

This module provides:
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


# ---------------------------------------------------------------
# Backward-compatible re-exports ONLY. No new code belongs here.
#
# Canonical locations (import from these in new code):
#   physics.sigma                        -> GfdSymmetricStage, auto_vortex_strength
#   physics.services.rotation.inference  -> optimize_inference
#
# optimize_inference is NOT re-exported here (circular import).
# Import it from physics.services.rotation.inference directly.
# ---------------------------------------------------------------
from physics.sigma import (                       # noqa: F401, E402
    GfdSymmetricStage,
    auto_vortex_strength,
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
            "gfd_velocity": "gfd_velocity",
            "gfd_structure": "gfd_structure",
            "gfd_symmetric": "gfd_symmetric",
            "mond": "mond",
            "cdm": "cdm",
            "gfd_topological": "gfd_topological",
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


def compute_fit_metrics(radii, gfd_velocities, observations, config):
    """
    Compute fit quality metrics for GFD vs observation data.

    This is the single source of truth for all scientific quality
    metrics displayed in the UI. It lives here so that the same
    function can be called from the prediction endpoint, the
    inference endpoint, and unit tests.

    Parameters
    ----------
    radii : list of float
        Galactocentric radii from the pipeline (kpc).
    gfd_velocities : list of float
        GFD model velocities at each radius (km/s).
    observations : list of dict or None
        Each dict has 'r' (kpc), 'v' (km/s), 'err' (km/s).
    config : GravisConfig
        Pipeline config for mass model and geometric parameters.

    Returns
    -------
    dict
        Metrics dictionary with keys:
        - fit_quality: { rms_km_s, chi2_reduced, within_1sigma,
                         within_2sigma, n_obs }
        - observation_summary: { n_points, r_min_kpc, r_max_kpc,
                                 mean_error_km_s }
        - mass_model: { total_baryonic_M_sun, gas_fraction_pct,
                        field_origin_kpc, field_horizon_kpc }
        - residuals: list of { r_kpc, v_obs, v_gfd, delta_v, sigma }
    """
    result = {}

    # --- Mass Model summary (always available) ---
    mm = config.mass_model
    m_bulge = mm.get("bulge", {}).get("M", 0)
    m_disk = mm.get("disk", {}).get("M", 0)
    m_gas = mm.get("gas", {}).get("M", 0)
    total = m_bulge + m_disk + m_gas
    gas_frac = (m_gas / total * 100.0) if total > 0 else 0.0
    r_env = config.galactic_radius
    r_origin = 0.30 * r_env  # THROAT_FRAC

    result["mass_model"] = {
        "total_baryonic_M_sun": round(total, 2),
        "gas_fraction_pct": round(gas_frac, 1),
        "field_origin_kpc": round(r_origin, 1),
        "field_horizon_kpc": round(r_env, 1),
    }

    # --- Observation-dependent metrics ---
    if not observations or len(observations) == 0:
        result["fit_quality"] = None
        result["observation_summary"] = None
        result["residuals"] = []
        return result

    # Observation summary
    obs_radii = [o["r"] for o in observations]
    obs_errors = [o.get("err", 0) for o in observations if o.get("err", 0) > 0]
    mean_err = sum(obs_errors) / len(obs_errors) if obs_errors else 0.0

    result["observation_summary"] = {
        "n_points": len(observations),
        "r_min_kpc": round(min(obs_radii), 2),
        "r_max_kpc": round(max(obs_radii), 2),
        "mean_error_km_s": round(mean_err, 1),
    }

    # Interpolate GFD velocity at each observation radius
    residuals = []
    sum_sq = 0.0
    chi2 = 0.0
    within_1s = 0
    within_2s = 0
    n_valid = 0

    for obs in observations:
        r_obs = obs["r"]
        v_obs = obs["v"]
        err = obs.get("err", 0)

        # Linear interpolation of GFD curve at r_obs
        v_gfd = _interpolate(radii, gfd_velocities, r_obs)
        if v_gfd is None:
            continue

        dv = v_obs - v_gfd
        sum_sq += dv * dv
        n_valid += 1

        sigma = None
        if err and err > 0:
            chi2 += (dv * dv) / (err * err)
            sigma = abs(dv) / err
            if sigma <= 1.0:
                within_1s += 1
            if sigma <= 2.0:
                within_2s += 1
        else:
            # No error bar: count as within both bands
            within_1s += 1
            within_2s += 1

        residuals.append({
            "r_kpc": round(r_obs, 2),
            "v_obs": round(v_obs, 2),
            "v_gfd": round(v_gfd, 2),
            "delta_v": round(dv, 2),
            "sigma": round(sigma, 2) if sigma is not None else None,
        })

    result["residuals"] = residuals

    if n_valid > 0:
        dof = max(n_valid - 1, 1)
        result["fit_quality"] = {
            "rms_km_s": round(math.sqrt(sum_sq / n_valid), 2),
            "chi2_reduced": round(chi2 / dof, 3),
            "within_1sigma": within_1s,
            "within_2sigma": within_2s,
            "n_obs": n_valid,
        }
    else:
        result["fit_quality"] = None

    return result


def _interpolate(xs, ys, x_target):
    """
    Linear interpolation of y at x_target given sorted (xs, ys).

    Returns None if x_target is outside the data range.
    """
    if not xs or not ys or len(xs) != len(ys):
        return None
    if x_target <= xs[0]:
        return ys[0]
    if x_target >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x_target:
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]
            if x1 == x0:
                return y0
            t = (x_target - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


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

        # Step 1: Compute radii from config.
        # The grid must extend to at least galactic_radius (R_env) so
        # the sigma stage's vortex reflection has full outer delta data.
        # Scale point count proportionally so the per-kpc density stays
        # the same as the originally requested grid.
        grid_max = max(self.config.max_radius, self.config.galactic_radius)
        n_pts = self.config.num_points
        if grid_max > self.config.max_radius and self.config.max_radius > 0:
            scale = grid_max / self.config.max_radius
            n_pts = min(int(self.config.num_points * scale), 500)
        radii = []
        for i in range(n_pts):
            r = (grid_max / n_pts) * (i + 1)
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
        from physics.sst_topological_velocity import gfd_velocity_sst_eq

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

        engine.add_stage(GravisStage(
            name="gfd_velocity",
            equation=gfd_velocity_sst_eq,
            equation_label="g_source=(17/13)*G*M/r^2, g_total=sqrt(g_source*a0+g_source^2), v=sqrt(r*g_total)",
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

        # GFD Topological: signed Burgers vortex (requires observations)
        observations = config.observations
        if observations and len(observations) >= 3:
            from physics.topological import GfdTopologicalStage
            engine.add_stage(GfdTopologicalStage(
                name="gfd_topological",
                equation_label=(
                    "Phase1: classify interior (absorbing/pumping/quiet), "
                    "Phase2: signed Burgers vortex fit, "
                    "v = sqrt(g_total * r)"
                ),
                parameters={
                    "accel_ratio": config.accel_ratio,
                    "observations": observations,
                    "mass_model": config.mass_model,
                },
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


# ===================================================================
# STOP: Do not add new functions, classes, or logic below this line.
#
# This module is the shared pipeline core. All domain-specific logic
# (inference, optimization, physics models) belongs in a GravisService
# under physics/services/ and is registered via the GravisRegistry in
# app.py. See the module docstring for the architectural pattern.
# ===================================================================
