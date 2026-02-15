"""
GRAVIS Core Pipeline: reusable traced computation infrastructure.

This module provides the generic building blocks that any GravisService
can use to build traced computation pipelines. Every computation step
records the equation used, the parameters supplied, all intermediate
values, and the final output series.

Classes:
    GravisStage   - Atomic computation unit with a callable equation
    StageResult   - Immutable record of one stage's execution
    PipelineRunner - Generic pipeline that feeds points through stages

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

from collections import OrderedDict


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
        Stage identifier. Used as the key in result dictionaries.
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


class PipelineRunner:
    """
    Generic traced pipeline: feed points through ordered stages.

    Each service can use this to run a sequence of GravisStage objects
    over a set of sample points. The runner collects all StageResults
    into an OrderedDict keyed by stage name.
    """

    def __init__(self):
        self._stages = []

    def add_stage(self, stage):
        """
        Add a stage to the pipeline.

        Parameters
        ----------
        stage : GravisStage
            The stage to add. Stages execute in insertion order.

        Returns
        -------
        PipelineRunner
            Self, for method chaining.
        """
        self._stages.append(stage)
        return self

    def run(self, radii, enclosed_masses):
        """
        Execute all stages over the given sample points.

        Parameters
        ----------
        radii : list of float
            Independent variable values.
        enclosed_masses : list of float
            Context values (enclosed mass at each radius).

        Returns
        -------
        OrderedDict
            Mapping of stage name to StageResult, in execution order.
        """
        results = OrderedDict()
        for stage in self._stages:
            results[stage.name] = stage.process(radii, enclosed_masses)
        return results
