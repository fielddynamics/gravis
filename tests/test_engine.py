"""
Tests for the GravisEngine traceable computation pipeline.

Verifies:
  1. Core class construction and serialization (GravisConfig, StageResult,
     GravisStage, GravisResult, GravisEngine).
  2. Numerical parity: engine output must exactly match the original
     inline computation for every theory (Newtonian, GFD, MOND, CDM).
  3. Stage introspection: intermediates, equation labels, and parameter
     recording are present and correct.
  4. Inference parity: engine inference produces identical results to
     the standalone infer_mass() function.
  5. Edge cases: zero mass, zero radius, missing stages.

These tests are independent of Flask and the API layer. They test the
engine classes directly.
"""

import math
import pytest

from physics.engine import (
    GravisConfig,
    GravisStage,
    StageResult,
    GravisEngine,
    GravisResult,
)
from physics.equations import (
    mass_model_eq,
    newtonian_eq,
    gfd_eq,
    mond_eq,
    cdm_eq,
    inference_eq,
)

# Reference implementations for parity checks
from physics.mass_model import enclosed_mass
from physics.newtonian import velocity as ref_newtonian
from physics.aqual import velocity as ref_gfd
from physics.mond import velocity as ref_mond
from physics.nfw import cdm_velocity as ref_cdm
from physics.inference import infer_mass as ref_infer_mass


# -----------------------------------------------------------------------
# Standard test fixture: Milky Way mass model
# -----------------------------------------------------------------------
MILKY_WAY_MODEL = {
    "bulge": {"M": 1.5e10, "a": 0.6},
    "disk":  {"M": 5.0e10, "Rd": 2.5},
    "gas":   {"M": 1.0e10, "Rd": 5.0},
}


# -----------------------------------------------------------------------
# GravisConfig tests
# -----------------------------------------------------------------------
class TestGravisConfig:
    """Verify GravisConfig construction and parameter bounds."""

    def test_basic_construction(self):
        config = GravisConfig(
            mass_model=MILKY_WAY_MODEL,
            max_radius=30.0,
            num_points=100,
        )
        assert config.mass_model == MILKY_WAY_MODEL
        assert config.max_radius == 30.0
        assert config.num_points == 100
        assert config.accel_ratio == 1.0
        assert config.m200 is None

    def test_num_points_capped_at_500(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 999)
        assert config.num_points == 500

    def test_num_points_minimum_10(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 3)
        assert config.num_points == 10

    def test_to_dict_includes_all_fields(self):
        config = GravisConfig(
            MILKY_WAY_MODEL, 30, 100, accel_ratio=1.5, m200=1e12
        )
        d = config.to_dict()
        assert d["max_radius"] == 30.0
        assert d["num_points"] == 100
        assert d["accel_ratio"] == 1.5
        assert d["m200"] == 1e12
        assert d["mass_model"] == MILKY_WAY_MODEL

    def test_to_dict_omits_none_m200(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 100)
        d = config.to_dict()
        assert "m200" not in d

    def test_to_dict_omits_none_observations(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 100)
        d = config.to_dict()
        assert "observations" not in d


# -----------------------------------------------------------------------
# StageResult tests
# -----------------------------------------------------------------------
class TestStageResult:
    """Verify StageResult construction and serialization."""

    def test_to_dict_structure(self):
        result = StageResult(
            name="test_stage",
            equation_label="v = sqrt(G*M/r)",
            parameters={"accel_ratio": 1.0},
            series=[100.0, 150.0, 200.0],
            intermediates={"g_N": [1e-10, 2e-10, 3e-10]},
        )
        d = result.to_dict()
        assert d["name"] == "test_stage"
        assert d["equation"] == "v = sqrt(G*M/r)"
        assert d["parameters"] == {"accel_ratio": 1.0}
        assert len(d["series"]) == 3
        assert "g_N" in d["intermediates"]
        assert len(d["intermediates"]["g_N"]) == 3


# -----------------------------------------------------------------------
# GravisStage tests
# -----------------------------------------------------------------------
class TestGravisStage:
    """Verify GravisStage processes equations correctly."""

    def test_process_collects_output_and_intermediates(self):
        """Stage.process runs the equation at each (r, m) pair."""
        stage = GravisStage(
            name="newtonian",
            equation=newtonian_eq,
            equation_label="v = sqrt(G*M/r)",
        )
        radii = [5.0, 10.0, 15.0]
        masses = [3e10, 5e10, 6e10]
        result = stage.process(radii, masses)

        assert result.name == "newtonian"
        assert len(result.series) == 3
        assert all(v > 0 for v in result.series)
        # Newtonian equation records g_N as an intermediate
        assert "g_N" in result.intermediates
        assert len(result.intermediates["g_N"]) == 3


# -----------------------------------------------------------------------
# Numerical parity: equations vs. reference implementations
# -----------------------------------------------------------------------
class TestEquationParity:
    """
    Every traced equation must produce the exact same numerical output
    as the original reference function. We test at multiple radii and
    mass values to ensure no unit conversion or arithmetic divergence.
    """

    @pytest.mark.parametrize("r_kpc,m_solar", [
        (0.5, 1e9),
        (2.0, 1e10),
        (8.0, 5e10),
        (15.0, 6.5e10),
        (30.0, 7.0e10),
    ])
    def test_newtonian_parity(self, r_kpc, m_solar):
        v_engine, _ = newtonian_eq(r_kpc, m_solar)
        v_ref = ref_newtonian(r_kpc, m_solar)
        assert v_engine == pytest.approx(v_ref, rel=1e-12)

    @pytest.mark.parametrize("r_kpc,m_solar", [
        (0.5, 1e9),
        (2.0, 1e10),
        (8.0, 5e10),
        (15.0, 6.5e10),
        (30.0, 7.0e10),
    ])
    def test_gfd_parity(self, r_kpc, m_solar):
        v_engine, _ = gfd_eq(r_kpc, m_solar, accel_ratio=1.0)
        v_ref = ref_gfd(r_kpc, m_solar, 1.0)
        assert v_engine == pytest.approx(v_ref, rel=1e-12)

    @pytest.mark.parametrize("r_kpc,m_solar", [
        (0.5, 1e9),
        (2.0, 1e10),
        (8.0, 5e10),
        (15.0, 6.5e10),
        (30.0, 7.0e10),
    ])
    def test_mond_parity(self, r_kpc, m_solar):
        v_engine, _ = mond_eq(r_kpc, m_solar, accel_ratio=1.0)
        v_ref = ref_mond(r_kpc, m_solar, 1.0)
        assert v_engine == pytest.approx(v_ref, rel=1e-12)

    @pytest.mark.parametrize("r_kpc,m_solar,m200", [
        (5.0, 3e10, 1e12),
        (8.0, 5e10, 1.5e12),
        (20.0, 6.5e10, 2e12),
    ])
    def test_cdm_parity(self, r_kpc, m_solar, m200):
        v_engine, _ = cdm_eq(r_kpc, m_solar, m200=m200)
        v_ref = ref_cdm(r_kpc, m_solar, m200)
        assert v_engine == pytest.approx(v_ref, rel=1e-10)

    @pytest.mark.parametrize("r_kpc,m_solar", [
        (2.0, 1e10),
        (8.0, 5e10),
        (15.0, 6.5e10),
    ])
    def test_mass_model_parity(self, r_kpc, m_solar):
        """Mass model equation produces the same total as the reference."""
        m_engine, _ = mass_model_eq(r_kpc, MILKY_WAY_MODEL)
        m_ref = enclosed_mass(r_kpc, MILKY_WAY_MODEL)
        assert m_engine == pytest.approx(m_ref, rel=1e-12)

    @pytest.mark.parametrize("r_kpc,v_km_s", [
        (5.0, 200.0),
        (8.0, 230.0),
        (15.0, 220.0),
        (30.0, 210.0),
    ])
    def test_inference_parity(self, r_kpc, v_km_s):
        m_engine, _ = inference_eq(r_kpc, v_km_s, accel_ratio=1.0)
        m_ref = ref_infer_mass(r_kpc, v_km_s, 1.0)
        assert m_engine == pytest.approx(m_ref, rel=1e-12)


# -----------------------------------------------------------------------
# Equation intermediates: verify physics variables are captured
# -----------------------------------------------------------------------
class TestEquationIntermediates:
    """Verify that each equation callable returns meaningful intermediates."""

    def test_newtonian_returns_g_N(self):
        _, intermed = newtonian_eq(8.0, 5e10)
        assert "g_N" in intermed
        assert intermed["g_N"] > 0

    def test_gfd_returns_full_chain(self):
        _, intermed = gfd_eq(8.0, 5e10)
        for key in ("g_N", "y_N", "x", "g_eff"):
            assert key in intermed
            assert intermed[key] > 0

    def test_mond_returns_mu_x(self):
        _, intermed = mond_eq(8.0, 5e10)
        for key in ("g_N", "y_N", "x", "mu_x", "g_eff"):
            assert key in intermed
        # mu(x) must be between 0 and 1
        assert 0 < intermed["mu_x"] <= 1.0

    def test_cdm_returns_component_velocities(self):
        _, intermed = cdm_eq(8.0, 5e10, m200=1.5e12)
        assert "v_baryon_km_s" in intermed
        assert "v_nfw_km_s" in intermed
        assert "m_nfw_enclosed" in intermed
        assert intermed["v_baryon_km_s"] > 0
        assert intermed["v_nfw_km_s"] > 0

    def test_mass_model_returns_components(self):
        _, intermed = mass_model_eq(8.0, MILKY_WAY_MODEL)
        assert "m_bulge" in intermed
        assert "m_disk" in intermed
        assert "m_gas" in intermed
        # All components should be positive at 8 kpc for the Milky Way
        assert intermed["m_bulge"] > 0
        assert intermed["m_disk"] > 0
        assert intermed["m_gas"] > 0

    def test_inference_returns_chain(self):
        _, intermed = inference_eq(8.0, 230.0)
        for key in ("g_eff", "x", "g_N"):
            assert key in intermed
            assert intermed[key] > 0


# -----------------------------------------------------------------------
# Edge cases: zero and negative inputs
# -----------------------------------------------------------------------
class TestEdgeCases:
    """Verify graceful handling of zero/negative inputs."""

    def test_newtonian_zero_radius(self):
        v, intermed = newtonian_eq(0.0, 5e10)
        assert v == 0.0
        assert intermed["g_N"] == 0.0

    def test_newtonian_zero_mass(self):
        v, intermed = newtonian_eq(8.0, 0.0)
        assert v == 0.0

    def test_gfd_zero_radius(self):
        v, intermed = gfd_eq(0.0, 5e10)
        assert v == 0.0
        assert all(val == 0.0 for val in intermed.values())

    def test_mond_zero_mass(self):
        v, intermed = mond_eq(8.0, 0.0)
        assert v == 0.0

    def test_cdm_zero_radius(self):
        v, intermed = cdm_eq(0.0, 5e10, m200=1e12)
        assert v == 0.0

    def test_mass_model_zero_radius(self):
        m, intermed = mass_model_eq(0.0, MILKY_WAY_MODEL)
        assert m == 0.0
        assert all(val == 0.0 for val in intermed.values())

    def test_inference_zero_radius(self):
        m, intermed = inference_eq(0.0, 230.0)
        assert m == 0.0

    def test_inference_zero_velocity(self):
        m, intermed = inference_eq(8.0, 0.0)
        assert m == 0.0


# -----------------------------------------------------------------------
# GravisEngine: full pipeline tests
# -----------------------------------------------------------------------
class TestGravisEnginePipeline:
    """Test the full rotation curve pipeline end to end."""

    def _make_config(self, num_points=50, max_radius=30.0):
        return GravisConfig(
            mass_model=MILKY_WAY_MODEL,
            max_radius=max_radius,
            num_points=num_points,
        )

    def test_rotation_curve_factory_produces_result(self):
        config = self._make_config()
        engine = GravisEngine.rotation_curve(config, m200=1.5e12)
        result = engine.run()
        assert isinstance(result, GravisResult)

    def test_result_has_all_stages(self):
        config = self._make_config()
        engine = GravisEngine.rotation_curve(config, m200=1.5e12)
        result = engine.run()
        assert "newtonian" in result.stage_results
        assert "gfd" in result.stage_results
        assert "mond" in result.stage_results
        assert "cdm" in result.stage_results

    def test_radii_length_matches_num_points(self):
        config = self._make_config(num_points=75)
        result = GravisEngine.rotation_curve(config).run()
        assert len(result.radii) == 75

    def test_all_series_lengths_consistent(self):
        config = self._make_config(num_points=50)
        result = GravisEngine.rotation_curve(config).run()
        n = len(result.radii)
        assert len(result.mass_result.series) == n
        for stage_result in result.stage_results.values():
            assert len(stage_result.series) == n

    def test_mass_model_result_has_component_intermediates(self):
        config = self._make_config()
        result = GravisEngine.rotation_curve(config).run()
        mass_intermed = result.mass_result.intermediates
        assert "m_bulge" in mass_intermed
        assert "m_disk" in mass_intermed
        assert "m_gas" in mass_intermed

    def test_series_accessor(self):
        config = self._make_config()
        result = GravisEngine.rotation_curve(config).run()
        newton_series = result.series("newtonian")
        assert len(newton_series) == config.num_points
        assert all(v > 0 for v in newton_series)

    def test_series_accessor_raises_on_invalid_name(self):
        config = self._make_config()
        result = GravisEngine.rotation_curve(config).run()
        with pytest.raises(KeyError):
            result.series("nonexistent_stage")

    def test_engine_raises_without_mass_stage(self):
        config = self._make_config()
        engine = GravisEngine(config)
        with pytest.raises(ValueError, match="No mass stage configured"):
            engine.run()

    def test_gfd_above_newtonian_at_outer_radii(self):
        """
        GFD velocity must exceed Newtonian at large radii where the
        field coupling enhancement is significant. This is a physics
        sanity check on the pipeline output.
        """
        config = self._make_config(num_points=100, max_radius=30.0)
        result = GravisEngine.rotation_curve(config).run()
        newton = result.series("newtonian")
        gfd = result.series("gfd")
        # Check the outer 20% of radii
        outer_start = int(0.8 * len(newton))
        for i in range(outer_start, len(newton)):
            assert gfd[i] >= newton[i], (
                f"GFD should exceed Newtonian at r={result.radii[i]} kpc"
            )


# -----------------------------------------------------------------------
# Parity: engine pipeline vs. inline computation
# -----------------------------------------------------------------------
class TestPipelineParityWithInline:
    """
    The engine pipeline must produce numerically identical results to
    the original inline computation loop that was in routes.py. This
    is the critical correctness guarantee for the refactor.
    """

    def test_full_milky_way_parity(self):
        """
        Run the engine and the original inline loop side by side for
        the Milky Way. Every velocity at every radius must match.
        """
        max_radius = 30.0
        num_points = 50
        accel_ratio = 1.0
        m200 = 1.5e12

        # Engine pipeline
        config = GravisConfig(MILKY_WAY_MODEL, max_radius, num_points)
        result = GravisEngine.rotation_curve(config, m200=m200).run()

        # Original inline computation
        for i in range(num_points):
            r = (max_radius / num_points) * (i + 1)
            m_at_r = enclosed_mass(r, MILKY_WAY_MODEL)

            # Verify radius
            assert result.radii[i] == pytest.approx(r, rel=1e-12)

            # Verify mass
            assert result.mass_result.series[i] == pytest.approx(
                m_at_r, rel=1e-12
            )

            # Verify each theory velocity
            assert result.series("newtonian")[i] == pytest.approx(
                ref_newtonian(r, m_at_r), rel=1e-12
            )
            assert result.series("gfd")[i] == pytest.approx(
                ref_gfd(r, m_at_r, accel_ratio), rel=1e-12
            )
            assert result.series("mond")[i] == pytest.approx(
                ref_mond(r, m_at_r, accel_ratio), rel=1e-12
            )
            assert result.series("cdm")[i] == pytest.approx(
                ref_cdm(r, m_at_r, m200), rel=1e-10
            )

    def test_accel_ratio_propagates(self):
        """Verify that accel_ratio flows through to GFD and MOND stages."""
        config = GravisConfig(
            MILKY_WAY_MODEL, 30.0, 20, accel_ratio=2.0
        )
        result = GravisEngine.rotation_curve(config).run()

        r = result.radii[10]
        m = result.mass_result.series[10]

        assert result.series("gfd")[10] == pytest.approx(
            ref_gfd(r, m, 2.0), rel=1e-12
        )
        assert result.series("mond")[10] == pytest.approx(
            ref_mond(r, m, 2.0), rel=1e-12
        )


# -----------------------------------------------------------------------
# API response format tests
# -----------------------------------------------------------------------
class TestApiResponseFormat:
    """Verify to_api_response() produces the correct flat dict format."""

    def test_api_response_has_required_keys(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 50)
        result = GravisEngine.rotation_curve(config, m200=1e12).run()
        response = result.to_api_response()
        for key in ("radii", "newtonian", "dtg", "mond", "cdm",
                     "enclosed_mass"):
            assert key in response, f"Missing API key: {key}"

    def test_api_response_gfd_mapped_to_dtg(self):
        """Internal stage name 'gfd' must map to API key 'dtg'."""
        config = GravisConfig(MILKY_WAY_MODEL, 30, 10)
        result = GravisEngine.rotation_curve(config).run()
        response = result.to_api_response()
        assert "dtg" in response
        # Verify it is actually the GFD series (not empty or wrong)
        assert len(response["dtg"]) == 10
        assert all(v > 0 for v in response["dtg"])

    def test_api_response_values_rounded(self):
        """All values in API response must be rounded to specified precision."""
        config = GravisConfig(MILKY_WAY_MODEL, 30, 10)
        result = GravisEngine.rotation_curve(config).run()
        response = result.to_api_response()

        # Radii rounded to 6 decimal places
        for r in response["radii"]:
            assert r == round(r, 6)

        # Velocities rounded to 4 decimal places
        for v in response["newtonian"]:
            assert v == round(v, 4)


# -----------------------------------------------------------------------
# Verbose response format tests
# -----------------------------------------------------------------------
class TestVerboseResponseFormat:
    """Verify to_verbose_response() produces the full computation chain."""

    def test_verbose_response_has_config(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 20)
        result = GravisEngine.rotation_curve(config).run()
        verbose = result.to_verbose_response()
        assert "config" in verbose
        assert verbose["config"]["max_radius"] == 30.0

    def test_verbose_response_has_mass_model_trace(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 20)
        result = GravisEngine.rotation_curve(config).run()
        verbose = result.to_verbose_response()
        mass_trace = verbose["mass_model"]
        assert "equation" in mass_trace
        assert "series" in mass_trace
        assert "intermediates" in mass_trace

    def test_verbose_response_has_all_stages(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 20)
        result = GravisEngine.rotation_curve(config, m200=1e12).run()
        verbose = result.to_verbose_response()
        stages = verbose["stages"]
        assert "newtonian" in stages
        assert "gfd" in stages
        assert "mond" in stages
        assert "cdm" in stages

    def test_verbose_stage_has_equation_and_intermediates(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 20)
        result = GravisEngine.rotation_curve(config).run()
        verbose = result.to_verbose_response()
        gfd_trace = verbose["stages"]["gfd"]
        assert "equation" in gfd_trace
        assert "parameters" in gfd_trace
        assert "intermediates" in gfd_trace
        # GFD intermediates should contain the field equation variables
        assert "y_N" in gfd_trace["intermediates"]
        assert "x" in gfd_trace["intermediates"]
        assert "g_eff" in gfd_trace["intermediates"]


# -----------------------------------------------------------------------
# Inference via engine
# -----------------------------------------------------------------------
class TestEngineInference:
    """Test GravisEngine.infer_mass() static method."""

    def test_inference_returns_stage_result(self):
        result = GravisEngine.infer_mass(8.0, 230.0)
        assert isinstance(result, StageResult)
        assert result.name == "inference"

    def test_inference_parity_with_reference(self):
        """Engine inference must match the standalone infer_mass function."""
        for r, v in [(5.0, 200.0), (8.0, 230.0), (15.0, 220.0)]:
            result = GravisEngine.infer_mass(r, v, accel_ratio=1.0)
            m_engine = result.series[0]
            m_ref = ref_infer_mass(r, v, 1.0)
            assert m_engine == pytest.approx(m_ref, rel=1e-12)

    def test_inference_records_intermediates(self):
        result = GravisEngine.infer_mass(8.0, 230.0)
        assert "g_eff" in result.intermediates
        assert "x" in result.intermediates
        assert "g_N" in result.intermediates

    def test_inference_records_parameters(self):
        result = GravisEngine.infer_mass(8.0, 230.0, accel_ratio=1.5)
        assert result.parameters["r_kpc"] == 8.0
        assert result.parameters["v_km_s"] == 230.0
        assert result.parameters["accel_ratio"] == 1.5

    def test_inference_with_accel_ratio(self):
        """Verify accel_ratio propagates through inference."""
        result_1 = GravisEngine.infer_mass(8.0, 230.0, accel_ratio=1.0)
        result_2 = GravisEngine.infer_mass(8.0, 230.0, accel_ratio=2.0)
        # Different accel_ratio should produce different inferred masses
        assert result_1.series[0] != result_2.series[0]

    def test_inference_zero_velocity(self):
        result = GravisEngine.infer_mass(8.0, 0.0)
        assert result.series[0] == 0.0


# -----------------------------------------------------------------------
# Method chaining
# -----------------------------------------------------------------------
class TestMethodChaining:
    """Verify fluent API style works correctly."""

    def test_set_mass_stage_returns_engine(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 10)
        engine = GravisEngine(config)
        returned = engine.set_mass_stage(
            equation=mass_model_eq,
            equation_label="test",
        )
        assert returned is engine

    def test_add_stage_returns_engine(self):
        config = GravisConfig(MILKY_WAY_MODEL, 30, 10)
        engine = GravisEngine(config)
        stage = GravisStage("test", newtonian_eq, "v = sqrt(G*M/r)")
        returned = engine.add_stage(stage)
        assert returned is engine

    def test_chained_construction(self):
        """Build an engine entirely via chaining."""
        config = GravisConfig(MILKY_WAY_MODEL, 30, 10)
        result = (
            GravisEngine(config)
            .set_mass_stage(mass_model_eq, "M(<r)")
            .add_stage(GravisStage("newtonian", newtonian_eq, "v=sqrt(GM/r)"))
            .add_stage(GravisStage("gfd", gfd_eq, "x^2/(1+x)=g_N/a0"))
            .run()
        )
        assert "newtonian" in result.stage_results
        assert "gfd" in result.stage_results
        assert len(result.radii) == 10
