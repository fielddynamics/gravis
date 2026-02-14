"""
Cross-galaxy rotation curve validation.

These tests verify that the physics predictions hold across ALL catalog
galaxies, not just the Milky Way.  They check theory ordering, observation
matching, and expected physical behavior across the full mass range from
gas-dominated dwarfs to massive spirals.
"""

import math
import pytest

from physics.aqual import velocity as dtg_velocity
from physics.newtonian import velocity as newtonian_velocity
from physics.mond import velocity as mond_velocity
from physics.mass_model import enclosed_mass, total_mass
from data.galaxies import get_prediction_galaxies


# Collect all prediction galaxies that have both mass_model and observations
GALAXIES = [g for g in get_prediction_galaxies()
            if g.get("mass_model") and g.get("observations")]


def _galaxy_ids():
    """Parametrize IDs for readable test output."""
    return [g["id"] for g in GALAXIES]


class TestGFDCloserThanNewton:
    """GFD should predict observations more accurately than Newtonian
    gravity for the majority of points.  This is the core claim:
    the Lagrangian enhancement goes in the right direction.

    Note: We do NOT require 2-sigma matching because the mass models
    are independently measured (not fitted to rotation curves), so
    systematic offsets in total mass are expected.  The test is that
    GFD improves over Newton, not that it perfectly fits."""

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_gfd_closer_than_newton_majority(self, galaxy):
        """For each galaxy, >60% of observation points should have
        |v_GFD - v_obs| < |v_Newton - v_obs|."""
        mm = galaxy["mass_model"]
        obs = galaxy["observations"]
        gfd_closer = 0
        total = 0
        for o in obs:
            r, v = o["r"], o["v"]
            m = enclosed_mass(r, mm)
            v_n = newtonian_velocity(r, m)
            v_g = dtg_velocity(r, m)
            if abs(v_g - v) < abs(v_n - v):
                gfd_closer += 1
            total += 1
        fraction = gfd_closer / total
        assert fraction >= 0.60, \
            f"{galaxy['id']}: GFD closer than Newton for only " \
            f"{gfd_closer}/{total} ({fraction:.0%}) points"

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_gfd_rms_less_than_newtonian(self, galaxy):
        """GFD RMS velocity residual should be smaller than Newtonian
        RMS residual for every galaxy."""
        mm = galaxy["mass_model"]
        obs = galaxy["observations"]
        rms_gfd = 0.0
        rms_newton = 0.0
        n = 0
        for o in obs:
            r, v = o["r"], o["v"]
            m = enclosed_mass(r, mm)
            v_n = newtonian_velocity(r, m)
            v_g = dtg_velocity(r, m)
            rms_newton += (v_n - v) ** 2
            rms_gfd += (v_g - v) ** 2
            n += 1
        rms_newton = math.sqrt(rms_newton / n)
        rms_gfd = math.sqrt(rms_gfd / n)
        assert rms_gfd < rms_newton, \
            f"{galaxy['id']}: GFD RMS={rms_gfd:.1f} >= Newton RMS={rms_newton:.1f}"


class TestTheoryOrdering:
    """At every observation radius for every galaxy:
    Newton <= MOND <= GFD (for default accel_ratio=1.0)."""

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_newton_le_mond_le_gfd(self, galaxy):
        """Theory ordering must hold at all observation radii."""
        mm = galaxy["mass_model"]
        for o in galaxy["observations"]:
            r = o["r"]
            m = enclosed_mass(r, mm)
            v_n = newtonian_velocity(r, m)
            v_m = mond_velocity(r, m)
            v_g = dtg_velocity(r, m)
            assert v_n <= v_m + 0.01, \
                f"{galaxy['id']} r={r}: Newton {v_n:.2f} > MOND {v_m:.2f}"
            assert v_m <= v_g + 0.01, \
                f"{galaxy['id']} r={r}: MOND {v_m:.2f} > GFD {v_g:.2f}"


class TestNewtonianUnderprediction:
    """Newton should underpredict observed velocities at outer radii
    (where enclosed fraction is high and the rotation curve is flat).
    This is the entire reason dark matter or modified gravity is needed."""

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_newton_below_observations_at_outer_radii(self, galaxy):
        """For points with >50% enclosed mass, Newton should usually
        underpredict (at least 60% of such points)."""
        mm = galaxy["mass_model"]
        m_total = total_mass(mm)
        outer_points = []
        for o in galaxy["observations"]:
            r = o["r"]
            m = enclosed_mass(r, mm)
            if m / m_total > 0.5:
                v_n = newtonian_velocity(r, m)
                outer_points.append(v_n < o["v"])
        if len(outer_points) >= 2:
            frac_below = sum(outer_points) / len(outer_points)
            assert frac_below >= 0.60, \
                f"{galaxy['id']}: Newton below obs only " \
                f"{frac_below:.0%} of outer points"


class TestGFDBoostScaling:
    """The GFD/Newton boost should scale with galaxy mass: dwarfs in the
    deep-field regime get a much larger enhancement than massive spirals
    in the transition regime."""

    def _boost_at_outer(self, galaxy):
        """Compute GFD/Newton velocity ratio at the outermost
        observation point."""
        mm = galaxy["mass_model"]
        r = galaxy["observations"][-1]["r"]
        m = enclosed_mass(r, mm)
        v_n = newtonian_velocity(r, m)
        v_g = dtg_velocity(r, m)
        if v_n <= 0:
            return 1.0
        return v_g / v_n

    def test_dwarfs_have_larger_boost(self):
        """Gas-dominated dwarfs (DDO 154, IC 2574, NGC 3109) should have
        larger GFD/Newton boosts than massive spirals (MW, M31)."""
        by_id = {g["id"]: g for g in GALAXIES}
        dwarf_ids = ["ddo154", "ic2574", "ngc3109"]
        spiral_ids = ["milky_way", "m31"]

        dwarf_boosts = [self._boost_at_outer(by_id[gid])
                        for gid in dwarf_ids if gid in by_id]
        spiral_boosts = [self._boost_at_outer(by_id[gid])
                         for gid in spiral_ids if gid in by_id]

        if dwarf_boosts and spiral_boosts:
            min_dwarf = min(dwarf_boosts)
            max_spiral = max(spiral_boosts)
            assert min_dwarf > max_spiral, \
                f"Dwarf min boost {min_dwarf:.2f} <= " \
                f"spiral max boost {max_spiral:.2f}"

    def test_massive_spiral_small_inner_boost(self):
        """For massive spirals, the GFD boost at r=1 kpc should be < 10%.
        Inner regions have high acceleration (Newtonian regime)."""
        by_id = {g["id"]: g for g in GALAXIES}
        for gid in ["milky_way", "m31"]:
            if gid not in by_id:
                continue
            mm = by_id[gid]["mass_model"]
            m = enclosed_mass(1.0, mm)
            v_n = newtonian_velocity(1.0, m)
            v_g = dtg_velocity(1.0, m)
            boost = (v_g / v_n) - 1.0 if v_n > 0 else 0.0
            assert boost < 0.10, \
                f"{gid} inner boost = {boost:.1%}, expected < 10%"


class TestFlatRotationCurves:
    """GFD should produce approximately flat rotation curves at large radii
    for galaxies with sufficient radial extent."""

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_outer_gfd_approximately_flat(self, galaxy):
        """For galaxies with observations spanning > 5 kpc,
        the GFD velocity variation in the outer half should be < 35%."""
        mm = galaxy["mass_model"]
        obs = galaxy["observations"]
        if obs[-1]["r"] - obs[0]["r"] < 5.0:
            pytest.skip("Radial extent too small")

        # Outer half of observation radii
        mid_idx = len(obs) // 2
        outer_obs = obs[mid_idx:]
        if len(outer_obs) < 2:
            pytest.skip("Not enough outer points")

        velocities = []
        for o in outer_obs:
            m = enclosed_mass(o["r"], mm)
            velocities.append(dtg_velocity(o["r"], m))

        v_max = max(velocities)
        v_min = min(velocities)
        variation = (v_max - v_min) / ((v_max + v_min) / 2.0)
        assert variation < 0.35, \
            f"{galaxy['id']}: outer variation = {variation:.1%}, " \
            f"expected < 35%"
