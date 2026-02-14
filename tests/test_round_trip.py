"""
Round-trip bidirectional consistency tests.

The field equation x^2/(1+x) = y_N is bidirectional:
  Forward (prediction): M -> g_N -> solve for x -> v
  Inverse (inference):  v -> g -> x -> evaluate g_N -> M

If the Lagrangian is correctly implemented, forward-then-inverse and
inverse-then-forward should both be exact identities (up to floating
point precision).  These tests verify this across all catalog galaxies,
across 7 decades of mass, and across 3 decades of radius.
"""

import math
import pytest

from physics.aqual import velocity as dtg_velocity
from physics.inference import infer_mass
from physics.mass_model import enclosed_mass
from data.galaxies import get_prediction_galaxies


GALAXIES = [g for g in get_prediction_galaxies()
            if g.get("mass_model") and g.get("observations")]


def _galaxy_ids():
    return [g["id"] for g in GALAXIES]


class TestForwardInverseAllGalaxies:
    """M -> v -> M must recover the original enclosed mass for every
    galaxy at every observation radius."""

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_m_to_v_to_m_recovery(self, galaxy):
        """Compute GFD velocity from model mass, then infer mass from
        that velocity.  Should recover the model's enclosed mass."""
        mm = galaxy["mass_model"]
        for o in galaxy["observations"]:
            r = o["r"]
            m_model = enclosed_mass(r, mm)
            if m_model <= 0:
                continue

            # Forward: M -> v
            v = dtg_velocity(r, m_model)

            # Inverse: v -> M
            m_recovered = infer_mass(r, v)

            assert m_recovered == pytest.approx(m_model, rel=1e-6), \
                f"{galaxy['id']} r={r}: M_model={m_model:.4e}, " \
                f"M_recovered={m_recovered:.4e}"


class TestInverseForwardAllGalaxies:
    """v -> M -> v must recover the observed velocity (modulo the fact
    that the observation may not match the model exactly)."""

    @pytest.mark.parametrize("galaxy", GALAXIES, ids=_galaxy_ids())
    def test_v_to_m_to_v_recovery(self, galaxy):
        """For each observation, infer mass, then compute GFD velocity
        from that inferred mass.  Should recover the input velocity."""
        for o in galaxy["observations"]:
            r, v = o["r"], o["v"]
            if r <= 0 or v <= 0:
                continue

            # Inverse: v -> M
            m_inferred = infer_mass(r, v)

            # Forward: M -> v
            v_recovered = dtg_velocity(r, m_inferred)

            assert v_recovered == pytest.approx(v, rel=1e-6), \
                f"{galaxy['id']} r={r}: v_obs={v}, v_recovered={v_recovered:.4f}"


class TestRoundTripAcrossMassDecades:
    """Round-trip M -> v -> M across 7 decades of mass at fixed radius."""

    @pytest.mark.parametrize("log_m", [6, 7, 8, 9, 10, 11, 12, 13])
    def test_mass_decade(self, log_m):
        """Round-trip at r=10 kpc for mass = 10^log_m M_sun."""
        m = 10.0 ** log_m
        r = 10.0

        v = dtg_velocity(r, m)
        if v <= 0:
            pytest.skip(f"Zero velocity for M=10^{log_m}")

        m_recovered = infer_mass(r, v)
        assert m_recovered == pytest.approx(m, rel=1e-6), \
            f"M=10^{log_m}: recovered={m_recovered:.4e}"


class TestRoundTripAcrossRadii:
    """Round-trip M -> v -> M across 3 decades of radius at fixed mass."""

    @pytest.mark.parametrize("r_kpc", [
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
    ])
    def test_radius_value(self, r_kpc):
        """Round-trip at fixed M=5e10 M_sun for various radii."""
        m = 5.0e10

        v = dtg_velocity(r_kpc, m)
        if v <= 0:
            pytest.skip(f"Zero velocity at r={r_kpc}")

        m_recovered = infer_mass(r_kpc, v)
        assert m_recovered == pytest.approx(m, rel=1e-6), \
            f"r={r_kpc}: M=5e10, recovered={m_recovered:.4e}"
