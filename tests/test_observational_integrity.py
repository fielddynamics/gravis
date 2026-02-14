"""
Observational data integrity tests.

These verify that the galaxy catalog's observational data is physically
plausible and internally consistent.  Bad input data would silently
propagate through all calculations, so these are critical for trust.
"""

import math
import pytest

from data.galaxies import get_prediction_galaxies, get_inference_galaxies
from physics.mass_model import total_mass


# All galaxies with observational data
PRED_GALAXIES = [g for g in get_prediction_galaxies()
                 if g.get("observations")]
INF_GALAXIES = [g for g in get_inference_galaxies()
                if g.get("observations")]
ALL_OBS_GALAXIES = PRED_GALAXIES + INF_GALAXIES


def _all_ids():
    return [g["id"] for g in ALL_OBS_GALAXIES]


class TestVelocitiesPhysicallyPlausible:
    """All observed velocities should be within the range of known
    galactic rotation velocities (roughly 5-500 km/s)."""

    @pytest.mark.parametrize("galaxy", ALL_OBS_GALAXIES, ids=_all_ids())
    def test_velocities_in_plausible_range(self, galaxy):
        for o in galaxy["observations"]:
            v = o["v"]
            assert 3.0 < v < 500.0, \
                f"{galaxy['id']} r={o['r']}: v={v} outside plausible range"


class TestErrorBarsReasonable:
    """Error bars should be a modest fraction of the velocity.
    Errors > 50% of velocity suggest corrupted data."""

    @pytest.mark.parametrize("galaxy", ALL_OBS_GALAXIES, ids=_all_ids())
    def test_errors_less_than_half_velocity(self, galaxy):
        for o in galaxy["observations"]:
            v, err = o["v"], o.get("err", 0)
            if err > 0:
                frac = err / v
                assert frac <= 0.50, \
                    f"{galaxy['id']} r={o['r']}: err/v = {frac:.1%}"


class TestRadiiWithinGalaxyExtent:
    """Observation radii should not exceed the galaxy's stated plotting
    distance.  An observation beyond the stated extent is suspect."""

    @pytest.mark.parametrize("galaxy", ALL_OBS_GALAXIES, ids=_all_ids())
    def test_radii_within_stated_distance(self, galaxy):
        max_r = galaxy.get("distance", 1000)
        for o in galaxy["observations"]:
            r = o["r"]
            # Allow 10% overshoot for rounding
            assert r <= max_r * 1.10, \
                f"{galaxy['id']}: observation at r={r} kpc exceeds " \
                f"galaxy extent {max_r} kpc"


class TestMilkyWayLiteratureValues:
    """The Milky Way has the most precisely measured rotation curve.
    Verify key data points against well-established values."""

    def _get_mw(self):
        for g in PRED_GALAXIES:
            if g["id"] == "milky_way":
                return g
        pytest.skip("Milky Way not in catalog")

    def test_solar_radius_velocity(self):
        """The circular velocity at the solar radius (~8 kpc) is one of
        the most precisely measured quantities in galactic astronomy.
        Should be 220-240 km/s (Eilers+2019, Jiao+2023)."""
        mw = self._get_mw()
        # Find observation nearest to 8 kpc
        near_8 = [o for o in mw["observations"]
                  if 7.0 <= o["r"] <= 9.0]
        assert len(near_8) >= 1, "No MW observation near 8 kpc"
        for o in near_8:
            assert 215 <= o["v"] <= 245, \
                f"MW at r={o['r']}: v={o['v']}, expected 220-240 km/s"

    def test_milky_way_total_mass_order(self):
        """MW total baryonic mass should be ~ 5-10 x 10^10 M_sun."""
        mw = self._get_mw()
        m = total_mass(mw["mass_model"])
        log_m = math.log10(m)
        assert 10.5 < log_m < 11.2, \
            f"MW total mass log10={log_m:.2f}, expected ~10.7-11.0"

    def test_milky_way_has_all_components(self):
        """MW should have bulge, disk, and gas components."""
        mw = self._get_mw()
        mm = mw["mass_model"]
        assert "bulge" in mm and mm["bulge"].get("M", 0) > 0
        assert "disk" in mm and mm["disk"].get("M", 0) > 0
        assert "gas" in mm and mm["gas"].get("M", 0) > 0

    def test_milky_way_disk_dominates(self):
        """The stellar disk should contribute >50% of total baryonic mass."""
        mw = self._get_mw()
        mm = mw["mass_model"]
        m_disk = mm["disk"]["M"]
        m_total = total_mass(mm)
        assert m_disk / m_total > 0.50, \
            f"MW disk fraction = {m_disk/m_total:.1%}, expected >50%"
