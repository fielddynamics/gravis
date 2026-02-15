"""
Solar System Constraints Service stub.

Will demonstrate Newtonian recovery in the high-acceleration
regime (g >> a0) for planetary orbits and spacecraft trajectories,
confirming DTG reduces to standard gravity where expected.

Status: coming_soon
"""

from physics.services import GravisService


class SolarService(GravisService):

    id = "solar"
    name = "Solar System Constraints"
    description = "Newtonian recovery in the high-acceleration regime"
    category = "cosmological"
    status = "coming_soon"
    route = "/solar"

    def validate(self, config):
        raise NotImplementedError(
            "Solar system service is not yet implemented")

    def compute(self, config):
        raise NotImplementedError(
            "Solar system service is not yet implemented")
