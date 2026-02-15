"""
Radial Acceleration Relation (RAR) Service stub.

Will plot observed vs. baryonic gravitational acceleration for
SPARC galaxies (175 galaxies, 2693 points) with GFD and MOND
predictions overlaid.

Status: coming_soon
"""

from physics.services import GravisService


class RARService(GravisService):

    id = "rar"
    name = "Radial Acceleration Relation"
    description = "Observed vs. baryonic acceleration across 175 SPARC galaxies"
    category = "galactic"
    status = "coming_soon"
    route = "/rar"

    def validate(self, config):
        raise NotImplementedError("RAR service is not yet implemented")

    def compute(self, config):
        raise NotImplementedError("RAR service is not yet implemented")
