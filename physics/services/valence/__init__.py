"""
Valence Predictions Service stub.

Will predict electron shell structure and valence from k=4
tetrahedral topology, comparing DTG predictions against the
periodic table for all stable elements.

Status: coming_soon
"""

from physics.services import GravisService


class ValenceService(GravisService):

    id = "valence"
    name = "Valence Predictions"
    description = "Electron shell structure from k=4 topology"
    category = "quantum_nuclear"
    status = "coming_soon"
    route = "/valence"

    def validate(self, config):
        raise NotImplementedError(
            "Valence service is not yet implemented")

    def compute(self, config):
        raise NotImplementedError(
            "Valence service is not yet implemented")
